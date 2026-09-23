"""Refuse a symbol whose tape is CONSTANT or STALE, before any statistic is computed on it.

THE DEFECT, measured pass 673 (Echo) and sized pass 673 (Kite). DRB-USDC
streamed ONE price, 0.0001663, across all 71 ``market_stream`` ticks in 168h,
newest tick 166.1 HOURS old. ``mean|move|`` on that series is EXACTLY 0.0bp, so
pass 480 computed h*=5.842 and published "unpayable at any skill". It was
measuring a dead feed. That wrong number sat priority-1 in the backlog for 193
passes and steered the crew. Nothing in the repo refused to compute on it.

IT IS NOT ONE SYMBOL -- the item's own complement is dead. ``frozen_tape_census``
(Kite, dc54b5b) over 168h: **185 symbols streamed, 88 have FEWER THAN 2 DISTINCT
PRICES (47.6%), 128 have a newest tick older than 24h, 70 are both.** DRB is
unremarkable in that population; XPL-USDC repeats one price 248 times. So the
guard is GENERAL.

THE THREE CATEGORIES, and why staleness alone is not the discriminator.
Criterion 4 as filed said "the staleness of the newest tick is the
discriminator, not the variance". That MISSES 18 SYMBOLS -- 88 frozen minus 70
frozen-and-stale. HOUSE-USDC (1.5h), SETZ (2.8h), HI (3.2h), PEPKING (4.0h),
REVERT, DAEMON, NEOBANK, BONG, JO, DTF, AOAS, BILLY, MILO, MOONSHOT, FLX, BREW,
BULLCOIN, WORLDWAR are all publishing RIGHT NOW and publishing a DEAD PRICE.
None is a stablecoin. So this guard uses BOTH tests:

    STALE    newest tick older than MAX_TICK_AGE_SEC   -> refuse, always
    FROZEN   fewer than 2 distinct prices in lookback  -> refuse, unless PEGGED

THE PEG CASE, which criterion 4 exists to protect. A genuinely pegged asset --
USDT-USDC, DAI-USDC -- is constant because it is SUPPOSED to be. Refusing it as
"frozen" would be a new defect. It is separated by a KNOWN-QUOTE-ASSET LIST
LOOKUP, not by any statistic, because no statistic can tell a peg from a corpse:
both are flat. ``PEG_ASSETS`` is that list, and it shares
``SYMBOL_EDGE_NEVER_BAN``'s membership so the two gates agree on what a stable
leg is.

A PEG IS EXEMPT FROM THE VARIANCE TEST AND NOT FROM THE STALENESS TEST. That is
the whole answer to criterion 4. "Not moving" is fine for a peg. "Not updating"
is never fine for anything: a stablecoin whose feed stopped 30 hours ago is not
telling you it is still worth a dollar, it is telling you nothing, and pricing a
leg off it is pricing off a memory.

TOO FEW TICKS IS NOT FROZEN. A symbol listed an hour ago with three ticks has
one distinct price trivially. Refusing it would ban exactly the new movers this
pipeline is meant to catch, so the frozen test needs ``MIN_TICKS`` before it
bites. Below that the tape is judged on staleness only -- which still catches
the dead ones, because a dead tape stops updating.

WHERE IT IS CALLED. There is NO chokepoint: 40+ files each open
``market_stream`` and write their own SELECT, and ``db.py`` has no helper (Kite
p673, criterion 2, answered plainly). So this predicate is called at the top of
the production consumers that can put money at risk -- the variance-shaped ones,
where a frozen symbol does not fail loudly but is scored as "no motion", which
reads identically to a symbol that is merely quiet. The 34 diagnostic scripts
are deliberately left out of the blast radius of a money-path change.

WHAT IT IS WORTH, stated honestly (Kite p673). The 18 frozen-but-fresh symbols
have produced ZERO trade_outcomes rows all-time; all 88 frozen tapes together
carry 9 round trips at -0.2222. This is a CORRECTNESS fix, not a money fix. Its
value is that a wrong statistic computed on a dead feed steered this crew for
193 passes.

FAILS OPEN. Any error reading the feed allows the symbol, matching
``symbol_motion_gate``. A gate that cannot read its evidence has no evidence to
refuse on, and a dark pipeline is a worse failure than a bad trade.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("TRADING_DB_PATH", str(ROOT / "storage" / "trading_cache.db")))

#: A newest tick older than this means the feed has stopped, whatever the price
#: says. 24h is the item's own figure and is generous: the live hold window is
#: 15 minutes, so a tape a full day old cannot price anything we would trade.
MAX_TICK_AGE_SEC = float(os.getenv("FROZEN_TAPE_MAX_TICK_AGE_SEC", str(24 * 3600)))

#: How much history the distinct-price count is taken over.
LOOKBACK_SEC = float(os.getenv("FROZEN_TAPE_LOOKBACK_SEC", str(7 * 86400)))

#: Distinct prices required inside the lookback. Two is the floor: one distinct
#: price IS the defect.
MIN_DISTINCT_PRICES = int(os.getenv("FROZEN_TAPE_MIN_DISTINCT", "2"))

#: Ticks required before the frozen test may fire. See "TOO FEW TICKS IS NOT
#: FROZEN" above -- below this the symbol is judged on staleness only.
MIN_TICKS = int(os.getenv("FROZEN_TAPE_MIN_TICKS", "8"))

#: Assets that are constant BY DESIGN. Membership, never a statistic -- no
#: statistic can tell a peg from a corpse, both are flat. Shares the motion
#: gate's list so the two gates agree on what a stable leg is.
PEG_ASSETS = {s.strip().upper() for s in
              os.getenv("SYMBOL_EDGE_NEVER_BAN", "USDC,USDT,DAI,USDBC,WETH,ETH").split(",")
              if s.strip()}

CACHE_SEC = float(os.getenv("FROZEN_TAPE_CACHE_SEC", "900"))

_cache: Dict[str, str] = {}
_cache_built_at: float = 0.0


def is_pegged(symbol: str) -> bool:
    """Is this symbol constant BY DESIGN rather than by death?

    A list lookup, deliberately. The base leg is what is being priced: in
    ``USDT-USDC`` the base is USDT and a flat line is the instrument working.
    """
    sym = str(symbol or "").upper()
    if not sym:
        return False
    return sym in PEG_ASSETS or sym.split("-")[0] in PEG_ASSETS


def verdict(prices: Sequence[Tuple[float, float]], *, symbol: str = "",
            now: Optional[float] = None,
            max_age_sec: Optional[float] = None,
            lookback_sec: Optional[float] = None,
            min_distinct: Optional[int] = None,
            min_ticks: Optional[int] = None) -> Optional[str]:
    """Why this tape cannot be computed on, or ``None`` to allow it.

    ``prices`` is ``(ts, price)`` in any order. PURE -- no database, no clock
    unless you omit ``now`` -- which is what makes the guard testable without a
    live feed, as the item's third criterion demands.

    Two independent refusals:

      * STALE  -- the newest tick is older than ``max_age_sec``. Applies to a
                  pegged asset exactly as it applies to anything else.
      * FROZEN -- fewer than ``min_distinct`` distinct prices across at least
                  ``min_ticks`` ticks inside ``lookback_sec``. Never applies to
                  a pegged asset.

    Staleness is checked FIRST so a dead peg is reported as dead rather than
    silently exempted.
    """
    now = time.time() if now is None else float(now)
    max_age_sec = MAX_TICK_AGE_SEC if max_age_sec is None else float(max_age_sec)
    lookback_sec = LOOKBACK_SEC if lookback_sec is None else float(lookback_sec)
    min_distinct = MIN_DISTINCT_PRICES if min_distinct is None else int(min_distinct)
    min_ticks = MIN_TICKS if min_ticks is None else int(min_ticks)

    clean: List[Tuple[float, float]] = []
    for row in prices or ():
        try:
            ts = float(row[0])
            price = float(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        if price > 0:
            clean.append((ts, price))
    if not clean:
        return None              # no evidence is not evidence of death

    newest = max(ts for ts, _ in clean)
    age = now - newest
    if age > max_age_sec:
        return (f"tape is STALE -- newest tick {age / 3600:.1f}h old "
                f"(limit {max_age_sec / 3600:.1f}h)")

    if is_pegged(symbol):
        return None              # constant by design; see the docstring

    window = [(ts, price) for ts, price in clean if ts >= now - lookback_sec]
    if len(window) < min_ticks:
        return None              # too new to judge, not frozen
    distinct = {round(price, 12) for _, price in window}
    if len(distinct) < min_distinct:
        only = next(iter(distinct))
        return (f"tape is FROZEN -- {len(distinct)} distinct price "
                f"({only:.10g}) across {len(window)} ticks in "
                f"{lookback_sec / 3600:.0f}h (need {min_distinct})")
    return None


def _load_feed(now: float) -> Dict[str, List[Tuple[float, float]]]:
    feed: Dict[str, List[Tuple[float, float]]] = {}
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    except Exception:  # noqa: BLE001 - no feed is not a reason to block trading
        return feed
    try:
        rows = conn.execute(
            "SELECT symbol, ts, price FROM market_stream "
            "WHERE ts >= ? AND price > 0",
            (now - LOOKBACK_SEC,),
        )
        for symbol, ts, price in rows:
            if not symbol:
                continue
            try:
                feed.setdefault(str(symbol).upper(), []).append((float(ts), float(price)))
            except (TypeError, ValueError):
                continue
    except Exception:  # noqa: BLE001
        return {}
    finally:
        conn.close()
    return feed


def _rebuild(now: float) -> None:
    global _cache_built_at
    verdicts: Dict[str, str] = {}
    for symbol, prices in _load_feed(now).items():
        reason = verdict(prices, symbol=symbol, now=now)
        if reason:
            verdicts[symbol] = reason
    _cache.clear()
    _cache.update(verdicts)
    _cache_built_at = now


def refusal_reason(symbol: str) -> Optional[str]:
    """Why this symbol's tape cannot be computed on, or ``None`` to allow it."""
    try:
        now = time.time()
        if now - _cache_built_at > CACHE_SEC:
            _rebuild(now)
        return _cache.get(str(symbol or "").upper())
    except Exception:  # noqa: BLE001
        return None


def refused_symbols() -> Dict[str, str]:
    """Current verdicts, for diagnostics and dashboards."""
    try:
        now = time.time()
        if now - _cache_built_at > CACHE_SEC:
            _rebuild(now)
    except Exception:  # noqa: BLE001
        return {}
    return dict(_cache)


if __name__ == "__main__":  # pragma: no cover - operator diagnostic
    _now = time.time()
    _feed = _load_feed(_now)
    _stale = _frozen = 0
    print(f"{len(_feed)} symbols streamed in the last {LOOKBACK_SEC / 3600:.0f}h; "
          f"stale limit {MAX_TICK_AGE_SEC / 3600:.0f}h, "
          f"min distinct {MIN_DISTINCT_PRICES}, min ticks {MIN_TICKS}\n")
    for _sym in sorted(_feed):
        _reason = verdict(_feed[_sym], symbol=_sym, now=_now)
        if not _reason:
            continue
        if _reason.startswith("tape is STALE"):
            _stale += 1
        else:
            _frozen += 1
        print(f"{_sym:22s} {_reason}")
    print(f"\nREFUSED {_stale + _frozen} of {len(_feed)}: "
          f"{_stale} stale, {_frozen} frozen-but-fresh")
