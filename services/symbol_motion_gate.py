"""Refuse symbols that cannot move far enough to pay for their own round trip.

``symbol_edge_gate`` bans a symbol once the BOOK has proven we lose on it: it
needs closed round trips, so a symbol has to cost real money before it can be
judged. This is the same question asked of the FEED instead, where the answer
is available before the first trade and does not depend on having lost it.

THE QUESTION. A round trip costs ``fixed + rate * notional`` -- 0.86% of
notional at the $0.75 clip, 0.59% at $1.50. A position is now held for
``GHOST_STALE_EXIT_SECONDS`` (900s) before the stale clock closes it. So the
symbol has one job: at some point inside that window, rise by more than the
round trip costs. If it does that rarely enough, no entry rule and no model can
make it profitable, because there is nothing there to capture.

MEASURED 2026-09-05 over 7 days of ``market_stream``, as the share of 15-minute
windows whose HIGH clears a 0.65% round trip. Every symbol with enough feed to
judge, in order:

    BASECAT-USDC    54.7%  |  TYBG-USDC          3.0%   <- floor
    BSTONK-USDC     48.6%  |  MOG-USDC           2.7%
    CP-USDC         41.0%  |  CBETH-USDC         1.3%
    LFG-USDC        40.7%  |  OPENHUMAN-USDC     1.3%
    MEME-USDC       33.2%  |  MOCHI-USDC         1.2%
    PRINTERINK      27.3%  |  CBLTC-USDC         1.1%
    BASEPEPE-USDC   15.7%  |  CBBTC-USDC         0.7%
    MOONBASE-USDC   12.3%  |  XCHAT-USDC         0.0%
    CBADA-USDC      11.4%  |  CBHYPE-USDC        0.0%
    CBMEGA-USDC      8.2%  |  GRASS-USDC         0.0%
    CBXRP-USDC       7.7%  |  SPACEX-USDC        0.0%
    CBZEC-USDC       6.4%  |  LIQUIDBGT-USDC     0.0%
    AERO-USDC        5.4%  |  SPCX-USDC          0.0%
    COMP-USDC        5.4%  |  ANTHROPIC-USDC     0.0%
                           |  1KTO100M-USDC      0.0%

The right-hand column is where the live wallet has been losing. Five of the 11
settled live round trips were on CBBTC and CBETH:

    CBBTC-USDC   -0.013177      CBETH-USDC   -0.005443
    CBBTC-USDC   -0.025137      CBETH-USDC   +0.010394
    CBBTC-USDC   -0.016878

They total -0.050240 against a lifetime net of +0.110050, so the account is
+0.160291 without them -- a 46% improvement from declining five trades. The
single largest win in the book, BSTONK +0.242029, came from the symbol second
from the top. Nothing here is about direction or about the model: it is about
whether the instrument moves enough to pay the toll.

THE THRESHOLD IS DELIBERATELY LOW. ``MIN_PAY_RATE`` is 3%, well under the 11.7%
pooled base rate, because the failure this gate must not cause is a quiet
pipeline. At 3% it refuses 14 symbols and leaves 15 -- including AERO, COMP and
CBXRP, where trading actually happens today -- so the tradeable universe
narrows without ever emptying. CBXRP passes here at 7.7% and is refused by the
edge gate instead, on its book: 6 closed round trips at a mean of -0.770%.

MEASURED ON THE HIGH, NOT THE CLOSE. A position carries a take-profit, a
trailing stop and a profit lock, so a move that happens inside the window and
gives back is still a move the bracket could have captured. Judging on the
window's close would ban a symbol for round-tripping rather than for standing
still, which is a different accusation and not the one this gate makes. The
high is therefore the generous reading: a symbol that fails THIS test never had
a moment worth trading.

BANS ONLY, NEVER PROMOTES -- the same asymmetry ``symbol_edge_gate`` documents.
A high pay rate is volatility, not edge, and BASECAT is the proof: it clears
its cost in 30.8% of windows and is still the book's largest destroyer of
capital, banned by the edge gate on 35 round trips at a mean of -1.565%. The
two gates are answering different questions and a symbol has to pass both.

FAILS OPEN. Any error reading the feed allows the trade. A gate that cannot
read its evidence has no evidence to refuse on, and a dark pipeline is a worse
failure than a bad trade.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from services import frozen_tape_guard
from services.logging_utils import log_message

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "storage" / "trading_cache.db"

#: How long a position is actually held before the stale clock closes it. This
#: is the window the symbol has to produce a payable move in, so it tracks the
#: exit rule rather than being tuned separately -- if the hold gets longer, the
#: bar a symbol has to clear gets easier, automatically.
def _hold_window_sec() -> float:
    raw = os.getenv("GHOST_NEG_EXIT_SECONDS", os.getenv("GHOST_STALE_EXIT_SECONDS", "900"))
    try:
        return max(60.0, float(raw))
    except (TypeError, ValueError):
        return 900.0


#: What a round trip costs as a fraction of notional, at the size actually
#: traded. Shares the edge gate's default so both gates price the same trade.
ROUND_TRIP_COST = float(os.getenv("SYMBOL_EDGE_ROUND_TRIP_COST", "0.0065"))

#: Share of hold windows that must clear the round trip. See the docstring for
#: why this sits well below the 11.7% pooled base rate.
MIN_PAY_RATE = float(os.getenv("SYMBOL_MOTION_MIN_PAY_RATE", "0.03"))

#: Trailing feed history the rate is measured over.
LOOKBACK_SEC = float(os.getenv("SYMBOL_MOTION_LOOKBACK_SEC", str(7 * 86400)))

#: Fewest windows before a symbol can be judged. A newly listed token with
#: twenty ticks has no measurable pay rate, and refusing it on that basis would
#: ban exactly the new movers this pipeline is meant to catch.
MIN_WINDOWS = int(os.getenv("SYMBOL_MOTION_MIN_WINDOWS", "200"))

#: A tick that moves the price by more than this is feed contamination, not a
#: move -- roughly 20% of symbols carry at least one. Counting them would hand
#: a dead symbol a perfect pay rate off a single bad print.
MAX_SANE_MOVE = float(os.getenv("SYMBOL_MOTION_MAX_SANE_MOVE", "1.0"))

#: Never refused: the stable legs every round trip routes through.
NEVER_BAN = {s.strip().upper() for s in
             os.getenv("SYMBOL_EDGE_NEVER_BAN", "USDC,USDT,DAI,USDBC,WETH,ETH").split(",")
             if s.strip()}

CACHE_SEC = float(os.getenv("SYMBOL_MOTION_CACHE_SEC", "900"))

_cache: Dict[str, Tuple[float, str]] = {}
_cache_built_at: float = 0.0


def pay_rate(prices: List[Tuple[float, float]], *, window_sec: float,
             cost: float) -> Tuple[float, int]:
    """Share of windows whose HIGH clears ``cost``, and how many there were.

    ``prices`` is (ts, price) ascending. Each tick opens one window; the window
    closes ``window_sec`` later, and it pays if the maximum price inside it
    exceeds the opening price by more than ``cost``.

    A rolling maximum would be O(n) rather than this O(n*k), but k is bounded
    by the ticks inside 15 minutes (~2 on this feed, 87 ticks per 10 minutes
    across 17 symbols) and the clarity is worth more than the constant here.
    """
    if not prices:
        return 0.0, 0
    paid = 0
    windows = 0
    end = 0
    last_ts = prices[-1][0]
    for start, (ts, price) in enumerate(prices):
        if not (price > 0):
            continue
        # A window can only be judged once the feed reaches its far edge.
        # Testing the LAST tick rather than the window's own contents is the
        # point: `end` stops at the final tick inside the window, which is by
        # construction earlier than ts+window_sec, so asking whether IT has
        # reached the edge is always false and judged nothing at all.
        if last_ts < ts + window_sec:
            break            # ...and every later window is shorter still
        if end < start:
            end = start
        while end + 1 < len(prices) and prices[end + 1][0] <= ts + window_sec:
            end += 1
        high = price
        for idx in range(start, end + 1):
            nxt = prices[idx][1]
            if not (nxt > 0):
                continue
            move = (nxt - price) / price
            if move > MAX_SANE_MOVE:
                continue     # contaminated print, not a tradeable high
            high = max(high, nxt)
        windows += 1
        if (high - price) / price > cost:
            paid += 1
    return (paid / windows if windows else 0.0), windows


def _load_feed(now: float) -> Dict[str, List[Tuple[float, float]]]:
    feed: Dict[str, List[Tuple[float, float]]] = {}
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    except Exception:  # noqa: BLE001 - no feed is not a reason to block trading
        return feed
    try:
        rows = conn.execute(
            "SELECT symbol, ts, price FROM market_stream "
            "WHERE ts >= ? AND price > 0 ORDER BY symbol, ts",
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
    window = _hold_window_sec()
    verdicts: Dict[str, Tuple[float, str]] = {}
    for symbol, prices in _load_feed(now).items():
        base = symbol.split("-")[0]
        if base in NEVER_BAN or symbol in NEVER_BAN:
            continue         # this gate's invariant: the stable legs every
                             # round trip routes through are never banned HERE,
                             # even for staleness. frozen_tape_guard still
                             # reports a dead peg to consumers that want it --
                             # banning WETH out of THIS gate would close the
                             # deep-pool route on a feed outage.
        # BEFORE any statistic, not after (backlog edb6da74). A frozen tape
        # does not fail the pay-rate test loudly -- it scores 0.0 on too few
        # windows, which this gate's own "not enough feed to judge" rule then
        # reads as a quiet symbol and ALLOWS. DRB-USDC was 71 ticks of one
        # constant, 0.0001663, and passed here.
        #
        # THE FROZEN HALF ONLY, deliberately. `frozen_tape_guard` also refuses a
        # STALE tape, and this gate does NOT act on that: `streamed_symbol_
        # candidates.FRESH_SEC` already requires a tick inside 600 SECONDS to
        # offer a symbol at all, which is 144x stricter than the guard's 24h and
        # runs upstream of here (Echo, p674). Duplicating it would add nothing
        # and would refuse on a looser rule than the one already binding. What
        # does NOT exist anywhere upstream is the constant-price test: a symbol
        # ticking every 30 seconds and publishing the same number forever is
        # FRESH by every check in this repo. That is the hole, and this is it
        # closed.
        dead = frozen_tape_guard.verdict(prices, symbol=symbol, now=now)
        if dead and "FROZEN" in dead:
            verdicts[symbol] = (0.0, dead)
            continue
        rate, windows = pay_rate(prices, window_sec=window, cost=ROUND_TRIP_COST)
        if windows < MIN_WINDOWS:
            continue         # not enough feed to judge -- allowed, by design
        if rate >= MIN_PAY_RATE:
            continue
        verdicts[symbol] = (
            rate,
            f"{rate * 100:.1f}% of {windows} {window / 60:.0f}-minute windows "
            f"cleared the {ROUND_TRIP_COST * 100:.2f}% round trip "
            f"(floor {MIN_PAY_RATE * 100:.1f}%)",
        )
    for symbol, (_, detail) in sorted(verdicts.items()):
        if symbol not in _cache:
            log_message(
                "trading",
                f"SYMBOL MOTION GATE: refusing {symbol} -- {detail}",
                severity="warning",
            )
    _cache.clear()
    _cache.update(verdicts)
    _cache_built_at = now


def refusal_reason(symbol: str) -> Optional[str]:
    """Why this symbol cannot pay for a round trip, or None to allow it."""
    try:
        now = time.time()
        if now - _cache_built_at > CACHE_SEC:
            _rebuild(now)
        verdict = _cache.get(str(symbol or "").upper())
        return verdict[1] if verdict else None
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
    return {symbol: detail for symbol, (_, detail) in _cache.items()}


if __name__ == "__main__":  # pragma: no cover - operator diagnostic
    _now = time.time()
    _window = _hold_window_sec()
    print(f"hold window {_window / 60:.0f}m, round trip {ROUND_TRIP_COST:.4%}, "
          f"floor {MIN_PAY_RATE:.1%}, min windows {MIN_WINDOWS}\n")
    print(f"{'symbol':22s} {'windows':>8s} {'pay rate':>9s}  verdict")
    scored = []
    for _sym, _prices in _load_feed(_now).items():
        _rate, _n = pay_rate(_prices, window_sec=_window, cost=ROUND_TRIP_COST)
        scored.append((_rate, _n, _sym))
    for _rate, _n, _sym in sorted(scored, reverse=True):
        _base = _sym.split("-")[0]
        if _base in NEVER_BAN or _sym in NEVER_BAN:
            _verdict = "never banned"
        elif _n < MIN_WINDOWS:
            _verdict = f"allowed (only {_n} windows)"
        elif _rate >= MIN_PAY_RATE:
            _verdict = "allowed"
        else:
            _verdict = "REFUSED"
        print(f"{_sym:22s} {_n:8d} {_rate:8.1%}  {_verdict}")
