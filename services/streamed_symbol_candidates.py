"""Offer candidates from the symbols we already stream and already trust.

WHY THIS EXISTS
---------------
Measured 2026-09-06, one hour after production came back up: **79% of ghost
candidates were offered on symbols the gates refuse**, and eleven of the
thirteen symbols that passed every gate were never offered a candidate at all.
The pipeline produced two entries in an hour while the feed carried 774 ticks
per ten minutes across 25 symbols.

The cause is structural rather than a bug. ``select_candidates`` in
tools/c0d3rV2/crypto_paper_trade.py builds its list from
``fetch_dexscreener_candidates() + fetch_gecko_new_pool_candidates()`` -- it
hunts NEW pools. That is a real job and worth keeping: new tokens are where
outsized moves live. But it means a symbol we have streamed for a week, whose
edge and motion and stop-survivability we have already measured, can never
become a candidate, because it is not new.

So the discovery engine spends its output where trades cannot happen, while
the symbols that CAN trade sit idle. Over seven days of feed history, these
cleared a 0.65% round trip inside a 15-minute window often enough to be worth
trading:

    AERO-WETH        44.4% of windows        EURC-WETH       33.3%
    CLANKER-USDC     25.0%                   VIRTUAL-WETH    16.7%
    TIBBIR-VIRTUAL   11.1%                   SOL-CBBTC       10.0%
    AERO-USDC         5.5%

Six of those seven had never been offered a candidate.

WHAT THIS DOES AND DOES NOT DO
------------------------------
It proposes symbols. It does not decide anything: every candidate it returns
still passes through the same gates in the same order -- symbol_edge_gate,
symbol_motion_gate, stop_survivability_gate, strategy_edge_gate -- and through
the solver that arbitrates them. This module cannot make a trade happen that
the gates would refuse; it can only stop the pipeline from starving while
eligible symbols go unexamined.

It also does not compete with new-pool discovery. The two answer different
questions -- "what just appeared" and "what have we already proven" -- and a
pipeline that only asks the first will keep rediscovering the same
unprofitable memecoins while ignoring the instruments it has a week of
evidence about.

THE SELECTION RULE
------------------
A symbol is offered when all three hold:

  1. It is ticking NOW (a stale symbol cannot be entered or exited).
  2. Its feed is dense enough that a stop can be evaluated.
  3. It has historically cleared the round-trip cost inside the hold window
     often enough to pay for a trade.

That third condition is the same question ``symbol_motion_gate`` asks, and it
is asked here deliberately rather than left to the gate: a candidate that will
be refused downstream is a wasted slot in the offer list, which is precisely
the failure this module exists to fix.

RANKED, NOT FILTERED. The caller takes the top N, so the ordering is the
product: symbols that clear their cost more often are offered first.

WEIGHTED BY POOL DEPTH  [ceadb8da, 2026-09-17]
----------------------------------------------
``clear_rate`` is a VOLATILITY statistic. It counts windows whose HIGH cleared
the round trip, so a symbol that thrashes scores above one that drifts the
same distance calmly. Ranking on it alone therefore spends the evidence budget
exactly where per-trade variance is highest -- and the trades needed to
resolve an edge of size ``e`` from zero go as ``(1.96*sigma/e)**2``, QUADRATIC
in sigma. Six times the noise is thirty-six times the trades.

Measured sigma of per-trade gross return by pool depth, ghost book,
``scripts/evidence_power_by_depth.py`` run 2026-09-17:

    >$1M          89.6bp  (n=56)        $10k-$100k     82.6bp  (n=27)
    $100k-$1M    552.5bp  (n=70)        <$10k         526.1bp  (n=19)
    UNRESOLVED   496.6bp  (n=34)

Resolving a 20bp edge takes 77 trades in the >$1M tier and 2,932 one tier
down; at the measured 3.9 tradeable/day that is 20 days against 752. Only
27.2% of the ghost book sits in the tier where the interval is tight enough
to resolve a realistic edge, and this module's ordering is what decides that.

So the offer list is ordered by ``clear_rate * depth_weight``. It is a
RE-ORDERING AND NOT A BAN: ``MIN_CLEAR_RATE`` admission is untouched and thin
pools are still offered, just later. That is deliberate rather than timid --
banning them would destroy the control group that makes this change
measurable, and a selector that can only be compared against itself cannot be
shown to have helped.

WHAT DEPTH IS BEING USED FOR, AND WHAT IT IS NOT. It is a proxy for per-trade
VARIANCE and for nothing else. It is NOT a claim that deeper pools fill
cheaper: ``scripts/tradeable_book_repricing.py`` section 5 tested precisely
that and it FAILED -- AERO sits in a pool 4x deeper than CBBTC and costs 9x
more to round-trip at $2 (0.6395% vs 0.0728%). No route may read this
ordering as buying cheaper fills.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "trading_cache.db"

#: Round-trip cost as a fraction of notional, measured from receipts. The same
#: figure symbol_edge_gate and symbol_motion_gate test against.
ROUND_TRIP_COST = float(os.getenv("STREAMED_CANDIDATE_COST", "0.0065"))

#: The hold window a move has to happen inside. Matches the 15-minute window
#: symbol_motion_gate uses so the two cannot disagree about the same symbol.
WINDOW_SEC = float(os.getenv("STREAMED_CANDIDATE_WINDOW_SEC", "900"))

#: Share of windows that must clear the cost before a symbol is worth offering.
MIN_CLEAR_RATE = float(os.getenv("STREAMED_CANDIDATE_MIN_CLEAR_RATE", "0.03"))

#: How much feed history to judge on.
LOOKBACK_SEC = float(os.getenv("STREAMED_CANDIDATE_LOOKBACK_SEC", str(7 * 86400)))

#: A symbol must have ticked this recently to be offered at all.
FRESH_SEC = float(os.getenv("STREAMED_CANDIDATE_FRESH_SEC", "600"))

#: Ticks needed in the lookback before the clear-rate means anything.
MIN_TICKS = int(os.getenv("STREAMED_CANDIDATE_MIN_TICKS", "50"))

#: Windows needed before a rate is a rate rather than an anecdote.
MIN_WINDOWS = int(os.getenv("STREAMED_CANDIDATE_MIN_WINDOWS", "5"))

CACHE_SEC = float(os.getenv("STREAMED_CANDIDATE_CACHE_SEC", "300"))

#: Pool liquidity per symbol in USD, deepest pool on base, from DexScreener.
#: READ-ONLY here -- scripts/pool_depth_sweep.py writes it. 160 rows today, of
#: which 70 carry a number and 90 are explicitly ``null``. That ``null`` means
#: ASKED AND UNRESOLVED, which is neither "thin" nor "unasked", so it must be
#: scored as UNKNOWN and never as dust.
DEPTH_SNAPSHOT_PATH = Path(os.getenv(
    "STREAMED_CANDIDATE_DEPTH_SNAPSHOT",
    str(Path(__file__).resolve().parents[1] / "data" / "pool_depth_snapshot.json"),
))

#: HOW LONG A DEPTH READING IS WORTH TRUSTING. Seven days, for two reasons
#: that agree: the snapshot's own sweep window is 168h (``_window_hours``), and
#: LOOKBACK_SEC above judges a symbol on the same seven days -- so a depth
#: reading and a clear-rate always describe the same stretch of tape. Pools
#: drain; trusting a point-in-time reading for longer than it measured is
#: asserting more than was measured. Past this age the weighting switches OFF
#: and the ordering falls back to pure clear_rate, VISIBLY: the reason is
#: carried in every candidate's rationale and logged once.
DEPTH_MAX_AGE_SEC = float(os.getenv(
    "STREAMED_CANDIDATE_DEPTH_MAX_AGE_SEC", str(7 * 86400)))

#: Tier edges in USD. The same four scripts/tradeable_book_repricing.py uses,
#: so a sigma quoted from that script and a decision taken here can never
#: refer to different tiers.
DEPTH_TIER_DEEP_USD = float(os.getenv("STREAMED_CANDIDATE_DEPTH_DEEP_USD", "1000000"))
DEPTH_TIER_MID_USD = float(os.getenv("STREAMED_CANDIDATE_DEPTH_MID_USD", "100000"))
DEPTH_TIER_THIN_USD = float(os.getenv("STREAMED_CANDIDATE_DEPTH_THIN_USD", "10000"))

#: THE WEIGHTS -- a multiplier on clear_rate, monotone NON-DECREASING in depth.
#: A symbol in a >$1M pool keeps its full clear_rate; everything thinner is
#: discounted, so a thin symbol has to clear its cost far more often to be
#: offered ahead of a deep one. Nothing is zero: a zero is a ban wearing a
#: weight's clothes, and it would delete the control group.
#:
#: WHY NOT 1/sigma^2, WHICH IS WHAT THE POWER ARGUMENT ACTUALLY IMPLIES --
#: READ THIS BEFORE TUNING THESE NUMBERS. **The measured sigma is NOT monotone
#: in depth.** The $10k-$100k tier reads 82.6bp against the >$1M tier's 89.6bp,
#: so a strict 1/sigma^2 weighting would rank $10k-$100k FIRST and invert the
#: whole premise of this change. The judgement taken here is that n=27 makes
#: that tier's sigma a noisy estimate and that a monotone-in-depth weight is
#: the defensible shape -- but that is a JUDGEMENT, not a measurement, and it
#: is the live threat to this route. If the $10k-$100k tier holds a low sigma
#: at n>100, these weights are wrong and the thin tier deserves the evidence.
DEPTH_WEIGHT_DEEP = float(os.getenv("STREAMED_CANDIDATE_DEPTH_WEIGHT_DEEP", "1.0"))
DEPTH_WEIGHT_MID = float(os.getenv("STREAMED_CANDIDATE_DEPTH_WEIGHT_MID", "0.5"))
DEPTH_WEIGHT_THIN = float(os.getenv("STREAMED_CANDIDATE_DEPTH_WEIGHT_THIN", "0.35"))
DEPTH_WEIGHT_DUST = float(os.getenv("STREAMED_CANDIDATE_DEPTH_WEIGHT_DUST", "0.2"))

#: An UNRESOLVED depth is neither a ban nor a reward. 0.4 sits between the
#: $100k-$1M and $10k-$100k weights deliberately: an unresolved symbol is
#: usually one DexScreener could not price, which is weak evidence of thinness
#: but not proof, so it is ranked as if mid-to-thin and left able to overtake
#: on a strong clear_rate. Its measured sigma (496.6bp, n=34) is close to the
#: thin tiers', which is the only evidence available about what it contains.
DEPTH_WEIGHT_UNKNOWN = float(os.getenv("STREAMED_CANDIDATE_DEPTH_WEIGHT_UNKNOWN", "0.4"))

_log = logging.getLogger(__name__)

_cache: List["StreamedCandidate"] = []
_cache_built_at: float = 0.0

_depths: Dict[str, float] = {}
_depths_taken_at: float = 0.0
_depths_loaded_at: float = 0.0
_depths_untrusted_reason: str = "depth snapshot not loaded yet"
_depths_fallback_logged: bool = False


@dataclass
class StreamedCandidate:
    """A symbol worth offering, with the evidence that says so."""

    symbol: str
    clear_rate: float
    windows: int
    ticks: int
    last_tick_age_sec: float
    #: Deepest base-pool liquidity in USD, or None when there is no TRUSTED
    #: reading -- which covers three different states (no snapshot, a stale
    #: snapshot, a symbol the sweep could not resolve) that ``depth_note`` and
    #: ``depth_tier`` tell apart.
    depth_usd: Optional[float] = None
    depth_tier: str = "UNKNOWN"
    #: The multiplier this candidate's clear_rate was ranked by. Exactly 1.0
    #: when depth weighting is off, which is what makes the fallback identical
    #: to the old ordering rather than merely similar to it.
    depth_weight: float = 1.0
    #: Why weighting was off, empty when it was on. Carried per candidate so
    #: the fallback is legible in a log six weeks from now.
    depth_note: str = ""

    @property
    def rank_score(self) -> float:
        """WHAT THE OFFER LIST IS ORDERED BY -- clear_rate discounted by depth.

        clear_rate measures how often a move pays; depth_weight discounts how
        expensive that symbol's evidence is to interpret. A symbol is offered
        first when the product is highest, never on either alone.
        """
        return self.clear_rate * self.depth_weight

    @property
    def rationale(self) -> str:
        if self.depth_note:
            depth_part = (f"depth weighting OFF ({self.depth_note}) -- "
                          f"ranked on clear_rate alone")
        elif self.depth_usd is None:
            depth_part = (f"pool depth UNRESOLVED ({self.depth_tier}) "
                          f"x{self.depth_weight:.2f} -> rank {self.rank_score:.4f}")
        else:
            depth_part = (f"pool ${self.depth_usd:,.0f} ({self.depth_tier}) "
                          f"x{self.depth_weight:.2f} -> rank {self.rank_score:.4f}")
        return (
            f"cleared {ROUND_TRIP_COST * 100:.2f}% in "
            f"{self.clear_rate * 100:.1f}% of {self.windows} "
            f"{int(WINDOW_SEC / 60)}-minute windows over {self.ticks} ticks; "
            f"{depth_part}"
        )


def _parse_stamp(value: object) -> Optional[float]:
    """The snapshot's OWN utc stamp as epoch seconds, or None.

    Its own stamp rather than the file's mtime, because a file that was copied,
    merged or touched is not a fresh sweep, and a stale reading that LOOKS
    fresh is exactly the failure the age guard exists to prevent. mtime is the
    fallback only when the stamp is missing or unparseable.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        return parsed.timestamp()
    except (OverflowError, OSError, ValueError):
        return None


def _load_depths(now: float) -> None:
    """Parse the depth snapshot into the module cache. NEVER raises.

    Cached for CACHE_SEC so a decision cycle costs at most one 4KB read, and
    re-read after that rather than pinned for the process lifetime: the sweep
    script can rewrite the file while production is up, and production serves
    stale code for 13 minutes per restart -- a depth cache that only refreshed
    on boot would be the slowest thing in the loop to correct.
    """
    global _depths, _depths_taken_at, _depths_loaded_at, _depths_untrusted_reason
    if _depths_loaded_at and now - _depths_loaded_at <= CACHE_SEC:
        return
    _depths_loaded_at = now
    try:
        doc = json.loads(DEPTH_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise ValueError("depth snapshot is not a JSON object")
        rows: Dict[str, float] = {}
        for key, value in doc.items():
            if not isinstance(key, str) or key.startswith("_"):
                continue  # metadata, not a symbol
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue  # null == asked and UNRESOLVED -> UNKNOWN, not dust
            if float(value) < 0.0:
                continue
            rows[key.strip().upper()] = float(value)
        taken = _parse_stamp(doc.get("_taken_utc_approx"))
        if taken is None:
            taken = float(DEPTH_SNAPSHOT_PATH.stat().st_mtime)
        _depths = rows
        _depths_taken_at = taken
        _depths_untrusted_reason = (
            "" if rows else "depth snapshot carries no numeric depths")
    except Exception as exc:  # noqa: BLE001 - a missing file is not an error
        _depths = {}
        _depths_taken_at = 0.0
        _depths_untrusted_reason = (
            f"depth snapshot unreadable ({exc.__class__.__name__})")


def depth_state(now: Optional[float] = None) -> Tuple[bool, str]:
    """``(weighting_is_on, why_not)``. FAILS OPEN.

    Every way of not having a depth reading -- no file, bad JSON, empty
    snapshot, a snapshot older than DEPTH_MAX_AGE_SEC -- returns False, and
    False means every symbol is weighted 1.0, which is the pure-clear_rate
    ordering this module shipped before depth existed. The fallback is not an
    approximation of the old behaviour: ``x * 1.0 == x`` exactly, so the order
    is bit-identical.
    """
    global _depths_fallback_logged
    moment = time.time() if now is None else now
    try:
        _load_depths(moment)
    except Exception:  # noqa: BLE001 - belt and braces; _load_depths catches
        return False, "depth snapshot load raised"
    reason = _depths_untrusted_reason
    if not reason and _depths_taken_at > 0.0:
        age = moment - _depths_taken_at
        if age > DEPTH_MAX_AGE_SEC:
            reason = (f"snapshot is {age / 86400.0:.1f}d old, past the "
                      f"{DEPTH_MAX_AGE_SEC / 86400.0:.1f}d trust window")
    if reason:
        if not _depths_fallback_logged:
            _depths_fallback_logged = True
            try:
                _log.warning(
                    "streamed_symbol_candidates: depth weighting OFF (%s); "
                    "ordering by clear_rate alone", reason)
            except Exception:  # noqa: BLE001 - logging must never stop a cycle
                pass
        return False, reason
    return True, ""


def depth_of(symbol: str, now: Optional[float] = None) -> Optional[float]:
    """Deepest base-pool liquidity in USD for a symbol, or None.

    The snapshot is keyed on the BASE token ("AERO"), while a streamed symbol
    is a pair ("AERO-WETH"), so the pair is split on the first dash -- the same
    split ``atf_static_strategy._add_streamed_candidates`` already does to
    resolve the address. The whole string is tried first, because two snapshot
    keys contain spaces rather than dashes and would otherwise be unreachable.
    """
    moment = time.time() if now is None else now
    trusted, _ = depth_state(moment)
    if not trusted:
        return None
    name = str(symbol or "").strip().upper()
    if not name:
        return None
    if name in _depths:
        return _depths[name]
    base = name.split("-", 1)[0] if "-" in name else name
    return _depths.get(base)


def tier_of(depth: Optional[float]) -> str:
    """Tier name for a depth reading. None is UNRESOLVED, never a thin tier."""
    if depth is None:
        return "UNRESOLVED"
    if depth >= DEPTH_TIER_DEEP_USD:
        return ">$1M"
    if depth >= DEPTH_TIER_MID_USD:
        return "$100k-$1M"
    if depth >= DEPTH_TIER_THIN_USD:
        return "$10k-$100k"
    return "<$10k"


def weight_of_depth(depth: Optional[float]) -> float:
    """The multiplier for a depth reading. Pure: no snapshot, no clock."""
    if depth is None:
        return DEPTH_WEIGHT_UNKNOWN
    if depth >= DEPTH_TIER_DEEP_USD:
        return DEPTH_WEIGHT_DEEP
    if depth >= DEPTH_TIER_MID_USD:
        return DEPTH_WEIGHT_MID
    if depth >= DEPTH_TIER_THIN_USD:
        return DEPTH_WEIGHT_THIN
    return DEPTH_WEIGHT_DUST


def depth_weight(symbol: str, now: Optional[float] = None) -> float:
    """How much this symbol's clear_rate counts, given how deep its pool is.

    1.0 whenever depth cannot be trusted, so an unusable snapshot leaves the
    ordering exactly as it was rather than silently scrambling it.
    """
    moment = time.time() if now is None else now
    trusted, _ = depth_state(moment)
    if not trusted:
        return 1.0
    return weight_of_depth(depth_of(symbol, moment))


def _clear_rate(rows: Sequence[Tuple[float, float]]) -> Tuple[float, int]:
    """Share of non-overlapping windows whose high cleared the round trip.

    Non-overlapping because overlapping windows would count one move many
    times and inflate the rate for a symbol that moved once.

    The high is measured from the window's OPEN, not from its low: an entry
    happens at the price available when the decision is made, and a move that
    only looks profitable measured from a low nobody could have bought at is
    the selection bias this repo has shipped before.
    """
    if len(rows) < 3:
        return 0.0, 0
    cleared = 0
    windows = 0
    index = 0
    total = len(rows)
    while index < total:
        start_ts, start_price = rows[index]
        if start_price <= 0:
            index += 1
            continue
        high = start_price
        cursor = index
        while cursor < total and rows[cursor][0] <= start_ts + WINDOW_SEC:
            if rows[cursor][1] > high:
                high = rows[cursor][1]
            cursor += 1
        if cursor - index >= 3:
            windows += 1
            if (high - start_price) / start_price >= ROUND_TRIP_COST:
                cleared += 1
        index = cursor if cursor > index else index + 1
    if windows == 0:
        return 0.0, 0
    return cleared / windows, windows


def _rebuild(now: float) -> None:
    global _cache_built_at
    found: List[StreamedCandidate] = []
    # Asked ONCE per rebuild, not once per symbol: the snapshot's age must not
    # be able to change halfway down a list and order it against two rules.
    depth_trusted, depth_why = depth_state(now)
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    except Exception:  # noqa: BLE001 - no feed is not a reason to raise
        _cache_built_at = now
        return
    try:
        symbols = [
            str(row[0]) for row in conn.execute(
                "SELECT DISTINCT symbol FROM market_stream WHERE ts > ?",
                (now - FRESH_SEC,),
            ) if row and row[0]
        ]
        for symbol in symbols:
            rows = [
                (float(ts), float(price))
                for ts, price in conn.execute(
                    "SELECT ts, price FROM market_stream "
                    "WHERE symbol = ? AND ts > ? ORDER BY ts",
                    (symbol, now - LOOKBACK_SEC),
                )
                if price is not None and float(price) > 0.0
            ]
            if len(rows) < MIN_TICKS:
                continue
            rate, windows = _clear_rate(rows)
            if windows < MIN_WINDOWS or rate < MIN_CLEAR_RATE:
                continue
            candidate = StreamedCandidate(
                symbol=symbol,
                clear_rate=rate,
                windows=windows,
                ticks=len(rows),
                last_tick_age_sec=now - rows[-1][0],
            )
            if depth_trusted:
                candidate.depth_usd = depth_of(symbol, now)
                candidate.depth_tier = tier_of(candidate.depth_usd)
                candidate.depth_weight = weight_of_depth(candidate.depth_usd)
            else:
                candidate.depth_note = depth_why
            found.append(candidate)
    except Exception:  # noqa: BLE001
        _cache_built_at = now
        return
    finally:
        conn.close()

    # THE ONE LINE THAT SPENDS THE EVIDENCE BUDGET. atf_static* takes 179 of
    # ~186 decision cycles in six hours and holds 70% of all ghost evidence,
    # and it takes the top N of this list -- so this sort decides where almost
    # every ghost trade the system makes is gathered. Ordering on clear_rate
    # alone ranked HIGHEST-VARIANCE first, which is the most expensive place
    # to buy evidence. See the module docstring for the sigma table.
    #
    # Ties are unchanged: Python's sort is stable, and when depth weighting is
    # off every weight is exactly 1.0, so this reduces to the old key bit for
    # bit rather than approximately.
    found.sort(key=lambda c: c.rank_score, reverse=True)
    _cache.clear()
    _cache.extend(found)
    _cache_built_at = now


def streamed_candidates(limit: Optional[int] = None) -> List[StreamedCandidate]:
    """Symbols we already stream that have earned a look, best first.

    Fails EMPTY rather than raising: a candidate source that throws would stop
    the trading cycle, and having no suggestions is a normal state (a quiet
    market, a cold feed) rather than an error.
    """
    try:
        now = time.time()
        if now - _cache_built_at > CACHE_SEC:
            _rebuild(now)
    except Exception:  # noqa: BLE001
        return []
    return list(_cache[:limit] if limit else _cache)


def as_dicts(limit: Optional[int] = None) -> List[Dict[str, object]]:
    """The same list in the shape the bus and dashboards consume."""
    return [
        {
            "symbol": candidate.symbol,
            "clear_rate": round(candidate.clear_rate, 4),
            "windows": candidate.windows,
            "ticks": candidate.ticks,
            "last_tick_age_sec": round(candidate.last_tick_age_sec, 1),
            # Both halves of the ranking, separately, and the product. A
            # consumer that re-sorts on "clear_rate" alone would silently undo
            # the depth ordering, so rank_score is published beside it.
            "depth_usd": candidate.depth_usd,
            "depth_tier": candidate.depth_tier,
            "depth_weight": round(candidate.depth_weight, 4),
            "rank_score": round(candidate.rank_score, 6),
            "rationale": candidate.rationale,
            "source": "streamed_symbol_candidates",
        }
        for candidate in streamed_candidates(limit)
    ]
