"""Refuse symbols whose price jumps further in one tick than the stop can bind.

WHY THIS EXISTS
---------------
Measured 2026-09-06. The live lane was frozen with ES95 tail risk at 0.1241
against a 0.10 guardrail, and the ENTIRE breach was one trade:

    MOONBASE-USDC   -12.41%   27h ago
    every other trade in the 48h window   under 3%

Remove that single trade and ES95 falls to 0.0291 -- comfortably passing. One
position on one symbol was holding the whole live lane shut.

The obvious readings were both wrong:

  * "the stop is too wide" -- it is not. `GHOST_STOP_LOSS_PCT` is 0.02 and
    trading/bot.py records the REALISED loss in the exit reason, so
    "stop_loss:-0.1241" means the position was already down 12.41% by the time
    a tick arrived to evaluate a 2% stop.
  * "the feed was too sparse" -- that is real but insufficient. Sparse feeds
    (<30 ticks/hour) breach by 15.70% on average against 5.00% for dense ones,
    3.1x worse, and the two -56% catastrophes had 0 and 1 ticks in the hour
    before exit. But MOONBASE had **70 ticks** in that hour and still lost
    12.41%, so density alone does not explain it.

What does explain it is the size of a SINGLE TICK JUMP:

    symbol          ticks   p50 jump   p99 jump    2% stop holds?
    MOONBASE-USDC     618     0.000%   99381.16%   NO
    OMARCHY-USDC      155     0.056%      17.55%   NO
    BSTONK-USDC      2640     0.050%      10.91%   NO
    CP-USDC           737     0.063%       3.96%   NO
    AERO-USDC        3418     0.002%       0.76%   yes
    CBBTC-USDC       3437     0.002%       0.47%   yes

A stop is a promise about the worst case. That promise is only keepable if the
price can be observed moving through the stop level -- if one tick can carry
the price from above the stop to far below it, the stop is decoration. Against
MOONBASE's 99,381% p99 jump (feed contamination: a denomination flip, not a
real move) no stop of any width binds.

WHAT THIS GATE ASKS
-------------------
Not "has this symbol lost money" (symbol_edge_gate asks that, and needs closed
round trips it does not have for a new symbol). Not "can this symbol move
enough to pay for a round trip" (symbol_motion_gate asks that). This asks the
question neither does, and the one that MOONBASE failed:

    can this symbol's stop actually be enforced on this symbol's feed?

It is answerable BEFORE the first trade, from the price feed alone, which is
what makes it useful on exactly the symbols the book-based gates cannot judge.

BANS ONLY, NEVER PROMOTES -- the same asymmetry symbol_edge_gate documents. A
tight p99 jump does not mean a symbol is worth trading; it only means a stop
placed on it means something.

FAILS OPEN. A symbol with too few ticks to measure is ALLOWED here, because
"unmeasurable" is not "dangerous" and this gate is not the one that polices
thin feeds -- symbol_motion_gate and the ATF scout's `_feed_is_dense_enough`
already refuse those, and stacking a third refusal on the same condition would
make a quiet feed look like three independent problems.

UNMEASURABLE IS NOT THE SAME AS UNOBSERVED
------------------------------------------
That abstention was applied to the whole verdict rather than to the estimate,
and it swallowed the symbol this module's own table names as failing. OMARCHY
carries 154 ticks against a ``MIN_TICKS`` of 200, so the gate returned no
verdict at all on a feed it had already watched jump 17.55% -- 8.8x the stop --
six separate times. Measured 2026-09-06 on the 5-day ghost book, priced at the
$6 live clip and restricted to round trips the live lane's 45-minute force exit
could reproduce:

    tradeable, hold <= 45m          30 trades   net -0.4656   PF 0.765
      of which OMARCHY-USDC          3 trades   net -1.5030   PF 0.069
    the same book without OMARCHY   27 trades   net +1.0374

Three trades on one abstained-on symbol were the whole loss, and two of them
realised -14.94% and -11.19% inside 1.0 and 3.1 MINUTES against a 2% stop.

A percentile needs samples. An observed breach does not: watching the price
jump past the ceiling twice is direct evidence that a stop cannot bind there,
not an inference from a thin sample. So the sample floor now governs only how
few observations may carry a ban -- below ``MIN_TICKS`` the gate still refuses
a symbol that has breached the ceiling at least ``MIN_BREACHES`` times, and a
single stray print still cannot ban anything. Symbols at or above ``MIN_TICKS``
are judged exactly as before.
"""
from __future__ import annotations

import os
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from services.logging_service import log_message
except Exception:  # noqa: BLE001 - logging must never block a trade
    def log_message(*_args, **_kwargs):  # type: ignore[misc]
        return None

DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "trading_cache.db"

#: Ticks needed before a p99 means anything on its own. Below this the gate
#: abstains UNLESS it has directly observed ``MIN_BREACHES`` jumps past the
#: ceiling -- see "unmeasurable is not the same as unobserved" above.
MIN_TICKS = int(os.getenv("STOP_SURVIVE_MIN_TICKS", "200"))

#: How many observed jumps past the ceiling can carry a ban on a feed too thin
#: for a percentile. Two, not one: a single bad print is a print, and this repo
#: has shipped enough denomination flips to know one of them proves nothing.
MIN_BREACHES = max(2, int(os.getenv("STOP_SURVIVE_MIN_BREACHES", "2")))

#: How far back to read the feed.
WINDOW_SEC = float(os.getenv("STOP_SURVIVE_WINDOW_SEC", str(7 * 86400)))

#: The stop the p99 jump is measured against. Defaults to the same value
#: trading/bot.py enforces, so the gate and the stop cannot drift apart.
STOP_PCT = float(os.getenv("GHOST_STOP_LOSS_PCT", "0.02"))

#: How much bigger than the stop a p99 jump may be before the symbol is
#: refused. 1.0 would demand the stop survive EVERY tick in the 99th
#: percentile, which is stricter than necessary: a jump slightly larger than
#: the stop overshoots slightly, and the tail guardrail already tolerates
#: that. 2.0 refuses symbols where a single tick can double the intended
#: loss.
MAX_JUMP_RATIO = float(os.getenv("STOP_SURVIVE_MAX_JUMP_RATIO", "2.0"))

#: How far apart two stored rows may be and still count as ONE TICK.
#:
#: A "SINGLE-TICK JUMP" ACROSS A 31-HOUR HOLE IS NOT A SINGLE-TICK JUMP.
#:
#: ``_tick_jumps`` selected ``price`` and never read ``ts``, so consecutive
#: ROWS were treated as consecutive TICKS however far apart in time they were.
#: On a feed that runs at 8-19 ticks/10m and has gone dark for hours at a time
#: -- our own outages, most of them since fixed: the news crawl on the stream
#: event loop, a 3-year backfill inside data_ingest, the live gate rebuilding
#: its dataset on the feed loop -- the gaps between stored rows reach 31 hours.
#: Measured 2026-09-07 over the 7-day window, max gap per symbol:
#:
#:     AAVE-USDC      111156s (30.9h)    VIRTUAL-USDC   115584s (32.1h)
#:     MORPHO-WETH     89952s (25.0h)    LFG-USDC       183319s (50.9h)
#:
#: So the gate was charging today's entries a MULTI-DAY return and calling it
#: the jump a 2% stop has to survive between two observations. It refused all
#: 46 symbols it had enough data to judge, which -- with the thin remainder
#: abstained on -- is why entry-refused-stop-survivability was 100% of refusals
#: (7-8/h) with the aggregate live gate wide open.
#:
#: This is the same conflation trading/pipeline.py:4988 records against
#: ``sparse``: the AGE of a measurement scored as a fault in the thing
#: measured. A market that gaps and a feed that went down last Tuesday are
#: different problems, and only the first one is this gate's.
#:
#: Restricting to pairs at most this far apart, over the same window and the
#: same 4.00% ceiling (2% stop x 2.0):
#:
#:     max gap      refused   allowed        AAVE     CLANKER    SPACEX    BSTONK
#:     none (old)        46       163       4.81%      46.11%    91.67%    10.83%
#:     900s              35       174
#:     300s              24       185
#:     120s              17       192       0.25%       0.51%     0.00%     5.03%
#:      60s               6       203
#:
#: 120s, and the sensitivity above is why the number is not arbitrary: the
#: per-symbol MEDIAN gap on every symbol dense enough to be judged is 1-65s, so
#: this keeps the whole body of the distribution, and it drops the p90 tail
#: (90-1800s) which is dominated by outages rather than by the market. It also
#: sits below the shortest holding period we trade -- round trips resolving in
#: single-digit minutes -- so a jump inside it is genuinely one the stop must
#: survive while a position is open.
#:
#: THE GUARD STILL BANS WHAT IT EXISTS TO BAN. 17 symbols stay refused, and
#: they are the right ones: BSTONK-USDC 5.03%, BASEPEPE-USDC 9.67%, MEME-USDC
#: 22.36%, CBETH-CBBTC 20.69%, MOONBASE-USDC 27.78% -- the symbol this module's
#: own docstring was written about -- and the contaminated pairs JITOSOL-CBBTC
#: 715127%, EURC-WETH 238626%, VVV-WETH 227177%. Nine become POSITIVELY allowed
#: on a measured p99 rather than by abstention: ANTHROPIC-USDC 0.00%,
#: MORPHO-WETH 0.00%, SPACEX-USDC 0.00%, TIBBIR-VIRTUAL 0.04%, VIRTUAL-WETH
#: 0.07%, CLANKER-USDC 0.51%, CBETH-WETH 0.96%, LFG-USDC 2.68%,
#: BASECAT-USDC 3.32%.
#:
#: The rest fall below ``MIN_TICKS`` once long-gap pairs are dropped and land
#: on the existing abstention, which is this module's stated policy rather than
#: a new hole: "this gate is not the one that polices thin feeds --
#: symbol_motion_gate and the ATF scout's _feed_is_dense_enough already refuse
#: those". The breach counter is filtered with the percentile deliberately: a
#: breach observed across a 31-hour hole is the same non-evidence as a
#: percentile computed from one, and letting it ban while the percentile may
#: not would put the old bug back through the thin-feed door.
#:
#: WHY BASECAT-USDC IS ALLOWED AND BSTONK-USDC IS NOT -- ANSWERED, do not
#: re-measure it. A 7-day census counting ADJACENT ROWS reported BASECAT at a
#: p99 of 5.559% with 31 jumps above 5%, against this module's 4.00% ceiling,
#: while ``refusal_reason('BASECAT-USDC')`` returned None, and the gate was
#: suspected of reading too short a window. IT IS NOT A WINDOW DISAGREEMENT:
#: ``WINDOW_SEC`` is 604800.0 and the gate reads the SAME seven days. The whole
#: difference is this constant. Re-measured 2026-09-10 over one 7d window,
#: splitting each symbol's >5% jumps by the gap they span:
#:
#:     symbol         pairs   p99 by row   p99 <=120s   >5% jumps   <=120s
#:     BASECAT-USDC    1809       5.272%       2.646%          23        0
#:     BSTONK-USDC     1455      12.318%       4.788%          79       10
#:     AERO-USDC       3668       0.862%       0.426%           1        0
#:
#: BASECAT's 23 large jumps span a MINIMUM of 281s and a median of 2135s;
#: inside 120s its largest move all week is 4.472%. AERO's single large jump
#: spans 22 hours. BSTONK's do not need a gap -- ten land between ticks as
#: little as 17s apart, which is why its capped p99 stays at 4.788% and it is
#: refused. The 120s measure is the right one: a stop is only enforceable on a
#: tick that ARRIVES, so a 5% move accumulated over 35 minutes of feed silence
#: is not a move the stop failed to bind on, it is one nobody was there to
#: evaluate. The discriminator is load-bearing rather than cosmetic -- it is
#: what separates the symbol that booked six 17-25% gross ghost rows from the
#: one that booked a single row, and row-adjacent p99 does not separate them.
#: Locked in by test_a_jump_nobody_could_trade_through_cannot_ban_a_symbol.py.
MAX_TICK_GAP_SEC = float(os.getenv("STOP_SURVIVE_MAX_TICK_GAP_SEC", "120"))

#: Symbols never refused regardless of feed. Empty by default.
NEVER_BAN = {
    s.strip().upper()
    for s in os.getenv("STOP_SURVIVE_NEVER_BAN", "").split(",")
    if s.strip()
}

CACHE_SEC = float(os.getenv("STOP_SURVIVE_CACHE_SEC", "600"))

_cache: Dict[str, Tuple[float, str]] = {}
_cache_built_at: float = 0.0


def _tick_jumps(conn: sqlite3.Connection, symbol: str,
                since: float) -> List[float]:
    """Absolute fractional price change between ticks ADJACENT IN TIME.

    Fractional rather than absolute because the stop is a fraction: a $0.01
    move means something entirely different on CBBTC than on a sub-cent
    memecoin, and comparing dollars to a percentage is the units error this
    repo has shipped more than once.

    Pairs separated by more than ``MAX_TICK_GAP_SEC`` are dropped rather than
    measured. This function used to select ``price`` alone, so it could not
    tell a 4% move in 30 seconds from a 4% move over 31 hours and scored both
    as one tick. See ``MAX_TICK_GAP_SEC`` for the measurement.
    """
    points: List[Tuple[float, float]] = []
    try:
        rows = conn.execute(
            "SELECT ts, price FROM market_stream WHERE symbol = ? AND ts > ? "
            "ORDER BY ts",
            (symbol, since),
        )
        for stamp, price in rows:
            try:
                value = float(price)
                when = float(stamp)
            except (TypeError, ValueError):
                continue
            # A row with no usable timestamp cannot be shown to be adjacent to
            # anything, and this gate bans on adjacency. Dropping it costs one
            # sample; keeping it would reintroduce the unbounded-gap pair the
            # cap exists to remove.
            if value > 0.0 and when == when:
                points.append((when, value))
    except Exception:  # noqa: BLE001
        return []

    gap_cap = MAX_TICK_GAP_SEC if MAX_TICK_GAP_SEC > 0 else float("inf")
    jumps: List[float] = []
    for index in range(len(points) - 1):
        previous = points[index][1]
        if previous <= 0.0:
            continue
        elapsed = points[index + 1][0] - points[index][0]
        if elapsed < 0.0 or elapsed > gap_cap:
            continue
        jump = abs(points[index + 1][1] - previous) / previous
        # A jump is only meaningful if it is finite. Feed contamination can
        # produce inf/nan, and those must not poison a percentile.
        if jump == jump and jump != float("inf"):
            jumps.append(jump)
    return jumps


def _percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))
    return ordered[index]


def _rebuild(now: float) -> None:
    global _cache_built_at
    verdicts: Dict[str, Tuple[float, str]] = {}
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    except Exception:  # noqa: BLE001 - no feed is not a reason to block trading
        _cache_built_at = now
        return

    try:
        since = now - WINDOW_SEC
        symbols = [
            str(row[0]) for row in conn.execute(
                "SELECT DISTINCT symbol FROM market_stream WHERE ts > ?",
                (since,),
            ) if row and row[0]
        ]
        ceiling = STOP_PCT * MAX_JUMP_RATIO
        for symbol in symbols:
            if symbol.upper() in NEVER_BAN:
                continue
            jumps = _tick_jumps(conn, symbol, since)
            if not jumps:
                continue
            breaches = sum(1 for jump in jumps if jump > ceiling)
            if len(jumps) < MIN_TICKS and breaches < MIN_BREACHES:
                # Too thin to estimate a percentile AND nothing directly
                # observed past the ceiling. Abstain -- see the module
                # docstring on why this gate does not police thin feeds.
                continue
            p99 = _percentile(jumps, 0.99)
            if p99 <= ceiling:
                continue
            thin = " (thin feed; %d observed breaches)" % breaches if len(jumps) < MIN_TICKS else ""
            verdicts[symbol.upper()] = (
                p99,
                f"p99 single-tick jump {p99 * 100:.2f}% over {len(jumps)} "
                f"ticks exceeds {ceiling * 100:.2f}% "
                f"({STOP_PCT * 100:.2f}% stop x {MAX_JUMP_RATIO:.1f}); "
                f"a stop cannot bind on this feed{thin}",
            )
    except Exception:  # noqa: BLE001
        _cache_built_at = now
        return
    finally:
        conn.close()

    for symbol, (_p99, detail) in sorted(verdicts.items()):
        if symbol not in _cache:
            log_message(
                "trading",
                f"STOP SURVIVABILITY GATE: refusing {symbol} -- {detail}",
                severity="warning",
            )
    _cache.clear()
    _cache.update(verdicts)
    _cache_built_at = now


def refusal_reason(symbol: str) -> Optional[str]:
    """Why a stop cannot be enforced on this symbol, or None to allow it.

    Fails OPEN on any error: a gate that cannot read the feed has no evidence
    to refuse on, and blocking every symbol because a database was locked
    would be far worse than letting one bad trade through.
    """
    try:
        name = str(symbol or "").strip().upper()
        if not name:
            return None
        now = time.time()
        if now - _cache_built_at > CACHE_SEC:
            _rebuild(now)
        verdict = _cache.get(name)
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
    return {symbol: detail for symbol, (_p99, detail) in _cache.items()}
