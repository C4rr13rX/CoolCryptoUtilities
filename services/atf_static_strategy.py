from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from db import get_db
from services.watchlists import load_watchlists, save_watchlists
from trading.portfolio import PortfolioState, STABLE_TOKENS


REPO_ROOT = Path(__file__).resolve().parents[1]
SIGNAL_KEY = "atf_static_strategy:signals"
LATEST_KEY = "atf_static_strategy:latest"
PENDING_BUS_KEY = "atf_static_strategy:pending_bus_actions"
FEEDBACK_KEY = "atf_static_strategy:feedback"
GHOST_POSITIONS_KEY = "atf_static_strategy:ghost_positions"
try:
    from services.symbol_edge_gate import refusal_reason as _symbol_edge_refusal
except Exception:  # noqa: BLE001 - a missing gate must not stop the scout
    def _symbol_edge_refusal(_symbol: str):  # type: ignore[misc]
        return None
try:
    from services.symbol_motion_gate import refusal_reason as _symbol_motion_refusal
except Exception:  # noqa: BLE001 - a missing gate must not stop the scout
    def _symbol_motion_refusal(_symbol: str):  # type: ignore[misc]
        return None
try:
    from services.stop_survivability_gate import (
        refusal_reason as _stop_survivability_refusal,
    )
except Exception:  # noqa: BLE001 - a missing gate must not stop the scout
    def _stop_survivability_refusal(_symbol: str):  # type: ignore[misc]
        return None
try:
    from services.strategy_edge_gate import refusal_reason as _strategy_edge_refusal
except Exception:  # noqa: BLE001 - a missing gate must not stop the scout
    def _strategy_edge_refusal(_strategy_id: str):  # type: ignore[misc]
        return None

try:
    from services.roundtrip_cost import (
        ghost_clip_usd as _ghost_clip_usd,
        net_profit_usd as _net_profit_usd,
        roundtrip_cost_rate as _roundtrip_cost_rate,
        roundtrip_cost_usd as _roundtrip_cost_usd,
    )
except Exception:  # noqa: BLE001 - never let a costing import stop the scout
    def _ghost_clip_usd() -> float:  # type: ignore[misc]
        return 2.0

    def _roundtrip_cost_usd(notional: float | None = None) -> float:  # type: ignore[misc]
        return 0.004047 + 0.003187 * float(notional or 2.0)

    def _roundtrip_cost_rate(notional: float | None = None) -> float:  # type: ignore[misc]
        n = float(notional or 2.0)
        return max(_roundtrip_cost_usd(n) / n, 0.003187)

    def _net_profit_usd(return_pct: float, notional: float | None = None) -> float:  # type: ignore[misc]
        n = float(notional or 2.0)
        return float(return_pct) * n - _roundtrip_cost_usd(n)

SOURCE = "c0d3rv2_atf_static"

#: Ledger identity for trades this module opens and closes ITSELF.
#:
#: Deliberately NOT "atf_static". Two different executors trade the ATF
#: signals: this scout, and ``trading/strategies/atf_static.py`` running
#: inside the bot. They share a signal source and nothing else -- the scout
#: enters on its own corroborated quote and exits on an 8% stop, a 1h hold
#: or its target, while the bot enters through the CDCL solver and exits on
#: triggers, a 2% stop, confidence drops and timed exits. Same entry idea,
#: entirely different realised P/L.
#:
#: They were reporting into ONE ledger id, and that is the root cause of
#: link 9. Measured 2026-09-02 over the whole database: of 376 closed
#: ``atf_static`` trades, **368 (97.9%) were taken by this scout** and 8 by
#: the bot. The scout hardcodes ``wallet="ghost"`` and ``mode="ghost"`` and
#: publishes ``live_execution_enabled: False`` -- it has no live branch and
#: cannot spend money at all. So ``atf_static`` was granted ``live_approved``
#: on the record of an executor that can never place a live trade, while the
#: executor that CAN place one had 8 trades against the 20 promotion needs.
#:
#: The bot then reached its live entry gate 264 times in 24h and the swap
#: guard PASSED 94 of them -- every one downgraded to ghost, because the only
#: graduated strategy never produces the directives that arrive there. The
#: one strategy allowed to spend could not, and the ones that could were not
#: allowed to. Splitting the id is what makes the ledger mean what the live
#: gate reads it to mean.
SCOUT_STRATEGY_ID = "atf_static_scout"

#: The signal identity. The bot-side plugin publishes under this and it is
#: what the live gate consults, so nothing this module executes may claim it.
SIGNAL_STRATEGY_ID = "atf_static"


def _record_ghost_outcome(strategy_id: str, profit: float, symbol: str = "") -> None:
    """
    Report a closed ghost trade to the strategy ledger.

    This loop runs its own ghost cycle rather than going through
    ``bot.py``'s exit path, which is the only other place that calls
    ``StrategyLedger.record()``. Without this the outcomes were written to
    ``trading_ops`` and nowhere else: 196 closed trades over four days that
    the graduation gate never saw, so the ledger sat unchanged and no
    strategy could ever accumulate the 20 trades promotion requires.

    Callers must pass ``SCOUT_STRATEGY_ID``. See its docstring for why an
    outcome this module produced must never be filed under the id the live
    gate reads.

    Deliberately best-effort. A ledger write must never abort a trading
    cycle -- losing one outcome is recoverable, stalling the loop is not.
    """
    try:
        from trading.strategies.ledger import StrategyLedger

        StrategyLedger().record(
            strategy_id, profit=float(profit), mode="ghost", symbol=symbol
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[atf-static] ledger record failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _now() -> float:
    return time.time()


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or default).strip().lower() in {"1", "true", "yes", "on"}


def _feed_price(db: Any, symbol: str, chain: str, max_age_sec: float) -> Optional[float]:
    """Representative streamed price for ``symbol``, or None if the feed is silent.

    Uses the MEDIAN of recent ticks rather than the single latest one.
    ``market_stream`` carries interleaved sources, and when one of them
    publishes a different denomination the series alternates between correct
    and wrong values on a scale of minutes. Observed 2026-08-27 against
    DexScreener ($29M liquidity) as ground truth:

        AERO-USDC  truth 0.5153  feed alternating 0.5138 and 1.14   (2.2x)
        MAMO-USDC  truth 0.01055 feed alternating 0.0105 and 0.1723 (16x)

    Comparing a good quote against whichever tick happened to land last made
    corroboration a coin flip -- it refused 100% of live signals, so no
    position could open at all. The median ignores a minority of bad ticks
    while still going silent when the whole series is wrong.
    """
    window = max(max_age_sec, 0.0)
    try:
        rows = db.recent_market_prices(symbol, chain, since_ts=_now() - window, limit=25)
    except Exception:
        rows = None
    prices: List[float] = []
    if rows:
        for row in rows:
            try:
                value = _float(row[0] if isinstance(row, (list, tuple)) else row, 0.0)
            except Exception:
                continue
            if value > 0.0:
                prices.append(value)
    if not prices:
        # Fall back to the single-tick lookup when the batch helper is absent.
        try:
            row = db.get_market_price(symbol, chain, ts=_now() - window, after=True)
        except Exception:
            return None
        if not row:
            return None
        price = _float(row[0], 0.0)
        return price if price > 0.0 else None
    prices.sort()
    mid = len(prices) // 2
    if len(prices) % 2:
        return prices[mid]
    return (prices[mid - 1] + prices[mid]) / 2.0


def _symbol_specific_hole(
    db: Any,
    symbol: str,
    chain: str,
    stamps: List[float],
    max_hole: float,
) -> float:
    """Longest stretch this symbol was silent *while the feed was up*, seconds.

    Every gap in ``stamps`` is re-measured against the rest of the feed. Time
    inside a gap when nothing ticked anywhere is a pipeline outage and is not
    charged to the symbol; time when other symbols were ticking normally is
    lost coverage and is.

    Feed liveness is bucketed at ``ATF_STATIC_FEED_LIVE_BUCKET_SEC`` (60s by
    default) and the answer is the longest RUN of consecutive live buckets in
    which this symbol produced nothing. Bucketing keeps the measure stable
    against the feed's own jitter -- a one-tick blip does not make the feed
    "up", and a two-second lull does not make it "down".

    Fails CLOSED on any error: returns the raw gap, so a database that cannot
    answer leaves the original guard exactly as strict as it was.
    """
    raw = 0.0
    for index in range(len(stamps) - 1):
        raw = max(raw, stamps[index + 1] - stamps[index])
    if raw <= max_hole:
        return raw
    bucket = max(1.0, _float_env("ATF_STATIC_FEED_LIVE_BUCKET_SEC", 60.0))
    worst = 0.0
    try:
        for index in range(len(stamps) - 1):
            start, end = stamps[index], stamps[index + 1]
            if end - start <= max_hole:
                continue
            others = db.feed_tick_times(
                chain, since_ts=start, until_ts=end, exclude_symbol=symbol
            )
            if not others:
                continue        # nothing ticked anywhere: an outage, not a hole
            live = sorted({int((tick - start) // bucket) for tick in others})
            run_start = previous = live[0]
            for slot in live[1:]:
                if slot != previous + 1:
                    worst = max(worst, (previous - run_start + 1) * bucket)
                    run_start = slot
                previous = slot
            worst = max(worst, (previous - run_start + 1) * bucket)
    except Exception:  # noqa: BLE001 - cannot check is not the same as safe
        return raw
    return worst


def _feed_coverage_ratio(
    db: Any,
    symbol: str,
    chain: str,
    stamps: List[float],
    window: float,
) -> Optional[float]:
    """Share of the feed's LIVE minutes in which this symbol also ticked.

    The question every other test here is trying to ask, asked directly. A
    stop can only fire on a tick, so what matters is how often we get to look
    at the symbol -- not the typical spacing between looks, and not the single
    worst gap.

    Both of the existing measures are defeated by the same shape, a symbol
    that ticks in bursts. BPAD-USDC at its 2026-09-05 17:48:02 live entry:

        22 ticks in the trailing hour, median gap 1.3s        -> passes
        max gap 2864s                                          -> escalates
        longest consecutive live-bucket run inside it: 300s    -> passes

    Twenty of those 22 ticks were one 90-second burst. The median describes
    the burst, not the coverage. And ``_symbol_specific_hole`` scores the
    escalation by the longest run of CONSECUTIVE live buckets, which needs the
    rest of the feed to be continuously up -- it never is here. During BPAD's
    2864s of silence the feed delivered 426 ticks, but they landed in only 12
    of the 47 minutes, so the longest unbroken run was 300s and a 47-minute
    symbol blackout scored as five. The entry was allowed. The position then
    went 1678s unpriced, and the first tick after the hole fired the stop at
    -19.21% against a 1.5% stop, closing -0.2549 on a $1.50 clip -- larger
    than the entire live book, which was -0.1864 over 18 round trips and
    +0.0686 without it.

    Counting TOTAL live minutes rather than consecutive ones is what fixes
    this: 12 of 47, and the ratio is scored against the feed's own liveness so
    a pipeline outage cancels from both sides. If nothing ticked anywhere,
    those minutes are in neither total and the symbol is not charged for them
    -- the property ``_symbol_specific_hole`` was written to protect, kept.

    Returns None when the feed's liveness cannot be established, which the
    caller treats as "no opinion" rather than as a refusal.
    """
    if not stamps:
        return None
    start = _now() - window
    bucket = max(1.0, _float_env("ATF_STATIC_FEED_LIVE_BUCKET_SEC", 60.0))
    try:
        others = db.feed_tick_times(
            chain, since_ts=start, until_ts=_now(), exclude_symbol=symbol
        )
    except Exception:  # noqa: BLE001 - cannot measure is not the same as sparse
        return None
    mine = {int((tick - start) // bucket) for tick in stamps if tick >= start}
    live = {int((tick - start) // bucket) for tick in (others or [])} | mine
    if not live:
        return None
    return len(mine) / len(live)


def _feed_is_dense_enough(db: Any, symbol: str, chain: str) -> bool:
    """Can a stop-loss actually be enforced on this symbol?

    The stop is evaluated when a tick arrives. If ticks are minutes or hours
    apart, price gaps past the stop and the realised loss is unbounded --
    measured 2026-08-27, 4 of 6 stop_loss exits breached an 8% stop (worst
    -22.2% on SOL-USDC, which had a 663-minute hole between ticks).

    Requires BOTH a minimum sample count and a recent median gap below the
    ceiling. Median rather than max: one long outage should not disqualify a
    symbol that is otherwise well covered, but a consistently thin feed should.
    """
    if not _bool_env("ATF_STATIC_REQUIRE_DENSE_FEED", "1"):
        return True
    window = _float_env("ATF_STATIC_FEED_DENSITY_WINDOW_SEC", 3600.0)
    max_gap = _float_env("ATF_STATIC_MAX_MEDIAN_TICK_GAP_SEC", 300.0)
    min_ticks = int(_float_env("ATF_STATIC_MIN_TICKS_FOR_ENTRY", 6))
    try:
        rows = db.recent_market_prices(
            symbol, chain, since_ts=_now() - window, limit=200
        )
    except Exception:
        return False
    stamps = sorted(float(r[1]) for r in (rows or []) if len(r) > 1)
    if len(stamps) < max(2, min_ticks):
        return False
    gaps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    if not gaps:
        return False
    gaps.sort()
    mid = len(gaps) // 2
    median_gap = gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2.0
    if median_gap > max_gap:
        return False
    # A healthy median is not enough: the stop is breached by the WORST gap,
    # not the typical one.
    #
    # Measured 2026-08-28 on every stop_loss exit in the ledger -- each breach
    # coincided with a single long hole in the feed while the position was
    # open, even though entry-time density looked fine:
    #
    #   BSTONK-USDC    -8.39%  20 ticks @ 42s median at entry, then a 10.9min
    #                          hole during the hold (2 ticks total)
    #   BASEJUICE-USDC -8.10%  29.8min hole
    #   BASECAT-USDC   -8.52%   9.6min hole
    #
    # Those three breaches are the whole of the tail: they held ES95 at 0.0834
    # against a 0.08 guardrail and blocked live trading entirely. The median
    # test passed all three, because one long hole barely moves a median.
    #
    # So bound the tail directly. A feed that has recently gone quiet for
    # longer than the stop can survive is a feed that cannot enforce the stop,
    # regardless of how good it looks on average.
    #
    # The budget is calibrated, not guessed. Sampled across the 1h windows the
    # gate actually reads, on 2026-08-27 12:00-20:00 (an uninterrupted
    # stretch), healthy symbols carried these median max-gaps:
    #
    #   CBBTC 624s   AERO 1052s   MAMO 770s   BASECAT 567s   BSTONK 983s
    #
    # Normal jitter therefore reaches ~1000s even on feeds that are fine. A
    # 600s budget would refuse every symbol permanently -- trading one bug for
    # a worse one. 1200s clears that jitter while still catching the holes
    # that actually breached the stop (BASEJUICE 1788s, and the hold-time
    # holes that produced all three breaches).
    # A HOLE ONLY INDICTS THE SYMBOL IF THE FEED WAS UP DURING IT.
    #
    # This test read the raw gap, so a single pipeline-wide outage was charged
    # against every symbol separately. Measured 2026-09-05 09:30, the largest
    # 1h gap for nine unrelated symbols:
    #
    #   AERO-USDC    2347s  08:32:34 -> 09:11:42
    #   CBETH-USDC   2421s  08:31:20 -> 09:11:42
    #   CBBTC-USDC   1387s  08:49:20 -> 09:12:27
    #   CBDOGE-USDC  1384s  08:48:38 -> 09:11:42
    #   BSTONK-USDC  1489s  08:46:53 -> 09:11:42
    #   CBXRP-USDC   1488s  08:46:53 -> 09:11:41
    #   BASECAT-USDC 1389s  08:48:32 -> 09:11:42
    #   COMP-USDC    1410s  08:48:50 -> 09:12:20
    #   DAI-USDC     2407s  08:32:35 -> 09:12:42
    #
    # Every one ends within a second of 09:11:42, across the production restart
    # at 09:00:03. It is ONE outage, not nine coverage failures -- and these are
    # the densest feeds we have (AERO 4.7s median, CBBTC 2.6s, CBETH 1.8s).
    #
    # Charging it per symbol refused 32 of 36 symbols; the only four that passed
    # were ones whose history was too short to contain the outage. Because the
    # window is an hour long, every restart then blocked EVERY entry for the
    # following hour -- a self-inflicted trading blackout, and the mechanical
    # reason the day booked zero live trades.
    #
    # So measure the hole over the time the feed was actually UP. If other
    # symbols were ticking while this one was silent, that is exactly the lost
    # coverage the guard was written to catch and it still refuses. If nothing
    # ticked anywhere, no stop could have been enforced on any symbol and the
    # gap is evidence about the pipeline, not about this symbol.
    max_hole = _float_env("ATF_STATIC_MAX_TICK_HOLE_SEC", 1200.0)
    if max_hole > 0.0 and gaps[-1] > max_hole:
        if _symbol_specific_hole(db, symbol, chain, stamps, max_hole) > max_hole:
            return False
    # The feed must also be live NOW, not merely dense in aggregate: a window
    # that ended twenty minutes ago describes a feed that has already stopped.
    # Budgeted separately so disabling the hole check does not also disable
    # the staleness check -- they answer different questions.
    max_stale = _float_env("ATF_STATIC_MAX_FEED_STALENESS_SEC", 1200.0)
    if max_stale > 0.0 and stamps and (_now() - stamps[-1]) > max_stale:
        return False
    # HOW OFTEN DO WE GET TO LOOK AT THIS SYMBOL AT ALL?
    #
    # The three tests above are all defeated by a bursty feed -- see
    # ``_feed_coverage_ratio`` for the BPAD-USDC entry that passed every one
    # of them on 22 ticks that were really one 90-second burst, then lost
    # -0.2549 through a 1678s hole. This asks the question directly.
    #
    # Calibrated on the book rather than chosen. Coverage measured at the
    # entry instant of all 18 live round trips, sorted:
    #
    #   21.6% -0.01688   25.7% -0.02514   27.3% -0.25493   27.8% -0.02142
    #   ------------------------------- 30% -------------------------------
    #   34.1% -0.00544   38.5% +0.24203   45.2% -0.02790   45.2% -0.02352
    #   47.2% -0.01908   50.0% -0.00308   51.2% +0.01039   56.0% -0.01318
    #   60.0% -0.00190   60.0% +0.00211   60.6% -0.00492   63.0% -0.03324
    #   81.0% +0.00187   81.0% +0.00784
    #
    # Every trade below 30% lost money, four for four, and they carry -0.3184
    # of the book's -0.1864. The threshold sits in the empty band between
    # 27.8% and 34.1% -- the widest gap in the distribution -- rather than on
    # a value tuned to a P/L outcome. Cutting at 35% or 40% instead would book
    # a better or worse backtest purely by including or excluding BSTONK's
    # +0.242 at 38.5%, which is fitting to one trade.
    #
    # Honest limit: n=18, and the size of the improvement rests on two tail
    # trades. The SIGN is what this rests on -- 4 of 4 below the line lost --
    # together with the mechanism, which is not statistical. A stop fires on a
    # tick; a symbol we see in a fifth of the live minutes has no enforceable
    # stop, and the 7d measurement across 24,053 inter-tick gaps on traded
    # symbols shows the p90 move reaching 6.16% by 900s of silence against a
    # 1.5% stop.
    #
    # Fails OPEN, unlike the tests above. If the feed's own liveness cannot be
    # read, the other three checks have already passed and this leaves the
    # gate exactly as strict as it was before -- a measurement failure must
    # not become a silent trading blackout.
    min_coverage = _float_env("ATF_STATIC_MIN_FEED_COVERAGE", 0.30)
    if min_coverage > 0.0:
        coverage = _feed_coverage_ratio(db, symbol, chain, stamps, window)
        if coverage is not None and coverage < min_coverage:
            return False
    return True


def _corroborated_price(
    db: Any,
    symbol: str,
    chain: str,
    quoted: float,
) -> Optional[float]:
    """Validate a DexScreener quote against the streamed feed.

    Positions were being opened and marked purely from ``signal["price_usd"]``,
    which no guard had ever checked. Observed 2026-08-27: BASELIFE-USDC entered
    at 2.05e-07 and "exited" at 4.39e-06 for +2038% -- on a symbol with ZERO
    rows in ``market_stream``. That single fabricated fill was 81% of the
    strategy's entire net profit and carried its ghost record to graduation.

    A quote is only usable when the feed both (a) has a recent tick for the
    symbol at all, and (b) agrees with the quote to within
    ``ATF_STATIC_MAX_FEED_DEV``. Anything else is priced from nothing and must
    not become a trade -- the same rule the synthetic-tick guard already
    applies to the stream itself.

    Returns the price to use, or None to skip the symbol entirely.
    """
    if quoted <= 0.0:
        return None
    if not _bool_env("ATF_STATIC_REQUIRE_FEED_PRICE", "1"):
        return quoted
    max_age = _float_env("ATF_STATIC_FEED_MAX_AGE_SEC", 900.0)
    feed = _feed_price(db, symbol, chain, max_age)
    if feed is None:
        return None
    max_dev = _float_env("ATF_STATIC_MAX_FEED_DEV", 0.35)
    if max_dev > 0.0:
        deviation = abs(quoted - feed) / feed
        if deviation > max_dev:
            return None
    # Prefer the feed: it is the corroborated number, and marking against it
    # keeps entry and exit on the same price basis.
    return feed


def refresh_feedback_scores(*, max_age_sec: float = 6 * 3600.0) -> Dict[str, Any]:
    """
    Feed ghost/live outcomes back into ATF's next candidate scoring pass.

    This is intentionally pair-level and model-agnostic: C0D3R/ATF publishes
    candidates, the existing ghost/live machinery produces outcomes, and this
    function converts those outcomes into small scheduler knobs instead of
    hiding failures.
    """
    db = get_db()
    since = _now() - max(300.0, float(max_age_sec))
    rows = db.fetch_trades(limit=int(os.getenv("ATF_STATIC_FEEDBACK_TRADE_LIMIT", "500")), since_ts=since)
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        details = row.get("details") if isinstance(row.get("details"), dict) else {}
        status = str(row.get("status") or details.get("status") or "").lower()
        action = str(row.get("action") or details.get("action") or "").lower()
        reason = str(details.get("reason") or "")
        sid = str(details.get("strategy_id") or details.get("strategy") or "")
        # Both executors of the ATF signals feed this pair-level scoring: the
        # scout's own round trips and the bot's. Splitting the ledger ids kept
        # the two apart where it decides who may spend money; here the
        # question is "how has this PAIR behaved", and both are evidence of
        # that. Matching only the bot's id would have silently emptied the
        # feedback loop the moment the scout was renamed -- 368 of the 376
        # closed trades are the scout's.
        if (
            sid not in {SIGNAL_STRATEGY_ID, SCOUT_STRATEGY_ID}
            and "ATF researched candidate" not in reason
        ):
            continue
        if action != "exit" and not status.endswith("-exit"):
            continue
        symbol = str(row.get("symbol") or details.get("symbol") or "").upper()
        if not symbol:
            continue
        try:
            profit = float(details.get("profit") or 0.0)
        except Exception:
            profit = 0.0
        ent = by_symbol.setdefault(symbol, {"symbol": symbol, "trades": 0, "wins": 0, "losses": 0, "profit": 0.0})
        ent["trades"] += 1
        ent["profit"] += profit
        if profit > 0:
            ent["wins"] += 1
        else:
            ent["losses"] += 1

    scores: Dict[str, Dict[str, Any]] = {}
    for symbol, ent in by_symbol.items():
        trades = max(1, int(ent["trades"]))
        wins = int(ent["wins"])
        win_rate = wins / trades
        profit = float(ent["profit"])
        # Conservative until there is a sample. Positive performers get more
        # allocation/priority; losers get throttled but remain visible.
        multiplier = 1.0
        priority = 0
        if trades >= 3:
            if win_rate >= 0.58 and profit > 0:
                multiplier = min(1.75, 1.0 + (win_rate - 0.5) + min(profit / 10.0, 0.5))
                priority = 8
            elif win_rate <= 0.42 or profit < 0:
                multiplier = max(0.25, 1.0 - (0.5 - win_rate) - min(abs(profit) / 10.0, 0.5))
                priority = -8
        scores[symbol] = {
            **ent,
            "win_rate": round(win_rate, 6),
            "profit": round(profit, 8),
            "allocation_multiplier": round(multiplier, 6),
            "priority": priority,
            "updated": _now(),
        }
        try:
            db.upsert_pair_adjustment(
                symbol,
                allocation_multiplier=multiplier,
                size_multiplier=max(0.25, min(1.75, multiplier)),
                priority=priority,
                details={"source": SOURCE, "feedback": scores[symbol]},
            )
        except Exception:
            pass

    payload = {"source": SOURCE, "ts": _now(), "max_age_sec": max_age_sec, "scores": scores}
    db.set_json(FEEDBACK_KEY, payload)
    return payload


def _feedback_for(symbol: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
    scores = feedback.get("scores") if isinstance(feedback, dict) else {}
    if not isinstance(scores, dict):
        return {}
    return scores.get(symbol.upper()) or {}


def _stable_source(portfolio: PortfolioState, chain: str) -> tuple[str, float]:
    portfolio.refresh(force=True)
    best_symbol = "USDC"
    best_qty = 0.0
    for (holding_chain, symbol), holding in portfolio.holdings.items():
        if holding_chain != chain.lower():
            continue
        if symbol.upper() not in STABLE_TOKENS and symbol.upper() not in {"USDBC", "USDC.E"}:
            continue
        qty = float(holding.quantity or 0.0)
        usd = float(holding.usd or qty)
        if usd > best_qty:
            best_symbol = symbol.upper()
            best_qty = qty
    return best_symbol, best_qty


def _quote_probe(
    *,
    chain: str,
    sell_token: str,
    buy_token: str,
    amount: float,
    from_address: str,
    slippage_bps: int,
    timeout_sec: int = 45,
) -> Dict[str, Any]:
    payload = {
        "chain": chain,
        "sell_token": sell_token,
        "buy_token": buy_token,
        "amount": f"{amount:.8f}",
        "from_address": from_address,
        "slippage_bps": slippage_bps,
    }
    cmd = [sys.executable, "-u", "main.py", "--action", "swap_quote", "--payload", json.dumps(payload)]
    started = _now()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            # The child inherits PYTHONUTF8 (services/utf8_mode.py) so it writes
            # UTF-8. Without these, `text=True` decodes with the locale encoding
            # -- cp1252 here -- and a non-ASCII token symbol in the quote log
            # came back as mojibake. `replace` keeps a partially undecodable
            # tail from raising in the parent: a quote must fail on its merits,
            # never on the encoding of the log that describes it.
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
        )
        output = proc.stdout or ""
        return {
            "ok": proc.returncode == 0 and "No quote providers available" not in output,
            "returncode": proc.returncode,
            "duration_sec": round(_now() - started, 3),
            "payload": payload,
            "output_tail": output[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "duration_sec": round(_now() - started, 3),
            "payload": payload,
            "error": f"quote_timeout:{exc}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": None,
            "duration_sec": round(_now() - started, 3),
            "payload": payload,
            "error": str(exc),
        }


def _run_ghost_quote_scout(
    *,
    db: Any,
    signals: List[Dict[str, Any]],
    chain: str,
    quote_token: str,
    max_positions: int,
) -> Dict[str, Any]:
    """Ghost-only ATF entries/exits from verified quote-probed candidates."""
    if not _bool_env("ATF_STATIC_QUOTE_GHOST_SCOUT_ENABLED", "1"):
        return {"enabled": False}
    now = _now()
    try:
        positions = db.get_json(GHOST_POSITIONS_KEY) or {}
    except Exception:
        positions = {}
    if not isinstance(positions, dict):
        positions = {}
    try:
        max_hold_sec = max(60.0, min(24 * 3600.0, float(os.getenv("ATF_STATIC_GHOST_MAX_HOLD_SEC", "3600"))))
    except Exception:
        max_hold_sec = 3600.0
    try:
        stop_loss = max(0.001, min(0.50, float(os.getenv("ATF_STATIC_GHOST_STOP_LOSS", "0.08"))))
    except Exception:
        stop_loss = 0.08
    try:
        min_profit = max(0.0, min(0.50, float(os.getenv("ATF_STATIC_GHOST_MIN_EXIT_PROFIT", "0.005"))))
    except Exception:
        min_profit = 0.005

    by_symbol = {
        str(sig.get("symbol") or "").upper(): sig
        for sig in signals
        if isinstance(sig, dict) and str(sig.get("symbol") or "").strip()
    }
    events: List[Dict[str, Any]] = []
    skipped_unpriced: List[str] = []
    skipped_sparse_feed: List[str] = []
    skipped_negative_edge: List[str] = []

    for symbol, pos in list(positions.items()):
        if not isinstance(pos, dict):
            positions.pop(symbol, None)
            continue
        sig = by_symbol.get(str(symbol).upper())
        entry = _float(pos.get("entry_price"), 0.0)
        # An exit is only meaningful if the ENTRY was real too.
        #
        # Corroborating just the exit still books fiction when the position was
        # opened before the feed was trustworthy. Observed 2026-08-27 after the
        # cross-chain fix landed: BASENOUN-USDC held an entry of 3.08e-05 while
        # the feed's entire history spans 1.5e-04..3.5e-04, and exiting it
        # against the now-correct price booked +402% -- and SOL-USDC booked a
        # -22% "stop_loss" it never took. Ten of twelve open positions carried
        # entries no tick could support.
        #
        # Such a position is not a trade, it is a stale record. Drop it without
        # recording an outcome rather than let it reach the ledger.
        if _corroborated_price(db, symbol, chain, entry) is None:
            try:
                from services.logging_utils import log_message

                log_message(
                    "atf-static",
                    "dropped stale position %s: entry %.10g has no corroborating tick"
                    % (symbol, entry),
                    severity="warning",
                )
            except Exception:  # noqa: BLE001
                pass
            positions.pop(symbol, None)
            continue
        quoted_mark = _float(
            (sig or {}).get("price_usd"),
            _float(pos.get("last_price"), _float(pos.get("entry_price"), 0.0)),
        )
        # An exit mark decides realised P/L, so it needs the same corroboration
        # the entry did. Marking against an unchecked quote is what booked a
        # +2038% "target_hit" on a symbol the feed had never carried.
        mark = _corroborated_price(db, symbol, chain, quoted_mark)
        if mark is None:
            # Hold the position and keep the last good mark. A silent feed is
            # not a reason to realise a price nothing can confirm.
            pos["last_seen_ts"] = now
            continue
        if entry <= 0 or mark <= 0:
            continue
        age = now - _float(pos.get("entry_ts"), now)
        profit = (mark / entry) - 1.0
        # What this round trip costs, as a fraction of the clip it is priced
        # at. Every "is this position worth closing" test below compares
        # against THIS, never against zero -- a position marked out at a return
        # smaller than one round trip is not a winner, it is a fee.
        clip_usd = _ghost_clip_usd()
        cost_rate = _roundtrip_cost_rate(clip_usd)
        target_return = _float(pos.get("target_return"), _float((sig or {}).get("expected_return"), 0.0))
        reason = ""
        # ``cost_rate`` joins the floor so a mis-set ATF_STATIC_GHOST_MIN_EXIT_PROFIT
        # cannot book a "target_hit" that loses money. At the defaults
        # (min_profit 0.005 vs cost 0.0039 on a $6 clip) it changes nothing.
        if profit >= max(min_profit, target_return, cost_rate):
            reason = "target_hit"
        elif profit <= -stop_loss:
            reason = "stop_loss"
        elif age >= max_hold_sec:
            # The hold timer must not realise a loss.
            #
            # Closing on the clock sells at whatever the price happens to be
            # when the timer expires. Measured on the last 20 atf_static
            # trades: target_hit won 4/4 (100%) while max_hold won only 9/16
            # (56%) -- so the timer was the direct source of every losing
            # trade that was not a stop-loss.
            #
            # A position that is merely slow is not a position that is wrong.
            # The stop-loss above already bounds the downside; letting the
            # clock crystallise a small loss converts a recoverable position
            # into a realised one for no reason.
            #
            # So the timer only closes a WINNER. An underwater position keeps
            # running until it either recovers past the profit floor or hits
            # its stop. ATF_STATIC_HOLD_FORCES_EXIT=1 restores the old
            # unconditional behaviour.
            #
            # And "winner" means AFTER the round trip, not above zero.
            #
            # This test read ``profit > 0.0``, which is a gross return compared
            # against nothing. Measured 2026-09-06 over the 5-day ghost book,
            # 83 exits closed on this timer and 47 of them (56.6%) marked out
            # inside one round trip -- CBXRP at +0.045%, AERO at +0.004%,
            # CBBTC at +0.333%, every one of them booked as a "win" and every
            # one of them a guaranteed loss once the 0.386% round trip on a $6
            # clip is charged. Across all 242 trades, 80 exits (33.1%) closed
            # inside the cost: +0.0597 of gross return between them, -$1.4951
            # net. That is the whole of the difference between a book that
            # earns and one that pays fees to stand still.
            #
            # A position between zero and the cost is NOT closed here. It falls
            # through to the stale bound below exactly as a losing one does, so
            # it is still capped -- it just is not crystallised at a price that
            # cannot pay for the crystallising.
            hold_forces_exit = _bool_env("ATF_STATIC_HOLD_FORCES_EXIT", "0")
            if hold_forces_exit or profit > cost_rate:
                reason = "max_hold"
            else:
                # Bound how long a losing position may be carried, so a dead
                # token cannot occupy a slot indefinitely. Beyond this it is
                # closed as a stop even if the stop threshold was never hit.
                stale_sec = max(
                    max_hold_sec * 2.0,
                    _float_env("ATF_STATIC_MAX_UNDERWATER_SEC", 4.0 * 3600.0),
                )
                if age >= stale_sec:
                    reason = "stale_underwater"
        if not reason:
            pos["last_price"] = mark
            pos["last_seen_ts"] = now
            continue
        entry_ts = _float(pos.get("entry_ts"), now - age)
        # ``profit`` is USD, net of the round trip -- the same unit and the same
        # cost model trading/bot.py records, because the live gate and the
        # strategy ledger sum both writers into ONE number.
        #
        # This module used to write the bare fraction here. Measured 2026-09-06,
        # 104 of the 242 trades in the 5-day ghost book were scout rows, so 43%
        # of the book the live gate reads was fractions added to dollars, and
        # none of those 104 had ever been charged a fee. See
        # services/roundtrip_cost.py for the measurement.
        #
        # ``profit_unit`` is written so a reader can tell a corrected row from a
        # legacy one by fact rather than by guessing from its magnitude.
        profit_usd = _net_profit_usd(profit, clip_usd)
        details = {
            "source": SOURCE,
            "strategy_id": SCOUT_STRATEGY_ID,
            "symbol": symbol,
            "chain": chain,
            "entry_price": entry,
            "exit_price": mark,
            "profit": profit_usd,
            "profit_unit": "usd",
            "return_pct": profit,
            "clip_usd": clip_usd,
            "roundtrip_cost_usd": _roundtrip_cost_usd(clip_usd),
            "age_sec": age,
            "reason": reason,
            # The risk layer reads exits through MetricsCollector, which keys on
            # entry_ts/exit_ts and reads the exit label from "exit_reason".
            # Publishing only "reason" and burying the timestamp inside
            # "position" meant 58 of 60 exits reported as "unspecified" and none
            # of them could be paired to their own entry. Emit the field names
            # the reader actually uses; "reason" stays for existing consumers.
            "exit_reason": reason,
            "entry_ts": entry_ts,
            "exit_ts": now,
            "timestamp": now,
            "position": pos,
            "signal": sig,
        }
        db.log_trade(wallet="ghost", chain=chain, symbol=symbol, action="exit", status="ghost-exit", details=details)
        # The ledger is USD too: trading/bot.py records ``economic_profit``
        # here, and graduation scores the two writers against one threshold.
        _record_ghost_outcome(SCOUT_STRATEGY_ID, profit_usd, symbol=symbol)
        # Both, named. An event feed that says "profit" without saying which
        # unit is how the ghost book came to add percentages to dollars.
        events.append({
            "symbol": symbol,
            "action": "exit",
            "profit": profit_usd,
            "profit_unit": "usd",
            "return_pct": profit,
            "reason": reason,
        })
        positions.pop(symbol, None)

    open_count = len(positions)
    # THIS SCOUT ITSELF, IF THE BOOK HAS CONDEMNED IT.
    #
    # Checked ONCE here rather than per symbol: every row this loop writes
    # carries SCOUT_STRATEGY_ID, so the verdict cannot differ between symbols
    # and re-asking inside the loop would only cost a cache lookup per
    # candidate.
    #
    # Placed AFTER the exit pass above and before the entry loop below, which
    # is the whole point: a condemned strategy must still be able to close
    # what it already holds. Gating exits would strand every open position and
    # recreate the disarming bug that left a demoted bot unable to sell what
    # it had bought.
    #
    # Entries only, and evidence is the reason. This scout took 352 of the
    # book's 458 ghost trades (77%) while containing zero swap calls -- it is
    # a simulator that can never place a live trade, and every entry it takes
    # is evidence denied to a strategy that could. See
    # services/strategy_edge_gate.py.
    scout_refusal = _strategy_edge_refusal(SCOUT_STRATEGY_ID)
    if scout_refusal:
        db.log_trade(
            wallet="ghost",
            chain=chain,
            symbol="ATF-STATIC",
            action="hold",
            status="entry-refused-strategy-edge",
            details={
                "reason": "strategy_cannot_pay_its_round_trip",
                "detail": scout_refusal,
                "strategy_id": SCOUT_STRATEGY_ID,
                "candidates_skipped": len(signals),
            },
        )
        signals = []
    for sig in signals:
        symbol = str(sig.get("symbol") or "").upper()
        if not symbol or symbol in positions:
            continue
        if open_count >= max(1, int(max_positions)):
            break
        quote_probe = sig.get("quote_probe") if isinstance(sig.get("quote_probe"), dict) else {}
        if not quote_probe.get("ok"):
            continue
        entry_price = _corroborated_price(
            db, symbol, chain, _float(sig.get("price_usd"), 0.0)
        )
        if not entry_price or entry_price <= 0:
            # No streamed tick to confirm the quote: refuse the entry rather
            # than open a position priced from a source nothing can check.
            skipped_unpriced.append(symbol)
            continue
        # A stop-loss is only as good as the feed that triggers it.
        #
        # The stop is checked when a tick ARRIVES, so on a sparse feed the
        # price gaps straight past it. Measured 2026-08-27: 4 of 6 stop_loss
        # exits breached the 8% stop, losing 11.2%, 11.3% and 22.2%, and
        # SOL-USDC had a 663-minute hole between ticks (14 ticks total). That
        # single -22% exit pushed tail risk to 0.095 against a 0.08 guardrail
        # and blocked live trading entirely.
        #
        # Entering a position we cannot bound the downside on is not a risk we
        # are choosing -- it is one we cannot see. Refuse it.
        if not _feed_is_dense_enough(db, symbol, chain):
            skipped_sparse_feed.append(symbol)
            continue
        # SYMBOLS THE BOOK HAS PROVEN WE LOSE ON.
        #
        # This scout writes `ghost-entry` rows directly and never passes
        # through trading/bot.py, so the gate wired into the bot's entry path
        # does not see it. Measured 2026-09-04 14:16, minutes after that gate
        # went live: BASECAT-USDC -- 37 closed round trips at mean -0.0517,
        # t=-3.30 -- was entered from HERE while the bot was correctly
        # refusing it. One rule, two entry paths, and only one of them was
        # holding the line.
        edge_refusal = _symbol_edge_refusal(symbol)
        if edge_refusal:
            skipped_negative_edge.append(symbol)
            db.log_trade(
                wallet="ghost",
                chain=chain,
                symbol=symbol,
                action="hold",
                status="entry-refused-symbol-edge",
                details={
                    "symbol": symbol,
                    "reason": "symbol_has_a_measured_negative_edge",
                    "detail": edge_refusal,
                    "strategy_id": SCOUT_STRATEGY_ID,
                },
            )
            continue
        # ...and neither is a symbol that cannot move far enough to pay for the
        # round trip. Same reasoning as the gate above and wired here for the
        # same reason: this scout writes `ghost-entry` rows directly, so a gate
        # that lives only in trading/bot.py does not see it.
        motion_refusal = _symbol_motion_refusal(symbol)
        if motion_refusal:
            skipped_negative_edge.append(symbol)
            db.log_trade(
                wallet="ghost",
                chain=chain,
                symbol=symbol,
                action="hold",
                status="entry-refused-symbol-motion",
                details={
                    "symbol": symbol,
                    "reason": "symbol_cannot_cover_a_round_trip",
                    "detail": motion_refusal,
                    "strategy_id": SCOUT_STRATEGY_ID,
                },
            )
            continue
        # ...and neither is a symbol whose stop cannot bind on its own feed.
        #
        # Wired here for the reason the two gates above are: this scout writes
        # `ghost-entry` rows directly and never passes through trading/bot.py,
        # so a gate that lives only there does not see it. MOONBASE-USDC --
        # p99 single-tick jump 99,381% -- was entered from this path, lost
        # 12.41% against a 2% stop, and single-handedly held ES95 tail risk
        # above its guardrail, freezing the live lane.
        stop_refusal = _stop_survivability_refusal(symbol)
        if stop_refusal:
            skipped_negative_edge.append(symbol)
            db.log_trade(
                wallet="ghost",
                chain=chain,
                symbol=symbol,
                action="hold",
                status="entry-refused-stop-survivability",
                details={
                    "symbol": symbol,
                    "reason": "a_stop_cannot_bind_on_this_feed",
                    "detail": stop_refusal,
                    "strategy_id": SCOUT_STRATEGY_ID,
                },
            )
            continue
        target_return = max(min_profit, _float(sig.get("expected_return"), 0.0))
        position = {
            "source": SOURCE,
            "strategy_id": SCOUT_STRATEGY_ID,
            "symbol": symbol,
            "chain": chain,
            "quote_token": quote_token.upper(),
            "entry_ts": now,
            "entry_price": entry_price,
            "last_price": entry_price,
            "target_return": target_return,
            "target_price": entry_price * (1.0 + target_return),
            "confidence": sig.get("confidence"),
            "score": sig.get("score"),
            "token_address": sig.get("token_address"),
            "pair_address": sig.get("pair_address"),
            "quote_probe": quote_probe,
        }
        positions[symbol] = position
        db.log_trade(
            wallet="ghost",
            chain=chain,
            symbol=symbol,
            action="enter",
            status="ghost-entry",
            details={
                **position,
                "reason": f"ATF researched candidate quote_ok=True target={target_return:.2%}",
                "signal": sig,
            },
        )
        events.append({"symbol": symbol, "action": "enter", "target_return": target_return})
        open_count += 1

    try:
        db.set_json(GHOST_POSITIONS_KEY, positions)
    except Exception:
        pass
    if skipped_unpriced:
        # Surface the refusal instead of silently trading less. A signal the
        # feed cannot corroborate is a data problem to fix, not a candidate to
        # quietly drop.
        try:
            from services.logging_utils import log_message

            log_message(
                "atf-static",
                "refused %d uncorroborated candidate(s): %s"
                % (len(skipped_unpriced), ", ".join(sorted(set(skipped_unpriced))[:8])),
                severity="warning",
            )
        except Exception:
            pass
    if skipped_sparse_feed:
        # Surface it: a symbol we cannot stop out of is a coverage problem to
        # fix, not a candidate to silently drop.
        try:
            from services.logging_utils import log_message

            log_message(
                "atf-static",
                "refused %d candidate(s) with a feed too sparse to enforce a stop: %s"
                % (len(skipped_sparse_feed),
                   ", ".join(sorted(set(skipped_sparse_feed))[:8])),
                severity="warning",
            )
        except Exception:
            pass
    if skipped_negative_edge:
        # Reported for the same reason the sparse-feed refusal is: a silent
        # refusal is indistinguishable from the scout never having found the
        # candidate, and that silence is what let BASECAT keep being entered
        # from this path while the bot's own gate refused it.
        try:
            from services.logging_utils import log_message

            log_message(
                "atf-static",
                "refused %d candidate(s) with a measured negative edge: %s"
                % (len(skipped_negative_edge),
                   ", ".join(sorted(set(skipped_negative_edge))[:8])),
                severity="warning",
            )
        except Exception:
            pass
    return {
        "enabled": True,
        "open": len(positions),
        "events": events,
        "skipped_unpriced": sorted(set(skipped_unpriced)),
        "skipped_sparse_feed": sorted(set(skipped_sparse_feed)),
        "skipped_negative_edge": sorted(set(skipped_negative_edge)),
    }


def _drop_already_refused(candidates: List[Any], *, quote_token: str) -> List[Any]:
    """Drop candidates the gates have already, standingly, refused.

    The gates below run per ENTRY and they are correct, but a candidate slot
    is spent long before the entry gate sees it: the scout quote-probes the
    symbol, writes a ghost_candidate row, and publishes a bus action, all for
    a verdict that was already on file.

    Measured 2026-09-06 over 30 minutes: BASECAT-USDC was offered 10 times and
    refused 20 times, against a symbol_edge_gate verdict standing on 35 closed
    round trips at a mean return of -1.565% versus a 0.650% cost. **58% of all
    candidate slots went to symbols with a standing refusal.** Discovery kept
    proposing what the gates kept declining, and every one of those slots was
    a slot an eligible symbol did not get.

    This is a PRE-FILTER, not a new gate. It asks the same three gates the
    same questions they would be asked a moment later, and refusing here
    changes no decision -- it only stops the pipeline paying for the same
    refusal twice. Every gate still runs at the entry site, so a symbol whose
    verdict changes between here and there is still judged correctly.

    Fails OPEN in every direction: a gate that cannot be imported, or that
    raises, leaves the candidate in the list to be judged downstream as
    before.
    """
    try:
        from services.symbol_edge_gate import refusal_reason as _edge
    except Exception:  # noqa: BLE001
        def _edge(_symbol: str):  # type: ignore[misc]
            return None
    try:
        from services.symbol_motion_gate import refusal_reason as _motion
    except Exception:  # noqa: BLE001
        def _motion(_symbol: str):  # type: ignore[misc]
            return None
    try:
        from services.stop_survivability_gate import refusal_reason as _stop
    except Exception:  # noqa: BLE001
        def _stop(_symbol: str):  # type: ignore[misc]
            return None

    kept: List[Any] = []
    for candidate in candidates:
        try:
            base = str(getattr(candidate, "symbol", "") or "").upper()
            if not base:
                continue
            pair = f"{base}-{str(quote_token or 'USDC').upper()}"
            if _edge(pair) or _motion(pair) or _stop(pair):
                continue
        except Exception:  # noqa: BLE001 - never drop on an error
            kept.append(candidate)
            continue
        kept.append(candidate)
    # If every candidate carries a standing refusal, hand back the original
    # list rather than nothing: an empty cycle produces no evidence at all,
    # and the entry gates will refuse them individually anyway.
    return kept or list(candidates)


def _certainly_refused_as_held(*, quote_token: str, strategy_id: str = "atf_static") -> set:
    """Pair symbols whose entry the book will refuse before any gate runs.

    ``_drop_already_refused`` above pre-filters the three GATE refusals, on the
    argument that a candidate slot spent re-proposing a standing refusal is a
    slot an eligible symbol did not get. That argument is right and it was
    aimed at the rarest third of the census. Measured 2026-09-07 over 6h from
    ``trading_ops``, ``atf_static`` -- the only executor that can spend real
    money -- was refused 57 times:

        entry-refused-duplicate           25   <- held by atf_static itself
        entry-refused-slot-busy           17   <- held by another strategy
        entry-refused-symbol-edge         11   } the 15 the pre-filter
        entry-refused-symbol-motion        3   } already catches
        entry-refused-stop-survivability   1   }

    So 42 of 57 (74%) were "the symbol is already held", 40 of them on
    AERO-USDC alone, against a position atf_static had been holding for up to
    3169s. Unlike a gate verdict, that is not an estimate: it is read off the
    position book, and the entry site will refuse it with certainty. Each one
    still cost a 0x quote probe on a feed that is already rate-limited.

    This matters because the ONLY thing between here and a live trade is
    atf_static's evidence rate. Re-arming needs 20 ghost round trips gathered
    since its demotion; it had 8 in 31.3h (0.26/h), while three quarters of
    its candidate slots were being spent on symbols it already held.

    Mirrors ``TradingBot._interpret_predictions`` exactly rather than
    approximating it, because both of its carve-outs are load-bearing:

      * A LIVE-APPROVED strategy drops NOTHING. Its entry may be live, and a
        live entry deliberately displaces a ghost position -- including its
        own, the ghost->live upgrade at bot.py:6807. Pre-filtering that away
        would silently re-close link 6, which cost 7 of 9 live-capable symbols
        on 2026-09-02. The waste this fixes only exists while the strategy is
        ghost-only, which is precisely when it is trying to earn its licence.
      * A position past ``MAX_HOLD_SECONDS``, one held with no ``strategy_id``,
        or another strategy's LIVE position is still enterable at the entry
        site, so none of those are dropped here.

    Returns an empty set on any failure: a book that cannot be read leaves the
    candidate list exactly as it was.
    """
    try:
        from trading.strategies.ledger import StrategyLedger

        if strategy_id in set(StrategyLedger().approved_ids() or ()):
            return set()          # its entries may be live, and live displaces
    except Exception:  # noqa: BLE001 - never narrow the funnel on an error
        return set()

    try:
        state = get_db().load_state()
        positions = ((state or {}).get("ghost_trading") or {}).get("positions") or {}
        if not isinstance(positions, dict):
            return set()
        max_hold_sec = float(os.getenv("MAX_HOLD_SECONDS", "3600"))
        now = _now()
        held: set = set()
        for symbol, position in positions.items():
            if not isinstance(position, dict):
                continue
            pair = str(symbol or "").strip().upper()
            if not pair:
                continue
            held_strategy = str(position.get("strategy_id") or "")
            if not held_strategy:
                continue      # books as "unclassified"; bot.py does not refuse
            age = now - float(position.get("entry_ts", position.get("ts", 0.0)) or 0.0)
            if age >= max_hold_sec:
                continue      # evictable -- the stale-slot escape hatch
            if held_strategy == strategy_id:
                held.add(pair)                       # entry-refused-duplicate
            elif str(position.get("mode") or "") != "live":
                held.add(pair)                       # entry-refused-slot-busy
        return held
    except Exception:  # noqa: BLE001
        return set()


def _add_streamed_candidates(candidates: List[Any], *, max_positions: int) -> List[Any]:
    """Append symbols we already stream and have already proven can pay.

    ``select_candidates`` builds its list from DexScreener and Gecko NEW
    POOLS. That is the right job for finding what just appeared, and it means
    a symbol we have streamed for a week -- whose edge, motion and
    stop-survivability are already measured -- can NEVER become a candidate,
    because it is not new.

    Measured 2026-09-06, one hour after production came back: 79% of ghost
    candidates were offered on symbols the gates refuse, and eleven of the
    thirteen symbols passing every gate had never been offered one. Two
    entries in an hour against a feed carrying 774 ticks per ten minutes.

    APPENDED, NEVER SUBSTITUTED. New-pool discovery keeps its slots; this only
    fills the space underneath. The two answer different questions and a
    pipeline that asks only the first keeps rediscovering the same
    unprofitable memecoins while ignoring instruments it has a week of
    evidence about.

    A REAL ADDRESS OR NOTHING. The consumer needs ``candidate.address`` to
    build a swap, and a fabricated one would produce a signal that cannot
    execute -- or worse, one that executes against the wrong token. A symbol
    the address book cannot resolve is skipped rather than guessed at; see
    the ticker-squatting note in services/token_address_book.py for why
    resolving by ticker is a safety failure rather than a convenience.

    Best-effort throughout: a candidate source that raises would stop the
    trading cycle, and having no suggestions is a normal state.
    """
    try:
        from services.streamed_symbol_candidates import streamed_candidates
        from services.token_address_book import lookup as _lookup_address
        from tools.c0d3rV2.crypto_paper_trade import Candidate
    except Exception:  # noqa: BLE001
        return candidates

    try:
        already = {str(getattr(c, "symbol", "") or "").upper() for c in candidates}
        added: List[Any] = []
        for proposal in streamed_candidates(limit=max(1, int(max_positions)) * 4):
            base = str(proposal.symbol or "").split("-")[0].upper()
            if not base or base in already:
                continue
            address = _lookup_address("base", base)
            if not address:
                continue
            already.add(base)
            added.append(Candidate(
                token=base,
                symbol=base,
                address=str(address),
                # Empty rather than invented: the scout keys its dedupe on
                # pair_address or address, and a fake pair would collide with
                # a real one.
                pair_address="",
                dex="streamed",
                url="",
                price_usd=0.0,
                liquidity_usd=0.0,
                volume_m5=0.0,
                volume_h1=0.0,
                # The scout derives expected_return from price_change_m5, and
                # the honest value here is zero: this proposal rests on how
                # OFTEN the symbol clears its cost, not on a recent move. The
                # target floor takes over from there.
                price_change_m5=0.0,
                price_change_h1=0.0,
                buys_m5=0,
                sells_m5=0,
                buys_h1=0,
                sells_h1=0,
                fdv=0.0,
                market_cap=0.0,
                # Clear-rate as the score, so a symbol that pays more often is
                # ranked above one that rarely does -- the same ordering the
                # proposer already applied.
                score=float(proposal.clear_rate),
                rationale=proposal.rationale,
            ))
        return list(candidates) + added
    except Exception:  # noqa: BLE001
        return candidates


def build_static_strategy_signals(
    *,
    budget_usd: float = 20.0,
    max_positions: int = 3,
    chain: str = "base",
    quote_token: str = "USDC",
    slippage_bps: int = 100,
    probe_quotes: bool = True,
) -> Dict[str, Any]:
    """
    Research Base candidates and publish them as normal scheduler-readable
    strategy signals.

    This does not broadcast transactions. It writes:
      * watchlists.stream / watchlists.ghost entries
      * trading_ops audit rows
      * kv_store persistent ATF strategy signals
      * quote/readiness probe results when possible
    """
    chain = (chain or "base").lower()
    db = get_db()
    started = _now()
    try:
        from tools.c0d3rV2.crypto_paper_trade import select_candidates
    except Exception as exc:
        raise RuntimeError(f"Unable to load C0D3R/ATF candidate selector: {exc}") from exc

    try:
        portfolio = PortfolioState(chains=(chain,))
        wallet = portfolio.wallet
        stable_symbol, stable_qty = _stable_source(portfolio, chain)
    except Exception as exc:
        portfolio = None  # type: ignore[assignment]
        wallet = os.getenv("PRIMARY_WALLET", "")
        stable_symbol, stable_qty = quote_token.upper(), 0.0
        db.log_trade(
            wallet="ghost",
            chain=chain,
            symbol="ATF-STATIC",
            action="wallet_read",
            status="warning",
            details={"source": SOURCE, "error": str(exc)},
        )

    effective_budget = max(1.0, float(budget_usd))
    if stable_qty > 0:
        effective_budget = min(effective_budget, max(1.0, stable_qty))
    per_position_usd = effective_budget / max(1, int(max_positions))
    probe_amount = max(0.01, min(float(os.getenv("ATF_STATIC_QUOTE_PROBE_USD", "0.25")), per_position_usd))

    candidates = select_candidates(budget_usd=effective_budget, max_positions=max_positions)
    candidates = _add_streamed_candidates(candidates, max_positions=max_positions)
    candidates = _drop_already_refused(candidates, quote_token=quote_token)
    # Read ONCE per cycle, not per candidate: the book does not move while we
    # iterate, and load_state() is not free.
    held_pairs = _certainly_refused_as_held(quote_token=quote_token)
    feedback = refresh_feedback_scores() if _bool_env("ATF_STATIC_FEEDBACK_ENABLED", "1") else {}
    signals: List[Dict[str, Any]] = []
    bus_actions: List[Dict[str, Any]] = []
    pairs: List[str] = []

    for idx, candidate in enumerate(candidates, start=1):
        symbol = str(candidate.symbol or "").upper()
        if not symbol or not candidate.address:
            continue
        pair_symbol = f"{symbol}-{quote_token.upper()}"
        pairs.append(pair_symbol)
        # The skip goes AFTER pairs.append, deliberately. `pairs` becomes the
        # stream watchlist below, and a symbol we HOLD is the last thing that
        # may lose its feed: every exit rule hangs off a market sample, so a
        # held position with no stream is never closed at all. Dropping the
        # candidate must cost the quote probe, the ghost_candidate row and the
        # bus action -- never the price feed keeping the position exitable.
        if pair_symbol in held_pairs:
            continue
        outcome = _feedback_for(pair_symbol, feedback)
        feedback_multiplier = max(0.25, min(1.75, _float(outcome.get("allocation_multiplier"), 1.0)))
        expected_return = max(0.0, min(0.15, (_float(candidate.price_change_m5) / 100.0) * 0.35 + (_float(candidate.score) * 0.04)))
        expected_return = max(0.0, min(0.15, expected_return * feedback_multiplier))
        confidence = max(0.05, min(0.9, _float(candidate.score) * feedback_multiplier))
        target_floor = max(0.005, min(0.10, _float(os.getenv("ATF_STATIC_TARGET_RETURN", "0.05"), 0.05)))
        target_return = max(0.015, min(0.15, max(expected_return, target_floor if confidence >= 0.35 else 0.015)))
        target_price = _float(candidate.price_usd) * (1.0 + target_return)
        signal = {
            "source": SOURCE,
            "ts": _now(),
            "chain": chain,
            "symbol": pair_symbol,
            "base_token": symbol,
            "quote_token": quote_token.upper(),
            "token_address": candidate.address,
            "pair_address": candidate.pair_address,
            "action": "enter",
            "strategy_id": "atf_static",
            "expected_return": round(target_return, 6),
            "target_price": target_price,
            "confidence": round(confidence, 6),
            "score": candidate.score,
            "budget_usd": round(per_position_usd, 6),
            "rationale": candidate.rationale,
            "feedback": outcome,
            "url": candidate.url,
            "liquidity_usd": candidate.liquidity_usd,
            "volume_h1": candidate.volume_h1,
            "price_usd": candidate.price_usd,
            "quote_probe": None,
        }
        if probe_quotes and wallet:
            signal["quote_probe"] = _quote_probe(
                chain=chain,
                sell_token=stable_symbol,
                buy_token=candidate.address,
                amount=min(probe_amount, max(stable_qty, probe_amount)),
                from_address=wallet,
                slippage_bps=slippage_bps,
            )
        status = "ghost_candidate_quote_ok" if (signal.get("quote_probe") or {}).get("ok") else "ghost_candidate"
        db.log_trade(
            wallet="ghost",
            chain=chain,
            symbol=pair_symbol,
            action="enter",
            status=status,
            details=signal,
        )
        signals.append(signal)
        bus_actions.append(
            {
                "action": "evaluate_atf_static_entry",
                "reason": "c0d3rv2_atf_candidate",
                "priority": 2,
                "chain": chain,
                "symbol": pair_symbol,
                "token_address": candidate.address,
                "target_usd": round(per_position_usd, 6),
                "quote_token": quote_token.upper(),
                "strategy_id": "atf_static",
                "window_sec": int(os.getenv("ATF_STATIC_BUS_WINDOW_SEC", "900")),
            }
        )

    if pairs:
        current = load_watchlists(db)
        pair_set = [p.upper() for p in pairs]
        current["stream"] = pair_set + [p for p in current.get("stream", []) if p not in pair_set]
        current["ghost"] = pair_set + [p for p in current.get("ghost", []) if p not in pair_set]
        save_watchlists(current, db=db)

    ghost_scout = _run_ghost_quote_scout(
        db=db,
        signals=signals,
        chain=chain,
        quote_token=quote_token,
        max_positions=max_positions,
    )

    payload = {
        "source": SOURCE,
        "ts": _now(),
        "duration_sec": round(_now() - started, 3),
        "chain": chain,
        "wallet": wallet,
        "quote_token": quote_token.upper(),
        "stable_source": stable_symbol,
        "stable_quantity": stable_qty,
        "budget_usd": budget_usd,
        "effective_budget_usd": effective_budget,
        "signals": signals,
        "bus_actions": bus_actions,
        "ghost_scout": ghost_scout,
        "live_execution_enabled": False,
        "live_execution_note": "Signals are ghost/scheduler inputs. Real swaps remain controlled by existing live readiness and dry-run gates.",
    }
    db.set_json(SIGNAL_KEY, signals)
    db.set_json(LATEST_KEY, payload)
    db.set_json(PENDING_BUS_KEY, bus_actions)
    db.log_trade(
        wallet="ghost",
        chain=chain,
        symbol="ATF-STATIC",
        action="strategy_publish",
        status="published" if signals else "no_candidates",
        details={k: v for k, v in payload.items() if k != "signals"},
    )
    return payload


def latest_signals(max_age_sec: float = 1800.0) -> List[Dict[str, Any]]:
    db = get_db()
    rows = db.get_json(SIGNAL_KEY) or []
    if not isinstance(rows, list):
        return []
    cutoff = _now() - max(30.0, float(max_age_sec))
    fresh = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if _float(row.get("ts")) < cutoff:
            continue
        fresh.append(row)
    return fresh


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Publish C0D3R/ATF static strategy signals into ghost trading.")
    parser.add_argument("--budget-usd", type=float, default=float(os.getenv("ATF_STATIC_BUDGET_USD", "20")))
    parser.add_argument("--max-positions", type=int, default=int(os.getenv("ATF_STATIC_MAX_POSITIONS", "3")))
    parser.add_argument("--chain", default=os.getenv("ATF_STATIC_CHAIN", "base"))
    parser.add_argument("--quote-token", default=os.getenv("ATF_STATIC_QUOTE_TOKEN", "USDC"))
    parser.add_argument("--slippage-bps", type=int, default=int(os.getenv("ATF_STATIC_SLIPPAGE_BPS", "100")))
    parser.add_argument("--no-probe-quotes", action="store_true")
    args = parser.parse_args(argv)
    payload = build_static_strategy_signals(
        budget_usd=args.budget_usd,
        max_positions=args.max_positions,
        chain=args.chain,
        quote_token=args.quote_token,
        slippage_bps=args.slippage_bps,
        probe_quotes=not args.no_probe_quotes,
    )
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
