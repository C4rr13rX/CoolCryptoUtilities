"""Replay EVERY registered strategy's shipped gate over stored ticks. Read-only.

WHY THIS EXISTS. Pass 441 built this measurement twice, for one strategy each
time (``scripts/money_button_dip_census.py``, ``scripts/money_button_momentum_
census.py``), and it paid both times: it answered in minutes what the
graduation funnel needs ~39.7 days of ghost evidence to answer, and it produced
two decisive closures. The funnel's own ranking -- "closest to the bar" -- is
built from whatever each strategy happened to be offered by the scheduler, so a
rule that was never handed a tick looks identical to a rule that was handed
thousands and declined them all. This hands EVERY strategy the SAME 168h of
tape and reports what its own shipped ``evaluate`` did with it.

WHAT IT IS NOT. This is the GATE's verdict, not the lane's. Nothing downstream
of ``Strategy.evaluate`` is replayed: no CDCL arbitration, no position cap, no
risk gate, no executor, no live arm. A strategy that tops this table has not
earned a live approval; it has earned somebody's attention.

THE 54 STRATEGIES THIS CANNOT REPLAY, and why they are listed rather than
dropped. ``build_default_registry()`` returns 72 strategies and 54 of them are
``MultiHorizonStrategy`` wrappers (``ema_cross@1d``, ``rsi_reversal@1w`` ...).
Their ``evaluate`` (trading/strategies/horizons.py) never reads
``state.samples`` at all -- it calls ``_PROVIDER.get_bars(...)``, and
``HistoryProvider._build`` anchors its window on ``now = time.time()`` and
queries the store for the LATEST bars. Handing such a wrapper a window from
tick *i* of last Tuesday returns its verdict on TODAY'S tape, carrying up to
168h of look-ahead. That is not a slow replay or an awkward one, it is a
meaningless one, so these are reported in a NOT REPLAYABLE section with the
reason. In production ``now`` really is now and the wrapper is not wrong there;
it is simply unauditable offline until a provider can be injected.

UPPER BOUNDS, printed in the output rather than buried here:
  * ``market_stream`` stores (ts, price) and NO volume, so every volume guard
    in every strategy is inert in this replay. Fire counts are UPPER BOUNDS.
  * Forward outcome is held to target or to the horizon on the same tick
    series; there is no book, no partial fill and no slippage beyond the
    per-token round trip. Gross here is a GHOST gross.
  * Cost is ``TOKEN_ROUND_TRIP`` -- the same table the dip census used. Pass
    492 measured live AERO fills at 37.45bp against the quoter's 13.58bp, so
    a round-trip constant is a LOWER BOUND on real cost and every ``mean net``
    below is therefore an OPTIMISTIC number.

Nothing here writes: no ledger, no registry, no threshold, no approval.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.money_button_dip_census import (  # noqa: E402
    base_of,
    load_ticks,
    round_trip_for,
)
from trading.strategies import build_default_registry  # noqa: E402
from trading.strategies.base import Strategy, StrategyContext  # noqa: E402
from trading.strategies.horizons import MultiHorizonStrategy  # noqa: E402

#: How many trailing samples each evaluation sees. The base contract's
#: ``min_samples`` is 16 and several strategies scale their own windows off the
#: median gap, so 40 gives every rule room without handing it a stale shelf.
#: It is the same constant the momentum census used, deliberately, so the
#: money_button row here is comparable with that measurement.
WINDOW = 40

#: Actions that open a position. Only these are resolved forward -- an EXIT
#: directive with no position is not a round trip, and pricing one as if it
#: were would invent a trade. Exit fires are counted and reported separately.
ENTER_ACTIONS = {"enter", "buy", "enter_long", "long", "open"}

#: Gates that are NOW-ANCHORED BY SOURCE, verified by reading them, not by a
#: heuristic. Each consults an external oracle with its own freshness window
#: and never reads ``state.samples``, so handing it a historical window asks it
#: nothing -- it answers about today. A replay row for these would be a
#: fabrication, and a zero-fire row would be a false negative.
ORACLE_GATES: Dict[str, str] = {
    "atf_static": (
        "NOW-ANCHORED: trading/strategies/atf_static.py:28 calls "
        "services.atf_static_strategy.latest_signals(max_age) with an 1800s "
        "freshness window and never reads state.samples -- at a historical "
        "tick it answers about TODAY, so its replay says nothing about then"
    ),
}


def _state(symbol: str, window: Sequence[Tuple[float, float]]):
    """A RouteState-alike carrying REAL timestamps, not synthetic spacing.

    Sample spacing is what several gates scale their windows off, so faking it
    uniform would change their behaviour. These are the feed's own timestamps.
    Volume is 0.0 because ``market_stream`` has none -- that is the upper-bound
    caveat printed in the report, not a silent substitution.
    """
    state = types.SimpleNamespace()
    state.symbol = symbol
    state.base_token = base_of(symbol)
    state.quote_token = symbol.split("-", 1)[1] if "-" in symbol else "USDC"
    state.samples = [(float(t), float(p), 0.0) for t, p in window]
    return state


def _t_stat(sample: np.ndarray) -> float:
    """One-sample t against zero. NaN when it cannot be formed.

    Printed beside every mean on purpose. These gates fire rarely and a mean
    over a handful of rows reads as a verdict when it is noise; ``|t| < 2`` on
    GROSS means UNMEASURED, which is a different finding from a gross edge
    that is genuinely absent.
    """
    if sample.size < 2:
        return float("nan")
    sd = float(sample.std(ddof=1))
    if sd <= 0.0:
        return float("nan")
    return float(sample.mean() / (sd / np.sqrt(sample.size)))


def _probe_windows(
    ticks: Dict[str, List[Tuple[float, float]]]
) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """Three windows no honest price gate can answer identically.

    A RAMP, a CRASH and a FLAT line, built on the REAL timestamps of the
    busiest symbol so that any median-gap-scaled window unit is unchanged and
    only the PRICES differ.
    """
    symbol = max(ticks, key=lambda s: len(ticks[s]))
    tail = ticks[symbol][-(WINDOW + 1):]
    stamps = [t for t, _ in tail]
    base = float(tail[0][1]) or 1.0
    n = len(stamps)
    shapes = {
        # +40% over the window and -40% over the window: any gate that reads
        # prices at all must treat these differently, and a flat line is the
        # third corner.
        "ramp": [base * (1.0 + 0.40 * i / max(n - 1, 1)) for i in range(n)],
        "crash": [base * (1.0 - 0.40 * i / max(n - 1, 1)) for i in range(n)],
        "flat": [base] * n,
    }
    return [(name, list(zip(stamps, prices))) for name, prices in shapes.items()]


def _verdict_signature(result) -> Tuple:
    """A comparable fingerprint of one evaluate() call."""
    if not result:
        return ("declined",)
    directive = result.get("directive")
    return (
        "fired",
        str(getattr(directive, "action", "") or ""),
        round(float(getattr(directive, "expected_return", 0.0) or 0.0), 8),
        round(float(getattr(directive, "target_price", 0.0) or 0.0), 8),
    )


def _window_blind(
    strategy: Strategy,
    probes: List[Tuple[str, List[Tuple[float, float]]]],
    gate_fee: float,
) -> bool:
    """True when the gate's verdict does not move between ramp, crash and flat.

    THIS IS A DIAGNOSTIC, NOT A CLASSIFIER, AND THE FIRST VERSION WAS WRONG.
    Written as a classifier it called 15 of 18 strategies blind -- including
    ``mean_reversion`` and ``macd_momentum``, which plainly read prices. The
    flaw: "declined" is ONE bucket, so three windows that all fail the same
    gate look identical to a gate that never looked. A monotone ramp is
    exactly what a mean-reversion rule refuses, so refusing all three is
    evidence of nothing.

    What it IS good for: a strategy that fires zero times over the whole
    replay is either refusing or blind, and those are the same observation.
    This probe cannot separate them either -- it is printed beside that zero so
    nobody reads the zero as a verdict. The two gates KNOWN to be blind are
    named from their source, not from this probe.
    """
    seen = set()
    for _name, window in probes:
        last_price = float(window[-1][1])
        ctx = StrategyContext(
            chain="base",
            last_price=last_price,
            last_volume=0.0,
            fee_rate=gate_fee,
            available_quote=1000.0,
            available_base=0.0,
        )
        try:
            seen.add(_verdict_signature(strategy.evaluate(_state("WETH-USDC", window), ctx)))
        except Exception:
            seen.add(("raised",))
    return len(seen) == 1


def _classify(
    strategy: Strategy,
    probes: List[Tuple[str, List[Tuple[float, float]]]],
    gate_fee: float,
) -> Optional[str]:
    """Return a NOT-REPLAYABLE reason, or None when the strategy can be replayed."""
    if isinstance(strategy, MultiHorizonStrategy):
        return ("reads the live DB at time.time() and ignores state.samples "
                "(horizons.py HistoryProvider) -- a replay would carry look-ahead")
    if not strategy.enabled():
        return f"disabled: STRATEGY_{strategy.strategy_id.upper()}_ENABLED is off"
    if strategy.strategy_id in ORACLE_GATES:
        return ORACLE_GATES[strategy.strategy_id]
    return None


def window_for(strategy: Strategy) -> int:
    """Samples to hand THIS strategy, from its own declared ``min_samples``.

    A FIXED WINDOW SILENTLY ZEROES A STRATEGY, and the first version of this
    script did exactly that. ``ema_cross`` declares ``min_samples = 40``; at a
    flat WINDOW = 40 it receives 41 samples, has no room for its slow EMA, and
    reports 0 fires in 31,984 evaluations -- which reads as "the gate refused
    everything" and is really "the harness never gave it a chance". At 120 it
    fires 210 times. ``omen_reversion`` declares 60 and could not fire at all.

    So the window is the strategy's OWN requirement with headroom, never a
    constant, and the size used is printed in its row so the number is not a
    hidden parameter.
    """
    return max(WINDOW, 3 * int(getattr(strategy, "min_samples", 16) or 16))


def _replay_one(
    strategy: Strategy,
    ticks: Dict[str, List[Tuple[float, float]]],
    *,
    gate_fee: float,
    horizon_sec: float,
) -> dict:
    """Evaluate one strategy at every tick of every symbol, then resolve forward."""
    fires: List[dict] = []
    evaluated = 0
    raised: Dict[str, int] = defaultdict(int)
    exit_fires = 0
    window_n = window_for(strategy)

    for symbol, series in ticks.items():
        n = len(series)
        for i in range(window_n, n):
            evaluated += 1
            window = series[i - window_n:i + 1]
            last_price = float(window[-1][1])
            ctx = StrategyContext(
                chain="base",
                last_price=last_price,
                last_volume=0.0,
                fee_rate=gate_fee,
                available_quote=1000.0,
                available_base=0.0,
            )
            try:
                # The SHIPPED gate. A hand-copy would measure the copy.
                result = strategy.evaluate(_state(symbol, window), ctx)
            except Exception as exc:  # a crash is a finding, never a skip
                raised[type(exc).__name__] += 1
                continue
            if not result:
                continue
            directive = result.get("directive")
            action = str(getattr(directive, "action", "") or "").strip().lower()
            if action and action not in ENTER_ACTIONS:
                exit_fires += 1
                continue
            target = float(getattr(directive, "target_price", 0.0) or 0.0)
            fires.append({
                "symbol": symbol, "i": i, "ts": float(window[-1][0]),
                "entry": last_price,
                # No target is not a reason to drop a fire; 1% is the same
                # placeholder the momentum census used so the two agree.
                "target": target if target > 0 else last_price * 1.01,
            })

    resolved: List[dict] = []
    for f in fires:
        series = ticks[f["symbol"]]
        entry, target = f["entry"], f["target"]
        exit_price, hit, held = entry, False, 0.0
        for ts, price in series[f["i"] + 1:]:
            held = ts - f["ts"]
            exit_price = float(price)
            if price >= target:
                hit = True
                break
            if held >= horizon_sec:
                break
        if held <= 0.0:
            continue  # no forward tape: unresolved, not a win and not a loss
        gross = (exit_price - entry) / entry
        cost = round_trip_for(f["symbol"])
        resolved.append({
            "symbol": f["symbol"], "gross": gross, "net": gross - cost,
            "hit": hit, "held": held, "cost": cost,
        })

    return {
        "strategy": strategy.strategy_id,
        "window": window_n,
        "evaluations": evaluated,
        "fires": len(fires),
        "exit_fires": exit_fires,
        "raised": dict(raised),
        "resolved": resolved,
    }


def _summarise(row: dict) -> dict:
    """Means, win rate and t-stats for one replayed strategy."""
    resolved = row["resolved"]
    out = dict(row)
    out["n"] = len(resolved)
    if not resolved:
        out.update(mean_gross=float("nan"), mean_net=float("nan"),
                   t_gross=float("nan"), t_net=float("nan"), win=float("nan"))
        return out
    gross = np.array([r["gross"] for r in resolved], dtype=float)
    net = np.array([r["net"] for r in resolved], dtype=float)
    # CONCENTRATION. Added because the FIRST run of this script produced
    # rsi_reversal at +0.489% gross, t=2.568, n=904 -- the only t>2 row in the
    # registry -- and SIX rows of TILCAYO-USDC were 56.3% of that pooled gross.
    # Dropping TILCAYO and LAPTOP (19 of 903 rows) took it to +0.0756% at
    # t=0.568. A t-stat does not catch this: it is a mean over a heavy tail,
    # and this repo has shipped the same defect before (98% of gross in 7 rows).
    # So the share of the pooled gross held by the single biggest symbol is
    # computed and printed beside the t.
    by_symbol_sum: Dict[str, float] = defaultdict(float)
    for r in resolved:
        by_symbol_sum[r["symbol"]] += r["gross"]
    total = float(gross.sum())
    out["top_symbol"], out["top_share"] = "", float("nan")
    if abs(total) > 1e-12 and by_symbol_sum:
        top = max(by_symbol_sum.items(), key=lambda kv: abs(kv[1]))
        out["top_symbol"] = top[0]
        out["top_share"] = 100.0 * top[1] / total
    out.update(
        median_gross=float(np.median(gross)),
        mean_gross=float(gross.mean()),
        mean_net=float(net.mean()),
        t_gross=_t_stat(gross),
        t_net=_t_stat(net),
        win=100.0 * float((net > 0).sum()) / len(resolved),
    )
    return out


def _fmt(value: float, width: int, digits: int = 3, pct: bool = True) -> str:
    if not np.isfinite(value):
        return f"{'--':>{width}}"
    if pct:
        return f"{100.0 * value:>{width}.{digits}f}%"
    return f"{value:>{width}.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--fee", type=float, default=None,
                    help="one-way fee handed to each GATE; default TRADE_FEE_RATE or 0.003")
    ap.add_argument("--horizon-min", type=float, default=15.0,
                    help="forward hold ceiling in minutes")
    ap.add_argument("--per-token", action="store_true",
                    help="also print the per-token split for every replayed strategy "
                         "(pass 441 saw mean gross span +0.114%% to -9.722%% inside "
                         "ONE cost cohort, so the pooled mean can hide the answer)")
    ap.add_argument("--min-token-n", type=int, default=3,
                    help="rows below this n are omitted from the per-token split")
    ap.add_argument("--only", default="",
                    help="comma-separated strategy_ids to replay (default: all)")
    args = ap.parse_args()

    gate_fee = args.fee if args.fee is not None else float(
        os.getenv("TRADE_FEE_RATE", "0.003"))
    horizon_sec = float(args.horizon_min) * 60.0

    ticks = load_ticks(args.hours)
    if not ticks:
        print("no ticks in the window -- nothing to replay")
        return 1
    tick_count = sum(len(v) for v in ticks.values())

    registry = build_default_registry()
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}

    probes = _probe_windows(ticks)
    replayable: List[Strategy] = []
    blocked: List[Tuple[str, str]] = []
    for strategy in registry.all():
        if wanted and strategy.strategy_id not in wanted:
            continue
        reason = _classify(strategy, probes, gate_fee)
        if reason:
            blocked.append((strategy.strategy_id, reason))
        else:
            replayable.append(strategy)

    print("=" * 78)
    print("STRATEGY REPLAY TABLE -- every registered gate, same tape, same cost")
    print("=" * 78)
    print(f"{args.hours:.0f}h of stored ticks: {len(ticks)} symbols, {tick_count} ticks, "
          f"window per-strategy (3x its own min_samples, floor {WINDOW}), horizon {args.horizon_min:.0f}m")
    print(f"registry: {len(registry.all())} strategies -- "
          f"{len(replayable)} replayed, {len(blocked)} not replayable")
    print(f"gate fee handed to every strategy: {gate_fee:.4f}")
    print()
    print("UPPER BOUNDS -- read these before reading a row:")
    print("  * market_stream carries NO volume, so every volume guard is inert")
    print("    here. Fire counts are UPPER BOUNDS on the production fire count.")
    print("  * no downstream arbitration, position cap, risk gate or live gate is")
    print("    replayed. This is the GATE's verdict, NOT the lane's.")
    print("  * cost is the TOKEN_ROUND_TRIP constant. Live AERO fills measured")
    print("    37.45bp against the quoter's 13.58bp (pass 492), so that constant")
    print("    is a LOWER bound and every 'mean net' below is OPTIMISTIC.")
    print()

    started = time.time()
    rows: List[dict] = []
    for idx, strategy in enumerate(replayable, 1):
        t0 = time.time()
        row = _replay_one(strategy, ticks, gate_fee=gate_fee, horizon_sec=horizon_sec)
        rows.append(_summarise(row))
        print(f"  [{idx}/{len(replayable)}] {strategy.strategy_id:<22} "
              f"{row['fires']:>6} fires / {row['evaluations']} evals "
              f"({time.time() - t0:.0f}s)", file=sys.stderr)

    # RANKED BY MEAN NET. A strategy with nothing resolved sorts last rather
    # than sorting as zero -- an unmeasured rule is not a flat rule.
    rows.sort(key=lambda r: (np.isfinite(r["mean_net"]), r["mean_net"]), reverse=True)

    header = (f"{'strategy':<22}{'win':>5}{'evals':>8}{'fires':>7}{'fire%':>8}"
              f"{'resolved':>9}{'win%':>7}{'mean gross':>12}{'t gross':>9}"
              f"{'mean net':>11}{'t net':>8}  verdict")
    print(header)
    print("-" * len(header))
    for r in rows:
        fire_rate = 100.0 * r["fires"] / max(r["evaluations"], 1)
        if r["fires"] == 0:
            # A gate that refuses and a gate that never read the window are
            # THE SAME OBSERVATION here, and the probe cannot separate them
            # either. Reported as an absence of measurement, never as a verdict.
            verdict = "0 FIRES -- refusing or window-blind, indistinguishable"
        elif r["n"] == 0:
            verdict = "fired but NO forward tape: unresolved"
        elif not np.isfinite(r["t_gross"]) or abs(r["t_gross"]) < 2.0:
            verdict = "UNMEASURED (|t| on gross < 2)"
        elif np.isfinite(r["top_share"]) and r["top_share"] >= 40.0:
            # A t>2 mean whose sign is held by one symbol is not a measurement
            # of the RULE, it is a measurement of that symbol's tape.
            verdict = (f"UNMEASURED -- CONCENTRATED: {r['top_symbol']} alone is "
                       f"{r['top_share']:.0f}% of the pooled gross")
        elif r["mean_gross"] > 0:
            verdict = "gross edge measured"
        else:
            verdict = "gross edge measured NEGATIVE"
        win_s = "--" if not np.isfinite(r["win"]) else f"{r['win']:.0f}%"
        print(f"{r['strategy']:<22}{r['window']:>5}{r['evaluations']:>8}{r['fires']:>7}"
              f"{fire_rate:>7.3f}%{r['n']:>9}{win_s:>7}"
              f"{_fmt(r['mean_gross'], 11)}{_fmt(r['t_gross'], 9, 3, pct=False)}"
              f"{_fmt(r['mean_net'], 10)}{_fmt(r['t_net'], 8, 3, pct=False)}"
              f"  {verdict}")

    print()
    print("VERDICT COLUMN. 'UNMEASURED' means the gross mean on that row is")
    print("indistinguishable from zero at its own n -- do NOT report it as")
    print("having, or as lacking, an edge. The worked example is pass 441's")
    print("momentum rule: -0.0184% at t=-0.103 over 5845 evaluations is not a")
    print("small negative edge, it is no measurement at all.")

    crashed = [(r["strategy"], r["raised"]) for r in rows if r["raised"]]
    if crashed:
        print()
        print("--- RAISED during replay (the base contract says evaluate must not) ---")
        for sid, tally in crashed:
            detail = ", ".join(f"{k} x{v}" for k, v in sorted(tally.items()))
            print(f"  {sid:<22} {detail}")

    exiters = [r for r in rows if r["exit_fires"]]
    if exiters:
        print()
        print("--- EXIT directives, counted and NOT priced (no position to close) ---")
        for r in exiters:
            print(f"  {r['strategy']:<22} {r['exit_fires']:>7} exit candidates")

    if blocked:
        print()
        print(f"--- NOT REPLAYABLE: {len(blocked)} strategies, each with its reason ---")
        by_reason: Dict[str, List[str]] = defaultdict(list)
        for sid, reason in blocked:
            by_reason[reason].append(sid)
        for reason, sids in by_reason.items():
            print(f"  {len(sids)} strategies -- {reason}")
            print(f"    {', '.join(sorted(sids))}")
        print()
        print("  These are LISTED, not skipped: a replay that silently dropped")
        print("  them would report an 18-row table as if it were the registry.")

    if args.per_token:
        print()
        print("=" * 78)
        print("PER-TOKEN SPLIT -- cost is a TOKEN choice, and pass 441 measured")
        print("mean gross spanning +0.114% to -9.722% INSIDE one cost cohort.")
        print("=" * 78)
        for r in rows:
            if r["n"] == 0:
                continue
            by_symbol: Dict[str, List[dict]] = defaultdict(list)
            for res in r["resolved"]:
                by_symbol[res["symbol"]].append(res)
            shown = [(s, g) for s, g in by_symbol.items() if len(g) >= args.min_token_n]
            if not shown:
                continue
            print()
            print(f"{r['strategy']}  ({r['n']} resolved over {len(by_symbol)} symbols; "
                  f"{len(by_symbol) - len(shown)} symbols below n={args.min_token_n} omitted)")
            print(f"  {'symbol':<22}{'n':>5}{'win%':>7}{'mean gross':>12}{'t gross':>9}"
                  f"{'mean net':>11}{'cost':>9}")
            for symbol, group in sorted(
                shown, key=lambda kv: -float(np.mean([x["net"] for x in kv[1]]))
            ):
                g = np.array([x["gross"] for x in group], dtype=float)
                nt = np.array([x["net"] for x in group], dtype=float)
                win = 100.0 * float((nt > 0).sum()) / len(group)
                print(f"  {symbol:<22}{len(group):>5}{win:>6.0f}%"
                      f"{_fmt(float(g.mean()), 11)}{_fmt(_t_stat(g), 9, 3, pct=False)}"
                      f"{_fmt(float(nt.mean()), 10)}"
                      f"{100.0 * group[0]['cost']:>8.2f}%")

    print()
    print(f"replayed {len(replayable)} strategies in {time.time() - started:.0f}s. "
          "Nothing was written: no ledger, no registry, no approval, no threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
