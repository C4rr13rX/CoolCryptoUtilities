"""Is the money_button cost gate correct caution, or a mis-set threshold?

The gate census (scripts/money_button_gate_census.py) says money_button fires
2 times in 33,392 evaluations and names `momentum_not_aligned` as the biggest
single decline at 57%. That framing is misleading in two ways, and this script
exists to correct both with measurement rather than argument.

FIRST, the census percentages are shares of ALL evaluations, but the gates run
in a CHAIN. A gate can only decline what reached it. `momentum_not_aligned`
declining 57% of everything is a different claim from it declining 57% of what
it saw, and only the second one is a statement about the threshold. This
script reports the CONDITIONAL rate at each gate.

SECOND, and this is the whole point: you cannot tell whether a gate is set
correctly by looking at how often it fires. You can only tell by looking at
what happened NEXT to the candidates it refused. A gate that refuses 3,000
candidates which would all have lost money is working perfectly. A gate that
refuses 3,000 candidates which would have won is the reason this lane has not
traded in 12.3 days.

So for every evaluation that a gate declined, this replays the tape forward
over the strategy's own holding period and records what the trade WOULD have
returned, gross and net of the round trip. The verdict on each threshold is
then arithmetic, not taste:

    refused candidates with a POSITIVE mean net  -> the gate is costing money
    refused candidates with a NEGATIVE mean net  -> the gate is earning its keep

No gate is re-implemented here. It calls MoneyButtonStrategy.evaluate and
reads the reason the strategy recorded, exactly as the census does, so this
cannot drift away from the code it measures. The term values (projected,
volatility, conservative_edge, required_edge) are recomputed from the same
window using the strategy module's OWN helpers.

Overlapping windows share ticks, so a run of adjacent evaluations is not a run
of independent samples. Every population is therefore reported twice: `raw`
counts every evaluation, and `sequential` admits a new one only after the
previous hold has closed. Believe the sequential number.

Usage:
    python -X utf8 scripts/money_button_edge_terms.py
    python -X utf8 scripts/money_button_edge_terms.py --hours 24
    python -X utf8 scripts/money_button_edge_terms.py --fee 0.0065   # live fee
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the census's loader and RouteState stand-in rather than making a second
# copy that can disagree with it.
from money_button_gate_census import FEED_GATES, _State, load_ticks  # noqa: E402

#: Gates that run AFTER the momentum triple. Used to reconstruct how many
#: evaluations actually reached each point in the chain.
POST_MOMENTUM = (
    "move_exhausted",
    "slope_not_positive",
    "projection_not_positive",
    "edge_below_cost",
    "volume_faded",
    "confidence_below_floor",
    "(fired)",
)


def _forward_return(
    ticks: List[Tuple[float, float, float]], end: int, hold_sec: float
) -> Tuple[float, float] | None:
    """Gross return over the holding period, and the ts it closed at.

    Returns None when the hold runs past the end of the recorded tape -- an
    unresolved trade must not be scored as a flat one, which would drag every
    mean toward zero and make a bad gate look harmless.
    """
    entry_ts, entry_px, _ = ticks[end]
    if entry_px <= 0:
        return None
    target = entry_ts + hold_sec
    for j in range(end + 1, len(ticks)):
        if ticks[j][0] >= target:
            exit_px = ticks[j][1]
            if exit_px <= 0:
                return None
            return (exit_px / entry_px) - 1.0, ticks[j][0]
    return None


def _summarise(rows: List[float], label: str, round_trip: float) -> None:
    """Print win rate and mean for one population, gross and net."""
    if not rows:
        print("    %-28s  (none)" % label)
        return
    arr = np.asarray(rows, dtype=float)
    net = arr - round_trip
    wins = int((net > 0).sum())
    print(
        "    %-28s  n=%-6d  net mean %+7.4f%%  win %5.1f%%  gross mean %+7.4f%%  up %5.1f%%"
        % (
            label,
            arr.size,
            net.mean() * 100.0,
            wins / arr.size * 100.0,
            arr.mean() * 100.0,
            float((arr > 0).sum()) / arr.size * 100.0,
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--fee", type=float, default=None)
    ap.add_argument("--exclude-recent-hours", type=float, default=0.0,
                    help="drop the most recent N hours, so --hours 336 "
                         "--exclude-recent-hours 168 measures a window that is "
                         "DISJOINT from the last 168h. This is how a lead found "
                         "in-sample gets tested out-of-sample without refitting "
                         "anything.")
    args = ap.parse_args()

    from trading.strategies.base import StrategyContext
    from trading.strategies.money_button import MoneyButtonStrategy
    from trading.strategies.money_button import log_slope_per_min

    fee = args.fee if args.fee is not None else float(os.getenv("TRADE_FEE_RATE", "0.003"))
    strat = MoneyButtonStrategy()
    lookback = strat.LOOKBACK_SEC

    hold_min = float(os.getenv("MONEY_BUTTON_HOLD_MIN", "12"))
    hold_sec = hold_min * 60.0
    slippage = float(os.getenv("MONEY_BUTTON_SLIPPAGE", "0.001"))
    decay = float(os.getenv("MONEY_BUTTON_DECAY", "0.5"))
    margin = float(os.getenv("MONEY_BUTTON_MARGIN", "1.1"))
    round_trip = 2.0 * fee + slippage
    required_edge = round_trip * margin

    by_symbol = load_ticks(args.hours)
    if args.exclude_recent_hours > 0:
        newest = max(t[0] for ticks in by_symbol.values() for t in ticks)
        cut = newest - args.exclude_recent_hours * 3600.0
        by_symbol = {
            sym: [t for t in ticks if t[0] < cut]
            for sym, ticks in by_symbol.items()
        }
        by_symbol = {s_: t for s_, t in by_symbol.items() if len(t) >= 2}
        print("OUT-OF-SAMPLE: dropped the most recent %.0fh; %d ticks remain."
              % (args.exclude_recent_hours,
                 sum(len(v) for v in by_symbol.values())))
    if not by_symbol:
        print("no ticks in the window; nothing to measure.")
        return 0

    #: gate -> list of gross forward returns, for every evaluation that gate
    #: declined and whose hold resolved inside the tape.
    fwd_raw: Dict[str, List[float]] = defaultdict(list)
    fwd_seq: Dict[str, List[float]] = defaultdict(list)
    #: gate -> how many evaluations it declined (resolved or not)
    seen: Dict[str, int] = defaultdict(int)
    #: the four terms of the cost gate, for the edge_below_cost population
    terms: List[Tuple[float, float, float]] = []   # projected, volatility, conservative
    #: counterfactuals: would this refusal have fired under a variant rule?
    variants: Dict[str, List[float]] = defaultdict(list)
    variants_seq: Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # The horizon sweep. The cost gate refuses because a 12-minute
    # extrapolation cannot reach 0.77%; the operator's stated opportunity is
    # "2-5% over 5-30 minutes, with room for 45-60". Those are not the same
    # clock, so measure the signal at both and let the tape decide.
    #
    # Three populations at every horizon, because a signal is only worth
    # anything if it beats doing nothing:
    #   null    -- every evaluation that got past the feed gates
    #   aligned -- passed the momentum triple
    #   clean   -- passed momentum AND move_exhausted AND slope_not_positive,
    #              i.e. the candidates ONLY the cost gate refused
    # ------------------------------------------------------------------
    # Pass 480: extended DOWNWARD. Everything ever screened here was 12 min+,
    # but the feed's median per-symbol gap is 66.6s and 101 of 111 symbols
    # sample at <=2 min, so a 3-min hold already contains ~3 ticks. The band
    # below 12 min is the only one never tested, and the tradeable book is
    # POSITIVE inside 15 min and negative after -- which is the whole reason
    # to look here. The old six are kept so the comparison is one table.
    HORIZONS = (2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 45.0, 60.0, 90.0)
    CLEAN_GATES = {"edge_below_cost", "volume_faded", "confidence_below_floor", "(fired)"}
    hz: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    hz_seq: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    #: sequential admission is per (population, horizon) -- a 90-min hold
    #: blocks far more re-entries than a 12-min one.
    hz_free: Dict[Tuple[str, float], float] = defaultdict(float)

    evaluated = 0
    unresolved = 0

    for symbol, ticks in by_symbol.items():
        start = 0
        free_at = 0.0     # sequential: no new sample until the last hold closed
        # Per SYMBOL, not per run. Timestamps are absolute epoch, so a single
        # global tracker lets one symbol's open hold block every other
        # symbol's next sample -- which silently collapsed the horizon sweep
        # from thousands of samples to tens and let one outlier set the mean.
        hz_free.clear()
        for end in range(len(ticks)):
            cutoff = ticks[end][0] - lookback
            while ticks[start][0] < cutoff:
                start += 1
            window = ticks[start:end + 1]
            ctx = StrategyContext(
                chain="base",
                last_price=window[-1][1],
                last_volume=window[-1][2],
                fee_rate=fee,
                available_quote=10.0,
                available_base=0.0,
            )
            result = strat.evaluate(_State(window, symbol), ctx)
            evaluated += 1
            gate = "(fired)" if result is not None else (strat.last_decline or "unknown")
            seen[gate] += 1

            # Feed gates never looked at the market; scoring a forward return
            # for them measures the tape, not the threshold.
            if gate in FEED_GATES:
                continue

            # Horizon sweep runs on every non-feed evaluation, independently of
            # whether the 12-min hold resolved -- a 12-min hold running off the
            # tape says nothing about a 90-min one and vice versa.
            pops = ["null"]
            if gate in POST_MOMENTUM:
                pops.append("aligned")
            if gate in CLEAN_GATES:
                pops.append("clean")

            # --- the INVERTED entry, on the same window and the same scaling.
            # money_button:278 requires r5>0 AND r10>0 AND r30>0, so it only
            # ever buys a confirmed uptrend. The operator is describing buying
            # 2-5% DIPS, which is the opposite predicate. Continuation is
            # closed by the table above; this tests reversion on identical
            # machinery so the two verdicts are comparable.
            #
            # `_return_over` and the unit scaling are the STRATEGY'S OWN, not
            # a second copy -- a dip measured with different windows than the
            # uptrend would not be a comparison.
            ts_w = np.asarray([t[0] for t in window], dtype=float)
            px_w = np.asarray([t[1] for t in window], dtype=float)
            if ts_w.size > 1:
                gap_w = float(np.median(np.diff(ts_w)))
                unit_w = max(5.0 * 60.0, 2.0 * gap_w)
                d5 = strat._return_over(ts_w, px_w, unit_w)
                d10 = strat._return_over(ts_w, px_w, unit_w * 2.0)
                d30 = strat._return_over(ts_w, px_w, unit_w * 6.0)
                if d5 < 0.0 and d10 < 0.0 and d30 < 0.0:
                    pops.append("dip")
                    # The operator's stated size: a 2%+ fall over the long
                    # window. A "dip" of 0.05% is noise wearing the name.
                    if d30 <= -0.02:
                        pops.append("dip>=2%")
            for h in HORIZONS:
                fwd_h = _forward_return(ticks, end, h * 60.0)
                if fwd_h is None:
                    continue
                g_h, x_ts = fwd_h
                for pop in pops:
                    hz[(pop, h)].append(g_h)
                    if ticks[end][0] >= hz_free[(pop, h)]:
                        hz_seq[(pop, h)].append(g_h)
                        hz_free[(pop, h)] = x_ts

            fwd = _forward_return(ticks, end, hold_sec)
            if fwd is None:
                unresolved += 1
                continue
            gross, exit_ts = fwd
            fwd_raw[gate].append(gross)
            is_seq = ticks[end][0] >= free_at
            if is_seq:
                fwd_seq[gate].append(gross)

            if gate != "edge_below_cost":
                if is_seq:
                    free_at = exit_ts
                continue

            # --- recompute the cost gate's own terms on this same window ---
            ts = np.asarray([t[0] for t in window], dtype=float)
            prices = np.asarray([t[1] for t in window], dtype=float)
            slope = log_slope_per_min(ts, prices)
            projected = float(np.expm1(slope * hold_min))
            steps = np.diff(np.log(np.clip(prices, 1e-12, None)))
            vol = float(np.std(steps)) if steps.size else 0.0
            conservative = max(0.0, projected * decay - 0.5 * vol)
            terms.append((projected, vol, conservative))

            # Counterfactuals, each removing ONE discount, so the term that is
            # actually binding can be named rather than guessed at.
            if projected * decay >= required_edge:
                variants["drop the volatility penalty"].append(gross)
                if is_seq:
                    variants_seq["drop the volatility penalty"].append(gross)
            if max(0.0, projected - 0.5 * vol) >= required_edge:
                variants["drop the momentum decay"].append(gross)
                if is_seq:
                    variants_seq["drop the momentum decay"].append(gross)
            if projected >= required_edge:
                variants["drop both discounts"].append(gross)
                if is_seq:
                    variants_seq["drop both discounts"].append(gross)
            if projected >= round_trip:
                variants["raw projection clears cost"].append(gross)
                if is_seq:
                    variants_seq["raw projection clears cost"].append(gross)
            if is_seq:
                free_at = exit_ts

    print("money_button edge terms -- %.0fh of stored ticks, one-way fee %.4f"
          % (args.hours, fee))
    print("%d symbols, %d evaluations, hold %.0f min, round trip %.4f, required edge %.4f"
          % (len(by_symbol), evaluated, hold_min, round_trip, required_edge))
    print("%d evaluations declined past the feed gates but ran off the end of the tape"
          % unresolved)

    # ------------------------------------------------------------------
    # The chain, with CONDITIONAL rates. This is the number the census
    # could not give, and it is the one that says whether a gate is tight.
    # ------------------------------------------------------------------
    feed_blocked = sum(seen[g] for g in FEED_GATES)
    reached_momentum = evaluated - feed_blocked - seen.get("mixed_denomination", 0) \
        - seen.get("window_exceeds_lookback", 0) - seen.get("feed_too_sparse_for_hold", 0)
    past_momentum = reached_momentum - seen.get("momentum_not_aligned", 0)
    print("\n--- the chain, as CONDITIONAL rates (a gate can only decline what reached it) ---")
    print("  %-26s %8d" % ("evaluations", evaluated))
    print("  %-26s %8d   (%.1f%% of evaluations never looked at the market)"
          % ("blocked by the feed", feed_blocked, feed_blocked / evaluated * 100.0))
    print("  %-26s %8d" % ("reached the momentum gate", reached_momentum))
    if reached_momentum:
        d = seen.get("momentum_not_aligned", 0)
        print("  %-26s %8d   declines %.2f%% OF WHAT IT SAW  (chance on 3 nested windows: 87.5%%)"
              % ("  momentum_not_aligned", d, d / reached_momentum * 100.0))
    print("  %-26s %8d" % ("passed the momentum gate", past_momentum))
    for g in POST_MOMENTUM:
        n = seen.get(g, 0)
        if n and past_momentum:
            print("  %-26s %8d   %.2f%% of survivors" % ("  " + g, n, n / past_momentum * 100.0))

    # ------------------------------------------------------------------
    # What happened NEXT to what each gate refused.
    # ------------------------------------------------------------------
    print("\n--- what the refused candidates actually did over the %.0f-min hold ---"
          % hold_min)
    print("  a gate whose refusals have a NEGATIVE net mean is earning its keep.")
    print("  RAW (overlapping windows, not independent samples):")
    for g in ("momentum_not_aligned",) + POST_MOMENTUM:
        if fwd_raw.get(g):
            _summarise(fwd_raw[g], g, round_trip)
    print("  SEQUENTIAL (no re-entry while still holding) -- believe these:")
    for g in ("momentum_not_aligned",) + POST_MOMENTUM:
        if fwd_seq.get(g):
            _summarise(fwd_seq[g], g, round_trip)

    # ------------------------------------------------------------------
    # Inside the cost gate: which discount is binding?
    # ------------------------------------------------------------------
    if terms:
        arr = np.asarray(terms, dtype=float)
        print("\n--- inside edge_below_cost (n=%d): which term is binding? ---" % arr.shape[0])
        for i, name in enumerate(("projected move", "per-tick volatility", "conservative edge")):
            col = arr[:, i]
            print("  %-22s median %+8.4f%%  p75 %+8.4f%%  p95 %+8.4f%%  max %+8.4f%%"
                  % (name, np.median(col) * 100, np.percentile(col, 75) * 100,
                     np.percentile(col, 95) * 100, col.max() * 100))
        print("  required edge          %+8.4f%%" % (required_edge * 100))
        zeroed = int((arr[:, 2] <= 0).sum())
        print("  refusals where the volatility penalty ate the WHOLE edge: %d (%.1f%%)"
              % (zeroed, zeroed / arr.shape[0] * 100.0))

    if variants:
        print("\n--- counterfactual: loosen ONE discount, and what do the new fires return? ---")
        print("  a variant is only progress if its net mean is POSITIVE.")
        for name in ("drop the volatility penalty", "drop the momentum decay",
                     "drop both discounts", "raw projection clears cost"):
            if variants.get(name):
                _summarise(variants[name], name + " [raw]", round_trip)
            if variants_seq.get(name):
                _summarise(variants_seq[name], name + " [seq]", round_trip)

    # ------------------------------------------------------------------
    # Does the signal have an edge at ANY horizon?
    # ------------------------------------------------------------------
    print("\n--- horizon sweep: does the momentum signal beat doing nothing? ---")
    print("  net = gross - %.4f round trip. 'lift' = aligned gross - null gross;" % round_trip)
    print("  a signal with skill has POSITIVE lift (medians; see note in source). Sequential only.")
    print("  %-7s | %-40s | %-40s | %-40s | lift" % ("horizon", "null", "aligned", "clean"))
    for h in HORIZONS:
        cells = []
        means = {}
        meds = {}
        for pop in ("null", "aligned", "clean"):
            rows = hz_seq.get((pop, h), [])
            if not rows:
                cells.append("%-40s" % "(none)")
                continue
            a = np.asarray(rows, dtype=float)
            means[pop] = a.mean()
            meds[pop] = float(np.median(a))
            # Median as well as mean: these tails are heavy and a single
            # -30% print can set a mean over a few hundred samples.
            cells.append("n=%-5d net %+7.4f%% med %+6.3f%% w%4.1f%%"
                         % (a.size, (a.mean() - round_trip) * 100.0,
                            (float(np.median(a)) - round_trip) * 100.0,
                            float(((a - round_trip) > 0).sum()) / a.size * 100.0))
        # Lift on MEDIANS, not means. The null population's mean is
        # outlier-contaminated -- it reads +9.56% net at 90 min while its own
        # median is -0.73% -- because `null` includes windows with
        # denomination breaks and two-regime tickers that the momentum gate's
        # price-scale guard removes from `aligned`. Differencing a
        # contaminated mean against a clean one measures the contamination,
        # not the signal. (That the aligned means are NOT blown up is itself
        # evidence those guards work.)
        lift = ""
        if "aligned" in meds and "null" in meds:
            lift = "%+7.4f%%" % ((meds["aligned"] - meds["null"]) * 100.0)
        print("  %-7.0f | %s | %s | %s | %s"
              % (h, cells[0], cells[1], cells[2] if len(cells) > 2 else "", lift))

    # ------------------------------------------------------------------
    # The same table for the INVERTED entry.
    # ------------------------------------------------------------------
    print("")
    print("--- the same sweep for DIPS (r5<0 and r10<0 and r30<0) ---")
    print("  continuation is closed above; this is the opposite predicate on")
    print("  the same windows. 'dip>=2%' is the operator's stated size.")
    print("  %-7s | %-40s | %-40s | %-40s | lift" % ("horizon", "null", "dip", "dip>=2%"))
    for h in HORIZONS:
        cells, meds2 = [], {}
        for pop in ("null", "dip", "dip>=2%"):
            rows = hz_seq.get((pop, h), [])
            if not rows:
                cells.append("%-40s" % "(none)")
                continue
            a = np.asarray(rows, dtype=float)
            meds2[pop] = float(np.median(a))
            cells.append("n=%-5d net %+7.4f%% med %+6.3f%% w%4.1f%%"
                         % (a.size, (a.mean() - round_trip) * 100.0,
                            (float(np.median(a)) - round_trip) * 100.0,
                            float(((a - round_trip) > 0).sum()) / a.size * 100.0))
        lift = ""
        if "dip" in meds2 and "null" in meds2:
            bps = (meds2["dip"] - meds2["null"]) * 10000.0
            lift = "%+7.4f%% (%+.2f bp)" % ((meds2["dip"] - meds2["null"]) * 100.0, bps)
        # CENSORED. The dip>=2% population reports a mean of +113% to +184%,
        # which is not a return, it is proof of contamination -- and the
        # selection rule is what causes it: a denomination break looks exactly
        # like a huge dip, so "fall 2% or more" preferentially SELECTS broken
        # rows, and the snap-back when the feed returns to the right
        # denomination books as an enormous profit. This population is
        # enriched for the artefact by construction, so the uncensored median
        # cannot be trusted either -- the contaminated rows may be sitting in
        # the middle of it, not only in the tail.
        #
        # 20% over the horizon is this repo's existing plausibility bar.
        raw2 = hz_seq.get(("dip>=2%", h), [])
        if raw2:
            a2 = np.asarray(raw2, dtype=float)
            keep = a2[np.abs(a2) <= 0.20]
            dropped = a2.size - keep.size
            if keep.size:
                print("  %-7.0f |   dip>=2%% CENSORED |%%|<=20%%: n=%-4d net %+7.4f%% med %+6.3f%% w%4.1f%%   (dropped %d of %d as implausible)"
                      % (h, keep.size, (keep.mean() - round_trip) * 100.0,
                         (float(np.median(keep)) - round_trip) * 100.0,
                         float(((keep - round_trip) > 0).sum()) / keep.size * 100.0,
                         dropped, a2.size))
            else:
                print("  %-7.0f |   dip>=2%% CENSORED: every one of %d rows was implausible"
                      % (h, a2.size))
    print("  cost floors to beat: 70.00 bp at this census fee, 11.00 bp at WETH-class depth.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
