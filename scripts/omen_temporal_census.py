#!/usr/bin/env python3
"""Is bar-count time being crossed with wall-clock time, and what IS the
existing temporal collection?

Two node-free audits, in the order they matter. Both are cheap, both are
decisive, and the FIRST one can invalidate a held-out number -- which is why
it runs before anything is added.

  SECTION 1 -- THE GRID AUDIT. The brain is TRAINED on a uniform hourly grid
  built from stored OHLCV, and QUERIED at live from a tick stream resampled to
  60-second buckets. A bar index means a different amount of wall clock in the
  two places. The horizon collection carries the bar COUNT and nothing else
  (``hzn h=12``), so the same atom is "twelve hours ahead" in training and
  "twelve minutes ahead" at inference. This section measures both cadences off
  real data rather than asserting them, prints the two frames side by side,
  and exits nonzero when they are the same token for different questions.

  SECTION 2 -- WHAT THE TEMPORAL COLLECTION ACTUALLY IS. Pass-107 reported
  ``collection_distinctness temporal = 1.0``. A stream unique on every sample
  is an INDEX: it can be memorised and never generalised, and the fabric's
  98.7%-recall / chance-generalisation gap is exactly what an index produces.
  The question this section answers is not "is it 1.0" but WHY -- whether one
  slot is a counter or a timestamp (a defect to fix), or whether the 1.0 is
  the conjunction of eleven honest slots each of which is fine on its own
  (not a defect, but it means the collection can only ever be queried as a
  whole and the fabric never sees a reusable part). Those two have opposite
  fixes, so the per-slot census is the thing, not the headline number.

    python -X utf8 scripts/omen_temporal_census.py \
        --corpus data/brain_experiments/p108_aero_up.json \
        --corpus data/brain_experiments/p108_aero_down.json \
        --horizon-minutes 720

The horizon is asked in MINUTES and converted with each corpus's OWN modal
gap, because two corpora at different cadences asked the same '--horizon 12'
and got two different wall-clock questions under one name -- which is the
exact crossing this script was written to detect. --horizon still takes bars
as an explicit override.

Exit codes: 0 when the grid audit finds no crossing, 2 when it does. Nothing
here trains, queries a node, or writes to a brain directory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.omen_brain import (  # noqa: E402
    COLLECTIONS, LOOKBACK_BARS, RETURN_SPANS, build_collections,
    horizon_frame, measure_bar_seconds,
    collection_distinctness,
)
from scripts.omen_experiment import (  # noqa: E402
    add_horizon_args, horizon_bars as horizon_bars_for, horizon_request,
    settle_horizon, validate_report_horizon,
)

#: The live resampling constants, read from the strategy rather than restated,
#: so this audit cannot drift away from the code it is auditing.
from trading.strategies.omen_reversion import (  # noqa: E402
    BAR_SECONDS as LIVE_BAR_SECONDS,
    HORIZON_BARS as LIVE_HORIZON_BARS,
)


def load_bars(path: Path) -> List[Dict[str, Any]]:
    """Same loader as omen_experiment, so the two see an identical corpus."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    bars = raw if isinstance(raw, list) else raw.get("bars") or raw.get("data") or []
    bars = [b for b in bars if b.get("close")]
    bars.sort(key=lambda b: int(b["timestamp"]))
    return bars


# --------------------------------------------------------------------------
# SECTION 1 -- the grid audit
# --------------------------------------------------------------------------

def grid_profile(bars: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Cadence and uniformity of a bar series, measured not assumed.

    ``uniform_share`` is the fraction of consecutive gaps equal to the modal
    gap. A training corpus off stored OHLCV should be at or near 1.0; anything
    materially below it means the corpus itself has holes and a bar index does
    not mean a fixed amount of wall clock even THERE.
    """
    stamps = [int(b["timestamp"]) for b in bars]
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    if not gaps:
        return {"bars": len(bars), "modal_gap_sec": 0, "uniform_share": 0.0,
                "median_gap_sec": 0, "max_gap_sec": 0}
    modal = Counter(gaps).most_common(1)[0][0]
    return {
        "bars": len(bars),
        "modal_gap_sec": int(modal),
        "median_gap_sec": int(statistics.median(gaps)),
        "max_gap_sec": int(max(gaps)),
        "uniform_share": sum(1 for g in gaps if g == modal) / len(gaps),
        "span_hours": (stamps[-1] - stamps[0]) / 3600.0,
    }


def horizon_frames(bars: Sequence[Mapping[str, Any]], horizon: int,
                   symbol: str) -> str:
    """The horizon frame the fabric is actually handed, for this horizon."""
    index = max(LOOKBACK_BARS, len(bars) - horizon - 1)
    frames = build_collections(bars, index, horizon_bars=horizon,
                               bar_seconds=measure_bar_seconds(bars),
                               symbol=symbol, chain="base")
    return frames["horizon"]


def live_tick_profile(hours: float, bar_seconds: int,
                      limit_symbols: int = 8) -> Optional[Dict[str, Any]]:
    """What the live tick stream's spacing looks like, per symbol.

    The live path buckets ticks into ``bar_seconds`` and DROPS empty buckets
    (``bars_from_samples``: "Gaps are dropped rather than forward-filled").
    Dropping is right for price integrity and wrong for the index: the bar
    list that comes out is indexed by bar COUNT, so bar i and bar i+1 are one
    bucket apart in the list and any number of buckets apart in wall clock.
    ``filled_share`` is how often that gap is exactly one bucket.

    Returns None when the database is unreachable -- the grid verdict does not
    depend on this, it only quantifies how bad the live side is.
    """
    import time as _time
    since = int(_time.time()) - int(hours * 3600)
    try:
        from db import get_db  # noqa: WPS433 -- optional, audit still runs

        db = get_db()
        with db._cursor() as cur:  # noqa: SLF001 -- read-only audit query
            cur.execute(
                "SELECT symbol, ts FROM market_stream WHERE ts >= ? ORDER BY ts",
                (since,))
            rows = cur.fetchall()
    except Exception as exc:  # pragma: no cover -- environment dependent
        return {"error": f"{type(exc).__name__}: {exc}"}
    by_symbol: Dict[str, List[int]] = {}
    for row in rows or []:
        symbol = str(row[0])
        by_symbol.setdefault(symbol, []).append(int(row[1]))
    out: List[Dict[str, Any]] = []
    ranked = sorted(by_symbol.items(), key=lambda kv: -len(kv[1]))
    for symbol, stamps in ranked[:limit_symbols]:
        buckets = sorted({ts // bar_seconds for ts in stamps})
        if len(buckets) < 2:
            continue
        gaps = [b - a for a, b in zip(buckets, buckets[1:])]
        span = buckets[-1] - buckets[0] + 1
        filled = len(buckets) / span
        out.append({
            "symbol": symbol,
            "ticks": len(stamps),
            "closed_bars": len(buckets),
            "possible_bars": span,
            "filled_share": filled,
            "adjacent_share": sum(1 for g in gaps if g == 1) / len(gaps),
            "max_gap_bars": max(gaps),
            # One step of the bar INDEX, in wall-clock seconds. Nominal is
            # bar_seconds; this is what it really is on this tape.
            "median_step_sec": statistics.median(gaps) * bar_seconds,
            # The live path fetches (LOOKBACK_BARS + 2) * bar_seconds of ticks
            # and refuses unless LOOKBACK_BARS + 1 CLOSED bars come out. At
            # this density, how many actually form?
            "bars_in_live_window": filled * (LOOKBACK_BARS + 2),
            "bars_required": LOOKBACK_BARS + 1,
        })
    return {"hours": hours, "bar_seconds": bar_seconds, "symbols": out,
            "total_ticks": sum(len(v) for v in by_symbol.values())}


# --------------------------------------------------------------------------
# SECTION 2 -- what the temporal collection contains
# --------------------------------------------------------------------------

def slot_census(frame_sets: Sequence[Mapping[str, str]],
                collection: str) -> List[Dict[str, Any]]:
    """Per-slot distinctness inside one collection's frame.

    A frame is ``prefix k=v k=v ...``. This splits it back into slots and
    measures each one on its own. The reading that matters:

      * a slot at distinctness ~1.0 with as many distinct values as samples is
        an INDEX -- a counter or a stamp -- and no fabric can generalise it;
      * every slot well below 1.0 while the WHOLE frame is 1.0 means the
        collection is an honest conjunction that happens to be unique, which
        is a topology problem (nothing reusable is ever bound) rather than a
        contamination one.
    """
    total = len(frame_sets)
    if total <= 0:
        return []
    slots: Dict[str, List[str]] = {}
    order: List[str] = []
    for frames in frame_sets:
        text = frames.get(collection)
        if not text:
            continue
        for token in text.split()[1:]:  # [0] is the prefix
            key, _, value = token.partition("=")
            if key not in slots:
                slots[key] = []
                order.append(key)
            slots[key].append(value)
    out = []
    for key in order:
        values = slots[key]
        counts = Counter(values)
        top_value, top_n = counts.most_common(1)[0]
        out.append({
            "slot": key,
            "distinct": len(counts),
            "distinctness": len(counts) / max(1, len(values)),
            "top_value": top_value,
            "top_share": top_n / max(1, len(values)),
        })
    return out


def build_frame_sets(bars: Sequence[Mapping[str, Any]], symbol: str,
                     horizon: int, start: int, stop: int
                     ) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    cadence = measure_bar_seconds(bars)
    for index in range(max(start, LOOKBACK_BARS), min(stop, len(bars) - horizon)):
        try:
            out.append(build_collections(bars, index, horizon_bars=horizon,
                                         bar_seconds=cadence,
                                         symbol=symbol, chain="base"))
        except (ValueError, IndexError):
            continue
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", required=True,
                        help="repeatable; one per window")
    parser.add_argument("--symbol", default="AERO-USDC")
    add_horizon_args(parser)
    parser.add_argument("--samples", type=int, default=900,
                        help="bars to census per corpus, newest-first window")
    # 48 and not 6, and the difference is a result this script got wrong once.
    # At 3600s buckets, six hours gives about six buckets per symbol, so
    # "every bucket filled" is six-for-six and the sweep's widest row reads a
    # meaningless 1.000. Over 48h the same row reads 0.879. A denominator this
    # small is exactly the error the width sweep exists to avoid making.
    parser.add_argument("--live-hours", type=float, default=48.0)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)

    try:
        asked = horizon_request(args)
    except ValueError as exc:
        print(f"cannot resolve horizon: {exc}")
        return 2

    report: Dict[str, Any] = {"symbol": args.symbol,
                              "horizon_source": ("bars" if asked["bars"]
                                                 is not None else "minutes"),
                              "corpora": [], "live": None}

    print("=" * 74)
    print("SECTION 1 -- THE GRID AUDIT: is bar-count time crossed with wall clock?")
    print("=" * 74)
    print(f"live resampling constants, read from trading/strategies/omen_reversion.py:")
    print(f"  OMEN_BAR_SECONDS   = {LIVE_BAR_SECONDS}")
    print(f"  OMEN_HORIZON_BARS  = {LIVE_HORIZON_BARS}")
    print(f"  LOOKBACK_BARS      = {LOOKBACK_BARS}")
    print()

    cadences: List[Tuple[str, Dict[str, Any]]] = []
    all_frames: Dict[str, List[Dict[str, str]]] = {}
    for path_text in args.corpus:
        path = Path(path_text)
        bars = load_bars(path)
        profile = grid_profile(bars)
        cadences.append((path.name, profile))
        print(f"  corpus {path.name}")
        print(f"    {profile['bars']} bars, modal gap {profile['modal_gap_sec']}s, "
              f"median {profile['median_gap_sec']}s, max {profile['max_gap_sec']}s")
        print(f"    uniform_share {profile['uniform_share']:.4f} "
              f"over {profile.get('span_hours', 0.0):.1f}h")
        # Converted with THIS corpus's own modal gap, which is the cadence the
        # frame is built at two lines down. One bar count across corpora of
        # different cadence is the crossing this whole script exists to name,
        # and the script was committing it in its own argument parsing.
        cadence_here = int(profile["modal_gap_sec"] or 3600)
        resolved = settle_horizon(args, cadence_here, label=path.name)
        bars_here = resolved["horizon_bars"]
        frame = horizon_frames(bars, bars_here, args.symbol)
        train_minutes = bars_here * cadence_here / 60.0
        print(f"    horizon frame handed to the fabric: {frame!r} "
              f"= {train_minutes:.0f} min ahead ({bars_here} bars)")
        report["corpora"].append({"corpus": path.name, "grid": profile,
                                  "horizon_frame": frame,
                                  "bar_seconds": cadence_here,
                                  "horizon_bars": bars_here,
                                  "horizon_minutes": train_minutes})
        all_frames[path.name] = build_frame_sets(
            bars, args.symbol, bars_here,
            max(LOOKBACK_BARS, len(bars) - bars_here - args.samples),
            len(bars) - bars_here)

    # The headline triple, at the median of the corpora's own cadences. The
    # per-corpus list above is the one that carries what each file asked; this
    # one exists so a reader of the top of the file knows the question, and so
    # validate_report_horizon can refuse a report that does not say.
    corpus_cadences = sorted(c["bar_seconds"] for c in report["corpora"])
    median_cadence = (corpus_cadences[len(corpus_cadences) // 2]
                      if corpus_cadences else 3600)
    report["bar_seconds"] = median_cadence
    report["horizon_bars"] = (asked["bars"] if asked["bars"] is not None
                              else horizon_bars_for(asked["minutes"],
                                                    median_cadence))
    report["horizon_minutes"] = round(
        report["horizon_bars"] * median_cadence / 60.0, 4)

    live_minutes = LIVE_HORIZON_BARS * LIVE_BAR_SECONDS / 60.0
    # Built by the SAME function the fabric is handed, not re-spelled here.
    # Re-spelling it is how the crossing hid: two literals that happened to
    # agree told nobody they were answering different questions.
    live_frame = horizon_frame(LIVE_HORIZON_BARS, LIVE_BAR_SECONDS)
    print()
    print(f"  live frame the SAME code emits: {live_frame!r} "
          f"= {live_minutes:.0f} min ahead")

    crossings = [c for c in report["corpora"]
                 if c["horizon_frame"] == live_frame
                 and abs(c["horizon_minutes"] - live_minutes) > 1.0]
    report["crossed"] = bool(crossings)
    print()
    if crossings:
        ratio = crossings[0]["horizon_minutes"] / max(live_minutes, 1e-9)
        print(f"  VERDICT: CROSSED. The horizon collection carries the bar COUNT and")
        print(f"  not the cadence, so {live_frame!r} is the same atom for two")
        print(f"  questions {ratio:.0f}x apart in wall clock "
              f"({crossings[0]['horizon_minutes']:.0f} min trained, "
              f"{live_minutes:.0f} min asked).")
    else:
        print("  VERDICT: NOT CROSSED on these corpora at this horizon.")

    print()
    live = live_tick_profile(args.live_hours, LIVE_BAR_SECONDS)
    report["live"] = live
    if not live or live.get("error"):
        print(f"  live tick census unavailable: {live and live.get('error')}")
    else:
        print(f"  live tick spacing, last {live['hours']:.1f}h, "
              f"{live['total_ticks']} ticks, bucketed at {live['bar_seconds']}s:")
        print(f"    {'symbol':<16} {'ticks':>7} {'bars':>6} {'possible':>9} "
              f"{'filled':>7} {'adjacent':>9} {'maxgap':>7}")
        for row in live["symbols"]:
            print(f"    {row['symbol']:<16} {row['ticks']:>7} "
                  f"{row['closed_bars']:>6} {row['possible_bars']:>9} "
                  f"{row['filled_share']:>7.3f} {row['adjacent_share']:>9.3f} "
                  f"{row['max_gap_bars']:>7}")
        print("    filled/adjacent below 1.0 means the live bar LIST is not a")
        print("    uniform grid: index i+1 is one bucket later in the list and")
        print("    any number of buckets later in wall clock.")
        print()
        print(f"    one step of the bar INDEX, nominal {LIVE_BAR_SECONDS}s, "
              f"and whether the live path can fire at all:")
        print(f"    {'symbol':<16} {'step_sec':>9} {'bars_formed':>12} "
              f"{'bars_needed':>12}  fires?")
        for row in live["symbols"]:
            fires = "yes" if row["bars_in_live_window"] >= row["bars_required"] else "NO"
            print(f"    {row['symbol']:<16} {row['median_step_sec']:>9.0f} "
                  f"{row['bars_in_live_window']:>12.1f} "
                  f"{row['bars_required']:>12}  {fires}")
        print(f"    bars_formed is filled_share x (LOOKBACK_BARS + 2) -- the live")
        print(f"    path asks for {(LOOKBACK_BARS + 2) * LIVE_BAR_SECONDS / 60:.0f} "
              f"minutes of ticks and refuses under "
              f"{LOOKBACK_BARS + 1} closed bars.")

    if live and not live.get("error") and live.get("symbols"):
        print()
        print(f"  WHICH BAR WIDTH WOULD LET THE LIVE PATH ANSWER AT ALL. Same 6h of")
        print(f"  tape, rebucketed. 'fetch' is (LOOKBACK_BARS + 2) x width, which is")
        print(f"  what the strategy asks the tick buffer for; 'formed' is how many")
        print(f"  closed bars come out of that fetch at the measured density.")
        print(f"    {'width_s':>8} {'fetch_min':>10} {'filled':>8} {'formed':>8} "
              f"{'needed':>7}  fires?  one bar means")
        sweep = []
        for width in (60, 120, 180, 300, 600, 900, 1800, 3600):
            wide = live_tick_profile(args.live_hours, width, limit_symbols=8)
            if not wide or wide.get("error") or not wide["symbols"]:
                continue
            filled = statistics.fmean(r["filled_share"] for r in wide["symbols"])
            formed = filled * (LOOKBACK_BARS + 2)
            fetch_min = (LOOKBACK_BARS + 2) * width / 60.0
            fires = "yes" if formed >= LOOKBACK_BARS + 1 else "NO"
            print(f"    {width:>8} {fetch_min:>10.0f} {filled:>8.3f} "
                  f"{formed:>8.1f} {LOOKBACK_BARS + 1:>7}  {fires:>5}   "
                  f"{width / 3600.0:.3f}h of tape")
            sweep.append({"bar_seconds": width, "fetch_minutes": fetch_min,
                          "mean_filled_share": filled, "bars_formed": formed,
                          "bars_required": LOOKBACK_BARS + 1,
                          "fires": fires == "yes"})
        report["bar_width_sweep"] = sweep
        print("    A WIDER BAR IS NOT A FREE FIX: it changes what one bar MEANS,")
        print("    and the fabric was trained where one bar is 1.000h of tape. The")
        print("    row whose last column reads 1.000h is the only width that both")
        print("    fires and matches the training cadence -- at the price of a")
        print(f"    {(LOOKBACK_BARS + 2) * 3600 / 3600.0:.0f}-hour tick buffer.")

    print()
    print("=" * 74)
    print("SECTION 2 -- WHAT THE EXISTING TEMPORAL COLLECTION CONTAINS")
    print("=" * 74)
    print(f"  RETURN_SPANS = {RETURN_SPANS}; the frame is "
          f"'tmp r<span>=... flip=N streak=N acc=...'")
    report["collections"] = {}
    for name, frames in all_frames.items():
        if not frames:
            print(f"  {name}: no frames built")
            continue
        distinctness = collection_distinctness(frames)
        print()
        print(f"  {name} -- {len(frames)} samples")
        print(f"    collection distinctness:")
        for collection in COLLECTIONS:
            value = distinctness.get(collection.name)
            if value is None:
                continue
            flag = "  <- INDEX-LIKE" if value >= 0.99 else ""
            print(f"      {collection.name:<18} {value:.4f}{flag}")
        census = slot_census(frames, "temporal")
        print(f"    temporal, per slot:")
        print(f"      {'slot':<10} {'distinct':>9} {'distinctness':>13} "
              f"{'top value':>10} {'top share':>10}")
        for row in census:
            print(f"      {row['slot']:<10} {row['distinct']:>9} "
                  f"{row['distinctness']:>13.4f} {row['top_value']:>10} "
                  f"{row['top_share']:>10.3f}")
        worst = max(census, key=lambda r: r["distinctness"]) if census else None
        report["collections"][name] = {
            "samples": len(frames),
            "distinctness": distinctness,
            "temporal_slots": census,
        }
        if worst and worst["distinctness"] >= 0.99:
            print(f"      -> slot {worst['slot']!r} is unique on every sample: "
                  f"an INDEX, and a defect to fix before adding a pool.")
        else:
            print(f"      -> no slot is an index (max slot distinctness "
                  f"{worst['distinctness']:.4f} on {worst['slot']!r}). The 1.0 "
                  f"is the CONJUNCTION of {len(census)} honest slots.")

    if args.json_out:
        validate_report_horizon(report)
        Path(args.json_out).write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    return 2 if report["crossed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
