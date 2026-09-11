#!/usr/bin/env python3
"""L2 cannot be a fixed-length path over a live L1. Measure the two fixes.

Item [fa75fa1a]. Fixing the encoder's dead slots (7d2a74e) made L1 right and
made the CURRENT L2 an identifier: a motif changes on 72.3% of bars under
relative banding, and a fixed-length path over an alphabet that changes nearly
every bar is near-unique BY CONSTRUCTION however small the alphabet is. L2
measured 0.6400/0.8267 DOWN and 0.5067/0.7850 UP at steps 2 and 3, against a
0.30 identifier ceiling, and it still failed at 0.3167 over a deliberately
coarse 21-symbol alphabet -- so this is not an alphabet-size problem and no
sweep of MOTIF_SEQUENCE_STEPS fixes it.

The design change is to STOP SAMPLING ONE MOTIF PER BAR. Two candidates, and
this probe measures both on one corpus in one process so nothing is cross-run:

  TRANSITIONS   key on the ordered motifs that actually CHANGED, dropping
                repeats, so a persistent regime contributes ONE symbol rather
                than N.
  RUN-LENGTH    carry motif plus a bucketed dwell count, so "this motif for a
                long time" and "this motif briefly" differ while a steady
                stretch stays one symbol.

Both preserve the ORDER that is the whole point of L2. Neither is worth a node
arm until its distinctness clears the ceiling, and this costs no node time:

    python -X utf8 scripts/omen_l2_scheme_probe.py \
        --corpus data/brain_experiments/p108_aero_down.json \
        --corpus data/brain_experiments/p108_aero_up.json

Exits nonzero when NEITHER scheme clears the ceiling in BOTH corpora, so it
gates a pass rather than merely informing it -- the same contract as
scripts/omen_layer_probe.py and scripts/omen_query_path_probe.py.

THIS FILE DELIBERATELY DOES NOT EDIT trading/omen_layers.py OR
scripts/omen_layer_probe.py. Cove holds both for [10140855]; the winning scheme
is wired there by whoever owns them, from the number this prints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.omen_brain import (  # noqa: E402
    LOOKBACK_BARS, build_collections, label_omen,
)
# THE canonical support floor and lift arithmetic, imported rather than
# recopied: two probes that disagree about what "supported" means produce two
# numbers nobody can put in one table.
from scripts.omen_layer_probe import label_skew  # noqa: E402
from trading.omen_layers import (  # noqa: E402
    IDENTIFIER_CEILING, L1_HYSTERESIS_MARGIN, L1_STREAMS, L2_TRANSITION_STEPS,
    MOTIF_SEQUENCE_STEPS, _band_of, _compact_motif, _numeric_of,
    cooccurrence_motif, layer_distinctness, relative_bands, sequence_motif,
    sticky_motifs, transition_motif,
)

# ``sticky_motifs`` was measured here first and now LIVES in
# trading/omen_layers.py ([2a53f971]); ``transition_motif`` was measured here
# and now lives there too ([fa75fa1a], pass 115) because it WON and the live
# path cannot import a probe. Both are re-exported under this module's name so
# callers and tests written against the probe keep working. A probe-local copy
# of a shipped encoder is how the two silently drift apart, which is exactly
# what the margin=0 byte-identity test exists to stop.
#
# ``run_length_motif`` below is deliberately NOT promoted: it is the rejected
# scheme and it stays here so its control number can go on being measured
# beside the winner, which the acceptance criterion requires.

#: Dwell buckets for the run-length scheme. Bucketed rather than raw, because a
#: raw count is an integer that grows without bound and would re-introduce the
#: near-uniqueness the scheme exists to remove. The boundaries are ordinal --
#: "just changed", "a couple of bars", "a stretch", "entrenched".
_DWELL_BUCKETS = ((1, "d1"), (2, "d2"), (4, "d4"), (8, "d8"))


#: The one reduction every scheme reads a motif through, so the comparison is
#: over an IDENTICAL alphabet. Imported rather than redefined: it moved into
#: trading/omen_layers with the winner.
_compact = _compact_motif


def _dwell_bucket(count: int) -> str:
    label = "dN"
    for bound, name in _DWELL_BUCKETS:
        if count <= bound:
            return name
        label = "dN"
    return label


def run_length_motif(motifs: Sequence[str],
                     steps: int = MOTIF_SEQUENCE_STEPS) -> str:
    """L2-R: the last ``steps`` CHANGED motifs, each with a bucketed dwell.

    The difference from L2-T is that a stretch is not merely collapsed, it is
    described: ``hhlmm:d8`` and ``hhlmm:d1`` are the same regime held for a
    long time and glimpsed for one bar, and those are different situations.

    The dwell counts only the run that is INSIDE the supplied window, so this
    frame is a function of the same inputs the other two schemes see.
    """
    if not motifs:
        return "co2r path=na"
    compact = [_compact(m) for m in motifs]
    runs: List[List[Any]] = []
    for item in compact:
        if runs and runs[-1][0] == item:
            runs[-1][1] += 1
        else:
            runs.append([item, 1])
    return "co2r path=%s" % "|".join(
        "%s:%s" % (sym, _dwell_bucket(count)) for sym, count in runs[-steps:])


def build_rows(bars: Sequence[Mapping[str, Any]], symbol: str, chain: str,
               horizon: int, bands: Optional[Mapping[str, Any]],
               window: int, margin: float,
               steps: int = L2_TRANSITION_STEPS) -> List[Dict[str, str]]:
    """One row per bar: its L1 motif and all three L2 schemes.

    ``margin`` IS PART OF THE MEASUREMENT, not a detail of it. L2's frames are
    a function of the L1 alphabet underneath them, so a table built at margin 0
    describes an encoder nobody ships and cannot gate the one we do -- that
    mistake is precisely what blocked [fa75fa1a] in pass 111, reporting "no
    order-carrying scheme can meet the guard" from the only banding under which
    that is true. margin 0 is still measured, as the CONTROL row.

    ``window`` is how many preceding bars the transition and run-length
    schemes may look back over. A fixed-length PATH cannot use a long window
    (that is the defect), but a change-keyed scheme can, because a long steady
    stretch costs it one symbol -- so the window is the knob that lets
    persistence show up at all.

    A bar whose L0 frames cannot be built records a HOLE rather than being
    skipped, so no scheme claims an adjacency the corpus does not have. The
    hole does NOT reset the held band: the market did not stop, only our view
    of it did.
    """
    built: List[Any] = []              # (index, frames) for every buildable bar
    order: List[Optional[int]] = []    # position in `built`, or None for a hole
    for index in range(LOOKBACK_BARS, len(bars)):
        try:
            frames = build_collections(bars, index, horizon_bars=horizon,
                                       symbol=symbol, chain=chain)
        except (ValueError, IndexError):
            order.append(None)
            continue
        order.append(len(built))
        built.append((index, frames))

    # Hysteresis is a fact about a STREAM of bars, so the whole window is
    # encoded in one call rather than bar by bar. margin=0 is byte-identical to
    # the per-bar cooccurrence_motif call, pinned by
    # tests/test_hysteresis_margin_zero_is_byte_identical.py.
    motifs = sticky_motifs([f for _, f in built], bands, margin)

    rows: List[Dict[str, str]] = []
    history: List[str] = []
    for slot in order:
        if slot is None:
            history.append("")            # a hole, not an adjacency
            continue
        index, _frames = built[slot]
        motif = motifs[slot]
        history.append(motif)
        recent = [m for m in history[-window:] if m]
        fixed = [m for m in history[-MOTIF_SEQUENCE_STEPS:] if m]
        rows.append({
            "L1_cooccurrence": motif,
            "L2_sequence": sequence_motif(fixed),
            "L2_transitions": transition_motif(recent, steps=steps),
            "L2_runlength": run_length_motif(recent, steps=steps),
            "_label": label_omen(bars, index, horizon_bars=horizon) or "",
        })
    return rows


def load_bars(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    bars = raw if isinstance(raw, list) else raw.get("bars") or raw.get("data") or []
    bars = [b for b in bars if b.get("close")]
    bars.sort(key=lambda b: int(b["timestamp"]))
    return bars


def measure(path: Path, symbol: str, chain: str, horizon: int,
            window: int, samples: int, margin: float,
            steps: int = L2_TRANSITION_STEPS,
            min_support: int = 20) -> Dict[str, Any]:
    """One corpus, all three schemes, at ONE banding, in ONE process."""
    bars = load_bars(path)
    if samples and len(bars) > samples + LOOKBACK_BARS:
        bars = bars[: samples + LOOKBACK_BARS]

    # Fit the relative bands on THIS corpus's own frames. The cut points are
    # returned rather than applied, so fitting and use are separable -- a
    # held-out arm must reuse the train cut points, and refitting on test
    # would leak the test distribution into the frame.
    frame_sets: List[Dict[str, str]] = []
    for index in range(LOOKBACK_BARS, len(bars)):
        try:
            frame_sets.append(build_collections(bars, index,
                                                horizon_bars=horizon,
                                                symbol=symbol, chain=chain))
        except (ValueError, IndexError):
            continue
    bands = relative_bands(frame_sets, streams=L1_STREAMS)

    rows = build_rows(bars, symbol, chain, horizon, bands, window, margin,
                      steps=steps)
    keys = ["L1_cooccurrence", "L2_sequence", "L2_transitions", "L2_runlength"]
    dist = layer_distinctness(rows, keys=keys)

    changed = sum(1 for a, b in zip(rows, rows[1:])
                  if a["L1_cooccurrence"] != b["L1_cooccurrence"])

    # DISTINCTNESS IS ONLY HALF THE GUARD, and the other half is the one that
    # decides whether a node arm is worth spending. A scheme can clear the
    # ceiling by collapsing to nearly a constant, and it can clear it while
    # having no group large enough to carry a label distribution at all: Jet
    # recorded that under PLAIN relative banding L2 has no group reaching
    # n=20 in either window, which means every L2 number measured before this
    # measured nothing. So the supported-group count rides in the same table.
    skew = {k: label_skew(rows, k, min_support=min_support)
            for k in keys}
    return {
        "corpus": str(path), "samples": len(rows),
        "window": window, "steps": steps, "margin": margin,
        "l1_change_rate": changed / max(1, len(rows) - 1),
        "distinctness": dist,
        "vocab": {k: len({r[k] for r in rows}) for k in keys},
        "supported": {k: len(skew[k].get("groups", [])) for k in keys},
        "covered": {k: skew[k].get("covered", 0.0) for k in keys},
        "best_lift": {k: (skew[k]["groups"][0]["lift"]
                          if skew[k].get("groups") else 0.0) for k in keys},
        "base_trough": skew["L1_cooccurrence"].get("base_trough", 0.0),
    }


def build_parser() -> argparse.ArgumentParser:
    """Separated from ``main`` so the DEFAULTS are testable without a corpus.

    ``--gate-margin``'s default is what the exit code means, and it is the one
    thing pass 111 got wrong. A default is only a guard if something asserts
    on it.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", action="append", required=True,
                        help="repeat for the UP and the DOWN corpus; both are "
                             "required before any scheme may be believed")
    parser.add_argument("--symbol", default="AERO-USDC")
    parser.add_argument("--chain", default="base")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--window", type=int, default=12,
                        help="bars a change-keyed scheme may look back over")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--margin", type=float, action="append",
                        help="hysteresis margins to sweep, as a fraction of "
                             "each band's own width. Repeat for a sweep. "
                             "0 is plain relative banding and is always "
                             "included so every table has its control.")
    parser.add_argument("--steps", type=int, action="append",
                        help="step counts to sweep for the two change-keyed "
                             "schemes. Defaults to MOTIF_SEQUENCE_STEPS.")
    parser.add_argument("--gate-margin", type=float,
                        default=L1_HYSTERESIS_MARGIN,
                        help="the hysteresis margin the GATE judges, which "
                             "must be the one the live path ships. Margin 0 "
                             "is always measured alongside it as the control.")
    parser.add_argument("--l2-steps", type=int, default=L2_TRANSITION_STEPS,
                        help="changed motifs the transition path carries")
    parser.add_argument("--min-support", type=int, default=20,
                        help="samples a motif group needs before its label "
                             "distribution is read; matches omen_layer_probe")
    parser.add_argument("--json-out", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    keys = ["L1_cooccurrence", "L2_sequence", "L2_transitions", "L2_runlength"]

    # THE CONTROL AND THE TREATMENT, in one process on one corpus each. The
    # control is plain relative banding; the treatment is the banding the live
    # path actually uses. Gating on the control is what produced the pass-111
    # "no order-carrying scheme can meet the guard" block, so both are printed
    # and the GATE reads the shipped one.
    control = [measure(Path(c), args.symbol, args.chain, args.horizon,
                       args.window, args.samples, 0.0, steps=args.l2_steps)
               for c in args.corpus]
    results = [measure(Path(c), args.symbol, args.chain, args.horizon,
                       args.window, args.samples, args.gate_margin,
                       steps=args.l2_steps)
               for c in args.corpus]

    def table(title: str, rows: List[Dict[str, Any]]) -> None:
        print(title)
        for res in rows:
            print("%s  n=%d  L1 changes on %.1f%% of bars  (trough rate %.1f%%)"
                  % (Path(res["corpus"]).name, res["samples"],
                     100 * res["l1_change_rate"], 100 * res["base_trough"]))
            for key in keys:
                value = res["distinctness"].get(key, 0.0)
                verdict = "" if key == "L1_cooccurrence" else (
                    "  PASS" if value <= IDENTIFIER_CEILING else "  FAIL")
                print("   %-16s distinct=%.4f  vocab=%4d  "
                      "groups(n>=%d)=%2d covering %5.1f%%  best lift %.2fx%s"
                      % (key, value, res["vocab"][key], args.min_support,
                         res["supported"][key], 100 * res["covered"][key],
                         res["best_lift"][key], verdict))
            print("")

    table("CONTROL -- plain relative banding (margin 0.00), window=%d, "
          "L2 steps=%d, ceiling=%.2f\n"
          % (args.window, args.l2_steps, IDENTIFIER_CEILING), control)
    table("TREATMENT -- SHIPPED banding: hysteresis margin %.2f, window=%d, "
          "L2 steps=%d, ceiling=%.2f\n"
          % (args.gate_margin, args.window, args.l2_steps,
             IDENTIFIER_CEILING), results)

    def clears(key: str) -> bool:
        return all(r["distinctness"].get(key, 1.0) <= IDENTIFIER_CEILING
                   for r in results)

    winners = [k for k in ("L2_transitions", "L2_runlength") if clears(k)]
    print("VERDICT TABLE -- worst of both corpora, at the SHIPPED banding")
    for key in ("L2_sequence", "L2_transitions", "L2_runlength"):
        worst = max(r["distinctness"].get(key, 1.0) for r in results)
        worst_ctl = max(r["distinctness"].get(key, 1.0) for r in control)
        print("%-16s margin 0.00 %.4f -> margin %.2f %.4f   %s"
              % (key, worst_ctl, args.gate_margin, worst,
                 "CLEARS" if clears(key) else "FAILS"))

    # A LAYER THAT ABSTRACTS PERFECTLY AND PREDICTS NOTHING IS STILL WORTHLESS.
    # Said as loudly as the distinctness verdict, because clearing the ceiling
    # by collapsing toward a constant clears it for the wrong reason.
    for key in winners:
        supported = min(r["supported"][key] for r in results)
        if supported <= 0:
            print("\nWARNING: %s clears the ceiling but has NO group reaching "
                  "n=%d in at least one corpus. It groups samples without "
                  "grouping enough of them to carry a label distribution, so a "
                  "node arm on it would measure nothing."
                  % (key, args.min_support))

    # THE SWEEP. Reported from the same process as the table above, so the
    # control row (margin 0) and every treatment row are one measurement --
    # the reason the pass-111 numbers are trustworthy at all.
    if args.margin or args.steps:
        margins = sorted({0.0, *(args.margin or ())})
        steps_list = sorted(set(args.steps or (MOTIF_SEQUENCE_STEPS,)))
        print("HYSTERESIS x STEPS SWEEP (margin 0 is the control)\n")
        for corpus in args.corpus:
            bars = load_bars(Path(corpus))
            if args.samples:
                bars = bars[: args.samples + LOOKBACK_BARS]
            frame_sets = []
            for index in range(LOOKBACK_BARS, len(bars)):
                try:
                    frame_sets.append(build_collections(
                        bars, index, horizon_bars=args.horizon,
                        symbol=args.symbol, chain=args.chain))
                except (ValueError, IndexError):
                    continue
            bands = relative_bands(frame_sets, streams=L1_STREAMS)
            total = len(frame_sets)
            print("%s  n=%d" % (Path(corpus).name, total))
            for margin in margins:
                motifs = sticky_motifs(frame_sets, bands, margin)
                rate = sum(1 for a, b in zip(motifs, motifs[1:])
                           if a != b) / max(1, total - 1)
                cells = []
                for steps in steps_list:
                    for fn, tag in ((transition_motif, "t"),
                                    (run_length_motif, "r")):
                        frames = [fn(motifs[max(0, i - args.window + 1):i + 1],
                                     steps=steps) for i in range(total)]
                        cells.append("%s%d=%.4f" % (tag, steps,
                                                    len(set(frames)) / total))
                print("   margin=%.2f  L1 change=%.1f%%  L1 distinct=%.4f "
                      "(vocab %d)  %s"
                      % (margin, 100 * rate, len(set(motifs)) / total,
                         len(set(motifs)), " ".join(cells)))
            print("")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"ceiling": IDENTIFIER_CEILING,
                        "gate_margin": args.gate_margin,
                        "l2_steps": args.l2_steps,
                        "control": control, "results": results},
                       indent=2), encoding="utf-8")
        print("\njson -> %s" % args.json_out)

    if not winners:
        print("\nVERDICT: NEITHER scheme clears %.2f in both corpora at the "
              "shipped banding. Do NOT spend a node arm." % IDENTIFIER_CEILING)
        return 1
    print("\nVERDICT: %s clears %.2f in BOTH corpora at hysteresis margin "
          "%.2f. The rejected scheme's own number is in the table above -- "
          "report it rather than leaving it unmeasured."
          % (" and ".join(winners), IDENTIFIER_CEILING, args.gate_margin))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
