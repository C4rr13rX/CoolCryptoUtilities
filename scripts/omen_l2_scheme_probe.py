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
from trading.omen_layers import (  # noqa: E402
    L1_STREAMS, MOTIF_SEQUENCE_STEPS, _band_of, _numeric_of, cooccurrence_motif,
    layer_distinctness, relative_bands, sequence_motif,
)

#: Above this, a layer NAMES samples rather than grouping them. Stated by
#: trading/omen_layers and used unchanged so this probe and the layer probe
#: cannot disagree about what passing means.
IDENTIFIER_CEILING = 0.30

#: Dwell buckets for the run-length scheme. Bucketed rather than raw, because a
#: raw count is an integer that grows without bound and would re-introduce the
#: near-uniqueness the scheme exists to remove. The boundaries are ordinal --
#: "just changed", "a couple of bars", "a stretch", "entrenched".
_DWELL_BUCKETS = ((1, "d1"), (2, "d2"), (4, "d4"), (8, "d8"))


def _compact(motif: str) -> str:
    """A motif's band pattern, which is all L2 ever reads of it.

    Same reduction ``sequence_motif`` applies, lifted out so the three schemes
    are compared over an IDENTICAL alphabet -- a scheme that quietly used a
    finer symbol would win on nothing but its symbol set.
    """
    bands = []
    for token in str(motif).split()[1:]:          # skip the "co1" prefix
        _, _, band = token.partition("=")
        bands.append({"lo": "l", "mid": "m", "hi": "h"}.get(band, "x"))
    return "".join(bands) or "x"


def _dwell_bucket(count: int) -> str:
    label = "dN"
    for bound, name in _DWELL_BUCKETS:
        if count <= bound:
            return name
        label = "dN"
    return label


def transition_motif(motifs: Sequence[str],
                     steps: int = MOTIF_SEQUENCE_STEPS) -> str:
    """L2-T: the last ``steps`` motifs that were DIFFERENT from their predecessor.

    Repeats are dropped, so a regime that holds for forty bars contributes one
    symbol and forty consecutive bars inside it share one frame. That is the
    persistence the old path threw away by sampling once per bar.

    ORDER SURVIVES: the symbols stay oldest-first, so A->B->C and C->B->A are
    different frames. That is the property L2 exists to carry and it is
    asserted directly in the test.
    """
    if not motifs:
        return "co2t path=na"
    compact = [_compact(m) for m in motifs]
    changed: List[str] = []
    for item in compact:
        if not changed or item != changed[-1]:
            changed.append(item)
    return "co2t path=%s" % "|".join(changed[-steps:])


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


def sticky_motifs(frame_sets: Sequence[Mapping[str, str]],
                  bands: Mapping[str, Any], margin: float) -> List[str]:
    """L1 motifs with HYSTERESIS: a slot holds its band until it is pushed out.

    THE MEASUREMENT THAT SENT ME HERE. Neither L2 scheme can work while L1
    changes on 73.1% of bars (DOWN) and 74.0% (UP) -- dropping repeats can only
    remove the ~27% that ARE repeats, so any ordered pair of recent motifs is
    near-unique by construction. The constraint is upstream, so the fix is
    upstream: stop the slot flickering across a tercile boundary.

    ``margin`` is a fraction of the band's own width (hi - lo), so it is in the
    stream's units rather than in absolute score units -- the same knob means
    the same thing on a stream whose scores span 0.01 and one whose scores span
    400. A slot already in ``lo`` stays there until the score climbs past
    ``lo + margin*width``; a slot in ``mid`` needs ``lo - margin*width`` to fall
    into ``lo``. margin=0 reproduces the plain relative banding exactly, which
    is what makes a both-settings comparison on one corpus honest.

    MEASURED, 600 samples per corpus, both computed in one process:

        margin  change rate DOWN/UP   L2_transitions steps=2 DOWN/UP
        0.00    73.1% / 74.0%         0.6017 / 0.4917   FAIL
        0.25    57.9% / 57.4%         0.5083 / 0.4050   FAIL
        0.50    37.6% / 38.2%         0.2633 / 0.2000   PASS
        1.00    18.9% / 17.9%         0.1483 / 0.1017   PASS

    So an ORDER-CARRYING L2 under the 0.30 ceiling in both windows exists, and
    it needed an L1 change rather than another L2 scheme.

    THE COST, stated because distinctness alone cannot see it: L1 itself
    coarsens (0.1983 -> 0.0817 DOWN, vocabulary 119 -> 49; 0.1383 -> 0.0683 UP,
    83 -> 41). Whether that coarser L1 still carries label skew is NOT measured
    here and must be, with ``omen_layer_probe``'s skew test, before any node arm
    is spent on it. A layer that abstracts perfectly and predicts nothing is
    still worthless.
    """
    scores = {name: [_numeric_of(frames.get(name)) for frames in frame_sets]
              for name in L1_STREAMS}
    columns: Dict[str, List[str]] = {}
    for name in L1_STREAMS:
        cuts = bands.get(name)
        if cuts is None:
            # relative_bands OMITS a stream whose terciles collapse, and
            # cooccurrence_motif falls back to absolute sign banding for it.
            # Emitting "na" here instead would make margin=0 a different
            # encoder from the comparison arm -- caught by
            # test_hysteresis_at_zero_margin_reproduces_plain_relative_banding.
            columns[name] = [_band_of(frames.get(name)) for frames in frame_sets]
            continue
        low, high = cuts
        reach = margin * (high - low)
        held: Optional[str] = None
        out: List[str] = []
        for score in scores[name]:
            if score is None:
                out.append("na")
                continue
            if held == "lo":
                band = "lo" if score <= low + reach else (
                    "hi" if score >= high else "mid")
            elif held == "hi":
                band = "hi" if score >= high - reach else (
                    "lo" if score <= low else "mid")
            else:
                band = "lo" if score <= low - reach else (
                    "hi" if score >= high + reach else "mid")
            held = band
            out.append(band)
        columns[name] = out
    return ["co1 " + " ".join("%s=%s" % (name[:3], columns[name][i])
                              for name in L1_STREAMS)
            for i in range(len(frame_sets))]


def build_rows(bars: Sequence[Mapping[str, Any]], symbol: str, chain: str,
               horizon: int, bands: Optional[Mapping[str, Any]],
               window: int) -> List[Dict[str, str]]:
    """One row per bar: its L1 motif and all three L2 schemes.

    ``window`` is how many preceding bars the transition and run-length
    schemes may look back over. A fixed-length PATH cannot use a long window
    (that is the defect), but a change-keyed scheme can, because a long steady
    stretch costs it one symbol -- so the window is the knob that lets
    persistence show up at all.

    A bar whose L0 frames cannot be built records a HOLE rather than being
    skipped, so no scheme claims an adjacency the corpus does not have.
    """
    rows: List[Dict[str, str]] = []
    history: List[str] = []
    for index in range(LOOKBACK_BARS, len(bars)):
        try:
            frames = build_collections(bars, index, horizon_bars=horizon,
                                       symbol=symbol, chain=chain)
        except (ValueError, IndexError):
            history.append("")
            continue
        motif = cooccurrence_motif(frames, bands=bands)
        history.append(motif)
        recent = [m for m in history[-window:] if m]
        fixed = [m for m in history[-MOTIF_SEQUENCE_STEPS:] if m]
        rows.append({
            "L1_cooccurrence": motif,
            "L2_sequence": sequence_motif(fixed),
            "L2_transitions": transition_motif(recent),
            "L2_runlength": run_length_motif(recent),
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
            window: int, samples: int) -> Dict[str, Any]:
    """One corpus, both bandings, all three schemes, in ONE process."""
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

    rows = build_rows(bars, symbol, chain, horizon, bands, window)
    keys = ["L1_cooccurrence", "L2_sequence", "L2_transitions", "L2_runlength"]
    dist = layer_distinctness(rows, keys=keys)

    changed = sum(1 for a, b in zip(rows, rows[1:])
                  if a["L1_cooccurrence"] != b["L1_cooccurrence"])
    return {
        "corpus": str(path), "samples": len(rows),
        "window": window, "steps": MOTIF_SEQUENCE_STEPS,
        "l1_change_rate": changed / max(1, len(rows) - 1),
        "distinctness": dist,
        "vocab": {k: len({r[k] for r in rows}) for k in keys},
    }


def main() -> int:
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
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    results = [measure(Path(c), args.symbol, args.chain, args.horizon,
                       args.window, args.samples) for c in args.corpus]

    print("L2 SCHEME PROBE -- relative banding, window=%d, steps=%d, "
          "ceiling=%.2f\n" % (args.window, MOTIF_SEQUENCE_STEPS,
                              IDENTIFIER_CEILING))
    keys = ["L1_cooccurrence", "L2_sequence", "L2_transitions", "L2_runlength"]
    for res in results:
        print("%s  n=%d  L1 changes on %.1f%% of bars"
              % (Path(res["corpus"]).name, res["samples"],
                 100 * res["l1_change_rate"]))
        for key in keys:
            value = res["distinctness"].get(key, 0.0)
            verdict = "" if key == "L1_cooccurrence" else (
                "  PASS" if value <= IDENTIFIER_CEILING else "  FAIL")
            print("   %-16s distinct=%.4f  vocab=%4d%s"
                  % (key, value, res["vocab"][key], verdict))
        print("")

    def clears(key: str) -> bool:
        return all(r["distinctness"].get(key, 1.0) <= IDENTIFIER_CEILING
                   for r in results)

    winners = [k for k in ("L2_transitions", "L2_runlength") if clears(k)]
    for key in ("L2_sequence", "L2_transitions", "L2_runlength"):
        worst = max(r["distinctness"].get(key, 1.0) for r in results)
        print("%-16s worst-of-both %.4f  %s"
              % (key, worst, "CLEARS" if clears(key) else "FAILS"))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"ceiling": IDENTIFIER_CEILING, "results": results},
                       indent=2), encoding="utf-8")
        print("\njson -> %s" % args.json_out)

    if not winners:
        print("\nVERDICT: NEITHER scheme clears %.2f in both corpora. Do NOT "
              "spend a node arm." % IDENTIFIER_CEILING)
        return 1
    print("\nVERDICT: %s clears %.2f in BOTH corpora. A node arm is now "
          "worth spending -- and only on the winner."
          % (" and ".join(winners), IDENTIFIER_CEILING))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
