"""A motif layer must SHRINK the vocabulary and CARRY the outcome, or be cut.

THE FAILURE THIS PREVENTS. A higher layer that is as distinct as its input has
abstracted nothing -- it is a lossy copy costing a consolidation and a query
per sample. This repo has already paid for the near-unique trap once
(``SEQUENCE_STEPS`` at 8 produced 0.76 distinct frames per sample, an
identifier, which maximises train recall and destroys generalisation), and the
guard that caught it only existed because someone wrote it down as a test.

The second half is the one that is easy to miss. Distinctness alone CANNOT say
whether a layer is worth querying: an abstraction layer is low-distinctness by
design, which puts it under the dilution law's 0.20 query floor automatically.
A layer that compresses its inputs neatly while telling you nothing about the
label would pass a distinctness check and then dilute every query it joined.
``label_skew`` is what separates those two cases, so it needs a test that goes
red when it stops separating them.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.omen_layer_probe import label_skew  # noqa: E402
from trading.omen_layers import (  # noqa: E402
    cooccurrence_motif, layer_distinctness, sequence_motif,
)


def _frames(geo: str, tem: str) -> dict:
    """L0-shaped frames whose bands are readable by ``_band_of``."""
    return {
        "geometry": f"geo z6={geo}12 z24={geo}8",
        "temporal": f"tmp z6={tem}12 z24={tem}8",
        "flow": "flw net=r12",
        "volatility": "vol rngv=r12",
        "cross": f"crs corr={tem}9",
    }


def test_the_motif_layer_shrinks_the_vocabulary_its_input_had():
    """L1 must come in materially below the streams it reads.

    Built so L0 is near-unique per sample and the motif is not: every sample
    gets its own bucket numbers, but only two band patterns exist.
    """
    rows = []
    for i in range(200):
        tem = "u" if i % 2 else "d"
        frames = {
            "geometry": f"geo z6=u{i} z24=u{i}",     # unique per sample
            "temporal": f"tmp z6={tem}{i} z24={tem}{i}",
            "flow": "flw net=r12",
            "volatility": "vol rngv=r12",
            "cross": f"crs corr={tem}9",
        }
        rows.append({**frames, "L1": cooccurrence_motif(frames)})

    scores = layer_distinctness(rows, keys=["geometry", "temporal", "L1"])
    assert scores["geometry"] == 1.0, "the L0 fixture must be near-unique"
    assert scores["L1"] < scores["geometry"] * 0.75, (
        f"L1 at {scores['L1']:.4f} did not shrink a 1.0 input -- it has "
        f"abstracted nothing and must be cut, not trained on")
    assert scores["L1"] <= 0.05, scores


def test_a_longer_motif_path_becomes_an_identifier_and_the_number_says_so():
    """The guard that took SEQUENCE_STEPS from 8 to 5, applied to L2.

    A path over a random motif stream gets MORE distinct as it lengthens.
    This test does not assert a policy -- it asserts the measurement moves in
    the direction that would expose the trap, so shortening the path can be
    justified by a number rather than by taste.
    """
    rng = random.Random(11)
    alphabet = [cooccurrence_motif(_frames("u", t)) for t in ("u", "d", "r")]
    stream = [rng.choice(alphabet) for _ in range(500)]

    short = [{"p": sequence_motif(stream[max(0, i - 3):i + 1], steps=3)}
             for i in range(len(stream))]
    long = [{"p": sequence_motif(stream[max(0, i - 9):i + 1], steps=9)}
            for i in range(len(stream))]

    short_d = layer_distinctness(short, keys=["p"])["p"]
    long_d = layer_distinctness(long, keys=["p"])["p"]
    assert long_d > short_d, (
        f"a 9-step path ({long_d:.3f}) must be more distinct than a 3-step "
        f"one ({short_d:.3f}) or the identifier guard cannot see the trap")


def test_a_layer_that_carries_no_outcome_reads_a_lift_of_one():
    """The dilution case distinctness cannot see.

    Two motifs, each perfectly compressing the input, each with the SAME
    label distribution. Vocabulary shrank; information about the outcome is
    zero. Lift must sit at 1.0, which is what tells a caller to keep the
    layer OUT of the query set.
    """
    rows = []
    for i in range(400):
        motif = "co1 a" if i % 2 else "co1 b"
        # label independent of the motif, by construction
        label = "trough" if i % 5 == 0 else "murk"
        rows.append({"L1": motif, "_label": label})

    skew = label_skew(rows, "L1")
    assert skew["base_trough"] == 0.2, skew["base_trough"]
    for group in skew["groups"]:
        assert abs(group["lift"] - 1.0) < 0.15, (
            f"a motif independent of the label read lift {group['lift']:.2f}; "
            f"at 1.00 it correctly says 'this layer would dilute a query'")


def test_a_layer_that_carries_the_outcome_reads_a_lift_above_one():
    """The counter-case, so the assertion above is not passing vacuously.

    A test that goes green whether or not the code works proves nothing, and
    this repo has shipped one. Here the motif DETERMINES the label, so lift
    must be far from 1.0 in both directions.
    """
    rows = []
    # 200 samples that are 80% trough, 200 that are 10% trough. Base rate is
    # 45%, so the two groups must read ~1.78x and ~0.22x. Written as explicit
    # counts because a modulo fixture put the base rate at 75% and squeezed
    # every lift toward 1.0 -- the fixture, not the code, was the bug.
    rows += [{"L1": "co1 buyable", "_label": "trough"} for _ in range(160)]
    rows += [{"L1": "co1 buyable", "_label": "murk"} for _ in range(40)]
    rows += [{"L1": "co1 flat", "_label": "trough"} for _ in range(20)]
    rows += [{"L1": "co1 flat", "_label": "murk"} for _ in range(180)]

    skew = label_skew(rows, "L1")
    assert abs(skew["base_trough"] - 0.45) < 1e-9, skew["base_trough"]
    lifts = sorted(g["lift"] for g in skew["groups"])
    assert lifts[-1] > 1.5, f"a determining motif read only {lifts[-1]:.2f}x"
    assert lifts[0] < 0.75, f"its complement read {lifts[0]:.2f}x"


def test_the_sell_high_half_is_scorable_and_is_not_the_buy_low_half():
    """The half a long-only lane scores NOWHERE.

    ``omen_experiment.py`` opens a position only on a buy-low omen, so a crest
    is an abstention and its accuracy is never measured. Scoring it needs
    ``label_skew`` to accept a target other than trough -- and needs the two
    targets to give genuinely different answers, or the parameter is
    decorative and the sell-high number is secretly the buy-low one.
    """
    rows = []
    rows += [{"L1": "co1 toppy", "_label": "crest"} for _ in range(80)]
    rows += [{"L1": "co1 toppy", "_label": "murk"} for _ in range(20)]
    rows += [{"L1": "co1 bottomy", "_label": "trough"} for _ in range(80)]
    rows += [{"L1": "co1 bottomy", "_label": "murk"} for _ in range(20)]

    crest = {g["frame"]: g["lift"]
             for g in label_skew(rows, "L1", target="crest")["groups"]}
    trough = {g["frame"]: g["lift"]
              for g in label_skew(rows, "L1", target="trough")["groups"]}

    assert crest["co1 toppy"] > 1.5 and crest["co1 bottomy"] < 0.5, crest
    assert trough["co1 bottomy"] > 1.5 and trough["co1 toppy"] < 0.5, trough
    assert crest["co1 toppy"] != trough["co1 toppy"], (
        "the two targets returned the same lift -- the sell-high half is "
        "reporting the buy-low half under a different name")


def test_support_below_the_floor_is_not_reported_as_a_signal():
    """A 2-sample motif at 5x lift is noise wearing a number's clothes."""
    rows = [{"L1": "co1 common", "_label": "murk"} for _ in range(200)]
    rows += [{"L1": "co1 rare", "_label": "trough"} for _ in range(3)]

    skew = label_skew(rows, "L1", min_support=20)
    frames = {g["frame"] for g in skew["groups"]}
    assert "co1 rare" not in frames, (
        "a 3-sample group was reported as a buy-low signal; min_support "
        "exists precisely to stop that becoming a strategy")
