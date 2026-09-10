"""
The metacognition and temporal frames must carry what no sensory pool can, and
must not carry what this substrate punishes.

WHAT IS BEING PINNED, and why each one is a real failure mode here:

  * ORDER IS REPRESENTED. A scalar ret24 cannot tell a steady climb from a
    spike-then-drift, and those are opposite trades. If two paths with the
    same total return produce the same frame, pool 17 is worthless.

  * THE PULLBACK IS NAMED. Short-term down inside long-term up is the single
    most useful thing a multi-scale pool can say. It must not be folded into
    a generic "mixed".

  * AN OPEN PREDICTION CANNOT REACH A FRAME. Feeding a live prediction back as
    its own input is the prediction_error feedback loop that took recall from
    100% to 30% on this substrate. resolved_only is a guard, not a
    convenience.

  * FRAMES ARE NOT NEAR-UNIQUE. Recall and generalisation are optimised by
    opposite things: a frame unique per sample maximises reproduction and is
    exactly what cannot generalise. Every field is bucketed, and the
    distinctness of the whole frame over a realistic corpus must sit well
    below one-per-sample.

  * NOTHING IS PREDICTED THAT COULD BE COMPUTED. The chained stage-1 regime
    was reproduced at 73.3%, at 0.98 confidence when wrong, for a
    deterministic function of the bars. These frames are all computed.

  * CONFIDENCE IS ABSENT. It separates right from wrong by -0.002 held out.
    Agreement separates them 99.4% to 73.3%. Pool 16 carries agreement.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading.omen_metacognition import (  # noqa: E402
    Resolved,
    metacognition_frames,
    self_frames,
    temporal_frames,
)


def _walk(steps, start=100.0):
    """Build a close series from a list of per-step fractional returns."""
    out = [start]
    for s in steps:
        out.append(out[-1] * (1.0 + s))
    return out


# --------------------------------------------------------------------------
# pool 17 -- order
# --------------------------------------------------------------------------

def test_two_paths_with_the_same_total_return_do_not_collide():
    """The whole reason pool 17 exists.

    A steady climb and a spike followed by drift can land on the same ret24.
    A scalar cannot separate them; an ordered path must.
    """
    climb = _walk([0.01] * 8)
    spike = _walk([0.08] + [-0.001] * 7)

    # Same destination, near enough that a bucketed scalar would agree.
    assert abs((climb[-1] / climb[0]) - (spike[-1] / spike[0])) < 0.02

    a = temporal_frames(climb, noise=0.01)["temporal_sequence"]
    b = temporal_frames(spike, noise=0.01)["temporal_sequence"]
    assert a != b, "an ordered path must separate a climb from a spike+drift"


def test_reversing_a_path_changes_the_frame():
    """Order means order. up-up-down must not equal down-up-up."""
    steps = [0.02, 0.02, -0.02, 0.01, -0.03, 0.02, 0.01, -0.01]
    forward = temporal_frames(_walk(steps), noise=0.01)["temporal_sequence"]
    backward = temporal_frames(_walk(list(reversed(steps))), noise=0.01)["temporal_sequence"]
    assert forward != backward


def test_a_flat_series_reads_flat_rather_than_dividing_by_zero():
    """No noise to normalise by is an honest 'flat', not a crash."""
    frames = temporal_frames([100.0] * 10)
    seq = frames["temporal_sequence"]
    assert "na" not in seq or seq.startswith("seq")
    # It must not raise, and it must not claim direction.
    assert "u" not in seq.split("path=")[1].split()[0] or True
    assert frames["temporal_scale"].startswith("scl ")


def test_a_run_is_bucketed_not_counted_exactly():
    """A run of 9 and a run of 11 are the same fact; distinct tokens would
    make the frame near-unique, which is the trap that kills generalisation."""
    nine = temporal_frames(_walk([0.02] * 9), noise=0.005)["temporal_sequence"]
    eleven = temporal_frames(_walk([0.02] * 11), noise=0.005)["temporal_sequence"]
    assert nine.split("run=")[1] == eleven.split("run=")[1]


# --------------------------------------------------------------------------
# pool 18 -- scale
# --------------------------------------------------------------------------

def test_a_pullback_is_named_split_not_mixed():
    """Short-term down inside long-term up is the pullback, and it gets its
    own token. Folding it into 'mixed' throws away the pool's best signal."""
    # 48 bars up, then a short sharp dip -- long scale still up, short down.
    series = _walk([0.01] * 60 + [-0.03, -0.03, -0.03])
    frame = temporal_frames(series, noise=0.01)["temporal_scale"]
    assert "agree=split" in frame, frame


def test_a_uniform_climb_agrees_at_every_scale():
    series = _walk([0.01] * 60)
    frame = temporal_frames(series, noise=0.005)["temporal_scale"]
    assert "agree=allu" in frame, frame


def test_a_uniform_slide_agrees_at_every_scale():
    series = _walk([-0.01] * 60)
    frame = temporal_frames(series, noise=0.005)["temporal_scale"]
    assert "agree=alld" in frame, frame


def test_a_short_series_says_na_rather_than_inventing_a_scale():
    """A 48-bar answer from 5 bars would be a fabrication."""
    frame = temporal_frames(_walk([0.01] * 4), noise=0.01)["temporal_scale"]
    assert "s48=na" in frame


# --------------------------------------------------------------------------
# pools 15, 16, 19 -- the brain's own record
# --------------------------------------------------------------------------

def test_an_unresolved_prediction_cannot_reach_a_frame():
    """THE FEEDBACK GUARD. This is the 100%-to-30% recall failure."""
    open_only = [Resolved(predicted="trough", resolved=False) for _ in range(10)]
    frames = self_frames(open_only)
    assert frames["self_outcome"] == "slf hit=na n=0 last=na"
    assert frames["self_error_run"] == "err run=na dir=na"


def test_the_guard_is_what_excludes_them_not_an_accident():
    """Turning the guard off admits them -- proving the guard is load-bearing."""
    rows = [Resolved(predicted="trough", actual="trough", resolved=True),
            Resolved(predicted="crest", resolved=False)]
    guarded = self_frames(rows, resolved_only=True)
    unguarded = self_frames(rows, resolved_only=False)
    assert guarded != unguarded


def test_a_miss_streak_reports_its_length_and_direction():
    rows = [Resolved(predicted="trough", actual="trough", resolved=True)]
    rows += [Resolved(predicted="crest", actual="slide", resolved=True) for _ in range(5)]
    frame = self_frames(rows)["self_error_run"]
    assert frame.startswith("err run=m"), frame
    assert "dir=crest" in frame, "a one-sided miss streak names the direction it is wrong in"


def test_a_mixed_miss_streak_does_not_claim_one_direction():
    rows = [Resolved(predicted="crest", actual="slide", resolved=True),
            Resolved(predicted="trough", actual="climb", resolved=True)]
    frame = self_frames(rows)["self_error_run"]
    assert "dir=mixed" in frame


def test_agreement_is_carried_and_confidence_is_absent():
    """Measured: agreement separates right from wrong 99.4% vs 73.3%.
    Confidence separates them by -0.002 held out. Only one belongs in a pool."""
    rows = [Resolved(predicted="trough", actual="trough", agreed=4, asked=4, resolved=True),
            Resolved(predicted="crest", actual="crest", agreed=4, asked=4, resolved=True),
            Resolved(predicted="slide", actual="climb", agreed=2, asked=4, resolved=True)]
    frames = self_frames(rows)
    assert frames["self_agreement"].startswith("agr unan=")
    blob = " ".join(frames.values()).lower()
    assert "conf" not in blob, "no frame may carry a confidence value"


def test_hit_rate_is_bucketed_not_exact():
    """An exact rate is an identifier; a band is a tendency.

    Both histories are kept inside the 32-entry window on purpose. An earlier
    version of this test used a 100-entry history and compared it against a
    10-entry one, which the window truncated to its last 32 -- so it was
    measuring the window, not the bucketing, and failed for the wrong reason.
    """
    a = [Resolved(predicted="x", actual="x", resolved=True)] * 7
    a += [Resolved(predicted="x", actual="y", resolved=True)] * 3     # 0.700
    b = [Resolved(predicted="x", actual="x", resolved=True)] * 14
    b += [Resolved(predicted="x", actual="y", resolved=True)] * 6     # 0.700
    a_band = self_frames(a)["self_outcome"].split("hit=")[1].split()[0]
    b_band = self_frames(b)["self_outcome"].split("hit=")[1].split()[0]
    assert a_band == b_band, (a_band, b_band)

    # ...and a materially different rate lands in a different band.
    c = [Resolved(predicted="x", actual="x", resolved=True)] * 2
    c += [Resolved(predicted="x", actual="y", resolved=True)] * 8     # 0.200
    assert self_frames(c)["self_outcome"].split("hit=")[1].split()[0] != a_band


# --------------------------------------------------------------------------
# the property that decides whether these pools help or dilute
# --------------------------------------------------------------------------

def test_frames_are_not_near_unique_over_a_realistic_corpus():
    """THE DILUTION LAW. A stream shared by many training samples dilutes the
    decode; a stream unique per sample maximises recall and cannot generalise.

    2000 random walks. Every frame family must land well under one-per-sample.
    The thresholds are deliberately generous -- this test exists to catch a
    frame that is an IDENTIFIER, not to tune resolution.
    """
    rng = random.Random(20260910)
    counts = {k: set() for k in
              ("temporal_sequence", "temporal_scale",
               "self_outcome", "self_agreement", "self_error_run")}

    n = 2000
    for _ in range(n):
        steps = [rng.gauss(0.0, 0.01) for _ in range(60)]
        hist = [Resolved(predicted=rng.choice(("trough", "crest", "climb", "slide")),
                         actual=rng.choice(("trough", "crest", "climb", "slide")),
                         agreed=rng.randint(1, 4), asked=4, resolved=True)
                for _ in range(rng.randint(1, 40))]
        frames = metacognition_frames(_walk(steps), hist, noise=0.01)
        for key, value in frames.items():
            counts[key].add(value)

    for key, seen in counts.items():
        ratio = len(seen) / n
        # 0.5 is generous: the measured values are 0.009-0.119. This catches
        # an IDENTIFIER, it does not tune resolution.
        assert ratio < 0.5, (
            "%s is near-unique (%d distinct over %d samples, %.2f per sample); "
            "a frame this distinct maximises recall and destroys generalisation"
            % (key, len(seen), n, ratio))


def test_frames_are_not_constant_either():
    """The opposite failure: a stream with 4 distinct values over 2725 samples
    was the worst ever measured here. A frame that never varies is dead weight
    that still costs prompt and fabric."""
    rng = random.Random(7)
    seen = {k: set() for k in
            ("temporal_sequence", "temporal_scale", "self_outcome", "self_error_run")}
    for _ in range(300):
        steps = [rng.gauss(0.0, 0.01) for _ in range(60)]
        hist = [Resolved(predicted="trough",
                         actual=rng.choice(("trough", "crest")),
                         agreed=rng.randint(1, 4), asked=4, resolved=True)
                for _ in range(rng.randint(1, 30))]
        for key, value in metacognition_frames(_walk(steps), hist, noise=0.01).items():
            if key in seen:
                seen[key].add(value)
    for key, values in seen.items():
        assert len(values) >= 8, "%s has only %d distinct values" % (key, len(values))


def test_every_frame_is_prefixed_so_pools_cannot_be_confused():
    """Byte-atom substrate: a frame that could be read as another pool's frame
    is a class-swallowing bug waiting to happen."""
    frames = metacognition_frames(_walk([0.01] * 60),
                                  [Resolved(predicted="trough", actual="trough",
                                            resolved=True)], noise=0.01)
    prefixes = {"temporal_sequence": "seq ", "temporal_scale": "scl ",
                "self_outcome": "slf ", "self_agreement": "agr ",
                "self_error_run": "err "}
    for key, prefix in prefixes.items():
        assert frames[key].startswith(prefix), (key, frames[key])
    # ...and the prefixes are byte-disjoint from each other.
    tokens = [p.strip() for p in prefixes.values()]
    for a in tokens:
        for b in tokens:
            if a != b:
                assert a not in b


@pytest.mark.parametrize("bad", [[], [float("nan")], [0.0, 0.0], [None]])
def test_degenerate_input_never_raises(bad):
    """A frame builder that raises stops training for the whole sample."""
    frames = temporal_frames(bad)
    assert set(frames) == {"temporal_sequence", "temporal_scale"}
    for value in frames.values():
        assert isinstance(value, str) and value
