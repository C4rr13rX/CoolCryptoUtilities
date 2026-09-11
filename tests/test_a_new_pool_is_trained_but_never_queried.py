"""The failure that made two arms report identical numbers to four decimals.

Pass 120 added a shape-class pool, trained it, and measured a held-out number
byte-identical to the arm WITHOUT the pool -- same exact rate, same omen
counts, same precisions, over 400 test samples. The pool was not inert: it was
never asked. ``build_collections`` returns the new frame and streams it to the
node, but ``OmenBrain.predict`` streams ``PREDICT_COLLECTIONS``, a fixed
three-name tuple, so a new collection reaches training and never reaches the
decode.

That is worth a test because the symptom is an UNCHANGED NUMBER, which reads
as "the idea did not work" rather than as "the wiring is incomplete", and the
relation pools (12/13/14) and metacognition pools (15-19) have exactly the
same exposure.

The second test pins the property the shape key exists for: a key that does
not survive the mutation buys nothing, and the whole arm is pointless without
it.
"""
from __future__ import annotations

import importlib
import random

import pytest


def _reload_brain(monkeypatch, **env):
    import trading.omen_brain as ob
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(ob)


@pytest.fixture(autouse=True)
def _restore():
    yield
    import trading.omen_brain as ob
    importlib.reload(ob)


def test_a_new_collection_reaches_training_but_not_the_query(monkeypatch):
    """The shape pool is streamed for training and absent from the decode."""
    ob = _reload_brain(monkeypatch, OMEN_SHAPE_COLLECTION="1")

    names = {c.name for c in ob.COLLECTIONS}
    assert "shape_class" in names, (
        "OMEN_SHAPE_COLLECTION=1 must put the pool in the training set")

    # THE BUG. The default query set does not contain it, so a run with the
    # pool on and a run with it off stream the SAME bytes at predict time.
    assert "shape_class" not in ob.PREDICT_COLLECTIONS, (
        "if this ever becomes false, the silent-no-op documented in "
        "data/brain_experiments/p120-jet-shape-relation-pool.md is closed and "
        "this test should assert the new default instead of the old one")

    # And the fix needs no code change -- the query set is env-driven.
    ob2 = _reload_brain(
        monkeypatch, OMEN_SHAPE_COLLECTION="1",
        OMEN_PREDICT_COLLECTIONS="temporal,geometry,cross,shape_class")
    assert "shape_class" in ob2.PREDICT_COLLECTIONS


def test_the_shape_key_survives_the_mutation_it_exists_to_bind(monkeypatch):
    """A deep-prefix jitter must not move the shape class.

    If it does, every mutant lands as another near-unique key and the arm is
    the pass-117 result again: more pairs is more memorisation.
    """
    ob = _reload_brain(monkeypatch, OMEN_SHAPE_COLLECTION="1",
                       OMEN_SHAPE_RESOLUTION="k4")
    rng = random.Random(11)

    # A shape with structure at the scale the key resamples onto -- one slow
    # swing across the window, not a fast sawtooth. A sawtooth whose period is
    # shorter than a segment is the degenerate case: every segment mean comes
    # out the same, the min-max span the key normalises against collapses to
    # noise, and the bands then flip on any jitter at all. That is a property
    # of the fixture, not of the encoder, and it is written down here because
    # it cost a test run to find.
    import math as _math
    base = [100.0 + 10.0 * _math.sin(i / 54.0) + i * 0.04 for i in range(169)]
    key = ob.shape_class_frame(base)
    assert key.startswith("shp k4="), key

    # Jitter only the DEEP prefix -- the bars the labeller cannot read.
    step = sum(abs(base[i] - base[i - 1]) for i in range(1, len(base))) / (len(base) - 1)
    held = 0
    trials = 40
    for _ in range(trials):
        moved = list(base)
        for i in range(len(base) - ob.RANGE_WINDOW):
            moved[i] = max(1e-12, base[i] + rng.gauss(0.0, 0.5 * step))
        if ob.shape_class_frame(moved) == key:
            held += 1
    assert held >= trials * 0.75, (
        f"the shape key held on only {held}/{trials} jittered copies; it has "
        "to collide across a base and its mutants or the pool teaches nothing")

    # And it must not be a constant: a genuinely different shape must differ.
    inverted = [200.0 - v for v in base]
    assert ob.shape_class_frame(inverted) != key, (
        "a mirrored chart must not share a shape class, or the key is blunt "
        "enough to be a constant")


def test_the_shape_frame_is_off_by_default(monkeypatch):
    """Production's frames must not change underneath this experiment."""
    import trading.omen_brain as ob
    ob = importlib.reload(ob)
    assert "shape_class" not in {c.name for c in ob.COLLECTIONS}
