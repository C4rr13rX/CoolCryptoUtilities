"""The shape descriptor must express SHAPE, not the instant it was read at.

The failure this prevents: a "same shape" relation stream that is secretly
keyed to price LEVEL or price AMPLITUDE. Such a stream looks like a relation,
ships as a relation, and teaches the fabric nothing but the instant -- which is
exactly the memorisation that inverted the pass-117 mutation arm. The three
invariances below are the descriptor's whole reason to exist, so they are
asserted rather than assumed.

The opposite failure is asserted too: a descriptor that collapses everything to
one token would pass every invariance test and be worthless. ``test_...
_distinguishes_genuinely_different_shapes`` is what makes the invariance tests
mean something.
"""
from __future__ import annotations

import math

import pytest

from scripts.omen_shape_relation import (
    BAND_SCHEMES, labelled_anchors, shape_descriptor,
)
from trading.omen_brain import LOOKBACK_BARS


def _ramp(n: int, lo: float, hi: float):
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def _hump(n: int, base: float, height: float):
    return [base + height * math.sin(math.pi * i / (n - 1)) for i in range(n)]


def test_descriptor_ignores_price_level():
    """The same path ten times higher is the same shape."""
    path = _hump(96, 1.0, 0.4)
    lifted = [p * 10.0 for p in path]
    assert shape_descriptor(path) == shape_descriptor(lifted)


def test_descriptor_ignores_amplitude():
    """The same path at a quarter of the amplitude is the same shape.

    z-normalisation divides by the path's own standard deviation, so a
    compressed copy bands identically. A descriptor that failed this would be
    reading volatility, not form.
    """
    path = _hump(96, 1.0, 0.4)
    anchor = path[0]
    squashed = [anchor + (p - anchor) * 0.25 for p in path]
    assert shape_descriptor(path) == shape_descriptor(squashed)


def test_descriptor_ignores_tempo():
    """The same shape drawn over twice as many bars is the same shape."""
    slow = _hump(192, 1.0, 0.4)
    fast = _hump(96, 1.0, 0.4)
    assert shape_descriptor(slow) == shape_descriptor(fast)


def test_descriptor_distinguishes_genuinely_different_shapes():
    """Without this, every invariance above is satisfied by a constant."""
    up = _ramp(96, 1.0, 2.0)
    down = _ramp(96, 2.0, 1.0)
    hump = _hump(96, 1.0, 0.4)
    assert len({shape_descriptor(up), shape_descriptor(down),
                shape_descriptor(hump)}) == 3


def test_a_flat_path_is_the_middle_band_and_does_not_raise():
    """A flat run has zero deviation. Dropping it would bias any census
    toward volatile anchors, so it gets the middle band instead."""
    flat = [1.25] * 96
    letters, _cuts = BAND_SCHEMES[5]
    middle = letters[len(letters) // 2]
    assert shape_descriptor(flat, segments=8) == f"shape8:{middle * 8}"


def test_every_band_scheme_uses_byte_disjoint_letters():
    """Atoms are bytes. If one band's token contained another's, the frequent
    band would swallow the rare one -- the trap that has already cost a pass.
    """
    for nbands, (letters, cuts) in BAND_SCHEMES.items():
        assert len(letters) == nbands
        assert len(set(letters)) == nbands
        assert len(cuts) == nbands - 1
        assert list(cuts) == sorted(cuts)


def test_the_descriptor_path_stops_at_the_anchor():
    """The descriptor must not be able to see its own label.

    ``labelled_anchors`` hands out the lookback path; if that path ever
    included a bar at or past the anchor's future, the whole measurement would
    be reading the answer.
    """
    bars = [{"timestamp": 1_700_000_000 + i * 3600,
             "open": 1.0 + 0.001 * i, "high": 1.02 + 0.001 * i,
             "low": 0.98 + 0.001 * i, "close": 1.0 + 0.001 * i,
             "volume": 100.0}
            for i in range(LOOKBACK_BARS + 60)]
    start = LOOKBACK_BARS
    rows = labelled_anchors(bars, start, start + 10, horizon=12)
    assert rows, "fixture produced no labelable anchors"
    for row in rows:
        assert len(row["path"]) == LOOKBACK_BARS + 1
        assert row["path"][-1] == pytest.approx(
            float(bars[row["index"]]["close"]))
