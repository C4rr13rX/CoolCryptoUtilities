"""The sample builder must FEED the self pools, not sentinel them.

THE BUG THIS PREVENTS, measured pass 110 on a real 19-pool node. The query
path probe reported ``QUERY PATH DEAD`` for a query set differing only by
pools 15/16/19 -- control 0/60, treatment 0/60 -- while the B arm demonstrably
fired six streams per prediction. The pools were sent and they were read. They
moved nothing because ``scripts/omen_experiment.build_samples`` called
``build_collections`` without ``history=``, so every self frame in every
training set was the ``na`` sentinel: the three pools trained as CONSTANTS.

A constant stream cannot move a query however good the pool is, and nothing in
the system said so -- the node returned 200, the streams fired, and the
verdict read DEAD with no error anywhere. That is the failure mode this file
exists to make loud.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bars(n: int = 900):
    """A deterministic wave with enough shape to move every self frame."""
    import math

    out = []
    price = 100.0
    for i in range(n):
        price *= 1.0 + 0.004 * math.sin(i / 7.0) + 0.001 * math.sin(i / 53.0)
        high = price * 1.003
        low = price * 0.997
        out.append({
            "timestamp": 1_700_000_000 + i * 3600,
            "open": price * 0.999, "high": high, "low": low,
            "close": price, "volume": 1000.0 + (i % 17) * 10,
        })
    return out


@pytest.fixture()
def meta_on(monkeypatch):
    """Reload omen_brain with the meta collections gated ON.

    The flag is read at import time, which is deliberate -- it must be one
    decision per process, not per call -- so exercising it means reloading.
    """
    import importlib

    monkeypatch.setenv("OMEN_META_COLLECTIONS", "1")
    import trading.omen_brain as omen_brain
    importlib.reload(omen_brain)
    import scripts.omen_experiment as experiment
    importlib.reload(experiment)
    yield experiment
    monkeypatch.delenv("OMEN_META_COLLECTIONS", raising=False)
    importlib.reload(omen_brain)
    importlib.reload(experiment)


SELF_NAMES = ("self_outcome", "self_agreement", "self_error_run")


def test_the_self_frames_are_not_one_constant_across_a_training_set(meta_on):
    """The regression itself: >1 distinct value for the outcome/error pools.

    Fails against the old loop, which emitted exactly one value -- the
    ``na`` sentinel -- for all three, on any corpus of any length.
    """
    bars = _bars()
    samples = meta_on.build_samples(bars, "TEST", "base", 12, 0, len(bars) - 12)
    assert len(samples) > 100, f"need a real training set, got {len(samples)}"

    distinct = {name: {s["frames"][name] for s in samples} for name in SELF_NAMES}
    constant = [name for name in ("self_outcome", "self_error_run")
                if len(distinct[name]) < 2]
    assert not constant, (
        f"{constant} carry ONE value across {len(samples)} samples -- the pools "
        f"train as constants and cannot move a query. "
        f"counts: { {k: len(v) for k, v in distinct.items()} }")


def test_an_unfed_build_emits_the_sentinel_so_the_test_above_can_fail(meta_on):
    """The negative control: without ``history=`` the frames ARE constant.

    Without this, the test above could pass for a reason unrelated to the
    fix -- it proves the sentinel path still exists and still collapses, so
    a green result upstairs means the history was actually threaded through.
    """
    from trading.omen_brain import build_collections

    bars = _bars()
    frames = [build_collections(bars, i, horizon_bars=12,
                                symbol="TEST", chain="base")
              for i in range(300, 500)]
    for name in SELF_NAMES:
        values = {f[name] for f in frames}
        assert len(values) == 1, (
            f"{name} varies without a history -- the sentinel path changed, "
            f"so the control no longer controls: {sorted(values)[:3]}")
        assert "na" in values.pop(), "the sentinel should read na"


def test_the_builder_never_hands_an_unresolved_row_to_a_frame(meta_on):
    """The prediction_error guard, checked at the seam rather than in theory.

    Feeding a live prediction back as its own input took recall from 100% to
    30% here. ``ResolvedHistory.as_of`` is what stops it, and this asserts the
    builder actually goes through it: every row a frame is built from must be
    settled, and settled no later than the bar being built.
    """
    from trading.omen_resolved_history import ResolvedHistory

    seen = []
    real_as_of = ResolvedHistory.as_of

    def spy(self, bar_index):
        rows = real_as_of(self, bar_index)
        seen.append((bar_index, rows))
        return rows

    ResolvedHistory.as_of = spy
    try:
        bars = _bars(500)
        meta_on.build_samples(bars, "TEST", "base", 12, 0, len(bars) - 12)
    finally:
        ResolvedHistory.as_of = real_as_of

    assert seen, "build_samples never consulted the history at all"
    for bar_index, rows in seen:
        for row in rows:
            assert row.resolved and row.actual, (
                f"bar {bar_index} was handed an UNRESOLVED row -- that is the "
                f"prediction_error feedback loop, not a history: {row}")

    # And it must actually accumulate: a history that is always empty would
    # satisfy the loop above while feeding the pools the same sentinel.
    assert max(len(rows) for _, rows in seen) > 1, (
        "the history never held more than one settled row, so the self pools "
        "saw nothing to be wrong about")


def test_history_is_a_no_op_when_the_meta_flag_is_off():
    """A non-meta run must be byte-identical to before the wiring landed.

    The flag defaults OFF because sending pool 15 to a v2/v3 node does not
    degrade training, it SILENTLY STOPS it. So the cost of the wiring in the
    default configuration has to be exactly zero, and that is checkable.
    """
    import importlib

    os.environ.pop("OMEN_META_COLLECTIONS", None)
    import trading.omen_brain as omen_brain
    importlib.reload(omen_brain)
    import scripts.omen_experiment as experiment
    importlib.reload(experiment)

    assert not omen_brain.META_ENABLED, "this test needs the flag OFF"
    bars = _bars(400)
    samples = experiment.build_samples(bars, "TEST", "base", 12, 0, len(bars) - 12)
    assert samples, "no samples built"
    streamed = {c.name for c in omen_brain.COLLECTIONS}
    for sample in samples:
        assert set(sample["frames"]) == streamed, (
            "the frame set diverged from what the node is sent")
        for name in SELF_NAMES:
            assert name not in sample["frames"], (
                f"{name} leaked into a non-meta build -- it would be an "
                f"unknown input pool id on a v2/v3 node, and the whole sample "
                f"would be reported as a MISS")
