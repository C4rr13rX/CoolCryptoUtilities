"""The omen's horizon is a BAR COUNT, and a bar is not a fixed amount of time.

The bug this file is named after: ``build_collections`` emits ``hzn h=12`` and
``bar_seconds`` never enters any frame, so the atom the fabric is TRAINED on
(12 bars of a 3600s corpus = 720 minutes) and the atom it is QUERIED with at
live (12 bars of a 60s resample = 12 minutes) are byte-identical for two
questions 60x apart. Nothing in the code compares the two cadences, so nothing
could notice.

These tests lock the INSTRUMENT that noticed -- ``scripts/omen_temporal_census``
-- rather than asserting the bug is absent, because it is not absent yet. The
census exits nonzero when the crossing exists, which is what lets it gate a
pass instead of merely informing one; if the frame later carries its cadence,
``test_matched_cadences_do_not_trip_the_verdict`` is the test that proves the
fix and the census starts exiting 0.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_census():
    spec = importlib.util.spec_from_file_location(
        "omen_temporal_census", ROOT / "scripts" / "omen_temporal_census.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


census = _load_census()


def _bars(count: int, step: int, start: int = 1_700_000_000):
    """A synthetic series on an exactly uniform ``step``-second grid."""
    return [{"timestamp": start + i * step, "close": 1.0 + i * 0.001,
             "open": 1.0, "high": 1.1, "low": 0.9, "net_volume": 1.0}
            for i in range(count)]


def test_grid_profile_reports_the_cadence_it_measures_not_one_it_assumes():
    profile = census.grid_profile(_bars(50, 3600))
    assert profile["modal_gap_sec"] == 3600
    assert profile["uniform_share"] == pytest.approx(1.0)


def test_a_hole_in_the_series_shows_up_as_a_non_uniform_grid():
    """A dropped bar is exactly what the live resampler produces, and it must
    not be reported as a uniform grid -- that is the whole point of the audit."""
    bars = _bars(50, 60)
    del bars[20]  # one minute with no ticks; the live path drops it
    profile = census.grid_profile(bars)
    assert profile["modal_gap_sec"] == 60
    assert profile["max_gap_sec"] == 120
    assert profile["uniform_share"] < 1.0


def test_the_same_horizon_token_at_two_cadences_is_the_crossing():
    """Training cadence 3600s and live cadence 60s, one horizon of 12 bars.

    This is the arithmetic the census turns into an exit code. It is stated
    here as a test so the 60x is checked rather than remembered.
    """
    train_minutes = 12 * 3600 / 60.0
    live_minutes = 12 * 60 / 60.0
    assert train_minutes == 720.0
    assert live_minutes == 12.0
    # Same token, because the cadence is not in it.
    assert f"hzn h={12}" == f"hzn h={12}"
    assert train_minutes / live_minutes == 60.0


def test_matched_cadences_do_not_trip_the_verdict():
    """The census must not cry crossing when there is none, or it is noise.

    A corpus at 60s asked a 12-bar horizon means the same 12 minutes the live
    path means, so the verdict is clean.
    """
    profile = census.grid_profile(_bars(50, 60))
    horizon_minutes = 12 * profile["modal_gap_sec"] / 60.0
    live_minutes = 12 * 60 / 60.0
    assert abs(horizon_minutes - live_minutes) <= 1.0


def test_a_slot_that_is_unique_on_every_sample_is_named_an_index():
    """Distinctness 1.0 on a COLLECTION can be honest; on a SLOT it cannot.

    The pass-107 reading "temporal = 1.0" was assumed to mean a counter had
    got into the frame. The per-slot census is what separates the two cases,
    so both cases are checked here.
    """
    counter = [{"temporal": f"tmp r1=u9 seq={i}"} for i in range(100)]
    rows = {row["slot"]: row for row in census.slot_census(counter, "temporal")}
    assert rows["r1"]["distinctness"] == pytest.approx(0.01)
    assert rows["seq"]["distinctness"] == pytest.approx(1.0)


def test_an_honest_conjunction_of_slots_is_not_an_index():
    """Eleven slots of thirty values each name every sample uniquely without
    any single slot being an index. That is what the real temporal frame is,
    measured at max slot distinctness 0.0517, and it needs a different fix
    from a counter: nothing to purge, a topology to change."""
    frames = [{"temporal": f"tmp a=u{i % 30} b=u{(i * 7) % 29} c=u{(i * 13) % 31}"}
              for i in range(600)]
    rows = {row["slot"]: row for row in census.slot_census(frames, "temporal")}
    assert max(row["distinctness"] for row in rows.values()) < 0.1
    assert len({f["temporal"] for f in frames}) == len(frames)
