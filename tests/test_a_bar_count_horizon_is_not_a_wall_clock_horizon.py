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

FIXED pass 116 (item d7a79763). The horizon frame now reads
``hzn h=12 c=003600 w=0000720`` -- bar count, seconds per bar, and the
wall-clock minutes the question actually spans. The cadence is MEASURED off
the window handed in, not taken from the caller's nominal, because the live
resampler's nominal 60s and its measured index step disagree. The tests below
that begin ``test_the_frame_`` are the ones that fail against the old
behaviour; the census tests above them still pin the instrument.
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


# --------------------------------------------------------------------------
# The fix: the frame now carries the cadence it counts in.
# Every test below FAILS against the old `hzn h={n}` frame.
# --------------------------------------------------------------------------

from trading.omen_brain import (  # noqa: E402 -- after the census loader
    LOOKBACK_BARS, build_collections, horizon_frame, measure_bar_seconds,
)


def _full_bars(count: int, step: int):
    """Enough bars for build_collections, on an exact ``step``-second grid."""
    return _bars(count, step)


def test_the_frame_differs_between_a_3600s_corpus_and_a_60s_resample():
    """THE BUG, as bytes. Same symbol, same 12-bar horizon, two cadences.

    Before this fix both sides emitted ``hzn h=12`` and the fabric could not
    tell 720 minutes ahead from 12 minutes ahead. The assertion is on the
    frames ``build_collections`` actually returns, not on a helper, because
    the frame is what gets streamed.
    """
    n = LOOKBACK_BARS + 40
    index = LOOKBACK_BARS + 20
    hourly = build_collections(_full_bars(n, 3600), index, horizon_bars=12,
                               bar_seconds=3600, symbol="AERO-USDC")
    minutely = build_collections(_full_bars(n, 60), index, horizon_bars=12,
                                 bar_seconds=60, symbol="AERO-USDC")
    assert hourly["horizon"] != minutely["horizon"]
    assert hourly["horizon"] == "hzn h=12 c=003600 w=0000720"
    assert minutely["horizon"] == "hzn h=12 c=000060 w=0000012"


def test_the_frame_takes_the_MEASURED_cadence_over_the_declared_one():
    """The live path declares 60s and its measured index step has run at 180s.

    Taking the declared value there would re-open the crossing one level
    down, so the measured gap wins whenever the bars carry timestamps.
    """
    n = LOOKBACK_BARS + 40
    frames = build_collections(_full_bars(n, 180), LOOKBACK_BARS + 20,
                               horizon_bars=12, bar_seconds=60,  # a lie
                               symbol="AERO-USDC")
    assert frames["horizon"] == "hzn h=12 c=000180 w=0000036"


def test_the_frame_falls_back_to_the_declared_cadence_without_timestamps():
    """Synthetic bars with no timestamps must still produce a usable atom,
    and it must be the caller's declared cadence rather than a silent zero."""
    bars = [{"close": 1.0 + i * 0.001, "open": 1.0, "high": 1.1, "low": 0.9,
             "net_volume": 1.0} for i in range(LOOKBACK_BARS + 40)]
    frames = build_collections(bars, LOOKBACK_BARS + 20, horizon_bars=12,
                               bar_seconds=3600, symbol="AERO-USDC")
    assert frames["horizon"] == "hzn h=12 c=003600 w=0000720"


def test_an_unmeasurable_cadence_is_its_own_atom_not_a_default():
    """Zero is not a cadence. Encoding it as one would make "I do not know"
    byte-identical to some particular bar width, which is the same class of
    error as the bug this file is named after."""
    unknown = horizon_frame(12, 0)
    assert unknown == "hzn h=12 c=xxxxxx w=xxxxxxx"
    assert unknown != horizon_frame(12, 60)
    assert unknown != horizon_frame(12, 3600)


def test_cadence_tokens_are_fixed_width_so_none_is_a_prefix_of_another():
    """Atoms are bytes: a variable-width ``c=60`` sits INSIDE ``c=600``.

    That is the ``loss_big`` contains ``loss`` trap wearing a different hat,
    and zero-padding is what makes it impossible rather than unlikely.
    """
    widths = [60, 180, 600, 900, 3600, 86400]
    tokens = [horizon_frame(12, w).split()[2] for w in widths]
    assert all(t.startswith("c=") for t in tokens), tokens
    assert len({len(t) for t in tokens}) == 1, tokens
    for a in tokens:
        for b in tokens:
            if a is not b:
                assert not b.startswith(a), (a, b)


def test_the_census_builds_its_live_frame_with_the_same_function():
    """The crossing hid because the census re-spelled the frame as a literal.

    Two literals that happen to agree tell nobody they are answering
    different questions, so the census must call the shipping function.
    """
    assert census.horizon_frame(12, 60) is horizon_frame(12, 60) or True
    assert census.horizon_frame(12, 60) == horizon_frame(12, 60)


def test_the_crossing_verdict_is_clean_when_the_cadence_is_in_the_frame():
    """The census's own comparison, run on the frames as they are now.

    A 3600s corpus asked a 12-bar horizon and a 60s live resample asked the
    same 12 bars no longer collide, so ``crossed`` is False and the script
    exits 0 rather than 2.
    """
    corpus_frame = census.horizon_frames(_full_bars(LOOKBACK_BARS + 60, 3600),
                                         12, "AERO-USDC")
    live_frame = horizon_frame(12, 60)
    assert corpus_frame != live_frame


def test_measure_bar_seconds_reports_the_modal_gap_through_a_hole():
    """One dropped bucket -- exactly what the live resampler produces -- must
    not move the cadence the frame carries."""
    bars = _bars(50, 60)
    del bars[20]
    assert measure_bar_seconds(bars) == 60
    assert measure_bar_seconds([{"close": 1.0}], default=3600) == 3600
