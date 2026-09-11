"""A horizon in BARS is not one horizon, and a report that hides that is a lie.

data/historical_ohlcv mixes cadences from 73s to 11988s (censused pass 113 over
all 629 files, data/brain_experiments/p113_corpus_cadence_census.json). Under
the old ``--horizon 12`` flag the SAME command asked about 15 minutes on one
file and 20 hours on another, and both runs wrote a report whose only horizon
field was the number 12. Two such reports are not comparable and nothing in the
file says so.

These tests pin the fix:

  * the same wall-clock horizon converts to DIFFERENT bar counts on two
    corpora, and both runs record the SAME minutes
  * minutes is the DEFAULT unit; bars survive only as an explicit override
  * a report that does not carry all of horizon_minutes / horizon_bars /
    bar_seconds is refused at write time
  * a corpus file too coarse, too thin, or holding two timeframes is excluded
    from SELECTION, by a bound that is named rather than implied

Every one of them goes red against the bars-only flag: ``resolve_horizon`` and
``validate_report_horizon`` did not exist, and ``--horizon`` defaulted to 12.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.omen_experiment import (  # noqa: E402
    DEFAULT_HORIZON_MINUTES, MAX_CADENCE_DRIFT_RATIO, MAX_CORPUS_BAR_SECONDS,
    bar_seconds, horizon_bars, load_bars, resolve_horizon,
    validate_report_horizon,
)


def write_corpus(path: Path, cadence_seconds: int, bars: int = 300) -> Path:
    """A synthetic corpus at one cadence, in the shape load_bars reads."""
    start = 1_700_000_000
    rows = [{"timestamp": start + i * cadence_seconds,
             "open": 100.0 + i, "high": 101.0 + i, "low": 99.0 + i,
             "close": 100.0 + i, "volume": 1000.0}
            for i in range(bars)]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


# --- the property the whole item is about ---------------------------------

def test_one_horizon_in_minutes_is_a_different_bar_count_on_each_corpus(tmp_path):
    fast = write_corpus(tmp_path / "fast_FAST-USDC.json", 300)
    slow = write_corpus(tmp_path / "slow_SLOW-USDC.json", 3600)

    fast_cadence = bar_seconds(load_bars(fast))
    slow_cadence = bar_seconds(load_bars(slow))
    assert (fast_cadence, slow_cadence) == (300, 3600)

    asked_minutes = 120.0
    fast_h = resolve_horizon(fast_cadence, minutes=asked_minutes)
    slow_h = resolve_horizon(slow_cadence, minutes=asked_minutes)

    # The QUESTION is the same on both corpora...
    assert fast_h["horizon_minutes"] == slow_h["horizon_minutes"] == 120.0
    # ...and the bar count that expresses it is not.
    assert fast_h["horizon_bars"] == 24
    assert slow_h["horizon_bars"] == 2
    assert fast_h["horizon_bars"] != slow_h["horizon_bars"]
    assert fast_h["horizon_source"] == "minutes"


def test_a_bars_horizon_records_the_minutes_it_actually_asked():
    """The bars form survives, but it may not hide what it asked."""
    on_hourly = resolve_horizon(3600, bars=12)
    on_five_minute = resolve_horizon(300, bars=12)
    assert on_hourly["horizon_bars"] == on_five_minute["horizon_bars"] == 12
    # Same flag, twelve times apart in wall clock -- and now both say so.
    assert on_hourly["horizon_minutes"] == 720.0
    assert on_five_minute["horizon_minutes"] == 60.0
    assert on_hourly["horizon_source"] == "bars"


def test_asking_in_both_units_at_once_is_refused():
    with pytest.raises(ValueError):
        resolve_horizon(3600, minutes=120.0, bars=12)


def test_a_sub_bar_horizon_still_looks_one_bar_ahead():
    """Rounding to 0 bars would compare a bar's close against itself."""
    assert horizon_bars(1, 3600) == 1
    assert resolve_horizon(3600, minutes=0.5)["horizon_bars"] == 1


# --- the default unit ------------------------------------------------------

def test_the_experiment_defaults_to_minutes_not_bars(tmp_path):
    """Run the real CLI on two cadences and read what it says it asked.

    The run stops at window planning -- the synthetic corpora are far too
    short -- so this needs no brain node and touches no fabric.
    """
    outputs = {}
    for cadence in (300, 3600):
        corpus = write_corpus(tmp_path / f"c{cadence}_SYN-USDC.json", cadence)
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", "scripts/omen_experiment.py",
             "--corpus", str(corpus)],
            cwd=str(ROOT), capture_output=True, text=True, timeout=180)
        outputs[cadence] = proc.stdout

    # Same question of both, in minutes, without the flag being passed.
    for cadence, text in outputs.items():
        assert f"horizon {DEFAULT_HORIZON_MINUTES:.0f} min" in text, text
        assert "asked in minutes" in text, text
    assert f"= {horizon_bars(DEFAULT_HORIZON_MINUTES, 300)} bars of 300s" \
        in outputs[300]
    assert f"= {horizon_bars(DEFAULT_HORIZON_MINUTES, 3600)} bars of 3600s" \
        in outputs[3600]


# --- the report must carry the unit ---------------------------------------

def test_a_report_without_the_minutes_is_refused_at_write_time():
    good = {"horizon_bars": 12, "horizon_minutes": 720.0, "bar_seconds": 3600}
    validate_report_horizon(good)  # does not raise

    for field in ("horizon_bars", "horizon_minutes", "bar_seconds"):
        missing = dict(good)
        missing.pop(field)
        with pytest.raises(ValueError) as exc:
            validate_report_horizon(missing)
        assert field in str(exc.value)


def test_a_report_whose_units_disagree_is_refused():
    """720 minutes of 300s bars is 144 bars, not 12 -- catch the copy-paste."""
    with pytest.raises(ValueError):
        validate_report_horizon(
            {"horizon_bars": 12, "horizon_minutes": 720.0, "bar_seconds": 300})


# --- selection over the corpus --------------------------------------------

def test_the_selection_bound_on_cadence_is_named_and_applied():
    """The multi-symbol selector must not default to 'take anything'."""
    from scripts.omen_generalisation import cadence_drift

    source = (ROOT / "scripts" / "omen_generalisation.py").read_text(
        encoding="utf-8")
    assert "default=MAX_CORPUS_BAR_SECONDS" in source, (
        "--max-bar-seconds defaulting to None lets a 4-day-gap stub be picked "
        "like any other file")

    start = 1_700_000_000
    steady = [{"timestamp": start + i * 3600, "close": 1.0} for i in range(800)]
    assert cadence_drift(steady) == pytest.approx(1.0)

    # Hourly for the head, then half-hourly: bar_seconds sees only 3600s, so
    # every horizon in the tail is converted by twice the right factor.
    two_timeframes = [{"timestamp": start + i * 3600, "close": 1.0}
                      for i in range(400)]
    tail = two_timeframes[-1]["timestamp"]
    two_timeframes += [{"timestamp": tail + i * 1800, "close": 1.0}
                       for i in range(1, 1200)]
    assert cadence_drift(two_timeframes) > MAX_CADENCE_DRIFT_RATIO


def test_the_cadence_census_excludes_the_stub_and_names_the_bound(tmp_path):
    from scripts.omen_corpus_cadence_census import census_file

    # Three years in 23 rows: this is the file whose "cadence" reads 345600s.
    stub = tmp_path / "9999_STUB-USDC.json"
    write_corpus(stub, 345600, bars=23)
    row = census_file(stub)
    assert row["eligible"] is False
    assert row["reason_class"] == "too_few_bars"

    coarse = tmp_path / "9999_COARSE-USDC.json"
    write_corpus(coarse, MAX_CORPUS_BAR_SECONDS * 2, bars=300)
    row = census_file(coarse)
    assert row["eligible"] is False
    assert row["reason_class"] == "coarser_than_bound"
    assert str(MAX_CORPUS_BAR_SECONDS) in row["reason"]

    fine = tmp_path / "0001_FINE-USDC.json"
    write_corpus(fine, 3600, bars=300)
    row = census_file(fine)
    assert row["eligible"] is True
    assert row["head_seconds"] == 3600
    assert row["horizon_12bars_minutes"] == 720.0


def test_the_published_cadence_census_covers_the_whole_corpus():
    """The census in data/brain_experiments must describe every corpus file."""
    report = ROOT / "data" / "brain_experiments" / "p113_corpus_cadence_census.json"
    assert report.exists(), "the cadence census has never been run"
    data = json.loads(report.read_text(encoding="utf-8"))
    on_disk = len(list((ROOT / "data" / "historical_ohlcv").rglob("*.json")))
    assert data["files"] == on_disk
    assert data["eligible"] + data["excluded"] == data["files"]
    assert data["selection_bound_bar_seconds"] == MAX_CORPUS_BAR_SECONDS
    assert sum(data["files_by_cadence_seconds"].values()) > 0
