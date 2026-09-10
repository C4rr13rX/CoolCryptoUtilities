"""A census must not call a window unsatisfiable when it only read the newest slice of it.

``scripts/entry_conjunct_census.py`` answered "can this entry conjunct EVER be
true on the current feed?" with ``ORDER BY ts DESC LIMIT 2000``. On 2026-09-10
the 24h window held 5540 prediction blocks, so it read the newest 8.7 hours,
printed "in the last 24h", and concluded::

    direction_prob   max 0.5000  floor 0.6000  reachable 0/2000
    UNSATISFIABLE ON THIS FEED -- never cleared its floor once in the window.

The whole window said the opposite: direction_prob cleared 0.6 on 1684 of 5540
ticks (30.4%) and cleared it TOGETHER with exit_conf on 141. Every one of those
141 was older than the truncation point, because the prediction head had
decayed over the preceding twelve hours -- so the truncated read turned a
recent regression into a permanent property of the feed, and the verdict told
the next reader to go and fix a number that had been fine half a day earlier.

These tests pin both halves: the default read covers the window, and a read
that IS truncated withholds the verdict instead of printing it.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from scripts.entry_conjunct_census import (
    read_window,
    census,
    count_snapshots,
    read_predictions,
    render,
    unsatisfiable,
)


def _db(tmp_path, rows):
    """A snapshot table holding ``rows`` of (age_seconds, direction_prob)."""
    path = str(tmp_path / "trading_cache.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE organism_snapshots (ts REAL, payload TEXT)")
    conn.executemany(
        "INSERT INTO organism_snapshots (ts, payload) VALUES (?, ?)",
        [
            (
                1_000_000.0 - age,
                json.dumps({"prediction": {
                    "direction_prob": prob,
                    "exit_conf": prob,
                    "net_margin": 0.0,
                }}),
            )
            for age, prob in rows
        ],
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture()
def decayed_feed(tmp_path):
    """The shape production actually had: healthy early, degenerate lately.

    30 old ticks clear the 0.6 floor; the 20 newest do not. A read capped at
    the newest 20 therefore sees a floor that "never" clears.
    """
    rows = [(3600.0 - i, 0.9) for i in range(30)]          # older, clearing
    rows += [(600.0 - i, 0.05) for i in range(20)]         # newest, not clearing
    return _db(tmp_path, rows)


def test_the_default_read_covers_the_whole_window(decayed_feed):
    """No limit by default -- every row in the window is scored."""
    preds = read_predictions(decayed_feed, hours=24.0, now=1_000_000.0)
    assert len(preds) == 50
    assert len(preds) == count_snapshots(decayed_feed, hours=24.0, now=1_000_000.0)


def test_a_conjunct_that_clears_only_in_the_older_half_is_not_unsatisfiable(
    decayed_feed,
):
    """The regression this file is named for.

    Against the old ``limit=2000``-style truncation the newest rows are all
    below the floor, so ``unsatisfiable`` named direction_prob. Reading the
    whole window shows it cleared 30 times.
    """
    whole = census(read_predictions(decayed_feed, hours=24.0, now=1_000_000.0))
    assert whole["direction_prob"]["reachable"] == 30
    assert "direction_prob" not in unsatisfiable(whole)

    truncated = census(
        read_predictions(decayed_feed, hours=24.0, limit=20, now=1_000_000.0)
    )
    assert truncated["direction_prob"]["reachable"] == 0
    assert "direction_prob" in unsatisfiable(truncated)


def test_a_partial_read_withholds_the_unsatisfiable_verdict(decayed_feed):
    """A window that was only partly read must not be reported as failing.

    The census already refuses to score a conjunct whose input is missing from
    the payload, because an unmeasured condition must not read as a failing
    one. A partly-read window is the same error one level up.
    """
    preds = read_predictions(decayed_feed, hours=24.0, limit=20, now=1_000_000.0)
    in_window = count_snapshots(decayed_feed, hours=24.0, now=1_000_000.0)
    out = render(census(preds), read=len(preds), in_window=in_window)

    assert "WINDOW ONLY PARTLY READ: 20 of 50" in out
    assert "UNSATISFIABLE ON THIS FEED" not in out


def test_a_row_with_no_prediction_block_does_not_read_as_truncation(tmp_path):
    """Coverage is counted in ROWS, not in predictions.

    A snapshot carrying no ``prediction`` block is fully read; it just has
    nothing to score. Counting coverage on predictions made a complete read of
    the live table report "5550 of 5552" and withhold a verdict it owed.
    """
    path = _db(tmp_path, [(600.0 - i, 0.05) for i in range(20)])
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO organism_snapshots (ts, payload) VALUES (?, ?)",
        (999_500.0, json.dumps({"no_prediction_here": True})),
    )
    conn.commit()
    conn.close()

    preds, rows_read, in_window = read_window(path, hours=24.0, now=1_000_000.0)
    assert len(preds) == 20          # one row had nothing to score
    assert rows_read == in_window == 21
    assert "WINDOW ONLY PARTLY READ" not in render(
        census(preds), read=rows_read, in_window=in_window
    )


def test_a_complete_read_still_reports_a_genuinely_dead_conjunct(tmp_path):
    """The withholding must not swallow the finding the census exists for."""
    path = _db(tmp_path, [(600.0 - i, 0.05) for i in range(20)])
    preds = read_predictions(path, hours=24.0, now=1_000_000.0)
    in_window = count_snapshots(path, hours=24.0, now=1_000_000.0)
    out = render(census(preds), read=len(preds), in_window=in_window)

    assert "WINDOW ONLY PARTLY READ" not in out
    assert "UNSATISFIABLE ON THIS FEED" in out
    assert "direction_prob" in out
