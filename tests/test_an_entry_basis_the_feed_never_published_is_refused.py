"""A contaminated exit must not become the next position's cost basis.

The real pair, measured 2026-09-10 over ``trade_outcomes``:

    AERO-USDC  entry 0.436805 -> exit 1.140000   +160.99%  net +3.2066
    AERO-USDC  entry 1.140000 -> exit 0.513839    -54.93%  net -1.1115

The second entry IS the first exit. Over 4976 AERO feed ticks the observed
range is 0.456 to 0.644 and 1.140000 appears nowhere in it, so the gate must
refuse 1.14 as an entry basis while still accepting 0.51.

These build a real ``market_stream`` table rather than mocking the query, so a
change to the SQL or to the column names fails here instead of silently
returning "corroborated" for everything -- which is the failure mode that would
switch the gate off without switching off any test.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.entry_price_corroboration import (  # noqa: E402
    DEFAULT_TOLERANCE, corroborating_ticks, entry_price_is_corroborated,
)

NOW = 1_756_000_000.0


@pytest.fixture()
def feed(tmp_path):
    """A market_stream carrying AERO's real observed band, and nothing else."""
    db = tmp_path / "trading_cache.db"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE market_stream (id INTEGER PRIMARY KEY, ts REAL, "
                "chain TEXT, symbol TEXT, price REAL, volume REAL, raw TEXT)")
    # 60 ticks across the two hours before NOW, inside the real 0.456-0.644 band.
    for i in range(60):
        px = 0.456316 + (0.643574 - 0.456316) * (i / 59.0)
        con.execute("INSERT INTO market_stream (ts, chain, symbol, price) "
                    "VALUES (?, ?, ?, ?)",
                    (NOW - 7000 + i * 100, "base", "AERO-USDC", px))
    con.commit()
    con.close()
    return db


def test_the_contaminated_exit_is_refused_as_the_next_entry(feed):
    """1.140000 is 77% above the top of everything the feed ever printed."""
    r = corroborating_ticks("AERO-USDC", 1.140000, at_ts=NOW, db_path=feed)

    assert r["coverage"] == 60, "fixture must be judgeable, not merely empty"
    assert r["within"] == 0
    assert r["corroborated"] is False, r["reason"]
    assert "NO feed tick" in r["reason"]
    assert entry_price_is_corroborated(
        "AERO-USDC", 1.140000, at_ts=NOW, db_path=feed) is False


def test_the_real_price_is_accepted(feed):
    """0.513839 is the exit the feed DID publish; refusing it would be a bug.

    A gate that refuses everything is switched off, not safe.
    """
    r = corroborating_ticks("AERO-USDC", 0.513839, at_ts=NOW, db_path=feed)

    assert r["corroborated"] is True, r["reason"]
    assert r["within"] > 0
    assert entry_price_is_corroborated(
        "AERO-USDC", 0.513839, at_ts=NOW, db_path=feed) is True


def test_a_symbol_the_feed_has_not_reached_is_unjudgeable_not_refused(feed):
    """The earlier AERO row closed before the feed's first AERO tick.

    Refusing entries in symbols the feed has not covered yet would stop the
    ghost harness gathering evidence in exactly the symbols that need it, so
    the default must fail OPEN -- and the live lane's ``strict`` reading must
    still refuse, because real money should not be spent on an unconfirmed
    basis.
    """
    r = corroborating_ticks("NEWCOIN-USDC", 1.23, at_ts=NOW, db_path=feed)

    assert r["coverage"] == 0
    assert r["corroborated"] is None, r["reason"]
    assert "unjudgeable" in r["reason"]

    assert entry_price_is_corroborated(
        "NEWCOIN-USDC", 1.23, at_ts=NOW, db_path=feed) is True
    assert entry_price_is_corroborated(
        "NEWCOIN-USDC", 1.23, at_ts=NOW, db_path=feed, strict=True) is False


def test_the_window_is_what_makes_a_stale_tick_stop_counting(feed):
    """Corroboration must come from the window, not from all history.

    Asked 10 hours after the last tick with a 2h window there is no coverage,
    so the answer is unjudgeable rather than a confident yes off stale data.
    """
    r = corroborating_ticks("AERO-USDC", 0.50, at_ts=NOW + 36_000,
                            db_path=feed)
    assert r["coverage"] == 0
    assert r["corroborated"] is None


def test_the_tolerance_band_is_the_measured_one(feed):
    """5% is calibrated (5.3% of entries refused); a silent change fails here.

    Also pins the band's edges: a price just outside it is refused and one just
    inside is accepted, so the comparison cannot quietly become >= or drop the
    multiplication.
    """
    assert DEFAULT_TOLERANCE == 0.05, (
        "tolerance moved; re-read the calibration table in "
        "services/entry_price_corroboration.py before changing it")

    top = 0.643574  # the highest price in the fixture
    assert corroborating_ticks(
        "AERO-USDC", top * 1.20, at_ts=NOW, db_path=feed)["corroborated"] is False
    assert corroborating_ticks(
        "AERO-USDC", top * 1.02, at_ts=NOW, db_path=feed)["corroborated"] is True
