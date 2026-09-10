"""A price series that is two series interleaved must be named, not averaged.

Measured 2026-09-10, 47 of 132 streamed symbols carry a second price regime.
The two shapes, both built here from their real numbers:

  COMP-USDC   4468 ticks, median 19.98, 170 ticks at 42.82-55.34
              -- two assets published under one ticker.
  CBETH-WETH  384 ticks, median 1.1386, 5 ticks at 2687-2851
              -- one asset published in two denominations (WETH, then USD).

The failure these prevent is the census going quiet: a ratio test that used
mean/sigma instead of the median, or that required a large off-regime FRACTION,
would report "no regimes" on exactly the symbols that have them, because a
bimodal series inflates its own sigma and the off-regime cluster can be 1% of
the ticks and still be every trade that matters.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.feed_regime_census import (  # noqa: E402
    MIN_REGIME_TICKS, REGIME_RATIO, census,
)


def _db(tmp_path, series_by_symbol):
    db = tmp_path / "trading_cache.db"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE market_stream (id INTEGER PRIMARY KEY, ts REAL, "
                "chain TEXT, symbol TEXT, price REAL, volume REAL, raw TEXT)")
    ts = 1_756_000_000.0
    for sym, prices in series_by_symbol.items():
        for i, px in enumerate(prices):
            con.execute("INSERT INTO market_stream (ts, chain, symbol, price) "
                        "VALUES (?, ?, ?, ?)", (ts + i, "base", sym, px))
    con.commit()
    con.close()
    return db


def test_two_assets_under_one_ticker_are_named(tmp_path):
    """COMP's real shape: a dominant low regime and a smaller high one."""
    prices = [19.98] * 200 + [42.82, 48.0, 51.0, 53.5, 55.34] * 4
    db = _db(tmp_path, {"COMP-USDC": prices})

    rep = census(db_path=db)

    assert rep["affected"] == 1, rep
    row = rep["rows"][0]
    assert row["symbol"] == "COMP-USDC"
    assert row["off_regime"] == 20
    assert row["far_min"] == pytest.approx(42.82)
    assert row["far_max"] == pytest.approx(55.34)
    # The off-regime cluster is a small FRACTION and must still be reported --
    # this is the assertion that fails a "needs >10% of ticks" rule.
    assert row["off_regime_pct"] < 10.0


def test_one_asset_in_two_denominations_is_named(tmp_path):
    """cbETH quoted in WETH and in USD under a single symbol."""
    prices = [1.1386] * 100 + [2687.06, 2750.0, 2800.0, 2820.0, 2851.21]
    db = _db(tmp_path, {"CBETH-WETH": prices})

    rep = census(db_path=db)

    assert rep["affected"] == 1
    row = rep["rows"][0]
    assert row["off_regime"] == MIN_REGIME_TICKS, (
        "a 5-tick denomination flip is exactly at the reporting floor and "
        "must still be named")
    assert row["far_min"] > 2000.0


def test_an_ordinary_volatile_symbol_is_not_named(tmp_path):
    """A symbol that merely MOVED must not be reported as two regimes.

    A census that flags everything is as useless as one that flags nothing.
    """
    prices = [1.0 + 0.4 * ((i % 7) / 7.0) for i in range(200)]
    db = _db(tmp_path, {"AERO-USDC": prices})

    rep = census(db_path=db)

    assert rep["symbols_examined"] == 1
    assert rep["affected"] == 0, rep["rows"]


def test_a_handful_of_bad_prints_is_below_the_floor(tmp_path):
    """Fewer than MIN_REGIME_TICKS off-regime ticks is noise, not a regime."""
    prices = [10.0] * 200 + [900.0] * (MIN_REGIME_TICKS - 1)
    db = _db(tmp_path, {"X-USDC": prices})

    assert census(db_path=db)["affected"] == 0

    # One more, and it crosses.
    other = tmp_path / "b"
    other.mkdir()
    prices2 = [10.0] * 200 + [900.0] * MIN_REGIME_TICKS
    assert census(db_path=_db(other, {"X-USDC": prices2}))["affected"] == 1


def test_the_ratio_is_the_documented_one():
    assert REGIME_RATIO == 1.5, (
        "ratio moved; the measured splits run 2x to 2500x, so re-read the "
        "note in scripts/feed_regime_census.py before changing it")
