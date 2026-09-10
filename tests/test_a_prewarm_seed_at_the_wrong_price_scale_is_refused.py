"""A historical prewarm seed is spliced into the SAME buffer the live stream
fills, so a seed at a different price scale -- or from a different week --
puts a foreign row inside the model's 60-bar window.  One such row sets the
scale of the entire window's distribution (price_mu -0.17 -> -1.8, b966158).

These tests fail against the pre-guard behaviour, where
``_prewarm_buffer_from_history`` seeded every file it could resolve without
comparing it to the live feed or checking its age.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from services.prewarm_seed_guard import (
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_MAX_LOG_RATIO,
    median_price,
    seed_verdict,
)

DAY = 86400.0


def test_a_seed_ten_log_units_from_the_live_price_is_refused():
    """PUMP-USDC, measured 2026-09-10: the newest prewarm file's median close
    is 0.0041 while the live feed reads 1.0220e-07 -- a log ratio of +10.6.
    That is not a price move, it is a different quantity."""
    now = time.time()
    v = seed_verdict(
        seed_median=0.0041,
        live_median=1.0220e-07,
        newest_bar_ts=now - 3600.0,   # FRESH: age must not be what refuses it
        now=now,
    )
    assert not v.ok
    assert v.reason == "seed_price_scale_mismatch"
    assert v.detail["log_ratio"] > 10.0


def test_a_refusal_names_both_numbers_it_compared():
    """A refusal that does not carry the seed price AND the live price sends
    the next reader back to re-measure it."""
    now = time.time()
    v = seed_verdict(
        seed_median=0.0041,
        live_median=1.0220e-07,
        newest_bar_ts=now - 30 * DAY,
        now=now,
    )
    line = v.log_line("PUMP-USDC", "9999_PUMP-USDC.json")
    assert "REFUSED" in line
    assert "seed_median=0.0041" in line
    assert "live_median=1.022e-07" in line
    # Refused on AGE, and the SCALE is still reported -- both measurements are
    # taken before either is judged.
    assert v.reason == "seed_too_old"
    assert "log_ratio=" in line
    assert "age_days=" in line


def test_a_twenty_day_old_seed_is_refused_even_at_the_right_scale():
    """ETH-USDT, measured: log ratio -0.0244 (the right series) and 20.14 days
    old.  Its rows are appended with their ORIGINAL timestamps into a buffer
    the rest of the bot reads as live ticks."""
    now = time.time()
    v = seed_verdict(
        seed_median=2650.0,
        live_median=2715.0,
        newest_bar_ts=now - 20.14 * DAY,
        now=now,
    )
    assert not v.ok
    assert v.reason == "seed_too_old"
    assert v.detail["age_days"] == pytest.approx(20.14, abs=0.01)
    assert abs(v.detail["log_ratio"]) < DEFAULT_MAX_LOG_RATIO


def test_a_fresh_seed_at_the_live_scale_still_seeds():
    """The guard must not switch the prewarm off.  Eight live symbols seed
    cleanly today at |log ratio| <= 0.0802 and 1.02 days old; every one of
    them must survive."""
    now = time.time()
    v = seed_verdict(
        seed_median=17.42,
        live_median=16.08,          # log ratio +0.0801, the worst clean row
        newest_bar_ts=now - 1.02 * DAY,
        now=now,
    )
    assert v.ok
    assert v.reason == "within_tolerance"
    assert v.detail["age_days"] < DEFAULT_MAX_AGE_DAYS


def test_no_live_reference_is_not_a_refusal():
    """The prewarm exists for the cold start where no live tick has arrived.
    Refusing there would switch the feature off for every new symbol rather
    than fixing a seam -- but it is REPORTED so it can be counted."""
    now = time.time()
    v = seed_verdict(
        seed_median=17.42,
        live_median=None,
        newest_bar_ts=now - 600.0,
        now=now,
    )
    assert v.ok
    assert v.reason == "no_live_reference"


def test_median_ignores_nonpositive_and_is_not_the_freshest_tick():
    """market_stream interleaves sources, so a source publishing a different
    denomination makes consecutive ticks alternate between right and wrong.
    One lookup is a coin flip; the median is not."""
    # four positives survive -- 0.98, 1.0, 1.02, 55.0 -- and the 55.0 outlier
    # moves the median not at all while it would move a mean to 14.5.
    assert median_price([1.0, 1.02, 0.98, 55.0, 0.0, -3.0]) == pytest.approx(1.01)
    assert median_price([]) is None
    assert median_price([0.0, -1.0]) is None


def test_the_bot_refuses_a_stale_wrong_scale_file_instead_of_seeding(tmp_path, monkeypatch, capsys):
    """End to end through the code the bot actually runs: a resolvable file at
    the wrong scale leaves the buffer EMPTY.  Against the pre-guard behaviour
    this seeded 60 foreign samples."""
    from trading import bot as bot_mod

    chain_dir = tmp_path / "data" / "historical_ohlcv" / "base"
    chain_dir.mkdir(parents=True)
    stale = time.time() - 20 * DAY
    rows = [
        {"timestamp": stale + i * 60.0, "close": 0.0041, "volume": 10.0}
        for i in range(80)
    ]
    (chain_dir / "9999_PUMP-USDC.json").write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    class _Db:
        def recent_market_prices(self, symbol, chain, *, since_ts=None, limit=25):
            return [(1.0220e-07, time.time() - i) for i in range(10)]

    class _Stub:
        window_size = 60
        primary_chain = "base"

        def __init__(self):
            self._buffer = []
            self.db = _Db()

        _prewarm_buffer_from_history = bot_mod.TradingBot._prewarm_buffer_from_history

    stub = _Stub()
    stub._prewarm_buffer_from_history("PUMP-USDC")

    assert stub._buffer == []
    out = capsys.readouterr().out
    assert "REFUSED" in out
    assert "PUMP-USDC" in out


def test_the_bot_still_seeds_a_fresh_file_at_the_live_scale(tmp_path, monkeypatch):
    """The counterpart: the guard must not be a gate that refuses everything."""
    from trading import bot as bot_mod

    chain_dir = tmp_path / "data" / "historical_ohlcv" / "base"
    chain_dir.mkdir(parents=True)
    fresh = time.time() - 3600.0
    rows = [
        {"timestamp": fresh + i * 60.0, "close": 16.0 + (i % 5) * 0.01, "volume": 10.0}
        for i in range(80)
    ]
    (chain_dir / "0082_LINK-USDC.json").write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    class _Db:
        def recent_market_prices(self, symbol, chain, *, since_ts=None, limit=25):
            return [(16.08, time.time() - i) for i in range(10)]

    class _Stub:
        window_size = 60
        primary_chain = "base"

        def __init__(self):
            self._buffer = []
            self.db = _Db()

        _prewarm_buffer_from_history = bot_mod.TradingBot._prewarm_buffer_from_history

    stub = _Stub()
    stub._prewarm_buffer_from_history("LINK-USDC")

    assert len(stub._buffer) == 60
    assert all(s["source"] == "history_prewarm" for s in stub._buffer)
