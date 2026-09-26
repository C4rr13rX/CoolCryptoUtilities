"""The OHLCV production readiness gate must test bar AGE, not file COUNT.

[9fd050f1] services/internal_cron.py decided a chain was ready with
``files_ready = self._ohlcv_file_count(chain) >= min_files`` and there was no
freshness test anywhere in that path.  MEASURED on base: 233 files existed so
the count gate passed trivially, while of the 19 files the bot actually
resolved for a prewarm seed ELEVEN were older than three days -- 5.4, 8.3,
9.5, 13.4 (x3), 20.1, 20.3, 20.6 and 94.6 days.  The gate said ready forever.

Every test here builds its own data/ tree in tmp_path and chdirs into it, so
nothing depends on the repo's real OHLCV files or on conftest fixtures.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from services.internal_cron import InternalCronSupervisor

_DAY = 86400.0


def _write_series(root, chain: str, name: str, newest_age_days: float, bars: int = 40):
    """One OHLCV file whose newest bar is ``newest_age_days`` old.

    st_mtime is deliberately left at NOW -- these files are written right here.
    A gate reading mtime therefore calls every one of them fresh, which is the
    bug: the 9999_ prefixed files are rewritten in place with old bars.
    """
    chain_dir = root / "data" / "historical_ohlcv" / chain
    chain_dir.mkdir(parents=True, exist_ok=True)
    newest = time.time() - newest_age_days * _DAY
    series = [
        {
            "timestamp": int(newest - (bars - 1 - i) * 900),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "net_volume": 1.0,
        }
        for i in range(bars)
    ]
    path = chain_dir / name
    path.write_text(json.dumps(series), encoding="utf-8")
    return path


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    """A repo-shaped tmp tree with base's pair index present."""
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "pair_index_base.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # Pin the age policy so the test does not move when the operator retunes
    # BOT_PREWARM_MAX_AGE_DAYS in .env.
    monkeypatch.setenv("BOT_PREWARM_MAX_AGE_DAYS", "3.0")
    return tmp_path


@pytest.fixture()
def sup(monkeypatch):
    s = InternalCronSupervisor()
    # The db branch is a separate source of readiness and is not what this item
    # is about; pin it to 0 so the FILE branch is what the assertions read.
    monkeypatch.setattr(type(s), "_ohlcv_db_count", staticmethod(lambda chain: 0))
    return s


_CFG = {"chains": ["base"], "min_files_per_chain": 1, "min_chains_ready": 1}


def test_stale_bars_are_not_ready_even_though_the_count_passes(tree, sup):
    """THE REGRESSION TEST. Fails against the count-only gate.

    Enough files to pass any plausible min_files, every one of them far older
    than the bound.  The old gate returned True here; that is the defect.
    """
    for i in range(12):
        _write_series(tree, "base", f"9999_STALE{i}-USDC.json", newest_age_days=20.6)

    assert sup._ohlcv_file_count("base") == 12  # the count gate passes trivially
    age = sup._ohlcv_newest_bar_age_days("base")
    assert age is not None and age > 3.0

    assert sup._production_ready({}, _CFG) is False


def test_fresh_bars_are_ready(tree, sup):
    _write_series(tree, "base", "0111_AAVE-USDC.json", newest_age_days=0.04)
    assert sup._production_ready({}, _CFG) is True


def test_one_fresh_file_among_stale_ones_is_ready(tree, sup):
    """The gate asks whether ANY usable fresh series exists, so a fresh file
    beside stale ones is ready.  Without this the gate would flip to the
    opposite failure -- refusing a chain that does have current tape."""
    for i in range(11):
        _write_series(tree, "base", f"9999_STALE{i}-USDC.json", newest_age_days=20.6)
    _write_series(tree, "base", "0111_AAVE-USDC.json", newest_age_days=0.04)
    assert sup._production_ready({}, _CFG) is True


def test_age_comes_from_the_newest_bar_not_from_mtime(tree, sup):
    """A file rewritten NOW with old bars is still stale.

    This is the case the item calls out: the 9999_ files are rewritten in
    place, so st_mtime is current while the series inside is weeks old.
    """
    path = _write_series(tree, "base", "9999_AERO-USDT.json", newest_age_days=23.07)
    now = time.time()
    os.utime(path, (now, now))

    assert abs(path.stat().st_mtime - now) < 5  # mtime says "written seconds ago"
    age = sup._ohlcv_newest_bar_age_days("base")
    assert age is not None and age > 23.0  # the bars say 23 days
    assert sup._production_ready({}, _CFG) is False


def test_unreadable_series_is_not_ready(tree, sup):
    """No bar timestamp readable at all => cannot assert freshness => NOT
    ready.  base really does hold files like this: 0004_CBBTC-USDC.json is a
    2-element list of the string "none"."""
    chain_dir = tree / "data" / "historical_ohlcv" / "base"
    chain_dir.mkdir(parents=True, exist_ok=True)
    (chain_dir / "0004_CBBTC-USDC.json").write_text('["none", "none"]', encoding="utf-8")

    assert sup._ohlcv_file_count("base") == 1  # non-zero size, so the count passes
    assert sup._ohlcv_newest_bar_age_days("base") is None
    assert sup._production_ready({}, _CFG) is False


def test_profile_may_state_its_own_bound(tree, sup):
    _write_series(tree, "base", "9999_CAKE-USDC.json", newest_age_days=15.77)
    assert sup._production_ready({}, _CFG) is False
    loose = dict(_CFG, max_ohlcv_age_days=30.0)
    assert sup._production_ready({}, loose) is True


def test_bound_defaults_to_the_prewarm_guard_policy(tree, sup, monkeypatch):
    """ONE age policy. If the guard's bound moves, the gate's moves with it --
    a gate that admits bars the bot will refuse to splice is the same defect
    in a different file."""
    from services import prewarm_seed_guard

    assert sup._ohlcv_max_age_days({}) == pytest.approx(prewarm_seed_guard.max_age_days())
    monkeypatch.setenv("BOT_PREWARM_MAX_AGE_DAYS", "9.0")
    assert sup._ohlcv_max_age_days({}) == pytest.approx(9.0)


def test_newest_bar_is_found_in_a_long_series(tree, sup):
    """The tail read must not miss the newest bar in a realistically long
    file -- base's largest is ~22k bars."""
    _write_series(tree, "base", "0004_AERO-USDC.json", newest_age_days=0.5, bars=22000)
    age = sup._ohlcv_newest_bar_age_days("base")
    assert age == pytest.approx(0.5, abs=0.05)
