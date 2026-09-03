"""Link 9 must count trades that happened, not trades that were refused.

``status LIKE 'live%'`` also matches live-entry-blocked, live-entry-failed and
live-dry-run-entry. The first live entry the swap guard refused would have
written a live-entry-blocked row and flipped link 9 to PASS -- the check would
have declared the path open on the very evidence that it was shut.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_check(db_path: Path):
    spec = importlib.util.spec_from_file_location(
        "live_path_check_under_test", ROOT / "scripts" / "live_path_check.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.DB_PATH = "file:%s?mode=ro" % db_path.as_posix()
    return mod


def _seed(db_path: Path, statuses):
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE trading_ops (id INTEGER PRIMARY KEY, ts REAL, wallet TEXT, "
        "chain TEXT, symbol TEXT, action TEXT, status TEXT, details TEXT)"
    )
    con.executemany(
        "INSERT INTO trading_ops (ts, wallet, chain, symbol, action, status, details) "
        "VALUES (?, 'live', 'base', 'T-USDC', 'enter', ?, '{}')",
        [(1_000_000.0, s) for s in statuses],
    )
    con.commit()
    con.close()


def test_refused_live_entries_do_not_count_as_live_trades(tmp_path) -> None:
    db_path = tmp_path / "ops.db"
    _seed(db_path, ["live-entry-blocked", "live-entry-failed", "live-dry-run-entry",
                    "guard-blocked-live", "live-entry-unfunded"])
    mod = _load_check(db_path)

    link = mod.check_live(1_000_100.0)

    assert link.ok is False
    assert "no live trades yet" in link.detail


def test_refusals_in_the_last_hour_are_surfaced(tmp_path) -> None:
    db_path = tmp_path / "ops.db"
    _seed(db_path, ["guard-blocked-live", "guard-blocked-live", "live-entry-blocked",
                    "live-entry-unfunded"])
    mod = _load_check(db_path)

    link = mod.check_live(1_000_100.0)

    assert link.ok is False
    # "live-entry-unfunded" is the wallet failing to fund a clip the guard had
    # already cleared. It is a refusal, so it must be counted here -- it was
    # the one exit on the live path that returned in total silence.
    assert "4 live entries REFUSED" in link.detail


def test_an_executed_live_entry_passes(tmp_path) -> None:
    db_path = tmp_path / "ops.db"
    _seed(db_path, ["live-entry", "live-exit", "live-entry-blocked"])
    mod = _load_check(db_path)

    link = mod.check_live(1_000_100.0)

    assert link.ok is True
    assert "2 executed live rows" in link.detail
