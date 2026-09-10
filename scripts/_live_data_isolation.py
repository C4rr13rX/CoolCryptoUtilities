"""A pytest plugin that starves the live-data predicates of the production DB.

Loaded with ``pytest -p scripts._live_data_isolation`` by
``scripts/live_data_predicate_census.py --prove``. It exists so the census can
ask ONE question empirically rather than by reading imports:

    does this test file's verdict change when the market data changes?

WHAT IT DOES. Three predicates decide, at runtime, whether a symbol may be
traded, and all three answer from ``storage/trading_cache.db`` -- this week's
tape:

    services.symbol_edge_gate.refusal_reason        the closed round-trip book
    services.symbol_motion_gate.refusal_reason      the price feed
    services.stop_survivability_gate.refusal_reason the price feed

Each is a module with a ``DB_PATH`` constant and a memo cache. The plugin
repoints ``DB_PATH`` at an EMPTY database carrying the same tables, and clears
the caches, so every gate returns "allow" for every symbol -- the answer a
fixture-fed test must be indifferent to, and the answer a tape-fed test will
change its mind about.

WHY EMPTY RATHER THAN A SECOND FIXTURE. All three gates FAIL OPEN by design
(see their docstrings): no evidence means no refusal. So the empty database is
the one alternative tape whose correct verdict is known in advance for every
symbol, which makes a difference between the two runs attributable to the data
rather than to the choice of replacement data.

The plugin never writes to the production database and never runs unless the
census asks for it.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

#: module path -> the tables its reader selects from. Creating them empty is
#: what separates "no rows" from "no table", which the gates' except-clauses
#: would treat identically and so would hide a schema mistake here.
#:
#: EVERY MODULE WITH ITS OWN ``DB_PATH`` MUST BE HERE, not only the ones a
#: predicate is named after. Leaving ``strategy_edge_gate`` out on 2026-09-10
#: repointed one of the two edge gates and not the other, and the census then
#: reported ``test_both_edge_gates_charge_the_same_cost`` as tape-dependent --
#: 0.004653 against 0.0065 -- when what had actually diverged was this list.
#: A partial repoint does not weaken the instrument, it INVERTS it: it
#: manufactures exactly the disagreement the suite exists to detect.
GATE_MODULES = {
    "services.symbol_edge_gate": ("trade_outcomes",),
    "services.strategy_edge_gate": ("trade_outcomes",),
    "services.symbol_motion_gate": ("market_stream",),
    "services.stop_survivability_gate": ("market_stream",),
}

_EMPTY_DB: Path | None = None


def _empty_db() -> Path:
    global _EMPTY_DB
    if _EMPTY_DB is not None:
        return _EMPTY_DB
    path = Path(tempfile.gettempdir()) / "ccu_live_data_isolation_empty.db"
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS market_stream "
            "(symbol TEXT, chain TEXT, ts REAL, price REAL, volume REAL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS trade_outcomes "
            "(id INTEGER PRIMARY KEY, symbol TEXT, ts REAL, entry_price REAL, "
            " exit_price REAL, quantity REAL, net_profit REAL, details TEXT)"
        )
        conn.commit()
    finally:
        conn.close()
    _EMPTY_DB = path
    return path


def _starve() -> list[str]:
    """Repoint every gate at the empty database. Returns the modules touched."""
    import importlib

    touched = []
    empty = _empty_db()
    for name in GATE_MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - a gate that will not import cannot decide anything
            continue
        if not hasattr(mod, "DB_PATH"):
            continue
        mod.DB_PATH = empty
        for attr in ("_cache", "_pair_cache", "_pair_seen"):
            holder = getattr(mod, attr, None)
            if hasattr(holder, "clear"):
                holder.clear()
        if hasattr(mod, "_cache_built_at"):
            mod._cache_built_at = 0.0
        touched.append(name)
    return touched


def pytest_configure(config):
    touched = _starve()
    config._live_data_isolation = touched
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            f"live-data isolation: {len(touched)} gate(s) repointed at an empty database"
        )


def pytest_runtest_setup(item):
    """Re-starve before every test.

    A test that imports a gate for the first time, or one that restores
    ``DB_PATH`` in a fixture teardown, would otherwise leak the production
    database into the tests that follow it -- and a leak here reads as "this
    file is tape-independent", the exact false negative the census exists to
    avoid.
    """
    _starve()
