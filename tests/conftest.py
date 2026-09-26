import importlib
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = REPO_ROOT / "web"
CACHE_ROOT = Path(os.getenv("PYTEST_CACHE_HOME", REPO_ROOT / "runtime" / "pytest_cache"))

# Ensure imports like coolcrypto_dashboard.* resolve without relying on external PYTHONPATH tweaks.
for path in (REPO_ROOT, WEB_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "reads_live_book: test deliberately reads the PRODUCTION trading store "
        "(storage/trading_cache.db) instead of per-test state",
    )
    cache_root = CACHE_ROOT
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        cache_root = Path("/tmp")
    kivy_home = cache_root / "kivy"
    mpl_home = cache_root / "matplotlib"
    for path in (kivy_home, mpl_home):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    os.environ.setdefault("KIVY_HOME", str(kivy_home))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_home))

    # Force Django to use a local SQLite database during tests to avoid Postgres dependencies.
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")
    os.environ.setdefault("DJANGO_SQLITE_PATH", str(cache_root / "django_test.sqlite3"))
    os.environ["DJANGO_DB_VENDOR"] = "sqlite"
    os.environ["TRADING_DB_VENDOR"] = "sqlite"
    os.environ["ALLOW_SQLITE_FALLBACK"] = "1"

    # And the SAME isolation for the trading store, which this function has
    # always given Django and never given the one database that matters.
    #
    # Measured (Dune, p709, item [82716309]): before this line, a single run of
    # tests/test_a_banned_strategy_does_not_take_the_tick.py -- 5 tests, 1.92s --
    # appended 7 rows to `trading_ops` in storage/trading_cache.db, 4 of them
    # `entry-predropped-edge-ban`. Two of its fixture strings had accumulated
    # 3025 and 3042 copies, 6067 rows in total: 19.7% of every ban reason in the
    # store, the ENTIRE n=9 bucket of a refusal census, and timestamped NEWER
    # than any real refusal. The complement -- "those fixtures were copied FROM
    # production, so the rows are genuine" -- was tested and is dead: a
    # before/after COUNT(*) around one run shows the rows are created by the run.
    #
    # With this line, the same file passes 5/5 and the real store moves by 0.
    #
    # setdefault, not assignment: a caller that deliberately points the suite at
    # a real store (a migration check, a one-off census) keeps that power.
    os.environ.setdefault("TRADING_DB_PATH", str(cache_root / "trading_cache_test.db"))


@pytest.fixture(autouse=True)
def _isolate_market_endpoint_selection(monkeypatch):
    """A developer .env must not silently change unit-test endpoint topology."""
    monkeypatch.delenv("MARKET_ENDPOINT_INCLUDE", raising=False)
    monkeypatch.setenv("MARKET_ENDPOINT_EXCLUDE", "dexscreener")
    try:
        from trading import data_stream

        monkeypatch.setattr(data_stream, "_ENV_ENDPOINT_INCLUDE", set())
        monkeypatch.setattr(data_stream, "_ENV_ENDPOINT_EXCLUDE", set())
    except Exception:
        pass


#: The three money-path gates that open the trading store THEMSELVES, with a
#: module-level ``DB_PATH`` hardcoded to ``storage/trading_cache.db``. They do
#: not consult ``TRADING_DB_PATH``, so the isolation added above at line 58
#: never reached them: a unit test that asserts on a gate verdict was reading
#: the 22GB PRODUCTION book at test time, and went red or green from TRADING
#: ACTIVITY ALONE with no code change.
#:
#: Each entry is (module path, attribute holding the db path, cache globals to
#: reset). The cache globals matter as much as the path: all three gates keep a
#: module-level verdict dict plus a ``_cache_built_at`` timestamp with a
#: CACHE_SEC TTL, so without the reset the FIRST test to touch a gate would
#: build a book and every later test in the same process would be served that
#: build -- test order would decide the verdict, which is the same defect one
#: level up.
_PINNED_GATES = (
    ("services.symbol_edge_gate", "DB_PATH",
     ("_cache", "_pair_cache", "_pair_seen", "_cache_built_at")),
    ("services.stop_survivability_gate", "DB_PATH",
     ("_cache", "_cache_built_at")),
    ("services.symbol_motion_gate", "DB_PATH",
     ("_cache", "_cache_built_at")),
)


@pytest.fixture(autouse=True)
def _pin_gate_books_to_test_state(request, monkeypatch):
    """Point the three money-path gates at per-test state, not production.

    Pins each gate's ``DB_PATH`` to ``TRADING_DB_PATH`` -- the same isolated
    store ``pytest_configure`` above already gives the rest of the suite -- so
    a test that seeds the trading store through the normal db layer and a test
    that reads a gate verdict are looking at ONE book, and neither is looking
    at the live one.

    All three gates guard their ``sqlite3.connect`` with ``except Exception``
    and fail OPEN (empty book, no refusal), so pinning to a store that does not
    exist yet yields "no ban" rather than an error. That is the correct default
    for a unit test: a ban has to be SET UP to be asserted on.

    Opt out with ``@pytest.mark.reads_live_book`` when a test genuinely means to
    read the production book -- a migration check or a census. The marker is
    explicit so those tests are greppable, instead of being swept into the
    pinned default and quietly changing meaning.
    """
    if request.node.get_closest_marker("reads_live_book"):
        return

    pinned = os.environ.get("TRADING_DB_PATH") or str(CACHE_ROOT / "trading_cache_test.db")
    for module_name, attr, cache_attrs in _PINNED_GATES:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - a gate that will not import cannot be read either
            continue
        current = getattr(module, attr, None)
        # Keep the declared type: these are pathlib.Path on all three gates and
        # the f-string URI below them formats either, but a test that asserts
        # on DB_PATH.name should not start seeing a str.
        monkeypatch.setattr(module, attr, type(current)(pinned) if isinstance(current, Path) else pinned)
        for cache_attr in cache_attrs:
            if not hasattr(module, cache_attr):
                continue
            existing = getattr(module, cache_attr)
            # monkeypatch restores the ORIGINAL object on teardown, so a fresh
            # empty container per test cannot leak either way.
            monkeypatch.setattr(module, cache_attr, type(existing)())


# pytest-django owns Django setup, test-environment setup, and database
# teardown. Keeping a second hand-rolled session fixture here caused Django's
# setup_test_environment() to run twice and prevented every ORM-backed suite
# from collecting.
