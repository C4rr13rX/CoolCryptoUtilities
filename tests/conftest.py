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


def pytest_configure():
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


# pytest-django owns Django setup, test-environment setup, and database
# teardown. Keeping a second hand-rolled session fixture here caused Django's
# setup_test_environment() to run twice and prevented every ORM-backed suite
# from collecting.
