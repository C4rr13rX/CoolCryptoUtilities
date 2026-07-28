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
