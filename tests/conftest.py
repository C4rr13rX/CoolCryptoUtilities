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


@pytest.fixture(scope="session", autouse=True)
def django_test_environment():
    """
    Stand up a Django test environment so DRF APITestCase suites run under pytest
    without needing pytest-django. Mirrors Django's DiscoverRunner setup/teardown.
    """
    if not os.environ.get("DJANGO_SETTINGS_MODULE"):
        os.environ["DJANGO_SETTINGS_MODULE"] = "coolcrypto_dashboard.settings"

    import django
    from django.test.utils import (
        setup_databases,
        setup_test_environment,
        teardown_databases,
        teardown_test_environment,
    )

    django.setup()
    setup_test_environment()
    old_config = None
    try:
        old_config = setup_databases(verbosity=0, interactive=False, keepdb=False)
    except RuntimeError as exc:
        # pytest-django blocks DB access unless explicitly marked; fallback to a
        # DB-less environment for unit tests that do not hit the ORM.
        if "Database access not allowed" not in str(exc):
            raise
    try:
        yield
    finally:
        if old_config:
            teardown_databases(old_config, verbosity=0)
        teardown_test_environment()
