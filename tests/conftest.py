import os
from pathlib import Path


def pytest_configure():
    cache_root = Path(os.getenv("PYTEST_CACHE_HOME", Path.cwd() / "runtime" / "pytest_cache"))
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
    os.environ["DJANGO_DB_VENDOR"] = "sqlite"
    os.environ["TRADING_DB_VENDOR"] = "sqlite"
    os.environ["ALLOW_SQLITE_FALLBACK"] = "1"
