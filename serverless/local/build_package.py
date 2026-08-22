#!/usr/bin/env python3
"""
Build the Lambda deployment bundle.

Produces ``dist/coolcrypto-lambda.zip`` containing the Django project, the
handlers, the services layer, and the pure-Python dependencies.

Two things this deliberately does NOT bundle:

* **The ML/vision stack** (tensorflow, opencv, matplotlib, kuzu, web3,
  torch-adjacent packages).  They blow past Lambda's 250 MB unzipped limit and
  are only reached by the trading/ML workers, which do not run in the request
  path.  A scheduled task that genuinely needs them belongs in a
  container-image Lambda or Fargate, not in this zip.  pandas and numpy *are*
  included despite their size: the telemetry and wallet views import them
  while the root URLconf is being built, so the site 500s without them.
* **pyarrow.**  It is 136 MB, which alone pushes the bundle past the 250 MB
  ceiling.  Nothing in the request path needs it: the market handler presigns
  S3 URLs and the browser parses the Parquet with hyparquet.  Only
  ``migrate_market.py`` reads Parquet in Python, and that runs on a
  workstation, not in Lambda.
* **The virtualenv itself.**  Dependencies are resolved fresh for the Lambda
  platform so Windows-built wheels never leak into a Linux runtime -- this is
  the single most common cause of "works locally, ImportError on Lambda".
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"
BUILD = DIST / "build"
ZIP_PATH = DIST / "coolcrypto-lambda.zip"

# Runtime dependencies that must exist inside the bundle. Kept separate from
# requirements.txt because that file pulls the full ML/trading stack.
LAMBDA_REQUIREMENTS = [
    "Django>=5.0,<6.1",
    "djangorestframework>=3.14",
    "django-cors-headers>=4.3",
    "mangum>=0.17",
    "whitenoise>=6.6",
    "psycopg[binary]>=3.1",
    "django-storages>=1.14",
    "boto3>=1.28",
    "python-dotenv>=1.0",
    "requests>=2.31",
    # services.secure_settings decrypts SecureVault entries with AES-GCM; the
    # cron handler reaches it through internal_cron -> secure_settings.
    "cryptography>=3.4.8",
    # Reached while importing the root URLconf:
    #   investigations.views -> services.web_research      -> bs4
    #   telemetry.views      -> services.wallet_*/trading  -> pandas
    #   streams/discovery    -> services.onchain_feed      -> aiohttp
    #   services.polite_news_crawler                       -> async_timeout
    "beautifulsoup4>=4.12",
    "pandas>=2.1",
    "aiohttp>=3.9",
    "async-timeout>=4.0.3",
    #   branddozer.views     -> services.branddozer_ai       -> openai
    #   services.news_ingestor                               -> feedparser
    #   services.branddozer_delivery                         -> networkx, yaml
    #   services.adaptive_control                            -> psutil
    "openai>=1.0",
    "feedparser>=6.0.11",
    "networkx>=3.0",
    "PyYAML>=6.0",
    "psutil>=5.9",
    # Quantum-safe auth: Argon2id verifier + ML-KEM-768 encapsulation.
    # kyber-py is pure Python, so it needs no platform wheel.
    "argon2-cffi>=23.1",
    "kyber-py>=1.0.1",
]

# Project source that ships with the bundle.
# config/ ships too: services.cron_profile reads config/cron_profile.json for
# the task schedule, and the bundle is the only place a Lambda can find it.
SOURCE_DIRS = [
    "web", "services", "serverless", "config",
    # Local packages the Django apps import: telemetry/opsconsole views reach
    # into trading.*, and the guardian/ops panels import monitoring_guardian
    # and tools. Missing these fails at request time, not at build time.
    "trading", "monitoring_guardian", "tools",
]

# Top-level .py modules at the repo root that the apps import by bare name.
# Note `cache`: several PyPI packages install a module of the same name, which
# would shadow this one and surface as a confusing ImportError deep in a
# request. Keep the bundle free of such collisions.
ROOT_MODULES = [
    "db", "cache", "balances", "router_wallet", "filter_scams",
    "token_decimals", "dotenv_fallback", "production",
]

EXCLUDE_PARTS = {
    "__pycache__", ".git", ".venv", "node_modules", ".pytest_cache",
    "collected_static", "frontend", "codex_transcripts", "logs",
    "runtime", "storage", "data", "tests", "benchmarks",
    # 229 MB of compiled Rust/native build output under tools/c0d3rV2. Nothing
    # in the request path imports it (the Python there is ~1 MB), and including
    # it alone pushes the bundle past Lambda's 250 MB unzipped ceiling.
    "native_os_service",
    "target", "dist-info-cache",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log", ".sqlite3", ".zip"}


def _keep(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    return path.suffix not in EXCLUDE_SUFFIXES


def install_dependencies(target: Path, platform: str) -> None:
    """Resolve dependencies for the *Lambda* platform, not the build host."""
    print(f"[deps] installing for {platform} -> {target}")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--target", str(target),
        "--platform", platform,
        "--implementation", "cp",
        "--python-version", "3.12",
        # Lambda ships no compiler, so a source build would produce a package
        # that cannot import. Force wheels and fail loudly instead.
        "--only-binary=:all:",
        "--upgrade",
        *LAMBDA_REQUIREMENTS,
    ]
    subprocess.run(cmd, check=True)


def copy_sources(target: Path) -> None:
    for name in SOURCE_DIRS:
        src = ROOT / name
        if not src.exists():
            print(f"[src] skip missing {name}")
            continue
        for path in src.rglob("*"):
            if not path.is_file() or not _keep(path.relative_to(ROOT)):
                continue
            dest = target / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        print(f"[src] copied {name}")

    # Root-level modules that services/ and the Django apps import as
    # top-level names (e.g. `from db import get_db`, `from cache import
    # CacheBalances`). Without these the URLconf import fails at request time,
    # not at build time, so the list is derived from the actual import graph.
    for extra in ROOT_MODULES:
        p = ROOT / f"{extra}.py"
        if p.exists():
            shutil.copy2(p, target / f"{extra}.py")
            print(f"[src] copied {extra}.py")
        else:
            print(f"[warn] root module missing: {extra}.py")


def collect_static(target: Path) -> None:
    """
    Run collectstatic into the bundle.

    WhiteNoise serves assets straight out of STATIC_ROOT inside the bundle, so
    without this the function boots with "No directory at: .../collected_static"
    and every /static/ request 404s. It runs against the build tree rather than
    the repo so the bundled copy is what gets hashed and compressed.
    """
    import os

    env = dict(
        os.environ,
        DJANGO_SETTINGS_MODULE="coolcrypto_dashboard.settings_lambda",
        # collectstatic touches neither the DB nor AWS; sqlite keeps it from
        # trying to reach a Postgres that may not be running at build time.
        DJANGO_DB_VENDOR="sqlite",
        ALLOW_SQLITE_FALLBACK="1",
        SECURE_ENV_HYDRATED="1",
        DJANGO_SECRET_KEY="build-time-only",
        PYTHONPATH=os.pathsep.join([str(target), str(target / "web")]),
    )
    print("[static] collectstatic")
    try:
        subprocess.run(
            [sys.executable, str(target / "web" / "manage.py"),
             "collectstatic", "--noinput", "--clear"],
            cwd=str(target / "web"), env=env, check=True,
            capture_output=True, text=True,
        )
        print("[static] done")
    except subprocess.CalledProcessError as exc:
        # Not fatal: the API still works, only /static/ assets are missing.
        print(f"[static] SKIPPED -- collectstatic failed:\n{exc.stderr[-800:]}")


def prune(build_dir: Path) -> None:
    """
    Strip what Lambda never executes.

    The 250 MB unzipped ceiling counts every byte in the bundle, and the
    scientific wheels spend most of their size on things a running function
    does not touch: vendored test suites (pandas alone ships ~30 MB of them),
    bundled headers used only when compiling against the library, and the
    dist-info metadata for packages nothing introspects at runtime.
    """
    freed = 0

    def _rm(path: Path) -> None:
        nonlocal freed
        try:
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            shutil.rmtree(path, ignore_errors=True)
            freed += size
        except OSError:
            pass

    for pkg in ("pandas", "numpy", "scipy", "networkx", "openai", "botocore"):
        root = build_dir / pkg
        if not root.exists():
            continue
        for d in list(root.rglob("tests")) + list(root.rglob("test")):
            if d.is_dir():
                _rm(d)
    # Header files are only needed to build against numpy/pandas, never to run.
    for inc in build_dir.rglob("include"):
        if inc.is_dir() and inc.parent.name in {"numpy", "pandas"}:
            _rm(inc)
    # Stale bytecode from the source copy; Lambda regenerates what it needs.
    for d in list(build_dir.rglob("__pycache__")):
        if d.is_dir():
            _rm(d)
    for f in build_dir.rglob("*.pyc"):
        try:
            freed += f.stat().st_size
            f.unlink()
        except OSError:
            pass

    print(f"[prune] freed {freed / 1_048_576:.1f} MB")


def make_zip(build_dir: Path, zip_path: Path) -> None:
    print(f"[zip] writing {zip_path}")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(build_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(build_dir).as_posix())
    size_mb = zip_path.stat().st_size / 1_048_576
    print(f"[zip] done: {size_mb:.1f} MB")
    if size_mb > 50:
        # deploy_local.sh already publishes through S3, which is the supported
        # path above this limit. Noted so the number is not a surprise.
        print("[note] >50 MB: too large for direct upload; deploying via S3.")
    if size_mb > 250:
        print("[warn] >250 MB unzipped is Lambda's hard ceiling -- "
              "move heavy deps to a layer or a container image.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="manylinux2014_x86_64")
    ap.add_argument("--skip-deps", action="store_true",
                    help="reuse the dependencies already in dist/build")
    args = ap.parse_args()

    DIST.mkdir(exist_ok=True)
    if BUILD.exists() and not args.skip_deps:
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True, exist_ok=True)

    if not args.skip_deps:
        install_dependencies(BUILD, args.platform)
    copy_sources(BUILD)
    collect_static(BUILD)
    prune(BUILD)
    make_zip(BUILD, ZIP_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
