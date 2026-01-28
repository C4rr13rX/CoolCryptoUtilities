from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_runtime_dirs() -> None:
    paths = [
        PROJECT_ROOT / "runtime" / "branddozer" / "solo_plans",
        PROJECT_ROOT / "runtime" / "branddozer" / "transcripts",
        PROJECT_ROOT / "runtime" / "branddozer" / "sessions",
    ]
    for path in paths:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def main() -> int:
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")
    _ensure_runtime_dirs()
    runpy.run_path(str(PROJECT_ROOT / "bin" / "brandozer"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
