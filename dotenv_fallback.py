from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

_FLAG = "ALLOW_DOTENV_LOADING"


def _dotenv_enabled() -> bool:
    value = os.environ.get(_FLAG, "")
    return value.lower() in {"1", "true", "yes", "on"}


try:  # pragma: no cover - prefer real package when available
    from dotenv import load_dotenv as _real_load_dotenv  # type: ignore
    from dotenv import find_dotenv as _real_find_dotenv  # type: ignore
    from dotenv import dotenv_values as _real_dotenv_values  # type: ignore
except Exception:  # pragma: no cover - fallback used when python-dotenv missing

    def _parse_dotenv(path: Path) -> Dict[str, str]:
        result: Dict[str, str] = {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key:
                        result[key] = value
        except Exception:
            return {}
        return result

    def _real_dotenv_values(path: str | os.PathLike[str], stream=None) -> Dict[str, str]:
        path_obj = Path(path)
        if not path_obj.exists():
            return {}
        return _parse_dotenv(path_obj)

    def _real_find_dotenv(usecwd: bool = False) -> str:
        candidates = []
        if usecwd:
            candidates.append(Path.cwd() / ".env")
        if sys.argv and sys.argv[0]:
            candidates.append(Path(sys.argv[0]).resolve().parent / ".env")
        try:
            candidates.append(Path(__file__).resolve().parent / ".env")
        except Exception:
            pass
        candidates.append(Path.home() / ".env")
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return str(candidate)
            except Exception:
                continue
        return ""

    def _real_load_dotenv(path: str | os.PathLike[str] | None = None, override: bool = False) -> bool:
        target = Path(path) if path else Path(_real_find_dotenv())
        if not target or not target.exists():
            return False
        values = _real_dotenv_values(target)
        for key, value in values.items():
            if override or key not in os.environ:
                os.environ[key] = value
        return True


def find_dotenv(usecwd: bool = False) -> str:
    if not _dotenv_enabled():
        return ""
    return _real_find_dotenv(usecwd=usecwd)


def dotenv_values(path: str | os.PathLike[str], stream=None) -> Dict[str, str]:
    if not _dotenv_enabled():
        return {}
    # Use positional args so this works for both python-dotenv and the fallback shim.
    return _real_dotenv_values(path, stream=stream)


def load_dotenv(path: str | os.PathLike[str] | None = None, override: bool = False) -> bool:
    if not _dotenv_enabled():
        return False
    # Use positional args so this works for both python-dotenv and the fallback shim.
    return _real_load_dotenv(path, override=override)
