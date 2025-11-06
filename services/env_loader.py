from __future__ import annotations
import os, sys
from pathlib import Path
from typing import List, Optional
from dotenv_fallback import load_dotenv, dotenv_values, find_dotenv

class EnvLoader:
    """Robust .env loader you can import anywhere."""
    @staticmethod
    def load() -> None:
        repo_root = Path(__file__).resolve().parents[1]
        web_dir = repo_root / "web"
        for path in (repo_root, web_dir):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

        # 1) try an auto-discovered .env in current working dir
        path = find_dotenv(usecwd=True)
        if path:
            try:
                load_dotenv(path, override=False)
                return
            except Exception:
                pass

        # 2) common fallbacks
        cands: List[Path] = []
        for p in (
            Path.cwd() / ".env",
            Path(sys.argv[0]).resolve().parent / ".env" if sys.argv and sys.argv[0] else None,
            Path(__file__).resolve().parent / ".env" if "__file__" in globals() else None,
            Path.home() / ".env",
        ):
            if p:
                cands.append(p)

        # 2a) load_dotenv on first readable candidate
        for p in cands:
            try:
                if p.is_file():
                    load_dotenv(p, override=False)
                    return
            except Exception:
                pass

        # 2b) last-resort: parse and set os.environ without load_dotenv
        for p in cands:
            try:
                if p.is_file():
                    for k, v in (dotenv_values(p) or {}).items():
                        os.environ.setdefault(k, v or "")
                    return
            except Exception:
                pass
