from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from services.handlers.base import FrameworkHandler, ProjectProfile


class GenericPythonHandler(FrameworkHandler):
    name = "python"

    def discover(self, root: Path) -> Optional[ProjectProfile]:
        if (root / "pyproject.toml").exists() or (root / "requirements.txt").exists():
            entry = "main.py" if (root / "main.py").exists() else None
            return ProjectProfile(
                root=root,
                language="python",
                framework="generic",
                test_runner="pytest" if (root / "tests").exists() else None,
                lint="ruff",
                security="bandit",
                entrypoint=entry,
            )
        return None

    def tests_for_path(self, profile: ProjectProfile, path: Optional[Path]) -> List[str]:
        cmds: List[str] = []
        if profile.test_runner != "pytest":
            return cmds
        cmd = "python -m pytest"
        if path:
            rel = None
            try:
                rel = path.resolve().relative_to(profile.root.resolve())
            except Exception:
                rel = None
            if rel:
                candidate = profile.root / "tests" / rel.name
                if candidate.exists():
                    cmd = f'{cmd} "{candidate}"'
                else:
                    cmd = f'{cmd} -k "{rel.stem}"'
        cmds.append(cmd)
        return cmds

    def full_tests(self, profile: ProjectProfile) -> List[str]:
        if profile.test_runner == "pytest":
            return ["python -m pytest"]
        return []

    def lint(self, profile: ProjectProfile) -> List[str]:
        return ["python -m ruff ."] if profile.lint else []

    def security(self, profile: ProjectProfile) -> List[str]:
        return ["python -m bandit -r ."] if profile.security else []
