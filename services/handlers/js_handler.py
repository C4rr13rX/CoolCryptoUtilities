from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import json

from services.handlers.base import FrameworkHandler, ProjectProfile


class GenericJSHandler(FrameworkHandler):
    name = "js"

    def discover(self, root: Path) -> Optional[ProjectProfile]:
        pkg = root / "package.json"
        if not pkg.exists():
            return None
        try:
            payload = json.loads(pkg.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        deps = payload.get("dependencies") or {}
        dev = payload.get("devDependencies") or {}
        framework = "generic"
        if "react" in deps or "react" in dev:
            framework = "react"
        if "vue" in deps or "vue" in dev:
            framework = "vue"
        if "svelte" in deps or "svelte" in dev:
            framework = "svelte"
        return ProjectProfile(
            root=root,
            language="javascript",
            framework=framework,
            test_runner="npm",
            lint=None,
            security=None,
            entrypoint=None,
        )

    def tests_for_path(self, profile: ProjectProfile, path: Optional[Path]) -> List[str]:
        cmds: List[str] = []
        if not (profile.root / "package.json").exists():
            return cmds
        if path and path.suffix.lower() in {".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte"}:
            # Try vitest or jest via npx
            cmds.append(f'npx vitest run "{path}"')
            cmds.append(f'npx jest "{path}"')
            return cmds
        cmds.append("npm test -- --watch=false")
        return cmds

    def full_tests(self, profile: ProjectProfile) -> List[str]:
        return ["npm test -- --watch=false"]
