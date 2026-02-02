from __future__ import annotations

from pathlib import Path
from typing import List


_FRAMEWORK_MARKERS = {
    "django": ["manage.py", "pyproject.toml", "requirements.txt"],
    "flask": ["app.py", "requirements.txt"],
    "fastapi": ["main.py", "pyproject.toml", "requirements.txt"],
    "vite": ["vite.config.js", "vite.config.ts"],
    "vue": ["vue.config.js", "src/main.ts", "src/main.js"],
    "react": ["src/index.tsx", "src/index.jsx"],
    "svelte": ["svelte.config.js", "src/App.svelte"],
    "nextjs": ["next.config.js", "app/page.tsx", "pages/index.tsx"],
    "nuxt": ["nuxt.config.ts", "nuxt.config.js"],
    "angular": ["angular.json"],
    "kivy": ["main.py", "buildozer.spec"],
}


def detect_frameworks(root: Path) -> List[str]:
    root = Path(root)
    hits: List[str] = []
    for name, markers in _FRAMEWORK_MARKERS.items():
        for marker in markers:
            if (root / marker).exists():
                hits.append(name)
                break
    # If package.json exists, hint at JS framework even if markers not found.
    if (root / "package.json").exists():
        if "javascript" not in hits:
            hits.append("javascript")
    return hits


__all__ = ["detect_frameworks"]
