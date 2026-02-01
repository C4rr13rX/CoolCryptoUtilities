from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from services.handlers.base import FrameworkHandler, ProjectProfile
from services.handlers.python_handler import GenericPythonHandler
from services.handlers.js_handler import GenericJSHandler


def _handlers() -> List[FrameworkHandler]:
    return [
        GenericPythonHandler(),
        GenericJSHandler(),
    ]


def detect_profile(root: Path) -> Optional[Tuple[FrameworkHandler, ProjectProfile]]:
    for handler in _handlers():
        profile = handler.discover(root)
        if profile:
            return handler, profile
    return None
