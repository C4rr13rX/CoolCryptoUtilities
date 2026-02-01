from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ProjectProfile:
    root: Path
    language: str
    framework: str
    test_runner: Optional[str]
    lint: Optional[str]
    security: Optional[str]
    entrypoint: Optional[str]


class FrameworkHandler:
    name = "generic"

    def discover(self, root: Path) -> Optional[ProjectProfile]:
        raise NotImplementedError

    def tests_for_path(self, profile: ProjectProfile, path: Optional[Path]) -> List[str]:
        return []

    def full_tests(self, profile: ProjectProfile) -> List[str]:
        return []

    def lint(self, profile: ProjectProfile) -> List[str]:
        return []

    def security(self, profile: ProjectProfile) -> List[str]:
        return []
