from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from django.test import SimpleTestCase

from services.branddozer_lifecycle import DEV_BRANCH, STABLE_BRANCH, ensure_lifecycle, finalize_cycle, prepare_cycle


class BrandDozerLifecycleTests(SimpleTestCase):
    def project(self, root: Path, **updates):
        value = {
            "id": "test-project",
            "name": "Test Project",
            "root_path": str(root),
            "default_prompt": "Continuously improve the test project",
            "workflow_kind": "generic",
            "workflow_config": {"mission": "Test the generic lifecycle"},
            "license_key": "unlicensed",
            "git_auto_promote": True,
        }
        value.update(updates)
        return value

    def test_successful_cycle_promotes_development_to_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "index.html").write_text("baseline", encoding="utf-8")
            project = self.project(root)
            ensure_lifecycle(project)
            prepare_cycle(project)
            (root / "index.html").write_text("working version", encoding="utf-8")
            result = finalize_cycle(project, success=True, message="validated update")
            self.assertTrue(result["promoted"])
            self.assertEqual(result["git"]["current_branch"], DEV_BRANCH)
            main_text = subprocess.run(
                ["git", "show", f"{STABLE_BRANCH}:index.html"], cwd=root,
                capture_output=True, text=True, check=True,
            ).stdout
            self.assertEqual(main_text, "working version")

    def test_failed_cycle_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "index.html").write_text("stable", encoding="utf-8")
            project = self.project(root)
            ensure_lifecycle(project)
            prepare_cycle(project)
            (root / "index.html").write_text("broken", encoding="utf-8")
            result = finalize_cycle(project, success=False, message="failed update")
            self.assertFalse(result["promoted"])
            stable = subprocess.run(
                ["git", "show", f"{STABLE_BRANCH}:index.html"], cwd=root,
                capture_output=True, text=True, check=True,
            ).stdout
            self.assertEqual(stable, "stable")

    def test_existing_license_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "LICENSE").write_text("Existing custom license", encoding="utf-8")
            ensure_lifecycle(self.project(root, license_key="mit"))
            self.assertEqual((root / "LICENSE").read_text(encoding="utf-8"), "Existing custom license")
