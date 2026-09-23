"""The end-of-pass census that stops a pass leaving shipped work on the floor.

The defect has recurred three times (pass 114, 458, 680) and the mechanism is
not a mis-scoped `git add`: it is that `git commit -- <paths>` succeeds and
says NOTHING about the path you forgot to name. A successful commit reads as
shipped work, and the file stays invisible to every other agent sharing this
worktree.

AGE IS THE WHOLE POINT, so it is what these tests exercise. An untracked file
written minutes ago is normal work in progress and must PASS; one older than a
day is a previous pass's rumour and must FAIL. A census that flags both is
noise, and noise gets ignored -- which is how this recurred.
"""

from __future__ import annotations

import os
import subprocess
import time

import pytest

from scripts.uncommitted_work_census import census, main, verdict


def _worktree(tmp_path):
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "scripts").mkdir()
    for args in (
        ["init", "-q"],
        ["config", "user.email", "a@b.c"],
        ["config", "user.name", "t"],
    ):
        subprocess.run(["git", *args], cwd=root, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return root


def _write(root, rel: str, age_days: float) -> None:
    p = root / rel
    p.write_text("# work\n", encoding="utf-8")
    stamp = time.time() - age_days * 86400.0
    os.utime(p, (stamp, stamp))


class TestAgeSeparatesWorkInProgressFromARumour:
    def test_a_file_written_this_pass_is_not_a_failure(self, tmp_path):
        """Otherwise every pass fails its own census and stops reading it."""
        root = _worktree(tmp_path)
        _write(root, "tests/test_fresh.py", age_days=0.0)

        result = census(str(root))
        assert [r["path"] for r in result["untracked"]] == ["tests/test_fresh.py"]
        assert verdict(result, max_age_days=1.0) == []
        assert main(["--root", str(root)]) == 0

    def test_a_week_old_untracked_test_fails_the_census(self, tmp_path):
        """The pass-114 case: seven money-path tests untracked for a week."""
        root = _worktree(tmp_path)
        _write(root, "tests/test_fresh.py", age_days=0.0)
        _write(root, "tests/test_stale.py", age_days=7.0)

        stale = verdict(census(str(root)), max_age_days=1.0)
        assert [r["path"] for r in stale] == ["tests/test_stale.py"], (
            "the fresh file must not be flagged alongside the stale one")
        assert stale[0]["age_days"] == pytest.approx(7.0, abs=0.05)
        assert main(["--root", str(root)]) == 1

    def test_committing_the_stale_file_clears_the_census(self, tmp_path):
        """A census that cannot be satisfied is a census nobody acts on."""
        root = _worktree(tmp_path)
        _write(root, "tests/test_stale.py", age_days=7.0)
        assert main(["--root", str(root)]) == 1

        subprocess.run(["git", "add", "tests/test_stale.py"], cwd=root,
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "commit", "-qm", "land it", "--",
                        "tests/test_stale.py"], cwd=root, check=True,
                       stdout=subprocess.DEVNULL)

        assert main(["--root", str(root)]) == 0
        assert census(str(root))["untracked"] == []


class TestTheTwoFailureModesAreReportedSeparately:
    def test_a_tracked_file_with_edits_is_modified_not_untracked(self, tmp_path):
        """Different danger, different repair.

        An untracked file is invisible to the other agents and dies to
        `git clean`. A modified tracked file is visible in `git status` and
        recoverable. Pooling them would make the dangerous case unreadable.
        """
        root = _worktree(tmp_path)
        _write(root, "scripts/tool.py", age_days=7.0)
        subprocess.run(["git", "add", "scripts/tool.py"], cwd=root, check=True,
                       stdout=subprocess.DEVNULL)
        subprocess.run(["git", "commit", "-qm", "base", "--",
                        "scripts/tool.py"], cwd=root, check=True,
                       stdout=subprocess.DEVNULL)
        _write(root, "scripts/tool.py", age_days=7.0)   # edit, do not commit
        (root / "scripts" / "tool.py").write_text("# edited\n", encoding="utf-8")

        result = census(str(root))
        assert result["untracked"] == []
        assert [r["path"] for r in result["modified"]] == ["scripts/tool.py"]
        # Modified files are REPORTED but do not fail: the scoping rule that
        # causes this defect also forbids sweeping another agent's in-flight
        # edits into a commit, and mid-pass edits are exactly that.
        assert main(["--root", str(root)]) == 0


class TestOnlyCodeCounts:
    def test_data_and_backup_artifacts_are_not_shipped_work(self, tmp_path):
        """This worktree carries ~30 untracked data/*.bak paths at all times.

        Counting them would bury the two real orphans in noise, which is the
        failure mode this census exists to prevent.
        """
        root = _worktree(tmp_path)
        (root / "data").mkdir()
        _write(root, "data/ledger.json.bak-something", age_days=30.0)
        _write(root, "tests/test_real.py", age_days=30.0)

        assert [r["path"] for r in census(str(root))["untracked"]] == [
            "tests/test_real.py"]
