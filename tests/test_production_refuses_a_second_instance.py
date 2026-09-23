"""The ``start_production`` singleton, measured rather than asserted.

Measured p679/p680: four production PIDs alive in two independent parent-child
pairs nine minutes apart, because SEVEN separate call sites launch
``main.py --action start_production`` and each carries its own non-atomic
"is it already running" cmdline scan. Two concurrent instances make every zero
this repo measures -- ghost entries, refusals, tick counts -- unfalsifiable.

Three things have to hold, and the third is the one that can take production
down if it is wrong:

  1. A production process outside our tree is REFUSED.
  2. Two processes cannot both hold the lease (the scan cannot see a process
     that has not spawned yet; the lease is what closes that race).
  3. A CHILD of a guarded instance INHERITS, it does not refuse. The live
     instance runs as a parent/child pair with an identical command line
     (pid 3584 -> pid 20452, same second, same argv), so a guard that is not
     tree-aware would make production refuse itself.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

psutil = pytest.importorskip("psutil")

import main  # noqa: E402
from services.guardian_lock import GuardianLease  # noqa: E402


def test_a_production_cmdline_is_recognised():
    assert main._production_cmdline(
        [sys.executable, "-X", "utf8", "-u", "main.py", "--action", "start_production", "--stay-alive"]
    )
    assert not main._production_cmdline([sys.executable, "-m", "pytest"])
    assert not main._production_cmdline([sys.executable, "main.py", "--action", "balances"])


def test_the_lease_is_mutually_exclusive_across_processes():
    """The scan cannot see a process that has not spawned yet; the lease can."""
    name = "production_singleton_test_mutex"
    first = GuardianLease(name, timeout=0)
    assert first.acquire(), "first lease should be free"
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, r'%s')\n"
                "from services.guardian_lock import GuardianLease\n"
                "print('ACQUIRED' if GuardianLease('%s', timeout=0).acquire() else 'REFUSED')"
                % (str(REPO), name),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO),
        )
        assert "REFUSED" in probe.stdout, (
            "a second process acquired the same lease -- the singleton is not "
            f"cross-process. stdout={probe.stdout!r} stderr={probe.stderr[-400:]!r}"
        )
    finally:
        first.release()

    # and it is genuinely released, not leaked, or the next real boot deadlocks
    again = GuardianLease(name, timeout=0)
    assert again.acquire(), "lease was not released"
    again.release()


def _run_guard_in_child(argv_marker: list[str]) -> subprocess.CompletedProcess:
    """Run the guard in a grandchild whose PARENT's argv is ``argv_marker``."""
    grandchild = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import main\n"
        "print('ANCESTOR=%%r' %% (main._production_ancestor_pid(),))\n"
        "print('INHERITED' if main._acquire_production_singleton() else 'REFUSED')\n"
        % (str(REPO),)
    )
    parent = (
        "import subprocess, sys\n"
        "r = subprocess.run([sys.executable, '-c', %r], capture_output=True, text=True, cwd=r'%s')\n"
        "sys.stdout.write(r.stdout)\n"
        "sys.stderr.write(r.stderr)\n" % (grandchild, str(REPO))
    )
    return subprocess.run(
        [sys.executable, "-c", parent] + argv_marker,
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(REPO),
    )


def test_a_child_of_a_production_process_inherits_rather_than_refusing():
    """The guard must not make production refuse its own re-spawned child."""
    result = _run_guard_in_child(["main.py", "--action", "start_production", "--stay-alive"])
    assert "INHERITED" in result.stdout, (
        "a child of a start_production process did NOT inherit the singleton. "
        "Shipping this would take production down on its next boot. "
        f"stdout={result.stdout!r} stderr={result.stderr[-600:]!r}"
    )


def test_the_guard_is_wired_into_the_start_production_branch():
    """A guard nothing calls is not a guard."""
    source = (REPO / "main.py").read_text(encoding="utf-8")
    branch = source.split('if action == "start_production":', 1)
    assert len(branch) == 2, "the start_production branch moved"
    head = branch[1][:600]
    assert "_acquire_production_singleton()" in head, (
        "start_production does not call the singleton guard before starting"
    )
    assert "return" in head, "the guard's refusal must return without starting"
