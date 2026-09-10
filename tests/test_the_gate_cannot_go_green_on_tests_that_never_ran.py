"""The pass gate printed OK on a run in which nothing ran.

Measured 2026-09-10 with the gate's own flags at the time (``-rf``, no
``--continue-on-collection-errors``) against a directory holding one
uncollectable file and one good one:

    pytest returncode : 2
    outcomes (named)  : {}
    passed            : 0
    failed            : 1
    regressions       : []          <- and this is what sets the verdict

``check()`` rejects only tests that are failing NOW and were not failing in the
snapshot. An empty ``outcomes`` is an empty regression set, so the gate printed
"OK -- nothing that was passing is broken." and exited 0. Two separate faults
produced that:

  1. ``-rf`` asks pytest for the failed short-summary only, so ERROR lines --
     which is how a file that cannot be COLLECTED is reported -- were omitted
     entirely and the broken file was never named.
  2. A collection error Interrupts the whole pytest session, so the good file
     in the same run never executed either. The gate was green on zero tests.

Two smaller ones in the same parser: ``(\\d+) (?:failed|error)`` stops at the
first alternation, so "1 failed, 2 errors" counted 1; and ``_targets()``
dropped GATE_TESTS entries with no file behind them, so renaming a gate test
silently removed it from the gate forever.

These tests use ``pass_gate.PYTEST_FLAGS`` and ``pass_gate._summarise``
directly. A copy of the flag list here would pass while the gate itself stayed
blind -- this repo has shipped exactly that shape before.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util as _iu

_spec = _iu.spec_from_file_location(
    "pass_gate_under_test", ROOT / "scripts" / "pass_gate.py")
pass_gate = _iu.module_from_spec(_spec)
_spec.loader.exec_module(pass_gate)


def _interpreter() -> str:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(venv) if venv.exists() else sys.executable


def _nested_pytest(target: Path):
    """Run pytest on ``target`` with the gate's OWN flags; return (text, rc).

    The inner run must not inherit the outer session's plugin/config state.
    DJANGO_SETTINGS_MODULE is the one that bites: pytest-django reads it from
    the environment, the child's rootdir is the temp directory rather than the
    repo, and the plugin then aborts at startup with "could not find a Django
    project" -- returncode 1, no summary line, no counts. That looks exactly
    like the blindness under test and would have made this test lie.
    """
    env = dict(os.environ)
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST",
                "DJANGO_SETTINGS_MODULE"):
        env.pop(key, None)
    out = subprocess.run(
        [_interpreter(), "-m", "pytest", str(target), "-p", "no:django",
         *pass_gate.PYTEST_FLAGS],
        cwd=str(ROOT), capture_output=True, text=True, env=env)
    return (out.stdout or "") + (out.stderr or ""), out.returncode


@pytest.fixture()
def one_broken_one_good(tmp_path: Path) -> Path:
    """A directory pytest cannot fully collect, plus a test that would pass."""
    (tmp_path / "test_uncollectable.py").write_text(
        "import a_module_that_is_not_installed_anywhere\n"
        "\n"
        "def test_never_reached():\n"
        "    assert True\n",
        encoding="utf-8")
    (tmp_path / "test_good.py").write_text(
        "def test_this_one_must_still_run():\n"
        "    assert True\n",
        encoding="utf-8")
    return tmp_path


def test_a_file_that_cannot_be_collected_is_named_as_a_failure(
        one_broken_one_good: Path) -> None:
    """The gate must NAME the broken file, not omit it.

    Against the pre-fix flags this fails: outcomes is {}, so the file that
    broke the gate is invisible and no regression is reported.
    """
    text, rc = _nested_pytest(one_broken_one_good)
    got = pass_gate._summarise(text, rc, [])

    # pytest names the uncollectable file under these flags. Under -rf it
    # printed no short-summary line for it at all.
    assert "test_uncollectable" in text, (
        "pytest did not name the uncollectable file with the gate's flags, "
        "which is what -rf did:\n%s" % text)
    # ...and the gate must turn that into a NAMED outcome, because an empty
    # outcomes dict is an empty regression set and therefore a green verdict.
    assert got["outcomes"], (
        "a file that failed to COLLECT produced no outcome, so it can never "
        "be reported as a regression: %r\n%s" % (got, text))
    assert got["failed"] >= 1, "%r\n%s" % (got, text)
    assert "-" not in got["outcomes"], (
        "a bare '-' names nothing and collapses every session-level error "
        "into one key: %r" % got)


def test_a_collection_error_does_not_stop_the_other_tests_running(
        one_broken_one_good: Path) -> None:
    """One uncollectable file used to Interrupt the whole session.

    Every other gate test then went unexecuted while the gate said OK. The
    good file in this directory must still run and be counted.
    """
    text, rc = _nested_pytest(one_broken_one_good)

    assert "Interrupted" not in text, (
        "collection error aborted the session, so the healthy tests in the "
        "same run never executed:\n%s" % text)
    got = pass_gate._summarise(text, rc, [])
    assert got["passed"] >= 1, (
        "test_good.py did not run beside the uncollectable file: %r\n%s"
        % (got, text))


def test_an_unusable_returncode_is_not_a_passing_run() -> None:
    """returncode 2/3/4 leave the counts blank; that is not evidence of a pass.

    ``ran`` False makes check() print REJECTED and exit 1. It previously
    printed INCONCLUSIVE and exited 0.
    """
    interrupted = pass_gate._summarise(
        "!!! Interrupted: 1 error during collection !!!\n1 error in 10.03s\n",
        2, [])
    assert interrupted["ran"] is False

    usage_error = pass_gate._summarise("ERROR: file or directory not found\n",
                                       4, [])
    assert usage_error["ran"] is False

    real = pass_gate._summarise("11 passed, 44 warnings in 12.23s\n", 0, [])
    assert real["ran"] is True
    assert real["passed"] == 11


def test_failures_and_errors_are_both_counted() -> None:
    """One alternation stopped at "failed" and never read the error count."""
    got = pass_gate._summarise("1 failed, 2 errors, 30 passed in 4.00s\n",
                               1, [])
    assert got["failed"] == 3, (
        "failed+errors must be summed; a single alternation reported 1 when "
        "three files were broken: %r" % got)
    assert got["passed"] == 30


def test_a_gate_test_with_no_file_is_a_failure_not_a_shorter_list() -> None:
    """Renaming a gate test used to remove it from the gate silently."""
    got = pass_gate._summarise("50 passed in 9.00s\n", 0,
                               ["test_that_was_renamed_away.py"])
    named = " ".join(got["outcomes"])
    assert "test_that_was_renamed_away.py" in named
    assert "MISSING FILE" in named
    assert got["failed"] == 1, (
        "a GATE_TESTS entry with no file behind it must count as a failure, "
        "otherwise a rename is indistinguishable from a pass: %r" % got)


def test_every_gate_test_entry_still_has_a_file() -> None:
    """The gate's own list, checked against the tree it claims to test."""
    missing = pass_gate._missing_targets()
    assert missing == [], (
        "GATE_TESTS names files that do not exist, so the gate is not running "
        "them: %s" % missing)
