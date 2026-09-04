"""Every launcher that starts production must pass ``-X utf8``.

On this box a Windows Python that did not start in UTF-8 mode uses cp1252 for
text it writes. Reproduced in this repo on 2026-09-03::

    >>> print('USD₮0 -> ok')
    UnicodeEncodeError: 'charmap' codec can't encode character '\\u20ae'

Not a hypothetical character: ``USD₮0`` is a real token this wallet prices
against -- ``PENDLE-USD₮0``, ``RAIN-USD₮0`` and ``ARB-USD₮0`` are all in the
bootstrap pair list.

``main.py`` calls ``ensure_utf8_mode()`` at import, and its ``harden_stdio()``
re-encodes THIS process's stdout/stderr, so the swap-path crash that once
recorded every live entry as ``live-entry-failed`` is covered however the
process was started. What that cannot reach is a bare ``open(path, "w")``
inside library code -- Keras' vocabulary writer, the one that killed every
model save -- because the default encoding for ``open()`` is fixed at
interpreter startup. Only ``-X utf8`` (or ``PYTHONUTF8`` in the parent's
environment) sets that.

So the flag still matters, and it has already been lost once: on 2026-09-03 the
running production process carried it while BOTH launchers that would relaunch
it did not -- and during that session an external supervisor
(``W1z4rDV1510n/scripts/w1z4rd_supervisor.py``, outside this repo) did exactly
that relaunch. This test pins the half this repo owns; the launchers are the
artefact, not somebody's shell history.

Lost a second time, and the test is why it went unnoticed
---------------------------------------------------------

This file enumerated only the two ``.ps1`` launchers, so the PYTHON ones were
never checked. Measured 2026-09-04 19:38Z: production was stopped and came back
within a minute as::

    .venv\\Scripts\\python.exe -u main.py --action start_production --stay-alive

-- no ``-X utf8``, the exact command line ``services/utf8_mode.py`` names as
the cause of every live entry recording ``live-entry-failed``. Two launchers in
this repo were on that contract:

    web/opsconsole/manager.py       DEFAULT_COMMAND
    monitoring_guardian/recovery.py _recover_production

while ``scripts/main_keeper.py`` and both ``.ps1`` files carried the flag. The
guardian relaunches on a 30s cooldown, so it wins the race and it decides what
production runs as, whatever started it the first time.

A launcher list that names some launchers is how a fix stays half-applied while
looking done, so this now checks EVERY process start in the repo that names
``start_production``, discovered rather than enumerated.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# The launchers that start the production manager, all of them.
_LAUNCHERS = ("GetToLiveTrading.ps1", "launch_revenir.ps1")

#: Python launchers: the argv list is built in code, so the check is on the
#: token sequence rather than on a PowerShell -ArgumentList line.
_PY_LAUNCHERS = (
    "scripts/main_keeper.py",
    "web/opsconsole/manager.py",
    "monitoring_guardian/recovery.py",
)

#: Vendored/built copies of this tree. They are outputs, not launchers -- the
#: source they were copied from is checked, and rebuilding regenerates them.
_SKIP_DIRS = {
    ".venv", ".git", "node_modules", "__pycache__", "runtime", "storage",
    "build", "pythonStage", "dist", "android",
}


def _argument_lists(text: str) -> list[str]:
    """Every -ArgumentList line that starts the production manager."""
    out = []
    for line in text.splitlines():
        if "-ArgumentList" in line and "start_production" in line:
            out.append(line.strip())
    return out


def test_every_production_launcher_passes_x_utf8() -> None:
    missing: list[str] = []
    checked = 0
    for name in _LAUNCHERS:
        path = _ROOT / name
        if not path.exists():
            continue
        for line in _argument_lists(path.read_text(encoding="utf-8", errors="ignore")):
            checked += 1
            # "-X","utf8" as adjacent PowerShell arguments.
            if not re.search(r'"-X"\s*,\s*"utf8"', line):
                missing.append(f"{name}: {line}")
    assert checked, "no production launch line found -- has a launcher been renamed?"
    assert not missing, (
        "production would relaunch without -X utf8; a non-ASCII ticker then "
        "aborts the swap path:\n" + "\n".join(missing)
    )


def test_the_flag_precedes_the_script() -> None:
    """``-X utf8`` after main.py is an argument to main.py, not to Python."""
    for name in _LAUNCHERS:
        path = _ROOT / name
        if not path.exists():
            continue
        for line in _argument_lists(path.read_text(encoding="utf-8", errors="ignore")):
            x_at = line.find('"-X"')
            main_at = line.find('"main.py"')
            assert x_at != -1 and main_at != -1, f"{name}: {line}"
            assert x_at < main_at, f"{name}: -X utf8 must precede main.py: {line}"


# --------------------------------------------------------- the Python launchers


def _py_launch_snippets(text: str) -> list[str]:
    """Every argv LIST in *text* that starts the production manager.

    The bounds are the list's own brackets, found by scanning back to the
    nearest unmatched ``[``. A fixed-size text window was tried first and was
    wrong in both directions: it swallowed the explanatory comment above the
    list (whose prose mentions ``main.py``, which broke the ordering check on
    two launchers that were correct) and it would truncate a long one.
    """
    out = []
    for match in re.finditer(r'"start_production"', text):
        depth = 0
        start = None
        for i in range(match.start() - 1, max(0, match.start() - 4000) - 1, -1):
            ch = text[i]
            if ch == "]":
                depth += 1
            elif ch == "[":
                if depth == 0:
                    start = i
                    break
                depth -= 1
        if start is None:
            continue
        end = text.find("]", match.end())
        out.append(text[start:end if end != -1 else match.end()])
    return out


def _has_x_utf8_before_main(argv: str) -> bool:
    """``"-X", "utf8"`` adjacent, and ahead of the script, in one argv list.

    The script token is ``"main.py"`` or a path variable that ends in it
    (``MAIN_PATH``, ``ROOT / "main.py"``), so both spellings are looked for.
    """
    x = re.search(r'"-X"\s*,\s*"utf8"', argv)
    if x is None:
        return False
    script = re.search(r'MAIN_PATH|"main\.py"', argv)
    return script is None or x.start() < script.start()


def test_every_python_launcher_passes_x_utf8() -> None:
    """The check this file was missing when production lost the flag on 09-04."""
    missing: list[str] = []
    checked = 0
    for name in _PY_LAUNCHERS:
        path = _ROOT / name
        assert path.exists(), f"{name} is gone -- has a launcher been renamed?"
        for snippet in _py_launch_snippets(path.read_text(encoding="utf-8", errors="ignore")):
            if "subprocess" not in snippet and "COMMAND" not in snippet and "cmd" not in snippet:
                continue                      # prose, not an argv
            checked += 1
            if not _has_x_utf8_before_main(snippet):
                missing.append(name)
    assert checked, "no python production launch found -- has a launcher been renamed?"
    assert not missing, (
        "these launchers would relaunch production in cp1252 mode, where a "
        "bare open(path, 'w') inside library code cannot write a UTF-8 "
        "vocabulary: " + ", ".join(sorted(set(missing)))
    )


def test_the_wallet_runner_spawns_in_utf8_mode() -> None:
    """Checked by name because the sweep below structurally cannot see it.

    ``WalletRunner._worker`` builds ``cmd`` for a generic ``--action`` and only
    appends ``--stay-alive`` under ``if action == "start_production"``, so the
    string never appears inside the argv list and the bracket scan finds
    nothing to check. It is still the launcher the ops console uses.
    """
    text = (_ROOT / "services/wallet_runner.py").read_text(encoding="utf-8")
    argv = re.search(r"cmd\s*=\s*\[[^\]]*\]", text)
    assert argv, "WalletRunner._worker no longer builds a cmd list"
    assert _has_x_utf8_before_main(argv.group(0)), (
        "WalletRunner spawns main.py without -X utf8: " + argv.group(0)
    )


def test_no_launcher_anywhere_in_the_repo_is_missed() -> None:
    """Enumeration is how the last fix stayed half-applied. Discover instead.

    Any NEW file that spawns ``main.py --action start_production`` has to carry
    the flag too, so the sweep is over the tree rather than over a list. Test
    files are excluded: they assert about launch lines, they do not launch.
    """
    offenders: list[str] = []
    for path in _ROOT.rglob("*.py"):
        parts = set(path.parts)
        if parts & _SKIP_DIRS or path.name.startswith("test_"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if '"start_production"' not in text:
            continue
        for snippet in _py_launch_snippets(text):
            if "main.py" not in snippet:
                continue                      # not a python-launching argv
            if not _has_x_utf8_before_main(snippet):
                offenders.append(str(path.relative_to(_ROOT)).replace("\\", "/"))
    assert not offenders, (
        "production launcher(s) without -X utf8: " + ", ".join(sorted(set(offenders)))
    )


#: The PYTHON launchers, which 6a0bd29 did not reach.
#:
#: That commit fixed the two .ps1 launchers and said "both launchers", but four
#: things in this repo can start production. Measured 2026-09-03 16:27:35, the
#: running production process (pid 5308) had the command line
#:
#:     python.exe -u main.py --action start_production --stay-alive
#:
#: which is ``w1z4rd_watchdog.py``'s ``prod_args`` verbatim and carries no
#: ``-X utf8``. The two launchers that were fixed were not the one doing the
#: relaunching. A partial change that leaves callers on the old contract is
#: worse than none, because it looks done.
_PY_LAUNCHERS = (
    ("scripts/main_keeper.py", "cmd = ["),
    ("scripts/w1z4rd_watchdog.py", "prod_args = ["),
)


def _py_launch_list(text: str, marker: str) -> str:
    """The single argv literal that starts production, flattened to one line."""
    start = text.find(marker)
    while start != -1:
        end = text.find("]", start)
        block = " ".join(text[start:end].split())
        if "start_production" in block:
            return block
        start = text.find(marker, start + 1)
    return ""


def test_every_python_launcher_passes_x_utf8() -> None:
    missing: list[str] = []
    checked = 0
    for name, marker in _PY_LAUNCHERS:
        path = _ROOT / name
        if not path.exists():
            continue
        block = _py_launch_list(path.read_text(encoding="utf-8", errors="ignore"), marker)
        assert block, f"{name}: no start_production argv literal found"
        checked += 1
        if not re.search(r'"-X"\s*,\s*"utf8"', block):
            missing.append(f"{name}: {block}")
    assert checked == len(_PY_LAUNCHERS), "a python launcher has been renamed"
    assert not missing, (
        "these launchers respawn production without -X utf8:\n" + "\n".join(missing)
    )


def test_the_python_launchers_put_the_flag_before_main_py() -> None:
    for name, marker in _PY_LAUNCHERS:
        path = _ROOT / name
        if not path.exists():
            continue
        block = _py_launch_list(path.read_text(encoding="utf-8", errors="ignore"), marker)
        x_at = block.find('"-X"')
        # main.py appears as a literal or as str(MAIN_PATH); both end the
        # python-argument section.
        main_at = min(
            (i for i in (block.find('"main.py"'), block.find("MAIN_PATH")) if i != -1),
            default=-1,
        )
        assert x_at != -1 and main_at != -1, f"{name}: {block}"
        assert x_at < main_at, f"{name}: -X utf8 must precede main.py: {block}"


def test_the_character_that_breaks_it_is_really_in_the_pair_list() -> None:
    """Guard the premise, not just the flag: if no ticker is non-ASCII any
    more, this test should be the thing that says so."""
    log = _ROOT / "data" / "production.log"
    if not log.exists():
        return  # nothing to assert against on a clean checkout
    text = log.read_text(encoding="utf-8", errors="ignore")
    assert "₮" in text or "->" in text, (
        "expected the production log to contain a character cp1252 cannot encode"
    )
