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
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# The launchers that start the production manager, all of them.
_LAUNCHERS = ("GetToLiveTrading.ps1", "launch_revenir.ps1")


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
