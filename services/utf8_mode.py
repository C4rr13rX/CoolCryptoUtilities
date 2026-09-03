"""Keep text that this process writes inside what the platform can encode.

Windows still defaults ``open()`` to the ANSI code page -- cp1252 on this box.
Every file this project writes is UTF-8: news headlines, token symbols, model
vocabularies. Where our own code opens a file we pass ``encoding="utf-8"``, but
library code does not, and one of those writes is on the critical path.

Measured 2026-09-02 in production. Keras persists an adapted
``TextVectorization`` vocabulary through
``keras/src/layers/preprocessing/index_lookup.py:837``::

    with open(vocabulary_filepath, "w") as f:
        f.write("\\n".join([str(w) for w in vocabulary]))

No ``encoding=``, so cp1252. The vocabulary is adapted from news headlines, and
headlines contain ``U+2192`` (an arrow). ``model.save()`` therefore died with
``UnicodeEncodeError: 'charmap' codec can't encode character '\\u2192'`` after
writing 49,057 bytes, ``models/active_model.keras`` was never published, and
the directory stayed empty. The log tail held 443 copies of that failure.

That is not a cosmetic failure. ``TrainingPipeline.ensure_active_model()`` runs
from ``bot._handle_sample`` on EVERY market tick, inside the market-stream's
asyncio event loop. With no artifact on disk, every tick rebuilt a Keras
baseline and re-adapted the vectorizers on that loop -- seconds of GIL-holding
TensorFlow work per tick. The REST poll coroutines never got scheduled, their
timeouts fired on requests the network answers in 0.4s, the stream declared a
"network outage" and backed off 24-51s, and the tick rate collapsed roughly
tenfold: 92.5% of a four-hour window produced no ticks at all. Per-symbol
density then fell under the six-ticks-per-hour floor ``atf_static`` requires
before it will open a position it can stop out of, so every ghost entry was
refused and ghost trading stopped.

One unencodable arrow, at the far end of that chain, is why nothing traded.

UTF-8 mode can only be selected at interpreter startup, and this module does
NOT try to re-exec into it. That was the first attempt and it is wrong here:
``os.execv`` on Windows does not overlay the process the way POSIX does. It
starts a new one and terminates this one, so the PID changes and the launcher's
stdout redirection is lost -- measured, the child's output never reached the
parent's pipe. Production is supervised and its log is a redirect, so a fix
that silently orphans both is worse than the bug.

Instead:

  * ``PYTHONUTF8`` is exported, which is inherited by the subprocesses this
    program spawns (``main.py --action swap_quote``, ``download2000.py``) and
    puts THEM in UTF-8 mode from startup, where it is free and safe.
  * ``harden_stdio`` re-encodes THIS process's stdout/stderr as UTF-8, which
    ``PYTHONUTF8`` cannot do after startup (see below).
  * For a process already running without it -- launched by an external
    supervisor, say -- ``sanitise_for_default_encoding`` makes the text we hand
    to libraries encodable, so the vocabulary write cannot fail whatever
    started us. In UTF-8 mode it is a no-op.

The second instance: it also killed every live trade
-----------------------------------------------------

Measured 2026-09-03. ``PYTHONUTF8`` is read by the interpreter at startup only,
so exporting it from ``ensure_utf8_mode`` fixes children and does nothing for
the process doing the exporting. Production runs as::

    .venv/Scripts/python.exe -u main.py --action start_production --stay-alive

with stdout redirected to ``data/production.log``. No ``-X utf8``, so
``sys.stdout.encoding`` is ``cp1252`` -- confirmed by running that exact command
shape: ``stdout.encoding='cp1252' isatty=False``, and ``ensure_utf8_mode()``
returned ``False`` while leaving it at ``cp1252``.

``SwapService.swap`` announces its route order before trying any route::

    services/swap_service.py:789
    print(f"[info] chainId={cid} taker={taker} routes={'->'.join(_routes)}")

That separator was ``U+2192``. The first arrow lands at index 83 of the
formatted line, so the ``print`` raised
``UnicodeEncodeError: 'charmap' codec can't encode character '\\u2192' in
position 83`` -- and ``trading/bot.py:4186`` catches everything out of
``swapper.swap`` as ``swap_error``. Every live entry was recorded
``live-entry-failed`` with that reason, before a single route was attempted.
The trades were never refused on their merits; a log line aborted them.

Eight such characters were on that path (``U+2192``, ``U+2014``, ``U+2026`` at
swap_service.py lines 520, 608, 659, 705, 747, 789, 871, 886). Line 789 is
unconditional, which is why the failure was total; 871 and 886 sit in the
Uniswap->Camelot fallback, so ASCII-ing only 789 would have moved the crash one
route down rather than removing it. Both were done: the characters are now
ASCII, and ``harden_stdio`` makes the class of failure impossible regardless of
what any future print contains -- a token symbol is attacker-chosen text that
reaches these logs, so the encoding must be right rather than the content lucky.
"""
from __future__ import annotations

import os
import sys
from typing import Iterable, List

#: Substituted for characters the ambient encoding cannot represent. ASCII, so
#: it survives every code page, and visible, so a vocabulary built from
#: degraded text is recognisable as degraded rather than silently plausible.
REPLACEMENT = "?"


def utf8_mode_active() -> bool:
    """Is this interpreter reading/writing text as UTF-8 by default?"""
    return bool(getattr(sys.flags, "utf8_mode", 0))


def default_encoding() -> str:
    """The encoding a bare ``open(path, "w")`` will use in this process."""
    if utf8_mode_active():
        return "utf-8"
    try:
        import locale

        return locale.getpreferredencoding(False) or "utf-8"
    except Exception:  # noqa: BLE001 - a missing locale is not worth raising over
        return "utf-8"


#: How stdout/stderr handle a character the stream cannot represent. Chosen
#: over ``"replace"`` because a log is read to diagnose: ``→`` names the
#: character that would otherwise vanish into an anonymous ``?``. Only reached
#: if the UTF-8 reconfigure below fails, since UTF-8 encodes everything.
STDIO_ERRORS = "backslashreplace"


def harden_stdio() -> List[str]:
    """Make ``print`` on this process's stdout/stderr incapable of raising.

    Returns the names of the streams that were successfully reconfigured, so a
    caller (or a test) can tell "hardened" from "silently did nothing".

    Why this exists rather than relying on ``PYTHONUTF8``: that variable is read
    at interpreter startup, so a process that did not receive it cannot opt in
    afterwards. Production is such a process, and one ``U+2192`` in a route-order
    log aborted every live trade (module docstring). Reconfiguring the streams
    is the only fix available from inside an already-running interpreter.

    Deliberately total -- it never raises. The streams here are:

      * ``TextIOWrapper``  -- the normal case, redirected or not; reconfigured.
      * ``None``           -- ``pythonw.exe`` has no stdout at all, and
                              ``scripts/loop_console.py`` runs under it.
      * a capture object   -- pytest and our own tee wrappers replace these and
                              need not expose ``reconfigure``.

    Only the first can be hardened; the others are skipped rather than made
    fatal, because a process that cannot configure its log must still trade.
    """
    hardened: List[str] = []
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors=STDIO_ERRORS)
        except (ValueError, OSError, AttributeError, TypeError):
            # Re-encoding was refused (a detached or non-text stream). Settle
            # for making it lenient, which still removes the raise.
            try:
                reconfigure(errors=STDIO_ERRORS)
            except Exception:  # noqa: BLE001 - logging must not break the caller
                continue
        hardened.append(name)
    return hardened


def ensure_utf8_mode() -> bool:
    """Export ``PYTHONUTF8`` for children, harden our own stdio, report state.

    Returns True when THIS interpreter is already in UTF-8 mode. It never
    re-execs (see the module docstring), so a False return is information for
    the caller, not a failure: ``harden_stdio`` has still run, and the
    sanitising path below covers file writes.
    """
    os.environ["PYTHONUTF8"] = "1"
    harden_stdio()
    return utf8_mode_active()


def is_encodable(text: str, encoding: str | None = None) -> bool:
    """Can ``text`` survive a bare ``open(..., "w").write(text)`` here?"""
    try:
        text.encode(encoding or default_encoding())
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def sanitise_for_default_encoding(text: str, encoding: str | None = None) -> str:
    """Return ``text`` with any character this process cannot write replaced.

    A no-op -- the same object back -- when nothing needs replacing, which is
    every string once the interpreter is in UTF-8 mode.
    """
    enc = encoding or default_encoding()
    if is_encodable(text, enc):
        return text
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def sanitise_all(texts: Iterable[str]) -> List[str]:
    """``sanitise_for_default_encoding`` over a sequence, resolving the
    encoding once rather than per item."""
    enc = default_encoding()
    if enc.lower().replace("-", "") == "utf8":
        return list(texts)
    return [sanitise_for_default_encoding(text, enc) for text in texts]
