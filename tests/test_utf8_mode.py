"""The process must be able to write text the platform cannot encode.

Two production failures, one cause. On Windows a redirected ``sys.stdout`` and a
bare ``open(path, "w")`` both default to cp1252, and this project's text is
UTF-8: news headlines, token symbols, route separators.

  1. Keras writes an adapted ``TextVectorization`` vocabulary through
     ``open(vocabulary_filepath, "w")`` (index_lookup.py:837). A U+2192 from a
     headline made ``model.save()`` raise, no model artifact was published, and
     the rebuild that missing artifact triggered ran on the market-stream event
     loop on every tick -- collapsing the price feed and stopping ghost trading.

  2. ``SwapService.swap`` announced its route order with a U+2192 separator
     before trying any route (services/swap_service.py:789). Printing it raised
     ``UnicodeEncodeError``, ``trading/bot.py:4186`` caught that as
     ``swap_error``, and EVERY live entry was recorded ``live-entry-failed``.
     No trade was ever refused on its merits; a log line aborted them all.

See services/utf8_mode.py for the measurements behind both.

These tests force ``PYTHONIOENCODING=cp1252`` in the children rather than
relying on the host's locale, so the failure reproduces identically on a
UTF-8 platform and the assertions mean the same thing everywhere.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services import utf8_mode  # noqa: E402

#: The character that did the damage, in both failures.
ARROW = "→"

#: Forced on the children so "what happens under a code page that cannot
#: represent U+2192" is asked of every platform, not just this Windows box.
NARROW_ENCODING = "cp1252"


def _run(code: str, *, env_extra: dict | None = None, utf8: bool = False):
    """Run ``code`` in a child whose stdio cannot encode U+2192.

    Starts from a hostile baseline: whatever this test runner inherited must
    not be what makes an assertion pass.
    """
    env = dict(os.environ)
    env.pop("PYTHONUTF8", None)
    env["PYTHONIOENCODING"] = NARROW_ENCODING
    if env_extra:
        env.update(env_extra)
    argv = [sys.executable]
    if utf8:
        argv += ["-X", "utf8"]
    argv += ["-c", code]
    return subprocess.run(
        argv, cwd=str(REPO_ROOT), env=env, text=True,
        encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
    )


class EnsureUtf8ModeTests(unittest.TestCase):
    """The contract ``ensure_utf8_mode`` actually offers."""

    def test_pythonutf8_is_exported_for_children(self) -> None:
        """Subprocesses (swap_quote, download2000) inherit the same rule.

        They read it at THEIR startup, where it works, which is the whole
        reason the variable is set rather than relied on for this process.
        """
        proc = _run(
            "import os\n"
            "from services.utf8_mode import ensure_utf8_mode\n"
            "ensure_utf8_mode()\n"
            "print('PYTHONUTF8=%s' % os.environ.get('PYTHONUTF8'))\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("PYTHONUTF8=1", proc.stdout)

    def test_does_not_reexec(self) -> None:
        """The process that called it is the process that continues.

        Re-exec was the first attempt and was abandoned: ``os.execv`` on Windows
        starts a new process and kills this one, so the PID changes and the
        launcher's stdout redirection is lost. Production is supervised through
        exactly such a redirect. Pinned because "just re-exec into UTF-8 mode"
        is the obvious-looking change someone will reach for again.
        """
        proc = _run(
            "import os\n"
            "before = os.getpid()\n"
            "from services.utf8_mode import ensure_utf8_mode\n"
            "ensure_utf8_mode()\n"
            "print('same_process=%s' % (os.getpid() == before))\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("same_process=True", proc.stdout)

    def test_reports_false_without_the_startup_flag(self) -> None:
        """A truthful report: the interpreter flag cannot be set after startup."""
        proc = _run(
            "from services.utf8_mode import ensure_utf8_mode\n"
            "print('flag=%s' % ensure_utf8_mode())\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("flag=False", proc.stdout)

    def test_reports_true_with_the_startup_flag(self) -> None:
        proc = _run(
            "from services.utf8_mode import ensure_utf8_mode\n"
            "print('flag=%s' % ensure_utf8_mode())\n",
            utf8=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("flag=True", proc.stdout)


class HardenStdioTests(unittest.TestCase):
    """``print`` must not be able to abort its caller."""

    def test_narrow_stdout_raises_without_the_guard(self) -> None:
        """Documents the platform behaviour the guard exists to override.

        If this ever stops failing, the rest of this class is proving nothing.
        """
        proc = _run(
            "try:\n"
            "    print('routes=' + %r)\n"
            "    print('NO-RAISE')\n"
            "except UnicodeEncodeError as exc:\n"
            "    print('RAISED %%s' %% exc)\n" % ARROW
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("RAISED", proc.stdout)

    def test_unencodable_print_survives_after_hardening(self) -> None:
        """The pin: the exact failure that aborted every live entry."""
        proc = _run(
            "from services.utf8_mode import harden_stdio\n"
            "harden_stdio()\n"
            "print('routes=' + %r)\n"
            "print('NO-RAISE')\n" % ARROW
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("NO-RAISE", proc.stdout)

    def test_ensure_utf8_mode_hardens_stdio_too(self) -> None:
        """main.py calls only ``ensure_utf8_mode``; it must be enough."""
        proc = _run(
            "import sys\n"
            "from services.utf8_mode import ensure_utf8_mode\n"
            "ensure_utf8_mode()\n"
            "print('encoding=%%s' %% sys.stdout.encoding)\n"
            "print('arrow=' + %r)\n" % ARROW
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("encoding=utf-8", proc.stdout)
        self.assertIn("arrow=" + ARROW, proc.stdout)

    def test_reports_which_streams_were_hardened(self) -> None:
        proc = _run(
            "from services.utf8_mode import harden_stdio\n"
            "print('hardened=%s' % ','.join(harden_stdio()))\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("hardened=stdout,stderr", proc.stdout)

    def test_survives_absent_stdout(self) -> None:
        """``pythonw.exe`` sets ``sys.stdout`` to None.

        scripts/loop_console.py runs under pythonw, so this is a real
        configuration and not a hypothetical.
        """
        proc = _run(
            "import sys\n"
            "real = sys.stdout\n"
            "sys.stdout = None\n"
            "from services.utf8_mode import harden_stdio\n"
            "names = harden_stdio()\n"
            "sys.stdout = real\n"
            "print('ok hardened=%s' % names)\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("ok hardened=['stderr']", proc.stdout)

    def test_survives_a_stream_without_reconfigure(self) -> None:
        """pytest's capture object, and our own tee wrappers, are not
        ``TextIOWrapper`` and need not offer ``reconfigure``."""
        proc = _run(
            "import io, sys\n"
            "class Tee:\n"
            "    def write(self, s): return len(s)\n"
            "    def flush(self): pass\n"
            "real = sys.stdout\n"
            "sys.stdout = Tee()\n"
            "from services.utf8_mode import harden_stdio\n"
            "names = harden_stdio()\n"
            "sys.stdout = real\n"
            "print('ok hardened=%s' % names)\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("ok hardened=['stderr']", proc.stdout)


class SwapRouteLogTests(unittest.TestCase):
    """The money path specifically: announcing a route must not abort a swap."""

    def test_route_order_announcement_is_printable(self) -> None:
        """Reproduces services/swap_service.py:789 under a narrow code page.

        ``SwapService.swap`` prints this BEFORE trying any route, so when it
        raised, no route was ever attempted -- the swap failed at its own log
        line. Uses the real ``default_route_order`` so a future route named in
        non-ASCII fails here rather than in production.
        """
        proc = _run(
            "from services.utf8_mode import ensure_utf8_mode\n"
            "ensure_utf8_mode()\n"
            "from services.swap_service import default_route_order\n"
            "cid, taker = 8453, '0x291c854811e92906a658Fb94Aa511bF919f968ad'\n"
            "print(f\"[info] chainId={cid} taker={taker} \"\n"
            "      f\"routes={'->'.join(default_route_order())}\")\n"
            "print('NO-RAISE')\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("NO-RAISE", proc.stdout)

    def test_swap_service_source_holds_no_unencodable_character(self) -> None:
        """No literal on the money path may need more than the platform's code page.

        Belt to ``harden_stdio``'s braces. Eight such characters were in this
        file (lines 520, 608, 659, 705, 747, 789, 871, 886); 789 is
        unconditional, and 871/886 sit in the Uniswap->Camelot fallback, so
        removing only the one that fired would have moved the crash one route
        down. This fails the moment someone types a nice arrow back in.
        """
        source = (REPO_ROOT / "services" / "swap_service.py").read_text(encoding="utf-8")
        offenders = [
            (n, line.strip())
            for n, line in enumerate(source.splitlines(), 1)
            if not _encodable(line, NARROW_ENCODING)
        ]
        self.assertEqual(
            offenders, [],
            "services/swap_service.py holds characters cp1252 cannot encode; "
            "a print of one of these aborted every live trade",
        )


def _encodable(text: str, encoding: str) -> bool:
    try:
        text.encode(encoding)
        return True
    except UnicodeEncodeError:
        return False


class SanitiseTests(unittest.TestCase):
    """The file-write path, which ``harden_stdio`` cannot help.

    ``open()``'s default encoding is fixed at interpreter startup, so text
    handed to a library that opens its own files must be made encodable
    instead.
    """

    def test_sanitise_replaces_only_what_cannot_be_encoded(self) -> None:
        out = utf8_mode.sanitise_for_default_encoding("btc rally " + ARROW + " up",
                                                      NARROW_ENCODING)
        self.assertNotIn(ARROW, out)
        self.assertIn("btc rally", out)

    def test_sanitise_is_identity_for_encodable_text(self) -> None:
        text = "btc rally up"
        self.assertIs(
            utf8_mode.sanitise_for_default_encoding(text, NARROW_ENCODING), text)

    def test_sanitise_is_identity_under_utf8(self) -> None:
        text = "btc rally " + ARROW + " up"
        self.assertIs(utf8_mode.sanitise_for_default_encoding(text, "utf-8"), text)

    def test_sanitise_all_matches_the_single_string_form(self) -> None:
        texts = ["plain", "btc " + ARROW + " up"]
        self.assertEqual(
            utf8_mode.sanitise_all(texts),
            [utf8_mode.sanitise_for_default_encoding(t) for t in texts],
        )

    def test_is_encodable_agrees_with_a_real_write(self) -> None:
        self.assertFalse(utf8_mode.is_encodable(ARROW, NARROW_ENCODING))
        self.assertTrue(utf8_mode.is_encodable("plain", NARROW_ENCODING))


if __name__ == "__main__":
    unittest.main()
