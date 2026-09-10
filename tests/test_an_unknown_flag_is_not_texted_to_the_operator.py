"""An option ``notify_sms`` does not know must not become the text message.

THE DEFECT, done live twice on 2026-09-10. ``main`` matched only ``--check``
and ``--test`` at ``argv[1]`` and let everything else fall through to
``send(" ".join(args))``. So::

    python scripts/notify_sms.py --body-file notice.txt

-- a plausible guess at an interface that did not exist yet -- sent the literal
string ``"--body-file notice.txt"`` to a real phone and printed::

    notify_sms: sent to 9194957881@vtext.com (23 chars) <message-id>

with exit code 0. The operator received two junk texts before anyone read the
source, and nothing downstream could have detected it: the caller sees 0, the
log says sent, and only the person holding the phone knows. Reporting SUCCESS
for delivering the wrong thing is the worst shape a failure can take.

THE SECOND DEFECT, WHICH HAS THE SAME FIX. Notices were arriving SHORT: one of
roughly 2,400 characters arrived as 978, and one of roughly 1,450 arrived as
331, cut mid-word at "Median price move is 0.0783 perce". A 34-character final
part means the body ENDED there. The script is NOT the culprit -- ``segments``
breaks on whitespace and preserves every word, and ``MAX_SEGMENTS`` is 12 so
the segment cap was not in play at those lengths. The body was ALREADY short
when it reached ``argv``, because a command line has a length limit and the
shell cut it.

``--body-file`` closes both: it takes the body off the command line entirely,
and it gives the parser a place to REFUSE an unknown flag instead of texting
it.

It matters more than a formatting nit. The standing orders put the ASK LAST in
every notice, so an argv cut removes precisely the decision request and leaves
only the evidence that motivated it -- which is what happened, and the ask had
to be re-sent on its own.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.notify_sms as notify_sms  # noqa: E402


@pytest.fixture()
def sent(monkeypatch):
    """Capture what would be transmitted; SES is never touched."""
    captured = []

    def _fake_send(body):
        captured.append(body)
        return 0

    monkeypatch.setattr(notify_sms, "send", _fake_send)
    return captured


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["notify_sms.py", *argv])
    return notify_sms.main()


def test_an_unknown_flag_is_refused_and_never_sent(monkeypatch, sent):
    """THE LOAD-BEARING ASSERTION -- the exact command that texted the operator.

    Against the old behaviour this sends "--body-file /tmp/x.txt" to a phone and
    returns 0. Both halves must change: nothing transmitted, and a nonzero exit
    so a caller can tell.
    """
    rc = _run(monkeypatch, "--body-fyle", "/tmp/x.txt")

    assert sent == [], "an unrecognised option must never be transmitted"
    assert rc != 0, "refusing to send must not be reported as success"


@pytest.mark.parametrize("flag", ["--verbose", "--dry-run", "-n", "--body-file="])
def test_no_unknown_option_shape_becomes_a_message(monkeypatch, sent, flag):
    """Not just the one flag that was typed -- the whole class of them.

    ``-n`` is included deliberately: a single-dash option is still an option to
    whoever typed it, but ``startswith("--")`` does not catch it, so this pins
    down what the guard actually promises rather than what it looks like it
    promises.
    """
    rc = _run(monkeypatch, flag)

    if flag.startswith("--"):
        assert sent == [], "%s must be refused, not texted" % flag
        assert rc != 0
    else:
        # Documented limit: single-dash arguments are still treated as message
        # text. Recorded so the next reader does not assume coverage that is
        # not here.
        assert sent == [flag]


def test_a_message_that_really_starts_with_dashes_still_sends(monkeypatch, sent):
    """The guard must not make a legitimate notice unsendable.

    ``--`` ends option parsing, which is the standard escape and the reason the
    refusal above is safe.
    """
    rc = _run(monkeypatch, "--", "--net", "P/L", "moved")

    assert rc == 0
    assert sent == ["--net P/L moved"]


def test_body_file_sends_the_file_contents_not_the_path(monkeypatch, sent, tmp_path):
    """The interface that was guessed at now exists and reads the FILE."""
    path = tmp_path / "notice.txt"
    path.write_text("Pass 111 finished. Held-out edge is unchanged.", encoding="utf-8")

    rc = _run(monkeypatch, "--body-file", str(path))

    assert rc == 0
    assert sent == ["Pass 111 finished. Held-out edge is unchanged."]
    assert str(path) not in (sent[0] if sent else ""), "the PATH must not be the message"


def test_a_notice_too_long_for_argv_round_trips_with_every_word_intact(tmp_path):
    """IRIS'S CASE, at the length that actually arrived short.

    A ~2,400-character notice -- the one that went out as 978 characters in four
    parts -- read from a file and segmented. Every word must survive, in order.
    Asserting on WORDS rather than on raw character totals is deliberate: the
    parts carry "(n/m) " prefixes, so summing ``len(part)`` compares the body
    against the body-plus-numbering and would pass while words were missing.
    """
    words = ["word%03d" % i for i in range(300)]
    body = " ".join(words)
    assert len(body) > 2000, "the fixture must exceed the length that was cut"

    path = tmp_path / "notice.txt"
    path.write_text(body, encoding="utf-8")

    parts = notify_sms.segments(notify_sms.read_body_file(str(path)))

    assert len(parts) > 1, "a 2400-character notice must segment, not fit in one"
    assert len(parts) <= notify_sms.MAX_SEGMENTS
    assert not any("TRUNCATED" in p for p in parts), (
        "a notice of this length must not hit the segment cap"
    )

    # Strip the "(n/m) " prefix each part carries and re-join.
    rejoined = " ".join(p.split(" ", 1)[1] for p in parts)
    assert rejoined.split() == words, "every word must survive, in order"


def test_a_missing_or_empty_body_file_sends_nothing(monkeypatch, sent, tmp_path):
    """Failing to read the notice must not send a truncated or empty one.

    The empty case is the one that matters: ``send`` already refuses an empty
    body, but it refuses it AFTER the caller believes a notice went out. Catch
    it at the parser.
    """
    assert _run(monkeypatch, "--body-file", str(tmp_path / "nope.txt")) != 0
    assert sent == []

    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    assert _run(monkeypatch, "--body-file", str(empty)) != 0
    assert sent == []

    assert _run(monkeypatch, "--body-file") != 0
    assert sent == [], "--body-file with no path must not text the flag itself"
