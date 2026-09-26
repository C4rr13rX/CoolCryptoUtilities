"""The notice transport must deliver the WHOLE notice, or say it did not.

Four agents reported a notice as sent when roughly half of it arrived, every
one of them off exit code 0. The reports blamed "a hard cap of six in the send
loop". Measured 2026-09-26 by Cove against scripts/notify_sms.py unchanged
since 1f08755: THERE IS NO SUCH CAP -- stub ``_send_one`` and ``send`` performs
one call per segment, 11 of them on a 3,001-character body. What was real is
``MAX_SEGMENTS`` = 12 binding on anything longer, dropping the tail, and
returning 0.

These tests go through ``main`` with ``--body-file``, which is the path the
agents used, and they measure the SUM OF WHAT THE TRANSPORT HANDED TO SES
against the sum of the body's full segmentation. Both numbers, every time: a
count of sends cannot tell truncation from a shorter body.

``test_long_notice_is_delivered_whole_or_reported_incomplete`` is the one that
fails against the pre-fix code -- it returned 0 there.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "notify_sms_under_test", os.path.join(ROOT, "scripts", "notify_sms.py")
)
notify_sms = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(notify_sms)


def _body(target_chars: int) -> str:
    """A notice-shaped body of a known length: numbers, clauses, newlines."""
    lines, i = [], 0
    while sum(len(x) + 1 for x in lines) < target_chars:
        i += 1
        lines.append(
            "line %d: measured number %0.6f and a clause that runs on a while"
            % (i, i / 7.0)
        )
    return "\n".join(lines)[:target_chars]


@pytest.fixture
def transport(monkeypatch, tmp_path):
    """Capture what would go to SES, and run main() the way an agent does."""
    sent: list = []
    monkeypatch.setattr(
        notify_sms, "_send_one", lambda aws, cfg, body: (sent.append(body), 0)[1]
    )
    monkeypatch.setattr(notify_sms.shutil, "which", lambda name: "/fake/aws")
    monkeypatch.delenv(notify_sms.MAX_SEGMENTS_ENV, raising=False)

    def run(body: str, *, argv_extra=()):
        path = tmp_path / "notice.txt"
        path.write_text(body, encoding="utf-8")
        monkeypatch.setattr(
            sys, "argv", ["notify_sms.py", "--body-file", str(path), *argv_extra]
        )
        sent.clear()
        rc = notify_sms.main()
        return rc, list(sent)

    return run


def _full_segment_sum(body: str) -> int:
    """The characters the body NEEDS, at the cap's own part boundaries.

    Not ``segments(body)`` -- that returns the capped list, so comparing the
    transport against it can never detect the cap. This re-splits at the same
    room and numbers every part.
    """
    room = notify_sms._room(notify_sms.SEGMENT_BODY, notify_sms.max_segments())
    return sum(len(p) for p in notify_sms._number(notify_sms._split(body, room)))


def test_short_notice_is_delivered_whole(transport):
    rc, sent = transport(_body(1434))
    assert rc == 0
    assert sum(len(s) for s in sent) == _full_segment_sum(_body(1434))


def test_3001_char_body_sends_11_segments_not_6(transport):
    """The exact body length from the report: 6 sends of 1581 was the claim."""
    body = _body(3001)
    rc, sent = transport(body)
    assert len(sent) == 11, "reported as 6 sends; the send loop has no cap of 6"
    assert sum(len(s) for s in sent) == 3069
    assert rc == 0


def test_long_notice_is_delivered_whole_or_reported_incomplete(transport):
    """FAILS AGAINST THE PRE-FIX CODE: it sent 3,600 of 4,067 and returned 0.

    Proven by loading ``git show HEAD:scripts/notify_sms.py`` at 1f08755 and
    running this same assertion: 12 sends, 3,600 characters delivered of the
    4,067 its own segmentation needed, rc 0, assertion False.

    Either the transport delivers every character the segmentation produced,
    or it exits nonzero. Silently keeping the first 88% is the one outcome
    this file exists to forbid.
    """
    body = _body(3977)
    rc, sent = transport(body)
    delivered = sum(len(s) for s in sent)
    needed = _full_segment_sum(body)
    assert delivered == needed or rc != 0, (
        "delivered %d of %d characters and returned %d"
        % (delivered, needed, rc)
    )
    # And specifically: this body cannot fit, so it must be the loud branch.
    assert rc == 3
    assert delivered < needed


def test_incomplete_send_names_how_much_was_dropped(transport, capsys):
    rc, sent = transport(_body(3977))
    err = capsys.readouterr().err
    assert rc == 3
    _parts, dropped_chars, dropped_segs = notify_sms.plan(_body(3977))
    assert dropped_chars > 0 and dropped_segs > 0
    assert str(dropped_chars) in err and "INCOMPLETE" in err
    assert notify_sms.MAX_SEGMENTS_ENV in err, "must say how to send it whole"


def test_raising_the_cap_delivers_the_whole_notice(transport, monkeypatch):
    """The escape hatch has to actually work, or exit 3 is just a nag."""
    body = _body(3977)
    monkeypatch.setenv(notify_sms.MAX_SEGMENTS_ENV, "40")
    rc, sent = transport(body)
    assert rc == 0
    assert sum(len(s) for s in sent) == _full_segment_sum(body)


def test_plan_and_segments_agree_on_what_is_sent():
    """``segments`` is the capped list; ``plan`` adds what the cap cost."""
    for target in (250, 1434, 3001, 3977, 9000):
        body = _body(target)
        parts, dropped_chars, dropped_segs = notify_sms.plan(body)
        assert notify_sms.segments(body) == parts
        assert (dropped_chars > 0) == (dropped_segs > 0)
        if dropped_segs:
            assert len(parts) == notify_sms.max_segments()


def test_truncation_marker_is_still_in_the_delivered_notice(transport):
    """stderr never reaches the person holding the phone."""
    _rc, sent = transport(_body(9000))
    assert "[TRUNCATED" in sent[-1]
