"""The operator's notice transport must never silently cut a message.

``scripts/notify_sms.py`` applied ``body[:MAX_BODY]`` with MAX_BODY = 300 and
then printed "sent ... (300 chars)" as a success. Measured 2026-09-10: a
1,791-character pass-109 notice went out as 300 characters, stopping inside a
sentence, and nothing anywhere said so.

That is the exact failure the standing instructions describe -- "the operator
reads these as texts and has been getting abbreviated fragments that stop
mid-phrase" -- while asserting that "the transport no longer truncates
anything" and blaming agents for hand-abbreviating. The transport was the
cause, so no amount of writing full sentences could have fixed it.

``segments`` is pure, so this is provable without SES.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "notify_sms.py"
_spec = importlib.util.spec_from_file_location("notify_sms_under_test", _PATH)
notify_sms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(notify_sms)


#: The real pass-109 notice length, rounded up. Well past one segment.
LONG = " ".join("word%d" % i for i in range(200))


def _rejoin(parts):
    """Every part minus its "(i/n) " prefix, back in order."""
    return " ".join(p.split(") ", 1)[1] for p in parts)


def test_a_long_notice_keeps_every_word():
    """The whole point: nothing is dropped, so the notice stands alone."""
    parts = notify_sms.segments(LONG)
    assert len(parts) > 1, "a notice this long must be split, not sent as one"
    assert _rejoin(parts) == LONG


def test_no_part_exceeds_the_segment_size():
    for part in notify_sms.segments(LONG):
        assert len(part) <= notify_sms.SEGMENT_BODY


def test_the_numbering_names_the_whole_so_a_reader_knows_what_is_missing():
    parts = notify_sms.segments(LONG)
    total = len(parts)
    for i, part in enumerate(parts, 1):
        assert part.startswith("(%d/%d) " % (i, total))


def test_a_notice_that_fits_is_sent_whole_and_unnumbered():
    """A one-part notice labelled "(1/1)" reads as though a part is missing."""
    assert notify_sms.segments("done: 22 failures -> 2") == [
        "done: 22 failures -> 2"
    ]


def test_a_word_is_never_cut_in_half():
    """The old slice cut mid-word; splitting on whitespace must not."""
    parts = notify_sms.segments(LONG)
    for part in parts:
        for word in part.split(") ", 1)[1].split():
            assert word in LONG.split(), "%r is not a whole word" % word


def test_an_over_long_notice_says_so_inside_the_notice():
    """Dropping content silently is the defect; announcing it is the fix."""
    huge = " ".join("w%d" % i for i in range(20000))
    parts = notify_sms.segments(huge, limit=3)
    assert len(parts) == 3
    assert "TRUNCATED" in parts[-1]


def test_an_empty_notice_sends_nothing():
    assert notify_sms.segments("") == []
    assert notify_sms.segments("   ") == []


@pytest.mark.parametrize("size", [40, 120, 300])
def test_no_word_is_lost_at_any_segment_size(size):
    parts = notify_sms.segments(LONG, size=size, limit=999)
    assert _rejoin(parts) == LONG
