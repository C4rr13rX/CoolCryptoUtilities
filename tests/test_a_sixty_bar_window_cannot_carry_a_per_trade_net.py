"""A held-out window too short to CONTAIN 30 trough labels may not be quoted.

The failure this prevents, measured pass 114 (node-free, 150 eligible 3600s
corpora, b1bbd12)
---------------------------------------------------------------------------
At the shipped omen threshold -- OMEN_COST_MULTIPLE 1.5, so 0.9750% against a
0.6500% round trip -- the MEDIAN corpus labels 14.46% of its bars ``trough``.
A 60-bar held-out window therefore CONTAINS 8.7 trough labels in total, and
8.7 is a ceiling on buy omens that no amount of recall can raise. Pass 111's
four arms reported ``buy_omens`` of 1, 9, 7 and 2 out of 60 held-out bars --
every one of them inside that ceiling. The UP-base cell read
``buy_net_per_trade`` +0.0031 at ``buy_hit_rate`` 1.0, the only positive cell
in the table, and it was ONE TRADE out of a window that could not have held
thirty.

Against HEAD before [3366105d] this whole file fails at import: there was no
``heldout_readability`` and no ``READABLE_LABEL_FLOOR``, and the report wrote
``buy_net_per_trade`` as ``total / max(1, len(trades))`` -- a bare percentage
on n=1, with the n stored in a different key twenty lines away.

The floor is on LABELS as well as on trades because the two say different
things. Too few labels means the WINDOW is too short and the fix is bars. Too
few calls in a window with plenty of labels means RECALL is the problem and
the fix is the brain. A report that shows one number for both sends the next
pass to fix the wrong one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.omen_experiment import (  # noqa: E402
    DEFAULT_HELDOUT_BARS, MIN_READABLE_HELDOUT_BARS_AT_3600S,
    READABLE_LABEL_FLOOR, UNREADABLE, heldout_readability, readability_cell,
    render_readability, validate_report_readability,
)
from trading.omen_brain import OMEN_CREST, OMEN_MURK, OMEN_TROUGH  # noqa: E402
from trading.omen_scoreboard import READABLE_TRADES  # noqa: E402


def _window(bars: int, troughs: int, crests: int = 0):
    """A held-out window of ``bars`` samples holding exactly ``troughs``."""
    labels = ([OMEN_TROUGH] * troughs + [OMEN_CREST] * crests
              + [OMEN_MURK] * (bars - troughs - crests))
    assert len(labels) == bars, "the fixture must fill the window exactly"
    return [{"label": label, "forward": 0.0} for label in labels]


def test_the_label_floor_is_the_same_thirty_as_the_trade_floor():
    """One floor, imported, so the two cannot drift apart in a later edit."""
    assert READABLE_LABEL_FLOOR == READABLE_TRADES == 30


def test_a_sixty_bar_window_writes_unreadable_where_it_used_to_write_a_percentage():
    """THE REGRESSION. Pass 111's UP-base cell: 60 bars, 9 troughs, 1 trade."""
    # 14.46% of 60 bars is 8.7 troughs; 9 is the median window, not a bad one.
    samples = _window(60, troughs=9)
    block = heldout_readability(samples, buy_trades=1, buy_net_total=0.000031)

    # The old behaviour wrote 0.000031 here and called it +0.0031% per trade.
    assert block["buy_net_per_trade"] == UNREADABLE
    assert block["heldout_readable"] is False

    buy = block["readability"]["buy"]
    assert buy["readable"] is False
    assert buy["label_n"] == 9, "the ceiling the window imposed"
    assert buy["trades"] == 1, "the n that must travel with the cell"
    assert buy["floor"] == 30
    # The raw number is kept -- a run must stay reproducible -- but under a
    # name nobody quotes by accident.
    assert buy["net_per_trade_raw"] == pytest.approx(0.000031)
    # BOTH reasons fire at n=1 in a 60-bar window, and each names its own fix.
    why = " ".join(buy["unreadable_because"])
    assert "9" in why and "ceiling" in why
    assert "1 buy calls" in why and "standard error" in why


def test_the_window_ceiling_and_the_call_count_are_separate_verdicts():
    """A long window with too few calls fails on RECALL, not on bars."""
    # 400 bars at the 14.46% median is 58 troughs: the window is fine.
    starved = heldout_readability(_window(400, troughs=58), buy_trades=4,
                                  buy_net_total=0.01)
    buy = starved["readability"]["buy"]
    assert buy["readable"] is False
    assert buy["label_n"] == 58, "the window CAN carry thirty calls"
    reasons = " ".join(buy["unreadable_because"])
    assert "ceiling" not in reasons, "58 labels is above the floor"
    assert "4 buy calls" in reasons

    # And the mirror: enough labels AND enough calls reads as a number.
    readable = heldout_readability(_window(400, troughs=58), buy_trades=40,
                                   buy_net_total=-0.40)
    assert readable["readability"]["buy"]["readable"] is True
    assert readable["buy_net_per_trade"] == pytest.approx(-0.01)
    assert readable["heldout_readable"] is True


def test_the_report_carries_the_base_rate_beside_the_count():
    """The ceiling is MEASURED on this window, never assumed from the median."""
    block = heldout_readability(_window(400, troughs=20, crests=60),
                                buy_trades=12, buy_net_total=0.0)
    assert block["heldout_window_bars"] == 400
    assert block["heldout_trough_labels"] == 20
    assert block["heldout_trough_base_rate"] == pytest.approx(0.05)
    assert block["heldout_crest_labels"] == 60
    assert block["heldout_crest_base_rate"] == pytest.approx(0.15)
    assert block["heldout_label_counts"][OMEN_MURK] == 320
    # A 5% trough rate is a third of the 14.46% median, and the window is
    # 400 bars -- which is exactly why the rate is measured rather than
    # inferred from the bar count.
    assert block["readability"]["buy"]["readable"] is False


def test_the_sell_cell_exists_and_says_it_was_never_scored():
    """An absent cell reads 'not applicable'; the truth is 'never measured'."""
    block = heldout_readability(_window(400, troughs=58, crests=60),
                                buy_trades=40, buy_net_total=-0.4)
    sell = block["readability"]["sell"]
    assert block["sell_net_per_trade"] == UNREADABLE
    assert sell["label_n"] == 60, "the sell cell still carries its n"
    assert sell["scored"] is False
    assert sell["readable"] is False
    assert any("NOT SCORED" in reason for reason in sell["unreadable_because"])
    # The buy half of the same report is readable: the sell verdict is about
    # the sell half only, not a blanket refusal.
    assert block["readability"]["buy"]["readable"] is True


def test_the_floor_and_the_default_window_are_named_in_the_block():
    """A default that lives only in argparse cannot be compared against a run."""
    block = heldout_readability(_window(400, troughs=58), buy_trades=40,
                                buy_net_total=-0.4)
    assert block["readable_label_floor"] == 30
    assert block["min_readable_heldout_bars_at_3600s"] == 208
    # 30 trough calls at the measured 14.46% median needs 207.5 bars at
    # PERFECT recall; the default is set above it, not at it.
    assert DEFAULT_HELDOUT_BARS >= MIN_READABLE_HELDOUT_BARS_AT_3600S
    assert MIN_READABLE_HELDOUT_BARS_AT_3600S == 208


def test_an_empty_window_reports_nothing_rather_than_a_flat_zero():
    """Zero bars is nothing measured, not a measured break-even."""
    block = heldout_readability([], buy_trades=0, buy_net_total=0.0)
    assert block["heldout_trough_base_rate"] is None
    assert block["buy_net_per_trade"] == UNREADABLE
    # No raw number is stashed when there was no number: an empty cell must
    # not offer a 0.0000% anyone could quote as a measured break-even.
    assert "net_per_trade_raw" not in block["readability"]["buy"]


def test_render_never_prints_a_percentage_without_its_n():
    """Same rule as the money scoreboard: the n travels with the number."""
    unreadable = render_readability(
        heldout_readability(_window(60, troughs=9), buy_trades=1,
                            buy_net_total=0.000031))
    for line in unreadable.splitlines():
        if "%" in line and "per trade" in line:
            assert "n=" in line, f"a per-trade percentage without its n: {line}"
    assert UNREADABLE in unreadable
    assert "+0.0031%" not in unreadable, "the quotable number must not appear"

    readable = render_readability(
        heldout_readability(_window(400, troughs=58), buy_trades=40,
                            buy_net_total=-0.4))
    assert "n=40" in readable and "per trade" in readable


def _report(block):
    """A report the way ``main`` builds it: fixed keys, readability spliced."""
    return {"buy_omens": block["readability"]["buy"]["trades"],
            "heldout_default_bars": DEFAULT_HELDOUT_BARS,
            "heldout_window_was_default": False,
            **block}


def test_the_write_guard_refuses_a_bare_percentage_on_a_short_window():
    """The sibling of validate_report_horizon: checked at WRITE time.

    Splice order in ``main`` is what puts the guarded value in the report, and
    splice order is exactly what a later edit reorders without noticing. So
    the guard is on the report, not on the code path that built it.
    """
    block = heldout_readability(_window(60, troughs=9), buy_trades=1,
                                buy_net_total=0.000031)
    report = _report(block)
    validate_report_readability(report)  # the control: the guarded form passes

    # Now do exactly what HEAD did -- write the bare number under the key a
    # reader quotes -- and the report must not reach the directory.
    report["buy_net_per_trade"] = 0.000031
    with pytest.raises(ValueError) as exc:
        validate_report_readability(report)
    assert "buy_net_per_trade" in str(exc.value)
    assert UNREADABLE in str(exc.value)
    assert "label n=9" in str(exc.value)


def test_the_write_guard_refuses_a_report_with_no_ceiling_at_all():
    """A report that never measured its own window cannot be placed."""
    block = heldout_readability(_window(400, troughs=58), buy_trades=40,
                                buy_net_total=-0.4)
    report = _report(block)
    validate_report_readability(report)  # complete, so it passes
    for field in ("heldout_trough_labels", "readability", "heldout_window_bars",
                  "readable_label_floor", "heldout_default_bars"):
        stripped = dict(report)
        stripped.pop(field)
        with pytest.raises(ValueError, match="missing"):
            validate_report_readability(stripped)


def test_the_write_guard_refuses_a_report_that_omitted_the_sell_cell():
    """An absent cell reads 'not applicable'; the truth is 'never measured'."""
    block = heldout_readability(_window(400, troughs=58, crests=60),
                                buy_trades=40, buy_net_total=-0.4)
    report = _report(block)
    report["readability"] = {"buy": report["readability"]["buy"]}
    with pytest.raises(ValueError, match="never"):
        validate_report_readability(report)


def test_a_cell_below_the_floor_cannot_be_built_holding_a_bare_float():
    """The guard is in the constructor, so no caller can route around it."""
    cell = readability_cell("buy", OMEN_TROUGH, 9, 60, trades=1,
                            net_per_trade=0.5)
    assert cell["net_per_trade"] == UNREADABLE
    assert cell["net_per_trade_raw"] == 0.5
