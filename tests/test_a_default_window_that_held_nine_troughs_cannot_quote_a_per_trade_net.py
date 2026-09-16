"""The two harnesses that published a bare per-trade net off a short window.

WHAT THIS PREVENTS, and both halves of it are a real published artifact:

  * ``omen_agreement_census`` defaulted to a 200-bar held-out window. That
    window holds ~29 trough labels on this feed's 14.46% median base rate --
    ONE SHORT of the 30-label floor, and close enough to read as adequate.
    Every ``buy_net_per_trade`` it published was a percentage its own window
    could not carry, and nothing in the report said so.
  * ``omen_scorable_windows`` defaulted to a 60-bar per-corpus window and
    published ``called_net`` per corpus. Measured node-free on 0004_AERO-USDC:
    656 of 656 candidate 60-bar windows hold fewer than 30 troughs, median 9.

Against the OLD behaviour both assertions below fail on a float: ``score``
returned ``total / len(trades)`` unconditionally, and the sweep wrote
``round(edge["called_net"], 6)`` into every row. The floor is imported from
``omen_experiment`` rather than restated, so a cell refused on one harness is
refused on all of them.

No node is contacted and no corpus is read: both halves are pure functions of
counts, which is the point -- a guard that needs :8091 to be checked is a
guard nobody checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.omen_experiment import (  # noqa: E402
    MIN_READABLE_HELDOUT_BARS_AT_3600S, READABLE_LABEL_FLOOR, UNREADABLE,
    readability_cell,
)
from trading.omen_brain import OMEN_TROUGH  # noqa: E402


def _row(action: str, forward: float, *, label: str = OMEN_TROUGH):
    """One held-out answer in the shape ``omen_agreement_census.score`` reads."""
    return {"verdict": "admitted", "label": label, "truth": label,
            "actionable": action == "buy", "action": action,
            "forward": forward, "confidence": 0.5}


def test_the_census_refuses_a_per_trade_net_from_a_two_hundred_bar_window():
    from scripts.omen_agreement_census import score

    # The exact geometry of the published reports: 200 held-out bars holding 29
    # troughs, and an arm that made 40 buy calls -- above the TRADE floor, so
    # the only thing that can refuse this cell is the window's own ceiling.
    rows = [_row("buy", 0.02) for _ in range(40)]
    result = score(rows, "primary", heldout_trough_labels=29,
                   heldout_window_bars=200)

    assert result["buy_omens"] == 40
    assert result["buy_net_per_trade"] == UNREADABLE, (
        "29 trough labels is the CEILING on correct buy calls in this window "
        "and it is below the %d-label floor, so the per-trade mean is a fact "
        "about the calendar" % READABLE_LABEL_FLOOR)
    # The float is kept, under a name that cannot be quoted by accident.
    assert isinstance(result["buy_net_per_trade_raw"], float)
    assert result["readability"]["buy"]["label_n"] == 29
    assert result["readability"]["buy"]["trades"] == 40
    assert not result["readability"]["buy"]["readable"]


def test_the_census_still_quotes_a_window_that_can_carry_it():
    """The guard must not refuse everything: a refusal that never lifts is off."""
    from scripts.omen_agreement_census import score

    rows = [_row("buy", 0.02) for _ in range(40)]
    result = score(rows, "primary", heldout_trough_labels=61,
                   heldout_window_bars=420)

    assert isinstance(result["buy_net_per_trade"], float)
    assert result["readability"]["buy"]["readable"]


def test_the_sweeps_per_corpus_cell_refuses_a_sixty_bar_window():
    """One corpus of the published 2026-09-10 sweep, at its own old default."""
    cell = readability_cell("buy", OMEN_TROUGH, 9, 60, trades=11,
                            net_per_trade=-0.013176)

    assert cell["net_per_trade"] == UNREADABLE
    assert cell["label_n"] == 9 and cell["window_bars"] == 60
    assert any("CONTAINS only 9" in why for why in cell["unreadable_because"])


def test_the_sweeps_pooled_cell_carries_the_ceiling_it_pooled():
    from scripts.omen_scorable_windows import pooled_readability

    # Seventeen 60-bar windows, nine troughs each. Pooling is additive, so the
    # pooled ceiling is 153 and the pooled cell IS readable -- that is honest
    # and it is why the per-corpus cell above is guarded separately. What the
    # pooled block may never do is hide the geometry it was pooled from.
    subset = [{"heldout_trough_labels": 9, "heldout_crest_labels": 8,
               "heldout_window_bars": 60} for _ in range(17)]
    block = pooled_readability(subset, trades=194, net_per_trade=-0.015575)

    assert block["heldout_trough_labels"] == 153
    assert block["heldout_window_bars"] == 1020
    assert block["corpora_pooled"] == 17
    assert block["heldout_trough_base_rate"] == pytest.approx(153 / 1020)
    # The sell half is NOT SCORED by this long-only harness, and an absent cell
    # would read as "not applicable" when what is true is "never measured".
    assert block["sell_net_per_trade"] == UNREADABLE
    assert not block["readability"]["sell"]["scored"]


def test_a_pooled_cell_of_two_short_windows_is_refused():
    from scripts.omen_scorable_windows import pooled_readability

    block = pooled_readability(
        [{"heldout_trough_labels": 9, "heldout_crest_labels": 8,
          "heldout_window_bars": 60} for _ in range(2)],
        trades=14, net_per_trade=-0.0131)

    assert block["buy_net_per_trade"] == UNREADABLE
    assert not block["heldout_readable"]


@pytest.mark.parametrize("module_name", [
    "scripts.omen_agreement_census",
    "scripts.omen_scorable_windows",
    "scripts.omen_layer_probe",
])
def test_the_default_heldout_window_can_hold_thirty_labels(module_name):
    """The default is the measurement: 200, 60 and 180 bars all sat below it.

    Checked through ``build_parser`` rather than by reading the source, so it
    is the default a run would actually get.
    """
    import importlib

    parser = importlib.import_module(module_name).build_parser()
    assert parser.get_default("test") >= MIN_READABLE_HELDOUT_BARS_AT_3600S, (
        "%s defaults to a held-out window that cannot reach the %d-label floor "
        "at this feed's median trough rate" % (module_name,
                                               READABLE_LABEL_FLOOR))


def test_the_shape_harness_default_window_can_hold_thirty_labels():
    """``omen_shape_mutations`` builds its parser inside ``main``.

    So the default is asserted on the constant the parser is built from. It was
    120 bars -- 17.4 troughs at the median rate -- published beside a POWER note
    about a 1.5pp effect, which is a weaker floor answering a different
    question.
    """
    from scripts.omen_shape_mutations import DEFAULT_HELDOUT_TEST_BARS

    assert DEFAULT_HELDOUT_TEST_BARS >= MIN_READABLE_HELDOUT_BARS_AT_3600S
