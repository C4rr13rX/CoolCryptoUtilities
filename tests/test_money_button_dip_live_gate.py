"""A borrowed live licence is not permission to spend into a losing book.

[f6055c6b] TWO CORRECT FIXES, LANDED TOGETHER, MADE A THIRD THING TRUE.

``f211df8`` restored the live licence that the buy-the-dip rule lost when
``70fc09c`` renamed it to ``money_button_dip``. That repair is right and is not
touched here: a renamed id silently de-arming the only rule that fires is a real
bug, ``_LIVE_APPROVAL_BASE_IDS`` is the fix, and every assertion in
``tests/test_the_ledger_lends_a_variant_id_its_base_licence.py`` still holds.

But read what ``f211df8``'s own commit message offers as the reason to arm it::

    the dip variant is the one that actually fires -- 333 dip evaluations
    last pass against 18 momentum

and then read what this project's own record says about that SAME sample, at
``data/attempts-revenir.md`` lines 7982 and 8054::

    money_button_dip   n=333   gross -0.3816%   loses BEFORE cost   CLOSED NEGATIVE
    money_button       n=18    gross -0.0184%   t=-0.10, unmeasured; loss is 100% cost

The arm was justified by the fire COUNT. Nobody checked the SIGN of those 333
fires. So the single id that holds a live licence today lends it to a rule our
own evidence closes as losing before a penny of cost is charged.

WHAT WAS MISSING IS NOT THE LICENCE. It is a check sited BETWEEN holding a
licence and placing an order. A licence answers "is this rule allowed to spend?"
and nothing anywhere asked "and does it make money when it does?".

These tests pin that gate and, just as hard, its BOUNDARY:

  * the licence SURVIVES -- ``is_live_approved('money_button_dip')`` is still
    True, so ``f211df8`` is preserved exactly and nothing was reverted;
  * the ORDER does not -- ``bot.py:_strategy_live_approved``, the one function
    every money path in ``bot.py`` consults, refuses it;
  * the refusal is a MEASUREMENT of the ledger's own recorded book, not a
    constant. Seed a positive book and the same code permits the same order.
    ``-0.3816`` appears nowhere in the implementation, and this file is where
    that is proven;
  * the momentum variant -- the bare ``money_button`` id the operator actually
    hand-approved -- is UNAFFECTED, because it spends on its own licence rather
    than on a borrowed one;
  * ghost firing is untouched, so the variant keeps earning the evidence that
    can one day overturn the refusal.

FAILS ON HEAD at ``564caf1``: ``_strategy_live_approved`` returns
``bool(strategy_ledger.is_live_approved(sid))`` and nothing else, so the live
order is ADMITTED on a licence borrowed from a row whose recorded book is
2 ghost round trips for -$0.0161.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from trading.scheduler import MONEY_BUTTON_DIP_STRATEGY_ID
from trading.strategies.ledger import _LIVE_APPROVAL_BASE_IDS, StrategyLedger

BASE_ID = "money_button"
DIP_ID = MONEY_BUTTON_DIP_STRATEGY_ID

#: The money_button row as data/strategy_ledger.json actually holds it, read
#: 2026-09-17 at 564caf1. This is the book the next production boot will judge,
#: so it is the book these tests are written against rather than a tidy
#: invention. Two ghost round trips, one win, one loss, net negative; no live
#: round trips at all.
REAL_GHOST_TRADES = 2
REAL_GHOST_WINS = 1
REAL_GHOST_PROFIT = -0.016149119465853047


def _mode(trades: int = 0, wins: int = 0, total_profit: float = 0.0) -> Dict[str, Any]:
    return {
        "trades": int(trades),
        "wins": int(wins),
        "losses": max(int(trades) - int(wins), 0),
        "total_profit": float(total_profit),
        "conf_ema": 0.0,
        "peak_profit": 0.0,
        "max_drawdown": 0.0,
        "consecutive_losses": 0,
        "last_ts": 0.0,
    }


def _row(
    *,
    live_approved: bool = False,
    graduation_blocked: bool = False,
    demotions: int = 0,
    ghost: Optional[Dict[str, Any]] = None,
    live: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A ledger row in the shape the real store holds.

    Same helper as ``test_the_ledger_lends_a_variant_id_its_base_licence.py``,
    with the two BOOKS opened up. That file always seeds an empty book, which
    is precisely why it cannot see this defect: an empty book measures nothing,
    and a gate on a measurement has to stay silent when there is none.
    """
    return {
        "live_approved": live_approved,
        "demotions": demotions,
        "demote_reason": None,
        "graduation_blocked": graduation_blocked,
        "graduation_blocked_reason": None,
        "ghost": ghost if ghost is not None else _mode(),
        "live": live if live is not None else _mode(),
    }


def _ledger(tmp_path: Path, rows: Dict[str, Dict[str, Any]]) -> StrategyLedger:
    """A ledger seeded on an explicit tmp path.

    Never the real store: it has concurrent writers and a live operator
    approval in it, and a test must neither read its way to a pass nor write
    to it.
    """
    path = tmp_path / "strategy_ledger.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return StrategyLedger(path=str(path))


def _order_path_admits(ledger: StrategyLedger, sid: str) -> bool:
    """Ask the LIVE ORDER PATH, not the licence.

    ``bot.py:_strategy_live_approved`` is the single function behind every seam
    in ``bot.py`` that spends real money -- ``entry_will_be_simulated``,
    ``entry_spends_real_money``, ``entry_is_live`` and the ``live_approved``
    conjunction all read it. Driving the real function with a real
    ``StrategyLedger`` is the difference between testing a gate and testing
    that a gate was written; a helper nothing calls would pass a unit test and
    still place the order.
    """
    from trading.bot import TradingBot

    return TradingBot._strategy_live_approved(
        SimpleNamespace(strategy_ledger=ledger), SimpleNamespace(strategy_id=sid)
    )


@pytest.fixture(autouse=True)
def _armed_live_lane(monkeypatch):
    """The state the next restart boots into: operator-armed, graduation enforced.

    ``.env`` carries ``ENABLE_LIVE_TRADING=1`` -- the operator armed the lane on
    2026-09-16. Under the arm ``_strategy_live_approved`` is the ONLY gate left
    between a directive and a swap, which is what makes this item p1.
    """
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "1")
    monkeypatch.delenv("LIVE_REQUIRE_MODEL_READY", raising=False)
    monkeypatch.delenv("STRATEGY_GRADUATION_ENFORCED", raising=False)
    monkeypatch.delenv("GHOST_ONLY_STRATEGY_IDS", raising=False)
    monkeypatch.delenv("STRATEGY_LIVE_ORDER_MIN_TRADES", raising=False)


# ----------------------------------------------------------------------
# The headline: the licence survives, the order does not
# ----------------------------------------------------------------------


def test_the_licence_survives_and_the_live_order_does_not(tmp_path):
    """THE ITEM, in one test, on the ledger production actually holds.

    Seeded exactly as ``f211df8``'s own test seeds it -- ONE ``money_button``
    row, NO ``money_button_dip`` row, which is the real shape of
    data/strategy_ledger.json -- with that row carrying the book the real file
    carries. Both halves are asserted together because the whole point is that
    they disagree: taking the licence away would have been the wrong fix, and
    leaving the order alone was the defect.
    """
    ledger = _ledger(
        tmp_path,
        {
            BASE_ID: _row(
                live_approved=True,
                ghost=_mode(REAL_GHOST_TRADES, REAL_GHOST_WINS, REAL_GHOST_PROFIT),
            )
        },
    )

    # (a) f211df8 is preserved. The renamed id still resolves to the operator's
    #     licence, so the rule that actually fires is not silently de-armed.
    assert ledger.is_live_approved(DIP_ID) is True, (
        "f211df8 was reverted: the dip variant lost the base licence again, "
        "which is the original [c1bd5cb7] defect and NOT what this item asks for"
    )

    # (b) ...and it still may not spend, because the book backing that licence
    #     loses money.
    assert _order_path_admits(ledger, DIP_ID) is False, (
        "money_button_dip placed a LIVE order on a borrowed licence whose "
        "recorded book is negative -- the exact state the next restart creates"
    )


def test_the_refusal_names_the_measurement(tmp_path):
    """A silent refusal is the defect f211df8's own commit message complains about.

    ``bot.py:_strategy_live_approved`` "returns a bare False and nothing logs a
    refusal that names the id". A second unnamed False would be the same bug
    with the sign flipped, so the reason has to carry the id, the sample size
    and the number.
    """
    ledger = _ledger(
        tmp_path,
        {
            BASE_ID: _row(
                live_approved=True,
                ghost=_mode(REAL_GHOST_TRADES, REAL_GHOST_WINS, REAL_GHOST_PROFIT),
            )
        },
    )

    reason = ledger.live_order_refusal(DIP_ID)
    assert reason, "the order was refused with no reason at all"
    assert DIP_ID in reason
    assert BASE_ID in reason, "the reason must name the row that lent the licence"
    assert "n=" in reason, "a refusal on evidence must state how much evidence"


# ----------------------------------------------------------------------
# It is a MEASUREMENT: seed a different book and the answer changes
# ----------------------------------------------------------------------


def test_a_proven_base_book_lends_a_spendable_licence(tmp_path):
    """The gate is not a ban on this id wearing a measurement's clothes.

    Same code, same ids, same borrowed licence -- only the recorded book
    differs, and the order is ADMITTED. This is the assertion that a later pass
    should run first if it suspects a constant crept in: this repo's single most
    repeated defect is a literal masquerading as a measurement (six
    ``TOKEN_ROUND_TRIP`` entries, ``roundtrip_cost``'s flat rate, the forge's
    60bp), and every one of them would have passed a test that only ever asked
    for a refusal.
    """
    ledger = _ledger(
        tmp_path,
        {
            BASE_ID: _row(
                live_approved=True,
                ghost=_mode(trades=40, wins=24, total_profit=+1.25),
            )
        },
    )

    assert ledger.is_live_approved(DIP_ID) is True
    assert ledger.live_order_refusal(DIP_ID) is None
    assert _order_path_admits(ledger, DIP_ID) is True, (
        "a borrowed licence backed by 40 profitable round trips was refused: "
        "the gate is reading something other than the book"
    )


def test_the_variant_is_refused_on_its_OWN_negative_book(tmp_path):
    """The durable refusal, once the variant has a book of its own.

    Today the dip id has no recorded outcomes anywhere -- it was named less
    than a day ago and production has not rebooted since -- so the only
    evidence behind its licence is the lender's. The moment it books its own
    round trips, its own book answers, and a profitable LENDER must not
    launder a losing BORROWER.
    """
    ledger = _ledger(
        tmp_path,
        {
            BASE_ID: _row(live_approved=True, ghost=_mode(40, 24, +1.25)),
            DIP_ID: _row(ghost=_mode(trades=333, wins=150, total_profit=-1.27)),
        },
    )

    assert ledger.is_live_approved(DIP_ID) is True
    reason = ledger.live_order_refusal(DIP_ID) or ""
    assert _order_path_admits(ledger, DIP_ID) is False
    assert "333" in reason, f"the variant's own 333 round trips are not in {reason!r}"


def test_a_live_loss_is_judged_as_well_as_a_ghost_loss(tmp_path):
    """Both books are recorded outcomes and both are evidence.

    Reading only ``ghost`` would let a rule that lost REAL money keep spending
    as long as its simulation looked fine, which inverts which book matters.
    """
    ledger = _ledger(
        tmp_path,
        {
            BASE_ID: _row(
                live_approved=True,
                ghost=_mode(30, 20, +0.40),
                live=_mode(12, 2, -0.90),
            )
        },
    )

    assert _order_path_admits(ledger, BASE_ID) is False
    assert "live" in (ledger.live_order_refusal(BASE_ID) or "")


# ----------------------------------------------------------------------
# The boundary: what must NOT change
# ----------------------------------------------------------------------


def test_the_momentum_variant_the_operator_approved_is_unaffected(tmp_path):
    """Do not widen or narrow what the operator approved.

    ``money_button`` is the single row a human hand-approved, on its own
    evidence, in full knowledge of it ("Dune (exec) pass 424, on USER
    instruction 2026-09-16 13:10"). It spends on ITS OWN licence, not a
    borrowed one, so this gate does not re-litigate that decision -- and
    refusing both variants would close the live lane outright, which is a
    different act needing a different mandate.

    Note what this means in production TODAY: the momentum rule keeps its
    eligibility on exactly the 2-round-trip book that refuses the dip variant,
    because the dip is spending on borrowed evidence and the momentum rule is
    not.
    """
    ledger = _ledger(
        tmp_path,
        {
            BASE_ID: _row(
                live_approved=True,
                ghost=_mode(REAL_GHOST_TRADES, REAL_GHOST_WINS, REAL_GHOST_PROFIT),
            )
        },
    )

    assert ledger.is_live_approved(BASE_ID) is True
    assert ledger.live_order_refusal(BASE_ID) is None
    assert _order_path_admits(ledger, BASE_ID) is True, (
        "the gate became a blanket disable of money_button -- that closes the "
        "live lane entirely and is not what this item asks for"
    )


def test_f211df8s_own_fixture_still_spends(tmp_path):
    """An EMPTY book must not refuse: no measurement is not a measurement of zero.

    This is the exact seed of
    ``test_the_dip_variant_inherits_the_base_licence``. If the gate refused
    here it would be refusing on the ABSENCE of evidence, every one of
    f211df8's cases would turn red, and no strategy could ever place its first
    live order.
    """
    ledger = _ledger(tmp_path, {BASE_ID: _row(live_approved=True)})

    assert ledger.is_live_approved(DIP_ID) is True
    assert ledger.live_order_refusal(DIP_ID) is None
    assert _order_path_admits(ledger, DIP_ID) is True


def test_an_unapproved_id_gains_nothing(tmp_path):
    """The gate only ever narrows. It can turn a True into a False, never back."""
    ledger = _ledger(
        tmp_path,
        {
            BASE_ID: _row(live_approved=True, ghost=_mode(40, 24, +1.25)),
            "rsi_reversal": _row(live_approved=False, ghost=_mode(40, 30, +9.0)),
        },
    )

    for sid in ("rsi_reversal", "no_such_strategy_xyz", "unclassified", "", None):
        assert ledger.may_place_live_order(sid) is False, f"{sid!r} gained a licence"
        if sid:
            assert _order_path_admits(ledger, sid) is False


def test_exactly_one_id_changes_answer_on_the_real_ledger():
    """The same census f211df8 ran, on the other side of the gate. Read-only.

    f211df8 proved over a 65-id universe that exactly one id GAINED a licence.
    This proves exactly one id LOSES the ability to spend it. A gate that
    quietly de-armed a second strategy would be the mirror of the defect it is
    fixing.
    """
    real = StrategyLedger.DEFAULT_PATH
    if not Path(real).exists():
        pytest.skip(f"no ledger at {real}")

    ledger = StrategyLedger()
    holders = {sid for sid in ledger.snapshot() if ledger.is_live_approved(sid)}
    holders.add(DIP_ID)

    refused = {sid for sid in holders if not ledger.may_place_live_order(sid)}
    assert refused <= {DIP_ID}, (
        f"the gate refused more than the dip variant: {sorted(refused)}"
    )


# ----------------------------------------------------------------------
# When the gate is evaluated, and what it does NOT stop
# ----------------------------------------------------------------------


def test_the_gate_is_evaluated_at_order_time_not_at_arm_time(tmp_path):
    """Arm-time only would ship this repo's demotion defect a second time.

    ``demotion-outlived-the-number-that-caused-it``: a verdict computed once
    and cached outlives the evidence behind it. Here the SAME ledger object,
    with no re-arm, no restart and no licence change, answers differently as
    soon as the book it is reading moves -- in both directions.
    """
    ledger = _ledger(
        tmp_path,
        {BASE_ID: _row(live_approved=True, ghost=_mode(30, 12, -0.50))},
    )

    assert _order_path_admits(ledger, DIP_ID) is False

    for _ in range(30):
        ledger.record(BASE_ID, profit=+0.05, mode="ghost", symbol="AAA-USDC",
                      mirror_registry=False)

    assert _order_path_admits(ledger, DIP_ID) is True, (
        "the refusal outlived the book that caused it"
    )

    for _ in range(40):
        ledger.record(BASE_ID, profit=-0.05, mode="ghost", symbol="AAA-USDC",
                      mirror_registry=False)

    assert _order_path_admits(ledger, DIP_ID) is False, (
        "the permission outlived the book that earned it"
    )


def test_ghost_evidence_collection_is_not_blocked(tmp_path):
    """The refusal must not starve the only thing that can overturn it.

    A gate that also stopped the variant trading in SIMULATION would make its
    own verdict permanent: the evidence that reverses it is exactly the
    evidence it would be preventing.
    """
    ledger = _ledger(
        tmp_path,
        {
            BASE_ID: _row(
                live_approved=True,
                ghost=_mode(REAL_GHOST_TRADES, REAL_GHOST_WINS, REAL_GHOST_PROFIT),
            )
        },
    )

    assert _order_path_admits(ledger, DIP_ID) is False

    ledger.record(DIP_ID, profit=+0.05, mode="ghost", symbol="AAA-USDC",
                  mirror_registry=False)

    assert ledger.stats(DIP_ID)["ghost"]["trades"] == 1
    assert ledger.stats(DIP_ID)["ghost"]["total_profit"] == pytest.approx(0.05)
    assert ledger.stats(BASE_ID)["ghost"]["trades"] == REAL_GHOST_TRADES, (
        "the books were pooled; 70fc09c's split is what keeps them apart"
    )


def test_the_licence_holders_list_is_still_unchanged(tmp_path):
    """``approved_ids()`` picks whose ghost book the live GATE judges.

    f211df8 deliberately left it alone and so does this. The gate refuses an
    ORDER; it does not edit a row, revoke an approval, or move which book the
    graduation machinery reads.
    """
    ledger = _ledger(
        tmp_path,
        {
            BASE_ID: _row(
                live_approved=True,
                ghost=_mode(REAL_GHOST_TRADES, REAL_GHOST_WINS, REAL_GHOST_PROFIT),
            )
        },
    )

    assert _order_path_admits(ledger, DIP_ID) is False
    assert ledger.approved_ids() == [BASE_ID]
    assert ledger.any_live_approved() is True
    assert ledger.stats(BASE_ID)["live_approved"] is True
    assert ledger.stats(BASE_ID)["demotions"] == 0
    assert DIP_ID not in ledger.snapshot(), (
        "the gate wrote a row for an id it was only asked about"
    )


def test_the_table_f211df8_added_is_still_the_only_way_in(tmp_path):
    """Drift check across the two commits.

    The gate reads the SAME variant->base table the licence resolves through,
    so the two cannot come apart: a future entry in that table inherits the
    expectancy check for free, and removing the table removes both at once.
    """
    assert set(_LIVE_APPROVAL_BASE_IDS) == {DIP_ID}
    assert _LIVE_APPROVAL_BASE_IDS[DIP_ID] == BASE_ID
