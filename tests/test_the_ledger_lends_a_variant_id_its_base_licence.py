"""Splitting a rule's ledger id silently de-armed the only rule that fires.

[c1bd5cb7] Commit 70fc09c gave the BUY-THE-DIP rule in ``trading/scheduler.py``
its own strategy id (``MONEY_BUTTON_DIP_STRATEGY_ID = 'money_button_dip'``,
scheduler.py:48) so its book would stop being pooled with the momentum plugin
that owns the bare ``money_button`` name. The two trade in opposite directions
and the pooled row could describe neither, so the split was right.

The LICENCE did not come with it. ``money_button`` is the single row an
operator hand-approved for live, and ``StrategyLedger.is_live_approved`` was an
exact dict lookup on the id a directive carries. Measured on the real ledger
before the fix::

    is_live_approved('money_button')     = True
    is_live_approved('money_button_dip') = False
    approved_ids()                       = ['money_button']

The dip variant is the one that actually fires -- 333 dip evaluations last pass
against 18 momentum -- so from the next boot the only firing variant would ask
for live eligibility under an id the store has never heard of and be refused
with a bare False that names nothing (``bot.py:_strategy_live_approved``, which
returns ``bool(self.strategy_ledger.is_live_approved(sid))``). The bug was
armed and had not fired only because production had not rebooted since 70fc09c.

These tests pin the narrow repair and, just as much, its BOUNDARY: exactly one
id resolves differently, the licence is derived from the base at call time
rather than copied, and every bar that could refuse the variant on its own
terms still refuses it. No threshold moved and no ledger data was edited.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from trading.scheduler import MONEY_BUTTON_DIP_STRATEGY_ID
from trading.strategies.ledger import _LIVE_APPROVAL_BASE_IDS, StrategyLedger
from trading.strategies.money_button import MoneyButtonStrategy

BASE_ID = "money_button"
DIP_ID = MONEY_BUTTON_DIP_STRATEGY_ID


def _row(
    *,
    live_approved: bool = False,
    graduation_blocked: bool = False,
    demotions: int = 0,
) -> Dict[str, Any]:
    """A ledger row in the shape the real store holds."""
    return {
        "live_approved": live_approved,
        "demotions": demotions,
        "demote_reason": None,
        "graduation_blocked": graduation_blocked,
        "graduation_blocked_reason": None,
        "ghost": {"trades": 0, "wins": 0, "total_profit": 0.0},
        "live": {"trades": 0, "wins": 0, "total_profit": 0.0},
    }


def _ledger(tmp_path: Path, rows: Dict[str, Dict[str, Any]]) -> StrategyLedger:
    """A ledger seeded on an explicit tmp path.

    The tmp path is not decoration: the real store has concurrent writers and
    a live operator approval in it, and a test must neither read its way to a
    pass nor write to it.
    """
    path = tmp_path / "strategy_ledger.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return StrategyLedger(path=str(path))


# ----------------------------------------------------------------------
# The repair
# ----------------------------------------------------------------------


def test_the_dip_variant_inherits_the_base_licence(tmp_path):
    """THE HEADLINE. Fails on 70fc09c: the dip id had no row, so a bare False.

    Note what is NOT seeded: there is no ``money_button_dip`` row at all. That
    is the real state of data/strategy_ledger.json, which holds exactly one
    money_button key.
    """
    ledger = _ledger(tmp_path, {BASE_ID: _row(live_approved=True)})

    assert ledger.is_live_approved(BASE_ID) is True
    assert ledger.is_live_approved(DIP_ID) is True, (
        "the dip variant is the rule that actually fires; with the base "
        "approved it must be allowed to spend, or the operator's live arm is "
        "attached to a rule that never trades"
    )


def test_the_licence_is_derived_from_the_base_not_copied(tmp_path):
    """"Exactly when": demoting the base must de-arm the variant too.

    This is why the table resolves through the base at call time instead of
    stamping ``live_approved`` onto a variant row. A copied row would keep the
    licence after the base lost it, which turns a bookkeeping repair into a
    widening of live eligibility.
    """
    ledger = _ledger(tmp_path, {BASE_ID: _row(live_approved=True)})
    assert ledger.is_live_approved(DIP_ID) is True

    ledger.demote(BASE_ID, "live P/L is not profitable")

    assert ledger.is_live_approved(BASE_ID) is False
    assert ledger.is_live_approved(DIP_ID) is False, (
        "the variant outlived its base's licence: the base row is the licence, "
        "so a demotion there must reach the variant"
    )


def test_an_unapproved_base_lends_nothing(tmp_path):
    """The table is a pointer to a licence, not a licence of its own."""
    ledger = _ledger(tmp_path, {BASE_ID: _row(live_approved=False)})

    assert ledger.is_live_approved(BASE_ID) is False
    assert ledger.is_live_approved(DIP_ID) is False


def test_a_base_that_is_not_in_the_ledger_at_all_lends_nothing(tmp_path):
    ledger = _ledger(tmp_path, {})

    assert ledger.is_live_approved(BASE_ID) is False
    assert ledger.is_live_approved(DIP_ID) is False


# ----------------------------------------------------------------------
# The boundary: nothing else may gain a licence
# ----------------------------------------------------------------------


def test_a_strategy_id_with_no_ledger_approval_is_still_refused(tmp_path):
    """The other half of the acceptance: unknown ids stay refused."""
    ledger = _ledger(
        tmp_path,
        {
            BASE_ID: _row(live_approved=True),
            "rsi_reversal": _row(live_approved=False),
        },
    )

    for sid in ("no_such_strategy_xyz", "rsi_reversal", "unclassified", "", None):
        assert ledger.is_live_approved(sid) is False, f"{sid!r} gained a licence"


def test_no_id_outside_the_table_gains_a_licence(tmp_path):
    """The set equality Echo asked for, at unit scale.

    The near-miss ids are the point. A suffix-stripping rule -- "anything
    before the last underscore is the base" -- would have admitted every one
    of ``atf_static_dip``, ``rsi_reversal_dip`` and ``money_button_dip_v2``
    off an approved ``money_button``/``rsi_reversal``/``atf_static``. An
    explicit table admits one id and no more.
    """
    rows = {
        BASE_ID: _row(live_approved=True),
        "rsi_reversal": _row(live_approved=True),
        "atf_static": _row(live_approved=True),
    }
    ledger = _ledger(tmp_path, rows)

    universe = set(rows) | {
        DIP_ID,
        "money_button_momentum",
        "money_button_dip_v2",
        "money_button_",
        "money_buttondip",
        "MONEY_BUTTON_DIP",
        "atf_static_dip",
        "atf_static_scout",
        "rsi_reversal_dip",
        "rsi_reversal@5h",
        "unclassified",
        "",
        "no_such_strategy_xyz",
    }

    approved = {sid for sid in universe if ledger.is_live_approved(sid)}
    assert approved == {BASE_ID, "rsi_reversal", "atf_static", DIP_ID}, (
        "an id outside _LIVE_APPROVAL_BASE_IDS changed answer: "
        f"{approved ^ {BASE_ID, 'rsi_reversal', 'atf_static', DIP_ID}}"
    )


def test_a_whitespace_padded_variant_id_resolves_the_same_way(tmp_path):
    """The exact-match path strips; the inherited path must strip identically."""
    ledger = _ledger(tmp_path, {BASE_ID: _row(live_approved=True)})

    assert ledger.is_live_approved(f"  {DIP_ID}  ") is True


# ----------------------------------------------------------------------
# Every bar that refuses the variant on its own terms still refuses it
# ----------------------------------------------------------------------


def test_the_variants_own_demotion_still_refuses_it(tmp_path):
    """Inheriting past a demotion would make the base row a way to undo one."""
    ledger = _ledger(
        tmp_path,
        {
            BASE_ID: _row(live_approved=True),
            DIP_ID: _row(live_approved=False, demotions=1),
        },
    )

    assert ledger.is_live_approved(BASE_ID) is True
    assert ledger.is_live_approved(DIP_ID) is False, (
        "the variant was demoted on its own evidence and the base's licence "
        "handed it back"
    )


def test_the_variants_own_permanent_block_still_refuses_it(tmp_path):
    """``block_graduation`` is about what a strategy IS. Nothing outranks it."""
    ledger = _ledger(tmp_path, {BASE_ID: _row(live_approved=True), DIP_ID: _row()})
    ledger.block_graduation(DIP_ID, "no live branch exists")

    assert ledger.is_live_approved(BASE_ID) is True
    assert ledger.is_live_approved(DIP_ID) is False


def test_the_ghost_only_bar_still_refuses_the_variant(tmp_path, monkeypatch):
    """GHOST_ONLY_STRATEGY_IDS names ids that can never spend, variant or not."""
    monkeypatch.setenv("GHOST_ONLY_STRATEGY_IDS", DIP_ID)
    ledger = _ledger(tmp_path, {BASE_ID: _row(live_approved=True)})

    assert ledger.is_live_approved(DIP_ID) is False


def test_a_ghost_only_base_lends_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_ONLY_STRATEGY_IDS", BASE_ID)
    ledger = _ledger(tmp_path, {BASE_ID: _row(live_approved=True)})

    assert ledger.is_live_approved(BASE_ID) is False
    assert ledger.is_live_approved(DIP_ID) is False


# ----------------------------------------------------------------------
# The seams either side of the repair
# ----------------------------------------------------------------------


def test_the_schedulers_constant_is_the_key_the_ledger_resolves(tmp_path):
    """Drift check. The table is keyed by a string the scheduler owns.

    Renaming ``MONEY_BUTTON_DIP_STRATEGY_ID`` without updating the table would
    re-arm the original defect in total silence, so the two ends are asserted
    against each other rather than against two literals.
    """
    assert DIP_ID in _LIVE_APPROVAL_BASE_IDS
    assert _LIVE_APPROVAL_BASE_IDS[DIP_ID] == MoneyButtonStrategy.strategy_id == BASE_ID
    # And the table stays a one-entry allow-list. Growing it is a deliberate
    # act that needs an operator's approval behind it, not a drive-by edit.
    assert set(_LIVE_APPROVAL_BASE_IDS) == {DIP_ID}


def test_the_evidence_books_stay_split(tmp_path):
    """The split 70fc09c bought must survive the licence being shared.

    Sharing a licence is not pooling a book. Outcomes still land under
    whichever id produced them.
    """
    ledger = _ledger(tmp_path, {BASE_ID: _row(live_approved=True)})
    ledger.record(DIP_ID, profit=+0.05, mode="ghost", symbol="AAA-USDC", mirror_registry=False)
    ledger.record(BASE_ID, profit=-0.05, mode="ghost", symbol="AAA-USDC", mirror_registry=False)

    assert ledger.stats(DIP_ID)["ghost"]["total_profit"] == pytest.approx(0.05)
    assert ledger.stats(BASE_ID)["ghost"]["total_profit"] == pytest.approx(-0.05)


def test_the_licence_holders_list_is_unchanged(tmp_path):
    """Names the boundary of this change rather than endorsing it.

    ``approved_ids()`` enumerates rows that HOLD a licence, and
    ``pipeline.py:_live_gate_candidates`` uses that list to pick whose ghost
    book the live gate judges. Adding the variant there would change which
    book is judged -- a separate decision, on its own evidence -- so this
    change deliberately does not. If a later pass decides the dip book is what
    the live gate should judge, this assertion is the one to change, on
    purpose.
    """
    ledger = _ledger(tmp_path, {BASE_ID: _row(live_approved=True)})

    assert ledger.approved_ids() == [BASE_ID]
    assert ledger.any_live_approved() is True


# ----------------------------------------------------------------------
# The real store, read-only
# ----------------------------------------------------------------------


def test_the_real_ledger_arms_the_variant_that_actually_fires():
    """The invariant on the file production boots from. Read-only.

    Asserted as an equality rather than ``is True`` so an operator who
    deliberately demotes money_button does not turn this red: the claim is
    that the two move together, not that either is armed today.
    """
    real = StrategyLedger.DEFAULT_PATH
    if not Path(real).exists():
        pytest.skip(f"no ledger at {real}")

    ledger = StrategyLedger()
    assert ledger.is_live_approved(DIP_ID) == ledger.is_live_approved(BASE_ID)
