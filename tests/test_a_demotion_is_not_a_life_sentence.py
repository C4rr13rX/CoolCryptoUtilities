"""A demoted strategy could never be re-armed, so nothing could ever trade live.

MEASURED 2026-09-06 on data/strategy_ledger.json, with LIVE TRADES TODAY: 0 and
``approved_ids()`` returning [] for the whole day. atf_static is the only entry
that has ever had a live execution branch, and it was pinned shut by two
independent copies of one defect -- both rules reading ``live["total_profit"]``,
a LIFETIME sum that a demotion freezes:

    live: 18 trades, 5W/13L, total_profit -0.186371
    demote_reason "live P/L -0.1585 over 17 trades is not profitable"
    demotions 7   graduation_blocked False   ghost 45 trades / 25 wins

  * ``_maybe_rearm_locked``  ``trades(18) >= 3 and net(-0.186371) <= 0`` returns
    before it reads one line of ghost evidence. No quantity of fresh ghost
    trades can ever re-arm it: the method is dead code for this strategy,
    permanently, and ``graduation_blocked`` was never set -- so nothing in the
    ledger says it is locked out for good.
  * ``_evaluate_demotion_locked``  ``trades(18) >= 8 and profit <= 0``
    re-demotes on the next ``record()`` call, so a hand-reinstatement is undone
    within minutes. The ledger records that happening: reinstated_ts
    1788636635 -> demoted_ts 1788642158, 5523s later, seven demotions in all.

That empty ``approved_ids()`` is what makes ``_live_gate_candidates()`` empty,
which silently switches the live gate's subject from "the strategy about to
spend money" to the pooled book of all 36 strategies -- the subject
``_ghost_validation_for_live``'s own docstring calls the wrong one. Every
refusal downstream of it is a symptom.

The drawdown brake sitting sixty lines below had the identical shape and was
already fixed, with ``dd_ref`` re-based in ``_grant_live_licence``; its comment
spells out the argument ("a ratchet with no exit"). It was never applied to the
two rules above it. This file is that argument, as a test.

Nothing here loosens the account's protection, and the last two tests are the
ones that prove it: a strategy that loses money UNDER ITS NEW LICENCE is still
demoted by the same floor on the same sample size, and re-arming still demands
a full graduation-grade ghost book gathered after the demotion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from trading.strategies.ledger import StrategyLedger


# atf_static's live branch as the ledger actually held it at 21:40 on
# 2026-09-06, and the ghost book it had at the same moment.
LIVE_AT_DEMOTION: Dict[str, Any] = {
    "trades": 18,
    "wins": 5,
    "losses": 13,
    "total_profit": -0.186371,
    "peak_profit": 0.2221068671625748,
    "consecutive_losses": 1,
    "dd_ref": -0.15847580194872188,
}


def _ledger(tmp_path: Path, entry: Dict[str, Any]) -> StrategyLedger:
    """A ledger on disk holding one entry, loaded through the real reader."""
    path = tmp_path / "strategy_ledger.json"
    path.write_text(json.dumps({"atf_static": entry}), encoding="utf-8")
    return StrategyLedger(path=str(path))


def _demoted_entry(*, fresh_ghost_trades: int, fresh_ghost_wins: int,
                   fresh_ghost_profit: float) -> Dict[str, Any]:
    """A demoted atf_static carrying a given amount of post-demotion evidence.

    ``pl_ref``/``trades_ref`` are stamped the way ``_demote_locked`` now stamps
    them, because that is the state the running system produces. Their absence
    is covered separately by the legacy test below.
    """
    live = dict(LIVE_AT_DEMOTION)
    live["pl_ref"] = live["total_profit"]
    live["trades_ref"] = live["trades"]
    # The re-arm rule counts evidence over the symbols the LIVE lane could
    # actually have placed, so the fixture has to carry that subset -- see
    # tests/test_a_licence_is_not_earned_on_symbols_the_live_lane_refuses.py.
    # This file is about a different question (a lifetime loss must not veto
    # the rule, and a re-arm must re-base the drawdown brake), so the whole
    # book here is tradeable and the assertions are unchanged.
    return {
        "live": live,
        "ghost": {
            "trades": 20 + fresh_ghost_trades,
            "wins": 8 + fresh_ghost_wins,
            "total_profit": 1.0 + fresh_ghost_profit,
            "tradeable": {
                "trades": 20 + fresh_ghost_trades,
                "wins": 8 + fresh_ghost_wins,
                "total_profit": 1.0 + fresh_ghost_profit,
            },
        },
        "ghost_at_demotion": {
            "trades": 20,
            "wins": 8,
            "total_profit": 1.0,
            "tradeable": {"trades": 20, "wins": 8, "total_profit": 1.0},
        },
        "live_approved": False,
        "demote_reason": "live P/L -0.1585 over 17 trades is not profitable",
        "demotions": 7,
        "graduation_blocked": False,
    }


def test_a_lifetime_loss_does_not_veto_the_rearm_rule(tmp_path: Path) -> None:
    """THE LOCKOUT. Fresh ghost evidence must be able to buy a second licence.

    Against the old code this fails: ``net`` read -0.186371 (lifetime) and the
    method returned at its first test, so 24 winning ghost trades gathered
    after the demotion bought nothing at all.
    """
    ledger = _ledger(
        tmp_path,
        _demoted_entry(
            fresh_ghost_trades=24, fresh_ghost_wins=18, fresh_ghost_profit=0.9
        ),
    )

    ledger._maybe_rearm_locked("atf_static")

    entry = ledger._entry("atf_static")
    assert entry["live_approved"] is True, entry
    assert entry["demote_reason"] is None, entry
    assert "atf_static" in ledger.approved_ids()


def test_a_new_licence_starts_its_pl_clock_at_zero(tmp_path: Path) -> None:
    """Granting a licence re-bases the P/L reference, as it does the drawdown one.

    Without this the strategy is re-armed and then convicted on the next
    ``record()`` by the sum that convicted it before -- the 5523-second
    reinstatement the ledger records.
    """
    ledger = _ledger(
        tmp_path,
        _demoted_entry(
            fresh_ghost_trades=24, fresh_ghost_wins=18, fresh_ghost_profit=0.9
        ),
    )

    ledger._maybe_rearm_locked("atf_static")
    live = ledger._entry("atf_static")["live"]

    assert live["pl_ref"] == pytest.approx(-0.186371)
    assert live["trades_ref"] == 18
    # The lifetime record is preserved, not rewritten. It is what the demote
    # reason quotes and what the dashboards report.
    assert live["total_profit"] == pytest.approx(-0.186371)
    assert live["trades"] == 18
    # And the licence's own record reads empty, which is the whole point.
    assert StrategyLedger._licence_net(live) == pytest.approx(0.0)
    assert StrategyLedger._licence_trades(live) == 0


def test_a_rearmed_strategy_survives_the_next_evaluation(tmp_path: Path) -> None:
    """The re-demotion loop. Re-arming is worthless if record() undoes it.

    Against the old code ``_evaluate_demotion_locked`` read trades=18 >= 8 and
    total_profit=-0.186371 <= 0 and demoted immediately, which is why the real
    ledger shows seven demotions and a reinstatement that lasted 92 minutes.
    """
    ledger = _ledger(
        tmp_path,
        _demoted_entry(
            fresh_ghost_trades=24, fresh_ghost_wins=18, fresh_ghost_profit=0.9
        ),
    )
    ledger._maybe_rearm_locked("atf_static")
    assert ledger._entry("atf_static")["live_approved"] is True

    ledger._evaluate_demotion_locked("atf_static")

    entry = ledger._entry("atf_static")
    assert entry["live_approved"] is True, entry.get("demote_reason")
    assert "atf_static" in ledger.approved_ids()


def test_a_losing_new_licence_is_still_demoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE GUARD. Re-basing must not forgive losses made under the new licence.

    Twelve round trips at -0.02 each, taken after the licence was granted, are
    exactly what the profitability floor exists to catch -- and it catches
    them, on the licence's own record, while the lifetime sum is irrelevant.

    The sample size is pinned to the deployed .env value rather than inherited,
    so this asserts about the rule and not about whichever number the
    environment happens to carry.
    """
    monkeypatch.setenv("STRATEGY_DEMOTE_MIN_LIVE_TRADES", "12")
    live = dict(LIVE_AT_DEMOTION)
    live["pl_ref"] = live["total_profit"]
    live["trades_ref"] = live["trades"]
    # Twelve further live round trips, losing 0.24 between them.
    live["trades"] = 30
    live["total_profit"] = -0.426371
    ledger = _ledger(
        tmp_path,
        {
            "live": live,
            "ghost": {"trades": 44, "wins": 26, "total_profit": 1.9},
            "ghost_at_demotion": {"trades": 20, "wins": 8, "total_profit": 1.0},
            "live_approved": True,
            "demote_reason": None,
            "demotions": 7,
        },
    )

    ledger._evaluate_demotion_locked("atf_static")

    entry = ledger._entry("atf_static")
    assert entry["live_approved"] is False, entry
    assert "-0.2400" in str(entry["demote_reason"]), entry["demote_reason"]
    assert "over 12 trades" in str(entry["demote_reason"]), entry["demote_reason"]


def test_rearming_still_needs_a_full_fresh_ghost_book(tmp_path: Path) -> None:
    """THE PRICE OF A SECOND LICENCE, unchanged.

    atf_static's real position on 2026-09-06: 6 fresh ghost trades, 2 wins.
    That is what it actually has, and it is not enough -- the fix makes
    recovery reachable, it does not hand it over.
    """
    ledger = _ledger(
        tmp_path,
        _demoted_entry(
            fresh_ghost_trades=6, fresh_ghost_wins=2, fresh_ghost_profit=0.6858
        ),
    )

    ledger._maybe_rearm_locked("atf_static")

    entry = ledger._entry("atf_static")
    assert entry["live_approved"] is False, entry
    assert ledger.approved_ids() == []


def test_a_ledger_without_the_reference_is_judged_as_before(tmp_path: Path) -> None:
    """Legacy entries keep their old verdict, exactly as ``_dd_ref`` does.

    An entry written before these fields existed has no licence boundary to
    measure from, so the lifetime record is the only record there is.
    """
    live = dict(LIVE_AT_DEMOTION)
    live.pop("pl_ref", None)
    live.pop("trades_ref", None)

    assert StrategyLedger._licence_net(live) == pytest.approx(-0.186371)
    assert StrategyLedger._licence_trades(live) == 18
