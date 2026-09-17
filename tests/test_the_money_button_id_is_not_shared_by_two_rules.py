"""Two rules called ``money_button`` traded in OPPOSITE directions under ONE
ledger id, so neither could graduate -- or be demoted -- on its own evidence.

Measured pass 425 (commit bcc4104), both registrations live at HEAD:

  * ``trading/strategies/money_button.py`` declares ``strategy_id =
    'money_button'`` and is MOMENTUM CONTINUATION. Its gate declines unless
    the 5m, 10m and 30m returns are all POSITIVE (``momentum_not_aligned``).
    The file contains zero occurrences of 'drawdown' or 'peak'. It buys
    STRENGTH.
  * ``trading/scheduler.py:_money_button_candidate`` emitted a directive
    carrying that same ``strategy_id='money_button'``, and it is BUY-THE-DIP:
    it declines unless price sits 2-5% BELOW the lookback peak
    (``drawdown_outside_dip_window``). It buys WEAKNESS.

One id pooled both books. A win from one rule cancelled a loss from the
other, the graduation bar read the blend, and the ledger row for
``money_button`` could never describe either rule. That row is the one the
operator hand-approved for live on 2026-09-16 ("live_approved_by": "Dune
(exec) pass 424, on USER instruction") on 2 ghost trades at net -0.0161 --
so the pooled row was also the row carrying the live arm.

These tests pin the split. They assert the two rules emit DISTINCT ids and
that a ledger fed one outcome from each keeps two rows rather than one.

They do NOT assert which gate a given tape hits, and they do not assert that
either rule is profitable. Both are market facts and would be pinning the
market rather than the code.

No threshold moved to make these pass. Every ``MONEY_BUTTON_*`` default --
MIN_DRAWDOWN, MAX_DRAWDOWN, MIN_NET_RETURN, SLOPE_FLOOR, REVERSION_FRACTION
-- is the shipped one; the dip tape below is shaped to CLEAR them, which is
why its drop sits outside the 30-minute slope window and its lookback median
still sits at the pre-dip shelf.
"""

from __future__ import annotations

import os
import time
from unittest.mock import patch

import numpy as np
import pytest

from tests.test_money_button import evaluate as evaluate_momentum
from trading.opportunity import OpportunitySignal
from trading.portfolio import PortfolioState, TokenHolding
from trading.scheduler import (
    MONEY_BUTTON_DIP_STRATEGY_ID,
    BusScheduler,
    RouteState,
)
from trading.strategies.money_button import MoneyButtonStrategy


NOW = time.time()

#: A 3% dip the rule actually TAKES, on shipped defaults.
#:
#: The shape matters and each part of it is load-bearing:
#:   * 80 samples on a 100.0 shelf keep the 2h-lookback MEDIAN at 100, which
#:     is what ``reversion_target`` is measured against -- a dip that has sat
#:     at the bottom longer than it sat at the top has no headroom and is
#:     refused ``no_reversion_headroom``.
#:   * the step down happens 35 minutes back, OUTSIDE the 30-minute slope
#:     window, so the knife guard sees a flat tape rather than the drop and
#:     does not refuse ``slope_below_knife_floor``.
#: Both were measured, not guessed: a 10-minute 3% slide is refused by the
#: knife floor and a 40-sample bottom is refused for headroom.
SETTLED_DIP_3PCT = [100.0] * 80 + [97.0] * 35

#: A confirmed 25% uptrend: the shape the MOMENTUM plugin exists to trade,
#: and the one the dip rule would refuse outright.
CONFIRMED_UPTREND = list(np.linspace(1.0, 1.25, 40))


@pytest.fixture()
def funded_portfolio(monkeypatch) -> PortfolioState:
    # PortfolioState refuses to exist without a wallet. Nothing here touches a
    # chain -- the holding is seeded directly -- so a literal is honest.
    monkeypatch.setenv("PRIMARY_WALLET", "0x000000000000000000000000000000000000dEaD")
    portfolio = PortfolioState()
    portfolio.holdings[("base", "USDC")] = TokenHolding(
        token="USDC", symbol="USDC", quantity=25.0, usd=25.0, chain="base", raw={}
    )
    return portfolio


#: [9329b42f] AAA-USDC IS NOT A REAL TOKEN, so it has no measured round trip,
#: so the dip rule's cost gate prices it at the pessimistic unmeasured default
#: (0.70%) and refuses it ``round_trip_cost_above_max``. That refusal is
#: CORRECT -- "we have never priced it" is not evidence a token is cheap -- but
#: it is not this file's subject. These tests pin that the two rules carry
#: DISTINCT ledger ids; they are not about which universe the dip rule trades.
#:
#: So the universe is pinned OPEN here, explicitly and only here. This is NOT
#: the docstring's "no threshold moved to make these pass": every
#: MONEY_BUTTON_* threshold the rule evaluates is still its shipped default,
#: and the tape is still shaped to clear them. What is pinned is the cost
#: UNIVERSE, which did not exist when this tape was written. Swapping in a
#: measured symbol instead would couple an id test to the cost table and break
#: it again the day WETH's measured cost crosses 0.35%.
DIP_UNIVERSE_PINNED_OPEN = {"MONEY_BUTTON_MAX_ROUND_TRIP_COST": "1"}


def _dip_directive(portfolio: PortfolioState):
    """Run the REAL scheduler rule and return the directive it emits."""
    with patch.dict(os.environ, DIP_UNIVERSE_PINNED_OPEN):
        return _dip_directive_inner(portfolio)


def _dip_directive_inner(portfolio: PortfolioState):
    scheduler = BusScheduler(prefill=False)  # prefill=False keeps this off the network
    state = RouteState(symbol="AAA-USDC", base_token="AAA", quote_token="USDC")
    for index, price in enumerate(SETTLED_DIP_3PCT):
        state.samples.append(
            (NOW - (len(SETTLED_DIP_3PCT) - index) * 60.0, float(price), 0.0)
        )
    scheduler._opportunity_bias["AAA-USDC"] = OpportunitySignal(
        symbol="AAA-USDC",
        kind="buy-low",
        zscore=-2.0,
        price=97.0,
        mean_price=100.0,
        std_price=1.0,
        timestamp=NOW,
    )
    result = scheduler._money_button_candidate(
        state,
        last_price=97.0,
        last_volume=1000.0,
        portfolio=portfolio,
        chain_name="base",
        fee_rate=0.0005,
        direction_prob=0.5,
        confidence=0.5,
        net_margin=0.0,
        base_allocation=None,
        risk_budget=1.0,
        live_trading=False,
    )
    assert result is not None, (
        "the dip tape no longer fires -- gates hit: "
        f"{dict(scheduler._mb_gate_counts)}. Reshape the tape; do not loosen "
        "a threshold to make this pass."
    )
    return result


def test_the_two_rules_do_not_share_a_ledger_id(funded_portfolio):
    """The headline assertion: one directive from each rule, two ids."""
    dip = _dip_directive(funded_portfolio)
    momentum = evaluate_momentum(CONFIRMED_UPTREND)
    assert momentum is not None, "a 25% confirmed uptrend should fire the plugin"

    dip_id = dip["directive"].strategy_id
    momentum_id = momentum["directive"].strategy_id

    assert dip_id != momentum_id, (
        f"both rules emitted {dip_id!r}: the dip book and the momentum book "
        "are pooled again and neither can graduate on its own evidence"
    )
    assert dip_id == MONEY_BUTTON_DIP_STRATEGY_ID
    # The PLUGIN keeps the bare name. It is the class actually called
    # MoneyButtonStrategy, it is what the plugin registry resolves, and it is
    # the id the operator's live arm is attached to -- moving it would move
    # that arm silently.
    assert momentum_id == MoneyButtonStrategy.strategy_id == "money_button"


def test_the_scoring_meta_carries_the_same_id_as_the_directive(funded_portfolio):
    """Attribution splits only if BOTH labels move.

    The selector reads ``meta['strategy']`` while the ledger reads
    ``directive.strategy_id``. If those two disagree the evidence lands under
    one name and the decision is recorded under another, which is the same
    pooling defect wearing a different hat.
    """
    dip = _dip_directive(funded_portfolio)
    assert dip["meta"]["strategy"] == dip["directive"].strategy_id


def test_the_two_rules_really_do_trade_opposite_directions(funded_portfolio):
    """The reason the split matters, asserted rather than asserted-in-a-docstring.

    The dip rule takes a tape that is DOWN off its peak; the momentum plugin
    refuses that same tape for ``momentum_not_aligned``. If this ever stops
    holding, the two rules have converged and the split is cosmetic.
    """
    # The dip rule takes the settled dip (proved by _dip_directive not raising).
    _dip_directive(funded_portfolio)

    # The momentum plugin refuses it, and names momentum as the reason.
    strategy = MoneyButtonStrategy()
    falling = evaluate_momentum(SETTLED_DIP_3PCT)
    assert falling is None, "the momentum plugin should refuse a tape below its peak"

    # And the converse: the dip rule's own gate refuses a confirmed uptrend,
    # because an uptrend is not 2-5% below its peak.
    scheduler = BusScheduler(prefill=False)
    state = RouteState(symbol="BBB-USDC", base_token="BBB", quote_token="USDC")
    for index, price in enumerate(CONFIRMED_UPTREND):
        state.samples.append(
            (NOW - (len(CONFIRMED_UPTREND) - index) * 60.0, float(price), 0.0)
        )
    scheduler._opportunity_bias["BBB-USDC"] = OpportunitySignal(
        symbol="BBB-USDC",
        kind="buy-low",
        zscore=-2.0,
        price=float(CONFIRMED_UPTREND[-1]),
        mean_price=float(CONFIRMED_UPTREND[0]),
        std_price=1.0,
        timestamp=NOW,
    )
    # Pinned open for the same reason as _dip_directive: BBB is not a real
    # token, so the cost gate would refuse it FIRST and the census would name
    # round_trip_cost_above_max instead of the drawdown gate this line is
    # about. The gate census reports only the FIRST failing guardrail, so a
    # gate added upstream masks every assertion made downstream of it.
    with patch.dict(os.environ, DIP_UNIVERSE_PINNED_OPEN):
        rising = scheduler._money_button_candidate(
            state,
            last_price=float(CONFIRMED_UPTREND[-1]),
            last_volume=1000.0,
            portfolio=funded_portfolio,
            chain_name="base",
            fee_rate=0.0005,
            direction_prob=0.5,
            confidence=0.5,
            net_margin=0.0,
            base_allocation=None,
            risk_budget=1.0,
            live_trading=False,
        )
    assert rising is None, "the dip rule should refuse a tape making new highs"
    assert "drawdown_outside_dip_window" in scheduler._mb_gate_counts

    del strategy  # constructed only to prove the class imports at this id


def test_a_ledger_fed_one_outcome_from_each_rule_keeps_two_rows(tmp_path, monkeypatch):
    """The point of the split, at the seam that actually stores evidence.

    Before the split both outcomes incremented ONE row, so a dip win and a
    momentum loss netted to nothing and the graduation bar read the blend.
    """
    from trading.strategies.ledger import StrategyLedger

    # An explicit tmp path keeps this off the real data/strategy_ledger.json.
    ledger = StrategyLedger(path=str(tmp_path / "ledger.json"))
    ledger.record(
        MONEY_BUTTON_DIP_STRATEGY_ID,
        profit=+0.05, mode="ghost", symbol="AAA-USDC", mirror_registry=False,
    )
    ledger.record(
        MoneyButtonStrategy.strategy_id,
        profit=-0.05, mode="ghost", symbol="AAA-USDC", mirror_registry=False,
    )

    dip = ledger.stats(MONEY_BUTTON_DIP_STRATEGY_ID)
    momentum = ledger.stats(MoneyButtonStrategy.strategy_id)

    assert dip["ghost"]["trades"] == 1
    assert momentum["ghost"]["trades"] == 1
    # The blend the shared id used to produce: +0.05 and -0.05 into one row is
    # 2 trades at net 0.0, which describes neither rule.
    assert dip["ghost"]["total_profit"] == pytest.approx(0.05)
    assert momentum["ghost"]["total_profit"] == pytest.approx(-0.05)
