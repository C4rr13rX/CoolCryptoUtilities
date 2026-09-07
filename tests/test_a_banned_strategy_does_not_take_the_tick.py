"""A tick spends one candidate; a condemned one must not be what it spends.

``BusScheduler.evaluate`` gathers candidates from every strategy and returns
exactly ONE directive. Everything else is discarded. So when the winner is a
proposal ``trading/bot.py`` refuses on sight -- because ``strategy_edge_gate``
has banned that strategy outright, or ``symbol_edge_gate`` has banned that
(strategy, symbol) pair -- the tick produces nothing at all, and the eligible
candidate standing behind it never gets asked.

Measured 2026-09-07 over 238 decision cycles in 2h of ``organism_snapshots``,
153 of which carried an ``enter`` directive:

    obv_accumulation@1w  CLANKER-USDC  22  entry-refused-strategy-edge
    obv_accumulation@3d  AERO/JITOSOL  22  entry-refused-strategy-edge
    donchian_breakout@5d COMP-USDC     19  entry-refused-symbol-edge
    atf_static           AERO-USDC     17  entry-refused-symbol-edge
                                       --
                                       84  of 153 (54.9%)

``atf_static`` is the only strategy with a live branch, and re-arming it needs
20 tradeable ghost round trips since its demotion (it has 1). In those two
hours it emitted 33 enter directives and 31 were on AERO-USDC, which it is
banned from -- one entry landed anywhere else.

These tests pin the two halves of the rule:

  * a banned enter candidate does not win the tick, even when it scores
    highest, as long as an eligible candidate exists;
  * an EXIT is never dropped, whatever the ban says. A ban means "do not open
    this", never "do not close this", and this repo has already stranded a live
    position by disarming the bot that held it.
"""

from __future__ import annotations

import time
from typing import Dict, List

import pytest

import trading.scheduler as scheduler_module
from trading.scheduler import BusScheduler, TradeDirective


BANNED_STRATEGY = "obv_accumulation@1w"
BANNED_PAIR_STRATEGY = "atf_static"
ELIGIBLE_STRATEGY = "rsi_reversal"
SYMBOL = "AERO-USDC"


class StubPortfolio:
    def __init__(self, quote_balance: float, base_balance: float, native_balance: float) -> None:
        self.quote_balance = quote_balance
        self.base_balance = base_balance
        self.native_balance = native_balance

    def get_quantity(self, symbol: str, chain: str = "base") -> float:
        if symbol.upper() == "USDC":
            return self.quote_balance
        if symbol.upper() == "AERO":
            return self.base_balance
        return 0.0

    def get_native_balance(self, chain: str = "base") -> float:
        return self.native_balance


def _directive(action: str, strategy_id: str, expected_return: float) -> TradeDirective:
    return TradeDirective(
        action=action,
        symbol=SYMBOL,
        base_token="AERO",
        quote_token="USDC",
        size=1.0,
        target_price=1.05,
        horizon="5m",
        confidence=0.8,
        expected_return=expected_return,
        reason=f"{strategy_id} {action}",
        strategy_id=strategy_id,
    )


def _candidate(action: str, strategy_id: str, score: float) -> Dict[str, object]:
    return {
        "directive": _directive(action, strategy_id, score),
        "score": float(score),
        "meta": {"strategy": strategy_id},
    }


def _pump(scheduler: BusScheduler, steps: int = 32) -> None:
    ts = time.time() - steps * 60.0
    for i in range(steps):
        scheduler._update_state(
            {
                "symbol": SYMBOL,
                "ts": ts + i * 60.0,
                "price": 1.0 * (1.0 + 0.002 * i),
                "volume": 1_000 + 10 * i,
                "chain": "base",
            }
        )


def _install_gates(monkeypatch) -> None:
    """The two real gates, stubbed to the verdicts production actually holds."""

    def strategy_refusal(strategy_id: str):
        if str(strategy_id) == BANNED_STRATEGY:
            return "9 closed round trips at mean return -1.668% vs 0.650% cost (t=-8.39)"
        return None

    def symbol_refusal(symbol: str, strategy_id=None):
        if str(strategy_id or "") == BANNED_PAIR_STRATEGY and str(symbol) == SYMBOL:
            return "17 closed round trips at mean return -0.992% vs 0.650% cost (t=-6.24)"
        return None

    monkeypatch.setattr(scheduler_module, "_strategy_edge_refusal", strategy_refusal)
    monkeypatch.setattr(scheduler_module, "_symbol_edge_refusal", symbol_refusal)


def _scheduler(monkeypatch, candidates: List[Dict[str, object]]) -> BusScheduler:
    sched = BusScheduler(horizons=[("5m", 300), ("30m", 1800)], min_profit=0.01, prefill=False)
    _pump(sched)
    _install_gates(monkeypatch)
    if sched.strategy_registry is None:
        pytest.skip("no strategy registry available in this build")
    monkeypatch.setattr(
        sched.strategy_registry, "evaluate_all", lambda state, ctx: list(candidates)
    )
    # The forecast lane must not add competing candidates of its own, or the
    # assertion below would be about the forecast rather than about the ban.
    monkeypatch.setattr(sched, "_forecast", lambda state: [])
    return sched


def _evaluate(sched: BusScheduler, *, base_balance: float = 0.0):
    portfolio = StubPortfolio(
        quote_balance=1_000.0, base_balance=base_balance, native_balance=1.0
    )
    return sched.evaluate(
        {
            "symbol": SYMBOL,
            "ts": time.time(),
            "price": 1.07,
            "volume": 1_200.0,
            "chain": "base",
        },
        {"direction_prob": 0.75, "exit_conf": 0.72, "net_margin": 0.02},
        portfolio,
    )


def test_a_strategy_wide_ban_does_not_take_the_tick(monkeypatch) -> None:
    """The highest-scoring candidate is banned outright; the tick must not go to it."""
    sched = _scheduler(
        monkeypatch,
        [
            _candidate("enter", BANNED_STRATEGY, 0.90),
            _candidate("enter", ELIGIBLE_STRATEGY, 0.10),
        ],
    )
    directive = _evaluate(sched)
    assert directive is not None, "the eligible candidate was dropped along with the banned one"
    assert directive.strategy_id == ELIGIBLE_STRATEGY, (
        f"the tick was spent on {directive.strategy_id!r}, which the entry gate refuses"
    )


def test_a_pair_ban_does_not_take_the_tick(monkeypatch) -> None:
    """atf_static/AERO-USDC: banned as a pair while the strategy is fine elsewhere."""
    sched = _scheduler(
        monkeypatch,
        [
            _candidate("enter", BANNED_PAIR_STRATEGY, 0.90),
            _candidate("enter", ELIGIBLE_STRATEGY, 0.10),
        ],
    )
    directive = _evaluate(sched)
    assert directive is not None
    assert directive.strategy_id == ELIGIBLE_STRATEGY, (
        f"the tick was spent on {directive.strategy_id!r}/{SYMBOL}, a banned pair"
    )


def test_an_edge_ban_never_drops_an_exit(monkeypatch) -> None:
    """A strategy that must not buy must still be able to sell what it holds."""
    sched = _scheduler(monkeypatch, [_candidate("exit", BANNED_STRATEGY, 0.90)])
    directive = _evaluate(sched, base_balance=10.0)
    assert directive is not None, "an edge ban swallowed an exit directive"
    assert directive.action == "exit"
    assert directive.strategy_id == BANNED_STRATEGY


def test_a_banned_pair_is_not_published_to_the_rotator(monkeypatch) -> None:
    """last_enter_candidates feeds the PortfolioRotator, which has no entry gate.

    Leaving a banned pair in that map re-proposes it on the rotation path,
    where nothing downstream has yet asked either edge gate about it.
    """
    sched = _scheduler(
        monkeypatch,
        [
            _candidate("enter", BANNED_PAIR_STRATEGY, 0.90),
            _candidate("enter", ELIGIBLE_STRATEGY, 0.10),
        ],
    )
    _evaluate(sched)
    published = sched.last_enter_candidates.get(SYMBOL, {}).get("candidates") or []
    ids = {str(getattr(c.get("directive"), "strategy_id", "")) for c in published}
    assert BANNED_PAIR_STRATEGY not in ids, (
        "the rotator was handed a candidate the entry gate refuses"
    )
    assert ELIGIBLE_STRATEGY in ids


def test_every_candidate_banned_names_the_ban_in_the_filter_reason(monkeypatch) -> None:
    """"no_candidates (thresholds not met)" would be a lie when the book banned them."""
    sched = _scheduler(monkeypatch, [_candidate("enter", BANNED_STRATEGY, 0.90)])
    directive = _evaluate(sched)
    assert directive is None
    reason = str(sched.routes[SYMBOL].last_filter_reason or "")
    assert "edge ban" in reason and BANNED_STRATEGY in reason, reason
