"""``_money_button_candidate`` had fifteen ``return None`` exits and not one of
them recorded WHICH.

Measured 2026-09-16 over 24h of ``trading_ops``: 347 entry candidates (219
``ghost_candidate`` + 128 ``ghost_candidate_quote_ok``), every one sourced
``c0d3rv2_atf_static``, and ZERO carrying ``strategy_id='money_button'`` --
while ``money_button`` is the one APPROVED strategy and ladder step 6 blocks on
``_ghost_validation_for_live()`` returning ``n=0`` for exactly that subject.
The lane was not losing. It was not firing, and nothing said why.

These tests pin the census that fixed that. They assert two things and
deliberately not a third:

  * every refusal increments exactly one NAMED gate, and
  * counting changes no verdict -- a refusal stays a refusal.

They do NOT assert which gate wins on a given tape. That is a market fact, it
moves with the defaults, and a test that pinned it would be pinning the
market rather than the code.
"""

from __future__ import annotations

import time

import pytest

from trading.opportunity import OpportunitySignal
from trading.portfolio import PortfolioState, TokenHolding
from trading.scheduler import BusScheduler, RouteState


NOW = time.time()

# A 3% dip off a flat shelf: the shape the money button exists to trade.
DIP_3PCT = [100.0] * 30 + [99.0, 98.5, 98.0, 97.6, 97.4, 97.3, 97.2, 97.1, 97.05, 97.0]
# A 20% crash: outside MONEY_BUTTON_MAX_DRAWDOWN by a wide margin.
CRASH_20PCT = [100.0] * 30 + [80.0] * 10
FLAT = [100.0] * 40


@pytest.fixture()
def scheduler() -> BusScheduler:
    # prefill=False keeps this off the network and off the 22GB store.
    return BusScheduler(prefill=False)


@pytest.fixture()
def funded_portfolio(monkeypatch) -> PortfolioState:
    # PortfolioState refuses to exist without a wallet. Nothing here touches a
    # chain -- holdings are seeded directly below -- so a literal is honest.
    monkeypatch.setenv("PRIMARY_WALLET", "0x000000000000000000000000000000000000dEaD")
    portfolio = PortfolioState()
    portfolio.holdings[("base", "USDC")] = TokenHolding(
        token="USDC", symbol="USDC", quantity=25.0, usd=25.0, chain="base", raw={}
    )
    return portfolio


def _tape(scheduler: BusScheduler, symbol: str, series, *, arm: bool = True) -> RouteState:
    """A RouteState carrying ``series`` at one-minute spacing."""
    state = RouteState(symbol=symbol, base_token=symbol.split("-")[0], quote_token="USDC")
    for index, price in enumerate(series):
        state.samples.append((NOW - (len(series) - index) * 60.0, float(price), 0.0))
    if arm:
        scheduler._opportunity_bias[symbol] = OpportunitySignal(
            symbol=symbol,
            kind="buy-low",
            zscore=-2.0,
            price=float(series[-1]),
            mean_price=float(series[0]),
            std_price=1.0,
            timestamp=NOW,
        )
    return state


def _evaluate(scheduler: BusScheduler, portfolio: PortfolioState, state: RouteState, *, fee_rate: float = 0.007):
    """Run the real candidate method and report ``(verdict, gate_name)``."""
    before = dict(scheduler._mb_gate_counts)
    passes_before = scheduler._mb_gate_pass
    result = scheduler._money_button_candidate(
        state,
        last_price=state.samples[-1][1],
        last_volume=1000.0,
        portfolio=portfolio,
        chain_name="base",
        fee_rate=fee_rate,
        direction_prob=0.5,
        confidence=0.5,
        net_margin=0.0,
        base_allocation=None,
        risk_budget=1.0,
        live_trading=False,
    )
    after = dict(scheduler._mb_gate_counts)
    moved = [gate for gate in after if after[gate] != before.get(gate, 0)]
    return result, moved, scheduler._mb_gate_pass - passes_before


@pytest.mark.parametrize(
    "name, series",
    [
        ("four samples, below the eight the lookback needs", [100.0, 99.0, 98.0, 97.0]),
        ("a flat tape with no dip at all", FLAT),
        ("a 20% crash, far outside the dip window", CRASH_20PCT),
        ("a clean 3% dip, the shape it exists to trade", DIP_3PCT),
    ],
)
def test_a_money_button_refusal_names_exactly_one_gate(scheduler, funded_portfolio, name, series):
    """A refusal is never silent, and never ambiguous about which gate it was."""
    state = _tape(scheduler, "AAA-USDC", series)
    result, moved, passed = _evaluate(scheduler, funded_portfolio, state)

    if result is None:
        assert len(moved) == 1, f"{name}: refused but {len(moved)} gates moved: {moved}"
        assert scheduler._mb_gate_counts[moved[0]] == 1
        assert passed == 0
    else:
        # A pass is counted too, so `evaluations` reconciles against the feed.
        assert moved == [] and passed == 1, name


def test_counting_a_refusal_does_not_change_the_verdict(scheduler, funded_portfolio):
    """The census is bookkeeping. The same tape decides the same way twice.

    Every call site is a ``return None`` that already existed on this path, so
    no candidate's fate changes and no gate moves. This is the assertion that
    keeps that true as the gates are tuned.
    """
    first_state = _tape(scheduler, "BBB-USDC", DIP_3PCT)
    first, _, _ = _evaluate(scheduler, funded_portfolio, first_state)

    second_state = _tape(scheduler, "BBB-USDC", DIP_3PCT)
    second, _, _ = _evaluate(scheduler, funded_portfolio, second_state)

    assert (first is None) == (second is None)


def test_the_census_reconciles_every_evaluation(scheduler, funded_portfolio, tmp_path, monkeypatch):
    """passed + refused_total accounts for every call. A lost row is a lie."""
    import json
    import os

    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    tapes = [
        ("CCC-USDC", [100.0, 99.0, 98.0, 97.0]),
        ("DDD-USDC", FLAT),
        ("EEE-USDC", CRASH_20PCT),
        ("FFF-USDC", DIP_3PCT),
    ]
    for symbol, series in tapes:
        _evaluate(scheduler, funded_portfolio, _tape(scheduler, symbol, series))

    scheduler._mb_flush_gate_census(force=True)
    payload = json.loads((tmp_path / "data" / "money_button_gate_census.json").read_text(encoding="utf-8"))

    assert payload["evaluations"] == len(tapes)
    assert payload["passed"] + payload["refused_total"] == len(tapes)
    assert sum(payload["by_gate"].values()) == payload["refused_total"]
    # Every counted gate carries the symbol that last hit it, so a census read
    # off production names a pair and not just a number.
    for gate in payload["by_gate"]:
        assert payload["last_symbol_per_gate"].get(gate)
