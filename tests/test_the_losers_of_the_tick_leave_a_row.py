"""Who was OFFERED the tick, not just who won it.

``BusScheduler.evaluate`` offers every registered strategy a chance on every
tick -- ``strategy_registry.evaluate_all`` (``trading/scheduler.py``) collects
a candidate from each -- and then spends exactly ONE of them, via
``_trident.select`` with a ``max(score)`` fallback. Before this row existed the
losers left no trace anywhere: the only per-strategy rows in ``trading_ops``
are the winner's (``ghost_candidate``, ``ghost-entry``) and the edge-ban
predrop's.

That gap is why the decision budget was mis-measured. The 2026-09-10 census
read "atf_static 179 cycles, every other strategy 7 between them" off
downstream rows -- but those rows count cycles WON, and reported them as
cycles OFFERED. The two numbers answer different questions and point at
different bugs:

  * a strategy offered zero cycles is a SCHEDULER defect -- it is never asked;
  * a strategy offered many and winning none is a SCORING defect -- it is
    asked and always loses the arbitration.

Nothing in the repo could tell those apart, so "the other 36 strategies never
get to trade" was an inference from the winners' table rather than a
measurement. These tests pin the row that makes it a measurement.

Against the old behaviour every test here fails: no ``entry-arbitration`` row
was written at all, so the losers are unrecoverable from the log.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pytest

import trading.scheduler as scheduler_module
from trading.scheduler import BusScheduler, HorizonSignal, TradeDirective


SYMBOL = "AERO-USDC"
WINNER = "atf_static"
LOSER_A = "rsi_reversal"
LOSER_B = "bollinger_squeeze"


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


def _scheduler(monkeypatch, candidates: List[Dict[str, object]]):
    """A scheduler whose only candidates are the ones under test, plus its log."""
    sched = BusScheduler(horizons=[("5m", 300), ("30m", 1800)], min_profit=0.01, prefill=False)
    _pump(sched)

    # No standing bans: this is about arbitration, not about the edge gates.
    monkeypatch.setattr(scheduler_module, "_strategy_edge_refusal", lambda sid: None)
    monkeypatch.setattr(scheduler_module, "_symbol_edge_refusal", lambda sym, sid=None: None)

    if sched.strategy_registry is None:
        pytest.skip("no strategy registry available in this build")
    monkeypatch.setattr(
        sched.strategy_registry, "evaluate_all", lambda state, ctx: list(candidates)
    )
    # A FLAT forecast signal: clears the "no_forecast_signals" short-circuit
    # without contributing a competing candidate of its own, so the offered
    # set under test is exactly the list above.
    monkeypatch.setattr(
        sched,
        "_forecast",
        lambda state: [
            HorizonSignal(
                label="5m",
                seconds=300,
                predicted_price=float(state.samples[-1][1]),
                expected_return=0.0,
                zscore=0.0,
            )
        ],
    )
    monkeypatch.setenv("ATF_STATIC_GHOST_SCOUT_ENABLED", "0")

    rows: List[Dict[str, Any]] = []

    def capture(**kwargs: Any) -> None:
        rows.append(kwargs)

    monkeypatch.setattr(sched.db, "log_trade", capture)
    return sched, rows


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


def _arbitration(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    found = [r for r in rows if r.get("status") == "entry-arbitration"]
    assert found, (
        "the tick was arbitrated between several strategies and wrote no "
        "entry-arbitration row: the losers are unrecoverable from trading_ops, "
        "so cycles-offered cannot be told from cycles-won"
    )
    assert len(found) == 1, f"one tick wrote {len(found)} arbitration rows"
    return found[0]


def test_the_strategies_that_lost_the_tick_are_named_in_the_row(monkeypatch) -> None:
    """Three strategies wanted the tick; the row must name all three, not the winner."""
    sched, rows = _scheduler(
        monkeypatch,
        [
            _candidate("enter", WINNER, 0.90),
            _candidate("enter", LOSER_A, 0.50),
            _candidate("enter", LOSER_B, 0.10),
        ],
    )
    directive = _evaluate(sched)
    assert directive is not None

    details = _arbitration(rows)["details"]
    offered = details["offered"]
    assert set(offered) == {WINNER, LOSER_A, LOSER_B}, (
        f"offered={sorted(offered)}: a strategy that was asked and lost is "
        "missing, which is exactly the row that made cycles-won look like "
        "cycles-offered"
    )
    assert details["offered_total"] == 3
    assert details["chosen"] == directive.strategy_id


def test_two_candidates_from_one_strategy_count_twice(monkeypatch) -> None:
    """``offered`` is a COUNT per strategy: the share of the budget is the question."""
    sched, rows = _scheduler(
        monkeypatch,
        [
            _candidate("enter", WINNER, 0.90),
            _candidate("enter", WINNER, 0.80),
            _candidate("enter", LOSER_A, 0.10),
        ],
    )
    _evaluate(sched)
    details = _arbitration(rows)["details"]
    assert details["offered"] == {WINNER: 2, LOSER_A: 1}, (
        f"offered={details['offered']}: deduping strategies hides concentration, "
        "which is the number the census exists to report"
    )
    assert details["offered_total"] == 3


def test_a_tick_with_nothing_to_enter_writes_no_arbitration_row(monkeypatch) -> None:
    """An exit-only tick allocates no entry budget, so it must not inflate the log.

    The row is bounded by the CONTESTED resource, not by the tick rate: 185
    enter candidates against 1611 ticks in the 6h to 2026-09-10 09:40. A row
    per tick would triple ``trading_ops`` and carry no allocation information.
    """
    sched, rows = _scheduler(monkeypatch, [_candidate("exit", WINNER, 0.90)])
    directive = _evaluate(sched, base_balance=10.0)
    assert directive is not None and directive.action == "exit"
    assert not [r for r in rows if r.get("status") == "entry-arbitration"], (
        "an exit-only tick wrote an arbitration row; the log grows with ticks "
        "rather than with contested entries"
    )


def test_a_failing_log_write_does_not_stop_the_tick(monkeypatch) -> None:
    """Diagnostics must never cost a directive -- the tick is the money path."""
    sched, _rows = _scheduler(monkeypatch, [_candidate("enter", WINNER, 0.90)])

    def explode(**kwargs: Any) -> None:
        raise RuntimeError("trading_ops is locked")

    monkeypatch.setattr(sched.db, "log_trade", explode)
    directive = _evaluate(sched)
    assert directive is not None, "a failed diagnostic write swallowed the directive"
    assert directive.strategy_id == WINNER


def test_a_strategy_the_registry_skipped_is_not_confused_with_one_that_lost(monkeypatch) -> None:
    """A skip and a loss must not look alike -- they are different bugs.

    ``StrategyRegistry.evaluate_all`` skips a strategy three ways: below
    ``min_samples``, ``enabled()`` false, or ``evaluate`` raised. None of them
    left a record, so a strategy that was NEVER ASKED was indistinguishable
    from one asked on every tick that always lost the arbitration. The first
    needs a scheduler fix and the second a scoring fix.

    The asymmetry is real and measured: across the 72 registered strategies on
    2026-09-10, ``atf_static`` needs 4 samples and ``ema_cross``,
    ``bollinger_squeeze``, ``macd_momentum`` and ``donchian_breakout`` need 40.
    """
    from trading.strategies.base import StrategyRegistry

    class _Strat:
        def __init__(self, sid: str, min_samples: int, *, on: bool = True,
                     raises: bool = False, signal: bool = True) -> None:
            self.strategy_id = sid
            self.min_samples = min_samples
            self._on = on
            self._raises = raises
            self._signal = signal

        def enabled(self) -> bool:
            return self._on

        def evaluate(self, state: Any, ctx: Any):
            if self._raises:
                raise ValueError("boom")
            return _candidate("enter", self.strategy_id, 0.5) if self._signal else None

    registry = StrategyRegistry([
        _Strat("cheap_warmup", 4),
        _Strat("needs_forty", 40),
        _Strat("switched_off", 0, on=False),
        _Strat("always_throws", 0, raises=True),
        _Strat("asked_but_silent", 0, signal=False),
    ])

    class _State:
        samples = [(0.0, 1.0)] * 10

    got = registry.evaluate_all(_State(), object())
    assert [c["directive"].strategy_id for c in got] == ["cheap_warmup"]

    skips = registry.last_skips
    assert skips["needs_forty"] == "min_samples 10<40", (
        f"{skips.get('needs_forty')!r}: the warm-up asymmetry is the reason a "
        "strategy gets no cycles, and it must say so in the row"
    )
    assert skips["switched_off"] == "disabled"
    assert skips["always_throws"] == "raised ValueError", (
        "a strategy that throws on every tick is skipped forever and looks "
        "exactly like one with no signal -- naming the exception is the only "
        "way that bug is ever seen"
    )
    assert skips["asked_but_silent"] == "no_signal"
    assert "cheap_warmup" not in skips


def test_the_arbitration_row_carries_the_skips(monkeypatch) -> None:
    """offered and skipped land in ONE row, so the shares are comparable."""
    sched, rows = _scheduler(monkeypatch, [_candidate("enter", WINNER, 0.90)])
    sched.strategy_registry.last_skips = {"needs_forty": "min_samples 10<40"}
    _evaluate(sched)
    details = _arbitration(rows)["details"]
    assert details["skipped"].get("needs_forty") == "min_samples 10<40", (
        "the arbitration row reports who competed but not who was never asked"
    )
