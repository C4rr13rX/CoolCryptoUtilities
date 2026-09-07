"""The omen strategy must never put a node round trip on the feed's thread.

``trading/brain_bridge.py`` carries the py-spy dump of the asyncio loop
parked inside a brain socket read: every market stream shares that loop, so
one slow brain call froze price collection for ALL symbols and writes
arrived in 12-minute bursts instead of once a second.

So the invariant is not "the brain call is fast", it is "``evaluate`` does
not make the brain call at all". These tests hold ``evaluate`` against a
node that takes ten seconds to answer, and against a node that is simply
dead, and require it to return promptly either way.

Also pinned here:
  * a half-formed bar is never emitted (it would be a second atom for the
    same minute);
  * gaps are dropped, not forward-filled (this feed has already shipped 82
    of 94 symbols holding a seed price forever);
  * an un-admitted omen produces no candidate;
  * the strategy is registered, so it is visible on the population page.
"""
from __future__ import annotations

import time
import types

import numpy as np
import pytest

from trading.omen_brain import (
    OMEN_CREST, OMEN_MURK, OMEN_TROUGH, SCHEMA_VERSION, Omen, omen_threshold,
)
from trading.strategies import build_default_registry
from trading.strategies.base import StrategyContext
from trading.strategies.omen_reversion import (
    BAR_SECONDS, HORIZON_BARS, OmenReversionStrategy, bars_from_samples,
)
import trading.strategies.omen_reversion as omen_reversion
from trading.omen_brain import LOOKBACK_BARS


class _SlowBrain:
    """Stands in for a node that answers, eventually."""

    def __init__(self, delay: float = 10.0) -> None:
        self.delay = delay
        self.calls = 0

    def predict(self, *args, **kwargs):
        self.calls += 1
        time.sleep(self.delay)
        raise AssertionError("should never be awaited by evaluate")


class _DeadBrain:
    def predict(self, *args, **kwargs):
        raise ConnectionRefusedError("node is down")


def _state(symbol="AERO-USDC", bars=LOOKBACK_BARS + 4, ticks_per_bar=3):
    """A RouteState-shaped stub with enough ticks to fill the window."""
    samples = []
    start = 1_700_000_000
    price = 100.0
    for bar in range(bars):
        for tick in range(ticks_per_bar):
            ts = start + bar * BAR_SECONDS + tick * (BAR_SECONDS // ticks_per_bar)
            price = 100.0 + np.sin(bar / 6.0) * 3.0 + tick * 0.01
            samples.append((float(ts), float(price), 1000.0))
    base_token, _, quote_token = symbol.partition("-")
    return types.SimpleNamespace(
        symbol=symbol, samples=samples,
        base_token=base_token, quote_token=quote_token or "USDC")


def _ctx(**overrides):
    base = dict(chain="base", last_price=100.0, last_volume=1000.0,
                fee_rate=0.001, available_quote=50.0, available_base=0.0)
    base.update(overrides)
    return StrategyContext(**base)


def _omen(label=OMEN_TROUGH, verdict="admitted", confidence=0.8):
    action = {"trough": "buy", "crest": "sell"}.get(label, "hold")
    if verdict != "admitted":
        label, action, confidence = OMEN_MURK, "hold", 0.0
    direction = 1 if action == "buy" else (-1 if action == "sell" else 0)
    return Omen(
        schema_version=SCHEMA_VERSION, symbol="AERO-USDC", chain="base",
        as_of_ts=1_700_000_000, price=100.0, horizon_bars=HORIZON_BARS,
        bar_seconds=BAR_SECONDS, omen=label, action=action,
        confidence=confidence, cost_fraction=0.0065,
        threshold_fraction=omen_threshold(),
        expected_move_fraction=direction * omen_threshold(), verdict=verdict,
    )


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(omen_reversion, "ENABLED", True)


# --- the feed invariant ---------------------------------------------------

def test_evaluate_returns_promptly_against_a_ten_second_node():
    """The whole reason this strategy has a cache."""
    strategy = OmenReversionStrategy(brain=_SlowBrain(delay=10.0))
    started = time.time()
    result = strategy.evaluate(_state(), _ctx())
    elapsed = time.time() - started
    assert elapsed < 1.0, (
        f"evaluate took {elapsed:.2f}s -- it is waiting on the node, which "
        f"is what froze the price feed for every symbol")
    assert result is None, "a cold cache must produce no candidate"


def test_evaluate_survives_a_dead_node():
    strategy = OmenReversionStrategy(brain=_DeadBrain())
    assert strategy.evaluate(_state(), _ctx()) is None
    # And again, to prove the failed refresh did not wedge the in-flight slot.
    assert strategy.evaluate(_state(), _ctx()) is None


def test_a_slow_refresh_does_not_pile_up_threads():
    """One in-flight refresh per symbol, however often evaluate is called."""
    brain = _SlowBrain(delay=5.0)
    strategy = OmenReversionStrategy(brain=brain)
    for _ in range(20):
        strategy.evaluate(_state(), _ctx())
    time.sleep(0.2)
    assert brain.calls <= 1, f"{brain.calls} concurrent refreshes for one symbol"


# --- the cache is the only thing evaluate reads ---------------------------

def test_an_admitted_trough_in_the_cache_produces_an_enter():
    strategy = OmenReversionStrategy(brain=_DeadBrain())
    strategy._cache["AERO-USDC"] = (time.time(), _omen(OMEN_TROUGH))
    candidate = strategy.evaluate(_state(), _ctx())
    assert candidate is not None
    assert candidate["directive"].action == "enter"
    assert candidate["meta"]["omen"] == OMEN_TROUGH
    assert candidate["meta"]["omen_schema"] == SCHEMA_VERSION


def test_an_admitted_crest_with_a_position_produces_an_exit():
    strategy = OmenReversionStrategy(brain=_DeadBrain())
    strategy._cache["AERO-USDC"] = (time.time(), _omen(OMEN_CREST))
    candidate = strategy.evaluate(
        _state(), _ctx(available_base=1.0, available_quote=0.0))
    assert candidate is not None
    assert candidate["directive"].action == "exit"


@pytest.mark.parametrize(
    "verdict", ["below_floor", "no_answer", "degenerate", "transport_error"])
def test_an_unadmitted_omen_produces_no_candidate(verdict):
    strategy = OmenReversionStrategy(brain=_DeadBrain())
    strategy._cache["AERO-USDC"] = (time.time(), _omen(verdict=verdict))
    assert strategy.evaluate(_state(), _ctx()) is None


def test_a_stale_cached_omen_is_not_traded():
    strategy = OmenReversionStrategy(brain=_DeadBrain())
    strategy._cache["AERO-USDC"] = (
        time.time() - omen_reversion.CACHE_SEC - 5.0, _omen(OMEN_TROUGH))
    assert strategy.evaluate(_state(), _ctx()) is None


def test_the_strategy_is_silent_while_disabled(monkeypatch):
    monkeypatch.setattr(omen_reversion, "ENABLED", False)
    strategy = OmenReversionStrategy(brain=_DeadBrain())
    strategy._cache["AERO-USDC"] = (time.time(), _omen(OMEN_TROUGH))
    assert strategy.evaluate(_state(), _ctx()) is None


def test_an_expected_move_that_does_not_clear_the_fee_is_refused():
    """The omen's own cost bar is not the last one it has to clear."""
    strategy = OmenReversionStrategy(brain=_DeadBrain())
    strategy._cache["AERO-USDC"] = (time.time(), _omen(OMEN_TROUGH))
    # A fee larger than the omen's threshold leaves nothing to capture.
    assert strategy.evaluate(_state(), _ctx(fee_rate=0.05)) is None


# --- bar building ---------------------------------------------------------

def test_the_forming_bar_is_never_emitted():
    """A half-formed bar has a different high/low than the finished one."""
    ts = np.array([0.0, 10.0, 20.0, 60.0, 70.0], dtype=np.float64)
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    volumes = np.ones(5, dtype=np.float64)
    bars = bars_from_samples(ts, prices, volumes, bar_seconds=60)
    assert len(bars) == 1, "only the closed bucket may be returned"
    assert bars[0]["close"] == 3.0 and bars[0]["high"] == 3.0


def test_gaps_are_dropped_not_forward_filled():
    """A forward-filled bar is a price that never traded."""
    # Ticks land in buckets 0, 10 and 20. Buckets 1-9 and 11-19 are a real
    # feed gap; a forward fill would invent 18 bars that never traded.
    ts = np.array([0.0, 30.0, 600.0, 630.0, 1200.0], dtype=np.float64)
    prices = np.array([1.0, 1.5, 9.0, 9.5, 9.9], dtype=np.float64)
    volumes = np.ones(5, dtype=np.float64)
    bars = bars_from_samples(ts, prices, volumes, bar_seconds=60)
    # Bucket 20 is the newest and still forming, so only 0 and 600 close.
    assert [b["timestamp"] for b in bars] == [0, 600]


def test_bars_are_ordered_oldest_first_and_ohlc_is_consistent():
    state = _state()
    ts = np.array([s[0] for s in state.samples])
    prices = np.array([s[1] for s in state.samples])
    volumes = np.array([s[2] for s in state.samples])
    bars = bars_from_samples(ts, prices, volumes)
    assert len(bars) >= LOOKBACK_BARS
    assert bars == sorted(bars, key=lambda b: b["timestamp"])
    for bar in bars:
        assert bar["low"] <= bar["open"] <= bar["high"]
        assert bar["low"] <= bar["close"] <= bar["high"]
        assert bar["buy_volume"] + bar["sell_volume"] == pytest.approx(
            bar["net_volume"])


def test_an_empty_tick_series_yields_no_bars():
    empty = np.empty(0, dtype=np.float64)
    assert bars_from_samples(empty, empty, empty) == []


# --- visibility -----------------------------------------------------------

def test_the_strategy_is_in_the_default_registry():
    """An invisible strategy is one nobody can judge."""
    assert "omen_reversion" in set(build_default_registry().ids())


def test_status_names_why_it_is_silent():
    strategy = OmenReversionStrategy(brain=_DeadBrain())
    status = strategy.status()
    assert status["strategy_id"] == "omen_reversion"
    assert status["threshold_fraction"] > 0.0
    assert status["horizon_sec"] == HORIZON_BARS * BAR_SECONDS
    assert "last_reason" in status


# --- the confidence floor is a measurement, not a guess -------------------

#: The two distributions the floor separates, from
#: data/brain_experiments/omen-AERO-USDC-h12-20260907-085010.json:
#: 40 pure-noise frames and 500 real held-out frames, 2725 trained pairs.
MEASURED_GARBAGE_MAX = 0.675
MEASURED_REAL_MIN = 0.921


def test_the_confidence_floor_sits_in_the_measured_gap():
    """A floor outside the gap either passes noise or refuses real frames.

    The 2026-08 regime gate shipped a floor of 0.45 picked by intuition and
    it rejected a valid reading. This one is pinned to the numbers it was
    read off, so moving it without re-measuring turns the test red.
    """
    assert MEASURED_GARBAGE_MAX < omen_reversion.CONFIDENCE_FLOOR < MEASURED_REAL_MIN, (
        f"floor {omen_reversion.CONFIDENCE_FLOOR} is outside the measured "
        f"gap ({MEASURED_GARBAGE_MAX}, {MEASURED_REAL_MIN}) -- re-run "
        f"scripts/omen_experiment.py and read a new one off the report")


def test_a_garbage_grade_confidence_is_refused_by_the_floor():
    """Every noise frame in the run scored at or below MEASURED_GARBAGE_MAX."""
    assert MEASURED_GARBAGE_MAX < omen_reversion.CONFIDENCE_FLOOR


def test_a_real_grade_confidence_still_clears_the_floor():
    """...and every real frame scored at or above MEASURED_REAL_MIN, so the
    floor costs no genuine signal. A floor that filters trades as well as
    noise would be doing two jobs and neither of them measurably."""
    assert omen_reversion.CONFIDENCE_FLOOR < MEASURED_REAL_MIN
