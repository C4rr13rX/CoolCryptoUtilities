"""Regression tests for the live Money Button Base funnel ([1252a1db]).

Covers the three exact mechanical blockers measured on pass 745:
  1. `_money_button_candidate` (`money_button_dip`) was refused 91/111 times on
     `tf_direction_or_confidence` because TensorFlow is absent and the fallback
     swarm/heuristic head leans `< 0.50` with `net_margin < 0` during a real
     `-0.7%` to `-1.6%` WETH dip.
  2. `BusScheduler.evaluate` passed the unmeasured `1.45%` fee buffer into
     `CDCLTradingSolver` `context["fee_rate"]` instead of `measured_rt_fee`
     (`0.0201%` on WETH) and let uncalibrated `discovered_edgeforge_*` claims
     crowd out live `money_button` / `money_button_dip` entries.
  3. Under `stale_ghost_book` (`recommended_live_usd = 0.0`) +
     `first_tranche_cap_usd = 0.50`, a live `WETH-USDC` entry clamped at `$0.50`
     could never amortize `$0.004047` fixed Base gas against the `0.558%` pooled
     `edge_estimate` (`1.25 * 0.830% = 1.037% > 0.558%`), whereas raising the
     clip to `$1.35` (inside `$0.50-$2.00` and `<= live_capital_cap_usd`) clears
     measured WETH cost (`1.25 * 0.322% = 0.402% <= 0.558%`) while still
     refusing unmeasured pairs (`UP-USDC`).
"""

from __future__ import annotations

import asyncio
import time
from unittest import mock

import pytest

from trading.bot import TradingBot
from trading.edge_estimate import _clear_cache
from trading.portfolio import PortfolioState, TokenHolding
from trading.scheduler import BusScheduler, MONEY_BUTTON_DIP_STRATEGY_ID, RouteState, TradeDirective


# Peak 1785 followed by a stabilized -1.50% dip shelf over the trailing 30m (slope > -0.00025).
WETH_DIP_SERIES = [1785.0] * 25 + [1759.0] * 12 + [1758.8, 1758.6, 1758.4, 1758.2]


@pytest.fixture(autouse=True)
def _clean_global_caches(monkeypatch):
    monkeypatch.setattr("trading.strategies.build_default_registry", lambda: None)
    _clear_cache()
    yield
    _clear_cache()


def _funded_portfolio(monkeypatch) -> PortfolioState:
    monkeypatch.setenv("PRIMARY_WALLET", "0x291c854811e92906a658Fb94Aa511bF919f968ad")
    portfolio = PortfolioState()
    portfolio.holdings[("base", "USDC")] = TokenHolding(
        token="USDC", symbol="USDC", quantity=4.018236, usd=4.018236, chain="base", raw={}
    )
    portfolio.holdings[("base", "USDT")] = TokenHolding(
        token="USDT", symbol="USDT", quantity=5.005437, usd=5.005437, chain="base", raw={}
    )
    portfolio.native_balances["base"] = 0.003222
    return portfolio


def test_money_button_dip_fires_when_fallback_head_leans_bearish_during_dip(monkeypatch) -> None:
    monkeypatch.delenv("MONEY_BUTTON_REQUIRE_TF", raising=False)
    sched = BusScheduler(prefill=False)
    sched.strategy_registry = None
    portfolio = _funded_portfolio(monkeypatch)
    now = time.time()
    state = RouteState(symbol="WETH-USDT", base_token="WETH", quote_token="USDT")
    for idx, px in enumerate(WETH_DIP_SERIES):
        state.samples.append((now - (len(WETH_DIP_SERIES) - idx) * 120.0, float(px), 100.0))

    cand = sched._money_button_candidate(
        state,
        last_price=float(WETH_DIP_SERIES[-1]),
        last_volume=100.0,
        portfolio=portfolio,
        chain_name="base",
        fee_rate=0.000201,
        direction_prob=0.44,
        confidence=0.50,
        net_margin=-0.0029,
        base_allocation=None,
        risk_budget=1.0,
        live_trading=True,
    )
    assert cand is not None
    assert cand["directive"].strategy_id == MONEY_BUTTON_DIP_STRATEGY_ID

    # Explicit opt-in still enforces the TF direction/confidence gate.
    monkeypatch.setenv("MONEY_BUTTON_REQUIRE_TF", "1")
    refused = sched._money_button_candidate(
        state,
        last_price=float(WETH_DIP_SERIES[-1]),
        last_volume=100.0,
        portfolio=portfolio,
        chain_name="base",
        fee_rate=0.000201,
        direction_prob=0.44,
        confidence=0.50,
        net_margin=-0.0029,
        base_allocation=None,
        risk_budget=1.0,
        live_trading=True,
    )
    assert refused is None
    assert sched._mb_gate_counts.get("tf_direction_or_confidence", 0) >= 1


def test_trident_judges_money_button_against_measured_cost_and_prioritizes_live_money_button(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MONEY_BUTTON_REQUIRE_TF", raising=False)
    monkeypatch.setenv("GAS_ALERT_MIN_NATIVE", "0.0002")
    monkeypatch.setattr("trading.scheduler._is_test_env", lambda: False)
    monkeypatch.setattr("trading.scheduler._strategy_edge_refusal", lambda _sid: None)
    monkeypatch.setattr("trading.scheduler._symbol_edge_refusal", lambda _sym, _sid=None: None)

    class _FakeRegistry:
        last_skips: dict = {}

        def evaluate_all(self, state, ctx):
            d = TradeDirective(
                action="enter",
                symbol=state.symbol,
                base_token=state.base_token,
                quote_token=state.quote_token,
                size=0.001,
                target_price=ctx.last_price * 1.028,
                horizon="4m",
                confidence=0.80,
                expected_return=0.028,
                reason="discovered_edgeforge_fake",
                strategy_id="discovered_edgeforge_fake_123456",
            )
            return [{"directive": d, "score": 0.028, "meta": {"confidence": 0.8, "direction_prob": 0.7, "quality": 0.8}}]

    sched = BusScheduler(prefill=False)
    sched.strategy_registry = None
    sched.strategy_registry = _FakeRegistry()
    portfolio = _funded_portfolio(monkeypatch)
    now = time.time()
    for idx, px in enumerate(WETH_DIP_SERIES[:-1]):
        sched._update_state(
            {"symbol": "WETH-USDC", "price": float(px), "volume": 100.0, "ts": now - (len(WETH_DIP_SERIES) - idx) * 120.0, "chain": "base"}
        )

    directive = sched.evaluate(
        {"symbol": "WETH-USDC", "price": float(WETH_DIP_SERIES[-1]), "volume": 100.0, "ts": now, "chain": "base"},
        {"direction_prob": 0.45, "exit_conf": 0.50, "net_margin": -0.002},
        portfolio,
        risk_budget=1.0,
        live_trading=True,
    )
    assert directive is not None
    assert directive.strategy_id == MONEY_BUTTON_DIP_STRATEGY_ID


class _Stub:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _PooledEdgeDB(_Stub):
    def __init__(self):
        self.logged: list = []

    def log_trade(self, **kwargs):
        self.logged.append(kwargs)
        return True

    def get_pair_adjustment(self, symbol):
        return {"size_multiplier": 1.0, "allocation_multiplier": 1.0}

    def fetch_trades(self, limit=4000, statuses=None):
        if statuses == ["ghost-exit"]:
            return [
                {"details": {"strategy_id": "other_strat", "profit": 0.00558}}
                for _ in range(30)
            ]
        return []

    def fetch_trade_outcomes(self, wallet="live", limit=400):
        return []

    def fetch_market_samples_for(self, *args, **kwargs):
        return []

    def fetch_trade_fills(self, *args, **kwargs):
        return []


def _make_live_bot(monkeypatch) -> TradingBot:
    _clear_cache()
    bot = TradingBot.__new__(TradingBot)
    TradingBot._ensure_runtime_state(bot)
    bot.db = _PooledEdgeDB()

    class _Ledger(_Stub):
        def is_live_approved(self, sid):
            return sid == MONEY_BUTTON_DIP_STRATEGY_ID

        def may_place_live_order(self, sid):
            return sid == MONEY_BUTTON_DIP_STRATEGY_ID

    class _Port(_Stub):
        holdings: dict = {}

        def get_quantity(self, sym, chain=None):
            return 4.02 if str(sym).upper() == "USDC" else 0.0

        def get_native_balance(self, *args, **kwargs):
            return 0.0032

    class _Validator(_Stub):
        def validate(self, **kwargs):
            return True, {}, []

    bot.strategy_ledger = _Ledger()
    bot.portfolio = _Port()
    bot.swap_validator = _Validator()
    bot.pipeline = type("P", (), {"decision_threshold": 0.58})()
    bot.metrics = _Stub()
    bot.scheduler = _Stub()
    bot.memory = _Stub()
    bot.graph = _Stub()
    bot.rotator = _Stub()
    bot.equilibrium = _Stub()
    bot._buffer = _Stub()
    bot.positions = {}
    bot.active_exposure = {}
    bot.bus_routes = {}
    bot.sim_native_balances = {}
    bot.sim_quote_balances = {("base", "USDC"): 25.0}
    bot.stable_tokens = {"USDC", "USDT"}
    bot.primary_chain = "base"
    bot.live_trading_enabled = True
    bot.ghost_session_id = 2
    bot._ghost_trade_counter = 0
    bot.max_trade_share = 0.05
    bot.gas_buffer_multiplier = 1.5
    bot.total_trades = 0
    bot.wins = 0
    bot.total_profit = 0.0
    bot.realized_profit = 0.0
    bot.stable_bank = 0.0
    bot._bridge = None
    bot._reflex_block_reason = None
    bot._reflex_blocked_until = 0.0
    bot._live_total_pnl = 0.0
    bot._live_peak_pnl = 0.0
    bot._live_consecutive_losses = 0
    bot._circuit_breaker_max_losses = 99
    bot._circuit_breaker_max_drawdown = 1e9
    bot._auto_execute_approved = False
    bot._nash_equilibrium_reached = True
    bot._insufficient_quote_last_ts = {}
    # Exact production stale_ghost_book + LIVE_FIRST_TRANCHE_USD=0.50 state:
    bot._transition_plan = {
        "capital_plan": {
            "recommended_live_usd": 0.0,
            "min_clip_usd": 0.75,
            "deployable_stable_usd": 9.02,
            "live_capital_cap_usd": 2.0,
            "live_ramp_schedule": {"first_tranche_cap_usd": 0.50},
        }
    }
    monkeypatch.setattr("trading.bot.symbol_edge_refusal", lambda _s, _sid=None: None)
    monkeypatch.setattr("trading.bot.strategy_edge_refusal", lambda _sid: None)
    monkeypatch.setattr("trading.bot.symbol_motion_refusal", lambda _s: None)
    monkeypatch.setattr("trading.bot.stop_survivability_refusal", lambda _s: None)
    return bot


def test_live_weth_clip_rises_to_amortize_fixed_base_gas_while_unmeasured_pair_stays_refused(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "1")
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "1")
    monkeypatch.setenv("STRATEGY_GRADUATION_ENFORCED", "1")

    async def _no_sync(self, *args, **kwargs):
        return None

    # 1. WETH-USDC (measured 2.01bp Base pool): raises clip to $1.35 inside [$0.50, $2.00] and passes micro_profit.
    bot_weth = _make_live_bot(monkeypatch)
    weth_dir = TradeDirective(
        action="enter",
        symbol="WETH-USDC",
        base_token="WETH",
        quote_token="USDC",
        size=0.25 / 1760.0,
        target_price=1760.0 * 1.008,
        horizon="30m",
        confidence=0.85,
        expected_return=0.008,
        reason="money_button dip",
        strategy_id=MONEY_BUTTON_DIP_STRATEGY_ID,
    )
    with mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync), mock.patch.object(
        TradingBot, "_live_trades_dry_run", lambda self: True
    ):
        dec_weth = asyncio.run(
            bot_weth._interpret_predictions(
                None,
                {"symbol": "WETH-USDC", "price": 1760.0, "ts": time.time(), "chain": "base", "volume": 100.0},
                weth_dir,
                pred_summary={"exit_conf": 0.5, "direction_prob": 0.5, "delta": 0.008, "net_margin": 0.008, "net_pnl": 0.008},
                brain_summary={},
            )
        )
    assert 1.20 <= dec_weth["micro_profit"]["notional_usd"] <= 2.00
    assert dec_weth["micro_profit"]["viable"] is True, dec_weth["micro_profit"]
    assert dec_weth["status"] == "live-dry-run-entry"

    # 2. UP-USDC (unmeasured token, 0.3187% + $0.004047 fixed -> 1.25 * cost_rate > 0.558% even at $1.35): stays refused!
    bot_up = _make_live_bot(monkeypatch)
    up_dir = TradeDirective(
        action="enter",
        symbol="UP-USDC",
        base_token="UP",
        quote_token="USDC",
        size=0.25 / 1.0,
        target_price=1.008,
        horizon="30m",
        confidence=0.85,
        expected_return=0.008,
        reason="money_button dip",
        strategy_id=MONEY_BUTTON_DIP_STRATEGY_ID,
    )
    with mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync):
        dec_up = asyncio.run(
            bot_up._interpret_predictions(
                None,
                {"symbol": "UP-USDC", "price": 1.0, "ts": time.time(), "chain": "base", "volume": 100.0},
                up_dir,
                pred_summary={"exit_conf": 0.5, "direction_prob": 0.5, "delta": 0.008, "net_margin": 0.008, "net_pnl": 0.008},
                brain_summary={},
            )
        )
    assert dec_up["micro_profit"]["viable"] is False
    assert any(r.get("status") == "live-entry-below-profit-floor" for r in bot_up.db.logged)
