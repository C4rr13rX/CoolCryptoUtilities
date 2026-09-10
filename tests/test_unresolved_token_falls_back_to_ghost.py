"""A token we cannot swap must not cost us the observation.

Link 9 (LIVE) had moved past ``swap_guard:volatility`` to a narrower refusal.
Measured 2026-09-02, the only live entry attempted after the 05:43 restart was
1KTO100M-USDC at 06:06:18, refused with ``reason=token_unresolved`` -- and
``trading_ops`` holds **no position row of any kind** for it, ghost or live.

That is two different verdicts collapsed into one. Not having a contract
address means this trade cannot *settle*; it says nothing about whether the
entry was worth *recording*. The old code returned outright, so the symbol
produced a refusal row and no evidence, and the ledger that gates graduation
never heard about the opportunity at all.

The split matters because the two sets barely overlap. Of the symbols the lane
was working that morning, five resolve AND pass the swap guard (CBBTC,
BASECAT, VIRTUAL, CBETH, BSTONK) while 1KTO100M, CBXRP, MTGA, SPCX and TOAD do
not resolve at all. Abandoning the second group silently removed them from the
book rather than keeping them in ghost where they can still earn a record.
"""
from __future__ import annotations

import asyncio
import time
from unittest import mock

import pytest

from trading.bot import TradingBot
from trading.scheduler import TradeDirective


class _Stub:
    """Answers any attribute with a no-op callable."""

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


class _DB(_Stub):
    def __init__(self):
        self.logged = []

    def log_trade(self, **kwargs):
        self.logged.append(kwargs)
        return True

    def get_pair_adjustment(self, symbol):
        return {}

    def fetch_market_samples_for(self, *args, **kwargs):
        return []

    def fetch_trade_fills(self, *args, **kwargs):
        return []


class _Ledger(_Stub):
    def is_live_approved(self, strategy_id):
        return True


class _Portfolio(_Stub):
    holdings: dict = {}

    def get_quantity(self, *args, **kwargs):
        return 1000.0

    def get_native_balance(self, *args, **kwargs):
        return 1.0


class _Validator(_Stub):
    """A guard that permits the trade, so only resolution is under test."""

    def validate(self, **kwargs):
        return True, {}, []


class _Pipeline(_Stub):
    decision_threshold = 0.58


def _bot() -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    TradingBot._ensure_runtime_state(bot)
    bot.db = _DB()
    bot.strategy_ledger = _Ledger()
    bot.portfolio = _Portfolio()
    bot.swap_validator = _Validator()
    bot.pipeline = _Pipeline()
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
    bot.sim_quote_balances = {}
    bot.stable_tokens = {"USDC"}
    bot.primary_chain = "base"
    bot.live_trading_enabled = True
    bot.ghost_session_id = 2
    bot._ghost_trade_counter = 0
    bot.max_trade_share = 0.5
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
    bot._auto_execute_approved = True
    bot._nash_equilibrium_reached = True
    bot._insufficient_quote_last_ts = {}
    return bot


def _directive(symbol: str, base: str) -> TradeDirective:
    return TradeDirective(
        action="enter", symbol=symbol, base_token=base, quote_token="USDC",
        size=10.0, target_price=1.05, horizon="atf", confidence=0.9,
        expected_return=0.05, reason="test", strategy_id="atf_static",
    )


def _enter(bot, directive, symbol, *, swap_token):
    """Drive the entry path with token resolution forced to a known answer."""
    sample = {"symbol": symbol, "price": 1.0, "ts": time.time(),
              "chain": "base", "volume": 100.0}
    summary = {"exit_conf": 0.9, "direction_prob": 0.9, "delta": 0.05,
               "net_margin": 0.05, "net_pnl": 0.05}

    async def _no_sync(self, *args, **kwargs):
        return None

    with mock.patch.object(
        TradingBot, "_resolve_live_trade_asset",
        # Third argument is the explicit contract the caller already knows the
        # trade is about (directive-supplied, or recorded on an open position).
        lambda self, chain, sym, explicit=None: (sym, explicit or swap_token),
    ), mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync):
        return asyncio.run(
            bot._interpret_predictions(
                None, sample, directive, pred_summary=summary, brain_summary={},
            )
        )


@pytest.fixture(autouse=True)
def _live_execution(monkeypatch):
    """Live approved and dry-run off: the state the refusal was observed in."""
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "0")
    monkeypatch.setenv("STRATEGY_GRADUATION_ENFORCED", "1")


def test_an_unresolvable_token_still_opens_a_ghost_position() -> None:
    """The 1KTO100M-USDC shape: refused for live, still measured."""
    bot = _bot()

    decision = _enter(bot, _directive("1KTO100M-USDC", "1KTO100M"),
                      "1KTO100M-USDC", swap_token=None)

    # The opportunity survives as evidence...
    assert decision["action"] == "enter"
    assert decision["status"] == "ghost-entry"
    assert decision["wallet"] == "ghost"
    assert bot.positions["1KTO100M-USDC"]["mode"] == "ghost"
    # ...and no real money was committed on a token we cannot swap.
    assert decision.get("executed") is not True


def test_the_live_refusal_is_still_recorded() -> None:
    """Falling back must not make the live failure invisible.

    The refusal is what tells us which symbols need an address backfilled. It
    is logged under ``action="hold"`` for the same reason the swap-guard block
    is: it has to stay visible without counting as an executed live trade.
    """
    bot = _bot()

    _enter(bot, _directive("1KTO100M-USDC", "1KTO100M"),
           "1KTO100M-USDC", swap_token=None)

    blocked = [row for row in bot.db.logged
               if row.get("status") == "live-entry-blocked"]
    assert len(blocked) == 1
    assert blocked[0]["action"] == "hold"
    assert blocked[0]["details"]["reason"] == "token_unresolved"


def test_a_resolvable_token_is_not_demoted_to_ghost() -> None:
    """The demotion must be about resolution, not a blanket retreat.

    With an address in hand the entry stays on the live path and fails later
    and louder (here: no bridge). If this ever returns a ghost entry, the
    fallback has swallowed the live lane entirely and no live trade can happen.

    The symbol is deliberately one with NO ledger history. This test used
    BASECAT-USDC and went red the moment the symbol-edge gate learned enough
    about it to refuse it -- 35 closed round trips at -0.852% against 0.465%
    of cost -- because the entry was then refused as entry-refused-symbol-edge
    before it ever reached the bridge branch under test. That refusal is
    correct; a unit test whose verdict depends on the live ledger's current
    contents is not. Do not put a traded symbol back here.
    """
    bot = _bot()

    decision = _enter(bot, _directive("ZQTESTLIVE-USDC", "ZQTESTLIVE"),
                      "ZQTESTLIVE-USDC", swap_token="0x" + "b2" * 20)

    assert decision["status"] == "live-entry-blocked"
    assert decision["reason"] == "bridge_unavailable"
    assert decision["wallet"] == "live"
    # Critically: it did NOT quietly become a ghost trade.
    assert "ZQTESTLIVE-USDC" not in bot.positions
