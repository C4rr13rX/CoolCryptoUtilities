"""A refused entry must not swallow the held position's stop-loss.

The dispatch in ``_interpret_predictions`` opened with::

    if directive and directive.action == "enter":
        should_enter = True

so any sample carrying an entry directive was routed into the entry path. For a
symbol already held by a LIVE position that path logs ``entry-refused-live-held``
and returns immediately -- and the protective bracket (take-profit, stop-loss,
break-even, trailing) lives in the final ``else``, which those samples never
reached.

The strategies that emit entry directives are exactly the ones that like a
symbol, so they re-emit for the symbol they already hold. The position was
therefore invisible to its own stop for as long as its own strategy kept liking
it. Measured on the live BSTONK-USDC position of 2026-09-03 (entered
1788455200, stopped 1788462735, $0.75 notional):

    72 of the 73 samples in those 2h05m were entry-refused-live-held and
    evaluated no trigger at all. The one sample that arrived WITHOUT an entry
    directive reached the stop on its first look and fired at once --
    ``stop_loss:-0.1840``, against a LIVE_STOP_LOSS_PCT of 0.015.

A 1.5% stop realised an 18.40% loss: -$0.1380 gross, 92% of the live P/L to
date. The first feed sample after entry was already -13.08%, so a stop that was
allowed to look would have exited there; the 5.32pp between -13.08% and -18.40%
is the pure cost of the skipped evaluations. CBBTC-USDC (127 refusals) and
CBETH-USDC (99) ran the same gauntlet and were only luckier.

Scope of the rule: the bracket outranks the directive only when the directive is
refused anyway. A live-approved entry still displaces what it lands on -- that
release path is pinned by test_entry_never_clobbers_a_position.py and deferring
it would trade real fills for simulated exits.
"""

from __future__ import annotations

import asyncio
import time
from unittest import mock

import pytest

from trading.bot import TradingBot
from trading.scheduler import TradeDirective

SYMBOL = "BSTONK-USDC"
BSTONK = "0x0F61Edbfe6Cd86024C0f210c0695B08df55fdfc9"

# The real numbers off the trade that paid for this test.
ENTRY_FILL = 0.0020818052696136586   # receipt fill price
FIRST_SAMPLE_AFTER_ENTRY = 0.0018095238   # -13.08%, 262s later
STOP_OUT_PRICE = 0.0016987796        # -18.40%, where it actually closed


class _Stub:
    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


class _DB(_Stub):
    def __init__(self):
        self.logged: list = []

    def log_trade(self, **kwargs):
        self.logged.append(kwargs)
        return True

    def get_pair_adjustment(self, symbol):
        return {}

    def fetch_market_samples_for(self, *args, **kwargs):
        return []

    def fetch_trade_fills(self, *args, **kwargs):
        return []


class _GhostOnlyLedger(_Stub):
    """Graduated nobody -- so an incoming entry is refused on a live slot.

    This is the state the BSTONK position actually sat in: atf_static held the
    slot live, and the directives arriving for it were not live-approved.
    """

    def is_live_approved(self, strategy_id):
        return False


class _Portfolio(_Stub):
    holdings: dict = {}

    def get_quantity(self, *args, **kwargs):
        return 1000.0

    def get_native_balance(self, *args, **kwargs):
        return 1.0


class _Validator(_Stub):
    def validate(self, **kwargs):
        return True, {}, []


class _Pipeline(_Stub):
    decision_threshold = 0.58


def _bot() -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    TradingBot._ensure_runtime_state(bot)
    bot.db = _DB()
    bot.strategy_ledger = _GhostOnlyLedger()
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


@pytest.fixture(autouse=True)
def _live_execution(monkeypatch):
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "1")
    monkeypatch.setenv("STRATEGY_GRADUATION_ENFORCED", "1")


def _directive(strategy_id: str = "atf_static_scout") -> TradeDirective:
    """The kind of directive that kept arriving for BSTONK while it was held."""
    return TradeDirective(
        action="enter", symbol=SYMBOL, base_token="BSTONK", quote_token="USDC",
        size=10.0, target_price=0.00201, horizon="atf", confidence=0.7851,
        expected_return=0.05, reason="atf_static: ATF researched candidate",
        strategy_id=strategy_id,
    )


def _live_position(target_price: float = 0.00201) -> dict:
    """atf_static's live BSTONK slot, holding real tokens."""
    now = time.time()
    return {
        "mode": "live",
        "strategy_id": "atf_static",
        "size": 360.264243225393,
        "entry_price": ENTRY_FILL,
        "target_price": target_price,
        "entry_ts": now - 262.0,
        "ts": now - 262.0,
        "trade_id": "2:BSTONK-USDC:32fff2d5ea0549bd917fcf25e23acc36",
        "tx_hash": "0xcd6fb05c92af5077f9be707727c1d57e0ac1dfedd54d9f87e860376b96ea560b",
        "base_token_address": BSTONK,
        "total_quote_spent": 0.75,
    }


def _tick(bot: TradingBot, price: float, directive: TradeDirective | None):
    sample = {"symbol": SYMBOL, "price": price, "ts": time.time(),
              "chain": "base", "volume": 100.0}
    summary = {"exit_conf": 0.5, "direction_prob": 0.5, "delta": 0.0,
               "net_margin": 0.0, "net_pnl": 0.0}

    async def _no_sync(self, *args, **kwargs):
        return None

    with mock.patch.object(
        TradingBot, "_resolve_live_trade_asset",
        lambda self, chain, sym, explicit=None: (sym, explicit or BSTONK),
    ), mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync):
        return asyncio.run(
            bot._interpret_predictions(
                None, sample, directive, pred_summary=summary, brain_summary={},
            )
        )


# --------------------------------------------------------------------------
# The bug that cost $0.1380.
# --------------------------------------------------------------------------

def test_a_refused_entry_does_not_hide_the_stop_loss() -> None:
    """The tick that used to return entry-refused-live-held now hits the stop."""
    bot = _bot()
    bot.positions[SYMBOL] = _live_position()

    decision = _tick(bot, FIRST_SAMPLE_AFTER_ENTRY, _directive())

    assert decision["status"] != "entry-refused-live-held", (
        "the entry refusal swallowed the sample again -- the held position "
        "never got to look at its own stop"
    )
    assert decision["action"] == "exit"
    assert str(decision.get("exit_reason", "")).startswith("stop_loss"), decision


def test_the_stop_fires_at_the_configured_distance_not_12x_it() -> None:
    """-13.08% is where the first sample landed; -18.40% is where it closed."""
    bot = _bot()
    bot.positions[SYMBOL] = _live_position()

    decision = _tick(bot, FIRST_SAMPLE_AFTER_ENTRY, _directive())

    assert decision["exit_reason"] == "stop_loss:-0.1308"
    assert decision["exit_price"] == FIRST_SAMPLE_AFTER_ENTRY
    # The realised loss is now bounded by the first sample that breaches the
    # stop, not by the next sample that happens to arrive without a directive.
    realised = (decision["exit_price"] - ENTRY_FILL) / ENTRY_FILL
    actual_history = (STOP_OUT_PRICE - ENTRY_FILL) / ENTRY_FILL
    assert realised > actual_history
    assert realised - actual_history == pytest.approx(0.0532, abs=5e-4)


def test_a_held_position_inside_its_bracket_is_left_alone() -> None:
    """The bracket outranks the directive only when it actually fires.

    Without this the fix would turn every refused entry into an exit, which
    closes live positions on the strength of another lane wanting to enter.
    """
    bot = _bot()
    # Well inside the bracket: above entry, below target, too young to time out.
    bot.positions[SYMBOL] = _live_position(target_price=ENTRY_FILL * 1.05)

    decision = _tick(bot, ENTRY_FILL * 1.001, _directive())

    assert decision["action"] == "hold"
    assert "exit_reason" not in decision
    held = bot.positions[SYMBOL]
    assert held["mode"] == "live"
    assert held["tx_hash"].startswith("0xcd6fb0")
    assert held["size"] == 360.264243225393
    assert not [r for r in bot.db.logged if str(r.get("action")) == "exit"]


def test_the_bracket_state_advances_on_a_refused_tick() -> None:
    """high_watermark must not stand still while entries are being refused.

    The trailing stop and break-even lock are both armed off the watermark, so
    a watermark that only moves on unaccompanied samples arms them late -- the
    same class of blindness as the stop, one step removed.
    """
    bot = _bot()
    bot.positions[SYMBOL] = _live_position(target_price=ENTRY_FILL * 1.50)

    peak = ENTRY_FILL * 1.03
    _tick(bot, peak, _directive())

    state = bot.positions[SYMBOL].get("trigger_state") or {}
    assert state.get("high_watermark") == pytest.approx(peak)
    assert state.get("break_even_armed") is True
