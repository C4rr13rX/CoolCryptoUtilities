"""An entry must never take an occupied position slot in silence.

Link 9 (LIVE) had every gate in front of it open on 2026-09-03: atf_static was
live_approved, dry-run was off, the arrow that aborted the last two swaps was
gone, and the router quoted USDC->CBBTC/AERO/CBETH/BASECAT for $3 apiece. What
was left was a bug waiting for the first real trade.

``self.positions`` is one slot per symbol, and both writes in
``_interpret_predictions`` are plain assignments that never asked whether the
slot was occupied. Measured by driving the method with a book already holding
the symbol:

    ghost entry over a LIVE position -> mode live->ghost, size 75.0 -> 10.0,
        tx_hash "0xrealhash" -> None. The bought tokens stay on-chain with
        nothing in the book pointing at them; the bot then "exits" a
        simulation -- no swap -- and the live P/L never settles.

    live entry over a ghost position -> the ghost trade_id, entry price and
        size vanish with no exit, no outcome and no ledger row: the orphaned
        entry class that _save_state already documents 152 of.

The first direction is the one that loses money, and it was the likely one. Of
the 17 symbols ticking at the time, 12 carried a ghost position, and the ghost
lane opens ~80 entries per 6h -- so the first live position would have been
overwritten within hours of being opened.

The rule these tests pin:

    * a non-live entry is REFUSED on a symbol held live (the live row is the
      only record of tokens we own, so it is never released), and
    * any other entry that takes an occupied slot RELEASES the old position
      loudly -- a trading_ops row naming what was abandoned and what replaced
      it -- rather than overwriting it in silence.

Release, not close: inventing an exit price and a hold time the strategy never
chose is how a strategy gets convicted on trades it did not make (ab74328).
One lost observation is the honest cost of the collision.
"""

from __future__ import annotations

import asyncio
import time
from unittest import mock

import pytest

from trading.bot import TradingBot
from trading.scheduler import TradeDirective

SYMBOL = "BASECAT-USDC"
BASECAT = "0xb2000000000000000000004c27f6523082f41d01"


class _Stub:
    """Answers any attribute with a no-op callable."""

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
    """A guard that permits the trade, so only the slot rule is under test."""

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


class _GhostOnlyLedger(_Stub):
    """A ledger that has graduated nobody -- every directive stays ghost."""

    def is_live_approved(self, strategy_id):
        return False


class _SwapOutcome:
    ok = True
    broadcast = True
    confirmed = True
    tx_hash = "0xnewhash"
    route = "uniswap"


@pytest.fixture(autouse=True)
def _live_execution(monkeypatch):
    """Live approved and dry-run off: the state link 9 fails in."""
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "0")
    monkeypatch.setenv("STRATEGY_GRADUATION_ENFORCED", "1")


def _directive(strategy_id: str, size: float = 10.0) -> TradeDirective:
    return TradeDirective(
        action="enter", symbol=SYMBOL, base_token="BASECAT", quote_token="USDC",
        size=size, target_price=1.05, horizon="atf", confidence=0.9,
        expected_return=0.05, reason="test", strategy_id=strategy_id,
    )


def _live_position() -> dict:
    """A position holding real tokens: the thing that must survive."""
    return {
        "mode": "live",
        "strategy_id": "atf_static",
        "size": 75.0,
        "entry_price": 0.9,
        "target_price": 1.2,
        "entry_ts": time.time() - 600,
        "ts": time.time() - 600,
        "trade_id": "live-real-money",
        "tx_hash": "0xrealhash",
        "base_token_address": BASECAT,
        "total_quote_spent": 3.0,
    }


def _ghost_position() -> dict:
    return {
        "mode": "ghost",
        "strategy_id": "donchian_breakout@5d",
        "size": 100.0,
        "entry_price": 0.9,
        "target_price": 1.2,
        "entry_ts": time.time() - 7200,
        "ts": time.time() - 7200,
        "trade_id": "ghost-old-trade-id",
    }


def _enter(bot: TradingBot, directive: TradeDirective, *, swapper=None):
    """Drive the entry path. Price matches the held entry price so the
    price-domain guard (which compares the two) is not what answers."""
    sample = {"symbol": SYMBOL, "price": 1.0, "ts": time.time(),
              "chain": "base", "volume": 100.0}
    summary = {"exit_conf": 0.9, "direction_prob": 0.9, "delta": 0.05,
               "net_margin": 0.05, "net_pnl": 0.05}

    async def _no_sync(self, *args, **kwargs):
        return None

    patches = [
        mock.patch.object(
            TradingBot, "_resolve_live_trade_asset",
            lambda self, chain, sym, explicit=None: (sym, explicit or BASECAT),
        ),
        mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync),
    ]
    if swapper is not None:
        patches.append(
            mock.patch.object(TradingBot, "_new_swapper", lambda self: swapper)
        )
    with patches[0], patches[1], (patches[2] if swapper is not None else mock.MagicMock()):
        return asyncio.run(
            bot._interpret_predictions(
                None, sample, directive, pred_summary=summary, brain_summary={},
            )
        )


def _settling_bot() -> TradingBot:
    """A bot whose swap moves real balances, so the live branch completes."""
    bot = _bot()
    bot._bridge = _Stub()
    quantities = {"USDC": 1000.0, "BASECAT": 0.0}

    class _Portfolio(_Stub):
        holdings: dict = {}

        def get_quantity(self, symbol, chain=None):
            return quantities.get(str(symbol).upper(), 0.0)

        def get_native_balance(self, *args, **kwargs):
            return 1.0

    class _Swapper:
        def swap(self, **kwargs):
            quantities["USDC"] -= 3.0
            quantities["BASECAT"] += 75.0
            return _SwapOutcome()

    bot.portfolio = _Portfolio()
    return bot, _Swapper()


# --------------------------------------------------------------------------
# The direction that loses money.
# --------------------------------------------------------------------------

def test_a_ghost_entry_never_overwrites_a_live_position() -> None:
    """The tokens are real; a simulation may not claim their slot."""
    bot = _bot()
    bot.strategy_ledger = _GhostOnlyLedger()
    bot.positions[SYMBOL] = _live_position()

    decision = _enter(bot, _directive("donchian_breakout@5d"))

    assert decision["status"] == "entry-refused-live-held"
    assert decision["action"] == "hold"
    held = bot.positions[SYMBOL]
    # Every field that ties the book to the chain survives untouched.
    assert held["mode"] == "live"
    assert held["trade_id"] == "live-real-money"
    assert held["tx_hash"] == "0xrealhash"
    assert held["size"] == 75.0
    assert held["strategy_id"] == "atf_static"


def test_the_refusal_of_a_ghost_entry_is_recorded() -> None:
    """A skipped observation must not look like a lane that wanted nothing."""
    bot = _bot()
    bot.strategy_ledger = _GhostOnlyLedger()
    bot.positions[SYMBOL] = _live_position()

    _enter(bot, _directive("donchian_breakout@5d"))

    rows = [r for r in bot.db.logged if r.get("status") == "entry-refused-live-held"]
    assert len(rows) == 1
    details = rows[0]["details"]
    assert details["held_trade_id"] == "live-real-money"
    assert details["incoming_strategy_id"] == "donchian_breakout@5d"
    # Refusals are logged under action="hold" so they stay visible without
    # counting as executed live trades (EXECUTED_LIVE_STATUSES in
    # scripts/live_path_check.py matches 'live-entry'/'live-exit' exactly).
    assert rows[0]["action"] == "hold"
    assert rows[0]["status"] not in {"live-entry", "live-exit"}


# --------------------------------------------------------------------------
# The direction that loses evidence.
# --------------------------------------------------------------------------

def test_a_live_entry_releases_a_ghost_position_out_loud() -> None:
    """Real money outranks a simulation -- but the swap is not silent."""
    bot, swapper = _settling_bot()
    bot.positions[SYMBOL] = _ghost_position()

    decision = _enter(bot, _directive("atf_static"), swapper=swapper)

    assert decision["status"] == "live-entry"
    assert bot.positions[SYMBOL]["mode"] == "live"

    released = [r for r in bot.db.logged if r.get("status") == "position-released"]
    assert len(released) == 1
    details = released[0]["details"]
    assert details["released_trade_id"] == "ghost-old-trade-id"
    assert details["released_strategy_id"] == "donchian_breakout@5d"
    assert details["released_mode"] == "ghost"
    assert details["released_size"] == 100.0
    assert details["incoming_mode"] == "live"
    assert details["incoming_strategy_id"] == "atf_static"


def test_a_released_position_is_not_recorded_as_a_completed_trade() -> None:
    """An abandoned observation is not an outcome.

    Recording one would credit or convict the strategy on an exit price and a
    hold time it never chose -- the failure ab74328 is named for. The release
    is a trading_ops row and nothing else.
    """
    bot, swapper = _settling_bot()
    recorded: list = []
    bot.strategy_ledger = _Stub()
    bot.strategy_ledger.is_live_approved = lambda sid: True  # type: ignore[method-assign]
    bot.strategy_ledger.record = lambda *a, **k: recorded.append((a, k))  # type: ignore[method-assign]
    bot.positions[SYMBOL] = _ghost_position()

    _enter(bot, _directive("atf_static"), swapper=swapper)

    assert not [
        call for call in recorded
        if "donchian_breakout@5d" in [str(x) for x in call[0]]
    ], "the abandoned ghost position must not reach the ledger as an outcome"
    assert not [r for r in bot.db.logged if str(r.get("status", "")).endswith("-exit")]


def test_the_released_symbol_is_claimed_so_it_cannot_be_resurrected() -> None:
    """_save_state only removes symbols this bot has claimed.

    Without the claim the abandoned row would be merged back in from the
    shared book on the next save, and the symbol would carry two histories.
    """
    bot, swapper = _settling_bot()
    bot.positions[SYMBOL] = _ghost_position()

    _enter(bot, _directive("atf_static"), swapper=swapper)

    assert SYMBOL in bot._owned_symbols


# --------------------------------------------------------------------------
# The ordinary case must not have changed.
# --------------------------------------------------------------------------

def test_an_entry_on_a_free_symbol_releases_nothing() -> None:
    bot, swapper = _settling_bot()
    assert SYMBOL not in bot.positions

    decision = _enter(bot, _directive("atf_static"), swapper=swapper)

    assert decision["status"] == "live-entry"
    assert not [r for r in bot.db.logged if r.get("status") == "position-released"]


def test_a_failed_live_entry_leaves_the_held_position_alone() -> None:
    """The release happens after the swap, so a swap that never fills is inert."""
    bot, _ = _settling_bot()
    bot.positions[SYMBOL] = _ghost_position()

    class _DeadSwapper:
        def swap(self, **kwargs):          # no balance movement -> no fill
            return _SwapOutcome()

    decision = _enter(bot, _directive("atf_static"), swapper=_DeadSwapper())

    assert decision["status"] == "live-entry-failed"
    assert decision["reason"] == "no_fill_detected"
    held = bot.positions[SYMBOL]
    assert held["mode"] == "ghost"
    assert held["trade_id"] == "ghost-old-trade-id"
    assert not [r for r in bot.db.logged if r.get("status") == "position-released"]
