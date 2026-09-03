"""A ghost position must be closed out of the ghost book, not the wallet.

``_interpret_predictions`` picks the balances it works from with one bot-level
switch::

    use_sim = not self.live_trading_enabled

so the day the bot went live, ``available_base`` for EVERY position -- ghost
ones included -- started coming from ``portfolio.get_quantity()``. A simulated
position holds no tokens, so that number is 0.0, and the exit is sized by it::

    exit_size = min(exit_target, available_base)   # -> 0.0
    if exit_size <= 0.0: ... "insufficient_base"; return decision

Measured on 2026-09-03 over the previous six hours: **1030** `insufficient_base`
refusals, ``available`` exactly 0.0 on every one, spread over 14 symbols the
wallet has never held -- CBXRP 201, BASECAT 209, BASEPEPE 137, TYBG 115, CP 109.
Ghost positions in the book at the time had been open for 3h to 15h with a
take-profit target long since passed.

Why it matters beyond the stuck book: a ghost exit is what writes the outcome
that reaches ``StrategyLedger.record()``, and graduation needs 20 of them. With
exits refused, every strategy sits at 1-8 closed trades forever (money_button
has exactly 1), and the un-closable positions keep their symbols occupied, so
live entries on them come back `entry-refused-live-held`.

And the truncation is worse than the block. When the wallet happened to hold
SOME of the ticker, `min()` silently sold that instead: CBETH-USDC closed
0.0001117 of a 0.0006270 ghost position at 10:49 UTC and booked it as a whole
trade -- a fabricated partial, of exactly the kind ab74328 was about.

The rule pinned here:

    * a GHOST position's exit size comes from the ghost book, whatever the
      wallet holds -- including a wallet holding none of it, and including a
      wallet holding a dust amount smaller than the position;
    * a LIVE position's exit size still comes from the wallet, because it sells
      real tokens and may not offer more than it owns.
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

ENTRY_PRICE = 0.90
TARGET_PRICE = 1.00
TICK_PRICE = 1.05  # past the target, so the exit is take-profit
GHOST_SIZE = 100.0


class _Stub:
    """Answers any attribute with a no-op callable."""

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


class _DB(_Stub):
    def __init__(self):
        self.logged: list = []
        self.outcomes: list = []

    def log_trade(self, **kwargs):
        self.logged.append(kwargs)
        return True

    def get_pair_adjustment(self, symbol):
        return {}

    def fetch_market_samples_for(self, *args, **kwargs):
        return []

    def fetch_trade_fills(self, *args, **kwargs):
        return []

    def record_trade_outcome(self, *args, **kwargs):
        # Truthy = "this outcome is new". A _Stub's None reads as a duplicate
        # and short-circuits the exit before it books anything.
        self.outcomes.append(kwargs or {"args": args})
        return True


class _Metrics(_Stub):
    """Captures feedback so the refusal itself can be asserted on."""

    def __init__(self):
        self.events: list = []

    def feedback(self, source, severity=None, label="", details=None, **kwargs):
        self.events.append({"source": source, "label": label, "details": details or {}})
        return None

    def by_label(self, label: str) -> list:
        return [e for e in self.events if e["label"] == label]


class _Ledger(_Stub):
    def __init__(self):
        self.recorded: list = []

    def is_live_approved(self, strategy_id):
        return True

    def record(self, *args, **kwargs):
        self.recorded.append((args, kwargs))
        return None


class _Validator(_Stub):
    def validate(self, **kwargs):
        return True, {}, []


class _Pipeline(_Stub):
    decision_threshold = 0.58


def _portfolio(base_quantity: float):
    """A wallet holding `base_quantity` of BASECAT and plenty of USDC."""

    class _Portfolio(_Stub):
        holdings: dict = {}

        def get_quantity(self, symbol, chain=None):
            if str(symbol).upper() == "BASECAT":
                return base_quantity
            return 1000.0

        def get_native_balance(self, *args, **kwargs):
            return 1.0

    return _Portfolio()


def _bot(*, wallet_base: float) -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    TradingBot._ensure_runtime_state(bot)
    bot.db = _DB()
    bot.strategy_ledger = _Ledger()
    bot.portfolio = _portfolio(wallet_base)
    bot.swap_validator = _Validator()
    bot.pipeline = _Pipeline()
    bot.metrics = _Metrics()
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
    # A funded sim purse, so the non-live control case does not trip the
    # "sim pool is zero" reinitialiser instead of exercising the exit.
    bot.sim_quote_balances = {("base", "USDC"): 100.0}
    bot.stable_tokens = {"USDC"}
    bot.primary_chain = "base"
    # The state this bug needs: the BOT is live, the POSITION is not.
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
    bot._sim_initial_pool = 100.0
    # Post-exit bookkeeping the non-live control case reaches. Off: promotion
    # is not what these tests are about.
    bot.auto_promote_live = False
    return bot


def _position(mode: str, size: float = GHOST_SIZE) -> dict:
    opened = time.time() - 7200
    return {
        "mode": mode,
        "strategy_id": "donchian_breakout@5d" if mode == "ghost" else "atf_static",
        "size": size,
        "entry_price": ENTRY_PRICE,
        "target_price": TARGET_PRICE,
        "entry_ts": opened,
        "ts": opened,
        "trade_id": f"{mode}-trade-id",
        "route": ["BASECAT", "USDC"],
        "quote_spent": ENTRY_PRICE * size,
        "base_token_address": BASECAT,
    }


def _tick(bot: TradingBot, directive: TradeDirective | None = None):
    """Drive one sample priced above the position's take-profit target."""
    sample = {"symbol": SYMBOL, "price": TICK_PRICE, "ts": time.time(),
              "chain": "base", "volume": 100.0}
    summary = {"exit_conf": 0.9, "direction_prob": 0.9, "delta": 0.05,
               "net_margin": 0.05, "net_pnl": 0.05}

    async def _no_sync(self, *args, **kwargs):
        return None

    with mock.patch.object(
        TradingBot, "_resolve_live_trade_asset",
        lambda self, chain, sym, explicit=None: (sym, explicit or BASECAT),
    ), mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync):
        return asyncio.run(
            bot._interpret_predictions(
                None, sample, directive, pred_summary=summary, brain_summary={},
            )
        )


@pytest.fixture(autouse=True)
def _live_execution(monkeypatch):
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "0")
    monkeypatch.setenv("STRATEGY_GRADUATION_ENFORCED", "1")
    # MAX_POSITION_PRICE_RATIO defaults to 5.0; 1.05/0.90 is well inside it, so
    # the price-domain guard is not what answers in any of these tests.
    monkeypatch.delenv("MAX_POSITION_PRICE_RATIO", raising=False)


# --------------------------------------------------------------------------
# The bug: a live bot could not close a simulated position.
# --------------------------------------------------------------------------

def test_ghost_position_exits_on_a_live_bot_with_an_empty_wallet() -> None:
    """The wallet holds none of the token. It is not supposed to."""
    bot = _bot(wallet_base=0.0)
    bot.positions[SYMBOL] = _position("ghost")

    decision = _tick(bot)

    assert decision["status"] == "ghost-exit", decision.get("reason")
    assert decision["action"] == "exit"
    assert decision["wallet"] == "ghost"
    # The whole simulated position closed, sized from the book.
    assert decision["size"] == pytest.approx(GHOST_SIZE)
    assert SYMBOL not in bot.positions
    # ... and the refusal that used to answer here never fired.
    assert bot.metrics.by_label("insufficient_base") == []


def test_ghost_exit_is_not_truncated_to_a_dust_wallet_balance() -> None:
    """A stray 0.5 BASECAT in the wallet must not resize a 100-unit ghost trade.

    This is the fabrication direction: `min(exit_target, wallet)` books a
    fraction of the position as if the strategy had chosen to sell it.
    """
    bot = _bot(wallet_base=0.5)
    bot.positions[SYMBOL] = _position("ghost")

    decision = _tick(bot)

    assert decision["status"] == "ghost-exit"
    assert decision["size"] == pytest.approx(GHOST_SIZE)
    assert decision["remaining_size"] == pytest.approx(0.0, abs=1e-9)


def test_ghost_exit_writes_an_outcome_the_ledger_can_grade() -> None:
    """Graduation counts closed ghost trades; a refused exit closes none."""
    bot = _bot(wallet_base=0.0)
    bot.positions[SYMBOL] = _position("ghost")

    decision = _tick(bot)

    assert decision["status"] == "ghost-exit"
    assert decision["strategy_id"] == "donchian_breakout@5d"
    # Sold above entry, so this is the profitable direction and the number is
    # a real subtraction, not a constant.
    assert decision["exit_price"] == pytest.approx(TICK_PRICE)
    assert decision["entry_price"] == pytest.approx(ENTRY_PRICE)
    # The outcome row is the artefact graduation reads; a refused exit writes
    # none, which is why every strategy stalled at 1-8 closed trades.
    assert bot.db.outcomes, "a closed ghost trade must commit a trade outcome"


# --------------------------------------------------------------------------
# The half that must NOT change: real tokens are still governed by the wallet.
# --------------------------------------------------------------------------

def test_live_position_exit_is_still_read_from_the_wallet() -> None:
    """A live position may not offer more tokens than the wallet holds.

    There is no signing bridge in a test process, so the live branch stops at
    `live-exit-blocked` -- which is already past the sizing gate, and is the
    proof that the wallet lookup (not the book) is what answered.
    """
    bot = _bot(wallet_base=40.0)
    bot.positions[SYMBOL] = _position("live", size=100.0)

    decision = _tick(bot)

    assert bot.metrics.by_label("insufficient_base") == []
    assert decision["status"] == "live-exit-blocked"
    assert decision["wallet"] == "live"
    assert decision["executed"] is False


def test_live_position_with_an_empty_wallet_is_refused_not_closed() -> None:
    """No tokens, no sale -- and no invented exit either."""
    bot = _bot(wallet_base=0.0)
    bot.positions[SYMBOL] = _position("live", size=100.0)

    decision = _tick(bot)

    assert decision["action"] != "exit"
    assert decision.get("status") != "live-exit"
    # The wallet, not the book, is what the live branch measured.
    refusals = bot.metrics.by_label("insufficient_base")
    assert refusals, "a live exit with no tokens must still be refused"
    assert refusals[0]["details"]["available"] == pytest.approx(0.0)
    assert refusals[0]["details"]["held"] == pytest.approx(100.0)
    # The position survives: it is the only record that the tokens were bought.
    assert bot.positions[SYMBOL]["size"] == pytest.approx(100.0)
    assert bot.positions[SYMBOL]["mode"] == "live"


def test_a_bot_that_is_not_live_is_unaffected() -> None:
    """The pre-existing sim path: ghost bot, ghost position, ghost book."""
    bot = _bot(wallet_base=0.0)
    bot.live_trading_enabled = False
    bot.positions[SYMBOL] = _position("ghost")

    decision = _tick(bot)

    assert decision["status"] == "ghost-exit"
    assert decision["size"] == pytest.approx(GHOST_SIZE)
