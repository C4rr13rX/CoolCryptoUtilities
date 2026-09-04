"""A slot holding real tokens refuses every incoming entry, live or ghost.

``_release_position_for_entry`` has always documented this as an invariant it
does not have to enforce::

    A live position is never released here -- ``_interpret_predictions``
    refuses a non-live entry before reaching this point, and a live entry
    replacing a live position on the same symbol would strand the first
    one's tokens. That case is asserted against in the tests rather than
    handled, because the entry path cannot produce it.

The entry path produced it four times on 2026-09-03. Every one is a
``position-released`` row in ``trading_ops`` carrying ``released_mode: "live"``:

    15:38:02  CBETH-USDC  released 2:CBETH-USDC:bfa397e6  size 0.000111750475
    16:24:26  CBBTC-USDC  released 2:CBBTC-USDC:c9d8cee9  size 0.00000928
    16:38:22  CBBTC-USDC  released 2:CBBTC-USDC:c83106b5  size 0.00000927
    16:40:45  CBBTC-USDC  released 2:CBBTC-USDC:12cd624b  size 0.00000926

all four ``incoming_mode: "live"``, all four ``atf_static`` onto its own live
slot, and each immediately followed by another 0.75 USDC buy of the same token.
The wallet ended the day holding 0.0000370900 CBBTC -- 3.008 USD at the
81099.62 cbBTC print -- behind FOUR settled buys and zero sells, with nothing in
the position book pointing at any of it. Every exit in this bot is driven off
``self.positions[symbol]``, so a released live position is not a lost
observation, it is money that nothing will ever attempt to sell.

The exemption was ``and not entry_is_live``, at both the refusal site and the
``entry_refused_by_live_slot`` predicate feeding the protective bracket. The
same-strategy guard added in c6dfa38 covers only the case where the incoming
directive carries the SAME ``strategy_id`` as the held position -- it does not
fire for a second live-approved strategy, nor for a directive with no
strategy_id at all. The slot's mode is the property that decides, so these
tests interrogate the slot, not the asker.
"""

from __future__ import annotations

import asyncio
import time
from unittest import mock

import pytest

from trading.bot import TradingBot
from trading.scheduler import TradeDirective

SYMBOL = "CBBTC-USDC"
CBBTC = "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf"

HELD_SIZE = 9.28e-06
HELD_ENTRY = 80818.96551724139
HELD_TX = "0x53f7303248255d6923b42c5920a58413c8b4c85b57cb54df01be0e14899b07ad"


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

    def fetch_trades(self, *args, **kwargs):
        return []

    def statuses(self):
        return [row.get("status") for row in self.logged]


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
    bot._insufficient_quote_last_ts = 0.0
    return bot


@pytest.fixture(autouse=True)
def _live_execution(monkeypatch):
    """Live approved and dry-run off: the state the four releases happened in."""
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "0")
    monkeypatch.setenv("STRATEGY_GRADUATION_ENFORCED", "1")


def _live_position() -> dict:
    """The CBBTC position released at 16:24:26, as the book held it."""
    return {
        "mode": "live",
        "strategy_id": "atf_static",
        "size": HELD_SIZE,
        "entry_price": HELD_ENTRY,
        "target_price": HELD_ENTRY * 1.05,
        "entry_ts": time.time() - 600,
        "ts": time.time() - 600,
        "trade_id": "2:CBBTC-USDC:c9d8cee9485047ec8fa7c10ab247e3b0",
        "entry_tx_hash": HELD_TX,
        "base_token_address": CBBTC,
        "quote_spent": 0.75,
    }


def _directive(strategy_id: str) -> TradeDirective:
    return TradeDirective(
        action="enter",
        symbol=SYMBOL,
        base_token="CBBTC",
        quote_token="USDC",
        size=9.2e-06,
        target_price=HELD_ENTRY * 1.06,
        horizon="atf",
        confidence=0.9,
        expected_return=0.05,
        reason="test",
        strategy_id=strategy_id,
    )


def _enter(bot: TradingBot, directive: TradeDirective, *, price: float = HELD_ENTRY):
    async def _no_sync(self, *args, **kwargs):
        return None

    sample = {
        "symbol": SYMBOL,
        "price": price,
        "ts": time.time(),
        "chain": "base",
        "volume": 100.0,
    }
    summary = {
        "exit_conf": 0.9,
        "direction_prob": 0.9,
        "delta": 0.05,
        "net_margin": 0.05,
        "net_pnl": 0.05,
    }
    with mock.patch.object(
        TradingBot,
        "_resolve_live_trade_asset",
        lambda self, chain, sym, explicit=None: (sym, explicit or CBBTC),
    ), mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync), mock.patch.object(
        TradingBot, "_adopt_orphaned_live_holding", lambda self, *a, **k: None
    ), mock.patch.object(
        # The premise of every test in this file is a live slot that is REAL:
        # HELD_SIZE of CBBTC bought by HELD_TX, on chain on 2026-09-03. The bot
        # now reconciles the book against the wallet before any refusal reads
        # it, and this stand-in has no chain to read, so without this the
        # position is correctly judged sold and dropped -- and then there is no
        # slot left to refuse anything, which is a different scenario.
        # Asserting the premise, not disabling the check: the phantom case is
        # covered in tests/test_phantom_position_never_blocks.py.
        TradingBot, "_position_is_real_on_chain", lambda self, *a, **k: True
    ):
        return asyncio.run(
            bot._interpret_predictions(
                None, sample, directive, pred_summary=summary, brain_summary={}
            )
        )


def test_a_live_approved_entry_by_the_same_strategy_does_not_release_the_slot():
    """atf_static onto its own live CBBTC slot: the 16:24, 16:38 and 16:40 rows."""
    bot = _bot()
    held = _live_position()
    bot.positions[SYMBOL] = held

    decision = _enter(bot, _directive("atf_static"))

    assert decision.get("status") != "live-entry"
    survivor = bot.positions.get(SYMBOL)
    assert survivor is not None, "the live position was released"
    assert survivor["mode"] == "live"
    assert survivor["size"] == HELD_SIZE
    assert survivor["entry_tx_hash"] == HELD_TX
    assert "position-released" not in bot.db.statuses()


def test_a_live_approved_entry_by_a_DIFFERENT_strategy_does_not_release_the_slot():
    """The gap the same-strategy guard leaves open.

    c6dfa38 keys its refusal on ``strategy_id == held strategy_id``. A second
    live-approved strategy therefore walked straight past it into the release.
    """
    bot = _bot()
    bot.positions[SYMBOL] = _live_position()

    decision = _enter(bot, _directive("supertrend_follow@5h"))

    assert decision.get("status") != "live-entry"
    survivor = bot.positions.get(SYMBOL)
    assert survivor is not None, "a second strategy released the live position"
    assert survivor["trade_id"] == "2:CBBTC-USDC:c9d8cee9485047ec8fa7c10ab247e3b0"
    assert survivor["size"] == HELD_SIZE
    assert "position-released" not in bot.db.statuses()


def test_a_live_approved_entry_with_no_strategy_id_does_not_release_the_slot():
    """The other gap: an empty strategy_id never equals the held one."""
    bot = _bot()
    bot.positions[SYMBOL] = _live_position()

    _enter(bot, _directive(""))

    survivor = bot.positions.get(SYMBOL)
    assert survivor is not None, "an unclassified directive released the live position"
    assert survivor["size"] == HELD_SIZE
    assert "position-released" not in bot.db.statuses()


def test_the_refusal_is_logged_as_a_live_row_when_the_entry_was_live():
    """A refused LIVE entry must be findable in a wallet='live' query.

    The refusal used to log ``wallet="ghost"`` unconditionally, on the reading
    that only ghost entries could ever be refused here. That is exactly why
    four live-over-live releases went unnoticed for a day.

    A second strategy, because the same-strategy case is caught earlier by the
    duplicate guard and logged as ``entry-refused-duplicate`` -- this site is
    the one that answers for everything that guard's strategy_id test misses.
    """
    bot = _bot()
    bot.positions[SYMBOL] = _live_position()

    _enter(bot, _directive("supertrend_follow@5h"))

    refusals = [
        row for row in bot.db.logged if row.get("status") == "entry-refused-live-held"
    ]
    assert refusals, "the refusal was not logged at all"
    assert refusals[-1]["wallet"] == "live"
    assert refusals[-1]["details"]["incoming_entry_is_live"] is True


def test_release_refuses_a_live_position_at_the_choke_point():
    """The enforcement, independent of which caller reaches it.

    ``_release_position_for_entry`` is where the position actually disappears.
    Guarding only its callers is what left the door open the first time.
    """
    bot = _bot()
    held = _live_position()
    bot.positions[SYMBOL] = held

    freed = bot._release_position_for_entry(
        SYMBOL,
        chain="base",
        incoming_mode="live",
        incoming_strategy="atf_static",
        incoming_trade_id="2:CBBTC-USDC:c83106b5b6414ebcb1da57a16afcef27",
    )

    assert freed is False, "the choke point still frees a live slot"
    assert bot.positions[SYMBOL] is held
    assert "live-position-release-refused" in bot.db.statuses()


def test_release_still_frees_a_ghost_slot_for_a_live_entry():
    """The ghost->live upgrade must keep working.

    Refusing it would mean a graduated strategy can never take the live entry
    for any symbol its own ghost lane happens to hold -- 7 of the 9 live-capable
    symbols on 2026-09-02.
    """
    bot = _bot()
    bot.positions[SYMBOL] = {
        "mode": "ghost",
        "strategy_id": "atf_static",
        "size": 100.0,
        "entry_price": HELD_ENTRY,
        "entry_ts": time.time() - 600,
        "ts": time.time() - 600,
        "trade_id": "ghost-trade",
    }

    freed = bot._release_position_for_entry(
        SYMBOL,
        chain="base",
        incoming_mode="live",
        incoming_strategy="atf_static",
        incoming_trade_id="incoming",
    )

    assert freed is True
    assert SYMBOL not in bot.positions
    assert "position-released" in bot.db.statuses()


def test_release_of_an_empty_slot_is_free():
    bot = _bot()
    assert (
        bot._release_position_for_entry(
            SYMBOL,
            chain="base",
            incoming_mode="live",
            incoming_strategy="atf_static",
            incoming_trade_id="incoming",
        )
        is True
    )
