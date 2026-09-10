"""The entry that settles on-chain must open a position, not report failure.

This is the end-to-end half of test_fill_is_read_from_the_receipt.py. On
2026-09-03 both of R3V3N!R's first real trades settled on Base and both were
written down as ``live-entry-failed / no_fill_detected``, leaving 19.488243
BASECAT and 1.546281 AERO in the wallet with nothing in the position book
pointing at them and no exit path. Production had to be stopped by hand: a
failed entry never fills the slot that would stop it re-firing, so it would
have re-spent $0.75 every cycle until the wallet was empty.

The bot below is set up in exactly the state that produced that, with both
independent causes present at once:

  * the portfolio reports 0.0 of the token being bought, because the transfer
    indexer has not discovered it -- true of every first purchase;
  * the quote balance falls by 1.5 for a 0.75 swap, because a sibling bot
    sharing the wallet settled inside the same window.

Read by wallet delta this is "no fill, and it cost double". Read from the
receipt it is 0.750000 USDC for 19.488243 BASECAT, which is what happened.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.fill_receipt import read_fill
from trading.bot import TradingBot
from trading.scheduler import TradeDirective

SYMBOL = "BASECAT-USDC"
BASECAT = "0xb2000000000000000000004c27f6523082f41d01"
USDC = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
WALLET = "0x291c854811e92906a658Fb94Aa511bF919f968ad"
REAL_HASH = "0xfd133cfe29d0018e778275bb8a9c2ba89b8d63cf4fabe87d773dfe8c9688b880"

_RECEIPT = json.loads(
    (Path(__file__).parent / "fixtures" / "base_live_swap_receipts.json").read_text()
)["BASECAT"]["receipt"]


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


class _Ledger(_Stub):
    def is_live_approved(self, strategy_id):
        return True


class _Validator(_Stub):
    def validate(self, **kwargs):
        return True, {}, []


class _Pipeline(_Stub):
    decision_threshold = 0.58


class _Bridge(_Stub):
    def get_address(self):
        return WALLET


class _RealisticSwapper:
    """Broadcasts, and reads its fill the way SwapService now does.

    ``read_fill`` runs the real parser over the real receipt; only the network
    fetch is replaced. So this exercises the production code path, not a
    hand-written stand-in for its answer.
    """

    def __init__(self, quantities: dict):
        self.quantities = quantities
        self.swaps = 0

    def swap(self, **kwargs):
        self.swaps += 1
        # The wallet moves by 1.5 -- our 0.75 plus a sibling bot's 0.75 -- and
        # BASECAT stays at 0.0 because the indexer has not seen it yet.
        self.quantities["USDC"] -= 1.5
        return type(
            "_Outcome", (), {"ok": True, "broadcast": True, "confirmed": True,
                             "tx_hash": REAL_HASH, "route": "UniswapV3"},
        )()

    def read_fill(self, chain, txh, *, sell, buy, wallet=None, **kwargs):
        assert txh == REAL_HASH
        return read_fill(
            _RECEIPT, wallet=wallet or WALLET, sell_token=sell, buy_token=buy,
            sell_decimals=6, buy_decimals=18,
        )


def _bot(quantities: dict) -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    TradingBot._ensure_runtime_state(bot)

    class _Portfolio(_Stub):
        holdings: dict = {}

        def get_quantity(self, symbol, chain=None):
            return quantities.get(str(symbol).upper(), 0.0)

        def get_native_balance(self, *args, **kwargs):
            return 1.0

    bot.db = _DB()
    bot.strategy_ledger = _Ledger()
    bot.portfolio = _Portfolio()
    bot.swap_validator = _Validator()
    bot.pipeline = _Pipeline()
    for name in ("metrics", "scheduler", "memory", "graph", "rotator",
                 "equilibrium", "_buffer"):
        setattr(bot, name, _Stub())
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
    bot._bridge = _Bridge()
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
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "0")
    monkeypatch.setenv("STRATEGY_GRADUATION_ENFORCED", "1")


def _enter(bot: TradingBot, swapper):
    directive = TradeDirective(
        action="enter", symbol=SYMBOL, base_token="BASECAT", quote_token="USDC",
        size=19.5, target_price=0.045, horizon="atf", confidence=0.9,
        expected_return=0.05, reason="test", strategy_id="atf_static",
    )
    sample = {"symbol": SYMBOL, "price": 0.0384847, "ts": time.time(),
              "chain": "base", "volume": 100.0}
    summary = {"exit_conf": 0.9, "direction_prob": 0.9, "delta": 0.05,
               "net_margin": 0.05, "net_pnl": 0.05}

    async def _no_sync(self, *args, **kwargs):
        return None

    with mock.patch.object(
        TradingBot, "_resolve_live_trade_asset",
        lambda self, chain, sym, explicit=None: (
            sym, explicit or (BASECAT if str(sym).upper() == "BASECAT" else USDC)
        ),
    ), mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync), mock.patch.object(
        TradingBot, "_new_swapper", lambda self: swapper
    ):
        return asyncio.run(
            bot._interpret_predictions(
                None, sample, directive, pred_summary=summary, brain_summary={},
            )
        )


def test_a_settled_entry_is_not_reported_as_no_fill_detected():
    """The exact 2026-09-03 failure, with both of its causes present."""
    quantities = {"USDC": 6.977334, "BASECAT": 0.0}
    bot = _bot(quantities)
    swapper = _RealisticSwapper(quantities)

    decision = _enter(bot, swapper)

    assert decision["status"] == "live-entry", decision.get("reason")
    assert decision.get("reason") != "no_fill_detected"
    assert decision["tx_hash"] == REAL_HASH
    assert decision["fill_source"] == "tx_receipt"
    # The portfolio still reports 0.0 BASECAT -- the fix does not depend on
    # the indexer catching up.
    assert quantities["BASECAT"] == 0.0


def test_the_position_records_what_the_chain_says_it_bought():
    quantities = {"USDC": 6.977334, "BASECAT": 0.0}
    bot = _bot(quantities)

    _enter(bot, _RealisticSwapper(quantities))

    held = bot.positions[SYMBOL]
    assert held["mode"] == "live"
    assert held["size"] == pytest.approx(19.488243173989281, rel=1e-12)
    assert held["entry_price"] == pytest.approx(0.0384847, abs=1e-7)
    assert held["entry_tx_hash"] == REAL_HASH
    assert held["fill_source"] == "tx_receipt"
    # The tokens have somewhere to be sold back to; the orphaned holdings of
    # 2026-09-03 had no exit path precisely because no position was written.
    assert held["base_token_address"].lower() == BASECAT


def test_the_sibling_bots_swap_does_not_double_the_recorded_cost():
    """The wallet fell by 1.5; this trade cost 0.75 and must say so.

    ``quote_spent`` is what the exit divides its proceeds against, so booking
    1.5 here turns a flat round trip into a 50% loss on the record that gates
    graduation.
    """
    quantities = {"USDC": 6.977334, "BASECAT": 0.0}
    bot = _bot(quantities)

    decision = _enter(bot, _RealisticSwapper(quantities))

    assert quantities["USDC"] == pytest.approx(5.477334, abs=1e-9)  # -1.5 really
    assert decision["quote_spent"] == pytest.approx(0.75, abs=1e-9)  # ours alone
    assert bot.positions[SYMBOL]["quote_spent"] == pytest.approx(0.75, abs=1e-9)


def test_the_entry_fires_once_and_does_not_respin_the_swap():
    """A failed entry re-fires every cycle; seven more would empty the wallet."""
    quantities = {"USDC": 6.977334, "BASECAT": 0.0}
    bot = _bot(quantities)
    swapper = _RealisticSwapper(quantities)

    _enter(bot, swapper)
    assert swapper.swaps == 1
    # The slot is now filled, so the next cycle has a position to manage
    # rather than an empty book inviting another purchase.
    assert SYMBOL in bot.positions


def test_a_swapper_that_cannot_read_receipts_falls_back_rather_than_raising():
    """The money has already moved; a bookkeeping gap must not raise.

    An exception after the broadcast loses a settled trade exactly the way the
    wallet delta did. Here the fallback correctly finds no fill (BASECAT is
    still 0.0 in the portfolio) -- the point is that it reports that, in the
    ordinary decision shape, instead of blowing up the entry path.
    """
    quantities = {"USDC": 6.977334, "BASECAT": 0.0}
    bot = _bot(quantities)

    class _OldSwapper(_RealisticSwapper):
        read_fill = None  # attribute exists but is not callable

        def __getattribute__(self, name):
            if name == "read_fill":
                raise AttributeError(name)
            return object.__getattribute__(self, name)

    decision = _enter(bot, _OldSwapper(quantities))

    assert decision["status"] == "live-entry-failed"
    assert decision["reason"] == "no_fill_detected"
    assert decision["fill_source"] == "wallet_delta"


def test_a_raising_fill_reader_does_not_lose_the_trade_path():
    quantities = {"USDC": 6.977334, "BASECAT": 0.0}
    bot = _bot(quantities)

    class _Exploding(_RealisticSwapper):
        def read_fill(self, *args, **kwargs):
            raise RuntimeError("rpc exploded")

    decision = _enter(bot, _Exploding(quantities))

    # Degraded, but still a decision -- no exception escaped the entry.
    assert decision["status"] == "live-entry-failed"
    assert decision["fill_source"] == "wallet_delta"


# THE FIXTURE PINS THE SYMBOL EDGE GATE, IT DOES NOT WEAKEN IT.
#
# Every test in this file is about what the ENTRY BOOKS once it fires --
# the receipt fill, the recorded cost, the single spin of the swap. None of
# them is about whether the edge gate lets the entry through, and the gate
# sits UPSTREAM of every one of those assertions.
#
# Without this pin these tests read production data. services/symbol_edge_gate
# opens storage/trading_cache.db at test time and answers from the live closed
# book, so on 2026-09-10 SYMBOL (BASECAT-USDC) crossed the ban threshold -- 35
# closed round trips at mean -0.852% against 0.465% cost, gross -1.4379 -- and
# turned six green tests red with no code change at all. A test whose verdict
# moves when the bots trade is not testing the code.
#
# The ban is CORRECT and stays: services.symbol_edge_gate.refusal_reason
# still returns it, and the gate's own coverage lives with the gate. What is
# pinned here is only this file's precondition -- that the entry is reached.
@pytest.fixture(autouse=True)
def _entry_reaches_the_booking_path(monkeypatch):
    monkeypatch.setattr(
        "trading.bot.symbol_edge_refusal", lambda _symbol, _strategy_id=None: None
    )
