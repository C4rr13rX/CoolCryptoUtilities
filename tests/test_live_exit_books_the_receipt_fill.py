"""The exit is where a wrong fill books a profit nobody earned.

The entry half of this is test_live_entry_books_the_receipt_fill.py. The exit
matters more, and had no test at all: an entry read wrong loses a position, but
an EXIT read wrong writes a number into the strategy ledger, and the ledger is
what gates whether a strategy may keep spending real money. This repo has
already convicted a strategy on trades it did not make (ab74328) and destroyed
a ledger by letting one fire on fabricated evidence.

Both wallet-delta failures apply to the exit as well, in mirror image:

  * ``quote_received`` is measured against a USDC balance that a sibling bot's
    swap moves inside the same window -- one wallet, one bot per symbol.
  * ``base_sold`` is measured against a token row the indexer may not have,
    and a sale that reads zero sold is an exit that never happened.

The receipt fixture here is a real swap of ours read in the sell direction:
0.750000 USDC received for 1.546280953235675228 AERO sold. Reading it as an
exit of a position opened at 0.485 gives a small real loss, which is what is
asserted -- a losing round trip must be recorded as a loss.
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

from services.fill_receipt import ReceiptFill, read_fill
from trading.bot import TradingBot
from trading.scheduler import TradeDirective

SYMBOL = "AERO-USDC"
AERO = "0x940181a94a35a4569e4529a3cdfb74e38fd98631"
USDC = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
WALLET = "0x291c854811e92906a658Fb94Aa511bF919f968ad"
EXIT_HASH = "0xd4c2d4df7886a80f4113314d518772e80113a68dd64c3321f2e873aec7c9c196"

_RAW = json.loads(
    (Path(__file__).parent / "fixtures" / "base_live_swap_receipts.json").read_text()
)["AERO"]["receipt"]


def _sell_receipt() -> dict:
    """The real receipt with its two Transfer legs reversed.

    Swapping ``topics[1]``/``topics[2]`` on each Transfer turns the recorded
    BUY of AERO into the SALE of the same amounts -- real log layout, real
    amounts, real gas, opposite direction. Nothing here is invented but the
    direction.
    """
    flipped = []
    for lg in _RAW["logs"]:
        topics = list(lg["topics"])
        if len(topics) == 3:
            topics[1], topics[2] = topics[2], topics[1]
        flipped.append({**lg, "topics": topics})
    return {**_RAW, "logs": flipped}


class _Stub:
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

    def record_trade_outcome(self, **kwargs):
        # Must return True. The real DB returns whether the row was INSERTed,
        # and a bare _Stub returns None -- which the bot correctly reads as
        # "this outcome_id is already committed" and abandons the exit AFTER
        # the swap has settled. That is a fake being less capable than the
        # class, and it silently skipped every assertion below.
        self.outcomes.append(kwargs)
        return True

    def get_pair_adjustment(self, symbol):
        return {}

    def fetch_market_samples_for(self, *args, **kwargs):
        return []

    def fetch_trade_fills(self, *args, **kwargs):
        return []


class _Ledger(_Stub):
    def __init__(self):
        self.recorded: list = []

    def is_live_approved(self, strategy_id):
        return True

    def record(self, *args, **kwargs):
        self.recorded.append((args, kwargs))
        return None


class _Bridge(_Stub):
    def get_address(self):
        return WALLET


class _ExitSwapper:
    """Sells, and reads its fill through the real parser on a real receipt."""

    def __init__(self, quantities: dict):
        self.quantities = quantities
        self.swaps = 0

    def swap(self, **kwargs):
        self.swaps += 1
        # A sibling bot buys 0.40 USDC of something in the same window, so the
        # quote balance rises by less than we received. The delta would book
        # that difference as our loss.
        self.quantities["USDC"] += 0.75 - 0.40
        self.quantities["AERO"] = 0.0
        return type(
            "_Outcome", (), {"ok": True, "broadcast": True, "confirmed": True,
                             "tx_hash": EXIT_HASH, "route": "UniswapV3"},
        )()

    def read_fill(self, chain, txh, *, sell, buy, wallet=None, **kwargs):
        assert txh == EXIT_HASH
        return read_fill(
            _sell_receipt(), wallet=wallet or WALLET,
            sell_token=sell, buy_token=buy,
            sell_decimals=18, buy_decimals=6,
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
    bot.swap_validator = type("_V", (_Stub,), {"validate": lambda s, **k: (True, {}, [])})()
    bot.pipeline = type("_P", (_Stub,), {"decision_threshold": 0.58})()
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


def _live_position() -> dict:
    """1.546281 AERO bought for 0.75 USDC -- the real open position."""
    opened = time.time() - 6 * 3600  # long past any minimum hold
    return {
        "mode": "live",
        "strategy_id": "atf_static",
        "size": 1.546280953235675,
        "entry_price": 0.4850347528569017,
        "quote_spent": 0.75,
        "gas_spent_native": 8.84178633624e-07,
        "target_price": 0.60,
        "entry_ts": opened,
        "ts": opened,
        "trade_id": "live-aero-round-trip",
        "entry_tx_hash": "0xd4c2d4df7886a80f4113314d518772e80113a68dd64c3321f2e873aec7c9c196",
        "base_token_address": AERO,
        "quote_token_address": USDC,
        "base_symbol": "AERO",
        "quote_symbol": "USDC",
        "fill_source": "tx_receipt",
        "exit_sequence": 0,
        "trigger_state": {"high_watermark": 0.4850347528569017},
    }


@pytest.fixture(autouse=True)
def _live_execution(monkeypatch):
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "0")
    monkeypatch.setenv("STRATEGY_GRADUATION_ENFORCED", "1")


def _exit(bot: TradingBot, swapper, *, price: float):
    directive = TradeDirective(
        action="exit", symbol=SYMBOL, base_token="AERO", quote_token="USDC",
        size=1.546280953235675, target_price=price, horizon="atf",
        confidence=0.9, expected_return=0.0, reason="target_reached",
        strategy_id="atf_static",
    )
    sample = {"symbol": SYMBOL, "price": price, "ts": time.time(),
              "chain": "base", "volume": 100.0}
    summary = {"exit_conf": 0.9, "direction_prob": 0.2, "delta": -0.02,
               "net_margin": -0.02, "net_pnl": -0.02}

    async def _no_sync(self, *args, **kwargs):
        return None

    with mock.patch.object(
        TradingBot, "_resolve_live_trade_asset",
        lambda self, chain, sym, explicit=None: (
            sym, explicit or (AERO if str(sym).upper() == "AERO" else USDC)
        ),
    ), mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync), mock.patch.object(
        TradingBot, "_new_swapper", lambda self: swapper
    ):
        return asyncio.run(
            bot._interpret_predictions(
                None, sample, directive, pred_summary=summary, brain_summary={},
            )
        )


def test_the_flipped_fixture_really_is_the_sale_of_what_we_bought():
    """Guard the fixture itself: a test built on a bad fixture proves nothing."""
    fill = read_fill(_sell_receipt(), wallet=WALLET, sell_token=AERO,
                     buy_token=USDC, sell_decimals=18, buy_decimals=6)
    assert fill.ok, fill.reason
    assert fill.sold == pytest.approx(1.546280953235675, rel=1e-12)
    assert fill.bought == pytest.approx(0.75, abs=1e-9)


def test_a_settled_exit_is_not_reported_as_no_fill():
    quantities = {"USDC": 5.477334, "AERO": 1.546280953235675}
    bot = _bot(quantities)
    bot.positions[SYMBOL] = _live_position()

    decision = _exit(bot, _ExitSwapper(quantities), price=0.4850347528569017)

    assert decision.get("reason") != "no_fill_detected"
    assert decision["status"] not in ("live-exit-failed", "live-exit-blocked"), decision
    assert decision["tx_hash"] == EXIT_HASH
    assert decision["fill_source"] == "tx_receipt"


def test_the_sibling_bots_purchase_does_not_become_our_loss():
    """The USDC balance rose by only 0.35; we received 0.75 and must say so.

    Measured by delta this round trip books roughly -0.40 on a 0.75 position:
    a >50% loss invented entirely by another bot's unrelated purchase, written
    into the record that decides whether this strategy keeps trading.
    """
    quantities = {"USDC": 5.477334, "AERO": 1.546280953235675}
    bot = _bot(quantities)
    bot.positions[SYMBOL] = _live_position()

    decision = _exit(bot, _ExitSwapper(quantities), price=0.4850347528569017)

    assert quantities["USDC"] == pytest.approx(5.827334, abs=1e-9)  # +0.35 really
    # Sold in, quote out, both from the receipt.
    assert decision["size"] == pytest.approx(1.546280953235675, rel=1e-9)
    # Bought for 0.75 and sold for 0.75: a flat round trip minus gas, not -0.40.
    outcome = bot.db.outcomes[-1]
    assert outcome["wallet"] == "live"
    assert outcome["quantity"] == pytest.approx(1.546280953235675, rel=1e-9)
    assert outcome["gross_profit"] == pytest.approx(0.0, abs=1e-9)
    # The only cost of a flat round trip is the gas the receipt reported.
    assert outcome["fee_cost"] == pytest.approx(8.5771473e-07, rel=1e-3)
    assert outcome["net_profit"] == pytest.approx(-8.5771473e-07, rel=1e-3)


def test_a_losing_round_trip_is_recorded_as_a_loss():
    """Never hide a loss. Gas alone makes this round trip negative."""
    quantities = {"USDC": 5.477334, "AERO": 1.546280953235675}
    bot = _bot(quantities)
    bot.positions[SYMBOL] = _live_position()

    _exit(bot, _ExitSwapper(quantities), price=0.4850347528569017)

    outcome = bot.db.outcomes[-1]
    assert outcome["net_profit"] < 0.0, f"a flat sale after gas must not book a gain: {outcome}"
    assert outcome["status"] == "closed"


def test_the_position_is_closed_so_the_tokens_are_not_stranded_twice():
    quantities = {"USDC": 5.477334, "AERO": 1.546280953235675}
    bot = _bot(quantities)
    bot.positions[SYMBOL] = _live_position()

    _exit(bot, _ExitSwapper(quantities), price=0.4850347528569017)

    remaining = bot.positions.get(SYMBOL)
    assert remaining is None or float(remaining.get("size", 0.0)) <= 1e-9, remaining


def test_an_unreadable_exit_receipt_falls_back_instead_of_raising():
    """The sale has already settled; a bookkeeping gap must not throw."""
    quantities = {"USDC": 5.477334, "AERO": 1.546280953235675}
    bot = _bot(quantities)
    bot.positions[SYMBOL] = _live_position()

    class _Blind(_ExitSwapper):
        def read_fill(self, *args, **kwargs):
            return ReceiptFill(ok=False, reason="no_receipt")

    decision = _exit(bot, _Blind(quantities), price=0.4850347528569017)

    assert decision["fill_source"] == "wallet_delta"
    assert isinstance(decision.get("status"), str)


def test_a_raising_exit_fill_reader_does_not_escape_the_exit_path():
    quantities = {"USDC": 5.477334, "AERO": 1.546280953235675}
    bot = _bot(quantities)
    bot.positions[SYMBOL] = _live_position()

    class _Exploding(_ExitSwapper):
        def read_fill(self, *args, **kwargs):
            raise RuntimeError("rpc exploded")

    decision = _exit(bot, _Exploding(quantities), price=0.4850347528569017)

    assert decision["fill_source"] == "wallet_delta"
