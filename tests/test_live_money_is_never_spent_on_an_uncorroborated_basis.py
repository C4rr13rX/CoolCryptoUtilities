"""The LIVE lane refuses an entry basis nothing corroborates, BEFORE the swap.

Item [781bf37c], following [d763940a]. The ghost entry site has refused a basis
the symbol's own book disputes since 3099104 -- it catches the AERO 1.140000 row
that inherited a fake +161% exit as its cost basis. The live lane had no such
refusal anywhere, and it could not be given one in the same place: the live
basis is booked from a SETTLED receipt, so refusing THERE strands a position
whose money has already left the wallet (disarming-stranded-the-live-position).

So the live reading is taken on the price the STRATEGY IS JUDGING, upstream of
the swap, with ``strict=True`` -- the reading services.entry_price_corroboration
documents for real money: a price nothing has confirmed is refused rather than
allowed. These tests drive ``_interpret_predictions`` end to end against a real
sqlite feed, so they prove the wiring and the threshold together:

1. A corroborated basis still reaches ``swapper.swap``. A gate that blocks
   everything is switched off, not safe, and that is the failure this class of
   change makes most often.
2. An uncorroborated basis never reaches it -- ``_new_swapper`` is a tripwire
   that fails the test if it is called at all.
3. The refusal DOWNGRADES to ghost rather than discarding the observation, so a
   refused live entry costs a live opportunity and never a measurement.
4. The settled-receipt booking site is still unguarded, so a refused basis can
   never strand money that has already left the wallet.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from unittest import mock

import pytest

from trading.bot import TradingBot
from trading.scheduler import TradeDirective

SYMBOL = "CBBTC-USDC"
CBBTC = "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf"

# The price the feed actually publishes for this symbol in the fixture below.
FEED_PRICE = 80818.96551724139


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


class _SwapReached(RuntimeError):
    """Raised by the stand-in swapper. Its presence means the swap was tried."""


class _Swapper:
    def swap(self, **kwargs):
        raise _SwapReached("the swap was submitted")


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
    bot._bridge = object()
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
def _live_execution(monkeypatch, tmp_path):
    """Live approved, dry run off, and a feed that publishes ONE symbol.

    The database is real sqlite with the two tables the guard reads, so the
    threshold and the SQL are exercised rather than mocked away. CBBTC-USDC is
    corroborated near FEED_PRICE; every other symbol has no coverage at all,
    which is the unjudgeable case ``strict=True`` exists to refuse.
    """
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "0")
    monkeypatch.setenv("STRATEGY_GRADUATION_ENFORCED", "1")

    db = tmp_path / "trading_cache.db"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE market_stream (symbol TEXT, ts REAL, price REAL)")
    con.execute("CREATE TABLE trade_outcomes (symbol TEXT, ts REAL, entry_price REAL)")
    now = time.time()
    con.executemany(
        "INSERT INTO market_stream (symbol, ts, price) VALUES (?, ?, ?)",
        [(SYMBOL, now - 60.0 * i, FEED_PRICE * (1.0 + 0.001 * i)) for i in range(1, 21)],
    )
    con.commit()
    con.close()

    import services.entry_price_corroboration as epc
    import trading.bot as bot_mod

    monkeypatch.setattr(epc, "DEFAULT_DB", db)
    # An unrelated gate, and it fires first: the symbol edge gate reads the
    # REAL trading_cache.db and currently bans every symbol this harness could
    # name. Silencing it here isolates the basis guard; it has its own tests.
    monkeypatch.setattr(bot_mod, "symbol_edge_refusal", lambda *a, **k: "")
    monkeypatch.setattr(bot_mod, "symbol_motion_refusal", lambda *a, **k: "")
    return db


def _directive(symbol: str = SYMBOL) -> TradeDirective:
    return TradeDirective(
        action="enter",
        symbol=symbol,
        base_token=symbol.split("-")[0],
        quote_token="USDC",
        size=9.2e-06,
        target_price=FEED_PRICE * 1.06,
        horizon="atf",
        confidence=0.9,
        expected_return=0.05,
        reason="test",
        strategy_id="atf_static",
    )


def _enter(bot: TradingBot, directive: TradeDirective, *, price: float):
    """Run one entry decision. ``_new_swapper`` is the tripwire."""
    async def _no_sync(self, *args, **kwargs):
        return None

    calls: list = []

    def _swapper(self, *args, **kwargs):
        calls.append(True)
        return _Swapper()

    sample = {
        "symbol": directive.symbol,
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
    ), mock.patch.object(TradingBot, "_new_swapper", _swapper):
        decision = asyncio.run(
            bot._interpret_predictions(
                None, sample, directive, pred_summary=summary, brain_summary={}
            )
        )
    return decision, calls


def test_a_corroborated_basis_still_reaches_the_swap():
    """The control. Without this the other tests pass on a dead lane."""
    bot = _bot()

    decision, calls = _enter(bot, _directive(), price=FEED_PRICE)

    assert calls, (
        "the live entry never reached the swapper even though the feed "
        "corroborates this price -- the gate is refusing everything, which is "
        "the same as being switched off"
    )
    assert "swap_error" in str(decision.get("reason") or ""), decision


def test_a_basis_the_feed_never_published_never_reaches_the_swap():
    """The AERO 1.140000 shape, priced in real money.

    2.6x the price the feed publishes, with twenty ticks of coverage in the
    window saying otherwise. No swap may be submitted on it.
    """
    bot = _bot()

    decision, calls = _enter(bot, _directive(), price=FEED_PRICE * 2.6)

    assert not calls, (
        "a live swap was submitted on a basis no feed tick corroborates -- "
        "this is real money spent at a price the feed never published"
    )
    assert decision.get("status") != "live-entry"
    assert decision.get("wallet") != "live"


def test_a_symbol_the_feed_has_not_reached_never_reaches_the_swap():
    """strict=True is the whole point: unjudgeable is refused for real money.

    The ghost lane allows an unjudgeable basis on purpose -- the feed reaches
    new symbols late and refusing there would stop the harness gathering
    evidence. The live lane must not inherit that leniency.
    """
    bot = _bot()

    decision, calls = _enter(bot, _directive("AERO-USDC"), price=1.14)

    assert not calls, (
        "a live swap was submitted on a symbol the feed has never published a "
        "tick for and the book holds no prior entry in"
    )
    assert decision.get("status") != "live-entry"


def test_the_refusal_is_logged_as_a_live_row_and_names_its_rule():
    """A decision about real money must be findable in a wallet='live' query."""
    bot = _bot()

    _enter(bot, _directive(), price=FEED_PRICE * 2.6)

    rows = [
        r for r in bot.db.logged
        if str(r.get("details", {}).get("reason") or "") == "uncorroborated_entry_basis"
    ]
    assert rows, "the live refusal was not logged at all: %s" % (bot.db.statuses(),)
    assert rows[-1]["wallet"] == "live"
    assert rows[-1]["status"] == "live-entry-blocked"
    assert rows[-1]["action"] == "hold"


def test_the_refused_entry_downgrades_to_ghost_rather_than_vanishing():
    """The observation survives the refusal.

    Discarding it would repeat the token_unresolved defect: a refusal recorded
    and the evidence that decides graduation thrown away with it. The ghost
    lane judges the same price under its own, lenient reading.
    """
    bot = _bot()

    decision, calls = _enter(bot, _directive("AERO-USDC"), price=1.14)

    assert not calls
    pos = bot.positions.get("AERO-USDC")
    assert pos is not None, (
        "the refused live entry took the observation with it: %r" % (decision,)
    )
    assert pos["mode"] == "ghost"
    assert pos["entry_price"] == pytest.approx(1.14)


def test_the_guard_runs_before_any_swap_and_not_on_the_settled_receipt():
    """Position in the source, which is the safety argument.

    The check must sit above ``swapper.swap`` and must NOT appear between the
    receipt being read and the position being booked -- refusing there strands
    a position whose money has already left the wallet.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "trading" / "bot.py").read_text(
        encoding="utf-8", errors="replace"
    )

    # Anchored on the ENTRY swap, not on ``_new_swapper`` -- the exit path
    # builds one of those too, hundreds of lines earlier.
    guard = src.index("entry_price_is_corroborated")
    swap = src.index('purpose="live_entry"')
    assert guard < swap, "the live basis check runs after the entry swap"

    booking = src.index("executed_entry_price = quote_spent / max(base_received, 1e-9)")
    ghost = src.index("# ghost / paper entry")
    settled = src[booking:ghost]
    assert "entry_price_is_corroborated" not in settled, (
        "the settled-receipt booking site is guarded -- a refusal there strands "
        "a position whose money has already left the wallet"
    )
    assert "book_disagreement" not in settled


def test_a_failure_inside_the_live_guard_does_not_close_the_lane(monkeypatch):
    """A gate that blocks everything is a bug, not safety.

    If the database cannot be read, the price is unjudgeable, not contaminated,
    and the live lane must keep working.
    """
    import services.entry_price_corroboration as epc

    def _boom(*args, **kwargs):
        raise RuntimeError("database is gone")

    monkeypatch.setattr(epc, "entry_price_is_corroborated", _boom)

    bot = _bot()
    decision, calls = _enter(bot, _directive(), price=FEED_PRICE)

    assert calls, (
        "a broken corroboration read closed the live lane: %r" % (decision,)
    )
