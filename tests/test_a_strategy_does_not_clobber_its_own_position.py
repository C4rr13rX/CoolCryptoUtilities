"""A strategy re-signalling the symbol it already holds must not restart it.

Link 5 (GRADUATION) failed with "no strategy approved for live" because no
strategy could reach the 20 ghost trades promotion requires. It was not that
the bot traded too little. Measured 2026-09-03 over 6h of ``trading_ops``:

    554  enter / ghost-entry
     29  exit  / ghost-exit
    523  hold  / position-released

    499 of the 523 releases (95.4%) were a strategy abandoning ITS OWN
    position on the SAME symbol -- rsi_reversal@12h 140 times, atf_static 68,
    donchian_breakout@5h 69, stochastic_reversal@1d 58.

    median hold before abandonment  20.0s
    p90                            167.2s

``StrategyLedger.record()`` is only ever called from the exit path, so a
position abandoned 20 seconds after it opened credits nobody. ~95% of every
ghost trade the bot opened was destroyed before a take-profit, a stop or a
timed exit could resolve it, and the ledger recorded well under one outcome an
hour against ~90 entries. money_button has 1 ghost trade in its lifetime and
needs 20; atf_static had 3.

The strategies that emit entry directives are precisely the ones that like a
symbol, so they re-emit for the symbol they are already in -- the churn is the
common case, not a corner.

The rule pinned here:

  * a strategy's entry directive for a symbol it already holds is REFUSED,
    logged as ``entry-refused-duplicate``, and the sample falls through to the
    held-position branch so the bracket, the target and the timed exit are all
    evaluated on it (the same lesson as the swallowed BSTONK stop);
  * EXCEPT when the entry is live and the held position is ghost. That is a
    real state change -- the ghost->live upgrade -- and it still displaces,
    because refusing it means a graduated strategy can never take the live
    entry for any symbol its own ghost lane happens to be holding, which is
    link 6 and was 7 of the 9 live-capable symbols on 2026-09-02.
  * a DIFFERENT strategy releases as before WHEN THE ENTRY IS LIVE. The
    ghost-over-ghost case was the whole of the remaining leak once this fix
    landed -- 67 cross-strategy evictions in the 24h to 2026-09-04, against
    zero same-strategy ones in the last 8h of it -- and it is now refused too;
    see tests/test_a_strategy_does_not_clobber_another_strategys_position.py.
    The live-over-ghost path is unchanged and stays pinned here and by
    tests/test_entry_never_clobbers_a_position.py.
"""

from __future__ import annotations

import asyncio
import time
from unittest import mock

import pytest

from services.fill_receipt import ReceiptFill
from trading.bot import TradingBot
from trading.scheduler import TradeDirective

SYMBOL = "BASECAT-USDC"
BASECAT = "0xb2000000000000000000004c27f6523082f41d01"


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
        # The real MarketDB returns True when the row is inserted and False on
        # a duplicate outcome_id. A _Stub no-op returns None, which the exit
        # path reads as "already committed" and bails with duplicate-outcome --
        # a fake with a smaller surface than the class under test hiding the
        # very behaviour being asserted.
        self.outcomes.append(kwargs)
        return True

    def get_pair_adjustment(self, symbol):
        return {}

    def fetch_market_samples_for(self, *args, **kwargs):
        return []

    def fetch_trade_fills(self, *args, **kwargs):
        return []


class _ApprovedLedger(_Stub):
    def is_live_approved(self, strategy_id):
        return True


class _GhostOnlyLedger(_Stub):
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


class _SwapOutcome:
    ok = True
    broadcast = True
    confirmed = True
    tx_hash = "0xnewhash"
    route = "uniswap"


def _bot() -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    TradingBot._ensure_runtime_state(bot)
    bot.db = _DB()
    bot.strategy_ledger = _ApprovedLedger()
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


def _settling_bot():
    """A bot whose swap moves real balances, so the live branch completes."""
    bot = _bot()
    bot._bridge = _Stub()
    quantities = {"USDC": 1000.0, "BASECAT": 0.0}

    class _P(_Stub):
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

        def read_fill(self, chain, txh, *, sell, buy, wallet=None, **kwargs):
            return ReceiptFill(
                ok=True, reason="", status=True,
                sold_raw=3_000_000, bought_raw=75 * 10**18,
                sold=3.0, bought=75.0, gas_native=0.000001,
            )

    bot.portfolio = _P()
    return bot, _Swapper()


@pytest.fixture(autouse=True)
def _live_execution(monkeypatch):
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "0")
    monkeypatch.setenv("STRATEGY_GRADUATION_ENFORCED", "1")


def _directive(strategy_id: str, size: float = 10.0) -> TradeDirective:
    return TradeDirective(
        action="enter", symbol=SYMBOL, base_token="BASECAT", quote_token="USDC",
        size=size, target_price=1.05, horizon="atf", confidence=0.9,
        expected_return=0.05, reason="test", strategy_id=strategy_id,
    )


def _position(strategy_id: str, *, mode: str, price: float = 0.9,
              held_secs: float = 20.0) -> dict:
    """A position 20 seconds old -- the measured median at abandonment."""
    return {
        "mode": mode,
        "strategy_id": strategy_id,
        "size": 100.0,
        "entry_price": price,
        "target_price": 1.2,
        "entry_ts": time.time() - held_secs,
        "ts": time.time() - held_secs,
        "trade_id": f"{mode}-original-trade-id",
        "exit_sequence": 0,
        "entry_confidence": 0.9,
        "tx_hash": "0xrealhash" if mode == "live" else "",
        "base_token_address": BASECAT,
        "total_quote_spent": 3.0,
    }


def _enter(bot: TradingBot, directive: TradeDirective, *, swapper=None,
           price: float = 1.0):
    sample = {"symbol": SYMBOL, "price": price, "ts": time.time(),
              "chain": "base", "volume": 100.0}
    summary = {"exit_conf": 0.9, "direction_prob": 0.9, "delta": 0.05,
               "net_margin": 0.05, "net_pnl": 0.05}

    async def _no_sync(self, *args, **kwargs):
        return None

    with mock.patch.object(
        TradingBot, "_resolve_live_trade_asset",
        lambda self, chain, sym, explicit=None: (sym, explicit or BASECAT),
    ), mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync), (
        mock.patch.object(TradingBot, "_new_swapper", lambda self: swapper)
        if swapper is not None else mock.MagicMock()
    ):
        return asyncio.run(
            bot._interpret_predictions(
                None, sample, directive, pred_summary=summary, brain_summary={},
            )
        )


# --------------------------------------------------------------------------
# The 499 releases.
# --------------------------------------------------------------------------

def test_a_ghost_strategy_does_not_restart_its_own_position() -> None:
    """The position that would have been abandoned 20s in survives intact."""
    bot = _bot()
    bot.strategy_ledger = _GhostOnlyLedger()
    original = _position("rsi_reversal@12h", mode="ghost")
    bot.positions[SYMBOL] = original

    _enter(bot, _directive("rsi_reversal@12h"))

    held = bot.positions[SYMBOL]
    assert held["trade_id"] == "ghost-original-trade-id"
    assert held["size"] == 100.0
    assert held["entry_price"] == 0.9
    assert held["entry_ts"] == original["entry_ts"]
    assert not [r for r in bot.db.logged if r.get("status") == "position-released"]


def test_the_duplicate_refusal_is_recorded() -> None:
    """An unlogged refusal is indistinguishable from a lane that wanted nothing."""
    bot = _bot()
    bot.strategy_ledger = _GhostOnlyLedger()
    bot.positions[SYMBOL] = _position("rsi_reversal@12h", mode="ghost")

    _enter(bot, _directive("rsi_reversal@12h"))

    rows = [r for r in bot.db.logged if r.get("status") == "entry-refused-duplicate"]
    assert len(rows) == 1
    row = rows[0]
    assert row["action"] == "hold"
    # Must not be countable as an executed live trade: live_path_check.py
    # matches 'live-entry'/'live-exit' exactly.
    assert row["status"] not in {"live-entry", "live-exit"}
    details = row["details"]
    assert details["strategy_id"] == "rsi_reversal@12h"
    assert details["held_trade_id"] == "ghost-original-trade-id"
    assert details["held_mode"] == "ghost"
    assert details["reason"] == "symbol_already_held_by_same_strategy"
    # Units: seconds, not ms, and non-negative.
    assert isinstance(details["held_secs"], float)
    assert 0.0 <= details["held_secs"] < 120.0


def test_a_refused_duplicate_still_reaches_the_take_profit() -> None:
    """The sample falls through to the held-position branch, not out of it.

    This is what turns the refusal into evidence: the position that used to be
    restarted every 20s now gets its target evaluated on the very ticks that
    used to destroy it, so it can close and reach StrategyLedger.record().
    """
    bot = _bot()
    recorded: list = []
    bot.strategy_ledger = _GhostOnlyLedger()
    bot.strategy_ledger.record = (  # type: ignore[method-assign]
        lambda *a, **k: recorded.append((a, k))
    )
    bot.positions[SYMBOL] = _position("rsi_reversal@12h", mode="ghost",
                                      price=0.9, held_secs=3600.0)

    # Price well through the stored target of 1.2.
    decision = _enter(bot, _directive("rsi_reversal@12h"), price=1.4)

    assert decision["action"] == "exit", decision
    assert SYMBOL not in bot.positions

    # The outcome the abandoned position could never produce.
    assert bot.db.outcomes, "the closed trade must reach trade_outcomes"
    outcome = bot.db.outcomes[-1]
    assert outcome["details"]["strategy_id"] == "rsi_reversal@12h"
    assert outcome["details"]["mode"] == "ghost"
    assert outcome["wallet"] == "ghost"
    # ...and the ledger row that graduation actually counts.
    assert recorded, "the closed trade must reach StrategyLedger.record"
    assert recorded[-1][0][0] == "rsi_reversal@12h"
    assert recorded[-1][1]["mode"] == "ghost"
    assert recorded[-1][1]["symbol"] == SYMBOL


def test_a_live_position_is_not_restarted_by_its_own_strategy() -> None:
    """Live-over-live on one symbol would strand the first fill's tokens."""
    bot, swapper = _settling_bot()
    bot.positions[SYMBOL] = _position("atf_static", mode="live")

    _enter(bot, _directive("atf_static"), swapper=swapper)

    held = bot.positions[SYMBOL]
    assert held["mode"] == "live"
    assert held["tx_hash"] == "0xrealhash"
    assert held["trade_id"] == "live-original-trade-id"
    assert not [r for r in bot.db.logged if r.get("status") == "position-released"]


# --------------------------------------------------------------------------
# What must NOT have changed.
# --------------------------------------------------------------------------

def test_the_ghost_to_live_upgrade_still_displaces() -> None:
    """A graduated strategy must still be able to take its own ghost slot live."""
    bot, swapper = _settling_bot()
    bot.positions[SYMBOL] = _position("atf_static", mode="ghost")

    decision = _enter(bot, _directive("atf_static"), swapper=swapper)

    assert decision["status"] == "live-entry", decision
    assert bot.positions[SYMBOL]["mode"] == "live"
    released = [r for r in bot.db.logged if r.get("status") == "position-released"]
    assert len(released) == 1
    assert released[0]["details"]["incoming_mode"] == "live"


def test_a_different_strategy_still_releases() -> None:
    """Cross-strategy collisions were 24 of 523 and keep the old behaviour."""
    bot, swapper = _settling_bot()
    bot.positions[SYMBOL] = _position("donchian_breakout@5d", mode="ghost")

    decision = _enter(bot, _directive("atf_static"), swapper=swapper)

    assert decision["status"] == "live-entry"
    released = [r for r in bot.db.logged if r.get("status") == "position-released"]
    assert len(released) == 1
    assert released[0]["details"]["released_strategy_id"] == "donchian_breakout@5d"
    assert not [
        r for r in bot.db.logged if r.get("status") == "entry-refused-duplicate"
    ]


def test_an_unattributed_position_is_not_treated_as_a_duplicate() -> None:
    """A position with no strategy_id has unknown ownership; do not guess."""
    bot, swapper = _settling_bot()
    pos = _position("", mode="ghost")
    bot.positions[SYMBOL] = pos

    decision = _enter(bot, _directive("atf_static"), swapper=swapper)

    assert decision["status"] == "live-entry"
    assert not [
        r for r in bot.db.logged if r.get("status") == "entry-refused-duplicate"
    ]


def test_an_entry_on_a_free_symbol_is_untouched() -> None:
    bot, swapper = _settling_bot()
    assert SYMBOL not in bot.positions

    decision = _enter(bot, _directive("atf_static"), swapper=swapper)

    assert decision["status"] == "live-entry"
    assert not [
        r for r in bot.db.logged if r.get("status") == "entry-refused-duplicate"
    ]
