"""A position holding real tokens must never be closed by simulation.

``_interpret_predictions`` decides how to close a position with::

    pos_mode    = pos["mode"]                                    # how it OPENED
    pos_is_live = pos_mode == "live" and self.live_trading_enabled

The ghost direction of that rule was enforced: a ghost-opened position never
routes through the live swap path even when the bot has gone live. The LIVE
direction was not. The moment the bot-level ``live_trading_enabled`` flag went
false -- and it flaps, because ``_refresh_auto_execute`` recomputes it from
graduation and readiness -- every position holding real tokens fell through to
the simulated branch and was marked out against the feed price. No swap, no tx
hash, no proceeds. The fiction was then booked as a completed trade.

And booked in the LIVE ledger. The database row is written with
``wallet = "live" if pos_is_live else "ghost"`` while ``StrategyLedger.record``
is called with ``mode=pos_mode``, so one simulated exit writes a row saying
ghost and a ledger entry saying live.

Measured against the chain on 2026-09-03 -- eth_getLogs over every ERC-20
Transfer touching 0x291c854811e92906a658Fb94Aa511bF919f968ad for the day --
four of atf_static's seven "live" outcomes have no settling transfer at all:

    17:40:03  CBETH   -0.000005   sold 0.00000023   no transfer
    19:12:15  BSTONK  -0.142865   sold 360.264243   no transfer; the wallet
                                  still holds all 360.2642432254
    20:03:27  CBBTC   +0.000267   sold 0.00000928   no transfer
    20:38:53  CBETH   -0.003011   sold 0.00015239   no transfer, and the wallet
                                  had held only 0.0000003733 since 16:46 -- it
                                  sold tokens that did not exist

The BSTONK line is -0.142865 against +0.011281 of wins across the whole live
book: 102% of the live P/L that demoted atf_static, the only strategy that has
ever spent real money here, and the source of the tail_risk 0.1429 and
profit_factor 0.152 the live gate refuses on. The position it "stopped out" is
still open on chain.

The rule pinned here: when a live-opened position cannot be closed on chain,
the exit is REFUSED and the position survives. An unclosed position is visible
in the book; a fictional close is not.

AMENDED 2026-09-04. The rule above is right; the implementation that carried it
was not. ``pos_is_live`` answered "can this close on chain?" with "is the bot
armed?", and those are different questions -- a disarmed bot with a signing
bridge and a resolved token can close on chain perfectly well. Because the bot
never re-arms itself once demoted, the refusal became permanent: atf_static
bought CBETH live at 00:17:32
(0x4ca1a606eb33ef24df951f15177532a2d9803554082ddb4c8677cc5d9bbc7e2d, settled on
base), its live record then flipped to ``live_approved: false`` /
``halt_live: 1.0``, and from 00:55:31 to 01:07:05 the exit was refused eighteen
times running with ``live_position_cannot_exit_in_simulation`` while the tokens
sat in the wallet. A position that cannot be closed is not a held trade, it is
a donation.

So ``pos_is_live`` is now ``pos_mode == "live"`` alone, and a disarmed bot takes
the same live swap path an armed one does -- disarming stops TAKING risk, not
shedding it. Entries are unaffected: they gate on ``entry_is_live`` /
``entry_spends_real_money``, both still
``live_trading_enabled AND _strategy_live_approved(directive)``.

Every safety assertion below is unchanged. What changed is where a disarmed bot
STOPS: at the live swap, not at a mark-out. When that swap cannot happen the
outcome is the same as before -- nothing booked, position survives.
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
LIVE_SIZE = 100.0


class _Stub:
    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


class _DB(_Stub):
    def __init__(self):
        self.outcomes: list = []

    def log_trade(self, **kwargs):
        return True

    def get_pair_adjustment(self, symbol):
        return {}

    def fetch_market_samples_for(self, *args, **kwargs):
        return []

    def fetch_trade_fills(self, *args, **kwargs):
        return []

    def record_trade_outcome(self, *args, **kwargs):
        self.outcomes.append(kwargs or {"args": args})
        return True


class _Metrics(_Stub):
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
    class _Portfolio(_Stub):
        holdings: dict = {}

        def get_quantity(self, symbol, chain=None):
            if str(symbol).upper() == "BASECAT":
                return base_quantity
            return 1000.0

        def get_native_balance(self, *args, **kwargs):
            return 1.0

    return _Portfolio()


def _bot(*, live_trading_enabled: bool, wallet_base: float = LIVE_SIZE) -> TradingBot:
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
    bot.sim_quote_balances = {("base", "USDC"): 100.0}
    bot.stable_tokens = {"USDC"}
    bot.primary_chain = "base"
    bot.live_trading_enabled = live_trading_enabled
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
    bot.auto_promote_live = False
    return bot


def _position(mode: str, size: float = LIVE_SIZE) -> dict:
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


def _tick(bot: TradingBot, price: float, directive: TradeDirective | None = None):
    sample = {"symbol": SYMBOL, "price": price, "ts": time.time(),
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
    monkeypatch.delenv("MAX_POSITION_PRICE_RATIO", raising=False)


# --------------------------------------------------------------------------
# The bug, in both directions a mark-out can point.
# --------------------------------------------------------------------------

def test_a_live_position_is_not_marked_out_at_a_loss() -> None:
    """The BSTONK case: a stop-loss with no sale behind it.

    Priced below entry and past the stop, this is exactly the path that booked
    -0.142865 against a position the wallet still holds in full.
    """
    bot = _bot(live_trading_enabled=False)
    bot.positions[SYMBOL] = _position("live")

    decision = _tick(bot, price=ENTRY_PRICE * 0.80)

    assert decision["status"] == "live-exit-blocked"
    assert decision["executed"] is False
    # It stopped at the SWAP, not at a mark-out. `bridge_unavailable` is the
    # live path failing to sign in a test process -- past this guard entirely.
    assert decision["reason"] != "live_position_cannot_exit_in_simulation"
    # Nothing was booked anywhere.
    assert bot.db.outcomes == [], "a simulated close of a live position wrote an outcome"
    assert bot.strategy_ledger.recorded == [], "a fiction reached the ledger"
    assert bot.total_trades == 0
    # The position survives -- it is the only record that the tokens are held.
    assert bot.positions[SYMBOL]["size"] == pytest.approx(LIVE_SIZE)
    assert bot.metrics.by_label("live_exit_would_be_simulated") == []


def test_a_live_position_is_not_marked_out_at_a_profit_either() -> None:
    """The CBBTC case: +0.000267 booked with no sale behind it.

    A fictional WIN is the more dangerous direction, because it graduates.
    """
    bot = _bot(live_trading_enabled=False)
    bot.positions[SYMBOL] = _position("live")

    decision = _tick(bot, price=TARGET_PRICE * 1.05)

    assert decision["status"] == "live-exit-blocked"
    assert decision["reason"] != "live_position_cannot_exit_in_simulation"
    assert bot.db.outcomes == []
    assert bot.strategy_ledger.recorded == []
    assert bot.positions[SYMBOL]["size"] == pytest.approx(LIVE_SIZE)


def test_a_disarmed_bot_takes_the_same_exit_path_as_an_armed_one() -> None:
    """The orphan fix, stated directly.

    Eighteen consecutive refusals on a live CBETH position between 00:55:31 and
    01:07:05 on 2026-09-04 were all the bot flag, not the chain. Disarming stops
    opening risk; it must not strand tokens already bought.
    """
    armed = _bot(live_trading_enabled=True)
    armed.positions[SYMBOL] = _position("live")
    disarmed = _bot(live_trading_enabled=False)
    disarmed.positions[SYMBOL] = _position("live")

    armed_decision = _tick(armed, price=TARGET_PRICE * 1.05)
    disarmed_decision = _tick(disarmed, price=TARGET_PRICE * 1.05)

    assert disarmed_decision["status"] == armed_decision["status"]
    assert disarmed_decision["reason"] == armed_decision["reason"]
    assert disarmed_decision["wallet"] == "live"


def test_a_disarmed_bot_still_cannot_open_with_real_money() -> None:
    """The half that must NOT move: exits re-armed, entries did not.

    ``entry_is_live`` / ``entry_spends_real_money`` still require
    ``live_trading_enabled``, so with no position held a disarmed bot's entry
    is a ghost entry -- it books no live wallet and no live ledger row.
    """
    bot = _bot(live_trading_enabled=False, wallet_base=0.0)
    assert bot.positions == {}

    directive = TradeDirective(
        action="enter",
        symbol=SYMBOL,
        base_token="BASECAT",
        quote_token="USDC",
        size=10.0,
        target_price=TARGET_PRICE,
        horizon="5m",
        confidence=0.9,
        expected_return=0.05,
        reason="test",
        strategy_id="atf_static",
        token_address=BASECAT,
    )
    _tick(bot, price=ENTRY_PRICE, directive=directive)

    live_records = [
        kwargs for _args, kwargs in bot.strategy_ledger.recorded
        if str(kwargs.get("mode", "")).lower() == "live"
    ]
    assert live_records == [], "a disarmed bot opened a LIVE position"
    for pos in bot.positions.values():
        assert pos.get("mode") != "live", "a disarmed bot booked a live position"


def test_the_refused_exit_never_reaches_the_live_ledger() -> None:
    """The specific corruption: wallet='ghost' on the row, mode='live' in the ledger.

    Whatever else happens, a simulated exit must not increment a strategy's
    LIVE book -- that book is what demotion and the live gate read.
    """
    bot = _bot(live_trading_enabled=False)
    bot.positions[SYMBOL] = _position("live")

    _tick(bot, price=ENTRY_PRICE * 0.80)

    live_records = [
        kwargs for _args, kwargs in bot.strategy_ledger.recorded
        if str(kwargs.get("mode", "")).lower() == "live"
    ]
    assert live_records == []


# --------------------------------------------------------------------------
# The halves that must NOT change.
# --------------------------------------------------------------------------

def test_a_ghost_position_still_closes_in_simulation() -> None:
    """The refusal is keyed on the POSITION's mode, not the bot's flag.

    A ghost position on a non-live bot is the ordinary case and must be
    entirely unaffected -- ghost exits are what feed graduation.
    """
    bot = _bot(live_trading_enabled=False, wallet_base=0.0)
    bot.positions[SYMBOL] = _position("ghost")

    decision = _tick(bot, price=TARGET_PRICE * 1.05)

    assert decision["status"] == "ghost-exit", decision.get("reason")
    assert decision["wallet"] == "ghost"
    assert decision["size"] == pytest.approx(LIVE_SIZE)
    assert bot.db.outcomes, "a closed ghost trade must still commit an outcome"
    assert SYMBOL not in bot.positions


def test_an_armed_bot_still_takes_the_live_swap_path() -> None:
    """With live execution armed, the live branch answers as before.

    There is no signing bridge in a test process, so it stops at
    ``bridge_unavailable`` -- which is past this guard, and is the proof that
    the guard did not swallow the armed case.
    """
    bot = _bot(live_trading_enabled=True)
    bot.positions[SYMBOL] = _position("live")

    decision = _tick(bot, price=TARGET_PRICE * 1.05)

    assert decision["status"] == "live-exit-blocked"
    assert decision["reason"] != "live_position_cannot_exit_in_simulation"
    assert bot.metrics.by_label("live_exit_would_be_simulated") == []
