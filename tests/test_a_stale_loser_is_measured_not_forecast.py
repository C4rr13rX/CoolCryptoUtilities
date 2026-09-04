"""The timed exit must read the position's P/L, not the model's forecast.

The held-position branch of ``_interpret_predictions`` documents its own order
of checks::

    #   1. take-profit  - the whole point of buy-low/sell-high
    #   2. stop-loss    - bounded downside
    #   3. model gates  - only when the model expresses a real opinion
    #   4. timed exit   - stale losers release capital

Rules 1-3 all read the position: ``target_price_held``, ``pnl_pct_held``,
``entry_price_held``. Rule 4 did not. It read::

    elif pnl < 0 and held_secs > GHOST_NEG_EXIT_SECONDS:

where ``pnl`` is bound hundreds of lines earlier as::

    pnl = float(summary.get("net_pnl", margin))

-- the model's predicted net margin for the NEXT step. Both values are
unitless fractions, so the comparison never raised and never looked wrong; it
just answered a different question than "is this position a stale loser".

MEASURED 2026-09-04, 13:15-16:15, over 136 organism-snapshot decisions:

    net_pnl > 0   49
    net_pnl == 0  39     <- what a neutral OR unavailable model returns, and
    net_pnl < 0   48        TensorFlow was unavailable for most of that window

So a genuinely stale losing position got its timed exit skipped on roughly two
ticks in three, and on any symbol the model stayed non-negative about, forever.
The live book at 16:13 held two positions and both were exactly that case:

    CBXRP-USDC  held  9,304s (3.4x GHOST_NEG_EXIT_SECONDS)  realised -1.46%
    CBBTC-USDC  held 11,998s (4.4x)                         realised -0.14%

Neither is deep enough to reach the 2% stop, neither had ever produced a
``timed-exit``, and while atf_static -- the only live-approved strategy -- held
them, its every new entry on those symbols came back ``entry-refused-duplicate``
(31 on CBBTC in three hours).

The rule pinned here:

  * a position past GHOST_NEG_EXIT_SECONDS at a REALISED loss exits, whatever
    the model forecasts for the next step;
  * a position with no usable entry price does NOT (an unknown cost basis is
    not evidence of a loss -- the same blind spot the stop-loss above has, and
    for the same reason); and
  * a LIVE position is still not force-closed at a nothing-move: "timed-exit"
    is not protective, so the live-exit margin gate refuses it as
    ``hold-negative``, which is what tests/test_a_close_must_cover_its_own_gas
    exists to protect. Fixing rule 4 must not undo that.
"""

from __future__ import annotations

import asyncio
import time
from unittest import mock

import pytest

from trading.bot import TradingBot

# The two positions that paid for this test, off the 2026-09-04 16:13 snapshot.
CBXRP = "CBXRP-USDC"
CBXRP_TOKEN = "0xcb585250f852C6c6bf90434AB21A00f02833a4af"
CBXRP_ENTRY = 1.4105672172894166
CBXRP_TICK = 1.39                 # realised -1.458%, inside the 2% stop
CBXRP_SIZE = 0.531701
CBXRP_HELD_SEC = 9304.0

CBBTC = "CBBTC-USDC"
CBBTC_TOKEN = "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf"
CBBTC_ENTRY = 79910.09593421653
CBBTC_TICK = 79799.78864298711    # realised -0.138%
CBBTC_SIZE = 2.189e-05
CBBTC_HELD_SEC = 11998.0

# Long past the 2700s default, so the clock is never what decides these tests.
NEG_EXIT_SEC = 2700.0


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

    def get_pair_adjustment(self, symbol):
        return {}

    def fetch_market_samples_for(self, *args, **kwargs):
        return []

    def fetch_trade_fills(self, *args, **kwargs):
        return []

    def fetch_trades(self, *args, **kwargs):
        return []

    def record_trade_outcome(self, *args, **kwargs):
        # Truthy = "new outcome". A _Stub's None reads as a duplicate and
        # short-circuits the exit before it books anything.
        self.outcomes.append(kwargs or {"args": args})
        return True


class _Metrics(_Stub):
    def __init__(self):
        self.events: list = []

    def feedback(self, source, severity=None, label="", details=None, **kwargs):
        self.events.append({"source": source, "label": label, "details": details or {}})
        return None


class _Ledger(_Stub):
    def is_live_approved(self, strategy_id):
        return True


class _Validator(_Stub):
    def validate(self, **kwargs):
        return True, {}, []


class _Pipeline(_Stub):
    decision_threshold = 0.58


class _Portfolio(_Stub):
    holdings: dict = {}

    def get_quantity(self, symbol, chain=None):
        return 1000.0

    def get_native_balance(self, *args, **kwargs):
        return 1.0


def _bot() -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    TradingBot._ensure_runtime_state(bot)
    bot.db = _DB()
    bot.strategy_ledger = _Ledger()
    bot.portfolio = _Portfolio()
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
    bot.auto_promote_live = False
    return bot


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # Dry run: these tests are about the DECISION, never about broadcasting.
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "1")
    monkeypatch.setenv("GHOST_NEG_EXIT_SECONDS", str(int(NEG_EXIT_SEC)))
    monkeypatch.setenv("GHOST_STOP_LOSS_PCT", "0.02")
    monkeypatch.setenv("MIN_HOLD_SECONDS", "300")
    # No take-profit target on any fixture below, so rule 1 cannot decide.


def _position(
    *,
    mode: str,
    entry_price: float,
    size: float,
    held_sec: float,
    token: str,
    strategy_id: str = "atf_static",
) -> dict:
    opened = time.time() - held_sec
    return {
        "mode": mode,
        "strategy_id": strategy_id,
        "size": size,
        "entry_price": entry_price,
        # 0.0 = no target, so the take-profit rule above cannot fire and the
        # timed exit is the only thing under test.
        "target_price": 0.0,
        "entry_ts": opened,
        "ts": opened,
        "trade_id": f"2:{mode}:stale-loser",
        "route": [token, "USDC"],
        "quote_spent": entry_price * size,
        "base_token_address": token,
    }


def _tick(bot: TradingBot, symbol: str, token: str, price: float, *, net_pnl: float):
    """One sample with the model saying `net_pnl` about the NEXT step."""
    sample = {"symbol": symbol, "price": price, "ts": time.time(),
              "chain": "base", "volume": 100.0}
    # direction_prob/exit_conf at 0.5 = model_neutral, which is what shuts off
    # rules 3a (confidence_drop) and 3b (negative_margin). Without that this
    # test would pass on the wrong rule.
    summary = {"exit_conf": 0.5, "direction_prob": 0.5, "delta": 0.0,
               "net_margin": 0.0, "net_pnl": net_pnl}

    async def _no_sync(self, *args, **kwargs):
        return None

    with mock.patch.object(
        TradingBot, "_resolve_live_trade_asset",
        lambda self, chain, sym, explicit=None: (sym, explicit or token),
    ), mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync):
        return asyncio.run(
            bot._interpret_predictions(
                # No directive: the sample goes straight to the held-position
                # branch, so nothing upstream can decide the outcome for us.
                None, sample, None, pred_summary=summary, brain_summary={},
            )
        )


# --------------------------------------------------------------------------
# The bug: the rule consulted the forecast and the forecast was not negative.
# --------------------------------------------------------------------------

def test_a_stale_loser_exits_when_the_model_forecast_is_zero() -> None:
    """net_pnl == 0.0 is the unavailable-model reading, and it froze the rule."""
    bot = _bot()
    bot.positions[CBXRP] = _position(
        mode="ghost", entry_price=CBXRP_ENTRY, size=CBXRP_SIZE,
        held_sec=CBXRP_HELD_SEC, token=CBXRP_TOKEN,
    )

    decision = _tick(bot, CBXRP, CBXRP_TOKEN, CBXRP_TICK, net_pnl=0.0)

    assert decision["action"] == "exit", (
        "a position held 3.4x GHOST_NEG_EXIT_SECONDS at a realised -1.46% was "
        "left open because the model forecast 0.0 for the next step"
    )
    assert decision.get("exit_reason") == "timed-exit", decision


def test_a_stale_loser_exits_when_the_model_forecast_is_positive() -> None:
    """The forecast may be bullish; the position is still a realised loser."""
    bot = _bot()
    bot.positions[CBXRP] = _position(
        mode="ghost", entry_price=CBXRP_ENTRY, size=CBXRP_SIZE,
        held_sec=CBXRP_HELD_SEC, token=CBXRP_TOKEN,
    )

    decision = _tick(bot, CBXRP, CBXRP_TOKEN, CBXRP_TICK, net_pnl=0.05)

    assert decision["action"] == "exit", decision
    assert decision.get("exit_reason") == "timed-exit", decision


def test_a_stale_WINNER_is_not_closed_by_the_timed_exit() -> None:
    """The rule is for losers. A position in profit is left to its target."""
    bot = _bot()
    bot.positions[CBXRP] = _position(
        mode="ghost", entry_price=CBXRP_ENTRY, size=CBXRP_SIZE,
        held_sec=CBXRP_HELD_SEC, token=CBXRP_TOKEN,
    )

    # Up 2%, old, and the model is bearish -- the forecast alone used to close
    # this and book a winner under the stale-loser name.
    decision = _tick(bot, CBXRP, CBXRP_TOKEN, CBXRP_ENTRY * 1.02, net_pnl=-0.05)

    assert decision.get("exit_reason") != "timed-exit", decision


def test_a_young_loser_is_not_closed_by_the_timed_exit() -> None:
    """The clock still has to run out. Same losing price, fresh position."""
    bot = _bot()
    bot.positions[CBXRP] = _position(
        mode="ghost", entry_price=CBXRP_ENTRY, size=CBXRP_SIZE,
        held_sec=NEG_EXIT_SEC - 600.0, token=CBXRP_TOKEN,
    )

    decision = _tick(bot, CBXRP, CBXRP_TOKEN, CBXRP_TICK, net_pnl=0.0)

    assert decision.get("exit_reason") != "timed-exit", decision


def test_an_unknown_cost_basis_is_not_treated_as_a_loss() -> None:
    """entry_price 0 reads pnl_pct_held == 0.0 -- not evidence of a loss.

    The stop-loss above has the same blind spot for the same reason. These
    positions are released by the max-hold eviction, not sold on a number
    nobody can compute.
    """
    bot = _bot()
    pos = _position(
        mode="ghost", entry_price=CBXRP_ENTRY, size=CBXRP_SIZE,
        held_sec=CBXRP_HELD_SEC, token=CBXRP_TOKEN,
    )
    pos["entry_price"] = 0.0
    bot.positions[CBXRP] = pos

    decision = _tick(bot, CBXRP, CBXRP_TOKEN, CBXRP_TICK, net_pnl=0.0)

    assert decision.get("exit_reason") != "timed-exit", decision


# --------------------------------------------------------------------------
# ...and the thing this fix must not break.
# --------------------------------------------------------------------------

def test_the_timed_exit_does_not_force_a_live_close_at_a_nothing_move() -> None:
    """A live stale loser reaches the margin gate and is held, not sold.

    CBBTC-USDC: $1.749 notional, realised -0.138%, held 11,998s. Closing it
    costs 0.65% of notional plus $0.0043 of round-trip gas -- est_net_profit
    -$0.0189 -- so the gate refuses. "timed-exit" is deliberately NOT in the
    protective set, so widening rule 4 must not reintroduce the forced closes
    that cost 0.43%-1.03% of notional on 2026-09-03.
    """
    bot = _bot()
    bot.positions[CBBTC] = _position(
        mode="live", entry_price=CBBTC_ENTRY, size=CBBTC_SIZE,
        held_sec=CBBTC_HELD_SEC, token=CBBTC_TOKEN,
    )

    decision = _tick(bot, CBBTC, CBBTC_TOKEN, CBBTC_TICK, net_pnl=0.0)

    assert decision["status"] == "hold-negative", decision
    assert CBBTC in bot.positions, "the live position must survive the refusal"
    cost = decision.get("exit_cost") or {}
    assert cost.get("est_net_profit_usd", 0.0) < 0.0, cost
