"""A close costs gas, and nothing was subtracting it.

Measured on the five settled live round trips on base, 2026-09-03/04
(``trade_outcomes``, mode=live, status=closed; gas re-read from the receipts
in commit a5a3385, so these are chain numbers, not estimates)::

    AERO  $0.75 notional   gross +0.001240   gas 0.004319   net -0.003079
    CBETH $0.44 notional   gross +0.014984   gas 0.004590   net +0.010394
    CBETH $0.32 notional   gross -0.002397   gas 0.003046   net -0.005443
    AERO  $0.75 notional   gross -0.001045   gas 0.003872   net -0.004917
    CBBTC $3.00 notional   gross -0.004102   gas 0.012776   net -0.016878

    gross  +0.008680        gas  -0.028603        net  -0.019922

The direction calls were right in aggregate and the round trips still lost.
Gas ran 0.43%-1.03% of these notionals, and every cost gate in the bot charged
a variable RATE (``fees`` = 0.0065 of notional) and nothing else -- so the one
cost that does not shrink with the clip was the one nobody modelled.

Three places were wrong, and they are pinned here:

1. THE LIVE EXIT MARGIN GATE (``_interpret_predictions``) did subtract a cost
   before closing, but only ``notional * fees``, and only while
   ``held_sec < MAX_HOLD_SECONDS``. Past the hold clock the check was not
   consulted at all and the position was closed at any price. AERO was entered
   at 1788489835 and closed at 1788494584 -- 4749s, past the 3600s clock -- on
   a -0.139% move, paying 0.516% of notional in gas. A hold clock decides
   whether to LOOK for an exit; it must not decide whether an exit is worth
   what it costs.

2. THE GHOST EXIT ACCOUNTING charged ``notional * fees`` and no gas, while the
   live branch charges realized gas (the DEX fee and slippage are inside the
   fill prices there, so they land in ``gross_profit``). The two books priced
   different games and the ghost one was cheaper. Graduation reads the ghost
   book: atf_static graduated on it and returned the record above.

3. THE ENTRY GATE passed ``fixed_cost_usd=MICRO_FIXED_COST_USD``, default
   "0". Measured over 600 organism snapshots: 6222 of 6222 entries returned
   ``profitable_after_costs``, every one sized at $0.75 against a flat 5%
   target, so the gate was not binding -- but it was also not charging gas,
   and it becomes binding the moment a strategy predicts an honest number.

What is deliberately NOT changed: protective exits. ``stop_loss``,
``break_even_lock``, ``profit_lock`` and ``trailing_stop`` still fire at any
age and at any cost. Cutting a real loss must never be gated on whether
cutting it is cheap -- "a single losing exit must never exceed the sum of the
wins" depends on the stop being unconditional.
"""

from __future__ import annotations

import asyncio
import os
import time
from unittest import mock

import pytest

from trading.bot import TradingBot
from trading.micro_profit import (
    _clear_roundtrip_gas_cache,
    evaluate_micro_profit,
    roundtrip_gas_usd,
)
from trading.scheduler import TradeDirective

SYMBOL = "AERO-USDC"
AERO = "0x940181a94a35a4569e4529a3cdfb74e38fd98631"

ENTRY_PRICE = 0.5018
LIVE_SIZE = 1.4946209380006292          # the real AERO clip: $0.75 of notional
NOTIONAL = ENTRY_PRICE * LIVE_SIZE

# The median of the five realized round trips above.
MEASURED_GAS_USD = 0.004319330367028045


# --------------------------------------------------------------------------
# 1. The cost itself: measured from our own receipts, and safe when it is not
#    measurable at all.
# --------------------------------------------------------------------------

class _OutcomeDB:
    """Only what ``roundtrip_gas_usd`` reads."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = 0

    def fetch_trade_outcomes(self, *, wallet=None, limit=200):
        self.calls += 1
        assert wallet == "live", "gas must be read from LIVE rows, not ghost marks"
        return [dict(row) for row in self.rows[: int(limit)]]


def _row(fee_cost, *, chain="base", status="closed", quantity=LIVE_SIZE,
         entry_price=ENTRY_PRICE):
    return {
        "chain": chain,
        "status": status,
        "fee_cost": fee_cost,
        "quantity": quantity,
        "entry_price": entry_price,
    }


@pytest.fixture(autouse=True)
def _no_gas_cache(monkeypatch):
    """The TTL cache is keyed by chain and would leak between tests."""
    _clear_roundtrip_gas_cache()
    monkeypatch.delenv("ROUNDTRIP_GAS_USD", raising=False)
    monkeypatch.delenv("ROUNDTRIP_GAS_USD_BASE", raising=False)
    monkeypatch.delenv("MICRO_FIXED_COST_USD", raising=False)
    monkeypatch.delenv("MAX_HOLD_FORCE_SECONDS", raising=False)
    yield
    _clear_roundtrip_gas_cache()


def test_gas_is_the_median_of_the_realized_round_trips() -> None:
    db = _OutcomeDB([
        _row(0.012775823216219494, quantity=3.709e-05, entry_price=80884.33540037746),
        _row(0.0038717637252734037),
        _row(0.0030455062644577867, quantity=0.000111, entry_price=2844.974188079904),
        _row(0.004589863754314283, quantity=0.000162, entry_price=2739.7212776504584),
        _row(0.004319330367028045, quantity=1.539156, entry_price=0.48727988389842986),
    ])

    assert roundtrip_gas_usd(db, "base") == pytest.approx(MEASURED_GAS_USD)


def test_an_unmeasurable_chain_charges_nothing_rather_than_guessing() -> None:
    """The fallback direction matters more than the fallback value.

    The other candidate was the existing ``_estimate_gas_cost``
    (ESTIMATED_GAS_NATIVE=0.001 ETH = $2.50, a mainnet-shaped number). At $5 a
    round trip it would refuse every exit on base and strand positions instead
    of costing them. A missing measurement degrades to exactly the old
    behaviour: charge the rate, charge no gas.
    """
    db = _OutcomeDB([_row(0.004, chain="base")])

    assert roundtrip_gas_usd(db, "ethereum") == 0.0


def test_a_broken_db_read_never_blocks_a_close() -> None:
    class _Exploding:
        def fetch_trade_outcomes(self, **kwargs):
            raise RuntimeError("db is locked")

    assert roundtrip_gas_usd(_Exploding(), "base") == 0.0


def test_annulled_and_ghost_rows_are_not_gas_readings() -> None:
    db = _OutcomeDB([
        _row(9.0, status="annulled"),
        _row(0.004319330367028045),
    ])

    assert roundtrip_gas_usd(db, "base") == pytest.approx(MEASURED_GAS_USD)


def test_a_repricing_bug_cannot_poison_every_gate_at_once() -> None:
    """The CBBTC row booked $0.4136 of "gas" on a $3.00 trade -- 13.8%.

    It was priced at the cbBTC price instead of the ETH price. That row is
    corrected now; this is the guard that stops the next one being adopted as
    the cost of every future trade.
    """
    db = _OutcomeDB([
        _row(0.4135511037032909, quantity=3.709e-05, entry_price=80884.33540037746),
        _row(0.004319330367028045),
    ])

    assert roundtrip_gas_usd(db, "base") == pytest.approx(MEASURED_GAS_USD)


def test_env_is_the_bootstrap_for_a_chain_with_no_live_history(monkeypatch) -> None:
    """A fresh database must not hand the ghost book a free round trip."""
    monkeypatch.setenv("ROUNDTRIP_GAS_USD_BASE", "0.0045")
    db = _OutcomeDB([])

    assert roundtrip_gas_usd(db, "base") == pytest.approx(0.0045)


def test_measurement_beats_the_env_bootstrap(monkeypatch) -> None:
    monkeypatch.setenv("ROUNDTRIP_GAS_USD_BASE", "0.0045")
    db = _OutcomeDB([_row(MEASURED_GAS_USD)])

    assert roundtrip_gas_usd(db, "base") == pytest.approx(MEASURED_GAS_USD)


def test_the_measurement_is_cached_off_the_hot_path() -> None:
    db = _OutcomeDB([_row(MEASURED_GAS_USD)])
    now = 1788500000.0

    for _ in range(50):
        roundtrip_gas_usd(db, "base", now=now)

    assert db.calls == 1, "the exit path must not re-query the DB per tick"


def test_the_cache_is_per_chain() -> None:
    db = _OutcomeDB([_row(MEASURED_GAS_USD, chain="base")])
    now = 1788500000.0

    assert roundtrip_gas_usd(db, "base", now=now) == pytest.approx(MEASURED_GAS_USD)
    assert roundtrip_gas_usd(db, "arbitrum", now=now) == 0.0


# --------------------------------------------------------------------------
# 2. The entry gate charges it.
# --------------------------------------------------------------------------

def test_the_entry_gate_refuses_an_edge_that_only_covers_the_rate() -> None:
    """0.70% clears the 0.65% rate and does not clear the rate plus gas.

    On $0.75 of notional the measured round trip is 0.576% on top -- so the
    true bar is 1.23%, not 0.65%. This is the gap every one of the five live
    round trips fell into.
    """
    rate_only = evaluate_micro_profit(
        notional_usd=NOTIONAL,
        gross_return=0.0070,
        variable_cost_rate=0.0065,
        fixed_cost_usd=0.0,
        minimum_net_profit_usd=0.0,
    )
    with_gas = evaluate_micro_profit(
        notional_usd=NOTIONAL,
        gross_return=0.0070,
        variable_cost_rate=0.0065,
        fixed_cost_usd=MEASURED_GAS_USD,
        minimum_net_profit_usd=0.0,
    )

    assert rate_only.viable is True
    assert with_gas.viable is False
    assert with_gas.reason == "net_profit_below_dollar_floor"
    assert with_gas.net_profit_usd < 0.0


# --------------------------------------------------------------------------
# 3. The exit gate charges it -- at any age.
# --------------------------------------------------------------------------

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

    def fetch_trade_outcomes(self, *, wallet=None, limit=200):
        return [_row(MEASURED_GAS_USD)]

    def record_trade_outcome(self, *args, **kwargs):
        self.outcomes.append(kwargs or {"args": args})
        return True


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


def _portfolio():
    class _Portfolio(_Stub):
        holdings: dict = {}

        def get_quantity(self, symbol, chain=None):
            if str(symbol).upper() == "AERO":
                return LIVE_SIZE
            return 1000.0

        def get_native_balance(self, *args, **kwargs):
            return 1.0

    return _Portfolio()


def _bot() -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    TradingBot._ensure_runtime_state(bot)
    bot.db = _DB()
    bot.strategy_ledger = _Ledger()
    bot.portfolio = _portfolio()
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


def _position(mode: str, *, held_sec: float) -> dict:
    opened = time.time() - held_sec
    return {
        "mode": mode,
        "strategy_id": "atf_static",
        "size": LIVE_SIZE,
        "entry_price": ENTRY_PRICE,
        "target_price": ENTRY_PRICE * 1.05,
        "entry_ts": opened,
        "ts": opened,
        "trade_id": f"{mode}-trade-id",
        "route": ["AERO", "USDC"],
        "quote_spent": NOTIONAL,
        "base_token_address": AERO,
    }


def _exit_tick(bot: TradingBot, price: float, reason: str):
    sample = {"symbol": SYMBOL, "price": price, "ts": time.time(),
              "chain": "base", "volume": 100.0}
    summary = {"exit_conf": 0.9, "direction_prob": 0.9, "delta": 0.05,
               "net_margin": 0.05, "net_pnl": 0.05}
    directive = TradeDirective(
        action="exit",
        symbol=SYMBOL,
        base_token="AERO",
        quote_token="USDC",
        size=LIVE_SIZE,
        target_price=ENTRY_PRICE * 1.05,
        horizon="5m",
        confidence=0.9,
        expected_return=0.0,
        reason=reason,
        tier="core",
        tranches=[],
        strategy_id="atf_static",
        token_address=AERO,
    )

    async def _no_sync(self, *args, **kwargs):
        return None

    with mock.patch.object(
        TradingBot, "_resolve_live_trade_asset",
        lambda self, chain, sym, explicit=None: (sym, explicit or AERO),
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
    monkeypatch.setenv("MAX_HOLD_SECONDS", "3600")
    monkeypatch.delenv("MAX_POSITION_PRICE_RATIO", raising=False)


def test_a_stale_position_is_not_closed_below_its_own_cost() -> None:
    """The AERO regression: 4749s held, -0.139% move, closed anyway.

    Before this change the margin check lived inside
    ``if held_sec < max_hold_sec``, so at 4749s it was skipped and the close
    went through -- gross -0.001045, gas -0.003872, net -0.004917.
    """
    bot = _bot()
    bot.positions[SYMBOL] = _position("live", held_sec=4749.0)

    decision = _exit_tick(bot, price=ENTRY_PRICE * 0.99861, reason="confidence_drop")

    assert decision["status"] == "hold-negative"
    cost = decision["exit_cost"]
    assert cost["held_sec"] > cost["max_hold_sec"], "this is the past-the-clock case"
    assert cost["est_gas_usd"] == pytest.approx(MEASURED_GAS_USD)
    assert cost["est_net_profit_usd"] < 0.0
    assert bot.db.outcomes == [], "a close that cannot pay for itself booked an outcome"
    assert bot.positions[SYMBOL]["size"] == pytest.approx(LIVE_SIZE)


def test_a_direction_win_that_cannot_pay_the_gas_is_not_closed() -> None:
    """The first AERO round trip: price rose 0.165% and the close lost money.

    gross +0.001240, gas -0.004319, net -0.003079. Exiting was the loss.
    """
    bot = _bot()
    bot.positions[SYMBOL] = _position("live", held_sec=4749.0)

    decision = _exit_tick(bot, price=ENTRY_PRICE * 1.00165, reason="confidence_drop")

    assert decision["status"] == "hold-negative"
    assert decision["exit_cost"]["est_gross_profit_usd"] > 0.0, "the direction call was right"
    assert decision["exit_cost"]["est_net_profit_usd"] < 0.0, "and the close still loses"
    assert bot.db.outcomes == []


def test_a_stop_loss_still_fires_past_the_clock_and_below_cost() -> None:
    """The guard that must NOT be gained.

    "A single losing exit must never exceed the sum of the wins" depends on
    the stop being unconditional. A protective exit is never asked whether it
    can pay for itself.
    """
    bot = _bot()
    bot.positions[SYMBOL] = _position("live", held_sec=4749.0)

    decision = _exit_tick(bot, price=ENTRY_PRICE * 0.90, reason="stop_loss:-0.1000")

    assert decision["status"] != "hold-negative", "the stop was gated on being cheap"


def test_an_edge_that_clears_gas_is_still_closed() -> None:
    """The gate must not simply freeze the book.

    2% on $0.75 is $0.015 gross against $0.0049 of rate and $0.0043 of gas --
    profitable, and it goes through.
    """
    bot = _bot()
    bot.positions[SYMBOL] = _position("live", held_sec=4749.0)

    decision = _exit_tick(bot, price=ENTRY_PRICE * 1.02, reason="confidence_drop")

    assert decision["status"] != "hold-negative"


def test_the_operator_escape_hatch_is_off_by_default(monkeypatch) -> None:
    """MAX_HOLD_FORCE_SECONDS exists; leaving it unset must change nothing."""
    bot = _bot()
    bot.positions[SYMBOL] = _position("live", held_sec=4749.0)
    monkeypatch.setenv("MAX_HOLD_FORCE_SECONDS", "3600")

    decision = _exit_tick(bot, price=ENTRY_PRICE * 0.99861, reason="confidence_drop")

    assert decision["status"] != "hold-negative", "the hatch did not open"


# --------------------------------------------------------------------------
# 4. The ghost book charges what the live book pays.
# --------------------------------------------------------------------------

def test_a_ghost_round_trip_is_charged_the_gas_a_live_one_pays() -> None:
    """Graduation reads the ghost book; it has to be a book about this game.

    Before: ``fee_cost = notional * 0.0065`` and no gas, so a simulated round
    trip cost $0.0049 where the real one cost $0.0049 of spread plus $0.0043
    of gas. atf_static graduated on the cheap book and returned gross
    +0.008680 against gas -0.028603.

    The 0.0065 flat rate was itself then replaced by the measured, SIZE-DEPENDENT
    model in services/roundtrip_cost.py -- ``fixed + rate * notional`` -- because
    the fixed part does not shrink with the clip. This test kept asserting the
    flat rate it had already replaced and so failed against its own code; it now
    pins the model the bot actually charges, with the constants written out
    independently so "whatever the code computes" cannot satisfy it.
    """
    bot = _bot()
    bot.positions[SYMBOL] = _position("ghost", held_sec=4749.0)

    _exit_tick(bot, price=ENTRY_PRICE * 1.05, reason="target_reached")

    assert bot.db.outcomes, "the ghost exit booked nothing"
    booked = bot.db.outcomes[-1]
    notional = float(booked["quantity"]) * float(booked["entry_price"])

    # The simulated clip a ghost round trip is priced at, and the measured cost
    # of one at that size -- spelled out here rather than imported, so a change
    # to either constant has to be made deliberately in two places.
    ghost_clip = float(os.getenv("GHOST_MIN_TRADE_USD", "0.75"))
    fee_rate = (0.004047 + 0.003187 * ghost_clip) / ghost_clip
    assert bot._roundtrip_fee_rate(notional_hint=None) == pytest.approx(
        fee_rate, rel=1e-9
    ), "the bot and this test disagree about what a round trip costs"

    expected = notional * fee_rate + MEASURED_GAS_USD
    assert booked["fee_cost"] == pytest.approx(expected, rel=1e-6)
    # The teeth: gas is really in there, not just the spread.
    assert booked["fee_cost"] > notional * fee_rate
    # And the accounting still closes: net = gross - fee, or
    # `validate_outcome_math` would have refused the row.
    assert float(booked["net_profit"]) == pytest.approx(
        float(booked["gross_profit"]) - float(booked["fee_cost"]), rel=1e-9
    )
