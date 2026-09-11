"""A live entry must be sized to the clip the plan approved, held slot or not.

Link 9 (LIVE) on 2026-09-03: every gate in front of it was open. atf_static was
live_approved, dry-run was off, the arrow that aborted the last two swaps was
gone, and 139 quote_ok ATF candidates were published in under three hours. Not
one reached a swap. Measured from ``organism_snapshots`` in the hour to 06:20,
85 enter directives were produced and all 85 died at the same gate::

    micro_profit = {"viable": false, "notional_usd": 0.0534107448,
                    "net_profit_usd": 0.0026363802,
                    "minimum_net_profit_usd": 0.02,
                    "reason": "net_profit_below_dollar_floor"}

$0.053 of notional against a $6.98 wallet, while the plan in the same snapshot
said ``recommended_live_usd = 0.75`` and ``min_clip_usd = 0.75``. The sizing
chain that produced it:

    risk_budget = max(0.05, recommended_live_ratio 0.10749)
                  * ghost_risk_multiplier 0.46571          = 0.0500596
    size_usd    = wallet 6.977334 * frac 0.12 * 0.0500596  = $0.041914
                  * pair size_multiplier 1.274296          = $0.053411

reproducing the recorded notional to six decimal places. At $0.042 a 5% target
nets $0.0018 against a $0.02 floor and can never be viable; at $0.75 it nets
$0.0326 and is. Nothing was refused on its merits.

A floor for exactly this already existed and had already stopped working.
``MIN_DIRECTIVE_NOTIONAL_USD`` is set to 0.75 in .env, under a comment naming
this failure -- but it is gated on ``pos is None``, and **all 85** of those
refusals were on a symbol already in the position book. The ghost lane holds 17
symbols, several for 5-15 hours, so the empty-slot condition was never true for
anything the live lane wanted. The lever was on, configured, commented, and
skipped every single time.

The rule pinned here: an entry that takes an OCCUPIED slot releases it and
spends the same money as one that finds it empty, so the clip cannot depend on
which of the two it is. Plus the caps that still bind, the lanes that must not
be touched, and the row the refusal must leave behind.
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

#: The capital_plan read off the live bot at 2026-09-03 06:18, verbatim.
CAPITAL_PLAN = {
    "recommended_live_usd": 0.75,
    "min_clip_usd": 0.75,
    "deployable_stable_usd": 6.977258,
    "live_capital_cap_usd": 6.0,
    "live_ramp_schedule": {
        "first_tranche_usd": 0.75,
        "max_live_usd": 6.0,
        "deployable_stable_usd": 6.977258,
        "first_tranche_cap_usd": 1.5,
    },
}

WALLET_USDC = 6.977334
PRICE = 0.03675055091146844
#: BASECAT-USDC's real pair_adjustments row that day.
SIZE_MULTIPLIER = 1.2742955025552438
#: The directive the scheduler actually emitted, and the notional it became.
DIRECTIVE_SIZE = 1.1404982678877242
BLOCKED_NOTIONAL = 0.05341074480113392


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
        return {"size_multiplier": SIZE_MULTIPLIER,
                "allocation_multiplier": SIZE_MULTIPLIER}

    def fetch_market_samples_for(self, *args, **kwargs):
        return []

    def fetch_trade_fills(self, *args, **kwargs):
        return []


class _Ledger(_Stub):
    """Only atf_static graduated -- the real ledger state on the day."""

    def is_live_approved(self, strategy_id):
        return strategy_id == "atf_static"


class _Portfolio(_Stub):
    holdings: dict = {}

    def get_quantity(self, symbol, chain=None):
        return WALLET_USDC if str(symbol).upper() == "USDC" else 0.0

    def get_native_balance(self, *args, **kwargs):
        return 1.0


class _Validator(_Stub):
    def validate(self, **kwargs):
        return True, {}, []


class _Pipeline(_Stub):
    decision_threshold = 0.58


def _bot(*, live: bool = True, plan=CAPITAL_PLAN) -> TradingBot:
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
    bot.sim_quote_balances = {("base", "USDC"): WALLET_USDC}
    bot.stable_tokens = {"USDC"}
    bot.primary_chain = "base"
    bot.live_trading_enabled = live
    bot.ghost_session_id = 2
    bot._ghost_trade_counter = 0
    bot.max_trade_share = 0.05
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
    bot._transition_plan = {"capital_plan": dict(plan)} if plan else {}
    return bot


def _ghost_position() -> dict:
    """The BASECAT row the ghost lane had been holding for five hours."""
    return {
        "mode": "ghost",
        "strategy_id": "donchian_breakout@5d",
        "size": 91.73506053167031,
        "entry_price": 0.03303479078560976,
        "target_price": 0.0396,
        "entry_ts": time.time() - 18600,
        "ts": time.time() - 18600,
        "trade_id": "ghost-old-trade-id",
    }


def _directive(strategy_id: str, *, action: str = "enter", size: float = DIRECTIVE_SIZE):
    return TradeDirective(
        action=action, symbol=SYMBOL, base_token="BASECAT", quote_token="USDC",
        size=size, target_price=PRICE * 1.05, horizon="atf", confidence=0.9,
        expected_return=0.05, reason="ATF researched candidate quote_ok=True",
        strategy_id=strategy_id, token_address=BASECAT,
    )


def _enter(bot: TradingBot, directive, *, swapper=None):
    sample = {"symbol": SYMBOL, "price": PRICE, "ts": time.time(),
              "chain": "base", "volume": 5000.0}
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


@pytest.fixture(autouse=True)
def _production_env(monkeypatch):
    """Production's own settings: the state link 9 was failing in.

    ``MIN_DIRECTIVE_NOTIONAL_USD`` is pinned to .env's 0.75 rather than left to
    whatever the loader hydrated, so these tests measure the slot gate and not
    the environment.
    """
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "0")
    monkeypatch.setenv("STRATEGY_GRADUATION_ENFORCED", "1")
    monkeypatch.setenv("MIN_DIRECTIVE_NOTIONAL_USD", "0.75")
    monkeypatch.setenv("SMALL_PROFIT_FLOOR_USD", "0.02")


# ---------------------------------------------------------------------------
# The regression: a held slot must not shrink a live entry
# ---------------------------------------------------------------------------

def test_a_held_ghost_position_no_longer_shrinks_a_live_entry() -> None:
    """The exact production condition: 85 of 85 refusals looked like this."""
    bot = _bot()
    bot.positions[SYMBOL] = _ghost_position()

    decision = _enter(bot, _directive("atf_static"), swapper=_Stub())

    micro = decision["micro_profit"]
    assert micro["notional_usd"] == pytest.approx(0.75, rel=1e-6), micro
    assert micro["viable"] is True, micro
    assert decision["live_clip_usd"] == pytest.approx(0.75)
    # What it would have been sized at, to six decimal places.
    assert decision["live_clip_raised_from_usd"] == pytest.approx(
        BLOCKED_NOTIONAL, rel=1e-6
    )


def test_without_the_clip_the_micro_notional_is_no_longer_deadlocked() -> None:
    """The $0.053 refusal that blocked link 9 all day is GONE, deliberately.

    This test was named ``..._reproduces_the_deadlock`` and asserted that an
    unclipped $0.053 entry is refused ``net_profit_below_dollar_floor``. That
    deadlock was the flat $0.02 SMALL_PROFIT_FLOOR, and it was removed on
    purpose -- a constant floor on a micro notional is a gate that blocks
    everything, which is a bug and not safety.

    Keeping the old assertion would have been a test demanding the deadlock
    back. What is asserted instead is the thing the clip is actually for: the
    notional is still the small one, and it is no longer refused on a
    constant. The floor that replaced it is covered by
    ``test_a_live_entry_faces_a_floor_that_scales_with_its_own_cost``.
    """
    bot = _bot(plan=None)
    bot.positions[SYMBOL] = _ghost_position()

    decision = _enter(bot, _directive("atf_static"), swapper=_Stub())

    micro = decision["micro_profit"]
    assert micro["notional_usd"] == pytest.approx(BLOCKED_NOTIONAL, rel=1e-6)
    assert micro["viable"] is True, micro
    assert micro["reason"] != "net_profit_below_dollar_floor", micro


def test_an_empty_slot_still_gets_the_clip() -> None:
    """The path that already worked must keep working."""
    bot = _bot()
    decision = _enter(bot, _directive("atf_static"), swapper=_Stub())
    assert decision["micro_profit"]["notional_usd"] == pytest.approx(0.75, rel=1e-6)
    assert decision["micro_profit"]["viable"] is True


# ---------------------------------------------------------------------------
# The clip itself
# ---------------------------------------------------------------------------

def test_clip_is_the_plans_recommendation() -> None:
    assert _bot()._live_clip_usd() == pytest.approx(0.75)


def test_clip_never_exceeds_the_first_tranche_cap() -> None:
    """A plan recommending more than one tranche is still capped at one."""
    plan = dict(CAPITAL_PLAN, recommended_live_usd=25.0, min_clip_usd=25.0)
    assert _bot(plan=plan)._live_clip_usd() == pytest.approx(1.5)


def test_clip_never_exceeds_deployable_stable() -> None:
    plan = {"recommended_live_usd": 500.0, "min_clip_usd": 500.0,
            "deployable_stable_usd": 6.977258}
    assert _bot(plan=plan)._live_clip_usd() == pytest.approx(6.977258)


def test_no_plan_means_no_floor() -> None:
    """Sizing is left exactly as it was when no plan has loaded."""
    assert _bot(plan=None)._live_clip_usd() == 0.0
    assert _bot(plan={"recommended_live_usd": 0.0,
                      "min_clip_usd": 0.0})._live_clip_usd() == 0.0
    assert _bot(plan={"recommended_live_usd": "nonsense"})._live_clip_usd() == 0.0


def test_the_clip_never_asks_for_more_than_the_wallet_holds() -> None:
    poor = _bot(plan={"recommended_live_usd": 5.0, "min_clip_usd": 5.0})

    class _Empty(_Portfolio):
        def get_quantity(self, symbol, chain=None):
            return 0.40 if str(symbol).upper() == "USDC" else 0.0

    poor.portfolio = _Empty()
    poor.positions[SYMBOL] = _ghost_position()
    decision = _enter(poor, _directive("atf_static"), swapper=_Stub())
    assert decision["micro_profit"]["notional_usd"] <= 0.40 + 1e-9


# ---------------------------------------------------------------------------
# What it must NOT touch
# ---------------------------------------------------------------------------

def test_a_ghost_bot_is_not_resized() -> None:
    """The ghost lane spends sim_quote_balances; the live plan is not its plan."""
    bot = _bot(live=False)
    bot.positions[SYMBOL] = _ghost_position()
    decision = _enter(bot, _directive("atf_static"))
    assert "live_clip_usd" not in decision


def test_a_non_graduated_strategy_is_not_resized() -> None:
    """Only a strategy the ledger graduated may be sized for real money."""
    bot = _bot()
    bot.positions[SYMBOL] = _ghost_position()
    decision = _enter(bot, _directive("rsi_reversal"))
    assert "live_clip_usd" not in decision


def test_an_exit_is_never_resized() -> None:
    """An exit sells the position; its size comes from the book, not the plan."""
    bot = _bot()
    bot.positions[SYMBOL] = {
        "mode": "live", "strategy_id": "atf_static", "size": 20.0,
        "entry_price": PRICE, "target_price": PRICE * 1.2,
        "entry_ts": time.time() - 7200, "ts": time.time() - 7200,
        "trade_id": "held", "tx_hash": "0xrealhash",
        "base_token_address": BASECAT, "total_quote_spent": 0.75,
    }
    decision = _enter(bot, _directive("atf_static", action="exit", size=20.0))
    assert "live_clip_usd" not in decision


# ---------------------------------------------------------------------------
# The silence
# ---------------------------------------------------------------------------

def test_a_profit_floor_refusal_of_a_live_entry_leaves_a_row(monkeypatch) -> None:
    """The gate that refused 85 of 85 entries must say so in the database.

    THE GUARANTEE IS THE ROW, NOT THE REFUSAL. This test used to reach the
    floor for free, because the flat $0.02 SMALL_PROFIT_FLOOR refused a $0.053
    entry on sight. trading/bot.py:7733 now scales the floor to a quarter of
    the trade's own estimated cost, so this fixture sails through it -- and a
    refusal that never happens cannot be checked for its audibility.

    So the fixture is made to hit the floor honestly, by charging a fixed
    round-trip cost large enough that the net lands under a quarter of it.
    That is the same branch the real refusals take: ``gross_return <=
    variable_cost_rate`` is tested first and on the RATE alone, so a fixed
    cost cannot divert this to ``edge_does_not_cover_variable_costs``.
    """
    monkeypatch.setenv("MICRO_FIXED_COST_USD", "1.00")
    bot = _bot(plan=None)
    bot.positions[SYMBOL] = _ghost_position()
    _enter(bot, _directive("atf_static"), swapper=_Stub())

    rows = [r for r in bot.db.logged
            if r.get("status") == "live-entry-below-profit-floor"]
    assert len(rows) == 1, bot.db.logged
    assert rows[0]["wallet"] == "live"
    details = rows[0]["details"]
    assert details["executed"] is False
    assert details["strategy_id"] == "atf_static"
    assert details["reason"].startswith("micro-profit-blocked:")
    assert details["micro_profit"]["reason"] == "net_profit_below_dollar_floor"


def test_a_ghost_refusal_leaves_no_live_row() -> None:
    """A simulation declining a simulation is not a live-path refusal."""
    bot = _bot(live=False, plan=None)
    bot.positions[SYMBOL] = _ghost_position()
    _enter(bot, _directive("atf_static"))
    assert not [r for r in bot.db.logged
                if r.get("status") == "live-entry-below-profit-floor"]


def test_the_refusal_status_is_not_an_executed_trade() -> None:
    """live_rows / pass_scorecard whitelist ('live-entry','live-exit') only."""
    from scripts.live_path_check import EXECUTED_LIVE_STATUSES

    assert "live-entry-below-profit-floor" not in EXECUTED_LIVE_STATUSES


# ---------------------------------------------------------------------------
# The dollar floor is a real-money argument
# ---------------------------------------------------------------------------
#
# SMALL_PROFIT_FLOOR_USD exists because a real swap costs gas and fees that do
# not scale with size. A simulated entry broadcasts nothing, and the same floor
# was killing the evidence pipeline every graduation depends on -- completely,
# on a live bot, where a ghost entry can never exceed wallet * max_trade_share
# = $6.977 * 0.05 = $0.349 while $0.02 net at a 5% target needs $0.46. Measured
# 2026-09-03 06:32 over the previous two hours: 205 enter directives refused
# here, 10 ghost entries taken. money_button was refused 17 times at
# $0.13-$0.35 and has ONE trade in a ledger that needs 20.


def test_a_simulated_entry_is_not_held_to_the_dollar_floor() -> None:
    """A trade that spends nothing is not refused for earning few cents."""
    bot = _bot()                                   # live bot...
    bot.positions[SYMBOL] = _ghost_position()
    decision = _enter(bot, _directive("rsi_reversal"))   # ...non-graduated

    micro = decision["micro_profit"]
    assert micro["minimum_net_profit_usd"] == 0.0
    assert micro["net_profit_usd"] < 0.02, micro   # would have been refused
    assert micro["viable"] is True, micro
    assert decision["action"] == "enter"
    assert decision["status"] == "ghost-entry"


def test_a_simulation_with_no_edge_is_still_refused() -> None:
    """The RATE test survives: this is the one money_button keeps failing."""
    bot = _bot()
    no_edge = TradeDirective(
        action="enter", symbol=SYMBOL, base_token="BASECAT", quote_token="USDC",
        size=DIRECTIVE_SIZE, target_price=PRICE * 1.0005,  # 5bp, under fees
        horizon="5m", confidence=0.9, expected_return=0.0005,
        reason="dip", strategy_id="money_button",
    )
    sample_summary_margin = 0.0005

    async def _no_sync(self, *args, **kwargs):
        return None

    with mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync):
        decision = asyncio.run(
            bot._interpret_predictions(
                None,
                {"symbol": SYMBOL, "price": PRICE, "ts": time.time(),
                 "chain": "base", "volume": 5000.0},
                no_edge,
                pred_summary={"exit_conf": 0.9, "direction_prob": 0.9,
                              "delta": sample_summary_margin,
                              "net_margin": sample_summary_margin,
                              "net_pnl": 0.0},
                brain_summary={},
            )
        )

    micro = decision["micro_profit"]
    assert micro["reason"] == "edge_does_not_cover_variable_costs", micro
    assert micro["viable"] is False
    assert decision["action"] == "hold"


def test_a_live_entry_faces_a_floor_that_scales_with_its_own_cost() -> None:
    """The floor is a QUARTER OF ESTIMATED COST, never a flat $0.02.

    This test used to assert ``minimum_net_profit_usd == 0.02`` and it was
    right to, until trading/bot.py:7733 replaced the flat SMALL_PROFIT_FLOOR
    with ``_entry_profit_floor_ratio() * estimated_cost_usd`` on purpose. A
    flat dollar floor demands a different RATE at every clip -- $0.02 on a
    $0.75 trade is 2.67% of notional, four times any edge this pipeline has
    ever measured -- and it refused 385 of 385 ghost entries. It only ever
    passed because the gate credited each entry with a fantasy 5%.

    So the guarantee worth testing is no longer a NUMBER, it is a SHAPE: the
    floor must track the cost being estimated. Asserting the constant back
    would restore the gate that blocked everything.
    """
    bot = _bot(plan=None)                          # no clip to rescue it
    bot.positions[SYMBOL] = _ghost_position()
    decision = _enter(bot, _directive("atf_static"), swapper=_Stub())

    micro = decision["micro_profit"]
    floor = micro["minimum_net_profit_usd"]
    assert floor == pytest.approx(0.25 * micro["estimated_cost_usd"], rel=1e-9), micro
    # And it is emphatically NOT the flat floor, which would be ~175x larger
    # here and would refuse this entry on a constant rather than on its cost.
    assert floor < 0.02


# THE FIXTURE PINS THE SYMBOL EDGE GATE, IT DOES NOT WEAKEN IT.
#
# This file is about the CLIP -- what size a live entry takes and which
# floors it faces. The symbol edge gate sits upstream of all of that and it
# answers from production: services/symbol_edge_gate opens
# storage/trading_cache.db at test time and reads the live closed book. On
# 2026-09-10 BASECAT-USDC crossed the ban threshold -- 35 closed round trips
# at mean -0.852% against 0.465% cost, gross -1.4379 -- and three tests here
# started reading 'entry-refused-symbol-edge' where they assert on the clip.
# They went red because the bots traded, not because the clip changed.
#
# The ban is CORRECT and stays: services.symbol_edge_gate.refusal_reason
# still returns it, and the gate's own coverage lives with the gate.
#
# THE STRATEGY GATE IS THE SAME STORY, ONE LAYER UP, AND IT IS WHY THIS FILE
# STOOD RED FOR EIGHT COMMITS. `test_a_simulated_entry_is_not_held_to_the_dollar
# _floor` asserted `action == "enter"` and read `hold`; the operator bisected it
# to b9d823b 879ea6e 5b86639 da32265 0aa5ba5 9c8715b 0894789 5880e3a and found it
# failing at every one, so it was nobody's regression. With only the symbol gate
# pinned the refusal moved to `entry-refused-strategy-edge`:
#
#   strategy_edge: rsi_reversal, 10 closed round trips at mean return -0.625%
#                  vs 0.587% cost (t=-1.76 on excess return)
#
# `rsi_reversal` is the strategy this test names, and it crossed the strategy
# ban threshold the same way BASECAT-USDC crossed the symbol one -- by trading.
# (Pass 100 re-priced that gate at c9ffb5c and recorded rsi_reversal lifting at
# t=-1.44; it has since gone back under.) Exactly like the symbol gate, the ban
# is CORRECT and is NOT weakened here: services.strategy_edge_gate.refusal_reason
# still returns it, the gate's own tests still assert it, and this file is about
# the CLIP -- it cannot measure which size an entry takes if an upstream gate
# refuses the entry on evidence that arrived after the assertion was written.
@pytest.fixture(autouse=True)
def _entry_reaches_the_clip(monkeypatch):
    monkeypatch.setattr(
        "trading.bot.symbol_edge_refusal", lambda _symbol, _strategy_id=None: None
    )
    monkeypatch.setattr(
        "trading.bot.strategy_edge_refusal", lambda _strategy_id: None
    )
