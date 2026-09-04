"""A ghost entry must be run at the size the live lane would actually spend.

Link 5 (GRADUATION) on 2026-09-04: no strategy approved for live. atf_static was
demoted on a genuinely negative live record (-0.019922 over five settled round
trips) and ``_demote_locked`` blanks the ghost book, so re-graduation needs 20
fresh ghost trades. Those trades are the evidence the whole promotion gate is
built on -- and they were being simulated at a size the live lane cannot take.

Measured on the running bot, 2026-09-04::

    wallet base:USDC              = 18.1906
    sim_quote_balances base:USDC  = 100.71569451706375
    max_trade_share               = 0.05
    capital_plan.recommended_live_usd = 0.75

    CBBTC-USDC ghost position notional  = 1.1275236071379516e-05 * 80666.28
                                        = 0.90953135
    BASECAT-USDC ghost position notional= 19.602516665270784 * 0.04639870306098964
                                        = 0.90953135
    18.1906 * 0.05                      = 0.9095300000000001

Both open ghost positions were sized at exactly 5% of the REAL wallet, to seven
significant figures. ``GHOST_MIN_TRADE_USD`` (2.00) had raised them 180 lines
earlier and the wallet cap silently put them back: the floor was gated on
``entry_will_be_simulated`` but the cap it fed was still keyed on the bot-level
wallet, so half the change had landed and it read as done.

Why the size decides whether any of it is evidence: gas is a FIXED
$0.00431933 per round trip on base (``roundtrip_gas_usd(db, "base")``, measured
from our own receipts), so the move a round trip must make to break even is

    $0.1368 -> 0.65% + 3.157% = 3.81%   (the CP-USDC ghost exit, 2026-09-04)
    $0.9095 -> 0.65% + 0.475% = 1.125%
    $0.7500 -> 0.65% + 0.576% = 1.226%  <- what the money actually pays
    $2.0000 -> 0.65% + 0.216% = 0.866%

Every one of those is a different game. A ghost book run at $2.00 while the
plan authorises $0.75 promotes strategies into a 1.4x more expensive lane --
the same mistake, in the same direction, as the ghost book that charged no gas
at all and graduated atf_static into gross +0.008680 against gas -0.028603.

So the rule pinned here: **a simulated entry is sized to ``_live_clip_usd()``,
the clip the transition plan authorised, and it is sized against the SIMULATED
purse.** The purse half matters on its own -- this wallet is a rotation book and
a low stable leg mid-rotation is normal, so the ghost lane must not shrink to
whatever happens to be un-deployed at the moment a signal fires.
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

#: The capital_plan the live bot was carrying on 2026-09-04.
CAPITAL_PLAN = {
    "recommended_live_usd": 0.75,
    "min_clip_usd": 0.75,
    "deployable_stable_usd": 18.18649773,
    "live_capital_cap_usd": 6.0,
    "live_ramp_schedule": {
        "first_tranche_usd": 0.75,
        "max_live_usd": 6.0,
        "deployable_stable_usd": 18.18649773,
        "first_tranche_cap_usd": 1.5,
    },
}

#: Real balances read off the running bot.
WALLET_USDC = 18.1906
SIM_PURSE_USDC = 100.71569451706375
#: The CBBTC-USDC entry price of the ghost position that was open at the time.
PRICE = 80666.28
#: wallet * max_trade_share -- the notional both open ghost positions carried.
WALLET_CAPPED_NOTIONAL = 0.90953135


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
        # Neutral, so the assertions measure the sizing chain and not a
        # per-pair multiplier that is applied to both lanes alike.
        return {"size_multiplier": 1.0, "allocation_multiplier": 1.0}

    def fetch_market_samples_for(self, *args, **kwargs):
        return []

    def fetch_trade_fills(self, *args, **kwargs):
        return []


class _Ledger(_Stub):
    """Production's ledger state: nothing is approved except atf_static."""

    def is_live_approved(self, strategy_id):
        return strategy_id == "atf_static"


class _Validator(_Stub):
    def validate(self, **kwargs):
        return True, {}, []


class _Pipeline(_Stub):
    decision_threshold = 0.58


def _bot(
    *,
    live: bool = True,
    plan=CAPITAL_PLAN,
    wallet: float = WALLET_USDC,
    purse: float = SIM_PURSE_USDC,
) -> TradingBot:
    class _Portfolio(_Stub):
        holdings: dict = {}

        def get_quantity(self, symbol, chain=None):
            return wallet if str(symbol).upper() == "USDC" else 0.0

        def get_native_balance(self, *args, **kwargs):
            return 1.0

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
    bot.sim_quote_balances = {("base", "USDC"): purse}
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


def _directive(strategy_id: str, *, size: float = 1.1275236071379516e-05):
    return TradeDirective(
        action="enter", symbol=SYMBOL, base_token="CBBTC", quote_token="USDC",
        size=size, target_price=PRICE * 1.05, horizon="atf", confidence=0.9,
        expected_return=0.05, reason="ATF researched candidate quote_ok=True",
        strategy_id=strategy_id, token_address=CBBTC,
    )


def _enter(bot: TradingBot, directive):
    sample = {"symbol": SYMBOL, "price": PRICE, "ts": time.time(),
              "chain": "base", "volume": 0.0}
    summary = {"exit_conf": 0.9, "direction_prob": 0.9, "delta": 0.05,
               "net_margin": 0.05, "net_pnl": 0.05}

    async def _no_sync(self, *args, **kwargs):
        return None

    with mock.patch.object(
        TradingBot, "_resolve_live_trade_asset",
        lambda self, chain, sym, explicit=None: (sym, explicit or CBBTC),
    ), mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync), mock.patch.object(
        TradingBot, "_new_swapper", lambda self: _Stub()
    ):
        return asyncio.run(
            bot._interpret_predictions(
                None, sample, directive, pred_summary=summary, brain_summary={},
            )
        )


@pytest.fixture(autouse=True)
def _production_env(monkeypatch):
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "0")
    monkeypatch.setenv("STRATEGY_GRADUATION_ENFORCED", "1")
    monkeypatch.setenv("MIN_DIRECTIVE_NOTIONAL_USD", "0.75")
    monkeypatch.setenv("SMALL_PROFIT_FLOOR_USD", "0.02")
    monkeypatch.setenv("GHOST_MIN_TRADE_USD", "2.0")


# ---------------------------------------------------------------------------
# The regression
# ---------------------------------------------------------------------------

def test_a_signal_with_no_size_is_simulated_at_the_clip_not_the_env_constant() -> None:
    """Most ticks carry volume=0, so this is the ordinary ghost entry.

    GHOST_MIN_TRADE_USD is 2.00 and the plan authorises 0.75. Simulating at
    $2.00 charges the round trip 0.866% while the money pays 1.226%.
    """
    bot = _bot()

    decision = _enter(bot, _directive("money_button", size=0.0))

    notional = decision["micro_profit"]["notional_usd"]
    assert notional == pytest.approx(0.75, rel=1e-9), decision["micro_profit"]
    assert notional != pytest.approx(2.0, rel=1e-6)


def test_the_two_lanes_size_the_same_signal_the_same_way() -> None:
    """Ghost and live differ in whose purse bounds them, and nothing else.

    Graduation reads the ghost book to decide whether to spend real money, so
    any gap here is a gap between the evidence and the thing it is evidence
    about.
    """
    directive_size = 1.1275236071379516e-05  # the CBBTC clip production emitted

    ghost = _enter(_bot(), _directive("money_button", size=directive_size))
    live = _enter(_bot(), _directive("atf_static", size=directive_size))

    ghost_usd = ghost["micro_profit"]["notional_usd"]
    live_usd = live["micro_profit"]["notional_usd"]
    # Same signal, same size -- to within the one difference that is meant to
    # remain: the live lane is re-capped by the wallet (18.1906 * 0.05 =
    # 0.90953000) while the ghost lane is capped by the $100.72 sim purse and
    # so keeps the directive's own 0.90953135. 1.5e-6 of $0.91.
    assert ghost_usd == pytest.approx(live_usd, rel=1e-5)
    assert ghost_usd >= live_usd, (ghost_usd, live_usd)
    assert ghost_usd == pytest.approx(WALLET_CAPPED_NOTIONAL, rel=1e-6)


def test_a_low_stable_leg_does_not_shrink_the_simulation() -> None:
    """The wallet is a rotation book; mid-rotation the stable leg is low.

    A simulated entry spends ``sim_quote_balances``, so the real wallet's
    momentary balance must not decide how large the simulation is. Before this,
    5% of a $0.50 stable leg sized the ghost book at $0.025 -- where the fixed
    $0.00431933 of gas is 17% of the notional and nothing can ever win.
    """
    bot = _bot(wallet=0.50)

    decision = _enter(bot, _directive("money_button", size=0.0))

    assert decision["micro_profit"]["notional_usd"] == pytest.approx(
        0.75, rel=1e-9
    ), decision["micro_profit"]
    assert bot._sizing_quote(
        "base", "USDC", 0.50, simulated=True
    ) == pytest.approx(SIM_PURSE_USDC)


def test_without_a_plan_the_env_floor_still_applies() -> None:
    """A pure sim run has no live clip to copy, so the constant governs."""
    bot = _bot(plan=None)

    decision = _enter(bot, _directive("money_button", size=0.0))

    assert decision["micro_profit"]["notional_usd"] == pytest.approx(2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# The lane that spends real money must be untouched
# ---------------------------------------------------------------------------

def _held_ghost_slot() -> dict:
    """A ghost position the lane has been sitting on for five hours."""
    return {
        "mode": "ghost",
        "strategy_id": "donchian_breakout@5d",
        "size": 1.0e-05,
        "entry_price": PRICE * 0.98,
        "target_price": PRICE * 1.05,
        "entry_ts": time.time() - 18600,
        "ts": time.time() - 18600,
        "trade_id": "ghost-old-trade-id",
    }


def test_a_held_slot_does_not_change_which_purse_bounds_the_entry() -> None:
    """The gating mistake this repo has already made once.

    ``MIN_DIRECTIVE_NOTIONAL_USD`` was keyed on ``pos is None`` and was
    therefore skipped 85 times out of 85, because the ghost lane holds most of
    the ticking symbols for hours -- an empty slot is the rare case, not the
    common one. So the purse an entry is bounded by must not depend on whether
    the slot it takes was already occupied.

    The SIZE legitimately does differ: on a held slot ``trade_size`` falls back
    to the position's own size so the exit path still gets evaluated, and
    overriding that would stop held positions ever being closed. What must not
    differ is whose balance caps it -- which is what this measures, by dropping
    the stable leg to $0.50 and showing the simulated entry does not follow it
    down.
    """
    rich = _bot()
    rich.positions[SYMBOL] = _held_ghost_slot()
    poor = _bot(wallet=0.50)
    poor.positions[SYMBOL] = _held_ghost_slot()

    rich_usd = _enter(rich, _directive("money_button", size=0.0))["micro_profit"][
        "notional_usd"
    ]
    poor_usd = _enter(poor, _directive("money_button", size=0.0))["micro_profit"][
        "notional_usd"
    ]

    assert rich_usd == pytest.approx(poor_usd, rel=1e-9), (rich_usd, poor_usd)
    # 5% of a $0.50 stable leg is $0.025, where the fixed $0.00431933 of gas is
    # 17% of the notional. The simulation must not be dragged there.
    assert poor_usd > 0.5, poor_usd


def test_a_live_entry_is_still_sized_against_the_real_wallet() -> None:
    """The sim purse may never fund, or size, a trade that spends money."""
    bot = _bot()

    # $100.72 of virtual bankroll against a $0.50 wallet: a live entry must see
    # the wallet, and only the wallet.
    assert bot._sizing_quote("base", "USDC", 0.50, simulated=False) == pytest.approx(0.50)
    assert bot._sizing_quote(
        "base", "USDC", WALLET_USDC, simulated=False
    ) == pytest.approx(WALLET_USDC)


def test_a_graduated_strategy_still_gets_the_live_clip() -> None:
    """atf_static is the approved lane; its sizing must not have moved."""
    bot = _bot()

    decision = _enter(bot, _directive("atf_static", size=0.0))

    micro = decision["micro_profit"]
    assert micro["notional_usd"] == pytest.approx(0.75, rel=1e-6), micro
    assert micro["viable"] is True, micro


def test_the_purse_is_zero_when_the_chain_has_none() -> None:
    """A missing sim balance falls back to the wallet, never below it."""
    bot = _bot(purse=0.0)

    assert bot._simulated_quote_purse("base", "USDC") == 0.0
    assert bot._simulated_quote_purse("base", "WETH") == 0.0
    # max(wallet, purse): a simulated entry is never sized below what the
    # wallet would already have allowed.
    assert bot._sizing_quote(
        "base", "USDC", WALLET_USDC, simulated=True
    ) == pytest.approx(WALLET_USDC)
