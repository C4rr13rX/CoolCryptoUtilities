"""The entry gate must not accept a strategy's own advertisement as its edge.

``bot.py`` sized the entry viability check off the directive's ``target_price``::

    gross_return = max(0.0, margin)
    if directive is not None and price > 0.0 and directive.target_price > price:
        gross_return = (float(directive.target_price) - price) / price

``atf_static`` builds that target as ``price * 1.05``. So the gate asked "is 5%
more than the cost?", answered yes, and approved -- for every symbol, in every
market, forever. The target also OVERRODE ``margin``, the brain's own estimate,
so a strategy could not fail the gate by being wrong, only by claiming less.

Measured 2026-09-05 across all 20 live entries this account has ever taken
(``trading_ops`` status ``live-entry``, ``micro_profit.gross_profit_usd /
notional_usd``): credited +5.00% on fourteen of them and +5.0%..+6.1% on the
rest. Delivered, over the 18 that settled: median **-0.25%** gross, net win
rate 27.8%, book **-0.186371**.

Nor is the claim merely optimistic -- it is derived from the live price, so a
corrupt price makes a spectacular one. Of 1665 recent entry decisions 11.4%
claimed better than +100%, the largest +1.047e12 (CBETH-USDC quoted at
2.74e-09 against a correct target of 2869.7 -- a decimals shift). A gate keyed
to the claim is inverted: the worse the data, the more certainly it approves.

What is pinned here:

  1. the claim is a CEILING, never the estimate -- a strategy is credited with
     what it has delivered and never with more than it asked for;
  2. a strategy with a losing record is refused, and one whose record clears
     its costs is not;
  3. a feed artifact cannot set a strategy's reputation;
  4. the ghost lane still runs on the claim, because a simulated entry is how a
     strategy EARNS the record this reads -- gating it on one it does not have
     is a closed loop with no entrance, the shape that once refused 385 of 385
     ghost entries.
"""

from __future__ import annotations

import pytest

from trading.edge_estimate import (
    MAX_PLAUSIBLE_RETURN,
    EdgeEstimate,
    _clear_cache,
    estimate_gross_return,
    trimmed_mean,
)
from trading.micro_profit import evaluate_micro_profit

# The cost model fitted from this account's receipts.
FIXED_USD = 0.004047
VARIABLE_RATE = 0.003187
GAS_USD = 0.00432

#: What atf_static actually advertises on every entry.
ATF_CLAIM = 0.05


class _Book:
    """Minimal stand-in for the trading database."""

    def __init__(self, ghost=(), live=()):
        self._ghost = list(ghost)
        self._live = list(live)

    def fetch_trades(self, *, limit=200, statuses=None, **kwargs):
        if statuses and "ghost-exit" not in statuses:
            return []
        return [
            {"status": "ghost-exit", "details": {"strategy_id": sid, "profit": r}}
            for sid, r in self._ghost
        ][:limit]

    def fetch_trade_outcomes(self, *, wallet=None, limit=200):
        if wallet not in (None, "live"):
            return []
        return [
            {
                "status": "closed",
                "entry_price": 1.0,
                "exit_price": 1.0 + r,
                "details": {"strategy_id": sid},
            }
            for sid, r in self._live
        ][:limit]


@pytest.fixture(autouse=True)
def _fresh_cache():
    _clear_cache()
    yield
    _clear_cache()


def _round_trip(notional_usd: float, gross_return: float, *, floor_ratio: float = 0.25):
    """The gate's arithmetic at one clip size, as bot.py assembles it."""
    fee_rate = FIXED_USD / notional_usd + VARIABLE_RATE
    cost = fee_rate * notional_usd + GAS_USD
    return evaluate_micro_profit(
        notional_usd=notional_usd,
        gross_return=gross_return,
        variable_cost_rate=fee_rate,
        fixed_cost_usd=GAS_USD,
        minimum_net_profit_usd=floor_ratio * cost,
    )


def test_the_claim_is_a_ceiling_not_the_estimate():
    """A strategy is never credited with more than it has delivered."""
    book = _Book(ghost=[("atf_static", 0.004)] * 40)
    edge = estimate_gross_return(book, "atf_static", claimed_return=ATF_CLAIM)

    assert edge.source == "measured"
    assert edge.samples == 40
    # The whole defect in one assertion: 5% claimed, 0.4% credited.
    assert edge.value == pytest.approx(0.004, abs=1e-9)
    assert edge.value < ATF_CLAIM


def test_a_modest_claim_is_not_inflated_to_the_measurement():
    """The cap runs one way only -- this can refuse, never approve."""
    book = _Book(ghost=[("winner", 0.20)] * 40)
    edge = estimate_gross_return(book, "winner", claimed_return=0.01)
    assert edge.value == pytest.approx(0.01, abs=1e-9)


def test_a_corrupt_price_cannot_set_a_reputation():
    """The 1.047e12 claim was a decimals shift, not an opportunity."""
    clean = [("atf_static", 0.004)] * 40
    poisoned = clean + [("atf_static", 1.047e12)]

    assert estimate_gross_return(
        _Book(ghost=poisoned), "atf_static", claimed_return=ATF_CLAIM
    ).value == pytest.approx(
        estimate_gross_return(
            _Book(ghost=clean), "atf_static", claimed_return=ATF_CLAIM
        ).value,
        abs=1e-9,
    )
    assert MAX_PLAUSIBLE_RETURN == 1.0


def test_a_strategy_with_no_record_gets_the_house_average_not_its_claim():
    """Falling back to the claim would reinstate the tautology for every new
    strategy, which is exactly where the losing ones start."""
    book = _Book(ghost=[("established", 0.006)] * 40)
    edge = estimate_gross_return(book, "brand_new", claimed_return=ATF_CLAIM)

    assert edge.source == "pooled"
    assert edge.value == pytest.approx(0.006, abs=1e-9)
    assert edge.value < ATF_CLAIM


def test_a_cold_database_does_not_switch_the_pipeline_off():
    """No evidence anywhere must degrade to today's behaviour, not to a refusal
    of everything. A gate that blocks everything is the same as being off."""
    edge = estimate_gross_return(_Book(), "anything", claimed_return=ATF_CLAIM)
    assert edge.source == "unmeasured"
    assert edge.value == pytest.approx(ATF_CLAIM, abs=1e-9)


def test_the_trimmed_mean_survives_the_tail_the_raw_mean_inherits():
    """Half of atf_static's raw mean was tail artifact: +1.35% -> +0.65%."""
    values = [0.001] * 18 + [0.90, 0.95]
    raw = sum(values) / len(values)
    assert raw > 0.09
    assert trimmed_mean(values) < 0.01


def test_the_measured_edge_refuses_the_clips_that_lost_the_money():
    """atf_static breaks even at $3.25; every live entry was $0.75-$3.00.

    The 18 settled round trips were all below break-even before they were
    placed, which is why the book is -0.186371 rather than unlucky.
    """
    book = _Book(ghost=[("atf_static", 0.00576)] * 40)
    edge = estimate_gross_return(book, "atf_static", claimed_return=ATF_CLAIM)

    for clip in (0.75, 1.50, 3.00):
        assert not _round_trip(clip, edge.value).viable, clip
    assert _round_trip(6.00, edge.value).viable

    # The same clips pass on the 5% fiction -- this is what the gate was doing.
    for clip in (0.75, 1.50, 3.00, 6.00):
        assert _round_trip(clip, ATF_CLAIM).viable, clip


def test_a_losing_strategy_is_refused_at_every_clip_it_can_afford():
    """rsi_reversal@5h: 16 ghost round trips, 12.5% win rate, trimmed -11.9%."""
    book = _Book(ghost=[("rsi_reversal@5h", -0.119)] * 40)
    edge = estimate_gross_return(book, "rsi_reversal@5h", claimed_return=ATF_CLAIM)

    assert edge.value < 0.0
    for clip in (0.75, 1.50, 3.00, 6.00, 12.00):
        assert not _round_trip(clip, edge.value).viable, clip


def test_a_winning_record_still_trades():
    """atf_static_scout: 103 ghost round trips, 93.2% win rate, trimmed +1.55%.

    Never removing a guard that correctly refuses a loser is only half the
    rule; the other half is that the pipeline must not stop.
    """
    book = _Book(ghost=[("atf_static_scout", 0.01551)] * 40)
    edge = estimate_gross_return(book, "atf_static_scout", claimed_return=ATF_CLAIM)

    for clip in (1.50, 3.00, 6.00):
        assert _round_trip(clip, edge.value).viable, clip


def test_live_outcomes_are_read_as_returns_not_dollars():
    """`gross_profit` on a live row is DOLLARS. Averaging it into a table of
    RATES is the boundary error this repo has shipped before, so the return is
    recomputed from the fills."""
    book = _Book(live=[("atf_static", 0.02)] * 40)
    edge = estimate_gross_return(book, "atf_static", claimed_return=ATF_CLAIM)
    assert edge.source == "measured"
    assert edge.value == pytest.approx(0.02, abs=1e-9)


def test_the_estimate_is_serialisable_for_the_refusal_log():
    """A rule that declines to spend real money must be as visible as one that
    spends it -- the estimate rides in the blocked decision."""
    book = _Book(ghost=[("atf_static", 0.004)] * 40)
    payload = estimate_gross_return(
        book, "atf_static", claimed_return=ATF_CLAIM
    ).to_dict()
    assert payload["source"] == "measured"
    assert payload["claimed"] == pytest.approx(ATF_CLAIM)
    assert payload["strategy_id"] == "atf_static"
    assert set(payload) == {"value", "source", "samples", "claimed", "strategy_id"}


class _Caps:
    """Just enough bot to exercise the two capital-cap helpers."""

    def __init__(self, positions=None, cap=6.0):
        self.positions = positions or {}
        self._transition_plan = (
            {"capital_plan": {"live_capital_cap_usd": cap}} if cap else {}
        )


def test_the_live_capital_cap_is_measured_across_the_whole_book():
    """`live_capital_cap_usd` was read only where it bounds ONE clip, so "cap
    live capital at $6.00" was in force as "cap each entry at $6.00" and the
    book could hold any number of them. Invisible at a $0.75 clip; at a clip
    sized to clear its costs it is $6.00 of risk versus the whole wallet."""
    from trading.bot import TradingBot

    bot = _Caps(
        {
            "AERO-USDC": {"mode": "live", "quote_spent": 2.5},
            "GHOST-USDC": {"mode": "ghost", "quote_spent": 99.0},
            "Y-USDC": {"mode": "live", "size": 4.0, "entry_price": 0.5},
        }
    )
    assert TradingBot._live_capital_cap_usd(bot) == pytest.approx(6.0)
    # Ghost money is not live money; the $99 must not count.
    assert TradingBot._live_deployed_usd(bot) == pytest.approx(4.5)
    # An add-on measures the symbol's NEW total, so it excludes its own basis.
    assert TradingBot._live_deployed_usd(
        bot, exclude_symbol="AERO-USDC"
    ) == pytest.approx(2.0)


def test_no_plan_disables_the_capital_cap_rather_than_blocking_every_entry():
    """Same convention as _live_clip_usd(): a missing plan must leave sizing
    exactly as it was, not refuse everything."""
    from trading.bot import TradingBot

    assert TradingBot._live_capital_cap_usd(_Caps(cap=None)) == 0.0


def test_the_profit_floor_is_a_rate_not_a_flat_dollar_amount():
    """$0.02 demands 2.67% of a $0.75 clip and 0.33% of a $6.00 one, so what
    the gate required moved with the clip for no reason anyone chose."""
    from trading.bot import _entry_profit_floor_ratio

    ratio = _entry_profit_floor_ratio()
    assert 0.0 <= ratio <= 1.0

    # The floor tracks the cost it is protecting against, in both directions.
    small = _round_trip(0.75, 0.05, floor_ratio=ratio).minimum_net_profit_usd
    large = _round_trip(6.00, 0.05, floor_ratio=ratio).minimum_net_profit_usd
    assert large > small
    # And it is reachable: the old flat $0.02 was not, at any clip this
    # wallet can fund on any edge this pipeline has measured.
    assert _round_trip(6.00, 0.00576, floor_ratio=ratio).viable


def test_a_broken_database_does_not_raise_on_the_money_path():
    class _Angry:
        def fetch_trades(self, **kwargs):
            raise RuntimeError("db down")

        def fetch_trade_outcomes(self, **kwargs):
            raise RuntimeError("db down")

    edge = estimate_gross_return(_Angry(), "atf_static", claimed_return=ATF_CLAIM)
    assert isinstance(edge, EdgeEstimate)
    assert edge.value == pytest.approx(ATF_CLAIM)
