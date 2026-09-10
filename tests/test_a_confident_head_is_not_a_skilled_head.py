"""A head's LEVEL must never be read as its SKILL.

THE FAILURE THIS PREVENTS. Backlog item 618d4c4b was filed as "the prediction
head collapsed: direction_prob p50 0.7833 -> 0.0317" and carried the
acceptance criterion "direction_prob p50 over the most recent 2h rises back
above 0.5". Measured against the realised tape on 2026-09-10, the window with
the HIGH level had AUC 0.35-0.40 against forward returns at 5m, 15m and 30m --
reliably wrong -- while the "collapsed" window was the only one with any skill
(0.56-0.60). The high level was ``price_mu`` foreign-row saturation, not
opinion. Satisfying that criterion would have re-opened the ghost lane onto
the one signal in the window that loses.

So these tests assert that ``scripts/head_skill_census`` ranks a head by the
ORDERING of its scores against what the tape did next, and that a confident,
anti-predictive head is called INVERTED however high it sits. An
implementation that judged by level would pass none of them.
"""

from __future__ import annotations

import pytest

from scripts.head_skill_census import (
    _is_no_prediction_sentinel,
    auc,
    cost_floor_sweep,
    cost_floor_verdict,
    move_size_filter,
    move_size_verdict,
    trailing_volatility,
    forward_return,
    score_window,
    verdict,
)


def _tape(symbol: str, moves: list[float], *, step: float = 60.0, start: float = 1000.0):
    """A price track that realises ``moves`` as successive fractional steps."""
    price = 100.0
    track = [(start, price)]
    for move in moves:
        price *= 1.0 + move
        track.append((track[-1][0] + step, price))
    return {symbol: track}


def test_a_confident_but_inverted_head_is_not_called_healthy():
    """High direction_prob, consistently wrong: the exact 12-24h-ago window.

    EVERY score here is bullish -- 0.90 before a fall and 0.60 before a rise
    -- so the level is high on every single row, exactly as it was in the
    window this item called healthy. The ordering is nonetheless perfectly
    backwards: the head is most confident precisely when the tape falls.
    """
    series = _tape("FAKE-USDC", [-0.02, 0.02] * 20)
    preds = []
    for index in range(40):
        rises = index % 2 == 1  # the tape alternates down, up, down, up...
        preds.append(
            {
                "ts": 1000.0 + index * 60.0,
                "symbol": "FAKE-USDC",
                "direction_prob": 0.60 if rises else 0.90,
                "direction_prob_raw": 0.60 if rises else 0.90,
                "price_mu": -1.2,
            }
        )
    window = score_window(preds, series, horizons=(60,))
    scored = window["horizons"][60]

    assert scored is not None, "a window with both up and down moves must score"
    assert scored["auc"] < 0.5, f"an inverted head must score below 0.5, got {scored['auc']}"
    assert window["level_p50"] > 0.5, "this fixture is only meaningful while the LEVEL is high"
    assert "INVERTED" in verdict(window)


def test_a_low_level_head_that_orders_correctly_is_called_skilled():
    """The mirror: the "collapsed" window, low and right, must read SKILLED."""
    series = _tape("FAKE-USDC", [-0.02, 0.02] * 20)
    preds = []
    for index in range(40):
        rises = index % 2 == 1
        # Both scores are far below the 0.6 entry floor -- a "collapsed" head --
        # but the higher of the two lands before the rise every time.
        preds.append(
            {
                "ts": 1000.0 + index * 60.0,
                "symbol": "FAKE-USDC",
                "direction_prob": 0.20 if rises else 0.05,
                "direction_prob_raw": 0.20 if rises else 0.05,
                "price_mu": -0.19,
            }
        )
    window = score_window(preds, series, horizons=(60,))

    assert window["level_p50"] < 0.5, "this fixture is only meaningful while the LEVEL is low"
    assert window["horizons"][60]["auc"] > 0.5
    assert "SKILLED" in verdict(window)


def test_the_verdict_needs_every_horizon_to_agree():
    """One flattering horizon must not carry a verdict on its own.

    A single horizon is how a long-only rule flatters itself in an up window,
    which has produced a fake 78% and a fake +0.9067% in this repo already.
    """
    window = {
        "n": 100,
        "level_p50": 0.4,
        "price_mu_p50": -0.2,
        "horizons": {
            300: {"auc": 0.62, "stderr": 0.02, "n": 100, "n_up": 50, "n_down": 50,
                  "realised_p50": 0.0},
            900: {"auc": 0.50, "stderr": 0.02, "n": 100, "n_up": 50, "n_down": 50,
                  "realised_p50": 0.0},
        },
    }
    assert "NO INFORMATION" in verdict(window)


def test_an_unscorable_window_is_not_a_verdict():
    """A flat tape yields no up/down pairs, and that is not evidence of skill."""
    assert auc([(0.9, 0.0), (0.1, 0.0)]) is None
    window = {"n": 2, "level_p50": 0.5, "price_mu_p50": 0.0, "horizons": {300: None}}
    assert "UNSCORED" in verdict(window)


def test_the_no_prediction_sentinel_is_not_a_prediction():
    """(direction_prob 0.5, net_margin 0.0) is bot.py's "no head ran" summary.

    Counting it reports a collapsed head sitting at its neutral ceiling, and
    it would drag every AUC toward 0.5 with rows carrying no opinion at all.
    """
    assert _is_no_prediction_sentinel({"direction_prob": 0.5, "net_margin": 0.0})
    assert _is_no_prediction_sentinel({"direction_prob": 0.5, "net_margin": None})
    assert not _is_no_prediction_sentinel({"direction_prob": 0.5, "net_margin": 0.02})
    assert not _is_no_prediction_sentinel({"direction_prob": 0.12, "net_margin": 0.0})


def test_an_implausible_realised_move_is_not_scored():
    """A 50%+ move inside the horizon is a denomination flip, not a price.

    Leaving one in moves an AUC by more than the effect being measured, which
    is how this feed's two-price-regime symbols have contaminated every
    earlier read.
    """
    series = {"FAKE-USDC": [(0.0, 100.0), (60.0, 400.0)]}
    assert forward_return(series, "FAKE-USDC", 0.0, 60.0) is None

    sane = {"FAKE-USDC": [(0.0, 100.0), (60.0, 101.0)]}
    assert forward_return(sane, "FAKE-USDC", 0.0, 60.0) == pytest.approx(0.01)


def test_the_forward_return_anchors_on_a_tradeable_quote():
    """A prediction between two ticks is scored from the next real quote.

    Anchoring on the prediction's own timestamp would score it against a
    price that never existed on the tape.
    """
    series = {"FAKE-USDC": [(0.0, 100.0), (100.0, 100.0), (160.0, 102.0)]}
    # Prediction at t=50 lands between the first two quotes; the anchor is the
    # quote at t=100, and 60s later the tape is at 102.
    assert forward_return(series, "FAKE-USDC", 50.0, 60.0) == pytest.approx(0.02)


def test_a_pooled_edge_that_lives_in_one_up_window_is_called_no_edge():
    """The trap that killed two apparent edges on 2026-09-10, both in one pass.

    One window carries a large positive; every other window is negative. The
    POOLED mean is positive, so a pooled read reports an edge. The regime
    split must refuse it, because the count of net-positive windows -- not
    their mean -- is what says whether a rule would have paid repeatedly.
    """
    from scripts.head_skill_census import regime_split, regime_verdict

    series = {"FAKE-USDC": []}
    preds = []
    now = 100_000.0
    # Six 2h windows. Window 0 (newest) rises hard; the rest drift down.
    for window in range(6):
        rising = window == 0
        for index in range(100):
            ts = now - (window * 7200.0 + 3600.0) - index
            price_move = 0.05 if rising else -0.01
            series["FAKE-USDC"].append((ts, 100.0))
            series["FAKE-USDC"].append((ts + 900.0, 100.0 * (1 + price_move)))
            preds.append(
                {
                    "ts": ts,
                    "symbol": "FAKE-USDC",
                    "direction_prob": 0.3,
                    "direction_prob_raw": 0.3,
                    "price_mu": -0.1,
                }
            )
    series["FAKE-USDC"].sort()

    summary = regime_split(preds, series, now=now, hours=24.0, min_rows=50)
    ups = summary["regimes"]["UP"]
    downs = summary["regimes"]["DOWN"]

    assert ups["n_windows"] >= 1 and downs["n_windows"] >= 1, "fixture needs both regimes"
    assert downs["n_positive"] == 0, "every down window here loses"
    assert "NO EDGE" in regime_verdict(summary)


def test_a_single_regime_window_cannot_claim_an_edge():
    """An up-only sample is UNPROVEN, never EDGE HOLDS.

    A verdict of "positive in 3/3 up windows" with no down window is exactly
    the shape of this repo's fake 78%.
    """
    from scripts.head_skill_census import regime_verdict

    summary = {
        "windows": [],
        "regimes": {
            "UP": {"n_windows": 3, "n_positive": 3, "mean_net": 0.004},
            "DOWN": {"n_windows": 0, "n_positive": 0, "mean_net": float("nan")},
        },
    }
    assert "UNPROVEN" in regime_verdict(summary)


def test_an_edge_is_only_claimed_when_both_regimes_hold():
    """The mirror, so the verdict is not merely a pessimist."""
    from scripts.head_skill_census import regime_verdict

    summary = {
        "windows": [],
        "regimes": {
            "UP": {"n_windows": 4, "n_positive": 3, "mean_net": 0.004},
            "DOWN": {"n_windows": 4, "n_positive": 3, "mean_net": 0.002},
        },
    }
    assert "EDGE HOLDS IN BOTH REGIMES" in regime_verdict(summary)


# ---------------------------------------------------------------------------
# THE FIXED COST LEG, AND THE RANK-THRESHOLD SWEEP.
#
# Two defects these cover, both found on 2026-09-10 in the census itself
# rather than in the code it measures:
#
#   1. The census charged only the 0.3187% notional RATE and silently dropped
#      the $0.004047 FIXED leg of the round trip. Under-billing an entry is
#      how a rule that loses money reads as an edge, and this repo has already
#      shipped a fee in the wrong currency once.
#   2. It scored ONE percentile (the top decile) and the acceptance criterion
#      on item 618d4c4b asks whether ANY rank threshold extracts the head's
#      ordering. "The top 10% does not pay" and "no rank threshold pays" are
#      different claims, and only the second decides whether to build a gate.
# ---------------------------------------------------------------------------


def test_the_fixed_leg_of_the_round_trip_is_charged_not_dropped():
    """$0.004047 is dollars and 0.3187% is a fraction; they must not be mixed.

    The fixed leg does not shrink with the trade, so it only becomes a
    fraction after being divided by the clip it is charged against. A census
    that omits it under-bills a $10 clip by 0.0405% and a $1 clip by 0.4047%
    -- the latter being larger than every effect this script measures.
    """
    from scripts.head_skill_census import ROUND_TRIP_FIXED_USD, total_cost_fraction

    rate = 0.003187
    assert total_cost_fraction(notional_rate=rate, clip_usd=10.0) == pytest.approx(
        rate + ROUND_TRIP_FIXED_USD / 10.0
    )
    # A SMALLER CLIP IS MORE EXPENSIVE, NOT LESS. Getting this backwards would
    # make the micro-clip lane look like the cheap one.
    assert total_cost_fraction(notional_rate=rate, clip_usd=1.0) > total_cost_fraction(
        notional_rate=rate, clip_usd=10.0
    )
    assert total_cost_fraction(notional_rate=rate, clip_usd=1.0) == pytest.approx(
        rate + ROUND_TRIP_FIXED_USD
    )
    with pytest.raises(ValueError):
        total_cost_fraction(notional_rate=rate, clip_usd=0.0)


def test_a_sweep_that_clears_only_up_windows_is_not_an_edge():
    """THE UP-WINDOW TRAP, AT THE SWEEP LEVEL.

    A long-only rule flatters itself in an up window. A sweep multiplies the
    danger: seven thresholds scored on one dataset produce a maximum whether
    or not any signal is present, so a verdict that named the best rung would
    manufacture an edge from noise. This fixture is net-positive in EVERY up
    window at every tightness and never in a down window -- the exact shape of
    this repo's fake 78% and fake +0.9067% -- and must still read as no edge.
    """
    from scripts.head_skill_census import sweep_verdict

    sweep = {
        "cost": 0.0036,
        "n_windows": 9,
        "rungs": [
            {
                "pct": pct,
                "regimes": {
                    "UP": {"n_windows": 4, "n_positive": 4, "mean_net": 0.006},
                    "DOWN": {"n_windows": 5, "n_positive": 0, "mean_net": -0.005},
                },
                "clears_both": False,
            }
            for pct in (0.01, 0.05, 0.10)
        ],
    }
    assert "NO RANK THRESHOLD PAYS" in sweep_verdict(sweep)


def test_a_threshold_clearing_both_regimes_is_a_hypothesis_not_an_edge():
    """The mirror, and it must still refuse to call the result shippable.

    Even when a rung clears both regimes, the cut was chosen after seeing the
    data. The verdict says HYPOTHESIS and asks for a held-out window, because
    the one unforgivable outcome here is inventing a positive.
    """
    from scripts.head_skill_census import sweep_verdict

    sweep = {
        "cost": 0.0036,
        "n_windows": 9,
        "rungs": [
            {
                "pct": 0.01,
                "regimes": {
                    "UP": {"n_windows": 4, "n_positive": 3, "mean_net": 0.006},
                    "DOWN": {"n_windows": 5, "n_positive": 4, "mean_net": 0.004},
                },
                "clears_both": True,
            }
        ],
    }
    spoken = sweep_verdict(sweep)
    assert "HYPOTHESIS" in spoken and "held-out" in spoken
    assert "NO RANK THRESHOLD PAYS" not in spoken


def test_the_sweep_scores_every_threshold_on_identical_rows():
    """A sweep whose rungs saw different samples compares nothing.

    Each rung must be a cut of the SAME joined rows -- if the tape join were
    redone per threshold, two rungs could differ because of which rows they
    happened to catch rather than because of the threshold. So every rung
    reports the same window counts.
    """
    from scripts.head_skill_census import percentile_sweep

    now = 100_000.0
    # A tape that alternates so both regimes and both classes are present.
    series = {"FAKE-USDC": []}
    preds = []
    price = 100.0
    for index in range(400):
        ts = now - 20_000.0 + index * 50.0
        series["FAKE-USDC"].append((ts, price))
        price *= 1.0 + (0.004 if index % 2 else -0.003)
        preds.append(
            {
                "ts": ts,
                "symbol": "FAKE-USDC",
                "direction_prob": (index % 10) / 10.0,
                "direction_prob_raw": (index % 10) / 10.0,
                "price_mu": 0.0,
            }
        )
    series["FAKE-USDC"].sort()

    sweep = percentile_sweep(
        preds, series, now=now, hours=6.0, horizon_sec=300, cost=0.0036, min_rows=20
    )
    assert sweep["rungs"], "fixture must produce at least one rung"
    counts = {
        (
            rung["regimes"]["UP"]["n_windows"],
            rung["regimes"]["DOWN"]["n_windows"],
        )
        for rung in sweep["rungs"]
    }
    assert len(counts) == 1, f"rungs scored different window sets: {counts}"
    total = sum(next(iter(counts)))
    assert total == sweep["n_windows"]


def test_a_grid_scan_is_judged_against_chance_not_by_its_best_cell():
    """SCANNING N CELLS AND REPORTING THE WINNER IS THE FAKE-EDGE MACHINE.

    A (horizon x threshold) grid has to win a majority of UP windows AND a
    majority of DOWN windows in each cell. Treat a window as a fair coin and
    each cell clears about 1/4 of the time, so a 28-cell grid yields roughly
    SEVEN clearing cells from noise alone. A verdict that named the best cell
    would therefore report an edge on pure noise every time it was run.
    """
    from scripts.head_skill_census import grid_verdict

    # Exactly the chance expectation: must NOT read as an edge.
    at_chance = {
        "n_cells": 28,
        "n_clearing": 7,
        "expected_by_chance": 7.0,
        "cells": [],
        "cost": 0.0036,
    }
    assert "INDISTINGUISHABLE FROM CHANCE" in grid_verdict(at_chance)

    # Below chance -- and still not an edge.
    below = dict(at_chance, n_clearing=3)
    assert "INDISTINGUISHABLE FROM CHANCE" in grid_verdict(below)

    # Nothing at all clears: say so plainly, and say what chance expected.
    none = dict(at_chance, n_clearing=0)
    spoken = grid_verdict(none)
    assert "NO CELL PAYS" in spoken and "7.0 expected" in spoken

    # Well above chance is still only a hypothesis needing a held-out window.
    above = dict(at_chance, n_clearing=21)
    spoken = grid_verdict(above)
    assert "ABOVE CHANCE" in spoken and "HYPOTHESIS" in spoken


def test_the_grid_counts_every_cell_it_scanned():
    """A grid that silently dropped cells would deflate the chance bar.

    The chance expectation is a fraction of the cell COUNT, so under-counting
    scanned cells makes an ordinary result look better than chance. Every
    horizon-threshold pair scanned must appear in n_cells.
    """
    from scripts.head_skill_census import GRID_HORIZONS_SEC, SWEEP_PERCENTILES
    from scripts.head_skill_census import horizon_threshold_grid

    now = 100_000.0
    series = {"FAKE-USDC": []}
    preds = []
    price = 100.0
    for index in range(1200):
        ts = now - 90_000.0 + index * 60.0
        series["FAKE-USDC"].append((ts, price))
        price *= 1.0 + (0.004 if index % 2 else -0.003)
        preds.append(
            {
                "ts": ts,
                "symbol": "FAKE-USDC",
                "direction_prob": (index % 10) / 10.0,
                "direction_prob_raw": (index % 10) / 10.0,
                "price_mu": 0.0,
            }
        )
    series["FAKE-USDC"].sort()

    grid = horizon_threshold_grid(
        preds, series, now=now, hours=24.0, cost=0.0036, min_rows=20
    )
    assert grid["n_cells"] == len(GRID_HORIZONS_SEC) * len(SWEEP_PERCENTILES)
    assert grid["n_cells"] == len(grid["cells"])
    assert grid["expected_by_chance"] == pytest.approx(0.25 * grid["n_cells"])


def _flat_tape_preds(symbol: str, moves: list[float], *, now: float, step: float = 60.0):
    """A tape and one prediction per bar, so every bar is a scorable row."""
    series = _tape(symbol, moves, step=step, start=now - (len(moves) + 2) * step)
    preds = [
        {
            "ts": ts,
            "symbol": symbol,
            "direction_prob": 0.5,
            "direction_prob_raw": 0.5,
            "price_mu": 0.0,
        }
        for ts, _ in series[symbol][:-1]
    ]
    return series, preds


def test_a_bigger_clip_cannot_buy_its_way_past_the_cost_floor():
    """"Trade bigger" is the standing suggestion, and the arithmetic refuses it.

    The clip amortises ONLY the fixed $0.004047 leg. The 0.3187% rate is
    charged on notional and is therefore invariant to size, so raising the
    clip 25x can never move the clearing share by more than the fixed leg was
    worth in the first place. This test pins that ceiling: every bar here
    moves 0.33%, which sits ABOVE the rate and BELOW the $10-clip floor, so a
    clip large enough to amortise the fixed leg to nothing flips every row --
    the largest swing the lever can possibly produce -- and the test asserts
    the sweep reports it as a clip effect rather than as an edge.
    """
    now = 2_000_000.0
    series, preds = _flat_tape_preds("FAKE-USDC", [0.0033] * 60, now=now)
    sweep = cost_floor_sweep(
        preds, series, now=now, hours=24.0, notional_rate=0.003187,
        horizons=(60,), clips=(10.0, 250.0), min_symbol_rows=10,
        symbol_horizon_sec=60,
    )
    small, large = sweep["clips"]
    # The floor falls, because the fixed leg is amortised -- and only by that.
    assert small["cost"] > large["cost"]
    assert large["cost"] == pytest.approx(0.003187, abs=1e-4)
    # A 0.33% move is under the $10 floor and over the $250 one.
    assert small["share_clearing"] == pytest.approx(0.0)
    assert large["share_clearing"] == pytest.approx(1.0)
    # And the verdict must still say the clip is not the lever, because the
    # rate it cannot touch is what the floor is made of.
    assert "CLIP CANNOT" in cost_floor_verdict(sweep)


def test_clearing_the_cost_floor_is_never_reported_as_an_edge():
    """The one misreading this whole arm invites, and it costs real money.

    A tape where every bar outruns the fee makes all three arms read 100%.
    That is a NECESSARY condition for a profitable round trip and nowhere near
    a sufficient one -- direction is still unmeasured here, and this repo has
    already shipped a fake edge by treating a favourable-looking aggregate as
    a green light. The verdict must carry the refusal even when every number
    in it is maximal.
    """
    now = 2_000_000.0
    series, preds = _flat_tape_preds("FAKE-USDC", [0.05, -0.05] * 30, now=now)
    sweep = cost_floor_sweep(
        preds, series, now=now, hours=24.0, notional_rate=0.003187,
        horizons=(60,), clips=(10.0,), min_symbol_rows=10, symbol_horizon_sec=60,
    )
    assert sweep["horizons"][0]["share_clearing"] == pytest.approx(1.0)
    assert sweep["symbols"][0]["share_clearing"] == pytest.approx(1.0)
    verdict_text = cost_floor_verdict(sweep)
    assert "HORIZON CAN" in verdict_text
    assert "CLEARING THE FLOOR IS NOT AN EDGE" in verdict_text
    assert "DOWN window" in verdict_text


def test_a_minority_of_clearing_ticks_is_not_a_clearing_horizon():
    """27% of 15m ticks clear the fee, and that horizon must not read CAN.

    A rule enters on what it can identify in advance, so a horizon where the
    move outruns the fee on a minority of ticks is one where the majority pay
    the fee for nothing. Judging the arm on "some tick clears" would have
    called 5m tradeable at 17.8%.
    """
    now = 2_000_000.0
    # Three bars in ten move far enough; the other seven do not.
    moves = ([0.02] * 3 + [0.0001] * 7) * 8
    series, preds = _flat_tape_preds("FAKE-USDC", moves, now=now)
    sweep = cost_floor_sweep(
        preds, series, now=now, hours=24.0, notional_rate=0.003187,
        horizons=(60,), clips=(10.0,), min_symbol_rows=10, symbol_horizon_sec=60,
    )
    share = sweep["horizons"][0]["share_clearing"]
    assert 0.0 < share < 0.5
    assert "HORIZON CANNOT" in cost_floor_verdict(sweep)


def test_the_move_size_filter_cannot_see_the_move_it_filters_on():
    """The easiest fake edge in this file, and the guard that stops it.

    A move-size condition is worth something only if it can be evaluated at
    the moment of entry. This tape is dead flat before t and violent after
    it, so a predictor computed from the FORWARD window would read high and
    one computed from the trailing window must read zero. If this assertion
    ever fails, the filter is selecting entries by hindsight and every number
    downstream of it is manufactured.
    """
    track = [(float(i) * 60.0, 100.0) for i in range(12)]
    price = 100.0
    for i in range(12, 24):
        price *= 1.10
        track.append((float(i) * 60.0, price))
    series = {"FAKE-USDC": track}

    at_the_turn = trailing_volatility(series, "FAKE-USDC", 12 * 60.0, 1800.0)
    assert at_the_turn == pytest.approx(0.0), "the trailing window is flat and must read flat"
    # ... while the FORWARD window from that same instant is violent. A
    # predictor that read zero here and non-zero from the same timestamp
    # forward is the proof that no hindsight leaked in.
    assert track[-1][1] / track[11][1] > 3.0, "this fixture needs a violent forward window"
    later = trailing_volatility(series, "FAKE-USDC", 23 * 60.0, 1800.0)
    assert later is not None and later > 0.05, "after the move the trailing vol must rise"


def test_a_filter_that_doubles_the_clearing_share_is_not_an_edge():
    """The measured 2026-09-10 result, pinned as a fixture.

    The high-volatility subset lifts the share of ticks outrunning the fee
    from 27.6% to 48.8% -- real, and pure arithmetic. Inside it the head's
    top decile is still UP 2/8 and DOWN 1/4. The verdict must say the filter
    worked AND that it bought no direction, because reporting only the first
    half is how a filter gets shipped as a strategy.
    """
    result = {
        "horizon_sec": 900,
        "cost": 0.003592,
        "lookback_sec": 1800.0,
        "quantile": 2.0 / 3.0,
        "n": 5881,
        "vol_cut": 0.00164,
        "arms": [
            {"label": "ALL TICKS", "n": 5881, "share_clearing": 0.276,
             "regimes": {"UP": {"n_positive": 2, "n_windows": 5},
                         "DOWN": {"n_positive": 0, "n_windows": 8}}},
            {"label": "HIGH-VOL", "n": 1961, "share_clearing": 0.488,
             "regimes": {"UP": {"n_positive": 2, "n_windows": 8},
                         "DOWN": {"n_positive": 1, "n_windows": 4}}},
        ],
    }
    text = move_size_verdict(result)
    assert "THE FILTER WORKS AS A FILTER" in text
    assert "+21.2 points" in text
    assert "IT BUYS NO DIRECTION" in text
    assert "AND THE HEAD THEN PAYS IN BOTH REGIMES" not in text


def test_a_filter_that_pays_in_both_regimes_is_called_a_hypothesis():
    """The mirror. Even a clean both-regimes result must not read as shippable.

    Every apparent edge this repo has found died on re-measurement, so the
    verdict for a positive says HYPOTHESIS and demands fresh windows. A
    census that graduated its own result to "edge" would be how the next
    fake one ships.
    """
    result = {
        "horizon_sec": 900, "cost": 0.003592, "lookback_sec": 1800.0,
        "quantile": 2.0 / 3.0, "n": 100, "vol_cut": 0.001,
        "arms": [
            {"label": "ALL TICKS", "n": 100, "share_clearing": 0.30,
             "regimes": {"UP": {"n_positive": 1, "n_windows": 5},
                         "DOWN": {"n_positive": 0, "n_windows": 5}}},
            {"label": "HIGH-VOL", "n": 40, "share_clearing": 0.60,
             "regimes": {"UP": {"n_positive": 4, "n_windows": 5},
                         "DOWN": {"n_positive": 4, "n_windows": 5}}},
        ],
    }
    text = move_size_verdict(result)
    assert "PAYS IN BOTH REGIMES" in text
    assert "HYPOTHESIS, NOT AN EDGE" in text


def test_the_move_size_filter_reports_both_arms_on_one_tape():
    """End to end: the filtered arm must be a strict subset, scored the same way."""
    now = 2_000_000.0
    moves = ([0.004, -0.004] * 5 + [0.0002, -0.0002] * 5) * 6
    series, preds = _flat_tape_preds("FAKE-USDC", moves, now=now, step=60.0)
    result = move_size_filter(
        preds, series, now=now, hours=24.0, horizon_sec=120, cost=0.003592,
        lookback_sec=600.0, min_rows=5,
    )
    labels = [arm["label"] for arm in result["arms"]]
    assert labels == ["ALL TICKS", "HIGH-VOL"]
    base, filtered = result["arms"]
    assert 0 < filtered["n"] < base["n"], "the filter must actually exclude ticks"
    assert filtered["share_clearing"] >= base["share_clearing"], (
        "selecting the high-volatility tercile cannot lower the share of moves "
        "that outrun the fee"
    )
