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
