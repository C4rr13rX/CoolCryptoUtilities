"""The sell-high half must be scored beside the buy-low half, or not at all.

The failure this prevents
-------------------------
Every omen report in this repo scored buying. ``buy_omens``,
``buy_hit_rate``, ``buy_net_per_trade`` -- and no crest cell anywhere, so the
DOWN window's crest half (pass 114's true-label ceiling put it at +1.8884%
per trade against the buy half's +1.6396%) was never measured by any
experiment, in either direction. An absent number reads as a zero, and a
half-book that reads as a zero gets sized as free headroom.

These tests hold three things:

  1. ``pool_scoreboards`` exists and pools TOTALS, not per-corpus averages --
     the arithmetic that lets a 208-bar window be read at all. Averaging the
     per-trade means would let a 2-trade corpus outvote a 90-trade one.
  2. A pooled cell under the readability floor is UNREADABLE with its n, and
     pooling cannot launder two small cells into a quotable one unless their
     combined n actually clears the floor.
  3. ``omen_both_halves`` emits crest_omens, crest_net_per_trade and
     crest_precision beside the buy cells, for BOTH windows, and its
     predictor is causal -- the range position at a bar must not move when a
     future bar changes.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.omen_both_halves import (  # noqa: E402
    calls_for, fit_bands, range_position, score_corpus, _verdict,
)
from trading.omen_brain import OMEN_CREST, OMEN_MURK, OMEN_TROUGH  # noqa: E402
from trading.omen_scoreboard import (  # noqa: E402
    READABLE_TRADES, money_scoreboard, pool_scoreboards, render_scoreboard,
)


def _bars(closes, start=1_000_000, step=3600):
    return [{"timestamp": start + i * step, "close": c}
            for i, c in enumerate(closes)]


def _board(buy_n, buy_total, buy_paid, sell_n, sell_total, sell_paid,
           *, horizon=12, cost=0.0065, every_n=0, every_mean=0.0):
    """A scoreboard-shaped dict, hand-built so pooling can be tested alone."""
    return {
        "horizon_bars": horizon, "round_trip_cost": cost,
        "omen_threshold": cost * 1.5, "readable_trades_floor": READABLE_TRADES,
        "scored_bars": every_n, "dropped_no_future": 0,
        "buy": {"n": buy_n, "paid": buy_paid, "net_total": buy_total,
                "net_per_trade": (buy_total / buy_n) if buy_n else None,
                "precision_paid": (buy_paid / buy_n) if buy_n else None,
                "readable": buy_n >= READABLE_TRADES},
        "sell": {"n": sell_n, "paid": sell_paid, "net_total": sell_total,
                 "net_per_trade": (sell_total / sell_n) if sell_n else None,
                 "precision_paid": (sell_paid / sell_n) if sell_n else None,
                 "readable": sell_n >= READABLE_TRADES},
        "every_bar_n": every_n, "every_bar_net_per_trade": every_mean,
    }


# ---------------------------------------------------------------- pooling


def test_pooling_adds_totals_rather_than_averaging_per_trade_means():
    """A 2-trade corpus must not outvote a 90-trade one.

    Against the old tree there was no pooling function at all, so the only
    way an experiment could report a pooled cell was to average the
    per-corpus per-trade nets -- which is the mean of the means, and is wrong
    whenever the corpora carry different trade counts. Here the mean-of-means
    is +0.0255 and the honest pooled number is +0.0010.
    """
    small = _board(2, 0.10, 2, 0, 0.0, 0, every_n=2, every_mean=0.05)
    large = _board(98, -0.08, 40, 0, 0.0, 0, every_n=98, every_mean=-0.001)
    pooled = pool_scoreboards([small, large])
    assert pooled["buy"]["n"] == 100
    assert pooled["buy_net_per_trade"] == pytest.approx(0.02 / 100)
    mean_of_means = (small["buy"]["net_per_trade"]
                     + large["buy"]["net_per_trade"]) / 2
    assert not math.isclose(pooled["buy_net_per_trade"], mean_of_means,
                            rel_tol=1e-6)
    # Precision pools on the counts too, not on the rates.
    assert pooled["trough_precision"] == pytest.approx(42 / 100)


def test_pooling_refuses_boards_scored_at_different_horizons_or_costs():
    """Two horizons is two games; pooling them invents a number."""
    with pytest.raises(ValueError):
        pool_scoreboards([_board(30, 0.1, 15, 30, 0.1, 15, horizon=12),
                          _board(30, 0.1, 15, 30, 0.1, 15, horizon=24)])
    with pytest.raises(ValueError):
        pool_scoreboards([_board(30, 0.1, 15, 30, 0.1, 15, cost=0.0065),
                          _board(30, 0.1, 15, 30, 0.1, 15, cost=0.0100)])
    with pytest.raises(ValueError):
        pool_scoreboards([])


def test_a_pooled_cell_under_the_floor_is_unreadable_and_says_its_n():
    """Pooling small cells does not launder them into a quotable edge."""
    boards = [_board(4, 0.2, 4, 3, 0.1, 3, every_n=40, every_mean=0.001)
              for _ in range(3)]
    pooled = pool_scoreboards(boards)
    assert pooled["buy"]["n"] == 12 and pooled["sell"]["n"] == 9
    assert not pooled["buy"]["readable"] and not pooled["sell"]["readable"]
    rendered = render_scoreboard(pooled, title="pooled")
    for line in rendered.splitlines():
        if "%" in line and "per trade" in line:
            assert "n=" in line, f"a per-trade net with no n: {line!r}"
    assert rendered.count("UNREADABLE") == 2
    # And the moment the pooled n clears the floor, it is quotable -- the
    # floor is on the POOLED n, not on any one corpus's n. Nine 4-trade
    # corpora make a readable buy cell (36) while the sell cell (27) is still
    # below it, so the two halves are judged separately.
    big = pool_scoreboards(boards * 3)
    assert big["buy"]["n"] == 36 and big["buy"]["readable"]
    assert big["sell"]["n"] == 27 and not big["sell"]["readable"]


def test_the_crest_keys_survive_pooling():
    """crest_omens / crest_net_per_trade / crest_precision, with n."""
    pooled = pool_scoreboards([_board(40, 0.4, 25, 50, -0.5, 10,
                                      every_n=200, every_mean=0.002)])
    assert pooled["crest_omens"] == 50
    assert pooled["crest_net_per_trade"] == pytest.approx(-0.01)
    assert pooled["crest_precision"] == pytest.approx(0.2)
    assert pooled["buy_omens"] == 40


# ------------------------------------------------------------- the predictor


def test_the_range_position_predictor_cannot_see_the_future():
    """A bar's call must not move when a LATER bar changes.

    The labeller looks forward; a predictor may not. This is the whole
    difference between pass 114's ceiling (100% precision by construction)
    and a held-out number.
    """
    closes = [10.0, 11.0, 12.0, 11.5, 10.2, 13.0, 9.0, 9.5]
    before = range_position(_bars(closes), 4)
    tampered = list(closes)
    tampered[5] = 99.0
    tampered[7] = 0.1
    assert range_position(_bars(tampered), 4) == before


def test_a_dead_flat_window_is_never_called_a_trough():
    """A frozen feed republishing one price has no low to be at."""
    flat = _bars([7.0] * 40)
    assert range_position(flat, 30) is None
    calls = calls_for(flat, 25, 40, 0.2, 0.8)
    assert {label for _, label in calls} == {OMEN_MURK}


def test_every_bar_gets_a_call_so_the_baseline_is_the_whole_window():
    """The every-bar baseline must cover the window, not the liked bars."""
    closes = [10.0 + (i % 7) * 0.3 for i in range(60)]
    bars = _bars(closes)
    calls = calls_for(bars, 10, 50, 0.2, 0.8)
    assert [index for index, _ in calls] == list(range(10, 50))
    labels = {label for _, label in calls}
    assert labels <= {OMEN_TROUGH, OMEN_CREST, OMEN_MURK}
    board = money_scoreboard(bars, calls, horizon_bars=3)
    assert board["every_bar_n"] == board["scored_bars"]


def test_the_bands_are_fitted_on_train_only_and_both_halves_get_their_own():
    """Each half fits its own band; neither inherits the other's."""
    closes = []
    for i in range(900):
        closes.append(100.0 + 8.0 * math.sin(i / 6.0) + i * 0.01)
    bars = _bars(closes)
    fitted = fit_bands(bars, 100, 700, horizon=6, cost=0.0065, multiple=1.5)
    assert fitted["low"]["band"] is not None
    assert fitted["high"]["band"] is not None
    # A band that fired too few times on train is not carried forward.
    assert fitted["low"]["n"] >= 20 and fitted["high"]["n"] >= 20
    # The fit is on the TRAIN range: moving bars AFTER train_stop must not
    # move the chosen bands.
    tampered = list(closes)
    for i in range(720, 900):
        tampered[i] = 500.0
    again = fit_bands(_bars(tampered), 100, 700, horizon=6, cost=0.0065,
                      multiple=1.5)
    assert again["low"]["band"] == fitted["low"]["band"]
    assert again["high"]["band"] == fitted["high"]["band"]


# ----------------------------------------------------------- the experiment


def test_the_experiment_reports_both_halves_in_both_windows(tmp_path):
    """crest cells beside buy cells, UP and DOWN, with n on every cell."""
    closes = []
    price = 100.0
    for i in range(1400):
        # A long rise then a long fall, so the corpus admits both regimes.
        drift = 0.0012 if i < 700 else -0.0012
        price *= (1.0 + drift)
        closes.append(price * (1.0 + 0.02 * math.sin(i / 5.0)))
    path = tmp_path / "SYNTH-USDC.json"
    path.write_text(__import__("json").dumps(_bars(closes)), encoding="utf-8")

    scored = score_corpus(path, horizon_minutes=720.0, test=208, train=600,
                          cadence_filter=3600, cost=0.0065, multiple=1.5)
    assert scored is not None
    assert scored["horizon_bars"] == 12 and scored["bar_seconds"] == 3600
    measured = [w for w in ("UP", "DOWN") if scored.get(w)]
    assert measured, "neither regime was found in a corpus built to hold both"
    for window in measured:
        board = scored[window]["board"]
        for key in ("buy_omens", "buy_net_per_trade", "trough_precision",
                    "crest_omens", "crest_net_per_trade", "crest_precision"):
            assert key in board, f"{window} board is missing {key}"
        assert board["buy"]["n"] == board["buy_omens"]
        assert board["sell"]["n"] == board["crest_omens"]
        # Held out: the test window starts after the train window's future.
        window_plan = scored[window]["window"]
        assert (window_plan["train_stop"] + scored["horizon_bars"]
                <= window_plan["test_start"])


def test_the_verdict_never_quotes_a_cell_below_the_floor():
    """An unreadable cell is named UNREADABLE with its n, never as a rate."""
    thin = pool_scoreboards([_board(5, 0.5, 5, 5, 0.5, 5,
                                    every_n=50, every_mean=0.001)])
    thin["corpora"] = 1
    text = _verdict({"UP": thin, "DOWN": None})
    assert "UNREADABLE" in text and "n=5" in text
    for line in text.splitlines():
        if "per trade" in line:
            assert "n=" in line
    assert "DOWN: no window" in text
