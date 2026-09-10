"""One row from another asset in a 60-bar window is the -1.2 ``price_mu``.

WHAT THIS PREVENTS, measured 2026-09-10 against the deployed
``models/active_model.keras`` (probe: ``scripts/model_window_probe.py``):

    clean live ETH-USDT window            price_mu -0.2065   net_margin -0.2130
    same window, one row multiplied 100x  price_mu -1.5111   net_margin -1.5176
    DRB window with one ETH row at t=30   price_mu -1.8395   net_margin -1.8460
    ETH and DRB rows interleaved          price_mu +1.3012   net_margin +1.2947

``organism_snapshots`` recorded ``price_mu`` p50 -1.2076 over 1613 live
predictions in the six hours to 09:00 that day, against a scheduler entry
conjunct of ``net_margin >= 0``. The contaminated windows reproduce that
number; the clean ones are an order of magnitude away from it. The model's
scale-free transform is NOT the defect -- probed at price levels 1e-4 and
1.2e4 it returns ``price_mu`` -0.000620 both times -- so a guard on the window
is what closes this, and it must not fire on an ordinary market move.

No TensorFlow here on purpose: the guard is arithmetic on a list of quotes and
must stay testable on the machines that run the gate without TF.
"""

import math

import pytest

from trading.data_loader import sanitize_model_price_window


def _window(level: float, step: float = 0.002, n: int = 60) -> list:
    return [level * (1.0 + step) ** i for i in range(n)]


def test_an_ordinary_market_move_is_left_exactly_alone():
    # The widest real window measured on this feed spans a log return of
    # 0.0632 (DRB-USDC). A +0.2%/step ramp spans 0.118 over 60 bars -- already
    # wider than anything the live feed produced -- and must still be untouched.
    window = _window(21.0)
    out, repaired = sanitize_model_price_window(window)
    assert repaired == 0
    assert out == pytest.approx(window, rel=1e-12)


def test_a_twenty_percent_crash_inside_the_window_is_a_market_not_a_fault():
    window = _window(2460.0, step=0.0)
    window[40:] = [p * 0.8 for p in window[40:]]
    out, repaired = sanitize_model_price_window(window)
    assert repaired == 0, "a 20% move is a market move and must survive the guard"
    assert out == pytest.approx(window, rel=1e-12)


def test_one_hundred_x_row_is_repaired_and_the_window_keeps_its_length():
    window = _window(2460.0)
    window[40] = window[40] * 100.0
    out, repaired = sanitize_model_price_window(window)
    assert repaired == 1
    assert len(out) == len(window), "the model reshapes on length; never shorten the window"
    # Carried forward from the last good row, not invented.
    assert out[40] == pytest.approx(window[39])


def test_a_row_from_another_asset_is_repaired_rather_than_anchored_on():
    # DRB-USDC trades at 2.4e-4 and ETH-USDT at 2.46e3: seven orders of
    # magnitude, which is a log return of ~16 once the window is anchored.
    window = _window(0.000238733)
    window[30] = 2460.57
    out, repaired = sanitize_model_price_window(window)
    assert repaired == 1
    assert out[30] == pytest.approx(window[29])
    anchor = out[0]
    assert max(abs(math.log(p / anchor)) for p in out) < 1.0, (
        "after the repair no bar may sit a factor of e away from the anchor"
    )


def test_a_leading_foreign_row_anchors_on_the_windows_own_median():
    window = _window(21.0)
    window[0] = 1e-9
    out, repaired = sanitize_model_price_window(window)
    assert repaired == 1
    # Nothing precedes bar 0, so the median of the window is the only honest
    # scale available. It must not be left at 1e-9, which is what divides the
    # whole window by ~0 inside PriceVolScaleNorm.
    assert out[0] > 1.0
    assert out[0] == pytest.approx(sorted(window[1:])[len(window[1:]) // 2], rel=0.05)


def test_zero_and_negative_quotes_are_repaired_not_passed_through():
    window = _window(21.0)
    window[10] = 0.0
    window[20] = -1.0
    out, repaired = sanitize_model_price_window(window)
    assert repaired == 2
    assert all(p > 0.0 for p in out)


def test_a_window_with_nothing_usable_is_returned_untouched():
    # There is no scale to anchor a repair on. Inventing one here would hand
    # the model a fabricated price, which is worse than the truth.
    window = [0.0, 0.0, -1.0]
    out, repaired = sanitize_model_price_window(window)
    assert repaired == 0
    assert out == [0.0, 0.0, -1.0]


def test_the_interleaved_two_asset_window_is_collapsed_onto_one_scale():
    eth = _window(2460.0)
    drb = _window(0.000238733)
    mixed = [eth[i] if i % 2 else drb[i] for i in range(60)]
    out, repaired = sanitize_model_price_window(mixed)
    assert repaired == 30, "half the rows belong to the other asset"
    anchor = next(p for p in out if p > 0.0)
    assert max(abs(math.log(p / anchor)) for p in out) < 1.0
