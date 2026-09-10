"""A single foreign price row must never reach the served model window.

WHAT THIS PREVENTS, AND WHY IT COST TWELVE HOURS OF THE GHOST LANE.

``model_definition.PriceVolScaleNorm`` makes the price channel scale-free
(``log(p_t / p_0)``), and it genuinely is: probed against the deployed
``models/active_model.keras``, a clean 60-bar window at price level 1e-4 and
at 1.2e4 produced ``price_mu`` -0.000620 both times, identical to six decimals
across eight orders of magnitude.

What being scale-free cannot absorb is a window whose rows are not all the
same asset. Scale-freedom is defined relative to the window's ANCHOR, so one
row from a differently-priced feed becomes a log return of ten or more and the
convolutions see a move no market makes. Measured against the same model, same
day, same live ETH-USDT window:

    clean ETH window                price_mu -0.2065   net_margin -0.2130
    ETH window, one row 100x        price_mu -1.5111   net_margin -1.5176
    DRB window, one ETH row t=30    price_mu -1.8395   net_margin -1.8460

ALL SIX HEADS COME OFF ONE MODEL CALL ON ONE TENSOR, so they do not fail
independently -- ``price_mu``, ``net_margin``, ``direction_prob`` and
``exit_conf`` saturate together. That is the whole mechanism behind the
collapse measured over 48h of ``organism_snapshots``: net_margin's MAXIMUM was
negative across ~3350 cycles and EVERY symbol for twelve hours, so the entry
conjunct ``net_margin >= 0`` at ``trading/scheduler.py:809`` could not fire for
any symbol at any price, and 1572 of 1574 cycles were holds.

``trading.data_loader.sanitize_model_price_window`` was written to repair this
and then sat UNWIRED -- its only callers in the tree were a probe script -- so
the guard existed while the contaminated window was served every tick. These
tests assert the guard is actually ON THE SERVING PATH, not merely importable.
They fail against the unwired ``_prepare_inputs``.
"""

from __future__ import annotations

import types

import numpy as np
import pytest


WINDOW_SIZE = 60
CLEAN_PRICE = 2500.0


def _stub() -> types.SimpleNamespace:
    """The collaborators ``_prepare_inputs`` actually touches, and nothing else.

    Called unbound against the real function so this cannot drift away from
    the served path without failing.
    """
    return types.SimpleNamespace(
        window_size=WINDOW_SIZE,
        pipeline=types.SimpleNamespace(
            sent_seq_len=12,
            tech_count=8,
            data_loader=types.SimpleNamespace(_get_asset_id=lambda _symbol: 0),
        ),
        _asset_vocab_limit=None,
    )


def _window(prices: list[float]) -> list[dict]:
    return [
        {
            "symbol": "ETH-USDC",
            "price": float(p),
            "volume": 1.0,
            "ts": 1_700_000_000 + i * 60,
        }
        for i, p in enumerate(prices)
    ]


def _served_prices(window: list[dict]) -> np.ndarray:
    from trading.bot import TradingBot

    served = TradingBot._prepare_inputs(_stub(), window)
    price_vol = np.asarray(served["price_vol_input"])
    assert price_vol.shape == (1, WINDOW_SIZE, 2), (
        f"price_vol_input is {price_vol.shape}, expected (1, {WINDOW_SIZE}, 2)"
    )
    return price_vol[0, :, 0].astype(np.float64)


def _max_adjacent_log_move(prices: np.ndarray) -> float:
    """The largest single-step log return the convolutions would see."""
    positive = prices[prices > 0.0]
    assert positive.size == prices.size, "a zero price reached the served window"
    return float(np.max(np.abs(np.diff(np.log(positive)))))


def test_a_hundred_x_foreign_row_is_repaired_before_it_is_served() -> None:
    """The 100x row that produced price_mu -1.5111 must not reach the tensor.

    Against the unwired ``_prepare_inputs`` the served channel carries the
    foreign row verbatim and this asserts ~4.6 against a bound of 1.0.
    """
    prices = [CLEAN_PRICE + i for i in range(WINDOW_SIZE)]
    clean_move = _max_adjacent_log_move(_served_prices(_window(prices)))

    prices[30] = prices[30] * 100.0
    served = _served_prices(_window(prices))

    assert CLEAN_PRICE * 100.0 not in set(served.tolist()), (
        "the foreign 100x row was served verbatim: sanitize_model_price_window "
        "is not on the serving path in trading/bot.py::_prepare_inputs"
    )

    dirty_move = _max_adjacent_log_move(served)
    assert dirty_move < 1.0, (
        f"served window still contains a {dirty_move:.4f} log move; a real "
        f"market does not move e^{dirty_move:.2f}x in one bar, and this is what "
        f"saturates all six heads (clean window moves {clean_move:.6f})"
    )


def test_the_repair_keeps_the_window_length_so_the_caller_sees_no_short_buffer() -> None:
    """Repair must carry forward, never drop.

    A short buffer is what ``_InsufficientHistory`` means, and it means
    something else entirely -- dropping rows here would raise that from the
    market-stream callback and kill the whole tick, ghost lane included.
    """
    prices = [CLEAN_PRICE + i for i in range(WINDOW_SIZE)]
    prices[10] = 1e-7
    prices[44] = prices[44] * 1000.0

    served = _served_prices(_window(prices))
    assert served.size == WINDOW_SIZE, (
        f"repair changed the window length to {served.size}; it must carry the "
        f"last good price forward, not drop the row"
    )


def test_a_clean_window_is_served_completely_unchanged() -> None:
    """The guard must not touch honest data.

    This is the half that stops the fix becoming its own bug: a filter that
    quietly rewrites good prices would be worse than the contamination.
    """
    prices = [CLEAN_PRICE * (1.0 + 0.001 * i) for i in range(WINDOW_SIZE)]
    served = _served_prices(_window(prices))

    np.testing.assert_allclose(
        served,
        np.asarray(prices, dtype=np.float32).astype(np.float64),
        rtol=1e-6,
        err_msg="the guard altered a clean window; it must only repair rows "
        "that are not a price of the same asset as the rest",
    )


def test_the_scale_free_channel_is_unchanged_across_orders_of_magnitude() -> None:
    """Two clean windows eight orders of magnitude apart must serve the same shape.

    Pins the premise the whole fix rests on: the saturation is a CONTAMINATED
    window, not the price level and not the units. If this ever fails, the
    diagnosis above is wrong and the guard is treating a symptom.
    """
    shape = [1.0 + 0.001 * i for i in range(WINDOW_SIZE)]
    cheap = _served_prices(_window([1e-4 * s for s in shape]))
    dear = _served_prices(_window([1.2e4 * s for s in shape]))

    cheap_returns = np.diff(np.log(cheap))
    dear_returns = np.diff(np.log(dear))
    np.testing.assert_allclose(
        cheap_returns,
        dear_returns,
        atol=1e-5,
        err_msg="the same shape at two price levels served different log "
        "returns; the price channel is not scale-free and the contaminated-"
        "window diagnosis needs re-deriving",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
