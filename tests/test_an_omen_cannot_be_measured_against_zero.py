"""The omen label set must be drawn at COST, never at zero.

Every one of these pins a shape that has already lost money in this repo:

  * a forward return compared against zero instead of against the round
    trip (``services/profit_logic_audit`` flags exactly this);
  * a label whose token is a SUBSTRING of another, so the frequent class
    swallows the rare one on a byte decode (measured 2026-07-09: every
    recall miss was ``loss_big`` decoding as ``loss``);
  * a "no opinion" answer that still carries a tradeable action;
  * a feature builder that pads short history instead of refusing, which
    silently creates a second atom for the same situation;
  * a feature builder that reads a bar it is supposed to be predicting.

None of these needs a running node.
"""
from __future__ import annotations

import math

import pytest

from trading.omen_brain import (
    COLLECTIONS, LOOKBACK_BARS, OMEN_ACTIONS, OMEN_CREST, OMEN_CLIMB,
    OMEN_DIRECTION, OMEN_LABELS, OMEN_MURK, OMEN_SLIDE, OMEN_TROUGH,
    REGIME_TOKENS, ROUND_TRIP_COST, SCHEMA_VERSION, Omen, build_collections,
    label_omen, label_regime, omen_frame, omen_threshold, parse_omen,
)


def _bars(closes, *, start_ts=1_700_000_000, step=3600):
    """Minimal OHLCV bars from a close series."""
    return [
        {
            "timestamp": start_ts + i * step,
            "open": c, "high": c * 1.001, "low": c * 0.999, "close": c,
            "net_volume": 1000.0, "buy_volume": 600.0, "sell_volume": 400.0,
        }
        for i, c in enumerate(closes)
    ]


# --- the cost bar ---------------------------------------------------------

def test_threshold_is_never_zero_and_never_below_the_round_trip():
    assert omen_threshold() > 0.0
    assert omen_threshold() >= ROUND_TRIP_COST, (
        "an omen that clears less than one round trip is a loss dressed as "
        "a signal")


def test_a_move_smaller_than_the_round_trip_is_murk_not_a_buy():
    """The whole point of the label set.

    A +0.30% forward move on a 0.65% round trip is a LOSS. Against a
    zero threshold it would be labelled a buy-low and would teach the
    substrate to lose money on every recurrence.
    """
    # 168 flat bars of history, then a dip to make the entry a low, then a
    # forward move of +0.30% -- real, positive, and not enough.
    closes = [100.0] * LOOKBACK_BARS + [99.0] + [99.0 * 1.003] * 12
    bars = _bars(closes)
    index = LOOKBACK_BARS
    forward = (closes[index + 12] - closes[index]) / closes[index]
    assert 0 < forward < ROUND_TRIP_COST, "fixture must be a real but unpaid move"
    assert label_omen(bars, index, horizon_bars=12) == OMEN_MURK


def test_a_move_that_clears_the_cost_from_a_low_is_a_trough():
    closes = [100.0] * LOOKBACK_BARS + [99.0] + [99.0 * 1.05] * 12
    bars = _bars(closes)
    assert label_omen(bars, LOOKBACK_BARS, horizon_bars=12) == OMEN_TROUGH


def test_a_move_that_clears_the_cost_downward_from_a_high_is_a_crest():
    closes = [100.0] * LOOKBACK_BARS + [101.0] + [101.0 * 0.95] * 12
    bars = _bars(closes)
    assert label_omen(bars, LOOKBACK_BARS, horizon_bars=12) == OMEN_CREST


def test_an_up_move_from_mid_range_is_a_climb_not_a_buy_low():
    """buy-LOW is the ask. An up-move already under way is not one."""
    closes = list(range(100, 100 + LOOKBACK_BARS))  # strictly rising: at the high
    closes = [float(c) for c in closes]
    closes += [closes[-1] * 1.05] * 12
    bars = _bars(closes)
    label = label_omen(bars, LOOKBACK_BARS - 1, horizon_bars=12)
    assert label == OMEN_CLIMB
    assert OMEN_ACTIONS[label] == "hold", "a climb must not open a position"


def test_a_missing_future_is_none_not_murk():
    """A sample whose future is off the end of the corpus must be dropped.

    Labelling it ``murk`` would train the substrate that the end of every
    data file is a quiet market.
    """
    bars = _bars([100.0] * (LOOKBACK_BARS + 5))
    assert label_omen(bars, len(bars) - 2, horizon_bars=12) is None


# --- byte-disjointness ----------------------------------------------------

def test_no_omen_token_is_a_substring_of_another():
    for a in OMEN_LABELS:
        for b in OMEN_LABELS:
            if a is not b:
                assert a not in b, (
                    f"{a!r} is a substring of {b!r} -- a byte decode of the "
                    f"frequent class would be read as the rare one")


def test_no_regime_token_can_be_read_as_an_omen():
    """Stage 1 and stage 2 decode from different pools, but a leaked frame
    must still be unreadable as the other stage's answer."""
    for regime in REGIME_TOKENS:
        assert parse_omen(regime) is None, (
            f"regime token {regime!r} parses as omen {parse_omen(regime)!r}")


def test_no_collection_prefix_collides_with_an_omen_token():
    for collection in COLLECTIONS:
        assert parse_omen(collection.prefix) is None
        assert parse_omen(collection.name) is None


def test_parse_omen_round_trips_every_label():
    for label in OMEN_LABELS:
        assert parse_omen(omen_frame(label)) == label
    assert parse_omen(None) is None
    assert parse_omen("") is None


# --- the schema -----------------------------------------------------------

def _omen(**overrides):
    base = dict(
        schema_version=SCHEMA_VERSION, symbol="AERO-USDC", chain="base",
        as_of_ts=1_700_000_000, price=1.0, horizon_bars=12, bar_seconds=3600,
        omen=OMEN_TROUGH, action="buy", confidence=0.5,
        cost_fraction=ROUND_TRIP_COST, threshold_fraction=omen_threshold(),
        expected_move_fraction=omen_threshold(), verdict="admitted",
    )
    base.update(overrides)
    return Omen(**base)


def test_an_unadmitted_omen_cannot_carry_a_tradeable_action():
    """The one invariant that keeps a broken brain off the money path."""
    for verdict in ("below_floor", "no_answer", "degenerate",
                    "transport_error", "not_trained", "unsupported_horizon"):
        with pytest.raises(ValueError):
            _omen(verdict=verdict, action="buy")
        held = _omen(verdict=verdict, action="hold", omen=OMEN_MURK,
                     confidence=0.0, expected_move_fraction=0.0)
        assert held.is_actionable is False


def test_a_zero_threshold_omen_is_refused_at_construction():
    with pytest.raises(ValueError):
        _omen(threshold_fraction=0.0)
    with pytest.raises(ValueError):
        _omen(threshold_fraction=-0.01)


def test_the_schema_refuses_an_unknown_label_or_verdict():
    with pytest.raises(ValueError):
        _omen(omen="moon")
    with pytest.raises(ValueError):
        _omen(verdict="probably")
    with pytest.raises(ValueError):
        _omen(confidence=1.5)


def test_expected_move_matches_the_direction_the_label_claims():
    for label in OMEN_LABELS:
        direction = OMEN_DIRECTION[label]
        action = OMEN_ACTIONS[label]
        if action == "buy":
            assert direction > 0
        elif action == "sell":
            assert direction < 0
    assert OMEN_DIRECTION[OMEN_MURK] == 0


# --- the feature builder --------------------------------------------------

def test_short_history_is_refused_rather_than_padded():
    bars = _bars([100.0] * (LOOKBACK_BARS + 2))
    with pytest.raises(ValueError):
        build_collections(bars, LOOKBACK_BARS - 1, horizon_bars=12, bar_seconds=3600,
                          symbol="X-USDC")


def test_collections_never_read_a_future_bar():
    """Truncating the corpus after the queried bar must not change a frame.

    If any collection peeked forward, the two frames would differ and every
    held-out number this module produces would be fiction.
    """
    closes = [100.0 + math.sin(i / 7.0) for i in range(LOOKBACK_BARS + 60)]
    bars = _bars(closes)
    index = LOOKBACK_BARS + 20
    full = build_collections(bars, index, horizon_bars=12, bar_seconds=3600, symbol="X-USDC")
    truncated = build_collections(bars[: index + 1], index, horizon_bars=12, bar_seconds=3600,
                                  symbol="X-USDC")
    assert full == truncated


def test_every_collection_has_its_own_byte_prefix():
    closes = [100.0 + math.sin(i / 5.0) for i in range(LOOKBACK_BARS + 10)]
    frames = build_collections(_bars(closes), LOOKBACK_BARS + 5,
                               horizon_bars=12, bar_seconds=3600, symbol="X-USDC")
    assert set(frames) == {c.name for c in COLLECTIONS}
    for collection in COLLECTIONS:
        assert frames[collection.name].startswith(collection.prefix + " ")
    prefixes = [c.prefix for c in COLLECTIONS]
    assert len(set(prefixes)) == len(prefixes), "prefixes must be unique"


def test_the_horizon_is_a_feature_not_a_hidden_constant():
    """Two horizons must produce different frames.

    'Buy low and sell high AT A LATER DATE' is a question about a specific
    later date. If the horizon did not reach the substrate, every horizon
    would train the same binding and the last one written would win.
    """
    closes = [100.0 + math.sin(i / 5.0) for i in range(LOOKBACK_BARS + 10)]
    bars = _bars(closes)
    short = build_collections(bars, LOOKBACK_BARS + 5, horizon_bars=6, bar_seconds=3600,
                              symbol="X-USDC")
    long = build_collections(bars, LOOKBACK_BARS + 5, horizon_bars=48, bar_seconds=3600,
                             symbol="X-USDC")
    assert short["horizon"] != long["horizon"]
    assert short["geometry"] == long["geometry"], (
        "only the horizon collection may change with the horizon")


def test_different_symbols_produce_different_instrument_frames():
    closes = [100.0 + math.sin(i / 5.0) for i in range(LOOKBACK_BARS + 10)]
    bars = _bars(closes)
    a = build_collections(bars, LOOKBACK_BARS + 5, horizon_bars=12, bar_seconds=3600, symbol="AERO-USDC")
    b = build_collections(bars, LOOKBACK_BARS + 5, horizon_bars=12, bar_seconds=3600, symbol="VVV-WETH")
    assert a["instrument"] != b["instrument"]


# --- the chain ------------------------------------------------------------

def test_stage_one_regime_never_reads_the_future():
    """The chain's first stage summarises the PAST.

    A stage-1 label that peeked forward would hand stage 2 the answer and
    every held-out number would be leakage.
    """
    closes = [100.0 + math.sin(i / 9.0) for i in range(LOOKBACK_BARS + 60)]
    bars = _bars(closes)
    index = LOOKBACK_BARS + 20
    assert label_regime(bars, index) == label_regime(bars[: index + 1], index)


def test_regime_tokens_are_pairwise_non_containing():
    for a in REGIME_TOKENS:
        for b in REGIME_TOKENS:
            if a is not b:
                assert a not in b
