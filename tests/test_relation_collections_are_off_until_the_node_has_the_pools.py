"""The relation streams must not reach a node that has no pools for them.

Named after the failure it prevents. Pools 12/13/14 exist only in
brains/market_predictor_v3_assoc.identity.toml. A v2 node declares 11, and
sending it pool 12 does NOT degrade gracefully -- the node answers
``unknown input pool id 12`` with ``consolidated: False``, and OmenBrain
counts the whole sample as a miss. So switching the relations on against the
wrong node silently stops training rather than weakening it, which is why the
default is off and why that default is worth a test.

The second half of the file tests the frames themselves: a relation that
buckets to a constant is a diluting stream (the dilution law), so a stream
that never varies is a bug even when it is wired correctly.
"""
from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import trading.omen_brain as omen_brain  # noqa: E402


def _reload(monkeypatch, value):
    """Re-import the module with OMEN_RELATION_COLLECTIONS set to ``value``."""
    if value is None:
        monkeypatch.delenv("OMEN_RELATION_COLLECTIONS", raising=False)
    else:
        monkeypatch.setenv("OMEN_RELATION_COLLECTIONS", value)
    return importlib.reload(omen_brain)


@pytest.fixture(autouse=True)
def _restore():
    """Leave the module in its default state whatever a test did to it."""
    yield
    os.environ.pop("OMEN_RELATION_COLLECTIONS", None)
    importlib.reload(omen_brain)


def _bars(n=260, *, trend=0.0, vol=0.004, seed=11):
    """Deterministic OHLCV. No RNG import: a fixed recurrence keeps the
    bars identical across runs, so a distinctness assertion cannot flake."""
    bars = []
    price = 100.0
    state = seed
    for i in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        wobble = ((state / 2147483648.0) - 0.5) * 2.0
        price *= 1.0 + trend + vol * wobble
        high = price * (1.0 + abs(wobble) * vol)
        low = price * (1.0 - abs(wobble) * vol)
        buy = 1000.0 * (1.0 + wobble)
        bars.append({
            "open": price * (1.0 - trend), "high": high, "low": low,
            "close": price, "net_volume": 1000.0 + 500.0 * wobble,
            "buy_volume": max(1.0, buy), "sell_volume": max(1.0, 2000.0 - buy),
        })
    return bars


# --- the wiring -----------------------------------------------------------

def test_relation_collections_are_absent_by_default(monkeypatch):
    """Default must be the v2-safe 7. This is the regression that matters:
    a bare import must never widen what gets sent to a node."""
    mod = _reload(monkeypatch, None)
    names = [c.name for c in mod.COLLECTIONS]
    assert names == ["geometry", "temporal", "flow", "volatility",
                     "cross", "horizon", "instrument"]
    for relation in ("rel_move_vol", "rel_shape_flow", "rel_trend_noise"):
        assert relation not in names


def test_relations_are_added_only_when_explicitly_enabled(monkeypatch):
    mod = _reload(monkeypatch, "1")
    names = [c.name for c in mod.COLLECTIONS]
    assert names[:7] == ["geometry", "temporal", "flow", "volatility",
                         "cross", "horizon", "instrument"]
    assert names[7:] == ["rel_move_vol", "rel_shape_flow", "rel_trend_noise"]
    assert [c.pool_id for c in mod.COLLECTIONS[7:]] == [12, 13, 14]


def test_a_disabled_relation_is_never_streamed_to_the_node(monkeypatch):
    """The seam that would actually break training: _streams must not emit a
    pool the node does not have."""
    mod = _reload(monkeypatch, None)
    frames = mod.build_collections(_bars(), 250, horizon_bars=6, bar_seconds=3600, symbol="TEST")
    brain = mod.OmenBrain.__new__(mod.OmenBrain)
    streams = mod.OmenBrain._streams(brain, frames)
    pool_ids = {s["pool_id"] for s in streams}
    assert pool_ids.isdisjoint({12, 13, 14}), (
        f"relation pools leaked to a node that may not have them: {pool_ids}")


@pytest.mark.parametrize("enabled", [None, "1"])
def test_what_is_built_is_exactly_what_is_streamed(monkeypatch, enabled):
    """The invariant test_every_collection_has_its_own_byte_prefix asserts,
    checked in BOTH states. One flag must gate the frame keys and the
    COLLECTIONS entries together: if they can disagree, either a pool gets
    nothing or work is done for a pool that will never receive it."""
    mod = _reload(monkeypatch, enabled)
    frames = mod.build_collections(_bars(), 250, horizon_bars=6, bar_seconds=3600, symbol="TEST")
    assert set(frames) == {c.name for c in mod.COLLECTIONS}


def test_enabling_relations_streams_the_new_pools(monkeypatch):
    mod = _reload(monkeypatch, "1")
    frames = mod.build_collections(_bars(), 250, horizon_bars=6, bar_seconds=3600, symbol="TEST")
    brain = mod.OmenBrain.__new__(mod.OmenBrain)
    streams = mod.OmenBrain._streams(brain, frames)
    assert {12, 13, 14} <= {s["pool_id"] for s in streams}


# --- the frames -----------------------------------------------------------

def test_every_relation_frame_carries_its_own_byte_prefix(monkeypatch):
    """Atoms are bytes, so the streams must not share a prefix or they bind
    into each other -- the same reason labels must be byte-disjoint."""
    mod = _reload(monkeypatch, "1")
    frames = mod.build_collections(_bars(), 250, horizon_bars=6, bar_seconds=3600, symbol="TEST")
    assert frames["rel_move_vol"].startswith("rmv ")
    assert frames["rel_shape_flow"].startswith("rsf ")
    assert frames["rel_trend_noise"].startswith("rtn ")
    prefixes = {f.split(" ", 1)[0] for f in frames.values()}
    assert len(prefixes) == len(frames), f"prefix collision: {prefixes}"


def test_relation_frames_vary_across_market_states(monkeypatch):
    """A relation that buckets to a constant carries nothing and would only
    dilute the query. Distinctness must be well above 1/n."""
    mod = _reload(monkeypatch, "1")
    seen = {"rel_move_vol": set(), "rel_shape_flow": set(),
            "rel_trend_noise": set()}
    samples = 0
    for trend, vol, seed in ((0.0, 0.004, 11), (0.004, 0.002, 29),
                             (-0.003, 0.010, 7)):
        bars = _bars(trend=trend, vol=vol, seed=seed)
        for index in range(200, 260, 6):
            frames = mod.build_collections(bars, index, horizon_bars=6, bar_seconds=3600,
                                           symbol="TEST")
            samples += 1
            for key in seen:
                seen[key].add(frames[key])
    for key, values in seen.items():
        assert len(values) > 1, f"{key} is a constant over {samples} samples"
        assert len(values) / samples > 0.1, (
            f"{key} distinctness {len(values)}/{samples} is near-constant")


def test_relations_are_dimensionless_not_raw_returns(monkeypatch):
    """The z-score fields must respond to VOLATILITY, not just to the move.

    Same drift at 20x the noise must bucket differently -- that is the whole
    point of the relation, and it is what a flat sibling pool cannot express.

    Measured honestly: this asserts the NORMALISATION happens (the divide by
    vol), not that _bucket_signed was the right bucketer. On this pair
    _bucket_return separates it too (u24 vs u18). The bucketer choice is
    justified by resolution instead, which the next test measures.
    """
    mod = _reload(monkeypatch, "1")
    quiet = mod.build_collections(_bars(trend=0.002, vol=0.001, seed=3),
                                  250, horizon_bars=6, bar_seconds=3600, symbol="TEST")
    noisy = mod.build_collections(_bars(trend=0.002, vol=0.020, seed=3),
                                  250, horizon_bars=6, bar_seconds=3600, symbol="TEST")
    assert quiet["rel_move_vol"] != noisy["rel_move_vol"], (
        "identical drift at 20x the volatility produced the same relation "
        "frame -- the normalisation is not happening")


def test_bucket_signed_spreads_the_band_a_z_score_lives_in():
    """_bucket_return puts z in [0.5, 3] into ~4 adjacent levels; the signed
    bucketer must do better or the stream is coarse where it matters."""
    signed = {omen_brain._bucket_signed(z / 10.0)
              for z in range(5, 31)}          # z = 0.5 .. 3.0
    returns = {omen_brain._bucket_return(z / 10.0) for z in range(5, 31)}
    assert len(signed) > len(returns), (
        f"signed {len(signed)} levels vs return {len(returns)} -- no gain")
    assert omen_brain._bucket_signed(0.0) != omen_brain._bucket_signed(2.0)
    assert omen_brain._bucket_signed(-2.0) != omen_brain._bucket_signed(2.0)
    assert omen_brain._bucket_signed(None) == "na"
    assert omen_brain._bucket_signed(math.nan) == "na"
    # Saturation is clamped, not wrapped: a huge z must stay at the top.
    assert omen_brain._bucket_signed(99.0) == omen_brain._bucket_signed(1e9)
