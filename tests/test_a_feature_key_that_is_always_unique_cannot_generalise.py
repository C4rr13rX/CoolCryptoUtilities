"""The omen features must land similar bars in the SAME bin.

The 2026-09-07 omen run reported 100% train recall beside 26.6% held-out
accuracy against a 31.2% majority class, and buy signals that lost 0.94bp per
trade against buying every bar. The cause was measurable before any training:
``trading.omen_brain.build_collections`` produced **2725 distinct signatures
from 2725 training samples** on AERO-USDC. Its bucket boundaries were made
"deliberately fine-grained" so that collisions could not cap train recall --
which is exactly the objective that guarantees memorisation and forbids
generalisation. Every key unique means every held-out bar is a key the model
has never seen.

These tests pin the property that has to hold instead: a fitted bin must be
SHARED.

They fail against the fine-grained-constants approach, and by how much was
measured on the same 732-row synthetic corpus these tests build. Per
COLLECTION frame, distinct values / rows, and the largest bucket:

    temporal     732/732   largest bucket 1     0.1%
    geometry     719/732   largest bucket 4     0.5%
    cross        354/732   largest bucket 13    1.8%
    flow         246/732   largest bucket 13    1.8%
    volatility    29/732   largest bucket 183  25.0%
    JOINT        732/732

``test_a_fitted_bin_is_shared_by_many_bars`` demands a largest bin of at least
15% of rows. The temporal frame -- the pool the omen architecture leans on
hardest -- delivers 0.1%, missing that bar by 150x, and every one of its 732
frames is unique. Only ``volatility`` generalises at all. That is the whole
finding: four of the five sensory pools were incapable of saying anything
about a bar they had not already been shown.
"""
from __future__ import annotations

import math
import random

import pytest

from trading.omen_features import (
    FEATURE_NAMES, LOOKBACK_BARS, MISSING_BIN, digitize, features, fit_bins,
    signature,
)


def _corpus(count: int = 900, seed: int = 3):
    """A random-walk corpus with the fields the feature builder reads."""
    rng = random.Random(seed)
    bars, price, ts = [], 100.0, 1_700_000_000
    for _ in range(count):
        step = rng.gauss(0.0, 0.01)
        open_p = price
        price = max(1e-6, price * (1.0 + step))
        high = max(open_p, price) * (1.0 + abs(rng.gauss(0.0, 0.003)))
        low = min(open_p, price) * (1.0 - abs(rng.gauss(0.0, 0.003)))
        volume = abs(rng.gauss(1000.0, 400.0)) + 1.0
        buy = volume * rng.uniform(0.2, 0.8)
        bars.append({
            "timestamp": ts, "open": open_p, "high": high, "low": low,
            "close": price, "net_volume": volume,
            "buy_volume": buy, "sell_volume": volume - buy,
        })
        ts += 3600
    return bars


def _rows(bars):
    return [features(bars, i) for i in range(LOOKBACK_BARS, len(bars))]


def test_a_fitted_bin_is_shared_by_many_bars():
    """Each feature's bins must hold a population, not one bar each.

    This is the whole difference from the substrate frames. With 4 bins and
    ~730 rows every bin should hold roughly 180 bars; the failure this guards
    against is a binning so fine that a bin holds one.
    """
    bars = _corpus()
    rows = _rows(bars)
    table = fit_bins(rows, 4)
    binned = [digitize(row, table) for row in rows]

    for name in FEATURE_NAMES:
        occupancy = {}
        for row in binned:
            occupancy[row[name]] = occupancy.get(row[name], 0) + 1
        biggest = max(occupancy.values())
        assert biggest >= 0.15 * len(rows), (
            f"feature {name}: its largest bin holds {biggest} of {len(rows)} "
            f"rows -- bins this fine cannot carry a population to average")


def test_the_joint_key_over_every_feature_is_unique_and_so_cannot_be_learned():
    """The diagnosis itself, as a test: joint keys do NOT collide.

    Even at 3 bins per feature the joint signature across 27 features is
    essentially unique -- 3**27 cells for a few hundred bars. This is why the
    substrate's one-frame-to-one-outcome binding memorises: the fix is not a
    coarser joint key, it is a model that reads features SEPARATELY. If this
    test ever starts failing, a joint-key lookup has become viable and the
    additive estimator can be revisited.
    """
    bars = _corpus()
    rows = _rows(bars)
    table = fit_bins(rows, 3)
    keys = [signature(digitize(row, table)) for row in rows]
    assert len(set(keys)) > 0.9 * len(keys), (
        f"joint keys collided more than expected: {len(set(keys))} distinct "
        f"of {len(keys)}")


def test_a_flat_window_gets_a_missing_bin_and_never_a_numeric_one():
    """``None`` is a state, not a zero.

    A dead-flat window has no position inside its own range. Folding that into
    the bottom numeric bin would tell the model "price is at its low" about a
    bar where there is no low -- inventing the single most tradeable-looking
    reading out of an absence of data.
    """
    bars = _corpus()
    for bar in bars[-60:]:  # freeze the tail dead flat
        bar["open"] = bar["high"] = bar["low"] = bar["close"] = 50.0
        bar["net_volume"] = bar["buy_volume"] = bar["sell_volume"] = 0.0

    row = features(bars, len(bars) - 1)
    assert row["p24"] is None, "a flat window must have no range position"

    table = fit_bins(_rows(bars), 4)
    assert digitize(row, table)["p24"] == MISSING_BIN
    assert MISSING_BIN not in range(0, 64), "MISSING_BIN must not alias a real bin"


def test_features_read_only_the_past():
    """Truncating everything after the bar must not change its features.

    Every lookahead this repo has shipped was invisible in the output and
    obvious in this comparison.
    """
    bars = _corpus()
    index = len(bars) - 40
    full = features(bars, index)
    truncated = features(bars[: index + 1], index)
    for name in FEATURE_NAMES:
        a, b = full[name], truncated[name]
        if a is None or b is None:
            assert a is b, f"{name}: {a!r} vs {b!r} -- the future changed it"
        else:
            assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15), (
                f"{name}: {a} with the future present, {b} without -- "
                f"the feature is reading forward")


def test_a_short_window_is_refused_rather_than_padded():
    """A padded window is a different situation wearing the same key."""
    bars = _corpus()
    with pytest.raises(ValueError):
        features(bars, LOOKBACK_BARS - 1)
    with pytest.raises(IndexError):
        features(bars, len(bars))


def test_fit_bins_never_emits_a_duplicate_edge():
    """Repeated quantiles must collapse, not create a bin nothing can enter.

    A feature that is constant over most of the corpus (``streak`` on a quiet
    market) produces the same quantile many times over. Emitting it once per
    request would silently cut the model's capacity while the bin count still
    read as 20.
    """
    rows = [{name: (0.0 if name != "r1" else float(i % 3))
             for name in FEATURE_NAMES} for i in range(500)]
    table = fit_bins(rows, 20)
    for name in FEATURE_NAMES:
        edges = table.edges[name]
        assert list(edges) == sorted(set(edges)), (
            f"{name} has duplicate or unsorted edges: {edges}")
        assert table.bin_count(name) <= 20
