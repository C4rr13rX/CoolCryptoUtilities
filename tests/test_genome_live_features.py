"""Live genome scoring must reuse the GA's feature builders, or abstain.

A genome evolved against one feature distribution and scored against a
different one is a model nobody validated. These tests pin the contract:
features come from the GA repo, incomplete vectors are refused, and a new
champion never inherits the incumbent's ghost record.
"""
from __future__ import annotations

import math

import pytest

from trading.genome import LiveFeatureBuilder, load_champion
from trading.genome.champion import ChampionGenome, champion_meets_objective
from trading.genome.features import (
    GENOME_REPO_AVAILABLE,
    REQUIRED_HISTORY_BARS,
    normalize_bar,
)


def make_bars(seed: int, count: int = 200):
    """Bars in the shape continuous_features requires."""
    out = []
    price = 100.0
    for i in range(count):
        price *= 1.0 + 0.002 * math.sin((i + seed) / 7.0)
        volume = 1000.0 + 10.0 * i
        buy = volume * (0.5 + 0.1 * math.sin((i + seed) / 5.0))
        out.append({
            "timestamp": 1_700_000_000 + i * 3600,
            "open": price * 0.999, "high": price * 1.004,
            "low": price * 0.996, "close": price,
            "volume": volume, "buy_volume": buy, "sell_volume": volume - buy,
        })
    return out


def test_short_history_abstains():
    """Fewer than 168 bars cannot produce the rolling windows."""
    short = {"BTC": make_bars(0, REQUIRED_HISTORY_BARS - 1)}
    assert LiveFeatureBuilder("BTC").build(short) == {}


def test_missing_reference_asset_abstains():
    """Reference returns are required; without BTC there is no baseline."""
    builder = LiveFeatureBuilder("BTC")
    assert builder.build({"ETH": make_bars(3)}) == {}


@pytest.mark.skipif(not GENOME_REPO_AVAILABLE,
                    reason="W1z4rDV1510n repo not importable")
def test_builds_cross_sectional_features():
    """Breadth features are medians across the universe at one timestamp."""
    data = {"BTC": make_bars(0), "ETH": make_bars(3), "SOL": make_bars(7)}
    features = LiveFeatureBuilder("BTC").build(data)
    assert set(features) == {"BTC", "ETH", "SOL"}
    for values in features.values():
        assert "market_median_r6" in values
        assert "r12" in values


@pytest.mark.skipif(not GENOME_REPO_AVAILABLE,
                    reason="W1z4rDV1510n repo not importable")
def test_the_live_champion_is_fully_scorable():
    """The whole point: every feature the champion needs must build.

    Previously only 29 of 41 built, because the builder ran market breadth
    but skipped the news, derived and causal-normalisation passes that
    load_dataset() applies. A partial vector was refused, so the champion
    could never ghost-trade.
    """
    champion = load_champion()
    if champion is None:
        pytest.skip("no champion.json present")
    data = {"BTC": make_bars(0), "ETH": make_bars(3), "SOL": make_bars(7)}
    features = LiveFeatureBuilder("BTC").build(data)
    assert features, "builder produced nothing"
    missing = champion.missing_features(features["BTC"])
    assert not missing, f"champion cannot be scored, missing: {missing}"
    assert champion.is_scorable(features["BTC"])


@pytest.mark.skipif(not GENOME_REPO_AVAILABLE,
                    reason="W1z4rDV1510n repo not importable")
def test_news_features_are_attached():
    """News sentiment must come from the archive the GA fitted against."""
    data = {"BTC": make_bars(0), "ETH": make_bars(3)}
    features = LiveFeatureBuilder("BTC").build(data)
    assert features
    for name in ("news_count_24h", "news_polarity_24h",
                 "asset_news_sentiment_24h", "news_macro_24h"):
        assert name in features["BTC"], f"{name} was never attached"


@pytest.mark.skipif(not GENOME_REPO_AVAILABLE,
                    reason="W1z4rDV1510n repo not importable")
def test_derived_and_causal_features_are_attached():
    """Derived/rolling features depend on breadth+news running first."""
    data = {"BTC": make_bars(0), "ETH": make_bars(3), "SOL": make_bars(7)}
    features = LiveFeatureBuilder("BTC").build(data)
    assert features
    for name in ("vol_adjusted_r6", "flow_basis_pressure",
                 "causal_z14_market_breadth_r6", "cross_rank_funding_rate"):
        assert name in features["BTC"], f"{name} was never attached"


@pytest.mark.skipif(not GENOME_REPO_AVAILABLE,
                    reason="W1z4rDV1510n repo not importable")
def test_ohlcv_without_order_flow_is_refused():
    """buy_volume/sell_volume are required, not optional."""
    plain = {}
    for asset, seed in (("BTC", 0), ("ETH", 3)):
        bars = make_bars(seed)
        for bar in bars:
            bar.pop("buy_volume", None)
            bar.pop("sell_volume", None)
        plain[asset] = bars
    # The builder swallows per-asset failures, so the result is empty rather
    # than a partial vector.
    assert LiveFeatureBuilder("BTC").build(plain) == {}


def test_incomplete_feature_vector_is_not_scorable():
    """A genome must refuse to score when any feature is absent."""
    genome = ChampionGenome(genome_id="abc123", features=["r2", "r12", "rv24"])
    assert genome.is_scorable({"r2": 0.1, "r12": 0.2, "rv24": 0.3})
    assert not genome.is_scorable({"r2": 0.1, "r12": 0.2})
    assert genome.missing_features({"r2": 0.1}) == ["r12", "rv24"]


def test_strategy_id_is_genome_specific():
    """A new champion must earn its own ghost record, not inherit one."""
    first = ChampionGenome(genome_id="aaaaaaaaaaaa1111", features=["r2"])
    second = ChampionGenome(genome_id="bbbbbbbbbbbb2222", features=["r2"])
    assert first.strategy_id != second.strategy_id
    assert first.strategy_id.startswith("genome_")


def test_objective_requires_measured_profit():
    """Backtest profit only qualifies a genome when fully measured."""
    good = ChampionGenome(genome_id="x", features=["r2"], profit_factor=1.12,
                          evaluated_folds=3, expectancy=0.001)
    thin = ChampionGenome(genome_id="x", features=["r2"], profit_factor=1.40,
                          evaluated_folds=1, expectancy=0.004)
    losing = ChampionGenome(genome_id="x", features=["r2"], profit_factor=1.12,
                            evaluated_folds=3, expectancy=-0.001)
    assert champion_meets_objective(good)
    assert not champion_meets_objective(thin), "single-fold luck must not qualify"
    assert not champion_meets_objective(losing)


def test_champion_loads_from_the_live_ga_state():
    """The strategy tracks whichever genome the GA currently favours."""
    champion = load_champion()
    if champion is None:
        pytest.skip("no champion.json present")
    assert champion.genome_id
    assert champion.features
    assert champion.strategy_id.startswith("genome_")


def test_total_volume_is_buy_plus_sell_not_net():
    """net_volume is a SIGNED imbalance and must never stand in for volume.

    The trading store writes buy_volume/sell_volume/net_volume but no total
    volume. Substituting net_volume would invert every volume-derived
    feature on sell-heavy bars.
    """
    bar = normalize_bar({
        "timestamp": 1_700_000_000, "open": 1.0, "high": 1.1,
        "low": 0.9, "close": 1.05,
        "buy_volume": 30.0, "sell_volume": 70.0, "net_volume": -40.0,
    })
    assert bar is not None
    assert bar["volume"] == 100.0, "volume must be buy + sell"


def test_explicit_volume_is_preserved():
    bar = normalize_bar({
        "timestamp": 1_700_000_000, "open": 1.0, "high": 1.1, "low": 0.9,
        "close": 1.05, "volume": 250.0, "buy_volume": 1.0, "sell_volume": 2.0,
    })
    assert bar is not None and bar["volume"] == 250.0


def test_malformed_bars_are_dropped_not_guessed():
    assert normalize_bar({"timestamp": 1, "open": 1.0}) is None
    assert normalize_bar({"timestamp": 0, "open": 1.0, "high": 1.0,
                          "low": 1.0, "close": 1.0}) is None
    assert normalize_bar({"timestamp": 1, "open": 1.0, "high": 1.0,
                          "low": 1.0, "close": 0.0}) is None

def test_the_reference_asset_resolves_to_what_the_corpus_carries():
    """A missing anchor makes build() abstain on every asset, silently.

    The GA anchors cross-asset features on WBTC (falling back to WETH) and the
    corpus carries those wrapped names. The live builder defaulted to "BTC",
    which is never present, so every tick returned zero signals in 0.0s -- a
    result indistinguishable from "not enough history yet". That is why the
    champion never produced a ghost trade.
    """
    builder = LiveFeatureBuilder()
    assert builder._resolve_reference({"WBTC": [], "AAVE": []}) == "WBTC"
    # WETH is the documented fallback when WBTC is absent.
    assert builder._resolve_reference({"WETH": [], "AAVE": []}) == "WETH"
    # An explicit choice wins when the universe actually carries it...
    assert LiveFeatureBuilder("AAVE")._resolve_reference({"AAVE": [], "WBTC": []}) == "AAVE"
    # ...but must not silently anchor on an asset that is missing.
    assert LiveFeatureBuilder("BTC")._resolve_reference({"WBTC": [], "AAVE": []}) == "WBTC"

def test_one_feed_per_process_not_one_per_pair():
    """The selector builds a TradingBot per pair; the build is universe-wide.

    A feed per bot would repeat the same ~27s cross-sectional build up to
    GHOST_PAIR_LIMIT times and publish identical signals each time.
    """
    import trading.genome.feed as feed_module

    class FakeScheduler:
        def __init__(self):
            self.external_signals = {}

    class FakeFeed:
        started = 0

        def __init__(self, scheduler):
            self._scheduler = scheduler

        def start(self):
            FakeFeed.started += 1
            return True

    original, shared = feed_module.GenomeSignalFeed, feed_module._SHARED_FEED
    feed_module.GenomeSignalFeed, feed_module._SHARED_FEED = FakeFeed, None
    try:
        first, second = FakeScheduler(), FakeScheduler()
        a = feed_module.ensure_feed(first)
        b = feed_module.ensure_feed(second)
        assert a is b, "every bot must share one feed"
        assert FakeFeed.started == 1, "the universe build must start once"
        # A rebuilt scheduler still receives publications.
        assert b._scheduler is second
    finally:
        feed_module.GenomeSignalFeed, feed_module._SHARED_FEED = original, shared

