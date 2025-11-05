from __future__ import annotations

from pathlib import Path

import pytest

from services.public_api_clients import (
    aggregate_market_data,
    fetch_coincap,
    fetch_coinlore,
    fetch_coinpaprika,
)


def test_fetch_coincap_returns_snapshots():
    try:
        snapshots = fetch_coincap(top=3)
    except Exception as exc:
        pytest.skip(f"CoinCap request failed: {exc}")
    assert snapshots, "CoinCap should return assets"
    assert all(snapshot.price_usd > 0 for snapshot in snapshots)


def test_fetch_coinpaprika_returns_snapshots():
    snapshots = fetch_coinpaprika(top=3)
    assert snapshots, "CoinPaprika should return assets"
    assert any(snapshot.symbol == "BTC" for snapshot in snapshots)


def test_fetch_coinlore_returns_snapshots():
    snapshots = fetch_coinlore(limit=3)
    assert snapshots, "CoinLore should return assets"
    assert all(snapshot.symbol for snapshot in snapshots)


def test_aggregate_market_data(tmp_path: Path):
    snapshots = aggregate_market_data(symbols=["BTC", "ETH"], coingecko_ids=["bitcoin", "ethereum"], top_n=5)
    assert snapshots, "Aggregation should produce snapshots"
    btc_entries = [snap for snap in snapshots if snap.symbol == "BTC"]
    assert btc_entries, "BTC data should be present"
