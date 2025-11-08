from __future__ import annotations

import requests

import pytest

from services.public_api_clients import (
    MarketSnapshot,
    aggregate_market_data,
    fetch_coincap,
    fetch_coinlore,
    fetch_coinpaprika,
)


def test_fetch_coincap_returns_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import public_api_clients as pac

    def boom(*_args: object, **_kwargs: object) -> None:
        raise requests.exceptions.ConnectionError("dns failure")

    sample = [
        MarketSnapshot(source="cache", symbol="ETH", name="Ethereum", price_usd=1800.0),
        MarketSnapshot(source="cache", symbol="BTC", name="Bitcoin", price_usd=42000.0),
    ]

    monkeypatch.setattr(pac, "_http_get", boom)
    monkeypatch.setattr(pac, "_cached_coincap_snapshots", lambda _limit: sample)

    snapshots = fetch_coincap(top=2)
    assert snapshots == sample


def test_fetch_coinpaprika_returns_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import public_api_clients as pac

    payload = [
        {
            "symbol": "btc",
            "name": "Bitcoin",
            "quotes": {"USD": {"price": 42000, "volume_24h": 10_000, "market_cap": 1_000_000}},
            "percent_change_24h": 1.2,
            "percent_change_7d": -2.3,
        }
    ]

    monkeypatch.setattr(pac, "_http_get", lambda *_args, **_kwargs: payload)

    snapshots = fetch_coinpaprika(top=1)
    assert snapshots, "CoinPaprika should return assets"
    assert snapshots[0].symbol == "BTC"


def test_fetch_coinlore_returns_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import public_api_clients as pac

    payload = {"data": [{"symbol": "SOL", "name": "Solana", "price_usd": 80.0}]}
    monkeypatch.setattr(pac, "_http_get", lambda *_args, **_kwargs: payload)
    snapshots = fetch_coinlore(limit=1)
    assert snapshots and snapshots[0].symbol == "SOL"


def test_aggregate_market_data(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import public_api_clients as pac

    base = MarketSnapshot(source="cache", symbol="BTC", name="Bitcoin", price_usd=1.0)
    monkeypatch.setattr(pac, "fetch_coincap", lambda top=5: [base])
    monkeypatch.setattr(pac, "fetch_coinpaprika", lambda top=5: [base])
    monkeypatch.setattr(pac, "fetch_coinlore", lambda limit=5: [base])
    monkeypatch.setattr(
        pac,
        "fetch_coingecko",
        lambda ids: [MarketSnapshot(source="cg", symbol="ETH", name="Ethereum", price_usd=2.0)],
    )

    snaps = aggregate_market_data(symbols=["BTC", "ETH"], coingecko_ids=["bitcoin", "ethereum"], top_n=5)
    assert {snap.symbol for snap in snaps} == {"BTC", "ETH"}
