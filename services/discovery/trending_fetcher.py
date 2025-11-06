from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests


DEXSCREENER_URL = "https://api.dexscreener.com/latest/dex/tokens"


@dataclass
class TrendingToken:
    symbol: str
    chain: str
    pair_address: str
    dex_id: str
    price_usd: Optional[float]
    volume_24h_usd: Optional[float]
    liquidity_usd: Optional[float]
    price_change_1h: Optional[float]
    price_change_6h: Optional[float]
    price_change_24h: Optional[float]
    metadata: Dict[str, float]


def fetch_trending_tokens(limit: int = 50, chains: Optional[List[str]] = None) -> List[TrendingToken]:
    params: Dict[str, str] = {"limit": str(limit)}
    if chains:
        params["filter"] = ",".join([chain.lower() for chain in chains])
    resp = requests.get(DEXSCREENER_URL, params=params, timeout=float(os.getenv("DISCOVERY_HTTP_TIMEOUT", "12")))
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("pairs") or []
    results: List[TrendingToken] = []
    for entry in data:
        try:
            results.append(
                TrendingToken(
                    symbol=str(entry.get("baseToken", {}).get("symbol") or "") + "-" + str(entry.get("quoteToken", {}).get("symbol") or ""),
                    chain=str(entry.get("chainId") or entry.get("chain", "unknown")),
                    pair_address=str(entry.get("pairAddress") or ""),
                    dex_id=str(entry.get("dexId") or entry.get("exchange", "unknown")),
                    price_usd=float(entry["priceUsd"]) if entry.get("priceUsd") else None,
                    volume_24h_usd=float(entry["volumeUsd24h"]) if entry.get("volumeUsd24h") else None,
                    liquidity_usd=float(entry["liquidity"].get("usd")) if entry.get("liquidity", {}).get("usd") else None,
                    price_change_1h=float(entry["priceChange"].get("h1")) if entry.get("priceChange", {}).get("h1") else None,
                    price_change_6h=float(entry["priceChange"].get("h6")) if entry.get("priceChange", {}).get("h6") else None,
                    price_change_24h=float(entry["priceChange"].get("h24")) if entry.get("priceChange", {}).get("h24") else None,
                    metadata={
                        "fdv": float(entry.get("fdv")) if entry.get("fdv") else None,
                        "transactions_1h": entry.get("txns", {}).get("h1", {}).get("buys"),
                    },
                )
            )
        except (TypeError, ValueError):
            continue
    return results
