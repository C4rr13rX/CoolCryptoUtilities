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


#: GeckoTerminal exposes a real trending-pools list per network. DexScreener's
#: /latest/dex/tokens requires a token address and has no trending variant, so
#: calling it with ?limit=&filter= returned **HTTP 404 every time** -- which is
#: why `discovery_discoveredtoken` sat at 0 rows and a token that started
#: moving could never enter the streamed universe.
GECKOTERMINAL_TRENDING = "https://api.geckoterminal.com/api/v2/networks/{network}/trending_pools"

#: Our chain names to GeckoTerminal network slugs.
_GECKO_NETWORKS = {
    "base": "base",
    "ethereum": "eth",
    "arbitrum": "arbitrum",
    "optimism": "optimism",
    "polygon": "polygon_pos",
    "bsc": "bsc",
}


def _gecko_trending(chain: str, timeout: float) -> List[Dict]:
    """Trending pools for one chain, as DexScreener-shaped dicts."""
    network = _GECKO_NETWORKS.get(chain.lower())
    if not network:
        return []
    url = GECKOTERMINAL_TRENDING.format(network=network)
    resp = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (compatible; R3V3N1R/1.0)",
                 "Accept": "application/json"},
    )
    resp.raise_for_status()
    out: List[Dict] = []
    for item in (resp.json().get("data") or []):
        attrs = item.get("attributes") or {}
        name = str(attrs.get("name") or "")
        # "Basecat / USDC 0.9%" -> base "Basecat", quote "USDC"
        parts = [p.strip() for p in name.split("/")]
        base_sym = parts[0] if parts else ""
        quote_sym = parts[1].split()[0] if len(parts) > 1 and parts[1] else ""
        change = attrs.get("price_change_percentage") or {}
        volume = attrs.get("volume_usd") or {}
        out.append({
            "baseToken": {"symbol": base_sym},
            "quoteToken": {"symbol": quote_sym},
            "chainId": chain.lower(),
            "pairAddress": str(attrs.get("address") or ""),
            "dexId": str(((item.get("relationships") or {}).get("dex") or {})
                         .get("data", {}).get("id") or "unknown"),
            "priceUsd": attrs.get("base_token_price_usd"),
            "volumeUsd24h": volume.get("h24"),
            "liquidity": {"usd": attrs.get("reserve_in_usd")},
            "priceChange": {"h1": change.get("h1"), "h6": change.get("h6"),
                            "h24": change.get("h24")},
            "fdv": attrs.get("fdv_usd"),
        })
    return out


def fetch_trending_tokens(limit: int = 50, chains: Optional[List[str]] = None) -> List[TrendingToken]:
    timeout = float(os.getenv("DISCOVERY_HTTP_TIMEOUT", "15"))
    data: List[Dict] = []
    for chain in (chains or ["base"]):
        try:
            data.extend(_gecko_trending(chain, timeout))
        except Exception as exc:
            print(f"[discovery] trending fetch failed for {chain}: {exc}")
    data = data[:limit]
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
