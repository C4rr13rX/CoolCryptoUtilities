from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

from services.token_address_book import is_token_address, record_many


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
    #: ERC-20 contract addresses -- what a swap actually needs. ``pair_address``
    #: identifies the POOL and is not interchangeable with these: on Uniswap v4
    #: it is a 32-byte pool id rather than an address at all. Defaulted and
    #: last so that every existing construction site stays valid; a caller that
    #: cannot supply them gets "" and the token simply stays unresolved.
    base_token_address: str = ""
    quote_token_address: str = ""


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


def _gecko_address(token_id: str) -> str:
    """``"base_0xabc..."`` -> ``"0xabc..."``; "" when it is not an address."""
    raw = str(token_id or "").strip()
    if "_" in raw:
        raw = raw.rsplit("_", 1)[1]
    return raw if is_token_address(raw) else ""


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
        # The token CONTRACT addresses ride along in relationships as
        # "base_0x<address>". They used to be dropped here, leaving the pool
        # address as the only identifier downstream -- and a pool is not
        # something you can swap. That is what made every live entry on a
        # discovered token fail with reason=token_unresolved.
        rel = item.get("relationships") or {}
        base_id = str(((rel.get("base_token") or {}).get("data") or {}).get("id") or "")
        quote_id = str(((rel.get("quote_token") or {}).get("data") or {}).get("id") or "")
        out.append({
            "baseToken": {"symbol": base_sym, "address": _gecko_address(base_id)},
            "quoteToken": {"symbol": quote_sym, "address": _gecko_address(quote_id)},
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
                    base_token_address=str(entry.get("baseToken", {}).get("address") or ""),
                    quote_token_address=str(entry.get("quoteToken", {}).get("address") or ""),
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

    # Write the addresses down as they arrive. The resolver that decides
    # whether a live trade can be built runs in a different process and long
    # after this fetch, so holding them only in memory is the same as dropping
    # them -- which is what used to happen.
    learned: Dict[str, Dict[str, str]] = {}
    for tok in results:
        base_sym, _, quote_sym = tok.symbol.partition("-")
        for sym, addr in ((base_sym, tok.base_token_address),
                          (quote_sym, tok.quote_token_address)):
            if sym and is_token_address(addr):
                learned.setdefault(tok.chain, {})[sym] = addr
    for chain, mapping in learned.items():
        try:
            record_many(chain, mapping, source="geckoterminal")
        except Exception as exc:  # never let bookkeeping break discovery
            print(f"[discovery] could not record token addresses for {chain}: {exc}")
    return results
