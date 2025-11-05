"""
Utilities for collecting market data from public cryptocurrency APIs that do
not require authentication keys.

The intent is to give the training pipeline multiple independent data feeds it
can lean on when a primary provider throttles or stalls.  Each fetcher returns
`MarketSnapshot` records with a common schema so downstream consumers can
combine or persist them without worrying about provider-specific quirks.

All calls obey polite rate limits and should be safe for periodic background
jobs (e.g. cron every 5-10 minutes).  No API keys are required for any of the
sources included here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import requests


DEFAULT_TIMEOUT = 15
USER_AGENT = "CoolCryptoUtilities/market-ingestor"


@dataclass
class MarketSnapshot:
    source: str
    symbol: str
    name: str
    price_usd: float
    volume_24h: Optional[float] = None
    market_cap_usd: Optional[float] = None
    percent_change_24h: Optional[float] = None
    extra: Optional[Dict[str, float]] = None


def _http_get(url: str, *, params: Optional[Dict[str, str]] = None) -> dict | list:
    resp = requests.get(
        url,
        params=params,
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# CoinCap (https://docs.coincap.io/)
# ---------------------------------------------------------------------------

def fetch_coincap(top: int = 25) -> List[MarketSnapshot]:
    payload = _http_get(
        "https://api.coincap.io/v2/assets",
        params={"limit": str(max(1, top))},
    )
    data = payload.get("data") or []
    snapshots: List[MarketSnapshot] = []
    for entry in data:
        try:
            snapshots.append(
                MarketSnapshot(
                    source="coincap",
                    symbol=str(entry.get("symbol", "")).upper(),
                    name=str(entry.get("name") or ""),
                    price_usd=float(entry.get("priceUsd", 0.0)),
                    volume_24h=float(entry.get("volumeUsd24Hr")) if entry.get("volumeUsd24Hr") else None,
                    market_cap_usd=float(entry.get("marketCapUsd")) if entry.get("marketCapUsd") else None,
                    percent_change_24h=float(entry.get("changePercent24Hr")) if entry.get("changePercent24Hr") else None,
                )
            )
        except (TypeError, ValueError):
            continue
    return snapshots


# ---------------------------------------------------------------------------
# CoinPaprika (https://api.coinpaprika.com)
# ---------------------------------------------------------------------------

def fetch_coinpaprika(top: int = 25) -> List[MarketSnapshot]:
    payload = _http_get(
        "https://api.coinpaprika.com/v1/tickers",
        params={"limit": str(max(1, top))},
    )
    snapshots: List[MarketSnapshot] = []
    for entry in payload:
        quotes = entry.get("quotes") or {}
        usd = quotes.get("USD") or {}
        try:
            snapshots.append(
                MarketSnapshot(
                    source="coinpaprika",
                    symbol=str(entry.get("symbol", "")).upper(),
                    name=str(entry.get("name") or ""),
                    price_usd=float(usd.get("price", 0.0)),
                    volume_24h=float(usd.get("volume_24h")) if usd.get("volume_24h") is not None else None,
                    market_cap_usd=float(usd.get("market_cap")) if usd.get("market_cap") is not None else None,
                    percent_change_24h=float(usd.get("percent_change_24h")) if usd.get("percent_change_24h") is not None else None,
                    extra={"percent_change_7d": float(usd.get("percent_change_7d"))} if usd.get("percent_change_7d") is not None else None,
                )
            )
        except (TypeError, ValueError):
            continue
    return snapshots


# ---------------------------------------------------------------------------
# CoinGecko (https://www.coingecko.com/api/documentations/v3)
# ---------------------------------------------------------------------------

def fetch_coingecko(ids: Sequence[str]) -> List[MarketSnapshot]:
    if not ids:
        return []
    params = {
        "vs_currency": "usd",
        "ids": ",".join(ids),
        "price_change_percentage": "24h,7d",
    }
    payload = _http_get("https://api.coingecko.com/api/v3/coins/markets", params=params)
    snapshots: List[MarketSnapshot] = []
    for entry in payload:
        try:
            snapshots.append(
                MarketSnapshot(
                    source="coingecko",
                    symbol=str(entry.get("symbol", "")).upper(),
                    name=str(entry.get("name") or ""),
                    price_usd=float(entry.get("current_price", 0.0)),
                    volume_24h=float(entry.get("total_volume")) if entry.get("total_volume") is not None else None,
                    market_cap_usd=float(entry.get("market_cap")) if entry.get("market_cap") is not None else None,
                    percent_change_24h=float(entry.get("price_change_percentage_24h")) if entry.get("price_change_percentage_24h") is not None else None,
                    extra={
                        "percent_change_7d": float(entry.get("price_change_percentage_7d_in_currency"))
                        if entry.get("price_change_percentage_7d_in_currency") is not None
                        else None
                    },
                )
            )
        except (TypeError, ValueError):
            continue
    return snapshots


# ---------------------------------------------------------------------------
# CoinLore (https://www.coinlore.com/cryptocurrency-data-api)
# ---------------------------------------------------------------------------

def fetch_coinlore(start: int = 0, limit: int = 25) -> List[MarketSnapshot]:
    payload = _http_get(
        "https://api.coinlore.net/api/tickers/",
        params={"start": str(max(0, start)), "limit": str(max(1, limit))},
    )
    data = payload.get("data") or []
    snapshots: List[MarketSnapshot] = []
    for entry in data:
        try:
            snapshots.append(
                MarketSnapshot(
                    source="coinlore",
                    symbol=str(entry.get("symbol", "")).upper(),
                    name=str(entry.get("name") or ""),
                    price_usd=float(entry.get("price_usd", 0.0)),
                    volume_24h=float(entry.get("volume24")) if entry.get("volume24") is not None else None,
                    market_cap_usd=float(entry.get("market_cap_usd")) if entry.get("market_cap_usd") is not None else None,
                    percent_change_24h=float(entry.get("percent_change_24h")) if entry.get("percent_change_24h") is not None else None,
                )
            )
        except (TypeError, ValueError):
            continue
    return snapshots


def aggregate_market_data(
    *,
    symbols: Optional[Sequence[str]] = None,
    coingecko_ids: Optional[Sequence[str]] = None,
    top_n: int = 25,
) -> List[MarketSnapshot]:
    """Fetch and merge snapshots from the public APIs."""

    snapshots: List[MarketSnapshot] = []
    try:
        snapshots.extend(fetch_coincap(top=top_n))
    except Exception as exc:
        print(f"[public-apis] CoinCap fetch failed: {exc}", file=sys.stderr)
    time.sleep(0.4)

    try:
        snapshots.extend(fetch_coinpaprika(top=top_n))
    except Exception as exc:
        print(f"[public-apis] CoinPaprika fetch failed: {exc}", file=sys.stderr)
    time.sleep(0.4)

    try:
        snapshots.extend(fetch_coinlore(limit=top_n))
    except Exception as exc:
        print(f"[public-apis] CoinLore fetch failed: {exc}", file=sys.stderr)
    time.sleep(0.4)

    if coingecko_ids:
        try:
            snapshots.extend(fetch_coingecko(coingecko_ids))
        except Exception as exc:
            print(f"[public-apis] CoinGecko fetch failed: {exc}", file=sys.stderr)

    # Filter to requested symbols if provided
    if symbols:
        wanted = {sym.upper() for sym in symbols}
        snapshots = [snap for snap in snapshots if snap.symbol.upper() in wanted]

    # Deduplicate by (source, symbol)
    unique: Dict[tuple[str, str], MarketSnapshot] = {}
    for snap in snapshots:
        key = (snap.source.lower(), snap.symbol.upper())
        unique[key] = snap
    return list(unique.values())


def save_snapshots(snapshots: Sequence[MarketSnapshot], path: Path) -> None:
    records = [asdict(snapshot) for snapshot in snapshots]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"generated_at": time.time(), "data": records}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect public cryptocurrency market data (no API keys).")
    parser.add_argument("--symbols", help="Comma separated list of ticker symbols to retain (default: all).")
    parser.add_argument("--coingecko-ids", help="Comma separated list of CoinGecko IDs (e.g. bitcoin,ethereum).")
    parser.add_argument("--top", type=int, default=25, help="How many assets to fetch from each provider (default: 25).")
    parser.add_argument("--output", help="Optional JSON path to write the combined snapshots.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    symbols = [sym.strip().upper() for sym in (args.symbols or "").split(",") if sym.strip()] or None
    gecko_ids = [gid.strip() for gid in (args.coingecko_ids or "").split(",") if gid.strip()] or None
    snapshots = aggregate_market_data(symbols=symbols, coingecko_ids=gecko_ids, top_n=args.top)
    print(f"Fetched {len(snapshots)} unique snapshots")
    if args.output:
        save_snapshots(snapshots, Path(args.output))
        print(f"Wrote {len(snapshots)} records to {args.output}")
    else:
        for snap in snapshots[:10]:
            print(f"{snap.source:<12} {snap.symbol:<8} ${snap.price_usd:,.2f}")


if __name__ == "__main__":
    main()
