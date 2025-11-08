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
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from services.adaptive_control import APIRateLimiter
from services.logging_bus import log_message


DEFAULT_TIMEOUT = 15
USER_AGENT = "CoolCryptoUtilities/market-ingestor"
_CACHE_ROOT = Path(os.getenv("PUBLIC_API_CACHE_DIR", "data/public_api_cache")).expanduser()
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_LOCAL_SNAPSHOT = Path(os.getenv("LOCAL_MARKET_CACHE", "data/market_snapshots.json")).expanduser()


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

RATE_LIMITER = APIRateLimiter(default_capacity=5.0, default_refill_rate=1.5)
RATE_LIMITER.configure("api.coingecko.com", capacity=3.0, refill_rate=0.5)
RATE_LIMITER.configure("api.coincap.io", capacity=3.0, refill_rate=0.8)
RATE_LIMITER.configure("api.coinpaprika.com", capacity=2.0, refill_rate=0.6)
RATE_LIMITER.configure("api.coinlore.net", capacity=2.0, refill_rate=0.7)
_HOST_BACKOFF: Dict[str, float] = {}
_DNS_BACKOFF_SEC = 300.0


def _http_get(url: str, *, params: Optional[Dict[str, str]] = None) -> dict | list:
    host = url.split("/")[2]
    now = time.time()
    resume = _HOST_BACKOFF.get(host, 0.0)
    if resume and now < resume:
        raise RuntimeError(f"{host} temporarily suppressed for {int(resume - now)}s")
    try:
        RATE_LIMITER.acquire(host, tokens=1.0, timeout=10.0)
    except TimeoutError as exc:
        log_message("public-api", f"Rate limit exceeded for {host}", severity="warning")
        raise
    resp = requests.get(
        url,
        params=params,
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    try:
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        _maybe_backoff_host(host, exc)
        raise


def _maybe_backoff_host(host: str, exc: Exception) -> None:
    message = str(exc).lower()
    backoff = False
    if isinstance(exc, requests.exceptions.ConnectionError):
        backoff = True
    elif "name or service not known" in message or "temporary failure in name resolution" in message:
        backoff = True
    if backoff:
        until = time.time() + _DNS_BACKOFF_SEC
        _HOST_BACKOFF[host] = until
        log_message(
            "public-apis",
            f"suppressing {host} due to connectivity errors",
            severity="warning",
            details={"resume_at": until},
        )


def _cache_path(name: str) -> Path:
    return _CACHE_ROOT / f"{name}.json"


def _persist_cache(name: str, payload: Any) -> None:
    path = _cache_path(name)
    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        path.unlink(missing_ok=True)


def _load_cached_payload(name: str) -> Optional[Any]:
    path = _cache_path(name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        path.unlink(missing_ok=True)
        return None


def _load_local_market_snapshots(limit: int, source: str = "coincap") -> List[MarketSnapshot]:
    if not _LOCAL_SNAPSHOT.exists():
        return []
    try:
        payload = json.loads(_LOCAL_SNAPSHOT.read_text(encoding="utf-8"))
    except Exception:
        return []
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    return _snapshots_from_records(data, source=source, limit=limit)


def _snapshots_from_records(records: Iterable[Dict[str, Any]], *, source: str, limit: int) -> List[MarketSnapshot]:
    snapshots: List[MarketSnapshot] = []
    for entry in records:
        if len(snapshots) >= limit:
            break
        try:
            snapshots.append(
                MarketSnapshot(
                    source=source,
                    symbol=str(entry.get("symbol", "")).upper(),
                    name=str(entry.get("name") or ""),
                    price_usd=float(entry.get("priceUsd") or entry.get("price_usd") or 0.0),
                    volume_24h=float(entry.get("volumeUsd24Hr") or entry.get("volume_24h"))
                    if entry.get("volumeUsd24Hr") or entry.get("volume_24h")
                    else None,
                    market_cap_usd=float(entry.get("marketCapUsd") or entry.get("market_cap_usd"))
                    if entry.get("marketCapUsd") or entry.get("market_cap_usd")
                    else None,
                    percent_change_24h=float(entry.get("changePercent24Hr") or entry.get("percent_change_24h"))
                    if entry.get("changePercent24Hr") or entry.get("percent_change_24h")
                    else None,
                    extra=entry.get("extra") if isinstance(entry.get("extra"), dict) else None,
                )
            )
        except (TypeError, ValueError):
            continue
    return snapshots


def _coincap_payload_to_snapshots(payload: Dict[str, Any], top: int) -> List[MarketSnapshot]:
    data = payload.get("data") if isinstance(payload, dict) else []
    if not isinstance(data, list):
        data = []
    return _snapshots_from_records(data, source="coincap", limit=top)


def _cached_coincap_snapshots(limit: int) -> List[MarketSnapshot]:
    cached_payload = _load_cached_payload("coincap_latest")
    if isinstance(cached_payload, dict):
        cached = _coincap_payload_to_snapshots(cached_payload, limit)
        if cached:
            return cached
    return _load_local_market_snapshots(limit)


# ---------------------------------------------------------------------------
# CoinCap (https://docs.coincap.io/)
# ---------------------------------------------------------------------------

def fetch_coincap(top: int = 25) -> List[MarketSnapshot]:
    limit = max(1, top)
    try:
        payload = _http_get(
            "https://api.coincap.io/v2/assets",
            params={"limit": str(limit)},
        )
    except Exception as exc:
        snapshots = _cached_coincap_snapshots(limit)
        if snapshots:
            log_message(
                "public-apis",
                "CoinCap offline fallback in use",
                severity="warning",
                details={"error": str(exc), "records": len(snapshots)},
            )
            return snapshots
        raise
    snapshots = _coincap_payload_to_snapshots(payload, limit)
    if snapshots:
        _persist_cache("coincap_latest", payload)
        _persist_cache("coincap_snapshots", [asdict(s) for s in snapshots])
        return snapshots
    cached = _cached_coincap_snapshots(limit)
    if cached:
        log_message(
            "public-apis",
            "CoinCap returned empty payload; using cached snapshots",
            severity="warning",
            details={"records": len(cached)},
        )
        return cached
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
        log_message("public-apis", f"CoinCap fetch failed: {exc}", severity="warning")
    time.sleep(0.4)

    try:
        snapshots.extend(fetch_coinpaprika(top=top_n))
    except Exception as exc:
        log_message("public-apis", f"CoinPaprika fetch failed: {exc}", severity="warning")
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
