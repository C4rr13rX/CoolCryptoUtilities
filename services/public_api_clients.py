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
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from services.adaptive_control import APIRateLimiter
from services.logging_utils import log_message


DEFAULT_TIMEOUT = 15
USER_AGENT = "CoolCryptoUtilities/market-ingestor"
_CACHE_ROOT = Path(os.getenv("PUBLIC_API_CACHE_DIR", "data/public_api_cache")).expanduser()
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_LOCAL_SNAPSHOT = Path(os.getenv("LOCAL_MARKET_CACHE", "data/market_snapshots.json")).expanduser()
_SNAPSHOT_HISTORY_DIR = Path(os.getenv("MARKET_SNAPSHOT_HISTORY_DIR", "data/market_snapshots_history")).expanduser()
_SNAPSHOT_HISTORY_LIMIT = max(12, int(os.getenv("MARKET_SNAPSHOT_HISTORY_LIMIT", "96")))
_HISTORICAL_DATA_ROOT = Path(os.getenv("HISTORICAL_DATA_ROOT", "data/historical_ohlcv")).expanduser()
_HISTORICAL_CACHE_LABEL = "historical_snapshots"
_HISTORICAL_CACHE_TTL = max(300, int(os.getenv("HISTORICAL_SNAPSHOT_CACHE_SEC", "1800")))
_GLOBAL_BACKOFF_SEC = max(60.0, float(os.getenv("PUBLIC_API_GLOBAL_BACKOFF_SEC", "600")))

_NETWORK_BLOCKED_UNTIL: float = 0.0


@dataclass
class MarketSnapshot:
    source: str
    symbol: str
    name: str
    price_usd: float
    volume_24h: Optional[float] = None
    market_cap_usd: Optional[float] = None
    percent_change_24h: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None

RATE_LIMITER = APIRateLimiter(default_capacity=5.0, default_refill_rate=1.5)
RATE_LIMITER.configure("api.coingecko.com", capacity=3.0, refill_rate=0.5)
RATE_LIMITER.configure("api.coincap.io", capacity=3.0, refill_rate=0.8)
RATE_LIMITER.configure("api.coinpaprika.com", capacity=2.0, refill_rate=0.6)
RATE_LIMITER.configure("api.coinlore.net", capacity=2.0, refill_rate=0.7)
_HOST_BACKOFF: Dict[str, float] = {}
_DNS_BACKOFF_SEC = 300.0
_FAILURE_LOG_INTERVAL = float(os.getenv("PUBLIC_API_FAILURE_LOG_SEC", "300"))
_LAST_PROVIDER_FAILURE: Dict[str, float] = {}


def _network_available() -> bool:
    return time.time() >= _NETWORK_BLOCKED_UNTIL


def _mark_network_blocked() -> None:
    global _NETWORK_BLOCKED_UNTIL
    _NETWORK_BLOCKED_UNTIL = time.time() + _GLOBAL_BACKOFF_SEC


def _http_get(url: str, *, params: Optional[Dict[str, str]] = None) -> dict | list:
    host = url.split("/")[2]
    now = time.time()
    if not _network_available():
        raise RuntimeError("network_offline")
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
        _mark_network_blocked()
        log_message(
            "public-apis",
            f"suppressing {host} due to connectivity errors",
            severity="warning",
            details={"resume_at": until},
        )


def _log_api_failure_once(provider: str, message: str, *, severity: str = "warning", details: Optional[Dict[str, Any]] = None) -> None:
    now = time.time()
    last = _LAST_PROVIDER_FAILURE.get(provider)
    if last and (now - last) < _FAILURE_LOG_INTERVAL:
        return
    _LAST_PROVIDER_FAILURE[provider] = now
    log_message("public-apis", message, severity=severity, details=details)


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


def _persist_snapshot_records(label: str, snapshots: Sequence[MarketSnapshot]) -> None:
    if not snapshots:
        return
    _persist_cache(label, [asdict(snapshot) for snapshot in snapshots])


def _load_snapshot_records(label: str, limit: int) -> List[MarketSnapshot]:
    payload = _load_cached_payload(label)
    if not isinstance(payload, list):
        return []
    return _snapshots_from_records(payload, source=None, limit=limit)


def _append_snapshot_history(records: Sequence[Dict[str, Any]]) -> None:
    if not records:
        return
    try:
        _SNAPSHOT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        path = _SNAPSHOT_HISTORY_DIR / f"{timestamp}.json"
        payload = {"generated_at": timestamp, "data": list(records)}
        path.write_text(json.dumps(payload), encoding="utf-8")
        history = sorted(_SNAPSHOT_HISTORY_DIR.glob("*.json"))
        excess = len(history) - _SNAPSHOT_HISTORY_LIMIT
        if excess > 0:
            for old_path in history[:excess]:
                old_path.unlink(missing_ok=True)
    except Exception:
        pass


def _load_local_market_snapshots(limit: int, sources: Optional[Sequence[str]] = None) -> List[MarketSnapshot]:
    if not _LOCAL_SNAPSHOT.exists():
        return []
    try:
        payload = json.loads(_LOCAL_SNAPSHOT.read_text(encoding="utf-8"))
    except Exception:
        return []
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    if sources:
        allowed = {src.lower() for src in sources if src}
        data = [
            entry
            for entry in data
            if str(entry.get("source", "")).lower() in allowed
        ]
    return _snapshots_from_records(data, source=None, limit=limit)


def _snapshots_from_records(records: Iterable[Dict[str, Any]], *, source: Optional[str], limit: int) -> List[MarketSnapshot]:
    snapshots: List[MarketSnapshot] = []
    for entry in records:
        if len(snapshots) >= limit:
            break
        try:
            entry_source = source or str(entry.get("source") or "unknown")
            snapshots.append(
                MarketSnapshot(
                    source=str(entry_source),
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
    return _load_local_market_snapshots(limit, sources=("coincap",))


def _cached_generic_snapshots(source: str, limit: int) -> List[MarketSnapshot]:
    cached = _load_snapshot_records(f"{source}_snapshots", limit)
    if cached:
        return cached
    return _load_local_market_snapshots(limit, sources=(source,))


def _historical_file_candidates(limit: int) -> List[Path]:
    if not _HISTORICAL_DATA_ROOT.exists():
        return []
    files: List[Path] = []
    try:
        entries = sorted(_HISTORICAL_DATA_ROOT.glob("*.json"))
        if not entries:
            entries = sorted(_HISTORICAL_DATA_ROOT.rglob("*.json"))
    except Exception:
        return []
    for path in entries:
        files.append(path)
        if len(files) >= limit * 4:
            break
    return files


def _historical_snapshot_from_file(path: Path) -> Optional[MarketSnapshot]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, list) or not payload:
        return None
    tail = payload[-min(len(payload), 288) :]
    last_row = tail[-1]
    try:
        price = float(last_row.get("close") or last_row.get("price") or 0.0)
    except Exception:
        price = 0.0
    if price <= 0:
        return None
    try:
        volumes = [abs(float(row.get("net_volume", 0.0))) for row in tail if row.get("net_volume") is not None]
    except Exception:
        volumes = []
    volume = float(sum(volumes)) if volumes else None
    timestamp = int(last_row.get("timestamp", 0))
    symbol = path.stem.split("_", 1)[-1].upper()
    first_price = 0.0
    try:
        first_price = float(tail[0].get("close") or tail[0].get("price") or 0.0)
    except Exception:
        first_price = 0.0
    pct_change = None
    if first_price > 0:
        pct_change = ((price - first_price) / first_price) * 100.0
    return MarketSnapshot(
        source="historical-cache",
        symbol=symbol,
        name=symbol.replace("-", "/"),
        price_usd=price,
        volume_24h=volume,
        market_cap_usd=None,
        percent_change_24h=pct_change,
        extra={
            "source_file": str(path),
            "updated_at": timestamp,
            "sample_span": len(tail),
        },
    )


def _historical_snapshot_fallback(limit: int, symbols: Optional[Sequence[str]] = None) -> List[MarketSnapshot]:
    cached = _load_snapshot_records(_HISTORICAL_CACHE_LABEL, limit)
    now = time.time()
    cache_file = _cache_path(_HISTORICAL_CACHE_LABEL)
    if cache_file.exists():
        try:
            cache_age = now - cache_file.stat().st_mtime
        except OSError:
            cache_age = float("inf")
    else:
        cache_age = float("inf")
    symbol_filter = {sym.upper() for sym in symbols} if symbols else None
    if cached and cache_age < _HISTORICAL_CACHE_TTL:
        if symbol_filter:
            cached = [snap for snap in cached if snap.symbol.upper() in symbol_filter]
        if cached:
            return cached[:limit]
    files = _historical_file_candidates(limit)
    snapshots: List[MarketSnapshot] = []
    for path in files:
        snap = _historical_snapshot_from_file(path)
        if not snap:
            continue
        if symbol_filter and snap.symbol.upper() not in symbol_filter:
            continue
        snapshots.append(snap)
        if len(snapshots) >= limit:
            break
    if snapshots:
        _persist_snapshot_records(_HISTORICAL_CACHE_LABEL, snapshots)
    return snapshots


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
        _log_api_failure_once(
            "coincap",
            f"CoinCap fetch failed: {exc}",
            details={"limit": limit},
        )
        snapshots = _cached_coincap_snapshots(limit)
        if snapshots:
            log_message(
                "public-apis",
                "CoinCap offline fallback in use",
                severity="info",
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
    limit = max(1, top)
    try:
        payload = _http_get(
            "https://api.coinpaprika.com/v1/tickers",
            params={"limit": str(limit)},
        )
    except Exception as exc:
        _log_api_failure_once(
            "coinpaprika",
            f"CoinPaprika fetch failed: {exc}",
            details={"limit": limit},
        )
        fallback = _cached_generic_snapshots("coinpaprika", limit)
        if fallback:
            log_message(
                "public-apis",
                "CoinPaprika offline fallback in use",
                severity="info",
                details={"error": str(exc), "records": len(fallback)},
            )
            return fallback
        raise
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
    if snapshots:
        _persist_snapshot_records("coinpaprika_snapshots", snapshots)
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
    try:
        payload = _http_get("https://api.coingecko.com/api/v3/coins/markets", params=params)
    except Exception as exc:
        _log_api_failure_once(
            "coingecko",
            f"CoinGecko fetch failed: {exc}",
            details={"ids": len(ids)},
        )
        raise
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
    if snapshots:
        _persist_snapshot_records("coingecko_snapshots", snapshots)
    return snapshots


# ---------------------------------------------------------------------------
# CoinLore (https://www.coinlore.com/cryptocurrency-data-api)
# ---------------------------------------------------------------------------

def fetch_coinlore(start: int = 0, limit: int = 25) -> List[MarketSnapshot]:
    try:
        payload = _http_get(
            "https://api.coinlore.net/api/tickers/",
            params={"start": str(max(0, start)), "limit": str(max(1, limit))},
        )
    except Exception as exc:
        _log_api_failure_once(
            "coinlore",
            f"CoinLore fetch failed: {exc}",
            details={"start": start, "limit": limit},
        )
        fallback = _cached_generic_snapshots("coinlore", max(1, limit))
        if fallback:
            log_message(
                "public-apis",
                "CoinLore offline fallback in use",
                severity="info",
                details={"error": str(exc), "records": len(fallback)},
            )
            return fallback
        raise
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
    if snapshots:
        _persist_snapshot_records("coinlore_snapshots", snapshots)
    return snapshots


def _merge_symbol_consensus(snapshots: Sequence[MarketSnapshot]) -> List[MarketSnapshot]:
    grouped: Dict[str, List[MarketSnapshot]] = {}
    for snap in snapshots:
        symbol = snap.symbol.upper()
        grouped.setdefault(symbol, []).append(snap)
    merged: List[MarketSnapshot] = []
    for symbol, group in grouped.items():
        valid_prices = [snap.price_usd for snap in group if snap.price_usd > 0]
        if not valid_prices:
            continue
        median_price = statistics.median(valid_prices)
        base = max(group, key=lambda s: ((s.volume_24h or 0.0), (s.market_cap_usd or 0.0)))
        contributors = sorted({snap.source for snap in group})
        spread = max(valid_prices) - min(valid_prices) if len(valid_prices) > 1 else 0.0
        merged.append(
            MarketSnapshot(
                source="consensus" if len(group) > 1 else base.source,
                symbol=symbol,
                name=base.name or symbol,
                price_usd=float(median_price),
                volume_24h=max((snap.volume_24h or 0.0) for snap in group) or None,
                market_cap_usd=base.market_cap_usd,
                percent_change_24h=base.percent_change_24h,
                extra={
                    "contributors": contributors,
                    "sample_size": len(group),
                    "price_spread": float(spread),
                },
            )
        )
    return merged


def aggregate_market_data(
    *,
    symbols: Optional[Sequence[str]] = None,
    coingecko_ids: Optional[Sequence[str]] = None,
    top_n: int = 25,
) -> List[MarketSnapshot]:
    """Fetch and merge snapshots from the public APIs."""

    snapshots: List[MarketSnapshot] = []
    errors: Dict[str, str] = {}
    fallback_used = False
    offline_mode = not _network_available()
    if offline_mode:
        log_message(
            "public-apis",
            "network unavailable; using cached/historical market data",
            severity="info",
        )
    if not offline_mode:
        try:
            snapshots.extend(fetch_coincap(top=top_n))
        except Exception as exc:
            errors["coincap"] = str(exc)
        time.sleep(0.4)

        try:
            snapshots.extend(fetch_coinpaprika(top=top_n))
        except Exception as exc:
            errors["coinpaprika"] = str(exc)
        time.sleep(0.4)

        try:
            snapshots.extend(fetch_coinlore(limit=top_n))
        except Exception as exc:
            errors["coinlore"] = str(exc)
        time.sleep(0.4)

        if coingecko_ids:
            try:
                snapshots.extend(fetch_coingecko(coingecko_ids))
            except Exception as exc:
                errors["coingecko"] = str(exc)

    # Filter to requested symbols if provided
    wanted = {sym.upper() for sym in symbols} if symbols else None
    if wanted:
        snapshots = [snap for snap in snapshots if snap.symbol.upper() in wanted]
        if not snapshots:
            offline_specific = _historical_snapshot_fallback(top_n, symbols=wanted or None)
            if offline_specific:
                snapshots.extend(offline_specific)
                fallback_used = True

    if not snapshots:
        archive = _load_local_market_snapshots(top_n, sources=None)
        if archive:
            log_message(
                "public-apis",
                "market snapshot archive used",
                severity="info",
                details={"records": len(archive)},
            )
            snapshots = archive
            fallback_used = True
            if wanted:
                snapshots = [snap for snap in snapshots if snap.symbol.upper() in wanted]

    if len(snapshots) < max(1, top_n):
        offline = _historical_snapshot_fallback(top_n, symbols=wanted or None)
        if offline:
            snapshots.extend(offline)
            fallback_used = True

    if not snapshots:
        return []

    merged = _merge_symbol_consensus(snapshots)
    if wanted:
        merged = [snap for snap in merged if snap.symbol.upper() in wanted]
    merged.sort(key=lambda snap: snap.volume_24h or 0.0, reverse=True)
    if top_n:
        merged = merged[: max(1, top_n)]
    if merged:
        try:
            save_snapshots(merged, _LOCAL_SNAPSHOT)
        except Exception:
            pass
    if errors:
        severity = "info" if fallback_used else "warning"
        message = "market data served from fallback" if fallback_used else "market data fetch degraded"
        details = {"providers": errors, "fallback": "historical" if fallback_used else "none"}
        log_message("public-apis", message, severity=severity, details=details)
    return merged


def save_snapshots(snapshots: Sequence[MarketSnapshot], path: Path) -> None:
    records = [asdict(snapshot) for snapshot in snapshots]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": time.time(), "data": records}
    path.write_text(json.dumps(payload, indent=2))
    _append_snapshot_history(records)


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
