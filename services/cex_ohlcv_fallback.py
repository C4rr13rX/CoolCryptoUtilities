"""
CEX OHLCV Fallback — download historical OHLCV data from free public exchange
APIs (Binance, CoinGecko) when on-chain RPC pool queries fail.

This module exists because the pair indexes (pair_index_base.json, etc.) contain
Ethereum mainnet Uniswap V2 pool addresses that don't exist on Base/Arbitrum/etc.
When the on-chain downloader (download2000.py) queries these addresses against
non-Ethereum RPCs, every pool fails with "non_uniswap_pool".

The CEX fallback downloads equivalent OHLCV data from centralized exchanges,
which trade the same token pairs. The output format matches what download2000.py
produces so the HistoricalDataLoader can consume it seamlessly.

Designed for: i5 CPU, 32GB RAM, no GPU.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from services.logging_utils import log_message

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = _REPO_ROOT / "data"
_DEFAULT_OUTPUT_ROOT = _DATA_ROOT / "historical_ohlcv"

# Binance kline intervals mapped to seconds
_BINANCE_INTERVAL = "5m"
_BINANCE_INTERVAL_SEC = 300
_BINANCE_MAX_LIMIT = 1000  # max candles per request

# Rate limiting
_REQUEST_DELAY = 0.35  # seconds between requests (stay well under limits)

# Symbols that Binance supports — map from our pair format to Binance format.
# Our format: "WETH-USDC" → Binance: "ETHUSDC"
_TOKEN_TO_BINANCE = {
    "WETH": "ETH",
    "WBTC": "BTC",
    "WMATIC": "MATIC",
    "WAVAX": "AVAX",
    "WBNB": "BNB",
}

# Binance quote assets (order of preference)
_BINANCE_QUOTES = ["USDC", "USDT", "BUSD", "BTC", "ETH"]

# CoinGecko IDs for fallback
_COINGECKO_IDS = {
    "ETH": "ethereum",
    "WETH": "ethereum",
    "BTC": "bitcoin",
    "WBTC": "bitcoin",
    "MATIC": "matic-network",
    "WMATIC": "matic-network",
    "SOL": "solana",
    "AVAX": "avalanche-2",
    "WAVAX": "avalanche-2",
    "LINK": "chainlink",
    "UNI": "uniswap",
    "AAVE": "aave",
    "ARB": "arbitrum",
    "OP": "optimism",
    "DAI": "dai",
    "PEPE": "pepe",
    "DOGE": "dogecoin",
    "SHIB": "shiba-inu",
    "CRV": "curve-dao-token",
    "MKR": "maker",
    "COMP": "compound-governance-token",
    "SNX": "havven",
    "SUSHI": "sushi",
    "YFI": "yearn-finance",
    "LDO": "lido-dao",
    "RPL": "rocket-pool",
    "GMX": "gmx",
    "PENDLE": "pendle",
    "RETH": "rocket-pool-eth",
    "CBETH": "coinbase-wrapped-staked-eth",
}

# Core pairs to always attempt downloading (high liquidity, reliable data).
# Includes both USDC-denominated (for direct trading) and key ETH-denominated pairs.
CORE_PAIRS = [
    "WETH-USDC",
    "WETH-USDT",
    "WBTC-USDC",
    "WBTC-WETH",
    "LINK-USDC",
    "UNI-USDC",
    "AAVE-USDC",
    "ARB-USDC",
    "OP-USDC",
    "DAI-USDC",
    "PEPE-USDC",
    "DOGE-USDC",
    "SHIB-USDC",
    "CRV-USDC",
    "MKR-USDC",
    "LDO-USDC",
    "SNX-USDC",
    "SUSHI-USDC",
    "COMP-USDC",
]


def _normalize_token(token: str) -> str:
    """Convert wrapped token names to their CEX equivalents."""
    return _TOKEN_TO_BINANCE.get(token.upper(), token.upper())


def _to_binance_symbol(base: str, quote: str) -> Optional[str]:
    """Convert our pair format to Binance symbol format."""
    b = _normalize_token(base)
    q = _normalize_token(quote)
    # Binance doesn't have ETH-ETH or similar
    if b == q:
        return None
    return f"{b}{q}"


def _fetch_binance_klines(
    symbol: str,
    interval: str = _BINANCE_INTERVAL,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: int = _BINANCE_MAX_LIMIT,
) -> List[List]:
    """Fetch klines from Binance public API (no key required)."""
    url = "https://api.binance.com/api/v3/klines"
    params: Dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_ms:
        params["startTime"] = start_ms
    if end_ms:
        params["endTime"] = end_ms
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 429:
            log_message("cex-fallback", f"Binance rate limited for {symbol}", severity="warning")
            time.sleep(10)
            return []
        if resp.status_code != 200:
            return []
        return resp.json()
    except Exception as exc:
        log_message("cex-fallback", f"Binance request failed: {exc}", severity="warning")
        return []


def _binance_klines_to_ohlcv(klines: List[List]) -> List[Dict[str, Any]]:
    """Convert Binance kline format to our OHLCV format."""
    rows = []
    for k in klines:
        # Binance kline: [open_time, open, high, low, close, volume, close_time, ...]
        try:
            rows.append({
                "timestamp": int(k[0]) // 1000,  # ms → seconds
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "net_volume": float(k[5]),
                # k[9] = taker_buy_base_asset_volume (base currency volume bought)
                "buy_volume": float(k[9]) if len(k) > 9 and float(k[9]) <= float(k[5]) else float(k[5]) * 0.5,
                "sell_volume": max(0.0, float(k[5]) - float(k[9])) if len(k) > 9 and float(k[9]) <= float(k[5]) else float(k[5]) * 0.5,
                "vwap": (float(k[2]) + float(k[3]) + float(k[4])) / 3.0,
            })
        except (IndexError, ValueError, TypeError):
            continue
    return rows


def download_pair_binance(
    base: str,
    quote: str,
    *,
    days_back: int = 90,
    interval: str = _BINANCE_INTERVAL,
) -> List[Dict[str, Any]]:
    """
    Download full OHLCV history for a pair from Binance.
    Paginates through the API to get up to `days_back` days of data.
    """
    binance_sym = _to_binance_symbol(base, quote)
    if not binance_sym:
        return []

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (days_back * 86400 * 1000)
    all_rows: List[Dict[str, Any]] = []
    cursor = start_ms

    while cursor < now_ms:
        klines = _fetch_binance_klines(binance_sym, interval=interval, start_ms=cursor, end_ms=now_ms)
        if not klines:
            # Try with swapped base/quote
            alt_sym = _to_binance_symbol(quote, base)
            if alt_sym and alt_sym != binance_sym:
                klines = _fetch_binance_klines(alt_sym, interval=interval, start_ms=cursor, end_ms=now_ms)
                if klines:
                    # Invert prices since we swapped base/quote
                    rows = _binance_klines_to_ohlcv(klines)
                    for r in rows:
                        if r["close"] > 0:
                            r["open"] = 1.0 / r["open"] if r["open"] > 0 else 0.0
                            r["close"] = 1.0 / r["close"]
                            orig_high = r["high"]
                            r["high"] = 1.0 / r["low"] if r["low"] > 0 else 0.0
                            r["low"] = 1.0 / orig_high if orig_high > 0 else 0.0
                    all_rows.extend(rows)
                    if len(klines) < _BINANCE_MAX_LIMIT:
                        break
                    cursor = int(klines[-1][0]) + 1
                    time.sleep(_REQUEST_DELAY)
                    continue
            break

        rows = _binance_klines_to_ohlcv(klines)
        all_rows.extend(rows)

        if len(klines) < _BINANCE_MAX_LIMIT:
            break
        cursor = int(klines[-1][0]) + 1
        time.sleep(_REQUEST_DELAY)

    return all_rows


def _to_coinbase_product(base: str, quote: str) -> Optional[str]:
    """Convert pair to Coinbase product ID format."""
    b = _normalize_token(base)
    q = _normalize_token(quote)
    if b == q:
        return None
    # Coinbase uses USD not USDC/USDT for primary markets
    if q in ("USDC", "USDT", "DAI", "BUSD"):
        q = "USD"
    return f"{b}-{q}"


def _coinbase_product_candidates(base: str, quote: str) -> List[str]:
    """Generate multiple Coinbase product ID candidates for a pair."""
    primary = _to_coinbase_product(base, quote)
    candidates = []
    if primary:
        candidates.append(primary)
    # Also try with USD as quote if not already
    b = _normalize_token(base)
    if f"{b}-USD" not in candidates:
        candidates.append(f"{b}-USD")
    return candidates


def download_pair_coinbase(
    base: str,
    quote: str,
    *,
    days_back: int = 90,
    granularity: int = 300,  # 5 minutes
) -> List[Dict[str, Any]]:
    """
    Download OHLCV from Coinbase Exchange API (free, no key, US-accessible).
    Coinbase limits to 300 candles per request, so we paginate.
    """
    candidates = _coinbase_product_candidates(base, quote)
    if not candidates:
        return []

    # Try each product candidate until one works
    product_id = None
    for candidate in candidates:
        test_url = f"https://api.exchange.coinbase.com/products/{candidate}/candles"
        try:
            test_resp = requests.get(test_url, params={"granularity": str(granularity)}, timeout=10)
            if test_resp.status_code == 200:
                product_id = candidate
                break
        except Exception:
            continue
        time.sleep(_REQUEST_DELAY)
    if not product_id:
        return []

    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    now = int(time.time())
    start = now - (days_back * 86400)
    all_rows: List[Dict[str, Any]] = []
    cursor = start
    max_candles = 300

    while cursor < now:
        end = min(cursor + granularity * max_candles, now)
        params = {
            "start": datetime.fromtimestamp(cursor, tz=timezone.utc).isoformat(),
            "end": datetime.fromtimestamp(end, tz=timezone.utc).isoformat(),
            "granularity": str(granularity),
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                break
            data = resp.json()
        except Exception:
            break

        if not data:
            cursor = end
            time.sleep(_REQUEST_DELAY)
            continue

        for candle in data:
            # Coinbase format: [time, low, high, open, close, volume]
            try:
                vol = float(candle[5])
                # Coinbase doesn't provide buy/sell split; estimate from
                # candle direction — attribute more volume to the dominant side.
                o, c = float(candle[3]), float(candle[4])
                if o > 0 and c > 0:
                    ratio = min(max((c - o) / o, -0.5), 0.5)  # clamp
                    buy_frac = 0.5 + ratio  # bullish → more buy, bearish → more sell
                else:
                    buy_frac = 0.5
                all_rows.append({
                    "timestamp": int(candle[0]),
                    "open": o,
                    "high": float(candle[2]),
                    "low": float(candle[1]),
                    "close": c,
                    "net_volume": vol,
                    "buy_volume": vol * buy_frac,
                    "sell_volume": vol * (1.0 - buy_frac),
                })
            except (IndexError, ValueError, TypeError):
                continue

        cursor = end
        time.sleep(_REQUEST_DELAY)

    # Coinbase returns newest first, sort by timestamp
    all_rows.sort(key=lambda r: r["timestamp"])
    return all_rows


def _fetch_coingecko_ohlc(
    coin_id: str,
    vs_currency: str = "usd",
    days: int = 90,
) -> List[Dict[str, Any]]:
    """Fetch OHLC from CoinGecko (free, no key, 4-hourly granularity for >30 days)."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": vs_currency, "days": str(days)}
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception:
        return []

    rows = []
    for candle in data:
        # CoinGecko OHLC: [timestamp_ms, open, high, low, close]
        try:
            o, h, l, c = float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4])
            # CoinGecko doesn't include volume.  Estimate a synthetic proxy
            # from candle range so downstream volume features aren't all zero.
            spread = (h - l) / max(o, 1e-12)
            synthetic_vol = spread * 1000.0  # scaled proxy
            buy_frac = 0.5 + min(max((c - o) / max(o, 1e-12), -0.5), 0.5) if o > 0 else 0.5
            rows.append({
                "timestamp": int(candle[0]) // 1000,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "net_volume": synthetic_vol,
                "buy_volume": synthetic_vol * buy_frac,
                "sell_volume": synthetic_vol * (1.0 - buy_frac),
            })
        except (IndexError, ValueError, TypeError):
            continue
    return rows


def download_pair_coingecko(
    base: str,
    quote: str,
    *,
    days: int = 90,
) -> List[Dict[str, Any]]:
    """Fallback: download OHLC from CoinGecko when Binance doesn't have the pair."""
    base_id = _COINGECKO_IDS.get(base.upper()) or _COINGECKO_IDS.get(_normalize_token(base))
    if not base_id:
        return []
    vs = "usd" if quote.upper() in ("USDC", "USDT", "DAI", "BUSD", "USD") else quote.lower()
    return _fetch_coingecko_ohlc(base_id, vs_currency=vs, days=days)


def download_pair(
    symbol: str,
    *,
    days_back: int = 90,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Download OHLCV for a pair symbol (e.g., "WETH-USDC").
    Tries Binance first, falls back to CoinGecko.
    Returns (source, rows).
    """
    parts = symbol.upper().replace("/", "-").split("-")
    if len(parts) != 2:
        return ("error", [])
    base, quote = parts

    # Try Coinbase first (US-accessible, good granularity)
    rows = download_pair_coinbase(base, quote, days_back=days_back)
    if rows:
        log_message("cex-fallback", f"Coinbase: {len(rows)} candles for {symbol}")
        return ("coinbase", rows)

    # Try Binance (best granularity, may be geo-restricted)
    rows = download_pair_binance(base, quote, days_back=days_back)
    if rows:
        log_message("cex-fallback", f"Binance: {len(rows)} candles for {symbol}")
        return ("binance", rows)

    # Fallback to CoinGecko (always available, lower granularity)
    rows = download_pair_coingecko(base, quote, days=days_back)
    if rows:
        log_message("cex-fallback", f"CoinGecko: {len(rows)} candles for {symbol}")
        return ("coingecko", rows)

    return ("none", [])


def save_ohlcv(
    rows: List[Dict[str, Any]],
    symbol: str,
    index: int,
    chain: str = "base",
    output_root: Optional[Path] = None,
) -> Optional[Path]:
    """Save OHLCV rows to the standard output path for the data loader."""
    if not rows:
        return None
    root = output_root or _DEFAULT_OUTPUT_ROOT
    out_dir = root / chain
    out_dir.mkdir(parents=True, exist_ok=True)
    label = symbol.upper().replace("/", "-")
    path = out_dir / f"{index:04d}_{label}.json"
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        log_message("cex-fallback", f"Failed to save {path}: {exc}", severity="error")
        return None
    return path


def bootstrap_core_pairs(
    *,
    chain: str = "base",
    days_back: int = 90,
    pairs: Optional[List[str]] = None,
    output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Download OHLCV data for core trading pairs from CEX APIs.
    This is the main entry point for bootstrapping the data pipeline.

    Returns a summary dict with counts and any errors.
    """
    target_pairs = pairs or CORE_PAIRS
    root = output_root or _DEFAULT_OUTPUT_ROOT
    results: Dict[str, Any] = {
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "pairs": {},
    }

    # Load pair index to get proper indices
    index_path = _DATA_ROOT / f"pair_index_{chain}.json"
    pair_indices: Dict[str, int] = {}
    if index_path.exists():
        try:
            with index_path.open("r", encoding="utf-8") as fh:
                idx_data = json.load(fh)
            for _addr, meta in idx_data.items():
                sym = str(meta.get("symbol", "")).upper()
                if sym:
                    pair_indices[sym] = int(meta.get("index", 0))
        except Exception:
            pass

    for i, symbol in enumerate(target_pairs):
        symbol_u = symbol.upper()
        index_num = pair_indices.get(symbol_u, 9000 + i)

        # Check if data already exists
        out_dir = root / chain
        existing = list(out_dir.glob(f"*_{symbol_u}.json")) if out_dir.exists() else []
        if existing:
            # Verify it has actual data
            try:
                with existing[0].open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list) and len(data) > 10:
                    results["skipped"] += 1
                    results["pairs"][symbol_u] = "exists"
                    continue
            except Exception:
                pass

        source, rows = download_pair(symbol, days_back=days_back)
        if rows:
            path = save_ohlcv(rows, symbol_u, index_num, chain=chain, output_root=root)
            if path:
                results["downloaded"] += 1
                results["pairs"][symbol_u] = f"{source}:{len(rows)}"
                log_message("cex-fallback", f"Saved {symbol_u}: {len(rows)} candles from {source} → {path}")
            else:
                results["failed"] += 1
                results["pairs"][symbol_u] = "save_failed"
        else:
            results["failed"] += 1
            results["pairs"][symbol_u] = "no_data"
            log_message("cex-fallback", f"No data available for {symbol_u}", severity="warning")

        time.sleep(_REQUEST_DELAY)

    log_message(
        "cex-fallback",
        f"Bootstrap complete: {results['downloaded']} downloaded, {results['skipped']} skipped, {results['failed']} failed",
    )
    return results


def run_cex_fallback_cycle(
    *,
    chain: str = "base",
    days_back: int = 90,
    max_pairs: int = 20,
) -> Dict[str, Any]:
    """
    Called by TokenDownloadSupervisor when OHLCV_CEX_FALLBACK=1.
    Downloads core pairs first, then expands to additional pairs from the index.
    """
    # Start with core pairs
    results = bootstrap_core_pairs(chain=chain, days_back=days_back)

    # If we have capacity, add more pairs from the index
    remaining = max_pairs - results["downloaded"] - results["skipped"]
    if remaining <= 0:
        return results

    index_path = _DATA_ROOT / f"pair_index_{chain}.json"
    if not index_path.exists():
        return results

    try:
        with index_path.open("r", encoding="utf-8") as fh:
            idx_data = json.load(fh)
    except Exception:
        return results

    core_set = {s.upper() for s in CORE_PAIRS}
    extra_pairs = []
    for _addr, meta in idx_data.items():
        sym = str(meta.get("symbol", "")).upper()
        if sym and sym not in core_set:
            extra_pairs.append((sym, int(meta.get("index", 0))))
        if len(extra_pairs) >= remaining:
            break

    for sym, idx in extra_pairs:
        source, rows = download_pair(sym, days_back=days_back)
        if rows:
            path = save_ohlcv(rows, sym, idx, chain=chain)
            if path:
                results["downloaded"] += 1
                results["pairs"][sym] = f"{source}:{len(rows)}"
        else:
            results["failed"] += 1
            results["pairs"][sym] = "no_data"
        time.sleep(_REQUEST_DELAY)

    return results
