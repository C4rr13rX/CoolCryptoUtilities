from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from db import TradingDatabase, get_db
from services.news_archive import CryptoNewsArchiver
from services.polite_news_crawler import collect_news as crawl_news
from trading.data_loader import TOKEN_SYNONYMS


REPO_ROOT = Path(__file__).resolve().parents[1]
HIST_ROOT = REPO_ROOT / "data" / "historical_ohlcv"

LINK_RE = re.compile(r"Link:\s*(https?://\S+)", re.I)
SOURCE_RE = re.compile(r"Source:\s*([A-Za-z0-9\.\-]+)", re.I)

TOKEN_OVERRIDES: Dict[str, Sequence[str]] = {
    "WETH": ["ETH"],
    "WBTC": ["BTC"],
    "WBNB": ["BNB"],
    "WAVAX": ["AVAX"],
    "WMATIC": ["MATIC"],
    "WFTM": ["FTM"],
}


@dataclass
class FileWindow:
    path: Path
    symbol: str
    start_ts: int
    end_ts: int

    @property
    def start_dt(self) -> datetime:
        return datetime.fromtimestamp(self.start_ts, tz=timezone.utc)

    @property
    def end_dt(self) -> datetime:
        return datetime.fromtimestamp(self.end_ts, tz=timezone.utc)


def _resolve_path(candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return (HIST_ROOT / path).resolve()


def _load_window(file_path: Path) -> FileWindow:
    if not file_path.exists():
        raise FileNotFoundError(f"Historical OHLCV file not found: {file_path}")
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Unable to parse OHLCV file {file_path}: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Historical file {file_path} is empty or malformed.")
    timestamps = [int(entry.get("timestamp", 0)) for entry in payload if isinstance(entry, dict) and entry.get("timestamp")]
    if not timestamps:
        raise ValueError(f"No timestamps found in {file_path}")
    stem = file_path.stem
    symbol = stem.split("_", 1)[-1].upper()
    return FileWindow(
        path=file_path,
        symbol=symbol,
        start_ts=min(timestamps),
        end_ts=max(timestamps),
    )


def _expand_tokens(symbols: Iterable[str]) -> List[str]:
    tokens: List[str] = []
    for symbol in symbols:
        parts = re.split(r"[-_/]", symbol.upper())
        for part in parts:
            if not part:
                continue
            tokens.append(part)
            if part in TOKEN_OVERRIDES:
                tokens.extend(TOKEN_OVERRIDES[part])
            if part in TOKEN_SYNONYMS:
                tokens.extend([tok.upper() for tok in TOKEN_SYNONYMS[part]])
    unique = sorted({tok.upper() for tok in tokens if tok})
    return unique


def _parse_article_link(article_text: str) -> Optional[str]:
    if not article_text:
        return None
    match = LINK_RE.search(article_text)
    if match:
        return match.group(1).strip()
    return None


def _parse_article_source(article_text: str) -> Optional[str]:
    if not article_text:
        return None
    match = SOURCE_RE.search(article_text)
    if match:
        return match.group(1).strip()
    return None


def _format_crypto_news(row: Dict[str, Any]) -> Dict[str, Any]:
    url = _parse_article_link(row.get("article", ""))
    source = _parse_article_source(row.get("article", "")) or "cryptopanic"
    timestamp = int(row.get("timestamp", 0))
    return {
        "timestamp": timestamp,
        "datetime": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
        "title": row.get("headline", ""),
        "summary": row.get("article", ""),
        "sentiment": row.get("sentiment", "neutral"),
        "tokens": row.get("tokens", []),
        "url": url,
        "origin": "cryptopanic",
        "source": source,
    }


def _format_crawler_news(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for entry in records:
        detected = entry.get("detected_date")
        ts = None
        if detected:
            try:
                ts = datetime.fromisoformat(detected.replace("Z", "+00:00"))
            except Exception:
                ts = None
        if ts is None:
            ts = datetime.now(timezone.utc)
        formatted.append(
            {
                "timestamp": int(ts.timestamp()),
                "datetime": ts.isoformat(),
                "title": entry.get("title") or entry.get("url"),
                "summary": entry.get("description") or "",
                "sentiment": "unknown",
                "tokens": [],
                "url": entry.get("url"),
                "origin": entry.get("query_match") or "crawler",
                "source": entry.get("source") or "crawler",
            }
        )
    return formatted


def _cache_key(symbols: Sequence[str], start: datetime, end: datetime) -> str:
    digest_source = "|".join(sorted(symbols)) + f"|{start.isoformat()}|{end.isoformat()}"
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()
    return f"lab_news:{digest}"


def collect_news_for_files(
    file_paths: Sequence[str],
    *,
    db: Optional[TradingDatabase] = None,
    hours_before: int = 12,
    hours_after: int = 12,
    cache_ttl_sec: int = 6 * 3600,
) -> Dict[str, Any]:
    if not file_paths:
        return {"items": [], "symbols": [], "start": None, "end": None}

    database = db or get_db()
    windows: List[FileWindow] = []
    for candidate in file_paths:
        resolved = _resolve_path(candidate)
        windows.append(_load_window(resolved))

    symbols = sorted({window.symbol for window in windows})
    start_ts = min(window.start_ts for window in windows)
    end_ts = max(window.end_ts for window in windows)
    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc) - timedelta(hours=hours_before)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc) + timedelta(hours=hours_after)

    cache_key = _cache_key(symbols, start_dt, end_dt)
    cached = database.get_json(cache_key)
    if cached and cached.get("expires", 0) > time.time():
        return cached["payload"]

    tokens = _expand_tokens(symbols)
    crypto_items: List[Dict[str, Any]] = []
    try:
        archiver = CryptoNewsArchiver()
        news_df = archiver.collect_window(symbols=tokens or symbols, start=start_dt, end=end_dt, persist=False)
        if not news_df.empty:
            crypto_items = [_format_crypto_news(row) for row in news_df.to_dict("records")]
    except RuntimeError:
        crypto_items = []
    except Exception:
        crypto_items = []

    crawler_items: List[Dict[str, Any]] = []
    try:
        query_terms = tokens + [" ".join(symbols)]
        crawler_records = crawl_news(queries=query_terms, start=start_dt, end=end_dt, max_pages=120)
        crawler_items = _format_crawler_news(crawler_records)
    except Exception:
        crawler_items = []

    combined: Dict[str, Dict[str, Any]] = {}
    for item in crypto_items + crawler_items:
        key = item.get("url") or item.get("title")
        if not key:
            key = f"{item['origin']}:{item['timestamp']}"
        if key in combined:
            existing = combined[key]
            if item["timestamp"] > existing["timestamp"]:
                combined[key] = item
        else:
            combined[key] = item

    items = sorted(combined.values(), key=lambda row: row["timestamp"], reverse=True)
    payload = {
        "symbols": symbols,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "items": items,
    }

    database.set_json(cache_key, {"expires": time.time() + cache_ttl_sec, "payload": payload})
    return payload


def collect_news_for_terms(
    *,
    tokens: Sequence[str],
    start: datetime,
    end: datetime,
    query: Optional[str] = None,
    max_pages: Optional[int] = None,
    cache_key: Optional[str] = None,
    cache_ttl_sec: int = 3 * 3600,
) -> Dict[str, Any]:
    tokens = sorted({tok.upper() for tok in tokens if tok})
    database = get_db()
    if cache_key:
        cached = database.get_json(cache_key)
        if cached and cached.get("expires", 0) > time.time():
            results = cached["payload"]
            if query:
                filtered_items = [
                    item
                    for item in results.get("items", [])
                    if query.lower() in (item.get("title", "") + " " + item.get("summary", "")).lower()
                ]
                results = {**results, "items": filtered_items}
            return results

    try:
        archiver = CryptoNewsArchiver()
        crypto_df = archiver.collect_window(symbols=tokens or ["BTC"], start=start, end=end, persist=False)
        crypto_items = [_format_crypto_news(row) for row in crypto_df.to_dict("records")] if not crypto_df.empty else []
    except Exception:
        crypto_items = []

    crawler_records = crawl_news(
        queries=tokens or ["crypto"],
        start=start,
        end=end,
        max_pages=max_pages or 200,
    )
    crawler_items = _format_crawler_news(crawler_records)

    combined: Dict[str, Dict[str, Any]] = {}
    for item in crypto_items + crawler_items:
        key = item.get("url") or f"{item['origin']}:{item['timestamp']}"
        if key in combined and item["timestamp"] <= combined[key]["timestamp"]:
            continue
        combined[key] = item

    items = sorted(combined.values(), key=lambda row: row["timestamp"], reverse=True)
    if query:
        query_lower = query.lower()
        items = [item for item in items if query_lower in (item["title"] + " " + item.get("summary", "")).lower()]

    result = {
        "symbols": tokens,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "items": items,
    }
    if cache_key:
        database.set_json(cache_key, {"expires": time.time() + cache_ttl_sec, "payload": result})
    return result
