from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from db import TradingDatabase, get_db
from services.news_archive import CryptoNewsArchiver
from services.polite_news_crawler import collect_news as crawl_news
from trading.data_loader import TOKEN_SYNONYMS

try:  # Optional dependency – falls back to lexical scoring when unavailable.
    from textblob import TextBlob
except Exception:  # pragma: no cover - TextBlob is optional in minimal deployments.
    TextBlob = None  # type: ignore[assignment]


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

BASE_CONTEXT_TOKENS: Dict[str, Sequence[str]] = {
    "BTC": ["bitcoin", "btc price", "bitcoin network", "hashrate"],
    "ETH": ["ethereum", "eth price", "ethereum mainnet", "gas fees"],
    "BNB": ["bnb chain", "binance smart chain", "bnb price"],
    "SOL": ["solana", "sol price", "solana transactions"],
    "MATIC": ["polygon", "matic price", "polygon pos"],
    "AVAX": ["avalanche", "avax price", "avax c-chain"],
    "ARB": ["arbitrum", "arb token", "arbitrum network"],
    "OP": ["optimism", "op token", "optimism superchain"],
    "FTM": ["fantom", "ftm price", "fantom opera"],
    "ADA": ["cardano", "ada price", "cardano updates"],
    "USDC": ["usd coin", "circle", "stablecoin regulation"],
    "USDT": ["tether", "usdt issuance", "stablecoin flows"],
    "DAI": ["makerdao", "dai stability", "stablecoin governance"],
}

NEWS_LOG_PREFIX = "news:log"

_POSITIVE_LEXICON = {
    "surge",
    "rally",
    "spike",
    "partnership",
    "upgrade",
    "bullish",
    "gain",
    "record",
    "milestone",
    "adopt",
    "expansion",
    "growth",
    "support",
    "boom",
}
_NEGATIVE_LEXICON = {
    "hack",
    "exploit",
    "breach",
    "lawsuit",
    "regulation",
    "selloff",
    "drop",
    "collapse",
    "bearish",
    "loss",
    "halt",
    "delay",
    "ban",
    "fraud",
    "scam",
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


def _sanitize_symbol_key(value: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9]+", "_", value.upper())
    cleaned = cleaned.strip("_")
    return cleaned or "UNKNOWN"


def _iter_days(start_dt: datetime, end_dt: datetime) -> Iterable[datetime]:
    cursor = start_dt.date()
    last = end_dt.date()
    while cursor <= last:
        yield datetime.combine(cursor, datetime.min.time(), tzinfo=timezone.utc)
        cursor += timedelta(days=1)


def _build_query_terms(symbols: Sequence[str], tokens: Sequence[str]) -> Tuple[List[str], List[str]]:
    base_tokens = sorted({tok.upper() for tok in tokens if tok})
    query_terms: Set[str] = set()
    for symbol in symbols:
        if not symbol:
            continue
        query_terms.add(symbol)
        query_terms.add(symbol.replace("-", " "))
        query_terms.add(symbol.replace("_", " "))
    if len(symbols) > 1:
        query_terms.add(" ".join(symbols))
    for token in base_tokens:
        query_terms.add(token)
        for ctx in BASE_CONTEXT_TOKENS.get(token, []):
            query_terms.add(ctx)
    query_terms = {term.strip() for term in query_terms if term and term.strip()}
    return base_tokens, sorted(query_terms)


def _derive_log_keys(symbols: Sequence[str], tokens: Sequence[str]) -> List[str]:
    keys = {_sanitize_symbol_key(symbol) for symbol in symbols if symbol}
    keys.update(_sanitize_symbol_key(token) for token in tokens if token)
    return sorted(keys)


def _load_seen_entries(
    db: TradingDatabase,
    symbol_keys: Sequence[str],
    start_dt: datetime,
    end_dt: datetime,
) -> Tuple[Set[str], Set[str]]:
    seen_urls: Set[str] = set()
    seen_titles: Set[str] = set()
    for day_dt in _iter_days(start_dt, end_dt):
        day = day_dt.date().isoformat()
        for key in symbol_keys:
            entry = db.get_json(f"{NEWS_LOG_PREFIX}:{key}:{day}") or {}
            seen_urls.update(entry.get("urls") or [])
            seen_titles.update(entry.get("titles") or [])
    return seen_urls, seen_titles


def _group_items_by_day(items: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        timestamp = item.get("timestamp")
        if not timestamp:
            continue
        try:
            ts = int(timestamp)
        except Exception:
            continue
        day = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
        grouped[day].append(item)
    return grouped


def _record_news_attempt(
    db: TradingDatabase,
    symbol_keys: Sequence[str],
    start_dt: datetime,
    end_dt: datetime,
    items: Sequence[Dict[str, Any]],
) -> None:
    grouped = _group_items_by_day(items)
    now_iso = datetime.now(timezone.utc).isoformat()
    for day_dt in _iter_days(start_dt, end_dt):
        day_str = day_dt.date().isoformat()
        day_items = grouped.get(day_str, [])
        for key in symbol_keys:
            existing = db.get_json(f"{NEWS_LOG_PREFIX}:{key}:{day_str}") or {}
            urls = set(existing.get("urls") or [])
            titles = set(existing.get("titles") or [])
            sources = existing.get("sources") or {}
            attempts = int(existing.get("attempts", 0)) + 1
            for item in day_items:
                url = item.get("url")
                title = item.get("title")
                source = (item.get("source") or item.get("origin") or "unknown").lower()
                if url:
                    urls.add(url)
                    source_urls = set(sources.get(source, []))
                    source_urls.add(url)
                    sources[source] = sorted(source_urls)
                if title:
                    titles.add(title)
            payload = {
                "symbol": key,
                "date": day_str,
                "urls": sorted(urls),
                "titles": sorted(titles),
                "sources": sources,
                "attempts": attempts,
                "updated": now_iso,
            }
            db.set_json(f"{NEWS_LOG_PREFIX}:{key}:{day_str}", payload)


def _summarize_sources(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for item in items:
        source = (item.get("source") or item.get("origin") or "unknown").lower()
        counts[source] = counts.get(source, 0) + 1
    return sorted(
        ({"source": key, "count": value} for key, value in counts.items()),
        key=lambda row: (-row["count"], row["source"]),
    )


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
    headline = row.get("headline", "")
    article = row.get("article", "")
    return {
        "timestamp": timestamp,
        "datetime": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
        "title": headline,
        "summary": article,
        "headline": headline,
        "article": article,
        "sentiment": row.get("sentiment", "neutral"),
        "tokens": row.get("tokens", []),
        "url": url,
        "origin": "cryptopanic",
        "source": source,
    }


def _score_sentiment(text: str) -> str:
    """
    Lightweight sentiment classifier used when crawling community sources.
    Prefers TextBlob when available, and falls back to a lexical heuristic so
    we always emit a label without external services.
    """

    content = (text or "").strip()
    if not content:
        return "neutral"
    if TextBlob is not None:
        try:
            polarity = TextBlob(content).sentiment.polarity  # type: ignore[call-arg]
            if polarity >= 0.12:
                return "positive"
            if polarity <= -0.12:
                return "negative"
            return "neutral"
        except Exception:
            pass
    tokens = re.findall(r"[A-Za-z]+", content.lower())
    score = 0
    for token in tokens:
        if token in _POSITIVE_LEXICON:
            score += 1
        elif token in _NEGATIVE_LEXICON:
            score -= 1
    if score > 0:
        return "positive" if score > 1 else "neutral"
    if score < 0:
        return "negative" if score < -1 else "neutral"
    return "neutral"


def _format_crawler_news(records: List[Dict[str, Any]], seed_tokens: Sequence[str]) -> List[Dict[str, Any]]:
    token_map = _build_token_map(seed_tokens)
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
        title = entry.get("title") or entry.get("url") or "Untitled"
        summary = entry.get("description") or ""
        source = entry.get("source") or "crawler"
        origin = entry.get("query_match") or "crawler"
        sentiment = _score_sentiment(f"{title} {summary}")
        tokens = _tokens_from_text(f"{title} {summary} {origin} {source}", token_map, seed_tokens)
        formatted.append(
            {
                "timestamp": int(ts.timestamp()),
                "datetime": ts.isoformat(),
                "title": title,
                "summary": summary,
                "headline": title,
                "article": summary or title,
                "sentiment": sentiment,
                "tokens": tokens,
                "url": entry.get("url"),
                "origin": origin,
                "source": source,
            }
        )
    return formatted


def _normalize_token(candidate: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9]", "", candidate.upper())
    return cleaned if len(cleaned) >= 2 else ""


def _build_token_map(seed_tokens: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for token in seed_tokens:
        normalized = _normalize_token(token)
        if not normalized:
            continue
        mapping[normalized.lower()] = normalized
        mapping[f"${normalized.lower()}"] = normalized
        if normalized.startswith("W") and len(normalized) > 2:
            naked = normalized[1:]
            mapping[naked.lower()] = normalized
            mapping[f"${naked.lower()}"] = normalized
        for synonym in TOKEN_SYNONYMS.get(normalized, []):
            syn = _normalize_token(synonym)
            if syn:
                mapping[syn.lower()] = normalized
    return mapping


def _tokens_from_text(text: str, token_map: Dict[str, str], seed_tokens: Sequence[str]) -> List[str]:
    tokens: Set[str] = set()
    haystack = (text or "").lower()
    for keyword, token in token_map.items():
        if keyword and keyword in haystack:
            tokens.add(token)
    for raw in re.findall(r"\\$?[A-Za-z0-9]{2,12}", text or ""):
        normalized = _normalize_token(raw)
        if not normalized:
            continue
        mapped = token_map.get(normalized.lower())
        tokens.add(mapped or normalized)
    if not tokens and seed_tokens:
        tokens.update([_normalize_token(tok) for tok in seed_tokens[:3] if _normalize_token(tok)])
    return sorted(tokens)


def _persist_free_news(items: Sequence[Dict[str, Any]]) -> None:
    if not items:
        return
    path = Path(os.getenv("FREE_NEWS_PATH", "data/news/free_news.parquet"))
    rows: List[Dict[str, Any]] = []
    for item in items:
        ts = item.get("timestamp")
        try:
            ts_int = int(ts)
        except Exception:
            continue
        headline = (item.get("headline") or item.get("title") or "").strip()
        article = (item.get("article") or item.get("summary") or "").strip()
        tokens = item.get("tokens") or []
        if isinstance(tokens, set):
            tokens = sorted(tokens)
        if not headline or not article or not tokens:
            continue
        rows.append(
            {
                "timestamp": ts_int,
                "headline": headline[:256],
                "article": article[:2048],
                "sentiment": item.get("sentiment") or "neutral",
                "tokens": list(tokens),
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            if not existing.empty:
                df = pd.concat([existing, df], ignore_index=True)
        except Exception:
            pass
    df.drop_duplicates(subset=["timestamp", "headline"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


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
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    if not file_paths:
        return {"items": [], "symbols": [], "start": None, "end": None}

    database = db or get_db()
    trace: List[Dict[str, Any]] = []

    windows: List[FileWindow] = []
    for candidate in file_paths:
        resolved = _resolve_path(candidate)
        windows.append(_load_window(resolved))

    symbols = sorted({window.symbol for window in windows})
    start_ts = min(window.start_ts for window in windows)
    end_ts = max(window.end_ts for window in windows)
    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc) - timedelta(hours=hours_before)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc) + timedelta(hours=hours_after)

    def _trace(stage: str, level: str = "info", **details: Any) -> None:
        entry = {"stage": stage, "level": level, **details}
        trace.append(entry)
        if progress_cb:
            try:
                progress_cb(dict(entry))
            except Exception:
                pass

    _trace(
        "resolved_windows",
        files=len(windows),
        symbols=symbols,
        window_start=start_dt.isoformat(),
        window_end=end_dt.isoformat(),
    )

    cache_key = _cache_key(symbols, start_dt, end_dt)
    cached = database.get_json(cache_key)
    if cached and cached.get("expires", 0) > time.time():
        payload = cached["payload"]
        _trace(
            "cache_hit",
            articles=len(payload.get("items", [])),
            sources=len(payload.get("sources", [])),
        )
        return payload

    tokens = _expand_tokens(symbols)
    api_tokens, query_terms = _build_query_terms(symbols, tokens or symbols)
    api_tokens = api_tokens or (symbols if symbols else ["BTC"])
    symbol_keys = _derive_log_keys(symbols, api_tokens)
    seen_urls, seen_titles = _load_seen_entries(database, symbol_keys, start_dt, end_dt)
    for window in windows:
        _trace(
            "window_loaded",
            file=str(window.path),
            symbol=window.symbol,
            start=window.start_dt.isoformat(),
            end=window.end_dt.isoformat(),
            span_hours=round((window.end_ts - window.start_ts) / 3600.0, 2),
        )
    _trace(
        "token_plan",
        tokens=api_tokens,
        query_preview=query_terms[:12],
        query_count=len(query_terms),
    )

    crypto_items: List[Dict[str, Any]] = []
    try:
        archiver = CryptoNewsArchiver()
        _trace("cryptopanic_start", tokens=api_tokens)
        news_df = archiver.collect_window(symbols=api_tokens, start=start_dt, end=end_dt, persist=False)
        if not news_df.empty:
            crypto_items = [_format_crypto_news(row) for row in news_df.to_dict("records")]
        _trace("cryptopanic_complete", articles=len(crypto_items))
        if crypto_items:
            for idx, item in enumerate(crypto_items[:2]):
                _trace(
                    "cryptopanic_sample",
                    title=item.get("title"),
                    source=item.get("source"),
                    timestamp=item.get("timestamp"),
                )
    except RuntimeError as exc:
        crypto_items = []
        _trace("cryptopanic_error", level="error", error=str(exc))
    except Exception as exc:
        crypto_items = []
        _trace("cryptopanic_error", level="error", error=str(exc))

    crawler_items: List[Dict[str, Any]] = []
    try:
        _trace("crawler_start", terms=query_terms)
        crawler_records = crawl_news(queries=query_terms, start=start_dt, end=end_dt, max_pages=160)
        crawler_items = _format_crawler_news(crawler_records, seed_tokens=api_tokens)
        _trace("crawler_complete", articles=len(crawler_items))
        if crawler_items:
            for idx, item in enumerate(crawler_items[:2]):
                _trace(
                    "crawler_sample",
                    title=item.get("title"),
                    origin=item.get("origin"),
                    timestamp=item.get("timestamp"),
                )
    except Exception as exc:
        crawler_items = []
        _trace("crawler_error", level="error", error=str(exc))

    total_attempted = len(crypto_items) + len(crawler_items)
    if crawler_items:
        _persist_free_news(crawler_items)
    dedup_skipped = 0
    combined: Dict[str, Dict[str, Any]] = {}
    for item in crypto_items + crawler_items:
        url = item.get("url")
        title = item.get("title")
        if url and url in seen_urls:
            dedup_skipped += 1
            continue
        if title and title in seen_titles:
            dedup_skipped += 1
            continue
        key = url or title or f"{item['origin']}:{item['timestamp']}"
        existing = combined.get(key)
        if existing and item["timestamp"] <= existing["timestamp"]:
            continue
        combined[key] = item
        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)

    items = sorted(combined.values(), key=lambda row: row["timestamp"], reverse=True)
    _trace(
        "dedupe_complete",
        attempted=total_attempted,
        deduplicated=dedup_skipped,
        retained=len(items),
    )
    for idx, entry in enumerate(items[:3]):
        _trace(
            "news_sample",
            slot=idx + 1,
            title=(entry.get("title") or "")[:180],
            source=entry.get("source"),
            timestamp=entry.get("timestamp"),
            sentiment=entry.get("sentiment"),
        )
    _record_news_attempt(database, symbol_keys, start_dt, end_dt, items)
    payload = {
        "symbols": symbols,
        "tokens": api_tokens,
        "query_terms": query_terms,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "items": items,
        "sources": _summarize_sources(items),
        "meta": {
            "attempted": total_attempted,
            "deduplicated": dedup_skipped,
            "returned": len(items),
        },
        "trace": trace,
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
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    else:
        start = start.astimezone(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    else:
        end = end.astimezone(timezone.utc)

    trace: List[Dict[str, Any]] = []

    def _trace(stage: str, level: str = "info", **details: Any) -> None:
        entry = {"stage": stage, "level": level, **details}
        trace.append(entry)
        if progress_cb:
            try:
                progress_cb(dict(entry))
            except Exception:
                pass

    input_symbols = [str(tok).upper() for tok in tokens if tok]
    symbol_seed = input_symbols or [str(tok).upper() for tok in tokens if tok] or ["BTC"]
    expanded_tokens = _expand_tokens(symbol_seed)
    api_tokens, query_terms = _build_query_terms(symbol_seed, expanded_tokens or symbol_seed)
    api_tokens = api_tokens or symbol_seed
    symbol_keys = _derive_log_keys(symbol_seed, api_tokens)
    _trace(
        "resolved_terms",
        tokens=api_tokens,
        start=start.isoformat(),
        end=end.isoformat(),
    )

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

    seen_urls, seen_titles = _load_seen_entries(database, symbol_keys, start, end)

    try:
        archiver = CryptoNewsArchiver()
        _trace("cryptopanic_start", tokens=api_tokens)
        crypto_df = archiver.collect_window(symbols=api_tokens, start=start, end=end, persist=False)
        crypto_items = [_format_crypto_news(row) for row in crypto_df.to_dict("records")] if not crypto_df.empty else []
        _trace("cryptopanic_complete", articles=len(crypto_items))
    except Exception as exc:
        crypto_items = []
        _trace("cryptopanic_error", level="error", error=str(exc))

    crawler_records = crawl_news(
        queries=query_terms or api_tokens,
        start=start,
        end=end,
        max_pages=max_pages or 200,
    )
    crawler_items = _format_crawler_news(crawler_records, seed_tokens=api_tokens)
    _trace("crawler_complete", articles=len(crawler_items))
    if crawler_items:
        _persist_free_news(crawler_items)

    total_attempted = len(crypto_items) + len(crawler_items)
    dedup_skipped = 0
    combined: Dict[str, Dict[str, Any]] = {}
    for item in crypto_items + crawler_items:
        url = item.get("url")
        title = item.get("title")
        if url and url in seen_urls:
            dedup_skipped += 1
            continue
        if title and title in seen_titles:
            dedup_skipped += 1
            continue
        key = url or f"{item['origin']}:{item['timestamp']}"
        existing = combined.get(key)
        if existing and item["timestamp"] <= existing["timestamp"]:
            continue
        combined[key] = item
        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)

    items = sorted(combined.values(), key=lambda row: row["timestamp"], reverse=True)
    if query:
        query_lower = query.lower()
        items = [item for item in items if query_lower in (item["title"] + " " + item.get("summary", "")).lower()]

    _trace(
        "dedupe_complete",
        attempted=total_attempted,
        deduplicated=dedup_skipped,
        retained=len(items),
    )

    _record_news_attempt(database, symbol_keys, start, end, items)
    result = {
        "symbols": input_symbols or api_tokens,
        "tokens": api_tokens,
        "query_terms": query_terms,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "items": items,
        "sources": _summarize_sources(items),
        "meta": {
            "attempted": total_attempted,
            "deduplicated": dedup_skipped,
            "returned": len(items),
        },
        "trace": trace,
    }
    if cache_key:
        database.set_json(cache_key, {"expires": time.time() + cache_ttl_sec, "payload": result})
    return result
