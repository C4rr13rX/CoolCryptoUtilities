from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import pandas as pd

from services.polite_news_crawler import collect_news
from trading.data_loader import TOKEN_SYNONYMS


ISO_FORMATS = ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z")


def _parse_date(value: str) -> datetime:
    for fmt in ISO_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _normalize_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    cleaned = re.sub(r"[^0-9A-Za-z]", "", token.upper())
    if len(cleaned) < 2:
        return None
    return cleaned


def _expand_tokens(symbols: Sequence[str], extra: Sequence[str]) -> List[str]:
    tokens: Set[str] = set()
    for symbol in symbols:
        for chunk in re.split(r"[-_/ ]", symbol.upper()):
            normalized = _normalize_token(chunk)
            if normalized:
                tokens.add(normalized)
                tokens.update(TOKEN_SYNONYMS.get(normalized, []))
    for term in extra:
        normalized = _normalize_token(term)
        if normalized:
            tokens.add(normalized)
    return sorted(tokens)


def _extract_tokens(text: str, base_tokens: Iterable[str]) -> List[str]:
    tokens: Set[str] = {tok for tok in base_tokens if tok}
    for word in re.findall(r"[A-Za-z0-9$]{3,}", text):
        normalized = _normalize_token(word)
        if normalized:
            tokens.add(normalized)
    return sorted(tokens)


def _ingest(
    *,
    symbols: Sequence[str],
    extra_terms: Sequence[str],
    start: datetime,
    end: datetime,
    max_pages: int,
) -> pd.DataFrame:
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    if end <= start:
        raise ValueError("End date must be after start date.")
    token_seeds = _expand_tokens(symbols, extra_terms)
    queries = sorted({*symbols, *extra_terms, *(tok.lower() for tok in token_seeds)})
    results = collect_news(queries=queries, start=start, end=end, max_pages=max_pages)
    rows: List[dict] = []
    for item in results:
        detected = item.get("detected_date")
        if not detected:
            continue
        try:
            detected_dt = datetime.fromisoformat(detected.replace("Z", "+00:00"))
        except ValueError:
            continue
        detected_dt = detected_dt.astimezone(timezone.utc)
        timestamp = int(detected_dt.timestamp())
        title = (item.get("title") or "").strip()
        desc = (item.get("description") or "").strip()
        url = item.get("url") or ""
        combined = " ".join(part for part in (title, desc, item.get("query_match") or "") if part)
        tokens = _extract_tokens(combined, token_seeds)
        if not tokens:
            continue
        article = "\n\n".join(part for part in (desc, f"Source: {item.get('source', 'community')}", f"Link: {url}") if part)
        headline = title or (desc[:120] if desc else f"Insight from {item.get('source', 'community')}")
        rows.append(
            {
                "timestamp": timestamp,
                "headline": headline[:256],
                "article": article[:2048],
                "tokens": tokens,
                "sentiment": "neutral",
                "metadata": {
                    "query_match": item.get("query_match"),
                    "source": item.get("source"),
                    "url": url,
                },
            }
        )
    return pd.DataFrame(rows)


def _persist(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        try:
            existing = pd.read_parquet(output)
        except Exception:
            existing = pd.DataFrame()
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)
    if df.empty:
        return
    df.drop_duplicates(subset=["timestamp", "headline"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_parquet(output, index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ethical, free crypto news via the polite crawler.")
    parser.add_argument("--symbols", help="Comma separated list of ticker symbols (e.g. ETH,BTC).", default="")
    parser.add_argument("--terms", help="Additional keywords to boost coverage.", default="")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD or ISO8601).")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD or ISO8601).")
    parser.add_argument("--output", default=os.getenv("ETHICAL_NEWS_PATH", "data/news/ethical_news.parquet"))
    parser.add_argument("--max-pages", type=int, default=200, help="Crawler page limit (default: 200).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    symbols = [sym.strip().upper() for sym in args.symbols.split(",") if sym.strip()]
    extra_terms = [term.strip() for term in args.terms.split(",") if term.strip()]
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    df = _ingest(symbols=symbols, extra_terms=extra_terms, start=start, end=end, max_pages=args.max_pages)
    if df.empty:
        print("No news collected for the requested window.")
        return
    output = Path(args.output)
    _persist(df, output)
    sample = df.tail(3)[["timestamp", "headline"]].to_dict(orient="records")
    print(f"Stored {len(df)} records to {output}")
    print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
