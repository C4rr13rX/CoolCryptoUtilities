"""Utility to backfill historical crypto news via the CryptoPanic API.

The archiver respects the public developer rate limits and stores the fetched
articles into ``data/news/cryptopanic_archive.parquet`` so that the training
pipeline can reuse them without hammering the API.  Headlines include basic
metadata (source, impact, etc.) when CryptoPanic exposes it.

Usage (examples)::

    # Backfill the last 30 days for ETH & BTC using defaults
    python -m services.news_archive

    # Explicit range and custom output file
    python -m services.news_archive --symbols ETH,BASE --start 2024-01-01 \
        --end 2024-03-01 --output data/news/custom_archive.parquet

The script only emits data for tokens that the API returns; it never scrapes
sites directly, keeping the workflow compliant with CryptoPanic's terms.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
import requests
from dotenv import load_dotenv


ISO_DATE = "%Y-%m-%d"
ARCHIVE_PATH = Path("data/news/cryptopanic_archive.parquet")


@dataclass
class ArchiveConfig:
    api_token: str
    rate_limit: int
    min_interval: float
    max_pages: int
    page_size: int
    cooldown_sec: int
    window_days: int


class CryptoNewsArchiver:
    def __init__(self, *, output_path: Path = ARCHIVE_PATH) -> None:
        load_dotenv()
        api_token = os.getenv("CRYPTOPANIC_API_KEY", "").strip()
        if not api_token:
            raise RuntimeError("CRYPTOPANIC_API_KEY missing; cannot fetch news")
        self.config = ArchiveConfig(
            api_token=api_token,
            rate_limit=max(1, int(os.getenv("CRYPTOPANIC_MAX_CALLS_PER_MIN", "45"))),
            min_interval=max(0.2, float(os.getenv("CRYPTOPANIC_REQUEST_INTERVAL", "1.2"))),
            max_pages=max(1, int(os.getenv("CRYPTOPANIC_MAX_PAGES", "8"))),
            page_size=max(5, min(100, int(os.getenv("CRYPTOPANIC_PAGE_SIZE", "50")))),
            cooldown_sec=max(60, int(os.getenv("CRYPTOPANIC_SYMBOL_COOLDOWN_SEC", "900"))),
            window_days=max(1, int(os.getenv("CRYPTOPANIC_ARCHIVE_WINDOW_DAYS", "3"))),
        )
        self.session = requests.Session()
        self.output_path = output_path
        self._request_log: deque[float] = deque(maxlen=512)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def backfill(
        self,
        *,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)
        if end <= start:
            raise ValueError("end must be after start")
        tokens = sorted({sym.strip().upper() for sym in symbols if sym.strip()})
        if not tokens:
            raise ValueError("No symbols specified for news archive")

        window = timedelta(days=self.config.window_days)
        cursor = start
        collected_frames: List[pd.DataFrame] = []
        while cursor < end:
            window_end = min(end, cursor + window)
            posts = self._fetch_posts(
                tokens=tokens,
                since_ts=int(cursor.timestamp()),
                until_ts=int(window_end.timestamp()),
            )
            if posts is not None and not posts.empty:
                collected_frames.append(posts)
            cursor = window_end

        if not collected_frames:
            return pd.DataFrame()

        archive_df = pd.concat(collected_frames, ignore_index=True)
        archive_df["headline"] = archive_df["headline"].astype(str)
        archive_df["article"] = archive_df["article"].astype(str)
        archive_df["sentiment"] = archive_df["sentiment"].astype(str)
        archive_df["timestamp"] = archive_df["timestamp"].astype("int64")
        archive_df.sort_values("timestamp", inplace=True)

        # Merge with existing archive
        existing = self._read_archive()
        combined = (
            pd.concat([existing, archive_df], ignore_index=True)
            if not existing.empty
            else archive_df
        )
        combined.drop_duplicates(subset=["timestamp", "headline"], inplace=True)
        combined.sort_values("timestamp", inplace=True)
        combined.reset_index(drop=True, inplace=True)
        self._write_archive(combined)
        return combined

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_archive(self) -> pd.DataFrame:
        if not self.output_path.exists():
            return pd.DataFrame(columns=["timestamp", "headline", "article", "sentiment", "tokens"])
        try:
            return pd.read_parquet(self.output_path)
        except Exception:
            return pd.DataFrame(columns=["timestamp", "headline", "article", "sentiment", "tokens"])

    def _write_archive(self, df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path, index=False)

    def _fetch_posts(
        self,
        *,
        tokens: Sequence[str],
        since_ts: int,
        until_ts: int,
    ) -> Optional[pd.DataFrame]:
        params = {
            "auth_token": self.config.api_token,
            "kind": os.getenv("CRYPTOPANIC_KIND", "news"),
            "filter": os.getenv("CRYPTOPANIC_FILTER", "all"),
            "public": "true",
            "limit": self.config.page_size,
            "currencies": ",".join(tok.lower() for tok in tokens),
        }
        if os.getenv("CRYPTOPANIC_INCLUDE_METADATA", "false").lower() in {"1", "true", "yes"}:
            params["metadata"] = "true"

        endpoint = "https://cryptopanic.com/api/v1/posts/"
        next_url: Optional[str] = endpoint
        pages = 0
        rows: List[Dict[str, object]] = []

        while next_url and pages < self.config.max_pages:
            resp = None
            try:
                self._throttle()
                if next_url == endpoint:
                    resp = self.session.get(next_url, params=params, timeout=20)
                else:
                    resp = self.session.get(next_url, timeout=20)
                resp.raise_for_status()
                payload = resp.json()
            except Exception:
                if resp is not None and 500 <= resp.status_code < 600:
                    time.sleep(self.config.min_interval * 2)
                break

            results = payload.get("results") or []
            for item in results:
                timestamp = self._extract_timestamp(item)
                if timestamp is None or timestamp < since_ts or timestamp > until_ts:
                    continue
                article, sentiment, normalized_tokens = self._normalise_entry(item, tokens)
                if not normalized_tokens:
                    continue
                rows.append(
                    {
                        "timestamp": timestamp,
                        "headline": str(item.get("title") or "Crypto market update")[:256],
                        "article": article,
                        "sentiment": sentiment,
                        "tokens": sorted(normalized_tokens),
                    }
                )

            next_url = payload.get("next")
            pages += 1
            if not results:
                break

        if not rows:
            return None
        return pd.DataFrame(rows)

    def _extract_timestamp(self, item: dict) -> Optional[int]:
        published = item.get("published_at") or item.get("created_at")
        if not published:
            return None
        ts = pd.to_datetime(published, errors="coerce", utc=True)
        if ts is None or pd.isna(ts):
            return None
        return int(ts.timestamp())

    def _normalise_entry(self, item: dict, default_tokens: Sequence[str]) -> tuple[str, str, List[str]]:
        description = item.get("description") or item.get("body") or ""
        sentiment = item.get("sentiment")
        if not sentiment:
            votes = item.get("votes") or {}
            sentiment = votes.get("sentiment") or "neutral"
        currency_tokens: List[str] = []
        for currency in item.get("currencies") or []:
            code = currency.get("code")
            slug = currency.get("slug")
            if code:
                currency_tokens.append(code)
            if slug:
                currency_tokens.append(slug)
        normalized_tokens = {tok.upper() for tok in currency_tokens if tok}
        if not normalized_tokens:
            normalized_tokens = {tok.upper() for tok in default_tokens}

        info_lines: List[str] = []
        metadata = item.get("metadata") or {}
        impact = metadata.get("impact")
        if impact:
            info_lines.append(f"Impact: {impact}")
        confidence = metadata.get("confidence")
        if confidence:
            info_lines.append(f"Confidence: {confidence}")
        labels = metadata.get("labels")
        if isinstance(labels, list) and labels:
            info_lines.append("Labels: " + ", ".join(labels[:5]))
        if item.get("kind"):
            info_lines.append(f"Kind: {item['kind']}")
        domain = item.get("domain")
        if domain:
            info_lines.append(f"Source: {domain}")
        slug = item.get("slug")
        fallback_url = f"https://cryptopanic.com/news/{slug}/" if slug else ""
        source_url = item.get("url") or fallback_url
        if source_url:
            info_lines.append(f"Link: {source_url}")
        info_lines.append("Focus tokens: " + ", ".join(sorted(normalized_tokens)))

        article_parts = [description.strip(), "\n".join(info_lines)]
        article_text = "\n\n".join(part for part in article_parts if part).strip()[:2048]
        if not article_text:
            article_text = item.get("title") or ""
        return article_text, str(sentiment or "neutral"), list(normalized_tokens)

    def _throttle(self) -> None:
        now = time.time()
        while self._request_log and now - self._request_log[0] > 60.0:
            self._request_log.popleft()
        if self._request_log:
            elapsed = now - self._request_log[-1]
            if elapsed < self.config.min_interval:
                time.sleep(self.config.min_interval - elapsed)
        if len(self._request_log) >= self.config.rate_limit:
            sleep_for = 60.0 - (now - self._request_log[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._request_log.append(time.time())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill CryptoPanic news into the local archive")
    parser.add_argument("--symbols", default="BTC,ETH", help="Comma-separated list of symbols (default: BTC,ETH)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD). Defaults to CRYPTOPANIC_ARCHIVE_LOOKBACK_DAYS ago.")
    parser.add_argument("--end", help="End date (YYYY-MM-DD). Defaults to now (UTC).")
    parser.add_argument("--output", help="Optional output parquet path")
    return parser.parse_args()


def _default_start() -> datetime:
    lookback_days = max(1, int(os.getenv("CRYPTOPANIC_ARCHIVE_LOOKBACK_DAYS", "30")))
    return datetime.now(timezone.utc) - timedelta(days=lookback_days)


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output) if args.output else ARCHIVE_PATH
    archiver = CryptoNewsArchiver(output_path=output_path)
    start_dt = _default_start() if not args.start else datetime.strptime(args.start, ISO_DATE).replace(tzinfo=timezone.utc)
    end_dt = datetime.now(timezone.utc) if not args.end else datetime.strptime(args.end, ISO_DATE).replace(tzinfo=timezone.utc)
    symbols = [sym.strip().upper() for sym in args.symbols.split(",") if sym.strip()]
    combined = archiver.backfill(symbols=symbols, start=start_dt, end=end_dt)
    print(f"Archived {len(combined)} total articles to {archiver.output_path}")


if __name__ == "__main__":
    main()
