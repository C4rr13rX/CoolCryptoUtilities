"""
Polite crypto news crawler used by the model lab to complement API-based news.

This module is inspired by the `polite_crypto_crawler.py` proof-of-concept
shared in the product brief.  The core goals remain the same:

* honour robots.txt directives and per-host crawl delays
* keep concurrency low and jitter requests to avoid hammering any origin
* focus on well-known, opt-in community sources that tolerate light scraping
* filter links by the requested query terms and time window

The crawler only touches a very small set of community-operated websites and
respects an allow-list coupled with an explicit ToS deny-list.  It favours
structured sources (sitemaps, index pages) to avoid crawling at depth.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
import random
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urldefrag, urlparse

import aiohttp
import async_timeout

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


DEFAULT_CONFIG: Dict[str, Any] = {
    "user_agent": "PoliteResearchBot/1.2 (contact: research@carrierx.dev)",
    "max_concurrency": 6,
    "per_host_default_qps": 1.0,
    "timeout_sec": 18,
    "max_retries": 3,
    "respect_tos_blocklist": [
        r"(^|\.)coindesk\.com$",
        r"(^|\.)cointelegraph\.com$",
    ],
    "allowed_hosts": [
        r"(^|\.)99bitcoins\.com$",
        r"(^|\.)bitcointalk\.org$",
        r"(^|\.)binance\.com$",
        r"(^|\.)bankless\.ghost\.io$",
        r"(^|\.)banklesshq\.com$",
        r"(^|\.)lists\.metzdowd\.com$",
        r"(^|\.)bitcoinmagazine\.com$",
        r"(^|\.)blog\.ethereum\.org$",
        r"(^|\.)ethereum\.org$",
        r"(^|\.)blockchain\.news$",
        r"(^|\.)coinbase\.com$",
        r"(^|\.)coinbureau\.com$",
        r"(^|\.)kraken\.com$",
        r"(^|\.)btctimes\.com$",
        r"(^|\.)decrypt\.co$",
        r"(^|\.)cryptobriefing\.com$",
        r"(^|\.)news\.bitcoin\.com$",
        r"(^|\.)coinjournal\.net$",
        r"(^|\.)cryptoslate\.com$",
        r"(^|\.)dlnews\.com$",
        r"(^|\.)coingecko\.com$",
        r"(^|\.)thedefiant\.io$",
        r"(^|\.)messari\.io$",
        r"(^|\.)chain\.link$",
        r"(^|\.)u\.today$",
        r"(^|\.)fxstreet\.com$",
        r"(^|\.)coinmarketcap\.com$",
        r"(^|\.)coinmarketcal\.com$",
        r"(^|\.)gemini\.com$",
        r"(^|\.)coincheckup\.com$",
        r"(^|\.)okx\.com$",
        r"(^|\.)bitfinex\.com$",
        r"(^|\.)bitstamp\.net$",
        r"(^|\.)livebitcoinnews\.com$",
        r"(^|\.)ambcrypto\.com$",
        r"(^|\.)beincrypto\.com$",
        r"(^|\.)coinspeaker\.com$",
        r"(^|\.)cryptopotato\.com$",
        r"(^|\.)blockworks\.co$",
        r"(^|\.)bitcoinist\.com$",
        r"(^|\.)coingape\.com$",
        r"(^|\.)coinpedia\.org$",
        r"(^|\.)cryptonews\.com$",
        r"(^|\.)coinedition\.com$",
        r"(^|\.)finbold\.com$",
        r"(^|\.)coinchapter\.com$",
        r"(^|\.)blockonomi\.com$",
        r"(^|\.)cryptoglobe\.com$",
        r"(^|\.)newsbtc\.com$",
        r"(^|\.)cryptopolitan\.com$",
        r"(^|\.)zycrypto\.com$",
        r"(^|\.)crypto-news-flash\.com$",
        r"(^|\.)btcmanager\.com$",
        r"(^|\.)altcoinbuzz\.io$",
        r"(^|\.)cryptodaily\.co\.uk$",
        r"(^|\.)coinquora\.com$",
        r"(^|\.)cryptonomist\.ch$",
        r"(^|\.)bsc\.news$",
        r"(^|\.)coinbold\.io$",
        r"(^|\.)coincu\.com$",
        r"(^|\.)cryptoninjas\.net$",
        r"(^|\.)dailyhodl\.com$",
        r"(^|\.)dailycoin\.com$",
        r"(^|\.)coinfomania\.com$",
        r"(^|\.)cryptotimes\.io$",
        r"(^|\.)cryptoticker\.io$",
        r"(^|\.)coinstatics\.com$",
        r"(^|\.)cryptonewsz\.com$",
        r"(^|\.)cryptomode\.com$",
        r"(^|\.)nftplazas\.com$",
        r"(^|\.)thecoinrepublic\.com$",
        r"(^|\.)thecryptobasic\.com$",
        r"(^|\.)tokenhell\.com$",
        r"(^|\.)bitcoinwarrior\.net$",
        r"(^|\.)cryptonewsland\.com$",
        r"(^|\.)crypto\.news$",
        r"(^|\.)nulltx\.com$",
        r"(^|\.)ethereumworldnews\.com$",
    ],
    "include_patterns": [r".+"],
    "exclude_patterns": [
        r".*\.(jpg|jpeg|png|gif|svg|webp|ico|css|mp4|mp3|webm|woff2?|ttf|eot)(\?.*)?$",
        r".*\?(replytocom|utm_|fbclid|gclid|ref|share)=.*",
        r".*#.*",
    ],
    "politeness": {"min_delay_sec": 0.5, "max_delay_sec": 4.0},
    "wordpress": {"pages_per_term": 3, "per_page": 40},
    "max_pages": 400,
    "allowed_statuses": [200, 304],
    "cache_dir": Path("storage/news_cache"),
    "db_path": Path("storage/news_cache/polite_crawler.sqlite3"),
}


def _ensure_directories(cfg: Dict[str, Any]) -> None:
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg["db_path"]).parent.mkdir(parents=True, exist_ok=True)


def _now_ts() -> int:
    return int(time.time())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ts_to_iso(ts: float) -> str:
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except Exception:
        dt = datetime.now(timezone.utc)
    return dt.isoformat()


def _iso_to_ts(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    except Exception:
        return None


def _iso_in_window(value: Optional[str], start_ts: float, end_ts: float) -> bool:
    if not value:
        return True
    ts = _iso_to_ts(value)
    if ts is None:
        return True
    return start_ts <= ts <= end_ts


# ---------------------------------------------------------------------------
# Robots policy & per-host throttling
# ---------------------------------------------------------------------------


class RobotsPolicy:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.rules: Dict[str, Tuple[str, Optional[float]]] = {}

    async def ensure_loaded(self, session: aiohttp.ClientSession, host: str) -> None:
        if host in self.rules:
            return
        robots_url = f"https://{host}/robots.txt"
        text, crawl_delay = "", None
        try:
            async with async_timeout.timeout(self.cfg["timeout_sec"]):
                async with session.get(
                    robots_url, headers={"User-Agent": self.cfg["user_agent"]}
                ) as resp:
                    if resp.status == 200 and int(resp.headers.get("Content-Length", "0")) < 2_000_000:
                        text = await resp.text(errors="ignore")
                        agent = None
                        for line in text.splitlines():
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            lower = line.lower()
                            if lower.startswith("user-agent"):
                                try:
                                    agent = line.split(":", 1)[1].strip()
                                except Exception:
                                    agent = None
                            elif lower.startswith("crawl-delay") and agent in {"*", self.cfg["user_agent"]}:
                                try:
                                    crawl_delay = float(line.split(":", 1)[1].strip())
                                except Exception:
                                    crawl_delay = None
        except Exception:
            text, crawl_delay = "", None
        self.rules[host] = (text, crawl_delay)

    def allowed(self, host: str, url: str) -> bool:
        text, _ = self.rules.get(host, ("", None))
        if not text:
            return True
        block_rules: List[str] = []
        current_agent: Optional[str] = None
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lower = line.lower()
            if lower.startswith("user-agent"):
                try:
                    current_agent = line.split(":", 1)[1].strip()
                except Exception:
                    current_agent = None
            elif lower.startswith("disallow") and current_agent in {"*", self.cfg["user_agent"]}:
                try:
                    path = line.split(":", 1)[1].strip()
                    if path:
                        block_rules.append(path)
                except Exception:
                    continue
        path = urlparse(url).path or "/"
        return not any(path.startswith(rule) for rule in block_rules)

    def crawl_delay(self, host: str) -> Optional[float]:
        return self.rules.get(host, ("", None))[1]


class HostGate:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.next_time: Dict[str, float] = {}

    async def wait(self, host: str, robots_delay: Optional[float]) -> None:
        base_delay = robots_delay if robots_delay is not None else 1.0 / self.cfg["per_host_default_qps"]
        base_delay = max(self.cfg["politeness"]["min_delay_sec"], min(self.cfg["politeness"]["max_delay_sec"], base_delay))
        ready_at = self.next_time.get(host, 0.0)
        now = time.time()
        if ready_at > now:
            await asyncio.sleep(ready_at - now)
        jitter = random.uniform(0.05, 0.25)
        self.next_time[host] = time.time() + base_delay + jitter


# ---------------------------------------------------------------------------
# Crawler storage (SQLite)
# ---------------------------------------------------------------------------


DDL_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS urls(
  id INTEGER PRIMARY KEY,
  url TEXT UNIQUE NOT NULL,
  host TEXT NOT NULL,
  discovered_at INTEGER NOT NULL,
  fetched_at INTEGER,
  status INTEGER,
  etag TEXT,
  last_modified TEXT,
  sha256 TEXT,
  size_bytes INTEGER,
  error TEXT,
  hint_date TEXT,
  query_match TEXT,
  robots_allowed INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_urls_host ON urls(host);
CREATE INDEX IF NOT EXISTS idx_urls_fetched ON urls(fetched_at);
CREATE TABLE IF NOT EXISTS robots(
  host TEXT PRIMARY KEY,
  robots_txt TEXT,
  fetched_at INTEGER,
  crawl_delay REAL
);
"""


class Store:
    def __init__(self, path: Path) -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        for stmt in DDL_SQL.strip().split(";"):
            s = stmt.strip()
            if s:
                self.conn.execute(s)
        self.conn.commit()

    def upsert_url(self, url: str, host: str, *, hint_date: Optional[str], query_match: Optional[str]) -> None:
        try:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO urls(url, host, discovered_at, hint_date, query_match)
                VALUES (?, ?, ?, ?, ?)
                """,
                (url, host, _now_ts(), hint_date, query_match),
            )
            if hint_date or query_match:
                self.conn.execute(
                    """
                    UPDATE urls
                    SET hint_date=COALESCE(hint_date, ?),
                        query_match=COALESCE(query_match, ?)
                    WHERE url=?
                    """,
                    (hint_date, query_match, url),
                )
            self.conn.commit()
        except Exception:
            pass

    def next_batch(self, limit: int) -> List[str]:
        rows = self.conn.execute(
            """
            SELECT url FROM urls
            WHERE fetched_at IS NULL AND robots_allowed=1
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [row[0] for row in rows]

    def set_robots_allowed(self, url: str, allowed: bool) -> None:
        self.conn.execute("UPDATE urls SET robots_allowed=? WHERE url=?", (1 if allowed else 0, url))
        self.conn.commit()

    def mark_result(
        self,
        url: str,
        *,
        status: Optional[int],
        etag: Optional[str],
        last_modified: Optional[str],
        sha256: Optional[str],
        size_bytes: Optional[int],
        error: Optional[str],
    ) -> None:
        self.conn.execute(
            """
            UPDATE urls
            SET fetched_at=?, status=?, etag=?, last_modified=?, sha256=?, size_bytes=?, error=?
            WHERE url=?
            """,
            (_now_ts(), status, etag, last_modified, sha256, size_bytes, error, url),
        )
        self.conn.commit()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


ALLOW_RE = [re.compile(p, re.I) for p in DEFAULT_CONFIG["allowed_hosts"]]
INCLUDE_RE = [re.compile(p, re.I) for p in DEFAULT_CONFIG["include_patterns"]]
EXCLUDE_RE = [re.compile(p, re.I) for p in DEFAULT_CONFIG["exclude_patterns"]]
TOS_BLOCK_RE = [re.compile(p, re.I) for p in DEFAULT_CONFIG["respect_tos_blocklist"]]


def _host_allowed(host: str) -> bool:
    if any(pattern.search(host) for pattern in TOS_BLOCK_RE):
        return False
    return any(pattern.search(host) for pattern in ALLOW_RE)


def _url_allowed(url: str) -> bool:
    if any(pattern.match(url) for pattern in EXCLUDE_RE):
        return False
    return any(pattern.match(url) for pattern in INCLUDE_RE)


def _normalise(url: str) -> str:
    url, _ = urldefrag(url)
    return url


TITLE_RE = re.compile(r"(?is)<title[^>]*>(.*?)</title>")
META_DESC_RE = re.compile(r'(?is)<meta\s+[^>]*?(name|property)\s*=\s*["\']description["\'][^>]*?>')
META_CONTENT_RE = re.compile(r'content\s*=\s*["\'](.*?)["\']', re.I)
DATE_META_RE = re.compile(
    r'(?is)<meta[^>]+(property|name)=["\'](article:published_time|og:updated_time|pubdate|publishdate|date|dc.date|parsely-pub-date)["\'][^>]*?>'
)
TIME_TAG_RE = re.compile(r'(?is)<time[^>]+datetime=["\'](.*?)["\']')
DATE_SPAN_RE = re.compile(r'(?is)<span[^>]+class=["\'][^"\']*(date|time|published)[^"\']*["\'][^>]*>(.*?)</span>')
LINK_RE = re.compile(r"""(?isx)<a\s+[^>]*?href=(?P<q>['"])(?P<u>.+?)(?P=q)""")


def _clean_text(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    return re.sub(r"\s+", " ", val).strip()


def _wp_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("rendered") or ""
    if not isinstance(value, str):
        return ""
    return _clean_text(value) or ""


def _extract_title_desc(html: str) -> Tuple[Optional[str], Optional[str]]:
    title = None
    match = TITLE_RE.search(html)
    if match:
        title = _clean_text(match.group(1))
    desc = None
    match_desc = META_DESC_RE.search(html)
    if match_desc:
        match_content = META_CONTENT_RE.search(match_desc.group(0))
        if match_content:
            desc = _clean_text(match_content.group(1))
    return title, desc


def _extract_links(base_url: str, html: str) -> List[str]:
    links: List[str] = []
    for match in LINK_RE.finditer(html or ""):
        href = match.group("u").strip()
        if href.startswith(("javascript:", "mailto:", "tel:")):
            continue
        links.append(urljoin(base_url, href))
    return links


def _try_parse_dates(candidates: Iterable[str]) -> Optional[str]:
    for raw in candidates:
        if not raw:
            continue
        text = raw.strip()
        iso = re.search(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(:\d{2})?(Z|[+-]\d{2}:?\d{2})?)?", text)
        if iso:
            return iso.group(0)
        rfc = re.search(
            r"[A-Z][a-z]{2}, \d{1,2} [A-Z][a-z]{2} \d{4} \d{2}:\d{2}(:\d{2})? (?:GMT|UTC|[+-]\d{4})",
            text,
        )
        if rfc:
            return rfc.group(0)
        ymd = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
        if ymd:
            return ymd.group(0)
    return None


def _extract_html_date(html: str) -> Optional[str]:
    candidates: List[str] = []
    for match in DATE_META_RE.finditer(html):
        content = META_CONTENT_RE.search(match.group(0))
        if content:
            candidates.append(content.group(1))
    for match in TIME_TAG_RE.finditer(html):
        candidates.append(match.group(1))
    for match in DATE_SPAN_RE.finditer(html):
        candidates.append(_clean_text(match.group(2) or ""))
    return _try_parse_dates(candidates)


def _parse_rss_timestamp(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        try:
            # fallback: ISO style already
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        except Exception:
            return None


def _renders_query_match(text: str, queries: Sequence[str]) -> bool:
    haystack = text.lower()
    for term in queries:
        needle = term.strip().lower()
        if not needle:
            continue
        if all(part in haystack for part in needle.split()):
            return True
    return False


# ---------------------------------------------------------------------------
# Core crawler
# ---------------------------------------------------------------------------


@dataclass
class CrawlResult:
    url: str
    title: Optional[str]
    description: Optional[str]
    detected_date: Optional[str]
    query_match: Optional[str]
    source: Optional[str]


class PoliteCryptoCrawler:
    """
    Minimal, search-first crawler tailored for the model lab.

    The workflow is:
      1. Seed URLs based on the query terms (Bitcoin Magazine sitemaps,
         Metzdowd Pipermail indexes, Bitcointalk board listings).
      2. Only enqueue URLs that contain at least one of the query terms.
      3. Fetch pages politely, parse lightweight metadata, and return
         structured results.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        _ensure_directories(self.cfg)
        self.store = Store(Path(self.cfg["db_path"]))
        self.robots = RobotsPolicy(self.cfg)
        self.gate = HostGate(self.cfg)
        self.sem = asyncio.Semaphore(self.cfg["max_concurrency"])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def collect_async(
        self,
        *,
        queries: Sequence[str],
        start_ts: float,
        end_ts: float,
        max_pages: Optional[int] = None,
    ) -> List[CrawlResult]:
        terms = [q.strip() for q in queries if q.strip()]
        if not terms:
            return []
        prev = self.cfg["max_pages"]
        if max_pages is not None:
            self.cfg["max_pages"] = max_pages
        try:
            await self._search(terms, start_ts, end_ts)
            results: List[CrawlResult] = []
            async with self._session() as session:
                collected = 0
                while collected < self.cfg["max_pages"]:
                    batch = self.store.next_batch(self.cfg["max_concurrency"] * 2)
                    if not batch:
                        break
                    tasks = [self._guarded_fetch(session, url, terms, start_ts, end_ts) for url in batch]
                    outputs = await asyncio.gather(*tasks)
                    for entry in outputs:
                        if entry is not None:
                            results.append(entry)
                    collected += len(batch)
            return results
        finally:
            self.cfg["max_pages"] = prev

    def collect(
        self,
        *,
        queries: Sequence[str],
        start_ts: float,
        end_ts: float,
        max_pages: Optional[int] = None,
    ) -> List[CrawlResult]:
        try:
            return asyncio.run(
                self.collect_async(
                    queries=queries,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    max_pages=max_pages,
                )
            )
        except RuntimeError:
            # In case we're already inside an event loop (e.g., within Django async view)
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(
                    self.collect_async(
                        queries=queries,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        max_pages=max_pages,
                    )
                )
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _session(self):
        connector = aiohttp.TCPConnector(limit=24, ssl=False)
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=self.cfg["timeout_sec"],
            sock_read=self.cfg["timeout_sec"],
        )
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": self.cfg["user_agent"]},
        ) as session:
            yield session

    async def _search(self, queries: Sequence[str], start_ts: float, end_ts: float) -> None:
        async with self._session() as session:
            tasks: List[asyncio.Task[None]] = []
            tasks.append(asyncio.create_task(self._search_bitcoinmagazine(session, queries, start_ts, end_ts)))
            tasks.append(asyncio.create_task(self._search_metzdowd(session, queries, start_ts, end_ts)))
            tasks.append(asyncio.create_task(self._search_bitcointalk(session, queries, start_ts, end_ts)))
            rss_sources = [
                ("https://blog.ethereum.org/en/feed.xml", "ethereum_blog"),
                ("https://blog.kraken.com/feed", "kraken_blog"),
                ("https://decrypt.co/feed", "decrypt"),
                ("https://cryptobriefing.com/feed", "cryptobriefing"),
                ("https://cryptoslate.com/feed", "cryptoslate"),
                ("https://crypto.news/feed/", "crypto_news"),
                ("https://www.coinspeaker.com/feed/", "coinspeaker"),
                ("https://ambcrypto.com/feed/", "ambcrypto"),
                ("https://beincrypto.com/feed/", "beincrypto"),
                ("https://www.btctimes.com/feed/", "btctimes"),
                ("https://u.today/rss", "utoday"),
                ("https://cryptopotato.com/feed/", "cryptopotato"),
                ("https://blog.chain.link/feed/", "chainlink_blog"),
                ("https://www.dlnews.com/feed/", "dlnews"),
                ("https://newsletter.thedefiant.io/feed", "thedefiant"),
                ("https://messari.io/rss", "messari"),
                ("https://blockworks.co/feed/", "blockworks"),
                ("https://bitcoinist.com/feed/", "bitcoinist"),
                ("https://www.coingape.com/feed/", "coingape"),
                ("https://coinpedia.org/feed/", "coinpedia"),
                ("https://www.cryptonews.com/feed/", "cryptonews"),
                ("https://coinedition.com/feed/", "coinedition"),
                ("https://finbold.com/feed/", "finbold"),
                ("https://www.coinbureau.com/feed/", "coinbureau"),
                ("https://coincu.com/feed/", "coincu"),
                ("https://coincheckup.com/blog/feed/", "coincheckup"),
                ("https://www.coinbold.io/feed/", "coinbold"),
                ("https://coinchapter.com/feed/", "coinchapter"),
                ("https://coinmarketcal.com/en/news/feed", "coinmarketcal"),
                ("https://blockonomi.com/feed/", "blockonomi"),
                ("https://cryptoglobe.com/feed/", "cryptoglobe"),
                ("https://newsbtc.com/feed/", "newsbtc"),
                ("https://cryptopolitan.com/feed/", "cryptopolitan"),
                ("https://zycrypto.com/feed/", "zycrypto"),
                ("https://www.crypto-news-flash.com/feed/", "crypto_news_flash"),
                ("https://btcmanager.com/feed/", "btcmanager"),
                ("https://www.altcoinbuzz.io/feed/", "altcoinbuzz"),
                ("https://cryptodaily.co.uk/feed/", "cryptodaily"),
                ("https://coinquora.com/feed/", "coinquora"),
                ("https://en.cryptonomist.ch/feed/", "cryptonomist"),
                ("https://bsc.news/feed/", "bscnews"),
                ("https://www.cryptoninjas.net/feed/", "cryptoninjas"),
                ("https://dailyhodl.com/feed/", "dailyhodl"),
                ("https://dailycoin.com/feed/", "dailycoin"),
                ("https://coinfomania.com/feed/", "coinfomania"),
                ("https://www.cryptonewsz.com/feed/", "cryptonewsz"),
                ("https://cryptomode.com/feed/", "cryptomode"),
                ("https://www.cryptotimes.io/feed/", "cryptotimes"),
                ("https://cryptoticker.io/en/feed/", "cryptoticker"),
                ("https://www.nftplazas.com/feed/", "nftplazas"),
                ("https://coinstatics.com/feed/", "coinstatics"),
                ("https://nulltx.com/feed/", "nulltx"),
                ("https://99bitcoins.com/feed/", "ninety9bitcoins"),
                ("https://www.fxstreet.com/rss/cryptocurrencies", "fxstreet"),
                ("https://coinmarketcap.com/community/articles/feed/", "coinmarketcap"),
                ("https://www.binance.com/en/blog/rss", "binance_blog"),
                ("https://blog.coinbase.com/feed", "coinbase_blog"),
                ("https://www.gemini.com/blog/rss", "gemini_blog"),
                ("https://www.okx.com/rss/blog", "okx_blog"),
                ("https://blog.bitfinex.com/feed/", "bitfinex_blog"),
                ("https://blog.kraken.com/feed/", "kraken_blog"),
                ("https://www.livebitcoinnews.com/feed/", "livebitcoinnews"),
                ("https://www.bitstamp.net/blog/rss/", "bitstamp_blog"),
                ("https://thecoinrepublic.com/feed/", "coinrepublic"),
                ("https://thecryptobasic.com/feed/", "cryptobasic"),
                ("https://tokenhell.com/feed/", "tokenhell"),
                ("https://bitcoinwarrior.net/feed/", "bitcoinwarrior"),
                ("https://cryptonewsland.com/feed/", "cryptonewsland"),
                ("https://bankless.ghost.io/rss/", "bankless"),
                ("https://www.blockchain.news/rss", "blockchainnews"),
                ("https://ethereumworldnews.com/feed/", "ethereumworldnews"),
            ]
            for feed_url, tag in rss_sources:
                tasks.append(
                    asyncio.create_task(
                        self._search_rss_feed(
                            session,
                            feed_url,
                            queries,
                            tag,
                            start_ts,
                            end_ts,
                        )
                    )
                )
            wordpress_sources = [
                ("https://news.bitcoin.com", "news_bitcoin"),
                ("https://coinjournal.net", "coinjournal"),
                ("https://cryptoslate.com", "cryptoslate_wp"),
                ("https://u.today", "utoday_wp"),
                ("https://ambcrypto.com", "ambcrypto_wp"),
                ("https://beincrypto.com", "beincrypto_wp"),
                ("https://www.coinspeaker.com", "coinspeaker_wp"),
                ("https://cryptopotato.com", "cryptopotato_wp"),
                ("https://blockworks.co", "blockworks_wp"),
                ("https://bitcoinist.com", "bitcoinist_wp"),
                ("https://www.coingape.com", "coingape_wp"),
                ("https://coinpedia.org", "coinpedia_wp"),
                ("https://www.cryptonews.com", "cryptonews_wp"),
                ("https://coinedition.com", "coinedition_wp"),
                ("https://finbold.com", "finbold_wp"),
                ("https://coincu.com", "coincu_wp"),
                ("https://www.coinbold.io", "coinbold_wp"),
                ("https://coinchapter.com", "coinchapter_wp"),
                ("https://coincheckup.com/blog", "coincheckup_wp"),
                ("https://blockonomi.com", "blockonomi_wp"),
                ("https://cryptoglobe.com", "cryptoglobe_wp"),
                ("https://newsbtc.com", "newsbtc_wp"),
                ("https://cryptopolitan.com", "cryptopolitan_wp"),
                ("https://zycrypto.com", "zycrypto_wp"),
                ("https://www.crypto-news-flash.com", "crypto_news_flash_wp"),
                ("https://btcmanager.com", "btcmanager_wp"),
                ("https://www.altcoinbuzz.io", "altcoinbuzz_wp"),
                ("https://cryptodaily.co.uk", "cryptodaily_wp"),
                ("https://coinquora.com", "coinquora_wp"),
                ("https://en.cryptonomist.ch", "cryptonomist_wp"),
                ("https://bsc.news", "bscnews_wp"),
                ("https://www.cryptoninjas.net", "cryptoninjas_wp"),
                ("https://dailyhodl.com", "dailyhodl_wp"),
                ("https://dailycoin.com", "dailycoin_wp"),
                ("https://coinfomania.com", "coinfomania_wp"),
                ("https://www.cryptonewsz.com", "cryptonewsz_wp"),
                ("https://cryptomode.com", "cryptomode_wp"),
                ("https://www.cryptotimes.io", "cryptotimes_wp"),
                ("https://cryptoticker.io", "cryptoticker_wp"),
                ("https://www.nftplazas.com", "nftplazas_wp"),
                ("https://coinstatics.com", "coinstatics_wp"),
                ("https://www.livebitcoinnews.com", "livebitcoinnews_wp"),
                ("https://blog.kraken.com", "kraken_blog_wp"),
                ("https://thecoinrepublic.com", "coinrepublic_wp"),
                ("https://thecryptobasic.com", "cryptobasic_wp"),
                ("https://tokenhell.com", "tokenhell_wp"),
                ("https://bitcoinwarrior.net", "bitcoinwarrior_wp"),
                ("https://cryptonewsland.com", "cryptonewsland_wp"),
                ("https://crypto.news", "crypto_news_wp"),
                ("https://nulltx.com", "nulltx_wp"),
                ("https://99bitcoins.com", "ninety9bitcoins_wp"),
                ("https://ethereumworldnews.com", "ethereumworldnews_wp"),
                ("https://www.blockchain.news", "blockchainnews_wp"),
                ("https://www.coinbureau.com", "coinbureau_wp"),
                ("https://blog.bitfinex.com", "bitfinex_wp"),
                ("https://www.bitstamp.net/blog", "bitstamp_wp"),
            ]
            for base_url, tag in wordpress_sources:
                tasks.append(
                    asyncio.create_task(
                        self._search_wordpress_site(
                            session,
                            base_url,
                            queries,
                            tag,
                            start_ts,
                            end_ts,
                        )
                    )
                )
            await asyncio.gather(*tasks)

    async def _search_bitcoinmagazine(
        self,
        session: aiohttp.ClientSession,
        queries: Sequence[str],
        start_ts: float,
        end_ts: float,
    ) -> None:
        sitemap_urls = [
            "https://bitcoinmagazine.com/sitemap.xml",
            "https://bitcoinmagazine.com/sitemap_index.xml",
            "https://bitcoinmagazine.com/sitemaps/sitemap.xml",
        ]
        allow_patterns = [re.compile(r"^https://bitcoinmagazine\.com/.+", re.I)]
        for sitemap in sitemap_urls:
            text = await self._get_text(session, sitemap)
            if not text:
                continue
            entries = re.findall(r"(?is)<url>(.*?)</url>", text) or re.findall(r"(?is)<sitemap>(.*?)</sitemap>", text)
            for block in entries[:1500]:
                loc = re.search(r"(?is)<loc>\s*(.*?)\s*</loc>", block)
                if not loc:
                    continue
                url = loc.group(1).strip()
                if not any(pattern.match(url) for pattern in allow_patterns):
                    continue
                slug = url.lower()
                if not any(q.lower() in slug for q in queries):
                    continue
                lastmod_val = None
                lm_match = re.search(r"(?is)<lastmod>\s*(.*?)\s*</lastmod>", block)
                if lm_match:
                    lastmod_val = lm_match.group(1).strip()
                host = urlparse(url).hostname or ""
                if _host_allowed(host) and _url_allowed(url):
                    self.store.upsert_url(_normalise(url), host, hint_date=lastmod_val, query_match="bitcoinmagazine")

    async def _search_metzdowd(
        self,
        session: aiohttp.ClientSession,
        queries: Sequence[str],
        start_ts: float,
        end_ts: float,
    ) -> None:
        root = "https://www.metzdowd.com/pipermail/cryptography/"
        index_html = await self._get_text(session, root)
        if not index_html:
            return
        month_pages = re.findall(r'href="(\d{4}-[A-Za-z]+)/"', index_html)
        month_pages = [urljoin(root, page) for page in month_pages][-36:]
        for page in month_pages:
            html = await self._get_text(session, page)
            if not html:
                continue
            for href, text in re.findall(r'href="(\d{6}\.html)"[^>]*>(.*?)</a>', html):
                subject = _clean_text(re.sub("<.*?>", "", text))
                if not subject:
                    continue
                if not any(q.lower() in subject.lower() for q in queries):
                    continue
                url = urljoin(page, href)
                hint_date = None
                match_date = re.search(r"(\d{4}-[A-Za-z]+)", page)
                if match_date:
                    try:
                        hint_date = datetime.strptime(match_date.group(1), "%Y-%B").strftime("%Y-%m-01")
                    except Exception:
                        hint_date = None
                host = urlparse(url).hostname or ""
                if _host_allowed(host) and _url_allowed(url):
                    self.store.upsert_url(_normalise(url), host, hint_date=hint_date, query_match="metzdowd")

    async def _search_bitcointalk(
        self,
        session: aiohttp.ClientSession,
        queries: Sequence[str],
        start_ts: float,
        end_ts: float,
    ) -> None:
        boards = [
            "https://bitcointalk.org/index.php?board=1.0",
            "https://bitcointalk.org/index.php?board=67.0",
        ]
        pages_per_board = 2
        for board in boards:
            base = re.sub(r"(\.\d+)$", "", board)
            for idx in range(pages_per_board):
                start = idx * 40
                url = f"{base}.{start}"
                html = await self._get_text(session, url)
                if not html:
                    continue
                for href, title in re.findall(r'href="([^"]+;topic=[^"]+)"[^>]*>(.*?)</a>', html):
                    clean_title = _clean_text(re.sub("<.*?>", "", title))
                    if not clean_title:
                        continue
                    if not any(q.lower() in clean_title.lower() for q in queries):
                        continue
                    abs_url = urljoin(url, href)
                    host = urlparse(abs_url).hostname or ""
                    if _host_allowed(host) and _url_allowed(abs_url):
                        self.store.upsert_url(_normalise(abs_url), host, hint_date=None, query_match="bitcointalk")

    async def _search_wordpress_site(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        queries: Sequence[str],
        query_tag: str,
        start_ts: float,
        end_ts: float,
    ) -> None:
        api = base_url.rstrip("/") + "/wp-json/wp/v2/posts"
        cfg = self.cfg.get("wordpress", {})
        per_page = int(cfg.get("per_page", 40))
        per_page = max(5, min(per_page, 100))
        pages_per_term = int(cfg.get("pages_per_term", 3))
        pages_per_term = max(1, min(pages_per_term, 6))
        start_iso = _ts_to_iso(start_ts)
        end_iso = _ts_to_iso(end_ts)
        terms = [term.strip() for term in dict.fromkeys(queries) if term.strip()]
        if not terms:
            return
        for term in terms:
            page = 1
            while page <= pages_per_term:
                params = {
                    "search": term,
                    "after": start_iso,
                    "before": end_iso,
                    "orderby": "date",
                    "order": "desc",
                    "page": page,
                    "per_page": per_page,
                    "_fields": "link,date,date_gmt,title,excerpt",
                }
                payload = await self._get_json(session, api, params=params)
                if not isinstance(payload, list) or not payload:
                    break
                queued = 0
                for entry in payload:
                    link = (entry.get("link") or "").strip()
                    if not link:
                        continue
                    normalized = _normalise(link)
                    host = urlparse(normalized).hostname or ""
                    if not host or not _host_allowed(host) or not _url_allowed(normalized):
                        continue
                    published = entry.get("date_gmt") or entry.get("date")
                    detected = None
                    if isinstance(published, str):
                        try:
                            detected = datetime.fromisoformat(published.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
                        except Exception:
                            detected = None
                    if detected and not _iso_in_window(detected, start_ts, end_ts):
                        continue
                    title = _wp_text(entry.get("title"))
                    excerpt = _wp_text(entry.get("excerpt"))
                    summary = " ".join(part for part in (title, excerpt) if part)
                    if summary and not _renders_query_match(summary, [term]):
                        continue
                    self.store.upsert_url(
                        normalized,
                        host,
                        hint_date=detected,
                        query_match=f"{query_tag}:{term.lower()}",
                    )
                    queued += 1
                if queued < per_page:
                    break
                page += 1

    async def _search_rss_feed(
        self,
        session: aiohttp.ClientSession,
        feed_url: str,
        queries: Sequence[str],
        query_tag: str,
        start_ts: float,
        end_ts: float,
        max_items: int = 150,
    ) -> None:
        text = await self._get_text(session, feed_url)
        if not text:
            return
        try:
            root = ET.fromstring(text)
        except Exception:
            return

        count = 0
        for item in root.findall(".//item"):
            if count >= max_items:
                break
            title = (item.findtext("title") or "").strip()
            description = (item.findtext("description") or "").strip()
            summary = f"{title} {description}".strip()
            matched = False
            if summary:
                matched = _renders_query_match(summary, queries)
            if not matched:
                categories = " ".join(cat.text or "" for cat in item.findall("category"))
                if categories:
                    matched = _renders_query_match(categories, queries)
            if not matched:
                continue
            link = (item.findtext("link") or "").strip()
            if not link:
                atom_link = item.find("{http://www.w3.org/2005/Atom}link")
                if atom_link is not None:
                    link = atom_link.attrib.get("href", "").strip()
            if not link:
                continue
            normalized = _normalise(link)
            host = urlparse(normalized).hostname or ""
            if not host or not _host_allowed(host) or not _url_allowed(normalized):
                continue

            hint_date = None
            for tag in ("pubDate", "published", "updated"):
                raw_val = item.findtext(tag)
                hint_date = _parse_rss_timestamp(raw_val)
                if hint_date:
                    break
            if hint_date and not _iso_in_window(hint_date, start_ts, end_ts):
                continue
            self.store.upsert_url(normalized, host, hint_date=hint_date, query_match=query_tag)
            count += 1

    async def _get_text(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        host = urlparse(url).hostname or ""
        if not _host_allowed(host):
            return None
        await self.robots.ensure_loaded(session, host)
        if not self.robots.allowed(host, url):
            return None
        await self.gate.wait(host, self.robots.crawl_delay(host))
        try:
            async with async_timeout.timeout(self.cfg["timeout_sec"]):
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return None
                    return await resp.text(errors="ignore")
        except Exception:
            return None

    async def _get_json(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        host = urlparse(url).hostname or ""
        if not _host_allowed(host):
            return None
        await self.robots.ensure_loaded(session, host)
        if not self.robots.allowed(host, url):
            return None
        await self.gate.wait(host, self.robots.crawl_delay(host))
        try:
            async with async_timeout.timeout(self.cfg["timeout_sec"]):
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        return None
                    return await resp.json(content_type=None)
        except Exception:
            return None

    async def _guarded_fetch(
        self,
        session: aiohttp.ClientSession,
        url: str,
        queries: Sequence[str],
        start_ts: float,
        end_ts: float,
    ) -> Optional[CrawlResult]:
        async with self.sem:
            return await self._fetch(session, url, queries, start_ts, end_ts)

    async def _fetch(
        self,
        session: aiohttp.ClientSession,
        url: str,
        queries: Sequence[str],
        start_ts: float,
        end_ts: float,
    ) -> Optional[CrawlResult]:
        host = urlparse(url).hostname or ""
        if not _host_allowed(host):
            self.store.set_robots_allowed(url, False)
            self.store.mark_result(url, status=None, etag=None, last_modified=None, sha256=None, size_bytes=None, error="host_blocked")
            return None

        await self.robots.ensure_loaded(session, host)
        allowed = self.robots.allowed(host, url)
        self.store.set_robots_allowed(url, allowed)
        if not allowed:
            self.store.mark_result(url, status=None, etag=None, last_modified=None, sha256=None, size_bytes=None, error="robots_disallow")
            return None

        await self.gate.wait(host, self.robots.crawl_delay(host))
        headers = {
            "User-Agent": self.cfg["user_agent"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate",
        }
        attempt = 0
        backoff = 1.5
        body_bytes: Optional[bytes] = None
        response_status: Optional[int] = None
        etag = None
        last_modified = None
        while attempt <= self.cfg["max_retries"]:
            try:
                async with async_timeout.timeout(self.cfg["timeout_sec"] + attempt * 2):
                    async with session.get(url, headers=headers, allow_redirects=True) as resp:
                        response_status = resp.status
                        if resp.status in (429, 503):
                            retry_after = resp.headers.get("Retry-After")
                            sleep_time = float(retry_after) if retry_after and retry_after.isdigit() else min(30, backoff * (attempt + 1))
                            await asyncio.sleep(sleep_time + random.uniform(0.1, 0.4))
                            attempt += 1
                            continue
                        if resp.status not in self.cfg["allowed_statuses"]:
                            self.store.mark_result(
                                url,
                                status=resp.status,
                                etag=None,
                                last_modified=None,
                                sha256=None,
                                size_bytes=None,
                                error=f"status_{resp.status}",
                            )
                            return None
                        if resp.status == 304:
                            self.store.mark_result(
                                url,
                                status=304,
                                etag=None,
                                last_modified=None,
                                sha256=None,
                                size_bytes=None,
                                error=None,
                            )
                            return None
                        body_bytes = await resp.read()
                        etag = resp.headers.get("ETag")
                        last_modified = resp.headers.get("Last-Modified")
                        break
            except asyncio.TimeoutError:
                attempt += 1
            except aiohttp.ClientError:
                attempt += 1
            except Exception as exc:  # pragma: no cover - defensive
                self.store.mark_result(url, status=None, etag=None, last_modified=None, sha256=None, size_bytes=None, error=f"exception:{exc.__class__.__name__}")
                return None

        if body_bytes is None:
            self.store.mark_result(url, status=response_status, etag=None, last_modified=None, sha256=None, size_bytes=None, error="retries_exhausted")
            return None

        content_type = ""
        try:
            content_type = session._default_headers.get("Content-Type", "")
        except Exception:
            content_type = ""

        if content_type and "text/html" not in content_type.lower():
            self.store.mark_result(url, status=response_status, etag=etag, last_modified=last_modified, sha256=None, size_bytes=len(body_bytes), error="unsupported_content_type")
            return None

        try:
            text = body_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text = ""

        title, desc = _extract_title_desc(text)
        detected_date = _extract_html_date(text)
        if not detected_date:
            hint_row = self.store.conn.execute("SELECT hint_date FROM urls WHERE url=?", (url,)).fetchone()
            if hint_row:
                detected_date = hint_row[0]

        if detected_date:
            try:
                parsed = datetime.fromisoformat(detected_date.replace("Z", "+00:00"))
                ts = parsed.timestamp()
                if ts < start_ts - 3 * 86400 or ts > end_ts + 3 * 86400:
                    self.store.mark_result(
                        url, status=response_status, etag=etag, last_modified=last_modified, sha256=_sha256_bytes(body_bytes), size_bytes=len(body_bytes), error="outside_window"
                    )
                    return None
            except Exception:
                pass

        sha = _sha256_bytes(body_bytes)
        self._cache_bytes(url, body_bytes)
        self.store.mark_result(
            url,
            status=response_status,
            etag=etag,
            last_modified=last_modified,
            sha256=sha,
            size_bytes=len(body_bytes),
            error=None,
        )
        query_row = self.store.conn.execute("SELECT query_match FROM urls WHERE url=?", (url,)).fetchone()
        query_match = query_row[0] if query_row and query_row[0] else None
        return CrawlResult(
            url=url,
            title=title,
            description=desc,
            detected_date=detected_date,
            query_match=query_match,
            source=host,
        )

    def _cache_bytes(self, url: str, payload: bytes) -> None:
        digest = _sha256_bytes(url.encode("utf-8"))
        path = Path(self.cfg["cache_dir"]) / f"{digest}.gz"
        try:
            with gzip.open(path, "wb") as handle:
                handle.write(payload)
        except Exception:
            pass


# Convenience ----------------------------------------------------------------


def collect_news(
    *,
    queries: Sequence[str],
    start: datetime,
    end: datetime,
    max_pages: Optional[int] = 200,
) -> List[Dict[str, Any]]:
    crawler = PoliteCryptoCrawler()
    start_ts = start.astimezone(timezone.utc).timestamp()
    end_ts = end.astimezone(timezone.utc).timestamp()
    results = crawler.collect(queries=queries, start_ts=start_ts, end_ts=end_ts, max_pages=max_pages)
    output: List[Dict[str, Any]] = []
    for item in results:
        output.append(
            {
                "url": item.url,
                "title": item.title,
                "description": item.description,
                "detected_date": item.detected_date,
                "query_match": item.query_match,
                "source": item.source,
            }
        )
    return output
