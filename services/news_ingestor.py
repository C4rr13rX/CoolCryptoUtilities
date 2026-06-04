from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from datetime import datetime

import feedparser

try:  # feedparser 6.x exposes util.parse_date, older builds do not
    from feedparser.util import parse_date as _fp_parse_date  # type: ignore
except Exception:  # pragma: no cover - fallback path
    _fp_parse_date = None
import pandas as pd
import requests
from bs4 import BeautifulSoup


@dataclass
class NewsSource:
    name: str
    url: str
    topics: Sequence[str] = ()
    # `kind` switches the fetcher used:
    #   "rss"     — default, feedparser handles RSS/Atom.
    #   "reddit"  — Reddit JSON listing (url = subreddit name like "ethereum").
    #   "hn"      — HackerNews Algolia search (url = query string).
    # All adapters return the same dict shape as _normalize_entry so
    # downstream code is uniform.
    kind: str = "rss"


def _crypto_rss_feeds() -> List["NewsSource"]:
    """Tier 1 + Tier 4 RSS feeds.  All free, no API key.  Some return
    bozo (malformed feed) on bad days — `_fetch_source` already
    silently swallows those and falls back to cache."""
    return [
        # === Tier 0: existing canonical crypto press ===
        NewsSource(name="CoinDesk", url="https://www.coindesk.com/arc/outboundfeeds/rss/", topics=("BTC", "ETH")),
        NewsSource(name="CoinTelegraph", url="https://cointelegraph.com/rss", topics=("L2", "DEFI")),
        NewsSource(name="Blockworks", url="https://blockworks.co/feed", topics=("MACRO", "MARKETS")),
        NewsSource(name="Ethereum Foundation", url="https://blog.ethereum.org/en/feed.xml", topics=("ETH", "STAKING", "L2")),
        NewsSource(name="US Treasury Press", url="https://home.treasury.gov/news/press-releases/rss", topics=("USD", "POLICY")),
        NewsSource(name="IMF Fintech", url="https://www.imf.org/external/pubs/ft/fandd/fintech/rss.xml", topics=("MACRO", "REGULATION")),
        NewsSource(name="GitHub Security Advisories", url="https://github.com/security-advisories.atom", topics=("SECURITY", "DEVOPS")),
        NewsSource(name="Decrypt", url="https://decrypt.co/feed", topics=("NFT", "GAMEFI", "DEFI")),
        NewsSource(name="The Defiant", url="https://thedefiant.io/feed", topics=("DEFI", "STABLECOIN")),
        NewsSource(name="ECB Press", url="https://www.ecb.europa.eu/rss/press.html", topics=("EURO", "MACRO")),
        NewsSource(name="BIS Press", url="https://www.bis.org/rss/press.xml", topics=("CBDC", "CENTRALBANK")),
        NewsSource(name="Federal Reserve Press", url="https://www.federalreserve.gov/feeds/press_all.xml", topics=("USD", "POLICY")),
        NewsSource(name="World Bank Blogs", url="https://blogs.worldbank.org/feed", topics=("MACRO", "DEVELOPMENT")),
        NewsSource(name="Financial Stability Board", url="https://www.fsb.org/feed/press/", topics=("REGULATION", "MACRO")),
        NewsSource(name="Chainalysis Insights", url="https://blog.chainalysis.com/feed", topics=("RISK", "ONCHAIN")),
        NewsSource(name="Bitcoin Magazine", url="https://bitcoinmagazine.com/feed", topics=("BTC", "MINING", "LIGHTNING")),
        NewsSource(name="DeFi Llama Blog", url="https://defillama.com/blog/rss.xml", topics=("DEFI", "TVL", "PROTOCOL")),
        NewsSource(name="Messari", url="https://messari.io/rss/news.xml", topics=("MARKETS", "RESEARCH", "ONCHAIN")),
        NewsSource(name="Dune Analytics Blog", url="https://dune.com/blog/rss.xml", topics=("ONCHAIN", "ANALYTICS")),
        NewsSource(name="Bankless", url="https://bankless.ghost.io/rss/", topics=("ETH", "DEFI", "STAKING")),
        NewsSource(name="The Block", url="https://www.theblock.co/rss.xml", topics=("BTC", "ETH", "MARKETS", "REGULATION")),
        NewsSource(name="CryptoSlate", url="https://cryptoslate.com/feed/", topics=("ALTCOIN", "MARKETS", "NFT")),
        NewsSource(name="Protos", url="https://protos.com/feed/", topics=("MARKETS", "REGULATION", "STABLECOIN")),
        NewsSource(name="SEC Press Releases", url="https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&dateb=&owner=include&count=20&search_text=&output=atom", topics=("REGULATION", "SEC", "ENFORCEMENT")),

        # === Tier 1: additional crypto press (no auth) ===
        NewsSource(name="U.Today",          url="https://u.today/rss",                       topics=("MARKETS", "ALTCOIN")),
        NewsSource(name="BeInCrypto",       url="https://beincrypto.com/feed/",              topics=("ALTCOIN", "DEFI", "NFT")),
        NewsSource(name="Crypto Briefing",  url="https://cryptobriefing.com/feed/",          topics=("DEFI", "L2", "RESEARCH")),
        NewsSource(name="Bitcoinist",       url="https://bitcoinist.com/feed/",              topics=("BTC", "ALTCOIN")),
        NewsSource(name="NewsBTC",          url="https://www.newsbtc.com/feed/",             topics=("BTC", "MARKETS")),
        NewsSource(name="AMBCrypto",        url="https://ambcrypto.com/feed/",               topics=("ALTCOIN", "MARKETS")),
        NewsSource(name="ZyCrypto",         url="https://zycrypto.com/feed/",                topics=("ALTCOIN", "MARKETS")),
        NewsSource(name="CoinGape",         url="https://coingape.com/feed/",                topics=("MARKETS", "REGULATION")),
        NewsSource(name="Crypto.News",      url="https://crypto.news/feed/",                 topics=("MARKETS", "ALTCOIN")),
        NewsSource(name="DLNews",           url="https://www.dlnews.com/arc/outboundfeeds/rss/?outputType=xml", topics=("DEFI", "REGULATION", "RESEARCH")),
        NewsSource(name="Forklog EN",       url="https://forklog.media/feed/",               topics=("ALTCOIN", "MARKETS")),
        NewsSource(name="BSC News",         url="https://www.bsc.news/rss/feed.xml",         topics=("BNB", "DEFI")),
        NewsSource(name="TheNewsCrypto",    url="https://thenewscrypto.com/feed/",           topics=("MARKETS", "ALTCOIN")),
        NewsSource(name="Bitcoin.com News", url="https://news.bitcoin.com/feed/",            topics=("BTC", "MARKETS")),
        NewsSource(name="CryptoPotato",     url="https://cryptopotato.com/feed/",            topics=("MARKETS", "ALTCOIN")),
        NewsSource(name="The Daily Hodl",   url="https://dailyhodl.com/feed/",               topics=("MARKETS", "ALTCOIN")),
        NewsSource(name="CCN",              url="https://www.ccn.com/feed/",                 topics=("MARKETS", "REGULATION")),
        NewsSource(name="DailyCoin",        url="https://dailycoin.com/feed/",               topics=("ALTCOIN", "MARKETS")),
        NewsSource(name="Coincu",           url="https://news.coincu.com/feed/",             topics=("ALTCOIN", "MARKETS")),
        NewsSource(name="Crypto News Z",    url="https://cryptonewsz.com/feed/",             topics=("ALTCOIN", "MARKETS")),

        # === Tier 4 part A: GitHub Atom feeds — protocol activity ===
        NewsSource(name="GH go-ethereum releases",    url="https://github.com/ethereum/go-ethereum/releases.atom",   topics=("ETH", "DEVOPS")),
        NewsSource(name="GH solana-labs releases",    url="https://github.com/solana-labs/solana/releases.atom",     topics=("SOL", "DEVOPS")),
        NewsSource(name="GH bitcoin releases",        url="https://github.com/bitcoin/bitcoin/releases.atom",        topics=("BTC", "DEVOPS")),
        NewsSource(name="GH op-stack releases",       url="https://github.com/ethereum-optimism/optimism/releases.atom", topics=("OP", "L2", "DEVOPS")),
        NewsSource(name="GH arbitrum releases",       url="https://github.com/OffchainLabs/nitro/releases.atom",     topics=("ARB", "L2", "DEVOPS")),
        NewsSource(name="GH base node releases",      url="https://github.com/base/node/releases.atom",              topics=("BASE", "L2", "DEVOPS")),
        NewsSource(name="GH uniswap-v3 releases",     url="https://github.com/Uniswap/v3-core/releases.atom",        topics=("UNI", "DEFI", "DEVOPS")),
        NewsSource(name="GH aave-v3 releases",        url="https://github.com/aave/aave-v3-core/releases.atom",      topics=("AAVE", "DEFI", "DEVOPS")),
        NewsSource(name="GH chainlink releases",      url="https://github.com/smartcontractkit/chainlink/releases.atom", topics=("LINK", "DEFI", "DEVOPS")),
        NewsSource(name="GH curve releases",          url="https://github.com/curvefi/curve-contract/releases.atom", topics=("CRV", "DEFI", "DEVOPS")),
        NewsSource(name="GH maker releases",          url="https://github.com/makerdao/dss/releases.atom",           topics=("MKR", "DAI", "DEFI", "DEVOPS")),
        NewsSource(name="GH lido releases",           url="https://github.com/lidofinance/lido-dao/releases.atom",   topics=("LDO", "DEFI", "DEVOPS")),
        NewsSource(name="GH cbETH releases",          url="https://github.com/coinbase/cbETH/releases.atom",         topics=("CBETH", "DEFI", "DEVOPS")),

        # === Tier 4 part B: arXiv — academic crypto/finance papers ===
        NewsSource(name="arXiv q-fin.TR",  url="https://export.arxiv.org/rss/q-fin.TR", topics=("RESEARCH", "MARKETS")),
        NewsSource(name="arXiv cs.CR",     url="https://export.arxiv.org/rss/cs.CR",    topics=("RESEARCH", "SECURITY")),
        NewsSource(name="arXiv q-fin.RM",  url="https://export.arxiv.org/rss/q-fin.RM", topics=("RESEARCH", "RISK")),
        NewsSource(name="arXiv q-fin.MF",  url="https://export.arxiv.org/rss/q-fin.MF", topics=("RESEARCH", "MATH")),
    ]


# === Tier 2: Reddit JSON adapter (no key, no auth) ===
# Reddit exposes every public subreddit as JSON via `/r/<name>/new.json`.
# Polite headers + 25-item fetch per sub keeps us well under rate limits.
_REDDIT_SUBS = [
    "CryptoCurrency", "Bitcoin", "ethereum", "solana", "BASE",
    "avax", "cardano", "Polkadot", "dogecoin", "Chainlink",
    "UniSwap", "Aave_Official", "MakerDAO", "ethfinance",
    "Optimism", "Arbitrum", "ethstaker", "ethdev", "defi",
    "CryptoMarkets", "altcoin", "CryptoCurrencyTrading",
    "CryptoCurrencies", "btc", "ETHFinance", "0xPolygon",
    "WBTC", "USDC", "LidoFinance", "RocketPool",
]

# === Tier 4 part C: HackerNews Algolia search ===
# Free, public, returns JSON with titles + URLs + scores.  We rotate
# the search term across the crypto vocabulary so every cycle picks
# up different threads.
_HN_TERMS = [
    "crypto", "bitcoin", "ethereum", "solana", "stablecoin",
    "defi", "l2 rollup", "smart contract", "blockchain",
    "uniswap", "chainlink", "base coinbase", "nft", "web3",
]

# === Tier 6 part A: Nitter — Twitter via RSS, no auth ===
# Nitter instances proxy Twitter and expose any user as RSS.  We
# rotate the instance because public Nitter mirrors go up and down.
_NITTER_INSTANCES = [
    "https://nitter.poast.org",
    "https://nitter.privacydev.net",
    "https://nitter.tiekoetter.com",
    "https://nitter.cz",
    "https://nitter.net",
]
_CRYPTO_TWITTER_USERS = [
    "VitalikButerin", "cz_binance", "brian_armstrong",
    "elonmusk", "saylor", "balajis", "TheBlock__",
    "DocumentingBTC", "Cointelegraph", "CoinDesk",
    "MessariCrypto", "DefiantNews", "BanklessHQ",
    "decryptmedia", "WatcherGuru", "thecryptolark",
    "0xDanielTehrani", "punk6529", "jessepollak",
    "rajgokal", "haydenzadams", "stanikulechov",
    "0xPolygon", "Optimism", "arbitrum", "Uniswap",
    "AaveAave", "chainlink", "lidofinance",
]

# === Tier 6 part B: RSSHub — Telegram channels + more ===
# RSSHub mirrors expose Telegram channels (and lots else) as RSS.
# Mirrors rotate frequently; we ship a small list and fail-soft.
_RSSHUB_INSTANCES = [
    "https://rsshub.app",
    "https://rss.shab.fun",
    "https://rsshub.zhouql.vip",
]
_TELEGRAM_CHANNELS = [
    "WhaleAlertCalls", "CoinDesk", "binance_announcements",
    "BybitOfficialAnnouncements", "DefiLlamaOfficial",
    "messaricrypto", "TheBlock__", "TheDefiant",
]


def _build_default_sources() -> Sequence[NewsSource]:
    sources = list(_crypto_rss_feeds())
    # Reddit
    for sub in _REDDIT_SUBS:
        sources.append(NewsSource(
            name=f"Reddit r/{sub}", url=sub, kind="reddit",
            topics=("SOCIAL", "REDDIT"),
        ))
    # HackerNews — one source per term so we get diverse threads
    for term in _HN_TERMS:
        sources.append(NewsSource(
            name=f"HN: {term}", url=term, kind="hn",
            topics=("DEV", "RESEARCH"),
        ))
    # Nitter (Twitter) — kind=nitter so the fetcher can rotate
    # through public mirrors when the primary is down.
    for user in _CRYPTO_TWITTER_USERS:
        sources.append(NewsSource(
            name=f"Nitter @{user}",
            url=user, kind="nitter",
            topics=("SOCIAL", "TWITTER"),
        ))
    # Telegram via RSSHub — kind=rsshub for the same reason
    for ch in _TELEGRAM_CHANNELS:
        sources.append(NewsSource(
            name=f"TG @{ch}",
            url=f"telegram/channel/{ch}", kind="rsshub",
            topics=("SOCIAL", "TELEGRAM"),
        ))
    return tuple(sources)


DEFAULT_SOURCES: Sequence[NewsSource] = _build_default_sources()

# Token synonym map: expand common ticker variants to canonical form for better matching.
# Key = variant (lowercase), value = canonical token(s) to add.
TOKEN_SYNONYMS: Dict[str, List[str]] = {
    "weth": ["eth", "ethereum"],
    "wbtc": ["btc", "bitcoin"],
    "eth": ["ethereum"],
    "btc": ["bitcoin"],
    "matic": ["polygon"],
    "pol": ["polygon"],
    "sol": ["solana"],
    "bnb": ["binance"],
    "avax": ["avalanche"],
    "link": ["chainlink"],
    "uni": ["uniswap"],
    "aave": ["aavefinance"],
    "arb": ["arbitrum"],
    "op": ["optimism"],
    "steth": ["eth", "lido", "staking"],
    "reth": ["eth", "rocketpool", "staking"],
    "cbeth": ["eth", "coinbase", "staking"],
    "usdc": ["stablecoin", "usd", "circle"],
    "usdt": ["stablecoin", "usd", "tether"],
    "dai": ["stablecoin", "usd", "makerdao"],
    "frax": ["stablecoin", "usd"],
    "defi": ["decentralizedfinance"],
    "l2": ["layer2", "rollup"],
    "nft": ["nonfungible", "collectible"],
    "cbdc": ["centralbank", "digitalcurrency"],
    "dex": ["decentralizedexchange", "swap"],
    "cex": ["centralizedexchange", "exchange"],
    "tvl": ["totalvaluelocked", "defi"],
    "yield": ["apr", "apy", "farming"],
}


def _expand_token_synonyms(tokens: Set[str]) -> Set[str]:
    """Return the input token set expanded with synonyms from TOKEN_SYNONYMS."""
    expanded = set(tokens)
    for tok in list(tokens):
        for synonym in TOKEN_SYNONYMS.get(tok.lower(), []):
            expanded.add(synonym.lower())
    return expanded


# ---------------------------------------------------------------------------
# Lightweight rule-based sentiment scorer (no external NLP dependency)
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = frozenset(
    "surge surging rally rallied breakout bullish buy buying adoption growth approve approved "
    "launch launched partner partnership upgrade upgraded integrat integrate invest investment "
    "profit profitable record high all-time gain gains win winner recovery recover milestone "
    "accelerat expand expansion institutional adoption strong outperform positive boost "
    "reward rewards staking yield airdrop grant approve legitimize mainstream".split()
)
_NEGATIVE_WORDS = frozenset(
    "crash crashing dump dumping hack hacked exploit exploited drain drained bear bearish "
    "sell selling ban banned restrict restricted fine fined penalt penalty lawsuit sue sued "
    "lose loss losing decline declining fall falling drop dropping plunge plunging breach "
    "fail failure bankrupt liquidat liquidation insolvenc insolvent fraud scam rug rugpull "
    "sanction sanctioned attack attacked vulnerability exposure delayed delay cancel cancelled "
    "suspend suspended withdraw withdrawal negative weak worse worst risk risky concern warning "
    "slump slumping collapse collapsing volatility doubt uncertain uncertainty fear".split()
)
_INTENSIFIERS = frozenset("major massive severe extreme unprecedented record significant sharply rapidly quickly".split())
_NEGATORS = frozenset("not no never neither nor without hardly barely scarcely".split())


def score_sentiment(text: str) -> Tuple[str, float]:
    """
    Return (label, score) where label ∈ {"positive","negative","neutral"} and
    score ∈ [-1, +1].  Uses a simple bag-of-words with negation handling.
    No external dependencies required.
    """
    words = re.findall(r"[a-z]+", text.lower())
    pos = 0.0
    neg = 0.0
    intensifier_active = False
    negation_active = False
    negation_window = 0

    for word in words:
        if word in _INTENSIFIERS:
            intensifier_active = True
            continue
        if word in _NEGATORS:
            negation_active = True
            negation_window = 4
            continue
        boost = 1.5 if intensifier_active else 1.0
        intensifier_active = False
        if negation_window > 0:
            negation_window -= 1
        else:
            negation_active = False

        is_pos = word in _POSITIVE_WORDS
        is_neg = word in _NEGATIVE_WORDS
        if is_pos:
            if negation_active:
                neg += boost
            else:
                pos += boost
        if is_neg:
            if negation_active:
                pos += boost * 0.5
            else:
                neg += boost

    total = pos + neg
    if total < 1.0:
        return "neutral", 0.0
    raw = (pos - neg) / total  # [-1, +1]
    if raw > 0.15:
        label = "positive"
    elif raw < -0.15:
        label = "negative"
    else:
        label = "neutral"
    return label, round(raw, 3)


class EthicalNewsIngestor:
    """Collects free/ethical crypto headlines and stores them locally for training."""

    def __init__(
        self,
        *,
        sources: Optional[Sequence[NewsSource]] = None,
        output_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        fetcher: Optional[Callable[[NewsSource], List[dict]]] = None,
        max_tokens: int = 12,
    ) -> None:
        base_sources = list(sources or DEFAULT_SOURCES)
        catalog_path = Path(os.getenv("ETHICAL_NEWS_SOURCES_PATH", "config/ethical_news_sources.json"))
        custom_sources = self._load_custom_sources(catalog_path)
        merged: Dict[str, NewsSource] = {source.name: source for source in base_sources}
        for source in custom_sources:
            merged[source.name] = source
        self.sources = tuple(merged.values())
        root_cache = cache_dir or Path(os.getenv("ETHICAL_NEWS_CACHE", "data/news/cache"))
        self.cache_dir = root_cache.expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = (output_path or Path(os.getenv("ETHICAL_NEWS_PATH", "data/news/ethical_news.parquet"))).expanduser()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.archive_root = Path(os.getenv("ETHICAL_NEWS_ARCHIVE_DIR", "data/news/free_news")).expanduser()
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self._fetcher = fetcher or self._fetch_source
        self.max_tokens = max_tokens
        self._request_timeout = max(1.0, float(os.getenv("ETHICAL_NEWS_REQUEST_TIMEOUT_SEC", "5")))
        self._http_headers = {
            "User-Agent": os.getenv("ETHICAL_NEWS_USER_AGENT", "CoolCryptoUtilities/ethical-news-ingestor")
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def harvest(
        self,
        *,
        tokens: Iterable[str],
        start_ts: int,
        end_ts: int,
        deadline: Optional[float] = None,
    ) -> List[dict]:
        keyword_set = {token.lower() for token in tokens if token}
        # Expand with synonyms so "WETH" searches also match "ETH" articles
        keyword_set_expanded = _expand_token_synonyms(keyword_set)
        rows: List[dict] = []
        for source in self.sources:
            if deadline and (deadline - time.time()) <= 0:
                break
            remaining = None
            if deadline:
                remaining = max(0.0, deadline - time.time())
                if remaining < 0.25:
                    break
            source_keywords = keyword_set_expanded | {topic.lower() for topic in getattr(source, "topics", []) if topic}
            timeout = None
            if remaining is not None:
                timeout = max(0.5, min(self._request_timeout, remaining))
            entries = self._pull_entries(source, timeout=timeout)
            for entry in entries:
                ts = self._entry_timestamp(entry)
                if ts is None or ts < start_ts or ts > end_ts:
                    continue
                title = self._clean_text(entry.get("title"))
                summary = self._clean_text(entry.get("summary") or entry.get("description") or "")
                body = f"{title}\n{summary}".strip()
                if not body:
                    continue
                article_tokens = self._extract_tokens(body)
                if source_keywords and not (article_tokens & source_keywords):
                    continue
                selected_tokens = self._select_tokens(article_tokens, source_keywords)
                if not selected_tokens:
                    continue
                sentiment_label, sentiment_score = score_sentiment(body)
                rows.append(
                    {
                        "timestamp": int(ts),
                        "headline": title[:256] or summary[:256],
                        "article": summary or title,
                        "sentiment": sentiment_label,
                        "sentiment_score": sentiment_score,
                        "tokens": sorted(selected_tokens),
                        "source": source.name,
                        "url": entry.get("link"),
                    }
                )
        if rows:
            self._write_parquet(rows)
            self._archive_rows(rows, start_ts, end_ts, keyword_set)
            return rows
        archived = self._load_archive_rows(start_ts, end_ts, keyword_set)
        if archived:
            self._write_parquet(archived)
        return archived

    def harvest_windows(
        self,
        *,
        tokens: Iterable[str],
        ranges: Sequence[Tuple[int, int]],
        deadline: Optional[float] = None,
    ) -> List[dict]:
        rows: List[dict] = []
        for start_ts, end_ts in ranges:
            start_int = int(min(start_ts, end_ts))
            end_int = int(max(start_ts, end_ts))
            rows.extend(self.harvest(tokens=tokens, start_ts=start_int, end_ts=end_int, deadline=deadline))
        return rows

    def harvest_window(
        self,
        *,
        tokens: Iterable[str],
        start: datetime,
        end: datetime,
        deadline: Optional[float] = None,
    ) -> List[dict]:
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        return self.harvest(tokens=tokens, start_ts=start_ts, end_ts=end_ts, deadline=deadline)

    def harvest_schedule_file(
        self,
        schedule_path: Path,
        *,
        fallback_tokens: Optional[Iterable[str]] = None,
    ) -> int:
        path = schedule_path.expanduser()
        if not path.exists():
            return 0
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        collected = 0
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            token_payload = entry.get("tokens") or fallback_tokens
            if not token_payload:
                continue
            start_ts = self._coerce_schedule_ts(entry.get("start_ts") or entry.get("start"))
            end_ts = self._coerce_schedule_ts(entry.get("end_ts") or entry.get("end"))
            if start_ts is None or end_ts is None:
                continue
            try:
                rows = self.harvest(tokens=token_payload, start_ts=start_ts, end_ts=end_ts)
                collected += len(rows)
            except Exception:
                continue
        return collected

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pull_entries(self, source: NewsSource, *, timeout: Optional[float] = None) -> List[dict]:
        try:
            if timeout is not None:
                try:
                    entries = self._fetcher(source, timeout=timeout)  # type: ignore[misc]
                except TypeError:
                    entries = self._fetcher(source)
            else:
                entries = self._fetcher(source)
            if entries:
                self._persist_cache(source, entries)
                return entries
        except Exception:
            entries = None
        cached = self._load_cache(source)
        return cached or []

    def _fetch_source(self, source: NewsSource, timeout: Optional[float] = None) -> List[dict]:
        # Dispatch on source kind so RSS, Reddit JSON, HN Algolia all
        # come back as the same dict shape downstream code expects.
        kind = getattr(source, "kind", "rss") or "rss"
        if kind == "reddit":
            return self._fetch_reddit(source, timeout=timeout)
        if kind == "hn":
            return self._fetch_hn(source, timeout=timeout)
        if kind == "nitter":
            return self._fetch_rotating_rss(source,
                instances=_NITTER_INSTANCES,
                path_template="{instance}/{path}/rss",
                timeout=timeout)
        if kind == "rsshub":
            return self._fetch_rotating_rss(source,
                instances=_RSSHUB_INSTANCES,
                path_template="{instance}/{path}",
                timeout=timeout)
        # Default: RSS / Atom via feedparser
        try:
            effective_timeout = self._request_timeout if timeout is None else max(0.5, min(timeout, self._request_timeout))
            resp = requests.get(source.url, timeout=effective_timeout, headers=self._http_headers)
            resp.raise_for_status()
            parsed = feedparser.parse(resp.content)
        except Exception:
            return []
        if getattr(parsed, "bozo", False):
            return []
        raw_entries = getattr(parsed, "entries", None) or getattr(parsed, "items", None) or []
        if not isinstance(raw_entries, list):
            return []
        return [self._normalize_entry(entry) for entry in raw_entries]

    def _fetch_reddit(self, source: NewsSource, timeout: Optional[float] = None) -> List[dict]:
        """Fetch a subreddit's NEW feed as RSS.

        Reddit deprecated unauthenticated JSON access (returns 403) but
        the RSS feed at /r/<sub>/new/.rss still works without OAuth.
        We pass it through feedparser so the dict shape matches the
        rest of the pipeline.
        """
        sub = (source.url or "").strip().strip("/")
        if not sub:
            return []
        url = f"https://www.reddit.com/r/{sub}/new/.rss"
        ua = self._http_headers.get("User-Agent") if isinstance(self._http_headers, dict) else None
        headers = {
            "User-Agent": ua or "w1z4rdv1510n-news/1.0 (+https://github.com/C4rr13rX/CoolCryptoUtilities)",
            "Accept": "application/atom+xml, application/rss+xml, application/xml;q=0.9, */*;q=0.8",
        }
        try:
            effective_timeout = self._request_timeout if timeout is None else max(0.5, min(timeout, self._request_timeout))
            resp = requests.get(url, timeout=effective_timeout, headers=headers)
            if resp.status_code != 200:
                return []
            parsed = feedparser.parse(resp.content)
        except Exception:
            return []
        if getattr(parsed, "bozo", False):
            return []
        raw = getattr(parsed, "entries", None) or []
        if not isinstance(raw, list):
            return []
        out: List[dict] = []
        for e in raw:
            d = self._normalize_entry(e)
            d["source_kind"] = "reddit"
            out.append(d)
        return out

    def _fetch_rotating_rss(
        self,
        source: NewsSource,
        *,
        instances: Sequence[str],
        path_template: str,
        timeout: Optional[float] = None,
    ) -> List[dict]:
        """Try each mirror in `instances` in turn until one returns
        usable RSS.  Used for Nitter (Twitter mirrors) and RSSHub
        (Telegram + many other feed sources) since public mirrors go
        up and down constantly."""
        path = (source.url or "").strip().strip("/")
        if not path:
            return []
        ua = self._http_headers.get("User-Agent") if isinstance(self._http_headers, dict) else None
        headers = {
            "User-Agent": ua or "w1z4rdv1510n-news/1.0",
            "Accept": "application/atom+xml, application/rss+xml, application/xml;q=0.9, */*;q=0.8",
        }
        effective_timeout = self._request_timeout if timeout is None else max(0.5, min(timeout, self._request_timeout))
        for inst in instances:
            try:
                url = path_template.format(instance=inst.rstrip("/"), path=path)
                resp = requests.get(url, timeout=effective_timeout, headers=headers)
                if resp.status_code != 200:
                    continue
                parsed = feedparser.parse(resp.content)
                if getattr(parsed, "bozo", False):
                    continue
                raw = getattr(parsed, "entries", None) or []
                if not isinstance(raw, list) or not raw:
                    continue
                kind = getattr(source, "kind", "rss")
                out: List[dict] = []
                for e in raw:
                    d = self._normalize_entry(e)
                    d["source_kind"] = kind
                    out.append(d)
                return out
            except Exception:
                continue
        return []

    def _fetch_hn(self, source: NewsSource, timeout: Optional[float] = None) -> List[dict]:
        """Search HackerNews via the public Algolia endpoint.

        source.url is the search term.  No auth, no rate-limit headers
        in practice (Algolia is generous for low-volume use).
        """
        term = (source.url or "").strip()
        if not term:
            return []
        api = ("https://hn.algolia.com/api/v1/search_by_date?"
               "tags=story&hitsPerPage=25&query=" + requests.utils.quote(term))
        headers = {"User-Agent": "w1z4rdv1510n-news/1.0"}
        try:
            effective_timeout = self._request_timeout if timeout is None else max(0.5, min(timeout, self._request_timeout))
            resp = requests.get(api, timeout=effective_timeout, headers=headers)
            if resp.status_code != 200:
                return []
            data = resp.json()
        except Exception:
            return []
        hits = data.get("hits") or []
        out: List[dict] = []
        for h in hits:
            try:
                title = (h.get("title") or h.get("story_title") or "").strip()
                if not title:
                    continue
                link = h.get("url") or h.get("story_url") or (
                    f"https://news.ycombinator.com/item?id={h.get('objectID')}" if h.get("objectID") else "")
                ts = h.get("created_at_i")
                entry = {
                    "title": title,
                    "summary": title,
                    "description": title,
                    "link": link,
                    "published": h.get("created_at") or "",
                    "source_kind": "hn",
                }
                if isinstance(ts, (int, float)):
                    entry["published_parsed"] = float(ts)
                out.append(entry)
            except Exception:
                continue
        return out

    def _normalize_entry(self, entry: Any) -> dict:
        summary = entry.get("summary") if isinstance(entry, dict) else getattr(entry, "summary", "")
        data = {
            "title": entry.get("title") if isinstance(entry, dict) else getattr(entry, "title", ""),
            "summary": summary,
            "description": entry.get("description") if isinstance(entry, dict) else getattr(entry, "description", ""),
            "link": entry.get("link") if isinstance(entry, dict) else getattr(entry, "link", ""),
            "published": entry.get("published") if isinstance(entry, dict) else getattr(entry, "published", ""),
        }
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            data["published_parsed"] = time.mktime(entry.published_parsed)
        elif isinstance(entry, dict) and isinstance(entry.get("published_parsed"), (tuple, list)):
            try:
                data["published_parsed"] = time.mktime(entry["published_parsed"])  # type: ignore[index]
            except Exception:
                pass
        return data

    def _entry_timestamp(self, entry: dict) -> Optional[int]:
        ts = entry.get("published_parsed")
        if isinstance(ts, (int, float)):
            return int(ts)
        published = entry.get("published")
        if isinstance(published, (int, float)):
            return int(published)
        if published:
            try:
                if _fp_parse_date:
                    parsed = _fp_parse_date(published)
                else:
                    parsed = feedparser._parse_date(published)  # type: ignore[attr-defined]
                if parsed:
                    return int(time.mktime(parsed))
            except Exception:
                return None
        return None

    def _coerce_schedule_ts(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except Exception:
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    return int(parsed.timestamp())
                except Exception:
                    return None
        return None

    def _clean_text(self, text: Any) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        soup = BeautifulSoup(raw, "html.parser")
        clean = soup.get_text(" ", strip=True)
        return re.sub(r"\s+", " ", clean).strip()

    def _extract_tokens(self, text: str) -> Set[str]:
        tokens = set(re.findall(r"[A-Za-z0-9]{3,}", text.lower()))
        return _expand_token_synonyms(tokens)

    def _select_tokens(self, article_tokens: Set[str], keyword_set: Set[str]) -> Set[str]:
        if keyword_set:
            matched = article_tokens & keyword_set
        else:
            matched = article_tokens
        trimmed = sorted(matched)[: self.max_tokens]
        return {token.upper() for token in trimmed}

    def _write_parquet(self, rows: List[dict]) -> None:
        df_new = pd.DataFrame(rows)
        if self.output_path.exists():
            try:
                df_old = pd.read_parquet(self.output_path)
            except Exception:
                df_old = pd.DataFrame()
            df = pd.concat([df_old, df_new], ignore_index=True)
            df = df.drop_duplicates(subset=["timestamp", "headline", "source"])
        else:
            df = df_new
        try:
            df.to_parquet(self.output_path, index=False)
        except Exception:
            pass

    def _archive_rows(self, rows: List[dict], start_ts: int, end_ts: int, tokens: Iterable[str]) -> None:
        if not rows:
            return
        payload = {
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "tokens": sorted({str(token).upper() for token in tokens if token}),
            "count": len(rows),
            "articles": rows,
        }
        filename = f"{int(start_ts)}_{int(end_ts)}.json"
        path = self.archive_root / filename
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_archive_rows(self, start_ts: int, end_ts: int, tokens: Set[str]) -> List[dict]:
        if not self.archive_root.exists():
            return []
        wanted = {token.upper() for token in tokens if token}
        rows: List[dict] = []
        seen: Set[Tuple[int, str, str]] = set()
        for path in self.archive_root.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            try:
                payload_start = int(payload.get("start_ts", 0))
                payload_end = int(payload.get("end_ts", 0))
            except Exception:
                continue
            if payload_end < start_ts or payload_start > end_ts:
                continue
            payload_tokens = {str(token).upper() for token in payload.get("tokens", []) if token}
            if wanted and payload_tokens and not (wanted & payload_tokens):
                continue
            for article in payload.get("articles", []) or []:
                if not isinstance(article, dict):
                    continue
                try:
                    ts = int(article.get("timestamp", 0))
                except Exception:
                    continue
                if ts < start_ts or ts > end_ts:
                    continue
                article_tokens = {str(token).upper() for token in article.get("tokens", []) if token}
                if wanted and article_tokens and not (wanted & article_tokens):
                    continue
                key = (ts, str(article.get("headline", "")), str(article.get("source", "")))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(article)
        return rows

    def _load_custom_sources(self, path: Path) -> List[NewsSource]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        sources: List[NewsSource] = []
        if isinstance(payload, list):
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "").strip()
                url = str(entry.get("url") or "").strip()
                topics = entry.get("topics") or ()
                if not name or not url:
                    continue
                sources.append(NewsSource(name=name, url=url, topics=tuple(topics)))
        return sources

    def _cache_file(self, source: NewsSource) -> Path:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", source.name.lower()).strip("-") or "source"
        return self.cache_dir / f"{slug}.json"

    def _persist_cache(self, source: NewsSource, entries: Iterable[dict]) -> None:
        path = self._cache_file(source)
        try:
            path.write_text(json.dumps(list(entries)), encoding="utf-8")
        except Exception:
            path.unlink(missing_ok=True)

    def _load_cache(self, source: NewsSource) -> Optional[List[dict]]:
        path = self._cache_file(source)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            path.unlink(missing_ok=True)
            return None
        if not isinstance(payload, list):
            return None
        return [entry for entry in payload if isinstance(entry, dict)]
