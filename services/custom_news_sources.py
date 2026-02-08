from __future__ import annotations

import time
from typing import Any, Dict, List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Gecko/20100101 Firefox/125.0",
    "Accept-Language": "en-US,en;q=0.9",
}


def _extract_text(node) -> str:
    if not node:
        return ""
    return " ".join(node.get_text(" ").split())


def parse_articles(html: str, *, base_url: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    item_selector = config.get("item_selector") or "article"
    title_selector = config.get("title_selector") or "a"
    link_selector = config.get("link_selector") or "a"
    summary_selector = config.get("summary_selector") or ""

    items = soup.select(item_selector) if item_selector else []
    if not items:
        items = soup.find_all("article")
    results: List[Dict[str, Any]] = []

    for item in items:
        title_node = item.select_one(title_selector) if title_selector else item
        link_node = item.select_one(link_selector) if link_selector else item
        summary_node = item.select_one(summary_selector) if summary_selector else None

        title = _extract_text(title_node)
        href = link_node.get("href") if link_node else ""
        url = urljoin(base_url, href) if href else ""
        summary = _extract_text(summary_node) if summary_node else ""
        if not title and not url:
            continue
        results.append({"title": title, "url": url, "summary": summary})
    return results


def fetch_source_articles(base_url: str, *, config: Dict[str, Any], max_items: int = 12, timeout: int = 12) -> List[Dict[str, Any]]:
    start = time.time()
    resp = requests.get(base_url, headers=DEFAULT_HEADERS, timeout=timeout)
    resp.raise_for_status()
    html = resp.text
    parsed = parse_articles(html, base_url=base_url, config=config)
    trimmed = parsed[:max_items]
    for entry in trimmed:
        entry["fetched_at"] = time.time()
        entry["elapsed_s"] = round(time.time() - start, 2)
    return trimmed
