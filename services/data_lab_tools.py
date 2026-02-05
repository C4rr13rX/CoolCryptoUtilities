from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from db import TradingDatabase, get_db


_STOPWORDS = {
    "the",
    "and",
    "for",
    "are",
    "but",
    "not",
    "you",
    "your",
    "with",
    "that",
    "this",
    "from",
    "have",
    "has",
    "had",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "how",
    "about",
    "into",
    "over",
    "under",
    "then",
    "than",
    "them",
    "they",
    "their",
    "ours",
    "can",
    "could",
    "should",
    "would",
    "will",
    "just",
    "like",
    "been",
    "did",
    "does",
    "doing",
    "its",
    "it's",
    "im",
    "i'm",
    "we",
    "our",
    "us",
    "me",
    "my",
    "mine",
    "yours",
}


@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: Sequence[str]
    description: str
    default_order: Optional[str] = None


TABLE_CATALOG: Dict[str, TableSpec] = {
    "balances": TableSpec(
        name="balances",
        columns=(
            "wallet",
            "chain",
            "token",
            "quantity",
            "usd_amount",
            "symbol",
            "name",
            "updated_at",
            "ts",
            "stale",
        ),
        description="Latest wallet/token balances (optionally stale).",
        default_order="updated_at DESC",
    ),
    "transfers": TableSpec(
        name="transfers",
        columns=(
            "wallet",
            "chain",
            "id",
            "hash",
            "block",
            "log_index",
            "ts",
            "from_addr",
            "to_addr",
            "token",
            "value",
            "inserted_at",
        ),
        description="Token transfer events for tracked wallets.",
        default_order="block DESC",
    ),
    "prices": TableSpec(
        name="prices",
        columns=("chain", "token", "usd", "source", "ts"),
        description="Latest price snapshots per token/chain.",
        default_order="ts DESC",
    ),
    "market_stream": TableSpec(
        name="market_stream",
        columns=("ts", "chain", "symbol", "price", "volume"),
        description="Streaming price ticks.",
        default_order="ts DESC",
    ),
    "trade_fills": TableSpec(
        name="trade_fills",
        columns=(
            "ts",
            "chain",
            "symbol",
            "expected_amount",
            "executed_amount",
            "expected_price",
            "executed_price",
        ),
        description="Execution fills recorded by trading ops.",
        default_order="ts DESC",
    ),
    "trading_ops": TableSpec(
        name="trading_ops",
        columns=("id", "ts", "wallet", "chain", "symbol", "action", "status"),
        description="Trading operations audit log.",
        default_order="id DESC",
    ),
    "experiments": TableSpec(
        name="experiments",
        columns=("id", "name", "status", "created", "updated"),
        description="Model/strategy experiments metadata.",
        default_order="updated DESC",
    ),
    "model_versions": TableSpec(
        name="model_versions",
        columns=("id", "version", "created", "path", "is_active"),
        description="Model version registry.",
        default_order="created DESC",
    ),
    "metrics": TableSpec(
        name="metrics",
        columns=("id", "ts", "stage", "category", "name", "value"),
        description="Operational metrics emitted by services.",
        default_order="ts DESC",
    ),
    "feedback_events": TableSpec(
        name="feedback_events",
        columns=("id", "ts", "source", "severity", "label"),
        description="Feedback events and guardrail alerts.",
        default_order="ts DESC",
    ),
    "advisories": TableSpec(
        name="advisories",
        columns=("id", "ts", "scope", "topic", "severity", "resolved", "resolved_ts"),
        description="Advisories and risk notices.",
        default_order="ts DESC",
    ),
    "pair_suppression": TableSpec(
        name="pair_suppression",
        columns=("symbol", "reason", "strikes", "last_failure", "release_ts"),
        description="Suppression list for problematic pairs.",
        default_order="last_failure DESC",
    ),
    "pair_adjustments": TableSpec(
        name="pair_adjustments",
        columns=(
            "symbol",
            "priority",
            "enter_offset",
            "exit_offset",
            "size_multiplier",
            "margin_offset",
            "allocation_multiplier",
            "label_scale",
            "updated",
        ),
        description="Per-pair tuning adjustments.",
        default_order="updated DESC",
    ),
}


def list_tables() -> Dict[str, Any]:
    tables = []
    for spec in TABLE_CATALOG.values():
        tables.append(
            {
                "table": spec.name,
                "columns": list(spec.columns),
                "description": spec.description,
                "default_order": spec.default_order,
            }
        )
    return {"count": len(tables), "tables": tables}


def _coerce_limit(value: Any, default: int = 50, maximum: int = 200) -> int:
    try:
        limit = int(value)
    except Exception:
        limit = default
    return max(1, min(limit, maximum))


def _sanitize_order(order: Any) -> str:
    raw = str(order or "desc").strip().lower()
    return "asc" if raw == "asc" else "desc"


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _coerce_row_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return value.hex()
    return value


def _normalize_filters(filters: Optional[Dict[str, Any]], allowed_cols: Sequence[str]) -> Tuple[List[str], List[Any]]:
    clauses: List[str] = []
    params: List[Any] = []
    if not filters:
        return clauses, params
    allowed = set(allowed_cols)
    for key, raw_val in filters.items():
        if key not in allowed:
            continue
        if isinstance(raw_val, dict):
            op = str(raw_val.get("op") or "=").strip().lower()
            value = raw_val.get("value")
            if op in {"contains", "like"}:
                clauses.append(f"LOWER({key}) LIKE ?")
                params.append(f"%{str(value or '').lower()}%")
                continue
            if op in {">", "<", ">=", "<=", "=", "!="}:
                clauses.append(f"{key} {op} ?")
                params.append(value)
                continue
            continue
        if isinstance(raw_val, (list, tuple, set)):
            values = [v for v in raw_val if v is not None]
            if not values:
                continue
            placeholders = ",".join("?" for _ in values)
            clauses.append(f"{key} IN ({placeholders})")
            params.extend(values)
            continue
        clauses.append(f"{key} = ?")
        params.append(raw_val)
    return clauses, params


def query_table(
    table: str,
    *,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 50,
    order_by: Optional[str] = None,
    order: str = "desc",
    columns: Optional[Sequence[str]] = None,
    db: Optional[TradingDatabase] = None,
) -> Dict[str, Any]:
    if table not in TABLE_CATALOG:
        raise ValueError(f"Unknown table: {table}")
    spec = TABLE_CATALOG[table]
    allowed_cols = list(spec.columns)
    selected_cols = [col for col in (columns or []) if col in allowed_cols]
    if not selected_cols:
        selected_cols = allowed_cols
    order = _sanitize_order(order)
    if order_by and order_by in allowed_cols:
        order_clause = f"{order_by} {order.upper()}"
    else:
        order_clause = spec.default_order or ""
    limit_val = _coerce_limit(limit)
    where_clauses, params = _normalize_filters(filters, allowed_cols)

    query = f"SELECT {', '.join(selected_cols)} FROM {table}"
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    if order_clause:
        query += f" ORDER BY {order_clause}"
    query += " LIMIT ?"
    params.append(limit_val)

    database = db or get_db()
    with database._cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    normalized = []
    for row in rows:
        data = dict(row)
        normalized.append({k: _coerce_row_value(v) for k, v in data.items()})
    return {
        "table": table,
        "count": len(normalized),
        "limit": limit_val,
        "order_by": order_clause or None,
        "filters": filters or {},
        "rows": normalized,
    }


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']{2,}", text.lower()) if t not in _STOPWORDS]


def _split_sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


def summarize_text(
    text: str,
    *,
    query_terms: Optional[Sequence[str]] = None,
    max_sentences: int = 3,
) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    tokens = _tokenize(text)
    freqs: Dict[str, int] = {}
    for token in tokens:
        freqs[token] = freqs.get(token, 0) + 1
    query_tokens = [t.lower() for t in (query_terms or []) if t and t.lower() not in _STOPWORDS]
    scored: List[Tuple[float, str]] = []
    for sent in sentences:
        sent_tokens = _tokenize(sent)
        if not sent_tokens:
            continue
        score = sum(freqs.get(tok, 0) for tok in sent_tokens)
        if query_tokens:
            score += sum(3 for tok in query_tokens if tok in sent.lower())
        scored.append((float(score), sent))
    if not scored:
        return " ".join(sentences[:max_sentences])
    scored.sort(key=lambda row: row[0], reverse=True)
    top = [sent for _, sent in scored[: max_sentences or 3]]
    return " ".join(top)


def fetch_news_with_summary(
    *,
    tokens: Sequence[str],
    start: Any,
    end: Any,
    query: Optional[str] = None,
    max_pages: Optional[int] = None,
    max_items: int = 40,
) -> Dict[str, Any]:
    start_dt = _coerce_datetime(start)
    end_dt = _coerce_datetime(end)
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt
    max_items = _coerce_limit(max_items, default=40, maximum=120)
    from services.news_lab import collect_news_for_terms

    result = collect_news_for_terms(tokens=list(tokens), start=start_dt, end=end_dt, query=query, max_pages=max_pages)
    items = list(result.get("items", []))
    if max_items:
        items = items[: max(1, int(max_items))]
    summary = []
    for item in items[:8]:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        source = item.get("source") or item.get("origin") or "unknown"
        date = (item.get("datetime") or "")[:10]
        summary.append(f"{date} - {title} ({source})".strip())
    payload = dict(result)
    payload["items"] = items
    payload["summary"] = summary
    return payload


def search_web(
    *,
    query: str,
    max_results: int = 5,
    max_bytes: int = 200_000,
    max_chars: int = 12_000,
    summary_sentences: int = 3,
    domains: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        return {"query": "", "results": [], "summary": [], "meta": {"errors": 0, "fetched": 0}}
    max_results = _coerce_limit(max_results, default=5, maximum=10)
    from services.web_search import WebSearch
    from services.web_research import WebResearcher

    search = WebSearch()
    researcher = WebResearcher(search=search)
    domain_list: List[str] = []
    if domains:
        if isinstance(domains, str):
            domain_list = [domains]
        else:
            domain_list = [str(d) for d in domains if d]
    results = search.search_domains(q, domain_list, total_limit=max_results)
    query_terms = _tokenize(q)
    compiled: List[Dict[str, Any]] = []
    errors = 0
    for result in results:
        text = ""
        try:
            text = researcher.fetch_text(result.url, max_bytes=max_bytes)
        except Exception:
            errors += 1
            compiled.append(
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "summary": "",
                    "error": "fetch_failed",
                }
            )
            continue
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        summary = summarize_text(text, query_terms=query_terms, max_sentences=summary_sentences)
        compiled.append(
            {
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "summary": summary,
                "excerpt": text[:400],
            }
        )

    overall = []
    for entry in compiled:
        if entry.get("summary"):
            overall.append(entry["summary"])
        if len(overall) >= 5:
            break
    return {
        "query": q,
        "results": compiled,
        "summary": overall,
        "meta": {"errors": errors, "fetched": len(compiled)},
    }


def summarize_payload(payload: Dict[str, Any], max_chars: int = 12_000) -> str:
    try:
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    except Exception:
        text = str(payload)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]..."
    return text
