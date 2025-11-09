from __future__ import annotations

import os
from typing import Dict, Iterable, List, Mapping, Optional

from services.token_catalog import get_core_token_map

_DISABLE_SET = {"0", "false", "no", "off"}


def core_watch_limit(default: int = 6) -> Optional[int]:
    """
    Return the configured cap for per-chain core-token watchers.
    - WALLET_CORE_WATCH_LIMIT=<int> sets an explicit limit.
    - 0/false disables the limit (track all core tokens for the chain).
    """
    raw = os.getenv("WALLET_CORE_WATCH_LIMIT")
    if raw is None or raw.strip() == "":
        limit = default
    else:
        try:
            limit = int(raw)
        except ValueError:
            limit = default
    if str(raw or "").strip().lower() in _DISABLE_SET:
        return None
    return limit if limit and limit > 0 else None


def build_core_watch_tokens(
    chains: Iterable[str],
    *,
    limit: Optional[int] = None,
    token_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, List[str]]:
    """
    Build a mapping of chain -> [core token addresses] capped by `limit`.
    Exposed so wallet backends can keep stable watchlists in multiple contexts.
    """
    mapping = token_map or get_core_token_map()
    result: Dict[str, List[str]] = {}
    for chain in chains:
        chain_l = str(chain or "").strip().lower()
        if not chain_l:
            continue
        entries = mapping.get(chain_l, {})
        if not entries:
            continue
        addrs = [addr for addr in entries.values() if addr]
        if not addrs:
            continue
        if limit is not None:
            addrs = addrs[:limit]
        result[chain_l] = addrs
    return result


def _as_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _quantity_score(meta: Mapping[str, object]) -> float:
    qty = _as_float(meta.get("quantity"))
    if qty > 0:
        return qty
    raw = meta.get("balance_hex") or meta.get("raw")
    if isinstance(raw, str):
        raw_s = raw.strip()
        try:
            if raw_s.startswith("0x"):
                return float(int(raw_s, 16))
            return float(raw_s)
        except Exception:
            return 0.0
    try:
        return float(raw or 0)
    except Exception:
        return 0.0


def select_cached_watch_tokens(
    tokens: Mapping[str, Mapping[str, object]],
    *,
    limit: Optional[int] = None,
    min_usd: float = 0.0,
    include_illiquid: bool = True,
) -> List[str]:
    """
    Pick candidate token addresses from cached wallet state ordered by USD value,
    falling back to quantity/amount when USD quotes are missing.
    """
    rows: List[tuple[float, float, str]] = []
    for addr, meta in (tokens or {}).items():
        if not addr:
            continue
        usd = _as_float(meta.get("usd_amount") or meta.get("usd"))
        if usd <= 0 and not include_illiquid:
            continue
        score = usd if usd > 0 else _quantity_score(meta)
        rows.append((usd, score, addr))
    rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    picks: List[str] = []
    for usd, _, addr in rows:
        if usd < min_usd:
            continue
        picks.append(addr)
        if limit is not None and len(picks) >= limit:
            break
    if limit is not None:
        return picks[:limit]
    return picks


def merge_watch_maps(*maps: Optional[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}
    for mapping in maps:
        if not mapping:
            continue
        for chain, tokens in mapping.items():
            bucket = merged.setdefault(chain, [])
            for token in tokens:
                if token not in bucket:
                    bucket.append(token)
    return merged


__all__ = [
    "core_watch_limit",
    "build_core_watch_tokens",
    "select_cached_watch_tokens",
    "merge_watch_maps",
]
