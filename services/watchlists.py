from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from db import TradingDatabase, get_db


DEFAULT_WATCHLISTS = {
    "stream": [],
    "ghost": [],
    "live": [],
}


def _normalize_symbol(symbol: str) -> str:
    cleaned = str(symbol or "").strip().upper().replace("/", "-")
    return cleaned


def _dedupe(symbols: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for raw in symbols:
        norm = _normalize_symbol(raw)
        if not norm or norm in seen:
            continue
        ordered.append(norm)
        seen.add(norm)
    return ordered


def load_watchlists(db: Optional[TradingDatabase] = None) -> Dict[str, List[str]]:
    database = db or get_db()
    payload = database.get_json("watchlists") or {}
    data: Dict[str, List[str]] = {}
    for key in DEFAULT_WATCHLISTS:
        entries = payload.get(key) or []
        if not isinstance(entries, list):
            entries = []
        data[key] = _dedupe(entries)
    return data


def save_watchlists(watchlists: Dict[str, Sequence[str]], db: Optional[TradingDatabase] = None) -> Dict[str, List[str]]:
    database = db or get_db()
    sanitized = {
        key: _dedupe(watchlists.get(key, []))
        for key in DEFAULT_WATCHLISTS
    }
    database.set_json("watchlists", sanitized)
    return load_watchlists(database)


def mutate_watchlist(
    target: str,
    *,
    add: Optional[Sequence[str]] = None,
    remove: Optional[Sequence[str]] = None,
    replace: Optional[Sequence[str]] = None,
    db: Optional[TradingDatabase] = None,
) -> Dict[str, List[str]]:
    target_key = str(target or "").lower()
    if target_key not in DEFAULT_WATCHLISTS:
        raise ValueError(f"Unknown watchlist '{target}'.")
    database = db or get_db()
    current = load_watchlists(database)
    entries = current.get(target_key, [])
    if replace is not None:
        entries = _dedupe(replace)
    else:
        if add:
            additions = _dedupe(add)
            for symbol in additions:
                if symbol not in entries:
                    entries.append(symbol)
        if remove:
            removals = {_normalize_symbol(sym) for sym in remove}
            entries = [sym for sym in entries if sym not in removals]
    current[target_key] = entries
    return save_watchlists(current, database)
