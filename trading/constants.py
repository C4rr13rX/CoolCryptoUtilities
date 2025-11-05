from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List


PRIMARY_CHAIN = os.getenv("PRIMARY_CHAIN", "base").strip().lower() or "base"


def _pair_index_path(chain: str) -> Path:
    override = os.getenv("PAIR_INDEX_PATH")
    if override:
        return Path(override)
    return Path("data") / f"pair_index_{chain}.json"


@lru_cache(None)
def _load_pair_index(chain: str) -> Dict[str, Dict[str, object]]:
    path = _pair_index_path(chain)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def top_pairs(limit: int = 10, *, chain: str | None = None) -> List[str]:
    chain = (chain or PRIMARY_CHAIN).lower()
    index = _load_pair_index(chain)
    if not index:
        return []
    items = sorted(
        index.values(),
        key=lambda entry: int(entry.get("index", 0)) if isinstance(entry, dict) else 0,
    )
    symbols: List[str] = []
    for entry in items:
        symbol = str(entry.get("symbol", "")).upper()
        if symbol:
            symbols.append(symbol)
        if len(symbols) >= limit:
            break
    return symbols


def pair_index_entries(chain: str | None = None) -> Dict[str, Dict[str, object]]:
    return dict(_load_pair_index((chain or PRIMARY_CHAIN).lower()))


_env_primary_symbol = os.getenv("PRIMARY_SYMBOL", "").strip().upper()
if _env_primary_symbol:
    PRIMARY_SYMBOL = _env_primary_symbol
else:
    candidates = top_pairs(limit=1)
    PRIMARY_SYMBOL = candidates[0] if candidates else "ETH-USDC"

PRIMARY_BASE, PRIMARY_QUOTE = (
    PRIMARY_SYMBOL.split("-", 1) if "-" in PRIMARY_SYMBOL else (PRIMARY_SYMBOL, "USDC")
)

MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE_REQUIRED", "0.9"))
SMALL_PROFIT_FLOOR = float(os.getenv("SMALL_PROFIT_FLOOR", "0.25"))
MAX_QUOTE_SHARE = float(os.getenv("MAX_QUOTE_SHARE", "0.25"))

GAS_PROFIT_BUFFER = float(os.getenv("GAS_PROFIT_BUFFER", "1.25"))
FALLBACK_NATIVE_PRICE = float(os.getenv("FALLBACK_NATIVE_PRICE", "1800.0"))
