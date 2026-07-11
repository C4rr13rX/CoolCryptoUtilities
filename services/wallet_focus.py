"""Single source of truth for "what the user actually holds and cares about."

The user's expectation is simple: *I have crypto in my wallet — trade with
what makes sense there.* This module answers "which of my holdings are worth
streaming, indexing, and predicting on" — above a small USD floor (dust is
explicitly not a priority), excluding pure stablecoins (nothing to predict
about a $1 peg), and mapping each to the trading pairs that represent it.

Used by:
  - make2000index.py  — seed the DexScreener pair index with held tokens
  - download2000.py    — ensure held-token pairs get OHLCV history
  - trading/selector.py — prioritise held tokens for live streaming

Historically make2000index tried to read the wallet via `from db import DB`
(a class that doesn't exist); the ImportError was swallowed, so wallet
seeding silently did nothing for the whole project's life. This module uses
the correct `get_db()` path and is import-safe.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

# Stable pegs — held as savings, not prediction targets. Native/wrapped
# gas tokens ARE prediction targets (ETH, MATIC, BNB move).
STABLE_SYMBOLS = {"USDC", "USDC.E", "USDBC", "USDT", "DAI", "BUSD", "TUSD",
                  "USDP", "FRAX", "LUSD", "GUSD", "PYUSD"}

# A held token maps to these quote legs, in preference order, to form pairs.
_QUOTE_LEGS = ("USDC", "USDT", "WETH")

# Native coins normalize to their wrapped ERC-20 for pair formation.
_WRAPPED = {"ETH": "WETH", "MATIC": "WMATIC", "POL": "WMATIC",
            "BNB": "WBNB", "AVAX": "WAVAX"}


def _min_usd() -> float:
    try:
        return float(os.getenv("WALLET_FOCUS_MIN_USD", "1.0"))
    except (TypeError, ValueError):
        return 1.0


def held_tokens(
    *,
    min_usd: Optional[float] = None,
    chains: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """Return meaningful (non-dust, non-stable) holdings, richest first.

    Each entry: {symbol, chain, usd, quantity}. Reads the balances table
    written by the wallet scan / portfolio refresh. Never raises.
    """
    floor = _min_usd() if min_usd is None else float(min_usd)
    chain_filter = {c.lower() for c in chains} if chains else None
    out: List[Dict[str, object]] = []
    seen: set[Tuple[str, str]] = set()
    try:
        from db import get_db
        db = get_db()
        rows = db.fetch_balances_flat(include_zero=False)
    except Exception:
        return out
    for row in rows:
        try:
            symbol = str(row["symbol"] or "").upper().strip()
            chain = str(row["chain"] or "").lower().strip()
            usd = float(row["usd_amount"] or 0.0)
        except Exception:
            continue
        if not symbol or symbol in STABLE_SYMBOLS:
            continue
        if usd < floor:
            continue
        if chain_filter and chain not in chain_filter:
            continue
        key = (chain, symbol)
        if key in seen:
            continue
        seen.add(key)
        out.append({"symbol": symbol, "chain": chain, "usd": usd,
                    "quantity": row["quantity"]})
    out.sort(key=lambda e: float(e["usd"]), reverse=True)
    return out


def held_symbols(*, min_usd: Optional[float] = None,
                 chain: Optional[str] = None) -> List[str]:
    """Just the distinct symbols (optionally for one chain), richest first.

    Includes the wrapped form of native coins so pair lookups resolve
    (ETH → also WETH), since DEX pairs trade the wrapped token.
    """
    chains = [chain] if chain else None
    syms: List[str] = []
    for entry in held_tokens(min_usd=min_usd, chains=chains):
        sym = str(entry["symbol"])
        if sym not in syms:
            syms.append(sym)
        wrapped = _WRAPPED.get(sym)
        if wrapped and wrapped not in syms:
            syms.append(wrapped)
    return syms


def ensure_wallet_streams(*, min_usd: Optional[float] = None) -> Dict[str, object]:
    """Guarantee every meaningful holding's pairs are in the stream watchlist.

    Called each production cycle so the moment the user funds a new token,
    it starts being monitored/predicted — without a full re-bootstrap. Dust
    stays out (min_usd floor); stablecoins stay out (nothing to predict).
    Returns a summary for logging. Never raises.
    """
    pairs = held_pairs(min_usd=min_usd)
    if not pairs:
        return {"status": "no_holdings", "added": []}
    try:
        from services.watchlists import load_watchlists, mutate_watchlist
        current = set((load_watchlists() or {}).get("stream") or [])
        missing = [p for p in pairs if p not in current]
        if missing:
            result = mutate_watchlist("stream", add=missing)
            return {"status": "added", "added": missing,
                    "stream_size": len((result or {}).get("stream") or [])}
        return {"status": "current", "added": [], "held_pairs": pairs}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "added": []}


def held_pairs(*, min_usd: Optional[float] = None,
               chain: Optional[str] = None) -> List[str]:
    """Trading-pair symbols representing the holdings (e.g. 'WETH-USDC').

    A held stable would be nothing to trade; a held non-stable pairs with
    each quote leg it isn't already. Ordered richest-holding first so
    downstream limit-capped consumers keep the most valuable positions.
    """
    pairs: List[str] = []
    seen: set = set()
    chains = [chain] if chain else None
    for entry in held_tokens(min_usd=min_usd, chains=chains):
        sym = str(entry["symbol"])
        base = _WRAPPED.get(sym, sym)
        if base in STABLE_SYMBOLS:
            continue
        for quote in _QUOTE_LEGS:
            if base == quote:
                continue
            pair = f"{base}-{quote}"
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs
