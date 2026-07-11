"""Bridge market discovery (bullish/bearish movers) into live-stream selection.

Discovery already scores trending tokens (bull = 1h change, bear = 24h change)
but only fed the dashboard. This bridge pulls the current movers, filters them
down to reliable/safe candidates, and adds the survivors to the ``stream``
watchlist — which ``trading.selector.select_pairs`` honors first — so the
system branches out beyond wallet holdings while staying scam-filtered.

Django-free: reads DexScreener trending directly (free API) and applies the
heuristic + GoPlus (when keyed) + token-safety screens locally, so it runs
inside the production subprocess without the web stack.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from services.logging_utils import log_message


_LAST_RUN_TS = 0.0

_STABLE_SYMBOLS = {"USDC", "USDBC", "USDT", "DAI", "BUSD", "TUSD", "USDP", "FRAX", "LUSD"}

# DexScreener chain ids → internal chain names
_CHAIN_ALIASES = {
    "ethereum": "ethereum",
    "base": "base",
    "arbitrum": "arbitrum",
    "optimism": "optimism",
    "polygon": "polygon",
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def refresh_stream_watchlist_from_discovery(
    *,
    chains: Optional[List[str]] = None,
    max_added: Optional[int] = None,
    min_interval_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Fetch movers → safety-filter → merge into the stream watchlist.

    Returns a summary dict; safe to call every supervisor cycle (internally
    rate-limited by DISCOVERY_STREAM_INTERVAL_SEC, default 30 min).
    """
    global _LAST_RUN_TS
    if os.getenv("DISCOVERY_STREAM_BRIDGE_ENABLED", "1").lower() not in {"1", "true", "yes", "on"}:
        return {"status": "disabled"}
    interval = min_interval_sec if min_interval_sec is not None else _env_float(
        "DISCOVERY_STREAM_INTERVAL_SEC", 1800.0
    )
    now = time.time()
    if now - _LAST_RUN_TS < max(60.0, interval):
        return {"status": "throttled", "next_in_sec": max(60.0, interval) - (now - _LAST_RUN_TS)}
    _LAST_RUN_TS = now

    from services.discovery.trending_fetcher import fetch_trending_tokens
    from services.discovery.security_checks import heuristic_screen

    if chains is None:
        env_chains = (os.getenv("FOCUS_CHAINS") or "").strip()
        chains = [c.strip().lower() for c in env_chains.split(",") if c.strip()] or None

    limit = int(_env_float("DISCOVERY_STREAM_FETCH_LIMIT", 40))
    tokens = fetch_trending_tokens(limit=limit, chains=chains)
    if not tokens:
        return {"status": "no_data"}

    min_liquidity = _env_float("DISCOVERY_STREAM_MIN_LIQUIDITY", 50000.0)
    min_volume = _env_float("DISCOVERY_STREAM_MIN_VOLUME", 50000.0)
    min_abs_move = _env_float("DISCOVERY_STREAM_MIN_ABS_MOVE_24H", 3.0)  # percent

    scored: List[Dict[str, Any]] = []
    for tok in tokens:
        symbol = (tok.symbol or "").strip().upper()
        chain = _CHAIN_ALIASES.get((tok.chain or "").strip().lower())
        if not symbol or not chain or symbol in _STABLE_SYMBOLS:
            continue
        liquidity = float(tok.liquidity_usd or 0.0)
        volume = float(tok.volume_24h_usd or 0.0)
        change_24h = float(tok.price_change_24h or 0.0)
        change_1h = float(tok.price_change_1h or 0.0)
        if liquidity < min_liquidity or volume < min_volume:
            continue
        # Interesting either direction: bullish momentum to ride, bearish
        # capitulation to buy low on. Flat movers add nothing.
        if abs(change_24h) < min_abs_move and abs(change_1h) < min_abs_move / 3.0:
            continue
        # Wash-trading heuristic: daily volume wildly above liquidity is fake.
        if liquidity > 0 and (volume / liquidity) > _env_float("DISCOVERY_STREAM_MAX_VOL_LIQ_RATIO", 50.0):
            continue
        report = heuristic_screen(
            tax_buy=0.0,
            tax_sell=0.0,
            liquidity_usd=liquidity,
            price_change_24h=change_24h,
        )
        if getattr(report, "verdict", "") == "honeypot":
            continue
        scored.append(
            {
                "symbol": symbol,
                "chain": chain,
                "pair_address": tok.pair_address,
                "bull_score": change_1h,
                "bear_score": change_24h,
                "liquidity": liquidity,
                "volume": volume,
                "weight": abs(change_24h) + abs(change_1h) * 2.0,
            }
        )

    if not scored:
        return {"status": "no_candidates", "fetched": len(tokens)}

    # GoPlus / registry safety on the raw token addresses when available.
    try:
        from services.token_safety import filter_token_pairs

        address_pairs = [
            (entry["chain"], entry["pair_address"])
            for entry in scored
            if entry.get("pair_address")
        ]
        safe = set(filter_token_pairs(address_pairs))
        scored = [
            entry
            for entry in scored
            if not entry.get("pair_address") or (entry["chain"], entry["pair_address"]) in safe
        ]
    except Exception:
        pass

    scored.sort(key=lambda e: e["weight"], reverse=True)
    cap = max_added if max_added is not None else int(_env_float("DISCOVERY_STREAM_MAX_ADDED", 8))
    additions = [f"{entry['symbol']}-USDC" for entry in scored[:cap]]

    try:
        from services.watchlists import mutate_watchlist

        result = mutate_watchlist("stream", add=additions)
        stream_size = len(result.get("stream", []))
    except Exception as exc:
        log_message("discovery-bridge", f"watchlist update failed: {exc}", severity="warning")
        return {"status": "error", "error": str(exc)}

    log_message(
        "discovery-bridge",
        f"streamed {len(additions)} movers into watchlist "
        f"({', '.join(additions)}); stream size now {stream_size}",
    )
    return {
        "status": "ok",
        "added": additions,
        "candidates": len(scored),
        "fetched": len(tokens),
        "stream_size": stream_size,
    }
