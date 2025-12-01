from __future__ import annotations

import os
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Dict, Iterable, List, Optional

from cache import CachePrices
from services.offline_market import OfflinePriceStore
from services.token_catalog import get_core_token_map


def _lower(value: str | None) -> str:
    return (value or "").strip().lower()


def _normalize_symbol(symbol: str | None) -> str:
    return (symbol or "").strip().upper()


def _quantity_to_decimal(value: object) -> Decimal:
    try:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(0)


@dataclass
class PriceResolution:
    usd: Optional[float]
    source: Optional[str]
    ts: Optional[float]
    refreshed: bool = False


class UsdValuation:
    """
    Resolve USD valuations for wallet balances by cross-referencing the cached
    price DB with offline market snapshots. This keeps wallet totals stable
    even when upstream RPC feeds are stale.
    """

    def __init__(
        self,
        *,
        cache_prices: Optional[CachePrices] = None,
        offline_store: Optional[OfflinePriceStore] = None,
        max_age_sec: Optional[float] = None,
    ) -> None:
        self.cache_prices = cache_prices or CachePrices()
        self.offline_store = offline_store or OfflinePriceStore()
        env_max_age = os.getenv("WALLET_USD_PRICE_MAX_AGE")
        self.max_age_sec = (
            float(env_max_age)
            if env_max_age is not None
            else (float(max_age_sec) if max_age_sec is not None else 900.0)
        )
        self._symbol_lookup: Dict[str, Dict[str, str]] = self._build_symbol_lookup()

    def _build_symbol_lookup(self) -> Dict[str, Dict[str, str]]:
        lookup: Dict[str, Dict[str, str]] = {}
        core_map = get_core_token_map()
        for chain, mapping in core_map.items():
            chain_l = _lower(chain)
            dest = lookup.setdefault(chain_l, {})
            for symbol, addr in (mapping or {}).items():
                dest[_lower(addr)] = _normalize_symbol(symbol)
        return lookup

    def _symbol_candidates(self, chain: str, token: str, symbol: str | None) -> List[str]:
        candidates: List[str] = []
        sym_norm = _normalize_symbol(symbol)
        if sym_norm:
            candidates.append(sym_norm)
        mapped = self._symbol_lookup.get(chain, {}).get(token)
        if mapped and mapped not in candidates:
            candidates.append(mapped)
        # Normalize wrapped/native cases so ETH/WETH share pricing
        native_aliases = {
            "WETH": "ETH",
            "WBTC": "BTC",
            "USDBC": "USDC",
            "USDCE": "USDC",
        }
        extra = []
        for entry in list(candidates):
            alias = native_aliases.get(entry)
            if alias:
                extra.append(alias)
        candidates.extend(alias for alias in extra if alias not in candidates)
        if not candidates:
            candidates.append("ETH")
        return candidates

    def _price_from_cache(self, chain: str, token: str) -> Optional[PriceResolution]:
        row = self.cache_prices.db.fetch_price(chain, token)
        if not row:
            return None
        usd_val = row["usd"]
        ts = float(row["ts"] or 0.0)
        age_ok = True
        if self.max_age_sec is not None:
            age_ok = (time.time() - ts) <= self.max_age_sec
        if not age_ok:
            return PriceResolution(usd=None, source=row["source"], ts=ts, refreshed=False)
        try:
            usd_float = float(usd_val)
        except Exception:
            usd_float = None
        return PriceResolution(usd=usd_float, source=row["source"], ts=ts, refreshed=False)

    def _price_from_offline(self, candidates: Iterable[str]) -> Optional[PriceResolution]:
        for symbol in candidates:
            snapshot = self.offline_store.get_price(symbol)
            if not snapshot:
                continue
            price = float(snapshot.price or 0.0)
            if price <= 0:
                continue
            return PriceResolution(
                usd=price,
                source=snapshot.source or "offline",
                ts=float(snapshot.ts or 0.0),
                refreshed=True,
            )
        return None

    def resolve_price(self, chain: str, token: str, symbol: str | None) -> Optional[PriceResolution]:
        chain_l = _lower(chain)
        token_l = _lower(token)
        cache_row = self._price_from_cache(chain_l, token_l)
        if cache_row and cache_row.usd:
            return cache_row
        candidates = self._symbol_candidates(chain_l, token_l, symbol)
        offline = self._price_from_offline(candidates)
        if offline:
            return offline
        return cache_row

    def refresh_row(self, row: Dict[str, object]) -> Optional[Dict[str, object]]:
        quantity = _quantity_to_decimal(row.get("quantity"))
        if quantity <= 0:
            return None
        chain = _lower(str(row.get("chain")))
        token = _lower(str(row.get("token")))
        symbol = row.get("symbol")
        resolution = self.resolve_price(chain, token, symbol)
        if not resolution or not resolution.usd:
            return None
        usd_value = float((quantity * Decimal(str(resolution.usd))).quantize(Decimal("0.0001")))
        current_usd = row.get("usd")
        try:
            current_float = float(current_usd)
        except Exception:
            current_float = None
        if current_float is not None and abs(current_float - usd_value) < max(0.01, current_float * 0.01):
            return None
        row["usd"] = usd_value
        row["usd_source"] = resolution.source
        row["usd_ts"] = resolution.ts
        return {
            "chain": row.get("chain"),
            "token": row.get("token"),
            "symbol": row.get("symbol"),
            "usd": usd_value,
            "source": resolution.source,
            "refreshed": resolution.refreshed,
        }
