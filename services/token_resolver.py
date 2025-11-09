from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from web3 import Web3

from cache import CacheBalances
from services.cli_utils import is_native
from services.token_catalog import get_core_token_map


@dataclass(frozen=True)
class ResolvedToken:
    """Canonical representation of a token identifier for wallet actions."""

    chain: str
    identifier: str
    symbol: Optional[str]
    is_native: bool


class TokenResolver:
    """
    Converts user-facing token inputs (symbol/address/native) into canonical
    identifiers understood by on-chain send/swap/bridge helpers. Prefers core
    tokenlists and cached wallet balances before falling back to direct input.
    """

    def __init__(
        self,
        *,
        cache_balances: CacheBalances | None = None,
        token_map: Optional[Mapping[str, Mapping[str, str]]] = None,
    ) -> None:
        self._cache_balances = cache_balances
        raw_map = token_map or get_core_token_map()
        self._token_map: Dict[str, Dict[str, str]] = {}
        for chain, entries in raw_map.items():
            chain_l = str(chain or "").strip().lower()
            if not chain_l:
                continue
            canon: Dict[str, str] = {}
            for symbol, addr in (entries or {}).items():
                sym = str(symbol or "").strip().upper()
                address = str(addr or "").strip()
                if sym and address:
                    canon[sym] = address
            if canon:
                self._token_map[chain_l] = canon

    def resolve(
        self,
        *,
        chain: str,
        token: str,
        wallet: Optional[str] = None,
        native_sentinel: str = "native",
        native_symbol: Optional[str] = None,
    ) -> ResolvedToken:
        chain_l = str(chain or "").strip().lower()
        if not chain_l:
            raise ValueError("chain is required")
        raw = str(token or "").strip()
        if not raw:
            raise ValueError("token is required")
        if is_native(raw):
            symbol = native_symbol or raw.upper() or chain_l.upper()
            return ResolvedToken(chain=chain_l, identifier=native_sentinel, symbol=symbol, is_native=True)
        normalized = self._normalize_address(raw)
        if normalized:
            return ResolvedToken(chain=chain_l, identifier=normalized, symbol=None, is_native=False)
        resolved = self._lookup_symbol(chain_l, raw)
        if not resolved and wallet:
            resolved = self._lookup_wallet_symbol(chain_l, wallet, raw)
        if not resolved:
            raise ValueError(f"Unknown token '{token}' on {chain}; provide checksum address.")
        checksum = Web3.to_checksum_address(resolved)
        return ResolvedToken(chain=chain_l, identifier=checksum, symbol=raw.upper(), is_native=False)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _normalize_address(value: str) -> Optional[str]:
        val = value.strip()
        if val.lower().startswith("0x") and len(val) == 42:
            return Web3.to_checksum_address(val)
        if len(val) == 40 and all(c in "0123456789abcdefABCDEF" for c in val):
            return Web3.to_checksum_address("0x" + val)
        return None

    def _lookup_symbol(self, chain: str, token: str) -> Optional[str]:
        entries = self._token_map.get(chain)
        if not entries:
            return None
        sym = token.strip().upper()
        return entries.get(sym)

    def _ensure_cache(self) -> CacheBalances:
        if self._cache_balances is None:
            self._cache_balances = CacheBalances()
        return self._cache_balances

    def _lookup_wallet_symbol(self, chain: str, wallet: str, token: str) -> Optional[str]:
        cb = self._ensure_cache()
        try:
            state = cb.get_state(wallet, chain)
        except Exception:
            return None
        target = token.strip().upper()
        tokens = (state or {}).get("tokens", {}) or {}
        for addr, meta in tokens.items():
            symbol = str(meta.get("symbol") or "").strip().upper()
            name = str(meta.get("name") or "").strip().upper()
            if target and (target == symbol or target == name):
                return addr
        return None


__all__ = ["ResolvedToken", "TokenResolver"]
