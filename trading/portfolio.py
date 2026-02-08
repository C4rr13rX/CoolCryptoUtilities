from __future__ import annotations

import os
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Iterable, Optional, Tuple

from cache import CacheBalances
from db import get_db, TradingDatabase
from trading.constants import PRIMARY_CHAIN

try:  # pragma: no cover - optional dependency
    from eth_account import Account  # type: ignore
except Exception:  # pragma: no cover - fallback
    Account = None  # type: ignore

STABLE_TOKENS = {"USDC", "USDT", "DAI", "USDP", "BUSD", "TUSD"}
NATIVE_SYMBOL = {"ethereum": "ETH", "arbitrum": "ETH", "optimism": "ETH", "base": "ETH", "polygon": "MATIC"}


@dataclass
class TokenHolding:
    token: str
    symbol: str
    quantity: float
    usd: float
    chain: str
    raw: Dict[str, str]

    def as_dict(self) -> Dict[str, float]:
        return {
            "token": self.token,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "usd": self.usd,
            "chain": self.chain,
        }


class PortfolioState:
    """
    Lightweight view of wallet balances used by the trading scheduler.
    Pulls cached balances via CacheBalances (fast) and refreshes on a cadence.
    """

    def __init__(
        self,
        *,
        wallet: Optional[str] = None,
        chains: Iterable[str] = (PRIMARY_CHAIN,),
        refresh_interval: float = 300.0,
        db: Optional[TradingDatabase] = None,
    ) -> None:
        self.db = db or get_db()
        self.chains = tuple(chains)
        self.refresh_interval = refresh_interval
        self._next_refresh: float = 0.0
        self._last_refresh: float = 0.0
        self.holdings: Dict[Tuple[str, str], TokenHolding] = {}
        self.native_balances: Dict[str, float] = {}
        self.native_usd: Dict[str, float] = {}
        self.native_prices: Dict[str, float] = {}
        self.wallet = (wallet or os.getenv("PRIMARY_WALLET") or "").lower()
        if not self.wallet:
            self.wallet = self._derive_wallet_from_mnemonic()
        self.cache = CacheBalances(db=self.db)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_wallet_from_mnemonic(self) -> str:
        mnemonic = os.getenv("MNEMONIC", "").strip()
        if not mnemonic or Account is None:
            raise RuntimeError(
                "PortfolioState requires a wallet address. "
                "Set PRIMARY_WALLET or provide MNEMONIC + DERIVATION_PATH."
            )
        path = os.getenv("DERIVATION_PATH", "m/44'/60'/0'/0/0")
        try:
            if hasattr(Account, "enable_unaudited_hdwallet_features"):
                try:
                    Account.enable_unaudited_hdwallet_features()  # type: ignore[attr-defined]
                except Exception:
                    pass
            acct = Account.from_mnemonic(mnemonic, account_path=path)  # type: ignore[arg-type]
            return acct.address.lower()
        except Exception as exc:  # pragma: no cover - depends on env secrets
            raise RuntimeError(f"Failed to derive wallet from mnemonic: {exc}") from exc

    def _parse_quantity(self, value: Optional[str]) -> float:
        if value is None:
            return 0.0
        try:
            return float(Decimal(str(value)))
        except Exception:
            return 0.0

    def _parse_usd(self, value: Optional[str]) -> float:
        if value is None:
            return 0.0
        try:
            return float(Decimal(str(value)))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now < self._next_refresh:
            return

        holdings: Dict[Tuple[str, str], TokenHolding] = {}
        native_balances: Dict[str, float] = {}
        native_usd: Dict[str, float] = {}

        for chain in self.chains:
            rows = self.cache.load(wallet=self.wallet, chains=[chain], include_zero=False)
            chain_lower = chain.lower()
            native_symbol = NATIVE_SYMBOL.get(chain_lower, chain_upper := chain.upper())
            native_balance = 0.0
            native_usd_total = 0.0
            for row in rows:
                quantity = self._parse_quantity(row.get("quantity"))
                usd = self._parse_usd(row.get("usd"))
                symbol = str(row.get("symbol") or "").upper()
                token_addr = str(row.get("token") or "").lower()
                if not symbol:
                    symbol = token_addr[:6] + "…" + token_addr[-4:]
                if symbol in (native_symbol, "ETH", "MATIC"):
                    native_balance += quantity
                    native_usd_total += usd
                holdings[(chain_lower, symbol)] = TokenHolding(
                    token=token_addr,
                    symbol=symbol,
                    quantity=quantity,
                    usd=usd,
                    chain=chain_lower,
                    raw=row,
                )
            native_balances[chain_lower] = native_balance
            native_usd[chain_lower] = native_usd_total

        holdings = self._filter_holdings(holdings)
        self.holdings = holdings
        self.native_balances = native_balances
        self.native_usd = native_usd
        self.native_prices = {
            chain: (native_usd[chain] / native_balances[chain])
            for chain in native_balances
            if native_balances.get(chain, 0.0) > 0 and native_usd.get(chain, 0.0) > 0
        }
        self._last_refresh = now
        self._next_refresh = now + self.refresh_interval

    def get_quantity(self, symbol: str, chain: str = "ethereum") -> float:
        key = (chain.lower(), symbol.upper())
        holding = self.holdings.get(key)
        if holding:
            return holding.quantity
        # allow address lookup
        for (c, sym), h in self.holdings.items():
            if c == chain.lower() and h.token.lower() == symbol.lower():
                return h.quantity
        return 0.0

    def get_native_balance(self, chain: str = "ethereum") -> float:
        return self.native_balances.get(chain.lower(), 0.0)

    def get_native_usd(self, chain: str = "ethereum") -> float:
        return self.native_usd.get(chain.lower(), 0.0)

    def get_native_price(self, chain: str = "ethereum") -> Optional[float]:
        return self.native_prices.get(chain.lower())

    def stable_liquidity(self, chain: str = "ethereum") -> float:
        chain_l = chain.lower()
        total = 0.0
        for (c, sym), holding in self.holdings.items():
            if c != chain_l:
                continue
            if sym in STABLE_TOKENS:
                total += holding.quantity
        return total

    def summary(self) -> Dict[str, float]:
        return {
            "wallet": self.wallet,
            "native_eth": self.get_native_balance("ethereum"),
            "stable_usd": self.stable_liquidity("ethereum"),
            "holdings": len(self.holdings),
            "last_refresh": self._last_refresh,
        }

    def _filter_holdings(self, holdings: Dict[Tuple[str, str], TokenHolding]) -> Dict[Tuple[str, str], TokenHolding]:
        key = os.getenv("GOPLUS_APP_KEY")
        secret = os.getenv("GOPLUS_APP_SECRET")
        if not key or not secret:
            return holdings
        try:
            from filter_scams import FilterScamTokens

            specs: List[str] = []
            addr_key: Dict[str, Tuple[str, str]] = {}
            for (chain, symbol), holding in holdings.items():
                token = holding.token
                if token and token.lower().startswith("0x") and len(token) >= 42:
                    spec = f"{chain}:{token}"
                    specs.append(spec)
                    addr_key[token.lower()] = (chain, symbol)
            if not specs:
                return holdings
            filterer = FilterScamTokens()
            result = filterer.filter(specs)
            flagged = set(result.flagged.keys())
            if not flagged:
                return holdings
            print(f"[portfolio] filtering {len(flagged)} honeypot/high-risk tokens from holdings")
            for addr_lower in flagged:
                key_tuple = addr_key.get(addr_lower)
                if key_tuple and key_tuple in holdings:
                    holdings.pop(key_tuple, None)
            return holdings
        except Exception as exc:
            print(f"[portfolio] scam filter unavailable: {exc}")
            return holdings

    @property
    def refresh_interval(self) -> float:
        return self._refresh_interval

    @refresh_interval.setter
    def refresh_interval(self, value: float) -> None:
        self._refresh_interval = max(60.0, float(value))
