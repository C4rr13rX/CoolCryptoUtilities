"""
Multi-Wallet Manager

Automatically creates new HD wallets when a single wallet's stable bank
crosses a configurable threshold.  All mnemonics are persisted in SecureVault
(MNEMONIC_0, MNEMONIC_1, …) and every wallet is used as an independent
trading portfolio.

Configuration (SecureVault / environment):
  MULTI_WALLET_THRESHOLD      – USD amount per wallet that triggers a new
                                 wallet creation (default: 150000)
  MULTI_WALLET_MAX_BALANCE    – Max USD a single wallet should hold before
                                 distributing to a new one (default: 300000)
  MULTI_WALLET_ENABLED        – "1" to enable auto-creation (default: "1")
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STATE_FILE = _PROJECT_ROOT / "runtime" / "multi_wallet_state.json"


class WalletInfo:
    """Lightweight descriptor for a managed wallet."""

    __slots__ = ("index", "mnemonic", "address", "derivation_path")

    def __init__(
        self,
        index: int,
        mnemonic: str,
        address: str,
        derivation_path: str = "m/44'/60'/0'/0/0",
    ) -> None:
        self.index = index
        self.mnemonic = mnemonic
        self.address = address
        self.derivation_path = derivation_path

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "address": self.address,
            "derivation_path": self.derivation_path,
        }


def _setting_name(index: int) -> str:
    """Return the SecureVault setting name for the given wallet index."""
    return f"MNEMONIC_{index}"


def _derive_address(mnemonic: str, path: str = "m/44'/60'/0'/0/0") -> str:
    from eth_account import Account  # type: ignore

    try:
        Account.enable_unaudited_hdwallet_features()
    except Exception:
        pass
    acct = Account.from_mnemonic(mnemonic, account_path=path)
    return acct.address


def _generate_mnemonic() -> str:
    from eth_account import Account  # type: ignore

    try:
        Account.enable_unaudited_hdwallet_features()
    except Exception:
        pass
    acct, mnemonic = Account.create_with_mnemonic()
    return mnemonic


class MultiWalletManager:
    """
    Manages multiple HD wallets.

    Wallets are stored in SecureVault as MNEMONIC_0, MNEMONIC_1, etc.
    The original MNEMONIC setting is migrated to MNEMONIC_0 on first use.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wallets: List[WalletInfo] = []
        self._last_check: float = 0.0

    # ── configuration helpers ──────────────────────────────────────────

    @staticmethod
    def threshold() -> float:
        try:
            return float(os.environ.get("MULTI_WALLET_THRESHOLD", "150000"))
        except (TypeError, ValueError):
            return 150_000.0

    @staticmethod
    def max_balance() -> float:
        try:
            return float(os.environ.get("MULTI_WALLET_MAX_BALANCE", "300000"))
        except (TypeError, ValueError):
            return 300_000.0

    @staticmethod
    def enabled() -> bool:
        return os.environ.get("MULTI_WALLET_ENABLED", "1").strip() in ("1", "true", "yes")

    # ── wallet discovery ───────────────────────────────────────────────

    def load_wallets(self, user=None) -> List[WalletInfo]:
        """Load all wallets from SecureVault.  Migrates MNEMONIC → MNEMONIC_0."""
        wallets: List[WalletInfo] = []
        try:
            settings = self._get_all_mnemonics(user)
        except Exception as exc:
            logger.warning("multi-wallet: failed to load mnemonics: %s", exc)
            return wallets

        # Migration: if MNEMONIC exists but MNEMONIC_0 does not, copy it.
        if "MNEMONIC" in settings and "MNEMONIC_0" not in settings:
            self._save_mnemonic(user, 0, settings["MNEMONIC"])
            settings["MNEMONIC_0"] = settings["MNEMONIC"]

        idx = 0
        while True:
            key = _setting_name(idx)
            mnemonic = settings.get(key, "").strip()
            if not mnemonic:
                break
            try:
                path = os.environ.get("DERIVATION_PATH", "m/44'/60'/0'/0/0")
                address = _derive_address(mnemonic, path)
                wallets.append(WalletInfo(idx, mnemonic, address, path))
            except Exception as exc:
                logger.warning("multi-wallet: failed to derive wallet %d: %s", idx, exc)
            idx += 1

        with self._lock:
            self._wallets = list(wallets)
        return wallets

    def get_wallets(self) -> List[WalletInfo]:
        """Return cached wallets (call load_wallets first)."""
        with self._lock:
            return list(self._wallets)

    def wallet_count(self) -> int:
        with self._lock:
            return len(self._wallets)

    # ── auto-creation logic ────────────────────────────────────────────

    def check_and_create(
        self,
        wallet_balances: Dict[str, float],
        user=None,
    ) -> Optional[WalletInfo]:
        """
        Check if any wallet has exceeded max_balance and create a new one.

        Args:
            wallet_balances: mapping of wallet address (lower) → USD value
            user: Django user for SecureVault storage

        Returns:
            Newly created WalletInfo, or None.
        """
        if not self.enabled():
            return None

        max_bal = self.max_balance()
        threshold = self.threshold()
        if max_bal <= 0 or threshold <= 0:
            return None

        # Cooldown: don't check more than once per minute.
        now = time.time()
        if now - self._last_check < 60:
            return None
        self._last_check = now

        with self._lock:
            wallets = list(self._wallets)

        if not wallets:
            return None

        # Check if any wallet exceeds max_balance.
        for w in wallets:
            addr = w.address.lower()
            balance = wallet_balances.get(addr, 0.0)
            if balance >= max_bal:
                logger.info(
                    "multi-wallet: wallet %d (%s) has $%.2f >= $%.2f max, creating new wallet",
                    w.index,
                    addr[:10],
                    balance,
                    max_bal,
                )
                return self._create_wallet(user)

        return None

    def create_wallet(self, user=None) -> WalletInfo:
        """Manually create a new wallet."""
        return self._create_wallet(user)

    def _create_wallet(self, user=None) -> WalletInfo:
        with self._lock:
            next_idx = len(self._wallets)

        mnemonic = _generate_mnemonic()
        path = os.environ.get("DERIVATION_PATH", "m/44'/60'/0'/0/0")
        address = _derive_address(mnemonic, path)
        wallet = WalletInfo(next_idx, mnemonic, address, path)

        self._save_mnemonic(user, next_idx, mnemonic)

        with self._lock:
            self._wallets.append(wallet)

        logger.info(
            "multi-wallet: created wallet %d → %s",
            next_idx,
            address,
        )
        self._persist_state()
        return wallet

    # ── aggregate state ────────────────────────────────────────────────

    def aggregate_state(self, per_wallet_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine per-wallet state snapshots into an aggregate view.

        Each item in per_wallet_states should be the result of
        ``load_wallet_state()`` keyed by wallet index.
        """
        total_usd = 0.0
        all_balances: List[Dict[str, Any]] = []
        all_transfers: Dict[str, Any] = {}
        all_nfts: List[Dict[str, Any]] = []
        wallet_summaries: List[Dict[str, Any]] = []

        for state in per_wallet_states:
            totals = state.get("totals") or {}
            usd = float(totals.get("usd", 0))
            total_usd += usd
            wallet_summaries.append({
                "index": state.get("wallet_index", 0),
                "wallet": state.get("wallet"),
                "usd": usd,
                "updated_at": state.get("updated_at"),
            })
            for bal in state.get("balances") or []:
                bal_copy = dict(bal)
                bal_copy["wallet_index"] = state.get("wallet_index", 0)
                all_balances.append(bal_copy)
            for chain, txs in (state.get("transfers") or {}).items():
                existing = all_transfers.get(chain, [])
                if isinstance(txs, list):
                    existing.extend(txs)
                all_transfers[chain] = existing
            for nft in state.get("nfts") or []:
                nft_copy = dict(nft)
                nft_copy["wallet_index"] = state.get("wallet_index", 0)
                all_nfts.append(nft_copy)

        return {
            "wallet_count": len(per_wallet_states),
            "wallets": wallet_summaries,
            "totals": {"usd": round(total_usd, 2)},
            "balances": all_balances,
            "transfers": all_transfers,
            "nfts": all_nfts,
        }

    # ── persistence helpers ────────────────────────────────────────────

    @staticmethod
    def _get_all_mnemonics(user=None) -> Dict[str, str]:
        """Load all MNEMONIC* settings from SecureVault."""
        try:
            from services.secure_settings import get_settings_for_user, default_env_user

            user = user or default_env_user()
            all_settings = get_settings_for_user(user)
        except Exception:
            all_settings = {}

        result: Dict[str, str] = {}
        for key, value in all_settings.items():
            if key == "MNEMONIC" or key.startswith("MNEMONIC_"):
                result[key] = value
        # Also check environment.
        for key, value in os.environ.items():
            if key == "MNEMONIC" or key.startswith("MNEMONIC_"):
                if key not in result:
                    result[key] = value
        return result

    @staticmethod
    def _save_mnemonic(user, index: int, mnemonic: str) -> None:
        """Persist a mnemonic to SecureVault."""
        try:
            from services.secure_settings import encrypt_secret, default_env_user

            if user is None:
                user = default_env_user()
            if user is None:
                logger.warning("multi-wallet: no user available to save mnemonic_%d", index)
                return

            from securevault.models import SecureSetting
            from django.db import transaction

            name = _setting_name(index)
            with transaction.atomic():
                setting, _ = SecureSetting.objects.get_or_create(
                    user=user,
                    name=name,
                    category="default",
                    defaults={"is_secret": True},
                )
                setting.is_secret = True
                payload = encrypt_secret(mnemonic)
                setting.value_plain = None
                setting.ciphertext = payload["ciphertext"]
                setting.encapsulated_key = payload["encapsulated_key"]
                setting.nonce = payload["nonce"]
                setting.save()

            logger.info("multi-wallet: saved %s to SecureVault", name)
        except Exception as exc:
            logger.warning("multi-wallet: failed to save mnemonic_%d: %s", index, exc)

    def _persist_state(self) -> None:
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    "wallet_count": len(self._wallets),
                    "wallets": [w.as_dict() for w in self._wallets],
                    "updated_at": time.time(),
                }
            _STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ── trading helpers ────────────────────────────────────────────────

    def get_bridges(self) -> List:
        """
        Create UltraSwapBridge instances for all loaded wallets.
        Returns a list of (wallet_index, bridge) tuples.
        """
        bridges = []
        with self._lock:
            wallets = list(self._wallets)
        for w in wallets:
            try:
                from router_wallet import UltraSwapBridge

                bridge = UltraSwapBridge(
                    mnemonic=w.mnemonic,
                    derivation_path=w.derivation_path,
                )
                bridges.append((w.index, bridge))
            except Exception as exc:
                logger.warning("multi-wallet: bridge init failed for wallet %d: %s", w.index, exc)
        return bridges

    def get_wallet_env(self, index: int) -> Dict[str, str]:
        """Return environment overrides to run a subprocess as a specific wallet."""
        with self._lock:
            for w in self._wallets:
                if w.index == index:
                    return {
                        "MNEMONIC": w.mnemonic,
                        "DERIVATION_PATH": w.derivation_path,
                        "MULTI_WALLET_INDEX": str(index),
                    }
        return {}


# Module-level singleton.
multi_wallet_manager = MultiWalletManager()
