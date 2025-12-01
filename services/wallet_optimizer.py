from __future__ import annotations

import math
import os
import time
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from services.wallet_logger import wallet_log
from services.token_catalog import get_core_token_map
from services.wallet_watch import build_core_watch_tokens, core_watch_limit

getcontext().prec = 60

DEFAULT_CHAINS = ("ethereum", "base", "arbitrum", "optimism", "polygon")
DEFAULT_NATIVE_TOKEN = "0x0000000000000000000000000000000000000000"
DEFAULT_NATIVE_SYMBOL = {
    "ethereum": "ETH",
    "base": "ETH",
    "arbitrum": "ETH",
    "optimism": "ETH",
    "polygon": "MATIC",
}


def _now() -> float:
    return time.time()


def _as_epoch(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        val = value.strip()
        if not val:
            return 0.0
        if val.endswith("Z"):
            val = val[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(val).timestamp()
        except Exception:
            pass
        try:
            return float(val)
        except Exception:
            return 0.0
    return 0.0


def _to_decimal_string(raw: int, decimals: int) -> str:
    if decimals <= 0:
        return str(raw)
    try:
        scale = Decimal(10) ** Decimal(decimals)
        if raw == 0:
            return "0"
        return str((Decimal(raw) / scale).normalize())
    except Exception:
        return str(raw)


def _parse_watch_tokens(blob: str) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    if not blob:
        return result
    entries = [chunk.strip() for chunk in blob.split(",") if chunk.strip()]
    for entry in entries:
        if ":" not in entry:
            continue
        chain, token = entry.split(":", 1)
        chain = chain.strip().lower()
        token = token.strip()
        if not chain or not token:
            continue
        result.setdefault(chain, []).append(token)
    return result


class AdaptiveGasOracle:
    """Adaptive EIP-1559 helper that keeps tips low without stalling txs."""

    def __init__(
        self,
        *,
        ttl_sec: int = 15,
        sample_size: int = 8,
        percentile_map: Optional[Mapping[str, float]] = None,
        default_strategy: str = "balanced",
    ) -> None:
        self.ttl_sec = max(3, int(ttl_sec))
        self.sample_size = max(4, int(sample_size))
        self.percentile_map = {
            "eco": 45.0,
            "balanced": 70.0,
            "urgent": 90.0,
        }
        if percentile_map:
            for key, value in percentile_map.items():
                try:
                    self.percentile_map[key.lower()] = float(value)
                except Exception:
                    continue
        self.default_strategy = default_strategy.lower()
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.max_jump_bps = max(0, int(os.getenv("GAS_MAX_JUMP_BPS", "2500") or "2500"))
        self.min_tip_gwei = float(os.getenv("GAS_MIN_TIP_GWEI", "0.5") or "0.5")
        self.max_tip_gwei = float(os.getenv("GAS_MAX_TIP_GWEI", "150") or "150")
        self._last_fee: Dict[str, Dict[str, int]] = {}
        self._chain_bias = self._load_chain_bias()

    def _load_chain_bias(self) -> Dict[str, float]:
        blob = os.getenv("GAS_CHAIN_BIASES")
        if not blob:
            return {}
        try:
            raw = json.loads(blob)
        except Exception:
            return {}
        out: Dict[str, float] = {}
        for chain, value in raw.items():
            try:
                out[str(chain).lower()] = float(value)
            except Exception:
                continue
        return out

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _env_strategy() -> str:
        val = os.getenv("GAS_STRATEGY") or os.getenv("GAS_URGENCY") or ""
        return val.strip().lower() or ""

    @staticmethod
    def _as_wei_from_gwei(value: str, w3) -> int:
        try:
            return int(w3.to_wei(float(value), "gwei"))
        except Exception:
            return int(float(value) * 1_000_000_000)

    def _pick_percentiles(self) -> List[float]:
        vals = sorted({max(1.0, min(99.0, v)) for v in self.percentile_map.values()})
        return vals or [70.0]

    def _strategy_for(self, urgency: Optional[str]) -> str:
        if urgency:
            return urgency.lower()
        env = self._env_strategy()
        if env:
            return env
        return self.default_strategy

    def _compute_eip1559(self, chain: str, w3, urgency: str) -> Dict[str, int]:
        percentiles = self._pick_percentiles()
        try:
            hist = w3.eth.fee_history(self.sample_size, "pending", percentiles)
        except Exception as exc:
            raise RuntimeError(f"fee_history unavailable: {exc}")
        base = int(hist.get("baseFeePerGas", [0])[-1])
        rewards = hist.get("reward") or []
        if not rewards:
            raise RuntimeError("fee_history rewards empty")
        pct_lookup = {round(p, 3): idx for idx, p in enumerate(percentiles)}
        target_pct = round(self.percentile_map.get(urgency, percentiles[-1]), 3)
        if target_pct not in pct_lookup:
            target_pct = min(percentiles, key=lambda x: abs(x - target_pct))
        idx = pct_lookup.get(target_pct, len(percentiles) - 1)
        tips: List[int] = []
        for block_entry in rewards[-self.sample_size:]:
            try:
                tips.append(int(block_entry[idx]))
            except Exception:
                continue
        if not tips:
            raise RuntimeError("fee_history tips empty")
        tips.sort()
        if len(tips) > 4:
            trim = max(1, len(tips) // 5)
            trimmed = tips[trim:-trim] if trim < len(tips) else tips
            if trimmed:
                tips = trimmed
        median_tip = int(statistics.median(tips))
        avg_tip = int(sum(tips) / len(tips))
        blended_tip = int(max(median_tip, avg_tip * 0.85))
        floor_tip_gwei = float(os.getenv("GAS_TIP_FLOOR_GWEI", "1") or "1")
        tip = max(blended_tip, self._as_wei_from_gwei(str(floor_tip_gwei), w3))
        tip_min = self._as_wei_from_gwei(str(self.min_tip_gwei), w3)
        tip_max = self._as_wei_from_gwei(str(self.max_tip_gwei), w3)
        if tip_max and tip_min and tip_max < tip_min:
            tip_max = tip_min
        tip = max(tip_min, min(tip, tip_max))
        mult = float(os.getenv("GAS_BASE_MULT", "1.35") or "1.35")
        cap_env = os.getenv("GAS_BASE_MAX_MULT")
        base_cap = float(cap_env) if cap_env else None
        if urgency == "eco":
            mult = max(mult * 0.85, 1.05)
        elif urgency == "urgent":
            mult = max(mult * 1.15, 1.45)
        if base_cap:
            mult = min(mult, base_cap)
        max_fee = int(base * mult) + tip
        if max_fee < base + tip:
            max_fee = base + tip
        chain_bias = self._chain_bias.get(str(chain).lower(), 1.0)
        if chain_bias and chain_bias != 1.0:
            max_fee = max(1, int(max_fee * chain_bias))
            tip = max(1, int(tip * chain_bias))
            tip = max(tip_min, min(tip, tip_max))
        return {"maxFeePerGas": max_fee, "maxPriorityFeePerGas": tip}

    def _smooth_fee_jump(self, chain: str, fees: Dict[str, int]) -> Dict[str, int]:
        chain_key = (chain or "default").lower()
        if self.max_jump_bps <= 0:
            self._last_fee[chain_key] = dict(fees)
            return fees
        prev = self._last_fee.get(chain_key)
        ref_key = "gasPrice" if "gasPrice" in fees else "maxFeePerGas"
        current_val = int(fees.get(ref_key, 0) or 0)
        if prev:
            prev_ref = ref_key if ref_key in prev else ("gasPrice" if "gasPrice" in prev else "maxFeePerGas")
            prev_val = int(prev.get(prev_ref, 0) or 0)
            if prev_val > 0 and current_val > 0:
                limit = int(prev_val * (1 + self.max_jump_bps / 10_000))
                if limit > 0 and current_val > limit:
                    scale = limit / current_val
                    fees[ref_key] = limit
                    if "maxPriorityFeePerGas" in fees and ref_key == "maxFeePerGas":
                        tip = int(fees["maxPriorityFeePerGas"])
                        scaled_tip = max(1, int(tip * scale))
                        fees["maxPriorityFeePerGas"] = min(scaled_tip, limit)
        if (
            "maxPriorityFeePerGas" in fees
            and "maxFeePerGas" in fees
            and fees["maxPriorityFeePerGas"] > fees["maxFeePerGas"]
        ):
            fees["maxPriorityFeePerGas"] = max(1, fees["maxFeePerGas"])
        self._last_fee[chain_key] = dict(fees)
        return fees

    def suggest(self, chain: str, w3, urgency: Optional[str] = None) -> Dict[str, int]:
        strategy = self._strategy_for(urgency)
        normalized_chain = (chain or "default").lower()
        cache_key = (normalized_chain, strategy)
        entry = self._cache.get(cache_key)
        now = _now()
        if entry and (now - entry["ts"]) <= self.ttl_sec:
            return dict(entry["fees"])

        override = os.getenv("GAS_PRICE_GWEI")
        if override:
            fee = {"gasPrice": self._as_wei_from_gwei(override, w3)}
            self._cache[cache_key] = {"fees": fee, "ts": now}
            return dict(fee)

        try:
            fees = self._compute_eip1559(normalized_chain, w3, strategy)
        except Exception:
            gp = int(getattr(w3.eth, "gas_price", 0) or self._as_wei_from_gwei("30", w3))
            fees = {"gasPrice": gp}
        fees = self._smooth_fee_jump(normalized_chain, dict(fees))
        self._cache[cache_key] = {"fees": fees, "ts": now}
        return dict(fees)

    def apply_to_tx(self, chain: str, w3, tx: MutableMapping[str, Any], urgency: Optional[str] = None) -> Dict[str, int]:
        fees = self.suggest(chain, w3, urgency=urgency)
        if "gasPrice" in fees:
            tx["gasPrice"] = int(fees["gasPrice"])
            tx.pop("maxFeePerGas", None)
            tx.pop("maxPriorityFeePerGas", None)
        else:
            tx["maxFeePerGas"] = int(fees["maxFeePerGas"])
            tx["maxPriorityFeePerGas"] = int(fees["maxPriorityFeePerGas"])
            tx.pop("gasPrice", None)
        return fees


@dataclass
class _TokenPlan:
    chain: str
    token: str
    reason: str
    meta: Dict[str, Any]


class RealtimeBalanceRefresher:
    """Lightweight balance updater driven by recent transfers + staleness checks."""

    def __init__(
        self,
        *,
        bridge,
        cache_balances,
        cache_transfers,
        native_token: str = DEFAULT_NATIVE_TOKEN,
        max_workers: Optional[int] = None,
        stale_seconds: Optional[int] = None,
        core_token_map: Optional[Mapping[str, Mapping[str, str]]] = None,
        core_watch_limit: Optional[int] = None,
    ) -> None:
        self.bridge = bridge
        self.cache_balances = cache_balances
        self.cache_transfers = cache_transfers
        self.native_token = (native_token or DEFAULT_NATIVE_TOKEN).lower()
        self.max_workers = max_workers or int(os.getenv("BALANCE_REFRESH_WORKERS", "6") or "6")
        self.stale_seconds = (
            stale_seconds if stale_seconds is not None else int(os.getenv("BALANCE_FAST_STALE_SEC", "45") or "45")
        )
        self.env_watch = _parse_watch_tokens(os.getenv("WALLET_PINNED_TOKENS", ""))
        self._core_token_map = dict(core_token_map or get_core_token_map())
        self._core_watch_limit = self._resolve_core_watch_limit(core_watch_limit)
        self._core_watch_cache = self._build_core_watch_cache()
        try:
            self._popular_watch_cap = max(3, int(os.getenv("BALANCE_FAST_POPULAR_LIMIT", "10") or "10"))
        except Exception:
            self._popular_watch_cap = 10
        try:
            self._popular_recent_minutes = max(
                1,
                int(os.getenv("BALANCE_FAST_POPULAR_MINUTES", "90") or "90"),
            )
        except Exception:
            self._popular_recent_minutes = 90

    # ------------------------------------------------------------------ planning
    def _default_chains(self, chains: Optional[Iterable[str]]) -> List[str]:
        if chains:
            ordered = [str(ch).lower() for ch in chains if str(ch).strip()]
            return ordered or list(DEFAULT_CHAINS)
        env_blob = os.getenv("WALLET_FAST_CHAINS")
        if env_blob:
            entries = [chunk.strip().lower() for chunk in env_blob.split(",") if chunk.strip()]
            if entries:
                return entries
        return list(DEFAULT_CHAINS)

    def _stale_tokens(self, meta_map: Mapping[str, Dict[str, Any]]) -> List[str]:
        if self.stale_seconds <= 0:
            return []
        cutoff = _now() - self.stale_seconds
        stale: List[str] = []
        for token, meta in (meta_map or {}).items():
            updated = _as_epoch(meta.get("updated_at"))
            if updated <= 0 or updated < cutoff:
                stale.append(token)
        return stale

    def _touched_since(self, wallet: str, chain: str, since_block: int) -> List[str]:
        try:
            touched = self.cache_transfers.touched_tokens_since(wallet, chain, since_block)
        except Exception:
            return []
        if not touched:
            return []
        return list(touched)

    def _resolve_core_watch_limit(self, override: Optional[int]) -> Optional[int]:
        if override is not None:
            return override if override > 0 else None
        env_val = os.getenv("BALANCE_FAST_CORE_LIMIT")
        if env_val is not None and env_val.strip() != "":
            lowered = env_val.strip().lower()
            if lowered in {"0", "false", "no", "none", "off"}:
                return None
            try:
                parsed = int(env_val)
                return parsed if parsed > 0 else None
            except ValueError:
                pass
        return core_watch_limit()

    def _build_core_watch_cache(self) -> Dict[str, List[str]]:
        if not self._core_token_map:
            return {}
        chains = list(self._core_token_map.keys())
        watch = build_core_watch_tokens(chains, limit=self._core_watch_limit, token_map=self._core_token_map)
        return {chain: addrs for chain, addrs in watch.items() if addrs}

    def _core_watch_for(self, chain: str) -> List[str]:
        return list(self._core_watch_cache.get(chain, []))

    def _popular_transfer_tokens(self, wallet: str, chain: str, limit: Optional[int]) -> List[str]:
        if not hasattr(self.cache_transfers, "popular_tokens"):
            return []
        cap = limit if limit is not None and limit > 0 else self._popular_watch_cap
        try:
            return self.cache_transfers.popular_tokens(
                wallet,
                chain,
                limit=cap,
                within_minutes=self._popular_recent_minutes,
            )
        except Exception:
            return []

    def _chain_plan(
        self,
        wallet: str,
        chain: str,
        *,
        lookback_blocks: int,
        watch_tokens: Optional[Mapping[str, Sequence[str]]],
        max_tokens: Optional[int],
        include_native: bool,
    ) -> List[_TokenPlan]:
        state = self.cache_balances.get_state(wallet, chain)
        tokens = state.get("tokens", {})
        max_block = 0
        for meta in tokens.values():
            try:
                max_block = max(max_block, int(meta.get("asof_block") or 0))
            except Exception:
                continue
        since_block = max(0, max_block - int(lookback_blocks))
        planned: List[_TokenPlan] = []
        stale = self._stale_tokens(tokens)
        for addr in stale:
            planned.append(_TokenPlan(chain=chain, token=addr, reason="stale", meta=tokens.get(addr, {})))
        touched = self._touched_since(wallet, chain, since_block)
        for addr in touched:
            meta = tokens.get(addr, {})
            planned.append(_TokenPlan(chain=chain, token=addr, reason="touched", meta=meta))
        chain_watch = list((watch_tokens or {}).get(chain, [])) + self.env_watch.get(chain, [])
        chain_watch += self._core_watch_for(chain)
        chain_watch += self._popular_transfer_tokens(wallet, chain, max_tokens)
        for addr in chain_watch:
            planned.append(_TokenPlan(chain=chain, token=addr, reason="watch", meta=tokens.get(addr.lower(), {})))
        if include_native:
            planned.append(
                _TokenPlan(
                    chain=chain,
                    token=self.native_token,
                    reason="native",
                    meta=tokens.get(self.native_token, {}),
                )
            )
        if not planned:
            return []
        dedup: Dict[str, _TokenPlan] = {}
        for item in planned:
            key = item.token.lower()
            if key in dedup:
                continue
            dedup[key] = item
        ordered = list(dedup.values())
        if max_tokens is not None and len(ordered) > max_tokens:
            ordered = ordered[:max_tokens]
        return ordered

    # ------------------------------------------------------------------ fetchers
    def _is_native(self, token: str) -> bool:
        val = (token or "").lower()
        if val in ("eth", "native"):
            return True
        return val == self.native_token

    def _native_symbol(self, chain: str) -> str:
        return DEFAULT_NATIVE_SYMBOL.get(chain, chain.upper())

    def _fetch_native(self, chain: str, wallet_addr: str, prev: Mapping[str, Any]) -> Dict[str, Any]:
        w3 = self.bridge._w3(chain)
        raw = int(w3.eth.get_balance(wallet_addr))
        symbol = prev.get("symbol") or self._native_symbol(chain)
        return {
            "balance_hex": hex(raw),
            "raw": hex(raw),
            "asof_block": int(w3.eth.block_number),
            "quantity": _to_decimal_string(raw, 18),
            "decimals": 18,
            "symbol": symbol,
            "name": prev.get("name") or symbol,
        }

    def _fetch_erc20(self, chain: str, token: str, wallet_addr: str, prev: Mapping[str, Any]) -> Dict[str, Any]:
        bal = int(self.bridge.erc20_balance_of(chain, token, wallet_addr))
        decimals = int(self.bridge.erc20_decimals(chain, token))
        qty = _to_decimal_string(bal, decimals)
        symbol = self.bridge.erc20_symbol(chain, token) or prev.get("symbol") or (token[:6] + "…")
        name = self.bridge.erc20_name(chain, token) or prev.get("name") or symbol
        block_num = int(self.bridge._w3(chain).eth.block_number)
        return {
            "balance_hex": hex(bal),
            "raw": hex(bal),
            "asof_block": block_num,
            "quantity": qty,
            "decimals": decimals,
            "symbol": symbol,
            "name": name,
        }

    def _fetch(self, plan: _TokenPlan, wallet_addr: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        token = plan.token
        chain = plan.chain
        prev = plan.meta or {}
        if self._is_native(token):
            data = self._fetch_native(chain, wallet_addr, prev)
        else:
            data = self._fetch_erc20(chain, token, wallet_addr, prev)
        usd = prev.get("usd_amount") or prev.get("usd") or "0"
        data.update({
            "usd_amount": usd,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        return (chain, {token: data})

    # ------------------------------------------------------------------ public
    def refresh(
        self,
        *,
        chains: Optional[Iterable[str]] = None,
        lookback_blocks: int = 16,
        watch_tokens: Optional[Mapping[str, Sequence[str]]] = None,
        include_native: bool = True,
        max_tokens_per_chain: Optional[int] = None,
    ) -> Dict[str, Any]:
        wallet_addr = self.bridge.get_address()
        chain_list = self._default_chains(chains)
        if max_tokens_per_chain is None:
            env_cap = os.getenv("BALANCE_FAST_MAX_TOKENS")
            if env_cap:
                try:
                    max_tokens_per_chain = int(env_cap)
                except Exception:
                    max_tokens_per_chain = None
        plan: List[_TokenPlan] = []
        for chain in chain_list:
            plan.extend(
                self._chain_plan(
                    wallet_addr,
                    chain,
                    lookback_blocks=lookback_blocks,
                    watch_tokens=watch_tokens,
                    max_tokens=max_tokens_per_chain,
                    include_native=include_native,
                )
            )
        if not plan:
            return {"wallet": wallet_addr, "updated": 0, "chains": []}

        updates: Dict[str, Dict[str, Dict[str, Any]]] = {}
        errors: List[str] = []
        workers = min(max(len(plan), 1), self.max_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._fetch, item, wallet_addr): item for item in plan}
            for fut in as_completed(futures):
                planned = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    errors.append(f"{planned.chain}:{planned.token} -> {exc}")
                    continue
                if not result:
                    continue
                chain, mapping = result
                bucket = updates.setdefault(chain, {})
                bucket.update(mapping)
        for chain, mapping in updates.items():
            self.cache_balances.upsert_many(wallet_addr, chain, mapping)
        summary = {
            "wallet": wallet_addr,
            "updated": sum(len(m) for m in updates.values()),
            "chains": sorted(updates.keys()),
            "errors": errors,
        }
        wallet_log("wallet.fast_refresh", **summary)
        return summary
