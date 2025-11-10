from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from cache import CacheBalances, CacheTransfers
from filter_scams import FilterScamTokens
from services.wallet_logger import wallet_log
from services.wallet_optimizer import RealtimeBalanceRefresher
from services.token_safety import TokenSafetyRegistry
from services.wallet_watch import (
    build_core_watch_tokens,
    core_watch_limit,
    merge_watch_maps,
    select_cached_watch_tokens,
)
from router_wallet import CHAINS, UltraSwapBridge

STATE_DIR = Path("storage/wallet_state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_STATE = {
    "wallet": None,
    "updated_at": None,
    "totals": {"usd": 0.0},
    "balances": [],
    "transfers": {},
    "nfts": [],
    "filtered_tokens": [],
}


def _bool_env(name: str, default: str = "1") -> bool:
    val = os.getenv(name, default)
    if val is None:
        val = default
    return str(val).strip().lower() not in {"0", "false", "no", "off"}


def _dynamic_watch_limit(default: int = 8) -> Optional[int]:
    raw = os.getenv("WALLET_DYNAMIC_WATCH_LIMIT")
    if raw is None or raw.strip() == "":
        return default
    lowered = raw.strip().lower()
    if lowered in {"0", "false", "no", "none"}:
        return None
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else None


def _dynamic_watch_min_usd(default: float = 5.0) -> float:
    raw = os.getenv("WALLET_DYNAMIC_WATCH_MIN_USD")
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def _dynamic_watch_include_illiquid(default: bool = False) -> bool:
    raw = os.getenv("WALLET_DYNAMIC_WATCH_INCLUDE_ILLIQUID")
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _ts_to_epoch(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return 0.0
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        from datetime import datetime

        return datetime.fromisoformat(s).timestamp()
    except Exception:
        try:
            return float(s)
        except Exception:
            return 0.0


def _build_cached_watch_tokens(
    wallet_addr: str,
    chain_list: Iterable[str],
    cache_balances: CacheBalances,
) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    limit = _dynamic_watch_limit()
    min_usd = _dynamic_watch_min_usd()
    include_illiquid = _dynamic_watch_include_illiquid()
    for chain in chain_list:
        chain_l = str(chain).strip().lower()
        if not chain_l:
            continue
        try:
            state = cache_balances.get_state(wallet_addr, chain_l)
        except Exception:
            continue
        tokens = state.get("tokens", {})
        picks = select_cached_watch_tokens(
            tokens,
            limit=limit,
            min_usd=min_usd,
            include_illiquid=include_illiquid,
        )
        if picks:
            result[chain_l] = picks
    return result


def _maybe_fast_balance_refresh(
    bridge: UltraSwapBridge | None,
    cb: CacheBalances,
    ct: CacheTransfers,
    chain_list: List[str],
    wallet_addr: Optional[str],
) -> None:
    if not bridge or not _bool_env("WALLET_FAST_BALANCES", "1"):
        return
    refresher = RealtimeBalanceRefresher(
        bridge=bridge,
        cache_balances=cb,
        cache_transfers=ct,
    )
    core_watch = build_core_watch_tokens(chain_list, limit=core_watch_limit())
    cached_watch: Dict[str, List[str]] = {}
    if wallet_addr:
        cached_watch = _build_cached_watch_tokens(wallet_addr, chain_list, cb)
    merged_watch = merge_watch_maps(core_watch, cached_watch)
    try:
        summary = refresher.refresh(chains=chain_list, watch_tokens=merged_watch, include_native=True)
        wallet_log("wallet_state.fast_balances", **summary)
    except Exception as exc:
        wallet_log("wallet_state.fast_balances_error", error=str(exc))


def _maybe_fast_transfer_refresh(
    bridge: UltraSwapBridge | None,
    ct: CacheTransfers,
    chain_list: List[str],
    wallet_addr: Optional[str],
) -> None:
    if not bridge or not _bool_env("WALLET_FAST_TRANSFERS", "1"):
        return
    if not wallet_addr:
        return
    try:
        pages = int(os.getenv("WALLET_FAST_TRANSFER_PAGES", "2") or "2")
    except ValueError:
        pages = 2
    pages = max(1, pages)
    try:
        max_age = float(os.getenv("WALLET_FAST_TRANSFER_MAX_AGE", "180") or "180")
    except ValueError:
        max_age = 180.0
    stale_chains = chain_list
    if max_age > 0:
        now = time.time()
        stale: List[str] = []
        for chain in chain_list:
            try:
                state = ct.get_state(wallet_addr, chain)
            except Exception:
                stale.append(chain)
                continue
            age = now - _ts_to_epoch(state.get("last_ts"))
            if age >= max_age:
                stale.append(chain)
        stale_chains = stale or []
    if not stale_chains:
        wallet_log("wallet_state.fast_transfers_skip", reason="fresh", chains=chain_list)
        return
    try:
        ct.rebuild_incremental(bridge, stale_chains, max_pages_per_dir=pages)
    except Exception as exc:
        wallet_log("wallet_state.fast_transfers_error", error=str(exc))


def _write_json(name: str, payload: Any) -> None:
    path = STATE_DIR / f"{name}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def _read_json(name: str) -> Any:
    path = STATE_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _resolve_wallet_address(bridge: UltraSwapBridge | None = None, wallet: Optional[str] = None) -> Optional[str]:
    if wallet:
        return wallet
    if bridge and hasattr(bridge, "get_address"):
        try:
            return bridge.get_address()
        except Exception:
            return None
    # Fallback: best effort by instantiating a bridge to derive wallet
    try:
        inst = UltraSwapBridge()
        addr = inst.get_address()
        return addr
    except Exception:
        return None


def _filter_scams(
    rows: List[Dict[str, Any]],
    *,
    registry: TokenSafetyRegistry | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not rows:
        return rows, []
    annotated: List[tuple[str, str]] = []
    for row in rows:
        token_addr = (row.get("token") or "").strip().lower()
        chain = (row.get("chain") or "").strip().lower()
        if token_addr.startswith("0x") and chain:
            annotated.append((chain, token_addr))
    if not annotated:
        return rows, []

    flagged_records: Dict[tuple[str, str], Dict[str, Any]] = {}
    used_registry = False
    if registry:
        try:
            survivors = registry.filter_pairs(annotated)
            survivor_set = {(ch, tok.lower()) for ch, tok in survivors}
            for entry in annotated:
                if entry not in survivor_set:
                    chain, token = entry
                    verdict = registry.verdict_for(chain, token)
                    flagged_records[(chain, token)] = {
                        "chain": chain,
                        "token": token,
                        "severity": verdict.get("severity", "unknown"),
                        "reasons": verdict.get("reasons", []),
                        "source": "registry",
                    }
            used_registry = True
            wallet_log(
                "wallet_state.scam_filter_cache",
                annotated=[f"{ch}:{tok}" for ch, tok in annotated],
                flagged=[f"{meta['chain']}:{meta['token']}" for meta in flagged_records.values()],
            )
        except Exception as exc:  # pragma: no cover - registry failures fall back
            wallet_log("wallet_state.scam_filter_registry_error", error=str(exc))

    if not used_registry:
        try:
            filt = FilterScamTokens()
            res = filt.filter([f"{ch}:{tok}" for ch, tok in annotated])
            for chain, token in annotated:
                reasons = res.reasons.get(token.lower(), [])
                if reasons:
                    flagged_records[(chain, token)] = {
                        "chain": chain,
                        "token": token,
                        "severity": TokenSafetyRegistry.classify_reasons(reasons),
                        "reasons": reasons,
                        "source": "api",
                    }
            wallet_log(
                "wallet_state.scam_filter",
                annotated=annotated,
                flagged=[f"{meta['chain']}:{meta['token']}" for meta in flagged_records.values()],
            )
        except Exception:
            return rows, []

    if not flagged_records:
        return rows, []

    survivors: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    for row in rows:
        chain = (row.get("chain") or "").strip().lower()
        token_addr = (row.get("token") or "").strip().lower()
        rec = flagged_records.get((chain, token_addr))
        if chain and token_addr.startswith("0x") and rec:
            wallet_log(
                "wallet_state.filtered_token",
                chain=row.get("chain"),
                token=row.get("token"),
                reason="scam_filter",
                severity=rec.get("severity"),
            )
            removed.append(rec)
            continue
        survivors.append(row)
    return survivors, removed


def _format_usd(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _collect_balances(
    wallet: str,
    *,
    cache_balances: CacheBalances | None = None,
    registry: TokenSafetyRegistry | None = None,
) -> Dict[str, Any]:
    cb = cache_balances or CacheBalances()
    rows = cb.load(wallet=wallet)
    wallet_log("wallet_state.raw_balances", wallet=wallet, rows=rows)
    rows, filtered_meta = _filter_scams(rows, registry=registry)
    wallet_log(
        "wallet_state.filtered_balances",
        wallet=wallet,
        rows=rows,
        filtered=len(filtered_meta),
    )
    totals_usd = sum(_format_usd(row.get("usd")) for row in rows)
    for row in rows:
        row["usd"] = _format_usd(row.get("usd"))
    rows.sort(key=lambda r: r.get("usd", 0), reverse=True)
    return {
        "totals": {"usd": round(totals_usd, 2)},
        "balances": rows,
        "filtered": filtered_meta,
    }


def _collect_transfers(
    wallet: str,
    chains: Optional[Iterable[str]] = None,
    limit: int = 25,
    *,
    cache_transfers: CacheTransfers | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    ct = cache_transfers or CacheTransfers()
    chains_iter = chains or CHAINS.keys()
    output: Dict[str, List[Dict[str, Any]]] = {}
    wallet_lower = wallet.lower()
    for chain in chains_iter:
        try:
            state = ct.get_state(wallet, chain)
        except Exception:
            continue
        items = state.get("items", [])
        if not items:
            continue
        formatted: List[Dict[str, Any]] = []
        for item in items[-limit:]:
            direction = "out" if (item.get("from_addr") == wallet_lower) else "in"
            formatted.append(
                {
                    "hash": item.get("hash"),
                    "direction": direction,
                    "token": item.get("token"),
                    "value": item.get("value"),
                    "block": item.get("block"),
                    "ts": item.get("ts"),
                }
            )
        output[chain] = list(reversed(formatted))
    return output


def _ipfs_to_https(url: str) -> str:
    if url.startswith("ipfs://"):
        return url.replace("ipfs://", "https://ipfs.io/ipfs/")
    if url.startswith("ar://"):
        return url.replace("ar://", "https://arweave.net/")
    return url


def _collect_nfts(wallet: str, chains: Optional[Iterable[str]] = None, max_items: int = 60) -> List[Dict[str, Any]]:
    api_key = (os.getenv("ALCHEMY_API_KEY") or "").strip()
    if not api_key:
        return []
    slug_map = {
        "ethereum": "eth-mainnet",
        "polygon": "polygon-mainnet",
        "base": "base-mainnet",
        "arbitrum": "arb-mainnet",
        "optimism": "opt-mainnet",
    }
    chains_iter = chains or slug_map.keys()
    session = requests.Session()
    session.headers.update({"accept": "application/json"})
    results: List[Dict[str, Any]] = []
    for chain in chains_iter:
        slug = slug_map.get(chain)
        if not slug:
            continue
        url = f"https://{slug}.g.alchemy.com/v2/{api_key}/getNFTs"
        params = {"owner": wallet, "withMetadata": "true", "pageSize": 50}
        fetched = 0
        while fetched < max_items:
            try:
                resp = session.get(url, params=params, timeout=12)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                break
            for nft in data.get("ownedNfts", []):
                media = nft.get("media") or []
                image = None
                if media:
                    image = media[0].get("gateway") or media[0].get("thumbnail") or media[0].get("raw")
                if not image:
                    meta = nft.get("metadata") or {}
                    image = meta.get("image") or meta.get("image_url")
                if image:
                    image = _ipfs_to_https(str(image))
                results.append(
                    {
                        "chain": chain,
                        "contract": nft.get("contract", {}).get("address"),
                        "token_id": nft.get("tokenId") or nft.get("id", {}).get("tokenId"),
                        "title": nft.get("title") or (nft.get("contract") or {}).get("name"),
                        "description": (nft.get("description") or ""),
                        "image": image,
                    }
                )
                fetched += 1
                if fetched >= max_items:
                    break
            page_key = data.get("pageKey")
            if not page_key:
                break
            params["pageKey"] = page_key
    return results


def capture_wallet_state(
    *,
    bridge: UltraSwapBridge | None = None,
    wallet: Optional[str] = None,
    chains: Optional[Iterable[str]] = None,
    cache_balances: CacheBalances | None = None,
    cache_transfers: CacheTransfers | None = None,
    registry: TokenSafetyRegistry | None = None,
) -> Dict[str, Any]:
    wallet_addr = _resolve_wallet_address(bridge=bridge, wallet=wallet)
    if not wallet_addr:
        return DEFAULT_STATE
    cb = cache_balances or CacheBalances()
    ct = cache_transfers or CacheTransfers()
    chain_list = [str(ch).lower() for ch in (chains or CHAINS.keys())]
    _maybe_fast_transfer_refresh(bridge, ct, chain_list, wallet_addr)
    _maybe_fast_balance_refresh(bridge, cb, ct, chain_list, wallet_addr)
    registry = registry or getattr(bridge, "_token_safety", None)
    balances_payload = _collect_balances(wallet_addr, cache_balances=cb, registry=registry)
    transfers_payload = _collect_transfers(wallet_addr, chains=chain_list, cache_transfers=ct)
    nfts_payload = _collect_nfts(wallet_addr, chains=chains)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state = {
        "wallet": wallet_addr,
        "updated_at": timestamp,
        "totals": balances_payload["totals"],
        "balances": balances_payload["balances"],
        "transfers": transfers_payload,
        "nfts": nfts_payload,
        "filtered_tokens": balances_payload.get("filtered", []),
    }
    _write_json("balances", balances_payload)
    _write_json("transfers", transfers_payload)
    _write_json("nfts", nfts_payload)
    _write_json("state", state)
    return state


def load_wallet_state() -> Dict[str, Any]:
    snapshot = _read_json("state")
    if snapshot:
        return snapshot
    balances = _read_json("balances") or {}
    transfers = _read_json("transfers") or {}
    nfts = _read_json("nfts") or []
    return {
        "wallet": None,
        "updated_at": None,
        "totals": balances.get("totals", {"usd": 0}),
        "balances": balances.get("balances", []),
        "transfers": transfers,
        "nfts": nfts,
        "filtered_tokens": balances.get("filtered", []),
    }
