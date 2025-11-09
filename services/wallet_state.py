from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from cache import CacheBalances, CacheTransfers
from filter_scams import FilterScamTokens
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
}


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


def _filter_scams(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    annotated = []
    for row in rows:
        token_addr = (row.get("token") or "").lower()
        chain = (row.get("chain") or "").lower()
        if token_addr:
            annotated.append(f"{chain}:{token_addr}")
    if not annotated:
        return rows
    try:
        filt = FilterScamTokens()
        res = filt.filter(annotated)
        flagged = {item.lower() for item in (res.flagged or [])}
        if not flagged:
            return rows
        survivors = []
        for row in rows:
            key = f"{row.get('chain','').lower()}:{row.get('token','').lower()}"
            if key and key in flagged:
                continue
            survivors.append(row)
        return survivors
    except Exception:
        return rows


def _format_usd(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _collect_balances(wallet: str) -> Dict[str, Any]:
    cb = CacheBalances()
    rows = cb.load(wallet=wallet)
    rows = _filter_scams(rows)
    totals_usd = sum(_format_usd(row.get("usd")) for row in rows)
    for row in rows:
        row["usd"] = _format_usd(row.get("usd"))
    rows.sort(key=lambda r: r.get("usd", 0), reverse=True)
    return {
        "totals": {"usd": round(totals_usd, 2)},
        "balances": rows,
    }


def _collect_transfers(wallet: str, chains: Optional[Iterable[str]] = None, limit: int = 25) -> Dict[str, List[Dict[str, Any]]]:
    ct = CacheTransfers()
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
) -> Dict[str, Any]:
    wallet_addr = _resolve_wallet_address(bridge=bridge, wallet=wallet)
    if not wallet_addr:
        return DEFAULT_STATE
    balances_payload = _collect_balances(wallet_addr)
    transfers_payload = _collect_transfers(wallet_addr, chains=chains)
    nfts_payload = _collect_nfts(wallet_addr, chains=chains)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state = {
        "wallet": wallet_addr,
        "updated_at": timestamp,
        "totals": balances_payload["totals"],
        "balances": balances_payload["balances"],
        "transfers": transfers_payload,
        "nfts": nfts_payload,
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
    }
