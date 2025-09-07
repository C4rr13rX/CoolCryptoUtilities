# cache.py
# MIT License
# © 2025 Your Name

from __future__ import annotations

import os
import json
import time
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set

from pathlib import Path

# --------------------------- basics ---------------------------

def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _hex_to_int(x: Any) -> int:
    if x is None: return 0
    if isinstance(x, int): return x
    s = str(x).strip().lower()
    if s.startswith("0x"):
        try: return int(s, 16)
        except Exception: return 0
    try: return int(s)
    except Exception: return 0

def _lower(a: Optional[str]) -> str:
    return (a or "").lower()

# Default cache root (override with env if you like)
CACHE_ROOT = Path(os.getenv("PORTFOLIO_CACHE_DIR", "~/.cache/mchain")).expanduser()

# Chain names you use elsewhere
CHAIN_IDS: Dict[str, int] = {
    "ethereum": 1,
    "base": 8453,
    "arbitrum": 42161,
    "optimism": 10,
    "polygon": 137,
}

# --- Price cache (per chain+token) -------------------------------------
import json, time, os
from pathlib import Path
from typing import Dict, List, Optional, Any

class CachePrices:
    """
    Simple JSON cache for USD prices keyed by (chain, token).
    - Values: {"usd": "<str>", "source": "alchemy|0x|custom", "ts": <epoch float>}
    - No wallet dimension (prices are global per chain+token).
    - TTL handled by the caller; helper returns only fresh entries.
    """
    def __init__(self, data_dir: str = ".cache", filename: str = "prices.json", cache_prices: Optional[CachePrices] = None, price_ttl_sec: Optional[int] = None, verbose: Optional[bool] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / filename
        self.cp: Optional[CachePrices] = cache_prices
        self.price_ttl_sec = int(price_ttl_sec if price_ttl_sec is not None
             else os.getenv("PRICE_TTL_SEC", "300"))
        self.cp: Optional[CachePrices] = cache_prices
        self.price_ttl_sec = int(price_ttl_sec if price_ttl_sec is not None
                                 else os.getenv("PRICE_TTL_SEC", "300"))

    # ---------- file IO ----------
    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "networks": {}}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    return {"version": 1, "networks": {}}
                data.setdefault("version", 1)
                data.setdefault("networks", {})
                return data
        except Exception:
            return {"version": 1, "networks": {}}

    def _save(self, data: Dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(self.path)

    # ---------- API ----------
    def get_price(self, chain: str, token: str, max_age_sec: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return a single fresh entry or None."""
        data = self._load()
        d = data["networks"].get(chain, {})
        ent = d.get(token.lower())
        if not ent:
            return None
        if max_age_sec is not None:
            ts = float(ent.get("ts", 0))
            if (time.time() - ts) > max_age_sec:
                return None
        return ent

    def get_many(self, chain: str, tokens: List[str], max_age_sec: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """Return only fresh entries for the requested tokens."""
        data = self._load()
        d = data["networks"].get(chain, {})
        out: Dict[str, Dict[str, Any]] = {}
        now = time.time()
        for t in tokens:
            ent = d.get(t.lower())
            if not ent:
                continue
            if max_age_sec is not None:
                ts = float(ent.get("ts", 0))
                if (now - ts) > max_age_sec:
                    continue
            out[t.lower()] = ent
        return out

    def upsert_many(self, chain: str, mapping: Dict[str, Dict[str, Any]]) -> None:
        """
        mapping: { token: {"usd": "<str or number>", "source": "alchemy|0x|custom"} }
        Adds ts automatically.
        """
        data = self._load()
        data["networks"].setdefault(chain, {})
        now = time.time()
        for addr, ent in mapping.items():
            if ent is None:
                continue
            usd = ent.get("usd")
            try:
                usd_str = str(usd)
            except Exception:
                usd_str = "0"
            source = (ent.get("source") or "custom").lower()
            data["networks"][chain][addr.lower()] = {"usd": usd_str, "source": source, "ts": now}
        self._save(data)

# ------------------------ CacheTransfers ------------------------

class CacheTransfers:
    """
    Wallet+chain ERC-20 transfers (full history, merged over time).

    File layout (JSON):
      cache/transfers/{chain}/{wallet}.json
      {
        "wallet": "0x...",
        "chain": "base",
        "last_block": 12345678, # highest block merged
        "last_ts": "2025-01-01T00:00:00Z",
        "items": [
          {
            "id": "uniqueId|hash:logIndex", # dedupe key
            "hash": "0x..",
            "logIndex": 12,
            "block": 123,
            "ts": "2025-01-01T..Z",
            "from": "0x..",
            "to": "0x..",
            "token": "0x<erc20>",
            "value": "123.45" | "0x..." | int (as returned)
          }, ...
        ]
      }
    """

    def __init__(self, root: Path = CACHE_ROOT) -> None:
        self.root = root

    # ---------- filenames ----------
    def _file(self, wallet: str, chain: str) -> Path:
        w = _lower(wallet)
        ch = _lower(chain)
        return self.root / "transfers" / ch / f"{w}.json"

    # ---------- public helpers ----------
    def get_state(self, wallet: str, chain: str) -> Dict[str, Any]:
        path = self._file(wallet, chain)
        state = _read_json(path)
        if not state:
            state = {"wallet": _lower(wallet), "chain": _lower(chain),
                     "last_block": 0, "last_ts": None, "items": []}
        # normalize & guard
        state["last_block"] = int(state.get("last_block") or 0)
        state["items"] = list(state.get("items") or [])
        return state

    def next_from_block(self, wallet: str, chain: str) -> int:
        """Start your API scan from this block (inclusive)."""
        st = self.get_state(wallet, chain)
        last = int(st.get("last_block") or 0)
        return max(last + 1, 0)

    def merge_new(self, wallet: str, chain: str, new_items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge a list of *raw* transfer dicts (Alchemy 'getAssetTransfers' style or similar).
        Returns the updated state dict.
        """
        st = self.get_state(wallet, chain)
        seen: Dict[str, Dict[str, Any]] = {}
        for it in st["items"]:
            k = str(it.get("id") or "")
            if k:
                seen[k] = it

        def _mk_record(t: Dict[str, Any]) -> Dict[str, Any]:
            # Accept multiple shapes; be permissive.
            h = t.get("hash") or t.get("transactionHash") or ""
            li = t.get("logIndex")
            if li is None:
                # Alchemy sometimes returns uniqueId
                uid = t.get("uniqueId")
                if uid:
                    li = uid.split("-")[-1] if isinstance(uid, str) else uid
            try:
                li = int(li)
            except Exception:
                li = None

            uid = t.get("uniqueId")
            if isinstance(uid, str) and uid:
                dedupe_id = uid
            elif h and li is not None:
                dedupe_id = f"{h}:{li}"
            else:
                # last resort: combine a few fields
                rc = (((t.get("rawContract") or {}).get("address")) or
                      t.get("erc20Contract") or t.get("contract") or "")
                ts = ((t.get("metadata") or {}).get("blockTimestamp") or
                      t.get("timestamp") or "")
                dedupe_id = f"{h}:{rc}:{ts}"

            blk = (t.get("blockNum") or t.get("blockNumber") or
                   (t.get("metadata") or {}).get("block") or 0)
            block_i = _hex_to_int(blk)

            ts = ((t.get("metadata") or {}).get("blockTimestamp") or
                  t.get("timestamp") or None)

            token = (((t.get("rawContract") or {}).get("address")) or
                     t.get("erc20Contract") or t.get("contract") or "")

            return {
                "id": dedupe_id,
                "hash": h,
                "logIndex": li,
                "block": block_i,
                "ts": ts,
                "from": _lower(t.get("from") or t.get("fromAddress") or ""),
                "to": _lower(t.get("to") or t.get("toAddress") or ""),
                "token": _lower(token),
                "value": t.get("value") if "value" in t else t.get("amount"),
            }

        max_block = int(st.get("last_block") or 0)
        for raw in (new_items or []):
            rec = _mk_record(raw)
            k = rec["id"]
            if not k:
                continue
            seen[k] = rec
            if rec["block"] and rec["block"] > max_block:
                max_block = rec["block"]

        # Rebuild list sorted by block then logIndex if present
        items = list(seen.values())
        items.sort(key=lambda x: (int(x.get("block") or 0), int(x.get("logIndex") or 0)))

        st["items"] = items
        st["last_block"] = max_block
        st["last_ts"] = _now_ts()
        _atomic_write(self._file(wallet, chain), st)
        return st

    def get_all(self, wallet: str, chain: str) -> List[Dict[str, Any]]:
        """Return ALL cached transfers for wallet+chain."""
        return list(self.get_state(wallet, chain).get("items") or [])

    def touched_tokens_since(self, wallet: str, chain: str, since_block: int) -> Set[str]:
        """
        Return the set of ERC-20 contract addresses touched after `since_block`.
        """
        out: Set[str] = set()
        st = self.get_state(wallet, chain)
        for it in st.get("items") or []:
            if int(it.get("block") or 0) > int(since_block or 0):
                tok = _lower(it.get("token"))
                if tok:
                    out.add(tok)
        return out

# ------------------------ CacheBalances ------------------------

class CacheBalances:
    """
    Per wallet+chain+token balance cache.

    File layout:
      cache/balances/{chain}/{wallet}.json
      {
        "wallet": "0x...",
        "chain": "base",
        "updated_at": "2025-..Z",
        "tokens": {
          "0xabc...": {
            "raw": "0x0123...", # raw hex string from RPC (or int/string)
            "at_block": 12345678, # block height when captured (optional)
            "updated_at": "2025-..Z"
          },
          ...
        }
      }
    """

    def __init__(self, root: Path = CACHE_ROOT) -> None:
        self.root = root

    def _file(self, wallet: str, chain: str) -> Path:
        return self.root / "balances" / _lower(chain) / f"{_lower(wallet)}.json"

    def get_state(self, wallet: str, chain: str) -> Dict[str, Any]:
        path = self._file(wallet, chain)
        st = _read_json(path)
        if not st:
            st = {"wallet": _lower(wallet), "chain": _lower(chain),
                  "updated_at": None, "tokens": {}}
        st["tokens"] = dict(st.get("tokens") or {})
        return st

    def get_hits_and_misses(
        self,
        wallet: str,
        chain: str,
        tokens: Sequence[str],
        changed_tokens: Optional[Iterable[str]] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Decide which balances we can serve from cache.

        - If `changed_tokens` is provided, only those addresses are considered stale.
        - If it is None, anything present in cache is a hit.
        """
        changed: Set[str] = set(_lower(a) for a in (changed_tokens or []))
        st = self.get_state(wallet, chain)
        tok_map = st.get("tokens", {})
        hits: Dict[str, Any] = {}
        misses: List[str] = []
        for a in tokens:
            key = _lower(a)
            entry = tok_map.get(key)
            if entry and (not changed or key not in changed):
                hits[a] = entry.get("raw")
            else:
                misses.append(a)
        return hits, misses

    def update(
        self,
        wallet: str,
        chain: str,
        updates: Dict[str, Any],
        at_block: Optional[int] = None,
    ) -> None:
        """
        Merge freshly-fetched balances into cache.
          updates: {token_addr -> raw_balance_value}
        """
        st = self.get_state(wallet, chain)
        tks = st["tokens"]
        ts = _now_ts()
        for addr, raw in (updates or {}).items():
            key = _lower(addr)
            tks.setdefault(key, {})
            tks[key]["raw"] = raw
            tks[key]["at_block"] = int(at_block) if at_block is not None else tks[key].get("at_block")
            tks[key]["updated_at"] = ts
        st["updated_at"] = ts
        _atomic_write(self._file(wallet, chain), st)

    def invalidate(
        self,
        wallet: str,
        chain: str,
        token_addrs: Iterable[str],
    ) -> None:
        """Drop specific tokens from cache (e.g., contract upgrade)."""
        st = self.get_state(wallet, chain)
        tks = st["tokens"]
        for a in token_addrs:
            tks.pop(_lower(a), None)
        st["updated_at"] = _now_ts()
        _atomic_write(self._file(wallet, chain), st)

# ------------------------ usage notes ------------------------

USAGE_NOTES = r"""
USAGE (main.py / UltraSwapBridge) — CacheTransfers
--------------------------------------------------
from cache import CacheTransfers

ct = CacheTransfers()

# Before calling Alchemy getAssetTransfers for a wallet+chain:
from_block = ct.next_from_block(wallet, chain) # int block number (inclusive)

# Example (pseudo):
# params = { "fromBlock": hex(from_block), "toBlock": "latest", ... }
# loop with pageKey fetching NEW transfers only

# After you fetch the NEW page(s):
ct.merge_new(wallet, chain, new_transfers_list)

# If you want the whole history without hitting the API again:
all_transfers = ct.get_all(wallet, chain)

# To know which token contracts changed since a block (so you can refresh only those balances):
changed = ct.touched_tokens_since(wallet, chain, since_block=previous_last_block)

NOTE on "entire history":
- The very first run should set from_block = 0 (ct.next_from_block returns 1 when empty; you can
  override and start at 0 if desired). After that, always use ct.next_from_block(...).
- If you ever want to force a full rescan: delete the cache file cache/transfers/{chain}/{wallet}.json.


USAGE (balances.py / MultiChainTokenPortfolio) — CacheBalances
--------------------------------------------------------------
from cache import CacheBalances, CacheTransfers

cb = CacheBalances()
ct = CacheTransfers() # optional, only if you want to refresh balances that actually changed

# Suppose you already know the token list ["0x..", ...] for wallet+chain
# Option A: naive — use whatever is cached; fetch only missing
hits, misses = cb.get_hits_and_misses(wallet, chain, token_list)

# Option B: smarter — refresh only tokens that had NEW transfers since last balance capture
# (Use ct.touched_tokens_since with the last_block value you remembered for this run; if you
# don't track it, you can query ct.get_state(wallet, chain)["last_block"] as the baseline.)
since = ct.get_state(wallet, chain)["last_block"]
changed = ct.touched_tokens_since(wallet, chain, since_block=since)
hits, misses = cb.get_hits_and_misses(wallet, chain, token_list, changed_tokens=changed)

# Now call your existing RPC path ONLY for 'misses', e.g. via alchemy_getTokenBalances/eth_call,
# producing a dict like: fetched = {addr: "0xTOKENBALANCEHEX", ...}
# Optionally include a block number (e.g., latest block from your RPC) when updating:
cb.update(wallet, chain, updates=fetched, at_block=latest_block_number)

# Finally, combine:
balances_raw = {**hits, **fetched}"""
