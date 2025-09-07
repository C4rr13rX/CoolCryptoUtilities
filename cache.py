# cache.py
# MIT License
# © 2025 Your Name

from __future__ import annotations

import os
import json
import time
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

class CachePrices:
    """
    Simple JSON cache for USD prices keyed by (chain, token).
    File shape:
    {
      "version": 1,
      "networks": {
        "<chain>": {
          "<token_addr_lower>": {"usd": "<str>", "source": "alchemy|0x|custom", "ts": <epoch float>}
        }
      }
    }
    """
    def __init__(
        self,
        data_dir: str = ".cache",
        filename: str = "prices.json",
        cache_prices: Optional["CachePrices"] = None,   # kept for backward compat; unused
        price_ttl_sec: Optional[int] = None,
        verbose: Optional[bool] = None,                 # kept for compat; unused
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / filename
        # default TTL 300s unless overridden by ctor or env
        env_ttl = os.getenv("PRICE_TTL_SEC")
        self.price_ttl_sec = int(price_ttl_sec if price_ttl_sec is not None else (env_ttl if env_ttl is not None else "300"))

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
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    # ---------- API ----------
    def get_price(
        self,
        chain: str,
        token: str,
        max_age_sec: Optional[int] = None,
        *,
        ttl_sec: Optional[int] = None,
    ) -> Optional[str]:
        """
        Return price string for (chain, token) if fresh, else None.
        Accepts either max_age_sec or ttl_sec (alias used by some callers).
        """
        data = self._load()
        d = data["networks"].get(chain, {})
        ent = d.get(token.lower())
        if not ent:
            return None
        age_limit = max_age_sec if max_age_sec is not None else ttl_sec
        if age_limit is not None:
            ts = float(ent.get("ts", 0))
            if (time.time() - ts) > float(age_limit):
                return None
        usd = ent.get("usd")
        return str(usd) if usd is not None else None

    def get_many(self, chain: str, tokens: List[str], max_age_sec: Optional[int] = None) -> Dict[str, str]:
        """Return only fresh USD prices (as strings) for the requested tokens."""
        data = self._load()
        d = data["networks"].get(chain, {})
        out: Dict[str, str] = {}
        now = time.time()
        for t in tokens:
            ent = d.get(t.lower())
            if not ent:
                continue
            if max_age_sec is not None:
                ts = float(ent.get("ts", 0))
                if (now - ts) > max_age_sec:
                    continue
            usd = ent.get("usd")
            if usd is not None:
                out[t.lower()] = str(usd)
        return out

    def get(self, chain: str, token: str, max_age_sec: Optional[int] = None) -> Optional[str]:
        """Alias for get_price."""
        return self.get_price(chain, token, max_age_sec=max_age_sec)

    def upsert_many(self, chain: str, mapping: Dict[str, Dict[str, Any]]) -> None:
        """
        mapping: { token_addr -> {"usd": "<str|number>", "source": "alchemy|0x|custom"} }
        Adds current epoch 'ts' automatically.
        """
        data = self._load()
        data["networks"].setdefault(chain, {})
        now = time.time()
        for addr, ent in (mapping or {}).items():
            if not addr or ent is None:
                continue
            try:
                usd_str = str(ent.get("usd"))
            except Exception:
                usd_str = "0"
            source = (ent.get("source") or "custom")
            data["networks"][chain][addr.lower()] = {"usd": usd_str, "source": str(source).lower(), "ts": now}
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
            "balance_hex": "0x0123...",    # raw hex string from RPC
            "asof_block": 12345678,        # block height when captured (optional)
            "ts": 1725640000.0,            # epoch seconds
            "decimals": 18,                # optional
            "quantity": "123.4567",        # optional pretty quantity
            "usd_amount": "12.34"          # optional cached USD amount
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

    # ---- expected by callers ----
    def get_token(self, wallet: str, chain: str, token: str) -> Dict[str, Any]:
        """
        Return the cached entry for `token` (lowercased) on (wallet, chain).
        If missing, returns {}.
        """
        st = self.get_state(wallet, chain)
        ent = st.get("tokens", {}).get(_lower(token))
        if isinstance(ent, dict):
            return ent
        return {}

    def upsert_many(self, wallet: str, chain: str, mapping: Dict[str, Dict[str, Any]]) -> None:
        """
        Merge a dict of {token -> entry_fields} into the cache.
        Recognized fields on each entry:
          - balance_hex or raw
          - asof_block or at_block
          - ts (epoch seconds) or updated_at (ISO string)
          - decimals, quantity, usd_amount (optional)
        Any unknown fields are merged as-is.
        """
        st = self.get_state(wallet, chain)
        tks = st["tokens"]
        now_iso = _now_ts()
        for addr, ent in (mapping or {}).items():
            if not addr or ent is None:
                continue
            key = _lower(addr)
            cur = dict(tks.get(key) or {})
            # Normalize synonyms
            balance_hex = ent.get("balance_hex")
            if balance_hex is None:
                raw = ent.get("raw")
                balance_hex = raw if raw is not None else cur.get("balance_hex")
            asof_block = ent.get("asof_block")
            if asof_block is None:
                asof_block = ent.get("at_block", cur.get("asof_block"))
            ts_val = ent.get("ts")
            updated_at_iso = ent.get("updated_at") if isinstance(ent.get("updated_at"), str) else None
            # Build merged entry
            merged = {**cur, **ent}
            if balance_hex is not None:
                merged["balance_hex"] = balance_hex
                # keep backward-compat mirror
                merged["raw"] = balance_hex
            if asof_block is not None:
                try:
                    merged["asof_block"] = int(asof_block)
                except Exception:
                    pass
            if ts_val is not None:
                try:
                    merged["ts"] = float(ts_val)
                except Exception:
                    pass
            # maintain a human-readable ISO timestamp too
            merged["updated_at"] = updated_at_iso or now_iso
            tks[key] = merged
        st["updated_at"] = now_iso
        _atomic_write(self._file(wallet, chain), st)

    # Legacy helper kept for compatibility with some callers
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
            tks[key]["balance_hex"] = raw
            tks[key]["raw"] = raw
            tks[key]["asof_block"] = int(at_block) if at_block is not None else tks[key].get("asof_block")
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

# For cache-only reads of a token entry:
ent = cb.get_token(wallet, chain, token_address)
# fields: balance_hex, asof_block, ts, decimals, quantity, usd_amount, updated_at (iso)

# To merge newly fetched balances and decorate with metadata:
payload = {
    token_address: {
        "balance_hex": "0x...",
        "asof_block": 1234,
        "ts": time.time(),
        "decimals": 18,
        "quantity": "12.34",
        "usd_amount": "34.56",
    },
    # ...
}
cb.upsert_many(wallet, chain, payload)
"""
