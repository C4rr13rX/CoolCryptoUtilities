from __future__ import annotations
import os, sys, time, json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set
from decimal import Decimal, getcontext
import requests
from dotenv_fallback import load_dotenv, find_dotenv, dotenv_values
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# NEW: caching (optional)
from cache import CacheBalances, CacheTransfers, CachePrices
from services.wallet_logger import wallet_log

getcontext().prec = 50 # high precision for token math

# ---------------- Simple Verbose Logger ----------------
class _V:
    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._indent = 0
    def on(self): self.enabled = True
    def off(self): self.enabled = False
    def _ts(self) -> str: return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3] + "Z"
    def log(self, msg: str, *args: Any) -> None:
        if self.enabled: print(f"[{self._ts()}] {' '*self._indent}{(msg if not args else msg.format(*args))}", flush=True)
    def section(self, title: str) -> None:
        if self.enabled: print(f"\n[{self._ts()}] {' '*self._indent}== {title} ==", flush=True)
    def warn(self, msg: str, *args: Any) -> None:
        if self.enabled: print(f"[{self._ts()}] {' '*self._indent}!! {(msg if not args else msg.format(*args))}", flush=True)
    def err(self, msg: str, *args: Any) -> None:
        if self.enabled: print(f"[{self._ts()}] {' '*self._indent}XX {(msg if not args else msg.format(*args))}", flush=True)
    def push(self, label: Optional[str] = None):
        if label: self.log(label); self._indent += 1
    def pop(self): 
        if self._indent > 0: self._indent -= 1

_VL = _V(enabled=os.getenv("TOKEN_PORTFOLIO_VERBOSE", "").strip().lower() in ("1","true","yes","on"))

def _expand_env(value: Optional[str]) -> str:
    if not value:
        return ""
    return os.path.expandvars(str(value)).strip()

def _mask_key(k: Optional[str]) -> str:
    if not k: return ""
    s = str(k); return "****" if len(s) <= 8 else s[:4] + "…" + s[-4:]

# ---------------- .env LOADER (PyDroid-safe) ----------------
def load_env_robust() -> None:
    _VL.section("ENV LOADER")
    path = find_dotenv(usecwd=True)
    if path: _VL.log("Found .env: {}", path); load_dotenv(path, override=False); return
    cands: List[Path] = []
    for p in (
        Path.cwd() / ".env",
        Path(sys.argv[0]).resolve().parent / ".env" if sys.argv and sys.argv[0] else None,
        Path(__file__).resolve().parent / ".env" if "__file__" in globals() else None,
        Path.home() / ".env",
    ):
        if p: cands.append(p)
    for p in cands:
        try:
            if p.is_file(): load_dotenv(p, override=False); _VL.log("Loaded .env from {}", p); return
        except Exception as e: _VL.warn("load_dotenv failed on {}: {}", p, e)
    for p in cands:
        try:
            if p.is_file():
                for k, v in (dotenv_values(p) or {}).items(): os.environ.setdefault(k, v or "")
                _VL.log("Loaded env via dotenv_values from {}", p); return
        except Exception as e: _VL.warn("dotenv_values failed on {}: {}", p, e)
    _VL.warn("No .env loaded; relying on process env only")
load_env_robust()

# ---------------- Helper types ----------------
Chains = ("ethereum", "base", "arbitrum", "optimism", "polygon")
CHAIN_CONFIG = {
    "ethereum": {
        "alchemy_slug": "eth-mainnet",
        "alchemy_prices_network": "eth-mainnet",
        "chain_id": 1,
        "env_alchemy_url": "ALCHEMY_ETH_URL",
        "env_infura_url": "INFURA_ETH_URL",
        "infura_base": "https://mainnet.infura.io/v3/",
        "public_rpcs": [
            "https://rpc.ankr.com/eth",
            "https://1rpc.io/eth",
            "https://eth.drpc.org",
            "https://eth.llamarpc.com",
            "https://rpc.flashbots.net",
        ],
    },
    "base": {
        "alchemy_slug": "base-mainnet",
        "alchemy_prices_network": "base-mainnet",
        "chain_id": 8453,
        "env_alchemy_url": "ALCHEMY_BASE_URL",
        "env_infura_url": None,
        "infura_base": None,
        "public_rpcs": [
            "https://mainnet.base.org",
            "https://base-rpc.publicnode.com",
            "https://1rpc.io/base",
            "https://base.drpc.org",
        ],
    },
    "arbitrum": {
        "alchemy_slug": "arb-mainnet",
        "alchemy_prices_network": "arb-mainnet",
        "chain_id": 42161,
        "env_alchemy_url": "ALCHEMY_ARB_URL",
        "env_infura_url": "INFURA_ARB_URL",
        "infura_base": "https://arbitrum-mainnet.infura.io/v3/",
        "public_rpcs": [
            "https://arb1.arbitrum.io/rpc",
            "https://arbitrum.drpc.org",
        ],
    },
    "optimism": {
        "alchemy_slug": "opt-mainnet",
        "alchemy_prices_network": "opt-mainnet",
        "chain_id": 10,
        "env_alchemy_url": "ALCHEMY_OP_URL",
        "env_infura_url": "INFURA_OP_URL",
        "infura_base": "https://optimism-mainnet.infura.io/v3/",
        "public_rpcs": [
            "https://mainnet.optimism.io",
            "https://optimism.drpc.org",
        ],
    },
    "polygon": {
        "alchemy_slug": "polygon-mainnet",
        "alchemy_prices_network": "polygon-mainnet",
        "chain_id": 137,
        "env_alchemy_url": "ALCHEMY_POLY_URL",
        "env_infura_url": "INFURA_POLY_URL",
        "infura_base": "https://polygon-mainnet.infura.io/v3/",
        "public_rpcs": [
            "https://polygon-rpc.com",
            "https://polygon.drpc.org",
        ],
    },
}
ZERO = "0x" + "0"*40

class MultiChainTokenPortfolio:
    """
    Build a snapshot FAST, preferring cache:
      • If every token has cached quantity + usd_amount and no transfers since its cached block → return immediately (ms).
      • Else, only refresh tokens whose contracts were touched since their cached block (or missing in cache).
      • Pricing is cache-only by default (no network). You can flip to hybrid via price_mode="hybrid".
    Result shape:
      { "<token_address>": {"quantity":"<str>", "usd_amount":"<str>", "transactions":{}} }
    """

    def __init__(
        self,
        wallet_address: str,
        tokens: Iterable[Any],
        default_chain: str = "ethereum",
        max_transfers_per_token: int = 0,
        max_retries: int = 3,
        retry_backoff_sec: float = 0.6,
        alchemy_api_key: Optional[str] = None,
        infura_api_key: Optional[str] = None,
        thegraph_api_key: Optional[str] = None,
        zerox_api_key: Optional[str] = None,
        verbose: Optional[bool] = None,
        # caches
        cache_balances: Optional[CacheBalances] = None,
        cache_transfers: Optional[CacheTransfers] = None,
        balance_ttl_sec: Optional[int] = None, # kept for compatibility; ignored when movement check is available
        reorg_safety_blocks: Optional[int] = None,
        cache_prices: Optional[CachePrices] = None,
        price_ttl_sec: Optional[int] = None,
        force_refresh: Optional[bool] = None,
        # NEW: pricing behavior
        price_mode: Optional[str] = None, # "cache_only" (default) | "hybrid"
    ):
        if verbose is None:
            verbose = os.getenv("TOKEN_PORTFOLIO_VERBOSE", "").strip().lower() in ("1","true","yes","on")
        self.v = _V(enabled=verbose)

        self.v.section("INIT")
        self.wallet = self._normalize_addr(wallet_address)
        if self.wallet.lower() == ZERO: raise ValueError("Refusing the zero address.")
        self.default_chain = default_chain.lower()
        if self.default_chain not in CHAIN_CONFIG: raise ValueError(f"Unsupported default chain '{default_chain}'")

        # Keys
        self.alchemy_key = (alchemy_api_key or os.getenv("ALCHEMY_API_KEY") or "").strip()
        self.alchemy_prices_key = (os.getenv("ALCHEMY_PRICES_API_KEY") or self.alchemy_key).strip()
        self.infura_key = (infura_api_key or os.getenv("INFURA_API_KEY") or "").strip()
        self.thegraph_key = (thegraph_api_key or os.getenv("THEGRAPH_API_KEY") or "").strip()
        self.zerox_key = (zerox_api_key or os.getenv("ZEROX_API_KEY") or "").strip()
        self.v.log("Keys — Alchemy: {}, AlchemyPrices: {}, Infura: {}, TheGraph: {}, 0x: {}",
                   _mask_key(self.alchemy_key), _mask_key(self.alchemy_prices_key),
                   _mask_key(self.infura_key), _mask_key(self.thegraph_key), _mask_key(self.zerox_key))

        # Endpoints
        self.alchemy_urls: Dict[str, str] = {}
        self.infura_urls: Dict[str, Optional[str]] = {}
        self.v.section("ENDPOINT BUILD")
        for ch, cfg in CHAIN_CONFIG.items():
            alc_env = cfg["env_alchemy_url"]
            alc_raw = os.getenv(alc_env) if alc_env else None
            alc_url = _expand_env(alc_raw)
            if alc_url:
                self.alchemy_urls[ch] = alc_url
            else:
                self.alchemy_urls[ch] = (
                    f"https://{cfg['alchemy_slug']}.g.alchemy.com/v2/{self.alchemy_key}" if self.alchemy_key else ""
                )
            inf_env = cfg["env_infura_url"]
            inf_raw = os.getenv(inf_env) if inf_env else None
            inf_url = _expand_env(inf_raw) if inf_raw else None
            if not inf_url and cfg["infura_base"] and self.infura_key:
                inf_url = cfg["infura_base"] + self.infura_key
            self.infura_urls[ch] = inf_url

        # Policy
        self.max_transfers_per_token = int(max_transfers_per_token)
        self.max_retries = int(max_retries)
        self.retry_backoff_sec = float(retry_backoff_sec)

        self.http = requests.Session()
        self.http.headers.update({"accept": "application/json", "content-type": "application/json"})

        # ---- caches + behavior ----
        # Auto-create caches if not provided (safe defaults).
        try:
            self.cb: Optional[CacheBalances] = cache_balances or CacheBalances()
        except Exception:
            self.cb = cache_balances
        try:
            self.ct: Optional[CacheTransfers] = cache_transfers or CacheTransfers()
        except Exception:
            self.ct = cache_transfers
        try:
            self.cp: Optional[CachePrices] = cache_prices or None
        except Exception:
            self.cp = cache_prices

        self.balance_ttl_sec = int(balance_ttl_sec if balance_ttl_sec is not None
                                   else os.getenv("BALANCE_TTL_SEC", "300"))
        self.price_ttl_sec = int(price_ttl_sec if price_ttl_sec is not None
                                   else os.getenv("PRICE_TTL_SEC", "300"))
        self.reorg_safety_blocks = int(
            reorg_safety_blocks if reorg_safety_blocks is not None
            else os.getenv("PORTFOLIO_REORG_SAFETY", "12")
        )

        # Default pricing: if you DIDN'T wire a price cache, do one-time network to prime USD.
        # If you DID wire a price cache, stay cache-only here.
        pm = (price_mode or ("hybrid" if self.cp is None else "cache_only")).strip().lower()
        self.price_mode = pm if pm in ("cache_only","hybrid") else "cache_only"
        force_refresh_flag = (
            force_refresh
            if force_refresh is not None
            else os.getenv("TOKEN_PORTFOLIO_FORCE_REFRESH", "").strip().lower() in ("1", "true", "yes", "force")
        )
        self.force_refresh = bool(force_refresh_flag)
        self.v.log(
            "Modes — price_mode={}, reorg_safety_blocks={}, force_refresh={}",
            self.price_mode,
            self.reorg_safety_blocks,
            self.force_refresh,
        )

        # Parse/resolve tokens
        self.tokens: List[Tuple[str,str]] = self._parse_tokens(tokens)
        self._ensure_detected_chains()


    def _touched_tokens_since(self, chain: str, since_block: int) -> Set[str]:
        # Use transfers cache if available.
        if self.ct:
            try:
                moved = self.ct.touched_tokens_since(self.wallet, chain, since_block) or set()
                return {self._normalize_addr(x).lower() for x in moved}
            except Exception as e:
                self.v.warn("touched_tokens_since(cache) failed: {}", e)

        # Fallback: minimal RPC probe.
        touched: Set[str] = set()
        hex_from = hex(max(since_block, 0))
        for direction in ("in", "out"):
            page_key, pages = None, 0
            while True:
                params = [{
                    "fromBlock": hex_from,
                    "toBlock": "latest",
                    "category": ["erc20"],
                    "withMetadata": False,
                    "maxCount": "0x3E8",
                    ("toAddress" if direction == "in" else "fromAddress"): self.wallet,
                }]
                if page_key:
                    params[0]["pageKey"] = page_key
                try:
                    res = self._alchemy(chain, "alchemy_getAssetTransfers", params)
                except Exception:
                    break
                for t in (res or {}).get("transfers", []):
                    rc = (t.get("rawContract") or {}).get("address")
                    if rc:
                        touched.add(self._normalize_addr(rc).lower())
                page_key = (res or {}).get("pageKey")
                pages += 1
                if not page_key or pages >= 2: # keep it snappy
                    break
        return touched
        
    # ---------------- Public API ----------------
    def build(self) -> Dict[str, Dict[str, Any]]:
        self.v.section("BUILD PORTFOLIO")
        wallet_log("portfolio.build_start", wallet=self.wallet, tokens=self.tokens)
        by_chain: Dict[str, List[str]] = {}
        for ch, addr in self.tokens: by_chain.setdefault(ch, []).append(addr)
        result: Dict[str, Dict[str, Any]] = {}

        for ch, token_list in by_chain.items():
            self.v.push(f"Chain: {ch} ({len(token_list)} tokens)")
            wallet_log("portfolio.chain_start", wallet=self.wallet, chain=ch, tokens=token_list)
            if not self._native_rpc_urls(ch):
                for a in token_list: result[a] = {"quantity":"0", "usd_amount":"0", "transactions":{}}
                self.v.pop(); continue

            # -------- Super-fast path: everything cached & no movement since
            fast = None
            if not self.force_refresh and self.alchemy_urls.get(ch):
                fast = self._fast_return_if_all_cached(ch, token_list)
            if fast is not None:
                self.v.log("Fast-return from cache for {}", ch)
                wallet_log("portfolio.chain_fast_path", wallet=self.wallet, chain=ch, tokens=token_list, reason="cache")
                result.update(fast); self.v.pop(); continue

            # -------- Determine which tokens must refresh (movement or missing)
            cached_entries: Dict[str, Dict[str, Any]] = {}
            meta_map: Dict[str, Dict[str, Any]] = {}
            min_block = None
            need_meta: set[str] = set()
            for a in token_list:
                ent = self.cb.get_token(self.wallet, ch, a) if self.cb else None
                if ent:
                    cached_entries[a] = ent
                    sym = ent.get("symbol") or ent.get("name")
                    if sym:
                        sym_str = str(sym).strip()
                        meta_map[a] = {
                            "symbol": sym_str.upper() if sym_str else None,
                            "name": ent.get("name"),
                        }
                    else:
                        need_meta.add(a)
                else:
                    need_meta.add(a)
                if ent and "asof_block" in ent:
                    b = int(ent["asof_block"])
                    min_block = b if min_block is None else min(min_block, b)
            wallet_log(
                "portfolio.cached_entries",
                wallet=self.wallet,
                chain=ch,
                cached={a: {"quantity": cached_entries[a].get("quantity"), "balance_hex": cached_entries[a].get("balance_hex")} for a in cached_entries},
            )

            touched: Set[str] = set()
            if min_block is not None:
                since = max(min_block - self.reorg_safety_blocks, 0)
                touched = self._touched_tokens_since(ch, since)

            refresh: List[str]
            keep: List[str]
            if self.force_refresh:
                refresh = list(token_list)
                keep = []
            else:
                refresh = [
                    a
                    for a in token_list
                    if (a not in cached_entries) or (a.lower() in {x.lower() for x in touched})
                ]
                keep = [a for a in token_list if a not in refresh]

            # No Alchemy endpoint means we cannot rely on touched-token detection; force refresh.
            if not (self.alchemy_urls.get(ch) or ""):
                refresh = list(token_list)
                keep = []

            self.v.log("To refresh: {} | keep from cache: {}", len(refresh), len(keep))
            wallet_log(
                "portfolio.refresh_plan",
                wallet=self.wallet,
                chain=ch,
                refresh=refresh,
                keep=keep,
                touched=list(touched),
            )

            # -------- Metadata & decimals
            decimals_map: Dict[str,int] = {}
            for a in token_list:
                d = None
                if a in cached_entries:
                    d = cached_entries[a].get("decimals")
                if isinstance(d, int):
                    decimals_map[a] = int(d)
                else:
                    need_meta.add(a)
            if need_meta:
                meta = self._get_metadata_bulk(ch, list(need_meta), max_workers=8)
                for a, m in meta.items():
                    if a not in decimals_map or not isinstance(decimals_map[a], int):
                        decimals_map[a] = int((m or {}).get("decimals", 18) or 18)
                    symbol = (m or {}).get("symbol")
                    name = (m or {}).get("name")
                    symbol_norm = None
                    if symbol:
                        sym_str = str(symbol).strip()
                        if sym_str:
                            symbol_norm = sym_str.upper()
                    elif name:
                        name_str = str(name).strip()
                        if name_str:
                            symbol_norm = name_str.upper()
                    if symbol or name:
                        meta_map[a] = {
                            "symbol": symbol_norm,
                            "name": name,
                        }

            # -------- Balances: fetch only refresh subset; use cache for keep
            balances_raw: Dict[str,str] = {}
            if refresh:
                fetched = self._safe_get_balances(ch, self.wallet, refresh)
                balances_raw.update(fetched)
                wallet_log(
                    "portfolio.fetched_balances",
                    wallet=self.wallet,
                    chain=ch,
                    fetched=fetched,
                )
                # choose an asof block and persist rich cache (incl. decimals)
                asof_block = self._chain_tip_block(ch)
                if self.cb:
                    payload = {}
                    for a in refresh:
                        meta_info = meta_map.get(a) or cached_entries.get(a, {}) or {}
                        symbol_val = meta_info.get("symbol") or meta_info.get("name")
                        symbol_norm = None
                        if symbol_val:
                            symbol_str = str(symbol_val).strip()
                            if symbol_str:
                                symbol_norm = symbol_str.upper()
                        payload[a] = {
                            "balance_hex": fetched.get(a, "0x0"),
                            "asof_block": asof_block,
                            "ts": time.time(),
                            "decimals": int(decimals_map.get(a, 18)),
                            "symbol": symbol_norm,
                            "name": meta_info.get("name") or symbol_val,
                        }
                    try: self.cb.upsert_many(self.wallet, ch, payload)
                    except Exception as e: self.v.warn("balance cache upsert_many failed: {}", e)

            for a in keep:
                if a in cached_entries:
                    balances_raw[a] = cached_entries[a].get("balance_hex", "0x0")
                else:
                    balances_raw[a] = "0x0"

            # -------- Compose quantities
            qty_map: Dict[str, Decimal] = {}
            for a in token_list:
                dec = int(decimals_map.get(a, 18))
                qty_map[a] = self._hex_to_decimal(balances_raw.get(a, "0x0"), dec)
            wallet_log(
                "portfolio.quantities",
                wallet=self.wallet,
                chain=ch,
                quantities={a: str(qty_map[a]) for a in qty_map},
                balances_raw=balances_raw,
            )

            # -------- USD amounts (cache-first)
            usd_map: Dict[str, str] = {}
            
            if self.price_mode == "cache_only":
                # Strictly from cache (balances cache or price cache); never hit the network.
                for a in token_list:
                    ent = cached_entries.get(a)
                    usd = ent.get("usd_amount") if ent else None
                    if usd is None and self.cp:
                        try:
                            px = self.cp.get_price(ch, a, ttl_sec=self.price_ttl_sec)
                            if px is not None and qty_map[a] > 0:
                                usd = str((Decimal(str(px)) * qty_map[a]).quantize(Decimal("0.00000001")))
                        except Exception:
                            pass
                    usd_map[a] = str(Decimal(str(usd))) if (usd is not None) else "0"
            
            else: # "hybrid"
                # 1) Reuse cached USD for tokens we kept (no movement).
                for a in keep:
                    usd_map[a] = str(cached_entries.get(a, {}).get("usd_amount", "0"))
                # 2) Price only the refresh subset that has non-zero quantity.
                refresh_nonzero = [a for a in refresh if qty_map[a] > 0]
                price_meta = {a: {"decimals": decimals_map[a]} for a in refresh_nonzero}
                prices = self._get_prices_usd(ch, refresh_nonzero, price_meta) if refresh_nonzero else {}
                for a in refresh:
                    px = prices.get(a.lower(), Decimal("0"))
                    usd_map[a] = str((qty_map[a] * px).quantize(Decimal("0.00000001")) if px > 0 else Decimal("0"))

            # -------- Persist pretty fields back to balance cache for next ms-run
            if self.cb:
                payload = {}
                for a in token_list:
                    meta_info = meta_map.get(a) or cached_entries.get(a, {}) or {}
                    symbol_val = meta_info.get("symbol") or meta_info.get("name")
                    symbol_norm = None
                    if symbol_val:
                        symbol_str = str(symbol_val).strip()
                        if symbol_str:
                            symbol_norm = symbol_str.upper()
                    payload[a] = {
                        "balance_hex": balances_raw.get(a, "0x0"),
                        "asof_block": self._chain_tip_block(ch), # conservative
                        "ts": time.time(),
                        "decimals": int(decimals_map.get(a, 18)),
                        "quantity": str(qty_map[a].normalize()),
                        "usd_amount": usd_map[a],
                        "symbol": symbol_norm,
                        "name": meta_info.get("name") or symbol_val,
                    }
                try: self.cb.upsert_many(self.wallet, ch, payload)
                except Exception as e: self.v.warn("balance cache upsert (final) failed: {}", e)

            # -------- Transfers (optional cap) — unchanged behavior
            for a in token_list:
                tx_dict: Dict[str, Dict[str, Any]] = {}
                if self.max_transfers_per_token and self.max_transfers_per_token > 0:
                    try:
                        txs = self._get_transfers_for_token(ch, a)
                        tx_dict = self._normalize_transfers(a, txs)
                    except Exception as e:
                        self.v.warn("Transfers failed for {}: {}", a, e)

                result[a] = {
                    "quantity": str(qty_map[a].normalize()),
                    "usd_amount": usd_map[a],
                    "transactions": tx_dict,
                }
            wallet_log(
                "portfolio.chain_complete",
                wallet=self.wallet,
                chain=ch,
                result={a: result[a] for a in token_list},
            )

            self.v.pop()

        return result

    # ---------------- Fast return if fully cached & no movement ----------------
    def _fast_return_if_all_cached(self, chain: str, token_list: List[str]) -> Optional[Dict[str, Dict[str, Any]]]:
        if not self.cb:
            return None

        entries: Dict[str, Dict[str, Any]] = {}
        blocks: List[int] = []

        for a in token_list:
            ent = self.cb.get_token(self.wallet, chain, a)
            if not ent:
                return None
            if not all(k in ent for k in ("quantity", "usd_amount", "asof_block")):
                return None
            entries[a] = ent
            try:
                blocks.append(int(ent["asof_block"]))
            except Exception:
                return None

        if not blocks:
            return None

        since_block = max(min(blocks) - self.reorg_safety_blocks, 0)

        # Prefer the transfers cache to avoid ANY RPC.
        if self.ct:
            try:
                st = self.ct.get_state(self.wallet, chain) or {}
                ct_last = int(st.get("last_block", 0))
                # If the transfers cache hasn't scanned past our cached snapshot → nothing moved.
                if ct_last <= since_block:
                    self.v.log("FAST-PATH: transfers cache behind or equal to snapshot; returning cached snapshot")
                    return {a: {"quantity": str(entries[a]["quantity"]),
                                "usd_amount": str(entries[a]["usd_amount"]),
                                "transactions": {}} for a in token_list}

                # Otherwise ask the cache which contracts touched since then.
                moved = self.ct.touched_tokens_since(self.wallet, chain, since_block) or set()
                moved = {self._normalize_addr(x).lower() for x in moved}
                if not moved:
                    self.v.log("FAST-PATH: transfers cache reports no movement; returning cached snapshot")
                    return {a: {"quantity": str(entries[a]["quantity"]),
                                "usd_amount": str(entries[a]["usd_amount"]),
                                "transactions": {}} for a in token_list}
                # Movement → not a fast return.
                return None
            except Exception as e:
                self.v.warn("FAST-PATH via transfers cache failed: {}", e)
                # fall back to RPC probe below

        # No transfers cache: do a tiny RPC probe.
        touched = self._touched_tokens_since(chain, since_block)
        if not touched:
            self.v.log("FAST-PATH: no movement by RPC probe; returning cached snapshot")
            return {a: {"quantity": str(entries[a]["quantity"]),
                        "usd_amount": str(entries[a]["usd_amount"]),
                        "transactions": {}} for a in token_list}
        return None

    # ---------------- Movement detection ----------------
    def _touched_tokens_since(self, chain: str, since_block: int) -> Set[str]:
        """
        Which token contracts touched the wallet since block N (both directions).
        Uses Alchemy getAssetTransfers with fromBlock filter; minimal pages.
        """
        touched: Set[str] = set()
        hex_from = hex(max(since_block, 0))
        for direction in ("in", "out"):
            page_key, pages = None, 0
            while True:
                params = [{
                    "fromBlock": hex_from,
                    "toBlock": "latest",
                    "category": ["erc20"],
                    "withMetadata": False,
                    "maxCount": "0x3E8",
                    ("toAddress" if direction=="in" else "fromAddress"): self.wallet,
                }]
                if page_key: params[0]["pageKey"] = page_key
                try:
                    res = self._alchemy(chain, "alchemy_getAssetTransfers", params)
                except Exception:
                    break
                for t in (res or {}).get("transfers", []):
                    rc = (t.get("rawContract") or {}).get("address")
                    if rc: touched.add(self._normalize_addr(rc).lower())
                page_key = (res or {}).get("pageKey")
                pages += 1
                if not page_key or pages >= 3: break # keep it snappy
        return touched

    # ---------------- RPC helpers & basics ----------------
    def _rpc(self, url: str, method: str, params: Any) -> Any:
        payload = {"jsonrpc":"2.0","id":1,"method":method,"params":params}
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = self.http.post(url, json=payload, timeout=20)
                if r.status_code in (429,500,502,503,504):
                    last_err = requests.HTTPError(f"{r.status_code} {r.reason}"); time.sleep(self.retry_backoff_sec*(attempt+1)); continue
                r.raise_for_status(); j = r.json()
                if "error" in j: raise RuntimeError(str(j["error"]))
                return j.get("result")
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                last_err = e; time.sleep(self.retry_backoff_sec*(attempt+1))
        raise last_err or RuntimeError("RPC failed")

    def _alchemy(self, chain: str, method: str, params: Any) -> Any:
        url = self.alchemy_urls.get(chain) or ""
        if not url: raise RuntimeError(f"No Alchemy URL for chain '{chain}'")
        return self._rpc(url, method, params)

    def _eth_call(self, chain: str, to_addr: str, data_hex: str) -> Optional[str]:
        params = [{"to": to_addr, "data": data_hex}, "latest"]
        for url in self._native_rpc_urls(chain):
            try:
                res = self._rpc(url, "eth_call", params)
                if isinstance(res, str):
                    return res
            except Exception:
                continue
        return None

    def _native_rpc_urls(self, chain: str) -> List[str]:
        urls: List[str] = []
        primary = self.alchemy_urls.get(chain)
        if primary:
            urls.append(primary)
        inf = self.infura_urls.get(chain)
        if inf:
            urls.append(inf)
        for candidate in CHAIN_CONFIG.get(chain, {}).get("public_rpcs", []) or []:
            if candidate and candidate not in urls:
                urls.append(candidate)
        return [u for u in urls if u]

    def _get_native_balance_hex(self, chain: str, wallet: str) -> str:
        params = [wallet, "latest"]
        for url in self._native_rpc_urls(chain):
            try:
                res = self._rpc(url, "eth_getBalance", params)
                if isinstance(res, str) and res.startswith("0x"):
                    return res
            except Exception:
                continue
        return "0x0"

    def _chain_tip_block(self, chain: str) -> int:
        for url in self._native_rpc_urls(chain):
            try:
                h = self._rpc(url, "eth_blockNumber", [])
                if isinstance(h, str) and h.startswith("0x"):
                    return int(h, 16)
                if isinstance(h, int):
                    return int(h)
            except Exception:
                continue
        return 0

    # ---------------- Balances (RPC fetchers) ----------------
    def _safe_get_balances(self, chain: str, wallet: str, tokens: List[str]) -> Dict[str, str]:
        native_tokens = [t for t in tokens if t.lower() == ZERO.lower()]
        erc_tokens = [t for t in tokens if t.lower() != ZERO.lower()]
        balances: Dict[str, str] = {}
        if erc_tokens:
            try:
                balances.update(self._get_balances_alchemy(chain, wallet, erc_tokens))
            except Exception:
                balances.update(self._get_balances_eth_call(chain, wallet, erc_tokens))
        if native_tokens:
            native_hex = self._get_native_balance_hex(chain, wallet)
            for addr in native_tokens:
                balances[addr] = native_hex
        wallet_log(
            "portfolio.safe_get_balances",
            wallet=wallet,
            chain=chain,
            tokens=tokens,
            result=balances,
        )
        return balances

    def _get_balances_alchemy(self, chain: str, wallet: str, tokens: List[str]) -> Dict[str, str]:
        balances: Dict[str, str] = {}
        chunk = 100
        for i in range(0, len(tokens), chunk):
            subset = tokens[i:i+chunk]
            params = [wallet, subset]
            res = self._alchemy(chain, "alchemy_getTokenBalances", params)
            for item in (res or {}).get("tokenBalances", []):
                addr = self._normalize_addr(item["contractAddress"]); balances[addr] = item.get("tokenBalance","0x0")
        return balances

    def _get_balances_eth_call(self, chain: str, wallet: str, tokens: List[str]) -> Dict[str, str]:
        balances: Dict[str, str] = {}; acct = wallet.lower().replace("0x","")
        for t in tokens:
            data = "0x70a08231" + "0"*24 + acct
            hexval = self._eth_call(chain, t, data); balances[t] = hexval or "0x0"
        return balances

    # ---------------- Metadata ----------------
    def _get_metadata_bulk(self, chain: str, token_list: List[str], max_workers: int = 8) -> Dict[str, Dict[str, Any]]:
        def _default() -> Dict[str, Any]: return {"decimals": 18, "name": None, "symbol": None, "logo": None}
        out: Dict[str, Dict[str, Any]] = {}; 
        if not token_list: return out
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(self._safe_get_token_metadata, chain, addr): addr for addr in token_list}
            for fut in as_completed(futs):
                addr = futs[fut]
                try: out[addr] = fut.result() or _default()
                except Exception: out[addr] = _default()
        return out

    def _safe_get_token_metadata(self, chain: str, token: str) -> Dict[str, Any]:
        try:
            res = self._alchemy(chain, "alchemy_getTokenMetadata", [token])
            d = res.get("decimals", 18); 
            try: d = int(d)
            except Exception: d = 18
            return {"decimals": d, "name": res.get("name"), "symbol": res.get("symbol"), "logo": res.get("logo")}
        except Exception:
            dec = self._erc20_decimals(chain, token); return {"decimals": dec, "name": None, "symbol": None, "logo": None}

    def _erc20_decimals(self, chain: str, token: str) -> int:
        res = self._eth_call(chain, token, "0x313ce567")
        try: return int(res, 16) if res else 18
        except Exception: return 18

    # ---------------- Pricing (hybrid only; cache-only handled inline) ----------------
    def _get_prices_usd(self, chain: str, tokens: List[str], meta: Dict[str, Any]) -> Dict[str, Decimal]:
        # Alchemy by-address then 0x fallback (unchanged); omitted here for brevity since cache-only is default.
        prices: Dict[str, Decimal] = {a.lower(): Decimal("0") for a in tokens}
        if not tokens: return prices
        if not self.alchemy_prices_key: return prices

        net = CHAIN_CONFIG[chain]["alchemy_prices_network"]
        base = f"https://api.g.alchemy.com/prices/v1/{self.alchemy_prices_key}/tokens/by-address"
        chunk = 25; headers = {"Content-Type":"application/json"}
        for i in range(0, len(tokens), chunk):
            addr_chunk = tokens[i:i+chunk]
            payload = {"addresses":[{"network": net, "address": a} for a in addr_chunk]}
            try:
                r = self.http.post(base, json=payload, headers=headers, timeout=30); r.raise_for_status()
                data = r.json().get("data", [])
                for item in data:
                    addr = self._normalize_addr(item.get("address",""))
                    usd = Decimal("0")
                    for p in item.get("prices", []):
                        if (p.get("currency") or "").upper() == "USD":
                            try: usd = Decimal(str(p.get("value","0")))
                            except Exception: usd = Decimal("0")
                            break
                    prices[addr.lower()] = usd
            except Exception: pass
        # 0x fallback disabled here to keep this fast; you can re-enable if you want.
        return prices

    # ---------------- Normalization, Transfers, Utils ----------------
    @staticmethod
    def _hex_to_decimal(x_hex: str, decimals: int) -> Decimal:
        if not x_hex: return Decimal(0)
        try: raw = int(x_hex, 16)
        except Exception: return Decimal(0)
        scale = Decimal(10) ** Decimal(max(decimals, 0))
        return Decimal(raw) / scale

    def _get_transfers_for_token(self, chain: str, token: str) -> List[Dict[str, Any]]:
        all_txs: List[Dict[str, Any]] = []
        def fetch(direction: str) -> None:
            page_key, fetched = None, 0
            while True:
                params = [{
                    "fromBlock": "0x0", "toBlock": "latest", "category": ["erc20"],
                    "contractAddresses": [token], "withMetadata": True, "maxCount": "0x3E8",
                    ("toAddress" if direction=="in" else "fromAddress"): self.wallet,
                }]
                if page_key: params[0]["pageKey"] = page_key
                res = self._alchemy(chain, "alchemy_getAssetTransfers", params)
                trs = (res or {}).get("transfers", [])
                for t in trs: t["_direction"] = direction
                all_txs.extend(trs); fetched += len(trs)
                page_key = (res or {}).get("pageKey")
                if not page_key or fetched >= self.max_transfers_per_token: break
                time.sleep(0.2)
        fetch("in"); fetch("out")
        all_txs.sort(key=lambda x: (x.get("metadata", {}) or {}).get("blockTimestamp", "") or "")
        return all_txs

    def _normalize_transfers(self, token: str, transfers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}; collisions: Dict[str, int] = {}
        for t in transfers:
            ts = (t.get("metadata", {}) or {}).get("blockTimestamp", "") or f"unknown-{(t.get('hash') or '')[:10]}"
            key = ts if ts not in out else f"{ts}#{collisions.get(ts,0)+1}"; collisions[ts] = collisions.get(ts,0)+1
            sender = (t.get("from") or "").lower(); receiver = (t.get("to") or "").lower()
            direction = t.get("_direction", "in")
            try: value = Decimal(str(t.get("value", 0)))
            except Exception: value = Decimal(0)
            tx_type = "in" if direction == "in" else "out"
            if sender == ZERO and receiver == self.wallet.lower(): tx_type = "mint"
            elif receiver == ZERO and sender == self.wallet.lower(): tx_type = "burn"
            elif sender == self.wallet.lower() and receiver == self.wallet.lower(): tx_type = "self"
            out[key] = {
                "type": tx_type,
                "token_address_to": token, "token_quantity_to": str(value),
                "token_address_from": token, "token_quantity_from": str(value),
                "sender_address": sender, "receiver_address": receiver,
            }
        return out

    @staticmethod
    def _normalize_addr(addr: str) -> str:
        if not addr: return addr
        a = addr.strip().lower()
        if a.startswith("0x"): a = a[2:]
        return "0x" + a.zfill(40)[:40]

    # ---------------- Token parsing / autodetect ----------------
    def _parse_tokens(self, tokens: Iterable[Any]) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for item in tokens:
            ch, addr = None, None
            if isinstance(item, str):
                s = item.strip().replace("\u201c", '"').replace("\u201d", '"').replace("\u2018","'").replace("\u2019","'").replace("\u00A0"," ")
                if ":" in s and s.split(":",1)[0].lower() in CHAIN_CONFIG:
                    ch, a = s.split(":",1); ch = ch.lower().strip(); addr = self._normalize_addr(a.strip())
                elif "@" in s:
                    a, ch = s.split("@",1); addr = self._normalize_addr(a.strip()); ch = ch.lower().strip()
                else:
                    addr = self._normalize_addr(s); ch = None
            elif isinstance(item, (tuple,list)) and len(item)==2:
                ch = str(item[0]).lower().strip(); addr = self._normalize_addr(str(item[1]).strip())
            elif isinstance(item, dict):
                ch = str(item.get("chain","")).lower().strip() or None
                addr = self._normalize_addr(str(item.get("address","")).strip())
            else:
                continue
            if not addr or len(addr)!=42: continue
            out.append((ch or self.default_chain, addr))
        return out

    def _ensure_detected_chains(self) -> None:
        resolved: List[Tuple[str,str]] = []
        for ch, addr in self.tokens:
            if ch in CHAIN_CONFIG:
                # Keep explicit chain hint even if we currently lack a dedicated Alchemy URL.
                resolved.append((ch, addr))
                continue
            detected = self._detect_chain_for_token(addr)
            resolved.append((detected or self.default_chain, addr))
        self.tokens = resolved

    def _detect_chain_for_token(self, addr: str) -> Optional[str]:
        for ch in CHAIN_CONFIG:
            if not (self.alchemy_urls.get(ch) or ""): continue
            res = self._eth_call(ch, addr, "0x313ce567")
            if res: return ch
        return None


if __name__ == "__main__":
    wallet = ""
    tokens = [
        "",
        "",
    ]

    # Optional caches (plug your real ones)
    cb = None # CacheBalances()
    ct = None # CacheTransfers()
    cp = None # CachePrices()

    tp = MultiChainTokenPortfolio(
        wallet, tokens,
        max_transfers_per_token=0,
        verbose=True,
        cache_balances=cb,
        cache_transfers=ct,
        cache_prices=cp,
        price_mode="cache_only", # <— default; no pricing network calls here
    )
    result = tp.build()
    print(json.dumps(result, indent=2))
