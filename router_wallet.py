
from __future__ import annotations
import os, time, json, certifi, requests, sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable
from pathlib import Path
from dotenv import load_dotenv, find_dotenv, dotenv_values
from filter_scams import FilterScamTokens
from cache import CacheBalances, CacheTransfers


# ---------- .env loader ----------
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import time
def load_env_robust() -> None:
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False); return

    cands: List[Path] = []
    for p in (
        Path.cwd() / ".env",
        Path(sys.argv[0]).resolve().parent / ".env" if sys.argv and sys.argv[0] else None,
        Path(__file__).resolve().parent / ".env" if "__file__" in globals() else None,
        Path.home() / ".env",
    ):
        if p:
            cands.append(p)

    for p in cands:
        try:
            if p.is_file():
                load_dotenv(p, override=False); return
        except Exception:
            pass

    for p in cands:
        try:
            if p.is_file():
                for k, v in (dotenv_values(p) or {}).items():
                    os.environ.setdefault(k, v or "")
                return
        except Exception:
            pass

load_env_robust()

# ---------- deps ----------
from web3 import Web3
from eth_account import Account
from eth_account.signers.local import LocalAccount
try:
    from web3.middleware import ExtraDataToPOAMiddleware as POA_MIDDLEWARE # v6
except ImportError:
    try:
        from web3.middleware import geth_poa_middleware as POA_MIDDLEWARE # v5
    except ImportError:
        POA_MIDDLEWARE = None

REQ_KW = {"timeout": 30, "verify": certifi.where()}

# Global Alchemy key (optional). If set, we'll build per-chain URLs automatically.
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY", "").strip()
ZEROX_API_KEY = os.getenv("ZEROX_API_KEY", "").strip() # optional for 0x fallback pricing

# Alchemy slugs per chain for auto URL building
ALCHEMY_SLUGS: Dict[str, str] = {
    "ethereum": "eth-mainnet",
    "base": "base-mainnet",
    "arbitrum": "arb-mainnet",
    "optimism": "opt-mainnet",
    "polygon": "polygon-mainnet",
}

JSON_HEADERS = {"accept": "application/json", "content-type": "application/json"}

# ---------- chains & RPCs ----------
def _env(url: Optional[str]) -> List[str]:
    return [url] if url and url.strip() else []

CHAINS: Dict[str, Dict[str, Any]] = {
    "ethereum": {
        "id": 1, "poa": False,
        "rpcs": (
            _env(os.getenv("ALCHEMY_ETH_URL"))
            + _env(os.getenv("INFURA_ETH_URL"))
            + [
                "https://rpc.ankr.com/eth",
                "https://1rpc.io/eth",
                "https://eth.drpc.org",
                "https://eth.llamarpc.com",
                "https://rpc.flashbots.net",
            ]
        ),
    },
    "base": {
        "id": 8453, "poa": True,
        "rpcs": (
            _env(os.getenv("ALCHEMY_BASE_URL"))
            + [
                "https://mainnet.base.org",
                "https://base-rpc.publicnode.com",
                "https://1rpc.io/base",
                "https://base.drpc.org",
            ]
        ),
    },
    "arbitrum": {
        "id": 42161, "poa": False,
        "rpcs": (
            _env(os.getenv("ALCHEMY_ARB_URL"))
            + _env(os.getenv("INFURA_ARB_URL"))
            + ["https://arb1.arbitrum.io/rpc", "https://arbitrum.drpc.org"]
        ),
    },
    "optimism": {
        "id": 10, "poa": False,
        "rpcs": (
            _env(os.getenv("ALCHEMY_OP_URL"))
            + _env(os.getenv("INFURA_OP_URL"))
            + ["https://mainnet.optimism.io", "https://optimism.drpc.org"]
        ),
    },
    "polygon": {
        "id": 137, "poa": True,
        "rpcs": (
            _env(os.getenv("ALCHEMY_POLY_URL"))
            + _env(os.getenv("INFURA_POLY_URL"))
            + ["https://polygon-rpc.com", "https://polygon.drpc.org"]
        ),
    },
}

# Alchemy env var names per chain (for discovery)
ALCHEMY_ENV: Dict[str, str] = {
    "ethereum": "ALCHEMY_ETH_URL",
    "base": "ALCHEMY_BASE_URL",
    "arbitrum": "ALCHEMY_ARB_URL",
    "optimism": "ALCHEMY_OP_URL",
    "polygon": "ALCHEMY_POLY_URL",
}

# ---------- LI.FI & ERC20 ----------
LIFI_BASE = os.getenv("LIFI_BASE", "https://li.quest/v1")
LIFI_API_KEY = os.getenv("LIFI_API_KEY", "").strip()
NATIVE = "0x0000000000000000000000000000000000000000"

ERC20_ABI = [
    {"constant": True, "inputs": [], "name": "decimals", "outputs":[{"name":"","type":"uint8"}], "type":"function"},
    {"constant": True, "inputs": [{"name":"account","type":"address"}], "name":"balanceOf", "outputs":[{"name":"","type":"uint256"}], "type":"function"},
    {"constant": True, "inputs": [{"name":"owner","type":"address"},{"name":"spender","type":"address"}], "name":"allowance", "outputs":[{"name":"","type":"uint256"}], "type":"function"},
    {"constant": False, "inputs": [{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}], "name":"approve", "outputs":[{"name":"","type":"bool"}], "type":"function"},
]

@dataclass
class QuotePreview:
    from_chain: int
    to_chain: int
    from_symbol: str
    to_symbol: str
    from_amount: float
    to_amount: float
    to_amount_min: float
    gas_usd: Optional[float]
    tx_request: Dict[str, Any]

# ---------- helpers ----------
from requests import Session
_session: Optional[Session] = None
def _http() -> Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(JSON_HEADERS)
    return _session

def _w3(chain: str) -> Web3:
    cfg = CHAINS[chain]; errs: List[str] = []
    for url in cfg["rpcs"]:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs=REQ_KW))
            if w3.is_connected():
                if cfg.get("poa") and POA_MIDDLEWARE:
                    w3.middleware_onion.inject(POA_MIDDLEWARE, layer=0)
                return w3
        except Exception as e:
            errs.append(f"{url} -> {type(e).__name__}: {e}")
    raise RuntimeError(f"RPC not reachable for {chain}. Tried:\n" + "\n".join(errs))

def _erc20(w3: Web3, addr: str):
    return w3.eth.contract(address=Web3.to_checksum_address(addr), abi=ERC20_ABI)

def _decimals(w3: Web3, token: str) -> int:
    if token.lower() == NATIVE: return 18
    return _erc20(w3, token).functions.decimals().call()

def _to_wei(amount: float, decimals: int) -> int:
    return int(round(amount * (10 ** decimals)))

def _from_wei(x: int | str, decimals: int) -> float:
    if isinstance(x, str): x = int(x,16) if x.startswith("0x") else int(x)
    return x / float(10 ** decimals)

def _eip1559_fees(w3: Web3) -> Dict[str, int]:
    try:
        mp = w3.eth.max_priority_fee
    except Exception:
        gp = w3.eth.gas_price
        mp = int(gp * 0.125)
    return {"maxPriorityFeePerGas": mp, "maxFeePerGas": int(w3.eth.gas_price * 2)}

def _normalize_addr(a: str) -> str:
    if not a: return a
    a = a.strip()
    if a.upper() in ("ETH", "NATIVE"): return NATIVE
    if not a.startswith("0x"): a = "0x" + a
    return Web3.to_checksum_address(a)

# ---------- chain detection (ERC-20) ----------
def detect_chain_for_token(addr: str, candidates: Iterable[str] = CHAINS.keys()) -> Optional[str]:
    addr = _normalize_addr(addr)
    if addr.lower() == NATIVE: return None
    for ch in candidates:
        try:
            w3 = _w3(ch)
            _erc20(w3, addr).functions.decimals().call()
            return ch
        except Exception:
            continue
    return None

def annotate_tokens(tokens: Iterable[Any], default_chain: str = "ethereum") -> List[Tuple[str, str]]:
    out: List[Tuple[str,str]] = []
    for it in tokens:
        ch, a = None, None
        if isinstance(it, str):
            s = it.strip()
            if ":" in s and s.split(":",1)[0].lower() in CHAINS:
                ch, a = s.split(":",1); ch = ch.lower(); a = a.strip()
            elif "@" in s:
                a, ch = s.split("@",1); a = a.strip(); ch = ch.lower()
            else:
                a = s
        elif isinstance(it, (tuple,list)) and len(it)==2:
            ch = str(it[0]).lower(); a = str(it[1])
        elif isinstance(it, dict):
            ch = str(it.get("chain","") or "").lower() or None
            a = str(it.get("address","") or "")
        else:
            continue

        addr = _normalize_addr(a)
        if addr.lower() == NATIVE:
            out.append(((ch or default_chain), NATIVE)); continue

        if ch and ch in CHAINS:
            out.append((ch, addr))
        else:
            detected = detect_chain_for_token(addr) or default_chain
            out.append((detected, addr))
    return out

# ---------- LI.FI HTTP ----------
def _lifi_quote(params: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"accept": "application/json"}
    if LIFI_API_KEY: headers["x-lifi-api-key"] = LIFI_API_KEY
    r = _http().get(f"{LIFI_BASE}/quote", params=params, timeout=30, verify=certifi.where())
    r.raise_for_status(); return r.json()

def _lifi_status(tx_hash: str) -> Dict[str, Any]:
    r = _http().get(f"{LIFI_BASE}/status", params={"txHash": tx_hash}, timeout=30, verify=certifi.where())
    r.raise_for_status(); return r.json()

# ---------- Optional: 0x pricing helpers (non-intrusive) ----------
from decimal import Decimal, InvalidOperation, getcontext as _getctx
_getctx().prec = 50

def _dec_to_str_plain(x: Decimal, max_dp: int) -> str:
    """No scientific notation; trim trailing zeros."""
    s = format(x, "f")
    if "." in s:
        intp, frac = s.split(".", 1)
        frac = frac[:max_dp].rstrip("0")
        return intp if not frac else f"{intp}.{frac}"
    return s

def _to_decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(0)

def normalize_snapshot_numbers(snapshot: Dict[str, Dict[str, Any]], qty_dp: int = 18, usd_dp: int = 8) -> None:
    """Post-process only formatting. Does NOT change balances."""
    for info in snapshot.values():
        q = _to_decimal(info.get("quantity", "0"))
        u = _to_decimal(info.get("usd_amount", "0"))
        info["quantity"] = _dec_to_str_plain(q, qty_dp)
        info["usd_amount"] = _dec_to_str_plain(u, usd_dp)

def _price_usd_0x_single(chain: str, token: str, decimals: int) -> Decimal:
    """
    Ask 0x for a price on `chain`:
    1) sellAmount = 1 token → get price or buyAmount USDC.
    2) if 0/failed: buyAmount = 1e6 USDC → invert tokens_for_1_usdc.
    Returns Decimal USD per 1 token, or 0 if unavailable.
    """
    base = "https://api.0x.org/swap/v1/price"
    headers = {"accept": "application/json"}
    if ZEROX_API_KEY:
        headers["0x-api-key"] = ZEROX_API_KEY
        headers["0x-version"] = "v2"

    chain_id = CHAINS[chain]["id"]
    addr = _normalize_addr(token)
    price = Decimal(0)

    # Path 1: sell 1 token
    try:
        sell_amount = str(int(Decimal(10) ** Decimal(decimals)))
    except (InvalidOperation, OverflowError):
        sell_amount = str(10 ** min(decimals, 36))

    try:
        r = _http().get(base, params={
            "chainId": str(chain_id),
            "sellToken": addr,
            "buyToken": "USDC",
            "sellAmount": sell_amount,
        }, headers=headers, timeout=20, verify=certifi.where())
        if r.status_code in (429, 500, 502, 503, 504):
            raise requests.HTTPError(f"{r.status_code} {r.reason}")
        r.raise_for_status()
        j = r.json()
        if "price" in j:
            price = Decimal(str(j["price"]))
        elif "buyAmount" in j:
            usdc = Decimal(j["buyAmount"]) / Decimal(10**6)
            price = usdc
    except Exception:
        price = Decimal(0)

    # Path 2: buy 1 USDC and invert
    if price <= 0:
        try:
            r = _http().get(base, params={
                "chainId": str(chain_id),
                "sellToken": addr,
                "buyToken": "USDC",
                "buyAmount": str(10**6), # 1 USDC
            }, headers=headers, timeout=20, verify=certifi.where())
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {r.reason}")
            r.raise_for_status()
            j = r.json()
            if "sellAmount" in j:
                tokens_for_1_usdc = Decimal(j["sellAmount"]) / (Decimal(10) ** Decimal(decimals))
                if tokens_for_1_usdc > 0:
                    price = Decimal(1) / tokens_for_1_usdc
            elif "price" in j:
                price = Decimal(str(j["price"]))
        except Exception:
            price = Decimal(0)

    return price if price > 0 else Decimal(0)

def enrich_portfolio_with_0x(snapshot: Dict[str, Dict[str, Any]],
                              addr_to_chain: Dict[str, str],
                              qty_dp: int = 18,
                              usd_dp: int = 8) -> None:
    """
    Only fills in USD for items with usd_amount==0 using 0x.
    Does NOT touch quantities. Adds minimal formatting after fill.
    """
    # Group missing by chain
    missing_by_chain: Dict[str, List[str]] = {}
    for addr, info in snapshot.items():
        usd = _to_decimal(info.get("usd_amount", "0"))
        if usd <= 0:
            ch = addr_to_chain.get(addr.lower())
            if ch:
                missing_by_chain.setdefault(ch, []).append(addr)

    # Query 0x per token and update USD
    for ch, addrs in missing_by_chain.items():
        if not addrs:
            continue
        try:
            w3 = _w3(ch)
        except Exception:
            continue
        for a in addrs:
            try:
                dec = _decimals(w3, a)
                px = _price_usd_0x_single(ch, a, dec)
                if px > 0:
                    qty = _to_decimal(snapshot[a].get("quantity", "0"))
                    usd_total = (qty * px)
                    snapshot[a]["usd_amount"] = _dec_to_str_plain(usd_total, usd_dp)
            except Exception:
                pass

# ======================================================================
# ROUTER
# ======================================================================
class UltraSwapBridge:
    """
    - connect wallet (mnemonic/private key)
    - discover wallet ERC-20s across chains (Alchemy getTokenBalances primary; transfers fallback)
    - annotate tokens for downstream use
    - quote/execute via LI.FI, send native/ERC-20
    """

    def __init__(
        self,
        mnemonic: Optional[str]=None,
        private_key: Optional[str]=None,
        derivation_path: str="m/44'/60'/0'/0/0",
        cache_transfers: Optional[CacheTransfers]=None, # <--- NEW
        reorg_safety_blocks: Optional[int]=None, # <--- NEW
    ):
        mnemonic = mnemonic or os.getenv("MNEMONIC")
        private_key = private_key or os.getenv("PRIVATE_KEY")
        if mnemonic:
            Account.enable_unaudited_hdwallet_features()
            self.acct: LocalAccount = Account.from_mnemonic(mnemonic, account_path=derivation_path)
        elif private_key:
            self.acct = Account.from_key(private_key)
        else:
            raise ValueError("Provide MNEMONIC or PRIVATE_KEY (in .env or constructor).")

        # cache for incremental transfer scans
        self.ct: Optional[CacheTransfers] = cache_transfers
        # small buffer to survive reorgs; env override PORTFOLIO_REORG_SAFETY
        self.reorg_safety = (
            int(os.getenv("PORTFOLIO_REORG_SAFETY", "12"))
            if reorg_safety_blocks is None else int(reorg_safety_blocks)
        )

    # -------- wallet getters
    def get_address(self) -> str:
        return self.acct.address

    # -------- token discovery (Alchemy)
    def _alchemy_url(self, chain: str) -> str:
        """Return a usable Alchemy URL for `chain` from env, or build from ALCHEMY_API_KEY."""
        env_key = ALCHEMY_ENV.get(chain, "")
        url = (os.getenv(env_key) or "").strip()
        if url:
            return url
        if ALCHEMY_API_KEY and chain in ALCHEMY_SLUGS:
            return f"https://{ALCHEMY_SLUGS[chain]}.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        return ""

    def _alchemy_post(self, url: str, method: str, params: Any) -> Dict[str, Any]:
        """Thin wrapper with gentle retry for transient 5xx/429."""
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        for attempt in range(3):
            r = _http().post(url, json=payload, timeout=60, verify=certifi.where())
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(0.4 * (attempt + 1))
                continue
            r.raise_for_status()
            j = r.json()
            if "error" in j:
                # surface as HTTPError-ish to trigger fallback
                raise requests.HTTPError(str(j["error"]))
            return j
        # final try
        r.raise_for_status()
        return r.json()

    def _discover_via_balances(self, url: str, min_balance_wei: int = 1) -> List[str]:
        """
        Use alchemy_getTokenBalances (paginated) to fetch ERC-20s with non-zero balances.
        Returns checksum addresses.
        """
        seen: List[str] = []
        page_key: Optional[str] = None
        while True:
            # Proper, minimal payload: [wallet, "erc20"] (+ pageKey on result)
            params: List[Any] = [self.acct.address, "erc20"]
            if page_key:
                pass
            res = self._alchemy_post(url, "alchemy_getTokenBalances", params + ([{"pageKey": page_key}] if page_key else [])).get("result") or {}
            token_balances = res.get("tokenBalances", []) or []
            for tb in token_balances:
                addr = tb.get("contractAddress")
                bal_hex = tb.get("tokenBalance")
                if not addr or not bal_hex or bal_hex in ("0x", "0x0"):
                    continue
                try:
                    wei = int(bal_hex, 16)
                except Exception:
                    wei = 0
                if wei >= min_balance_wei:
                    try:
                        seen.append(Web3.to_checksum_address(addr))
                    except Exception:
                        pass
            page_key = res.get("pageKey")
            if not page_key:
                break
            time.sleep(0.15)
        return seen

    def _discover_via_transfers(self, url: str, chain: str, max_pages_per_dir: int = 5) -> List[str]:
        """
        Fallback discovery via transfer history.
        NOW INCREMENTAL: if CacheTransfers is provided, only scan from last cached block + reorg buffer,
        merge new pages to cache, and return distinct ERC-20 contract addresses seen (new+old if cache exists).
        """
        wallet = self.acct.address
        from_block_hex = "0x0"
        baseline_last = -1

        if self.ct is not None:
            st = self.ct.get_state(wallet, chain) or {}
            baseline_last = int(st.get("last_block", -1))
            start = max(0, baseline_last + 1 - self.reorg_safety)
            from_block_hex = hex(start)
            print(f"[discover/transfers] {chain}: cache baseline last_block={baseline_last}, scanning from {from_block_hex}")

        new_pages: List[Dict[str, Any]] = []

        def _scan(direction: str) -> None:
            page_key, pages = None, 0
            while True:
                filt = {
                    "fromBlock": from_block_hex,
                    "toBlock": "latest",
                    "category": ["erc20"],
                    "withMetadata": False,
                    "maxCount": "0x3E8",
                }
                if direction == "in":
                    filt["toAddress"] = wallet
                else:
                    filt["fromAddress"] = wallet
                if page_key:
                    filt["pageKey"] = page_key

                payload = {"jsonrpc": "2.0", "id": 1, "method": "alchemy_getAssetTransfers", "params": [filt]}
                r = _http().post(url, json=payload, timeout=60, verify=certifi.where())
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(0.6); pages += 1
                    if pages >= max_pages_per_dir: break
                    continue
                r.raise_for_status()
                res = (r.json() or {}).get("result") or {}
                trs = res.get("transfers", []) or []
                if not trs and not res.get("pageKey"):
                    break
                new_pages.extend(trs)
                page_key = res.get("pageKey")
                pages += 1
                if not page_key or pages >= max_pages_per_dir:
                    break

        _scan("in")
        _scan("out")

        # Merge new transfers into cache (idempotent), compute addresses & new last_block
        addrs: set[str] = set()
        if self.ct is not None:
            try:
                self.ct.merge_new(wallet, chain, new_pages)
                # collect all known addresses from cache (complete history)
                all_transfers = self.ct.get_all(wallet, chain) or []
                for t in all_transfers:
                    rc = (t.get("rawContract") or {}).get("address")
                    if rc:
                        try: addrs.add(Web3.to_checksum_address(rc))
                        except Exception: pass
                # optional: quick report of tokens touched since baseline
                if baseline_last >= 0:
                    changed = self.ct.touched_tokens_since(wallet, chain, since_block=baseline_last) or set()
                    if changed:
                        print(f"[discover/transfers] {chain}: tokens changed since {baseline_last}: {len(changed)}")
            except Exception as e:
                print(f"[discover/transfers] {chain}: cache merge error {type(e).__name__}: {e}")
                # fall back to addresses from just-downloaded pages
                for t in new_pages:
                    rc = (t.get("rawContract") or {}).get("address")
                    if rc:
                        try: addrs.add(Web3.to_checksum_address(rc))
                        except Exception: pass
        else:
            for t in new_pages:
                rc = (t.get("rawContract") or {}).get("address")
                if rc:
                    try: addrs.add(Web3.to_checksum_address(rc))
                    except Exception: pass

        return sorted(addrs)

    def discover_tokens_pairs(
        self,
        chains: Optional[Iterable[str]] = None,
        min_balance_wei: int = 1,
    ) -> List[Tuple[str, str]]:
        """
        Returns [(chain, checksum_token_address), ...] for non-zero ERC-20 balances.
        Primary: alchemy_getTokenBalances (paginated).
        Fallback: transfer scan (in/out, cached incrementally if CacheTransfers provided).
        """
        chains = list(chains) if chains else list(CHAINS.keys())
        out: List[Tuple[str, str]] = []

        for ch in chains:
            
            # FAST PATH: use cached token list for this chain if available
            try:
                cb = CacheBalances()
                cached_map = (cb.get_state(self.acct.address, ch) or {}).get("tokens", {}) or {}
                if cached_map:
                    addrs = []
                    for a in cached_map.keys():
                        try:
                            addrs.append(Web3.to_checksum_address(a))
                        except Exception:
                            pass
                    out.extend((ch, a) for a in addrs)
                    print(f"[discover] {ch}: using {len(cached_map)} cached tokens (balances)")
                    continue
            except Exception:
                pass
            url = self._alchemy_url(ch)
            if not url:
                print(f"[discover] {ch}: no Alchemy URL")
                continue

            try:
                addrs = self._discover_via_balances(url, min_balance_wei=min_balance_wei)
                out.extend((ch, a) for a in addrs)
                print(f"[discover] {ch}: found {len(addrs)} tokens via balances")
            except Exception as e:
                print(f"[discover] {ch}: balances error {type(e).__name__}: {e}")
                try:
                    addrs = self._discover_via_transfers(url, chain=ch)
                    out.extend((ch, a) for a in addrs)
                    print(f"[discover] {ch}: found {len(addrs)} tokens via transfers (cached)")
                except Exception as e2:
                    print(f"[discover] {ch}: transfer fallback error {type(e2).__name__}: {e2}")

        return out

    def discover_tokens(
        self,
        style: str = "portfolio", # canonical default for balances.py inputs
        chains: Optional[Iterable[str]] = None
    ) -> List[str]:
        """
        style='portfolio' -> base uses '0x...@base', others 'chain:0x...'
        style='prefix' -> 'chain:0x...'
        style='suffix' -> '0x...@chain'
        style='pairs' -> returns [('chain','0x...'), ...]
        """
        pairs = self.discover_tokens_pairs(chains=chains)
        if style == "pairs":
            return pairs # type: ignore[return-value]

        if style == "prefix":
            annotated = [f"{ch}:{addr}" for ch, addr in pairs]
        elif style == "suffix":
            annotated = [f"{addr}@{ch}" for ch, addr in pairs]
        else: # "portfolio" canonical for balances.py
            out: List[str] = []
            for ch, addr in pairs:
                ch_l = (ch or "").lower()
                out.append(f"{addr}@base" if ch_l == "base" else f"{ch_l}:{addr}")
            annotated = out

        # de-dupe while preserving discovery order
        return list(dict.fromkeys(annotated))

    # -------- passthrough annotate utility
    @staticmethod
    def annotate(tokens: Iterable[Any], default_chain: str = "ethereum") -> List[Tuple[str,str]]:
        return annotate_tokens(tokens, default_chain=default_chain)

    # -------- quoting / execution / send
    def quote(self, src_chain: str, dst_chain: str, from_token: str, to_token: str, amount: float, slippage: float=0.003) -> QuotePreview:
        from_token = _normalize_addr(from_token); to_token = _normalize_addr(to_token)
        w3 = _w3(src_chain); from_dec = _decimals(w3, from_token)
        params = {
            "fromChain": CHAINS[src_chain]["id"], "toChain": CHAINS[dst_chain]["id"],
            "fromToken": from_token, "toToken": to_token,
            "fromAmount": str(_to_wei(amount, from_dec)), "fromAddress": self.acct.address,
            "slippage": slippage,
        }
        q = _lifi_quote(params); est, act = q["estimate"], q["action"]
        gas_usd = None
        try:
            if est.get("gasCosts"): gas_usd = sum(float(x.get("amountUSD") or 0) for x in est["gasCosts"])
        except Exception: pass
        return QuotePreview(
            from_chain=act["fromChainId"], to_chain=act["toChainId"],
            from_symbol=act["fromToken"]["symbol"], to_symbol=act["toToken"]["symbol"],
            from_amount=_from_wei(int(est["fromAmount"]), act["fromToken"]["decimals"]),
            to_amount=_from_wei(int(est["toAmount"]), act["toToken"]["decimals"]),
            to_amount_min=_from_wei(int(est.get("toAmountMin", est["toAmount"])), act["toToken"]["decimals"]),
            gas_usd=gas_usd, tx_request=q.get("transactionRequest") or {},
        )

    def execute(self, src_chain: str, dst_chain: str, from_token: str, to_token: str, amount: float, slippage: float=0.003, wait: bool=False) -> Dict[str, Any]:
        w3 = _w3(src_chain)
        from_token = _normalize_addr(from_token); to_token = _normalize_addr(to_token)
        pv = self.quote(src_chain, dst_chain, from_token, to_token, amount, slippage)
        txreq = pv.tx_request
        if not txreq: raise RuntimeError("LI.FI returned no transactionRequest to execute.")

        if from_token.lower() != NATIVE:
            needed = _to_wei(amount, _decimals(w3, from_token))
            spender = Web3.to_checksum_address(txreq["to"])
            current = _erc20(w3, from_token).functions.allowance(self.acct.address, spender).call()
            if current < needed:
                tx = _erc20(w3, from_token).functions.approve(spender, needed).build_transaction({
                    "from": self.acct.address, "nonce": w3.eth.get_transaction_count(self.acct.address),
                    "chainId": w3.eth.chain_id, **_eip1559_fees(w3),
                })
                tx["gas"] = int(w3.eth.estimate_gas(tx) * 1.1)
                signed = self.acct.sign_transaction(tx)
                ah = w3.eth.send_raw_transaction(signed.rawTransaction).hex()
                w3.eth.wait_for_transaction_receipt(ah, timeout=240)

        if "to" not in txreq or "data" not in txreq: raise RuntimeError("Invalid LI.FI transactionRequest.")
        to = Web3.to_checksum_address(txreq["to"])
        value = txreq.get("value", 0); value = int(value,16) if isinstance(value,str) and value.startswith("0x") else value
        tx = {
            "to": to, "from": self.acct.address, "data": txreq["data"], "value": value,
            "chainId": w3.eth.chain_id, "nonce": w3.eth.get_transaction_count(self.acct.address),
            **_eip1559_fees(w3),
        }
        gl = txreq.get("gasLimit")
        tx["gas"] = int(gl,16) if isinstance(gl,str) and gl.startswith("0x") else (int(w3.eth.estimate_gas(tx) * 1.12) if not gl else int(gl))
        signed = self.acct.sign_transaction(tx)
        txh = w3.eth.send_raw_transaction(signed.rawTransaction).hex()
        out: Dict[str,Any] = {"preview": pv.__dict__, "txHash": txh}
        if pv.from_chain != pv.to_chain and wait:
            while True:
                try:
                    s = _lifi_status(txh); out["bridgeStatus"] = s
                    if s.get("status") in ("DONE","FAILED"): break
                except Exception: pass
                time.sleep(20)
        return out

    def send(self, chain: str, token: str, to_address: str, amount: float) -> str:
        w3 = _w3(chain); token = _normalize_addr(token); to = Web3.to_checksum_address(to_address)
        if token.lower() == NATIVE:
            tx = {
                "to": to, "from": self.acct.address, "value": _to_wei(amount, 18),
                "chainId": w3.eth.chain_id, "nonce": w3.eth.get_transaction_count(self.acct.address),
                **_eip1559_fees(w3),
            }
            tx["gas"] = int(w3.eth.estimate_gas(tx) * 1.1)
            signed = self.acct.sign_transaction(tx)
            return w3.eth.send_raw_transaction(signed.rawTransaction).hex()
        c = _erc20(w3, token); dec = _decimals(w3, token)
        tx = c.functions.transfer(to, _to_wei(amount, dec)).build_transaction({
            "from": self.acct.address, "nonce": w3.eth.get_transaction_count(self.acct.address),
            "chainId": w3.eth.chain_id, **_eip1559_fees(w3),
        })
        tx["gas"] = int(w3.eth.estimate_gas(tx) * 1.1)
        signed = self.acct.sign_transaction(tx)
        return w3.eth.send_raw_transaction(signed.rawTransaction).hex()


# ======== Live send helpers (appended safely) ========
try:
    UltraSwapBridge
except NameError:
    pass
else:
    import os
    from web3 import Web3

    def _rb__w3(self, chain: str):
        url = self._alchemy_url(chain)
        if not url:
            raise RuntimeError(f"No RPC URL configured for chain '{chain}'")
        try:
            provider = Web3.HTTPProvider(url, request_kwargs={"timeout": float(os.getenv("ALCHEMY_TIMEOUT_SEC", "10"))})
        except TypeError:
            provider = Web3.HTTPProvider(url)
        return Web3(provider)

    def _rb__fee_fields(self, w3: "Web3", max_priority_gwei=None, max_fee_gwei=None):
        # Prefer EIP-1559; fallback to legacy gasPrice
        try:
            pending = w3.eth.get_block("pending")
            base = pending.get("baseFeePerGas")
            if base is not None:
                # defaults: tip 2 gwei, maxFee = 2*base + tip
                tip = int(Web3.to_wei(max_priority_gwei if max_priority_gwei is not None else 2, "gwei"))
                max_fee = int(Web3.to_wei(max_fee_gwei, "gwei")) if max_fee_gwei is not None else (2*int(base) + tip)
                return {"maxPriorityFeePerGas": tip, "maxFeePerGas": max_fee}
        except Exception:
            pass
        # legacy
        try:
            return {"gasPrice": w3.eth.gas_price}
        except Exception:
            # last resort constant 5 gwei
            return {"gasPrice": Web3.to_wei(5, "gwei")}

    def _rb__ensure_live(self):
        if str(os.getenv("WALLET_LIVE", "0")).lower() not in ("1", "true", "yes", "y"):
            raise RuntimeError("Live transactions disabled. Set WALLET_LIVE=1 to enable broadcasting.")

    def send_native(self, chain: str, to: str, value: int, *, gas: int=None, gas_limit: int=None, max_priority_gwei=None, max_fee_gwei=None, nonce: int=None):
        """
        Send native currency (e.g., ETH) on `chain` to `to` with `value` (wei).
        Requires WALLET_LIVE=1 in environment.
        Returns tx hash hex string.
        """
        self._rb__ensure_live()
        w3 = self._rb__w3(chain)
        to_cs = Web3.to_checksum_address(to)
        sender = self.acct.address
        tx = {
            "chainId": w3.eth.chain_id,
            "nonce": w3.eth.get_transaction_count(sender, "pending") if nonce is None else int(nonce),
            "to": to_cs,
            "value": int(value),
            "gas": int(gas or gas_limit or 21000),
        }
        tx.update(self._rb__fee_fields(w3, max_priority_gwei, max_fee_gwei))
        signed = self.acct.sign_transaction(tx)
        txh = w3.eth.send_raw_transaction(signed.rawTransaction)
        return w3.to_hex(txh)

    def send_erc20(self, chain: str, token: str, to: str, amount: int, *, gas: int=None, gas_limit: int=None, max_priority_gwei=None, max_fee_gwei=None, nonce: int=None):
        """
        Send ERC-20 `amount` on `chain` from wallet to `to`.
        Requires WALLET_LIVE=1 in environment.
        Returns tx hash hex string.
        """
        self._rb__ensure_live()
        w3 = self._rb__w3(chain)
        token_cs = Web3.to_checksum_address(token)
        to_cs = Web3.to_checksum_address(to)
        sender = self.acct.address

        abi = [{
            "name": "transfer",
            "type": "function",
            "stateMutability": "nonpayable",
            "inputs": [{"name": "to", "type": "address"}, {"name": "value", "type": "uint256"}],
            "outputs": [{"name": "", "type": "bool"}],
        }]
        c = w3.eth.contract(address=token_cs, abi=abi)

        # base tx fields (EIP-1559 if available)
        base = {
            "chainId": w3.eth.chain_id,
            "from": sender,
            "nonce": w3.eth.get_transaction_count(sender, "pending") if nonce is None else int(nonce),
            "to": token_cs,
            "value": 0,
        }
        base.update(self._rb__fee_fields(w3, max_priority_gwei, max_fee_gwei))

        # estimate gas, then build, sign, send
        try:
            est = c.functions.transfer(to_cs, int(amount)).estimate_gas({"from": sender})
        except Exception:
            est = 100000  # conservative fallback
        gas_final = int(gas or gas_limit or int(est * 1.2))

        tx = c.functions.transfer(to_cs, int(amount)).build_transaction({**base, "gas": gas_final})
        signed = self.acct.sign_transaction(tx)
        txh = w3.eth.send_raw_transaction(signed.rawTransaction)
        return w3.to_hex(txh)

    # Attach helpers if not already present
    if not hasattr(UltraSwapBridge, "_rb__w3"):
        UltraSwapBridge._rb__w3 = _rb__w3
    if not hasattr(UltraSwapBridge, "_rb__fee_fields"):
        UltraSwapBridge._rb__fee_fields = _rb__fee_fields
    if not hasattr(UltraSwapBridge, "_rb__ensure_live"):
        UltraSwapBridge._rb__ensure_live = _rb__ensure_live
    UltraSwapBridge.send_native = send_native
    UltraSwapBridge.send_erc20 = send_erc20
# ======== End live send helpers ========


# ======== Live send helpers v2 (web3 v5/v6 compatible) ========
try:
    UltraSwapBridge
except NameError:
    pass
else:
    import os
    from web3 import Web3

    def _rb__raw_tx_bytes(signed):
        """Return raw tx bytes across web3 versions."""
        for attr in ("rawTransaction", "raw_transaction", "raw"):
            raw = getattr(signed, attr, None)
            if raw is not None:
                return raw
        # Some versions may already return bytes/HexBytes
        return signed

    def _rb__w3(self, chain: str):
        url = self._alchemy_url(chain)
        if not url:
            raise RuntimeError(f"No RPC URL configured for chain '{chain}'")
        try:
            provider = Web3.HTTPProvider(url, request_kwargs={"timeout": float(os.getenv("ALCHEMY_TIMEOUT_SEC", "10"))})
        except TypeError:
            provider = Web3.HTTPProvider(url)
        return Web3(provider)

    def _rb__fee_fields(self, w3: "Web3", max_priority_gwei=None, max_fee_gwei=None):
        try:
            pending = w3.eth.get_block("pending")
            base = pending.get("baseFeePerGas")
            if base is not None:
                tip = int(Web3.to_wei(max_priority_gwei if max_priority_gwei is not None else 2, "gwei"))
                max_fee = int(Web3.to_wei(max_fee_gwei, "gwei")) if max_fee_gwei is not None else (2*int(base) + tip)
                return {"maxPriorityFeePerGas": tip, "maxFeePerGas": max_fee}
        except Exception:
            pass
        try:
            return {"gasPrice": w3.eth.gas_price}
        except Exception:
            return {"gasPrice": Web3.to_wei(5, "gwei")}

    def _rb__ensure_live(self):
        if str(os.getenv("WALLET_LIVE", "0")).lower() not in ("1", "true", "yes", "y"):
            raise RuntimeError("Live transactions disabled. Set WALLET_LIVE=1 to enable broadcasting.")

    def send_native(self, chain: str, to: str, value: int, *, gas: int=None, gas_limit: int=None, max_priority_gwei=None, max_fee_gwei=None, nonce: int=None):
        self._rb__ensure_live()
        w3 = self._rb__w3(chain)
        to_cs = Web3.to_checksum_address(to)
        sender = self.acct.address
        tx = {
            "chainId": w3.eth.chain_id,
            "nonce": w3.eth.get_transaction_count(sender, "pending") if nonce is None else int(nonce),
            "to": to_cs,
            "value": int(value),
            "gas": int(gas or gas_limit or 21000),
        }
        tx.update(self._rb__fee_fields(w3, max_priority_gwei, max_fee_gwei))
        signed = self.acct.sign_transaction(tx)
        raw = _rb__raw_tx_bytes(signed)
        txh = w3.eth.send_raw_transaction(raw)
        return w3.to_hex(txh)

    def send_erc20(self, chain: str, token: str, to: str, amount: int, *, gas: int=None, gas_limit: int=None, max_priority_gwei=None, max_fee_gwei=None, nonce: int=None):
        self._rb__ensure_live()
        w3 = self._rb__w3(chain)
        token_cs = Web3.to_checksum_address(token)
        to_cs = Web3.to_checksum_address(to)
        sender = self.acct.address
        abi = [{
            "name": "transfer",
            "type": "function",
            "stateMutability": "nonpayable",
            "inputs": [{"name": "to", "type": "address"}, {"name": "value", "type": "uint256"}],
            "outputs": [{"name": "", "type": "bool"}],
        }]
        c = w3.eth.contract(address=token_cs, abi=abi)
        base = {
            "chainId": w3.eth.chain_id,
            "from": sender,
            "nonce": w3.eth.get_transaction_count(sender, "pending") if nonce is None else int(nonce),
            "to": token_cs,
            "value": 0,
        }
        base.update(self._rb__fee_fields(w3, max_priority_gwei, max_fee_gwei))
        try:
            est = c.functions.transfer(to_cs, int(amount)).estimate_gas({"from": sender})
        except Exception:
            est = 100000
        gas_final = int(gas or gas_limit or int(est * 1.2))
        tx = c.functions.transfer(to_cs, int(amount)).build_transaction({**base, "gas": gas_final})
        signed = self.acct.sign_transaction(tx)
        raw = _rb__raw_tx_bytes(signed)
        txh = w3.eth.send_raw_transaction(raw)
        return w3.to_hex(txh)

    # Rebind latest helpers
    UltraSwapBridge._rb__w3 = _rb__w3
    UltraSwapBridge._rb__fee_fields = _rb__fee_fields
    UltraSwapBridge._rb__ensure_live = _rb__ensure_live
    UltraSwapBridge.send_native = send_native
    UltraSwapBridge.send_erc20 = send_erc20
# ======== End live send helpers v2 ========


# ======== Live send helpers v3 (ERC-20 build_transaction compat: no explicit 'to') ========
from web3 import Web3

def send_erc20(self, chain: str, token: str, to: str, amount: int, *, gas: int=None, gas_limit: int=None, max_priority_gwei=None, max_fee_gwei=None, nonce: int=None):
    """
    ERC-20 transfer compatible with web3.py v5/v6:
    - Do NOT set 'to' in build_transaction; function injects contract address.
    - EIP-1559 fee fields if available; fallback to gasPrice.
    """
    self._rb__ensure_live()
    w3 = self._rb__w3(chain)
    token_cs = Web3.to_checksum_address(token)
    to_cs = Web3.to_checksum_address(to)
    sender = self.acct.address

    abi = [{
        "name": "transfer",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [{"name": "to", "type": "address"}, {"name": "value", "type": "uint256"}],
        "outputs": [{"name": "", "type": "bool"}],
    }]
    c = w3.eth.contract(address=token_cs, abi=abi)

    base = {
        "chainId": w3.eth.chain_id,
        "from": sender,
        "nonce": w3.eth.get_transaction_count(sender, "pending") if nonce is None else int(nonce),
        "value": 0,
    }
    base.update(self._rb__fee_fields(w3, max_priority_gwei, max_fee_gwei))

    try:
        est = c.functions.transfer(to_cs, int(amount)).estimate_gas({"from": sender})
    except Exception:
        est = 100000
    gas_final = int(gas or gas_limit or int(est * 1.2))

    tx = c.functions.transfer(to_cs, int(amount)).build_transaction({**base, "gas": gas_final})
    signed = self.acct.sign_transaction(tx)
    raw = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None) or getattr(signed, "raw", None) or signed
    txh = w3.eth.send_raw_transaction(raw)
    return w3.to_hex(txh)

UltraSwapBridge.send_erc20 = send_erc20
# ======== End live send helpers v3 ========




def _rb__hexint(v, default=0):
    if v is None: return default
    if isinstance(v, int): return v
    try:
        s=str(v).strip().lower()
        return int(s,16) if s.startswith('0x') else int(s)
    except Exception:
        return default
# ======== 0x swap helpers v1 (multi-chain) ========
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from web3 import Web3
import os, json

def _rb__0x_base_url(self, chain: str) -> str:
    ch = (chain or "").lower()
    m = {
        "ethereum": "https://api.0x.org",
        "base": "https://base.api.0x.org",
        "arbitrum": "https://arbitrum.api.0x.org",
        "optimism": "https://optimism.api.0x.org",
        "polygon": "https://polygon.api.0x.org",
    }
    return m.get(ch, "")

def _rb__http_get_json(self, url: str, headers: dict | None = None, timeout_sec: float | None = None):
    hdrs = {"Accept": "application/json", "User-Agent": os.getenv("HTTP_UA", "Mozilla/5.0 WalletCLI/1.0")}
    if headers: hdrs.update(headers)
    req = Request(url, headers=hdrs, method="GET")
    to = float(os.getenv("ZEROX_HTTP_TIMEOUT_SEC", "12")) if timeout_sec is None else float(timeout_sec)
    last_err = None
    for attempt in range(3):
        try:
            with urlopen(req, timeout=to) as r:
                data = r.read()
            return json.loads(data.decode("utf-8"))
        except HTTPError as e:
            body = e.read()
            msg = body.decode("utf-8", errors="ignore") if body else ""
            try:
                j = json.loads(msg) if msg else {}
            except Exception:
                j = {"raw": msg}
            code = j.get("code") or j.get("reason") or j.get("validationErrors") or j.get("message") or "HTTPError"
            # retry on 403/429 once or twice (CF or rate limit)
            if e.code in (403, 429) and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                last_err = (e.code, code, j); 
                continue
            raise RuntimeError(f"0x {e.code} {code}: {j}") from None
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(0.75 * (attempt + 1))
                continue
            raise
def get_0x_quote(self, chain: str, sell_token: str, buy_token: str, sell_amount_wei: int, slippage: float = 0.01, taker: str | None = None, api_key: str | None = None) -> dict:
    base = _rb__0x_base_url(self, chain)
    if not base:
        raise RuntimeError(f"0x not supported for chain '{chain}'")
    taker_addr = taker or self.acct.address
    q = {
        "sellToken": sell_token,
        "buyToken": buy_token,
        "sellAmount": str(int(sell_amount_wei)),
        "takerAddress": taker_addr,
        "slippagePercentage": f"{slippage:.6f}",
    }
    url = f"{base}/swap/v1/quote?{urlencode(q)}"
    headers = {}
    ak = api_key or os.getenv("ZEROX_API_KEY") or ""
    if ak:
        headers["0x-api-key"] = ak
    return _rb__http_get_json(self, url, headers=headers)

_ERC20_APPROVE_ABI = [
    {"name":"approve","type":"function","stateMutability":"nonpayable","inputs":[{"name":"spender","type":"address"},{"name":"value","type":"uint256"}],"outputs":[{"name":"","type":"bool"}]},
    {"name":"allowance","type":"function","stateMutability":"view","inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],"outputs":[{"name":"","type":"uint256"}]},
    {"name":"decimals","type":"function","stateMutability":"view","inputs":[],"outputs":[{"name":"","type":"uint8"}]},
    {"name":"symbol","type":"function","stateMutability":"view","inputs":[],"outputs":[{"name":"","type":"string"}]},
]

def erc20_allowance(self, chain: str, token: str, owner: str, spender: str) -> int:
    w3 = self._rb__w3(chain)
    c = w3.eth.contract(address=Web3.to_checksum_address(token), abi=_ERC20_APPROVE_ABI)
    try:
        return int(c.functions.allowance(Web3.to_checksum_address(owner), Web3.to_checksum_address(spender)).call())
    except Exception:
        return 0

def approve_erc20(self, chain: str, token: str, spender: str, amount: int, *, gas: int | None = None, max_priority_gwei=None, max_fee_gwei=None, nonce: int | None = None):
    self._rb__ensure_live()
    w3 = self._rb__w3(chain)
    token_cs = Web3.to_checksum_address(token)
    spend_cs = Web3.to_checksum_address(spender)
    sender = self.acct.address
    c = w3.eth.contract(address=token_cs, abi=_ERC20_APPROVE_ABI)
    base = {
        "chainId": w3.eth.chain_id,
        "from": sender,
        "nonce": w3.eth.get_transaction_count(sender, "pending") if nonce is None else int(nonce),
        "value": 0,
    }
    base.update(self._rb__fee_fields(w3, max_priority_gwei, max_fee_gwei))
    try:
        est = c.functions.approve(spend_cs, int(amount)).estimate_gas({"from": sender})
    except Exception:
        est = 60000
    gas_final = int(gas or int(est * 1.2))
    tx = c.functions.approve(spend_cs, int(amount)).build_transaction({**base, "gas": gas_final})
    signed = self.acct.sign_transaction(tx)
    raw = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None) or getattr(signed, "raw", None) or signed
    txh = w3.eth.send_raw_transaction(raw)
    return w3.to_hex(txh)

def send_swap_via_0x(self, chain: str, quote: dict, *, gas: int | None = None, max_priority_gwei=None, max_fee_gwei=None, nonce: int | None = None):
    self._rb__ensure_live()
    w3 = self._rb__w3(chain)
    sender = self.acct.address
    to = Web3.to_checksum_address(quote["to"])
    value = _rb__hexint(quote.get("value"), 0)
    data = quote.get("data") or "0x"

    # Try on-chain gas estimation first; fall back to quote hints or a conservative default
    try:
        gas_est = int(w3.eth.estimate_gas({"from": sender, "to": to, "value": value, "data": data}))
    except Exception:
        gas_hint = _rb__hexint(quote.get("gas") or quote.get("estimatedGas"), 0)
        gas_est = gas_hint or 250000

    gas_final = int(gas if gas is not None else int(gas_est * 1.2))

    tx = {
        "chainId": w3.eth.chain_id,
        "nonce": w3.eth.get_transaction_count(sender, "pending") if nonce is None else int(nonce),
        "from": sender,
        "to": to,
        "value": value,
        "data": data,
        "gas": gas_final,
    }
    tx.update(self._rb__fee_fields(w3, max_priority_gwei, max_fee_gwei))

    signed = self.acct.sign_transaction(tx)
    raw = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None) or getattr(signed, "raw", None) or signed
    txh = w3.eth.send_raw_transaction(raw)
    return w3.to_hex(txh)

# bind helpers
UltraSwapBridge._rb__0x_base_url = _rb__0x_base_url
UltraSwapBridge._rb__http_get_json = _rb__http_get_json
UltraSwapBridge.get_0x_quote = get_0x_quote
UltraSwapBridge._ERC20_APPROVE_ABI = _ERC20_APPROVE_ABI
UltraSwapBridge.erc20_allowance = erc20_allowance
UltraSwapBridge.approve_erc20 = approve_erc20
UltraSwapBridge.send_swap_via_0x = send_swap_via_0x
# ======== End 0x swap helpers v1 ========

# ======== 1inch v6 helpers (multi-chain) ========
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import os, json
from web3 import Web3

_ONEINCH_NATIVE = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
_ONEINCH_CHAIN_IDS = {
    "ethereum": 1,
    "base": 8453,
    "arbitrum": 42161,
    "optimism": 10,
    "polygon": 137,
}

def _rb__1inch_base(self, chain: str) -> tuple[str,int]:
    ch = (chain or "").lower()
    cid = _ONEINCH_CHAIN_IDS.get(ch)
    if not cid: return "", 0
    return f"https://api.1inch.dev/swap/v6.0/{cid}", cid

def _rb__http_get_json_auth(self, url: str, headers: dict | None = None, timeout_sec: float | None = None):
    hdrs = {"Accept": "application/json", "User-Agent": os.getenv("HTTP_UA", "Mozilla/5.0 WalletCLI/1.0")}
    if headers: hdrs.update(headers)
    req = Request(url, headers=hdrs, method="GET")
    to = float(os.getenv("ONEINCH_HTTP_TIMEOUT_SEC", "12")) if timeout_sec is None else float(timeout_sec)
    last = None
    for attempt in range(3):
        try:
            with urlopen(req, timeout=to) as r:
                data = r.read()
            return json.loads(data.decode("utf-8"))
        except HTTPError as e:
            body = e.read()
            msg = body.decode("utf-8", errors="ignore") if body else ""
            try:
                j = json.loads(msg) if msg else {}
            except Exception:
                j = {"raw": msg}
            # 1inch often sits behind Cloudflare (1010); retry 403/429
            if e.code in (403, 429) and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                last = (e.code, j)
                continue
            code = j.get("description") or j.get("error") or j.get("message") or "HTTPError"
            raise RuntimeError(f"1inch {e.code} {code}: {j}") from None
        except Exception as e:
            last = e
            if attempt < 2:
                time.sleep(0.75 * (attempt + 1))
                continue
            raise
def get_1inch_swap_tx(self, chain: str, sell_token: str, buy_token: str, sell_amount_wei: int, slippage: float = 0.01, taker: str | None = None, api_key: str | None = None) -> dict:
    """
    Ask 1inch /swap for an executable tx. Returns a dict normalized like 0x:
      { "to": <addr>, "data": <0x...>, "value": <int>, "estimatedGas": <int>,
        "buyAmount": <int>, "price": <str> (if provided) }
    """
    base, cid = _rb__1inch_base(self, chain)
    if not base:
        raise RuntimeError(f"1inch not supported for '{chain}'")
    ak = api_key or os.getenv("ONEINCH_API_KEY")
    if not ak:
        raise RuntimeError("ONEINCH_API_KEY is not set")
    taker_addr = taker or self.acct.address

    # native token mapping
    s_id = _ONEINCH_NATIVE if (sell_token.lower() in ("eth","native")) else Web3.to_checksum_address(sell_token)
    b_id = _ONEINCH_NATIVE if (buy_token.lower()  in ("eth","native")) else Web3.to_checksum_address(buy_token)

    q = {
        "src": s_id,
        "dst": b_id,
        "amount": str(int(sell_amount_wei)),
        "from": taker_addr,
        "slippage": f"{slippage*100:.4f}",  # 1inch expects percent
        # you could add: "disableEstimate": "true"  (but we want gas in response)
    }
    url = f"{base}/swap?{urlencode(q)}"
    hdrs = {"Authorization": f"Bearer {ak}"}
    j = self._rb__http_get_json_auth(url, headers=hdrs)

    tx = j.get("tx") or {}
    # Normalize to the same shape used by 0x sender
    out = {
        "to": tx.get("to") or j.get("to"),
        "data": tx.get("data") or j.get("data"),
        "value": int(tx.get("value") or j.get("value") or 0),
        "estimatedGas": int(j.get("gas") or tx.get("gas") or 0),
        "buyAmount": int(j.get("dstAmount") or j.get("toAmount") or 0),
        "price": j.get("price"),
    }
    if not out["to"] or not out["data"]:
        raise RuntimeError(f"1inch swap response missing tx fields: {j}")
    return out

# bind
UltraSwapBridge.get_1inch_swap_tx = get_1inch_swap_tx
UltraSwapBridge._rb__1inch_base = _rb__1inch_base
UltraSwapBridge._rb__http_get_json_auth = _rb__http_get_json_auth
# ======== End 1inch v6 helpers ========

# ======== OpenOcean v4 helpers (multi-chain, no API key) ========
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import os, json
from web3 import Web3
import time

_OO_NATIVE = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"  # OpenOcean accepts this for native
_OO_CHAINS = {
    "ethereum": "eth",
    "base": "base",
    "arbitrum": "arbitrum",
    "optimism": "optimism",
    "polygon": "polygon",
}

def _rb__http_get_json_ua(self, url: str, headers: dict | None = None, timeout_sec: float | None = None, retries: int = 2):
    # Same as other helpers but with UA + small retry (CF/rate-limit friendly)
    hdrs = {"Accept": "application/json", "User-Agent": os.getenv("HTTP_UA", "Mozilla/5.0 WalletCLI/1.0")}
    if headers: hdrs.update(headers)
    req = Request(url, headers=hdrs, method="GET")
    to = float(os.getenv("OO_HTTP_TIMEOUT_SEC", "12")) if timeout_sec is None else float(timeout_sec)
    last = None
    for a in range(retries+1):
        try:
            with urlopen(req, timeout=to) as r:
                data = r.read()
            return json.loads(data.decode("utf-8"))
        except HTTPError as e:
            body = e.read()
            msg = body.decode("utf-8", errors="ignore") if body else ""
            try:
                j = json.loads(msg) if msg else {}
            except Exception:
                j = {"raw": msg}
            # retry on 403/429 transient
            if e.code in (403, 429) and a < retries:
                time.sleep(1.2*(a+1))
                last = (e.code, j)
                continue
            code = j.get("message") or j.get("error") or j.get("description") or "HTTPError"
            raise RuntimeError(f"OpenOcean {e.code} {code}: {j}") from None
        except Exception as e:
            last = e
            if a < retries:
                time.sleep(0.6*(a+1))
                continue
            raise

def get_openocean_swap_tx(self, chain: str, sell_token: str, buy_token: str, sell_amount_wei: int, slippage: float = 0.01, taker: str | None = None) -> dict:
    """
    Ask OpenOcean /v4/{chain}/swap for an executable tx.
    Normalized return: {to,data,value(int),estimatedGas(int),buyAmount(int)}
    """
    ch = (chain or "").lower()
    oo_ch = _OO_CHAINS.get(ch)
    if not oo_ch:
        raise RuntimeError(f"OpenOcean not supported for '{chain}'")
    taker_addr = taker or self.acct.address

    s_id = _OO_NATIVE if (sell_token.lower() in ("eth","native")) else Web3.to_checksum_address(sell_token)
    b_id = _OO_NATIVE if (buy_token.lower()  in ("eth","native")) else Web3.to_checksum_address(buy_token)

    # OpenOcean expects slippage as percent (e.g., 1 for 1%)
    slip_pct = f"{slippage*100:.4f}".rstrip('0').rstrip('.') if slippage else "1"

    q = {
        "inTokenAddress": s_id,
        "outTokenAddress": b_id,
        "amount": str(int(sell_amount_wei)),
        "slippage": slip_pct,
        "account": taker_addr,
        # "gasPrice": "",          # optional
        # "onlyDexId": "",         # optional targeting
        # "disabledDexIds": "",    # optional blacklist
    }
    url = f"https://open-api.openocean.finance/v4/{oo_ch}/swap?{urlencode(q)}"
    j = _rb__http_get_json_ua(self, url)

    # Typical shape: {"data":{"to":"0x..","data":"0x..","value":"123","gasPrice":"..","gasLimit":".."},"outAmount":"..",...}
    tx = (j.get("data") or {})
    to_addr = tx.get("to") or j.get("to")
    data_hex = tx.get("data") or j.get("data") or "0x"
    value = int(tx.get("value") or j.get("value") or 0)
    est_gas = int(tx.get("gasLimit") or j.get("gasLimit") or j.get("estimatedGas") or 0)
    out_amt = int(j.get("outAmount") or j.get("toAmount") or 0)
    if not to_addr or not data_hex:
        raise RuntimeError(f"OpenOcean swap response missing tx fields: {j}")
    return {"to": to_addr, "data": data_hex, "value": value, "estimatedGas": est_gas, "buyAmount": out_amt}

# bind
UltraSwapBridge.get_openocean_swap_tx = get_openocean_swap_tx
# ======== End OpenOcean v4 helpers ========


# ----------------------- Local Uniswap v3 (Arbitrum) -----------------------
# Minimal, module-level monkey-patch so we don't disturb class layout.

try:
    from web3 import Web3
except Exception:
    Web3 = None  # will raise at runtime if missing

# Canonical addresses (Arbitrum One)
_V3_ADDR = {
    "arbitrum": {
        # Uniswap V3 SwapRouter02
        "router": "0x68b3465833FB72A70eCDf485E0e4C7bD8665Fc45",
        # QuoterV2
        "quoter": "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
        # WETH
        "weth":   "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
    }
}

# Minimal ABIs
_V3_QUOTER_ABI = [
    {
        "inputs":[
            {"internalType":"address","name":"tokenIn","type":"address"},
            {"internalType":"address","name":"tokenOut","type":"address"},
            {"internalType":"uint24","name":"fee","type":"uint24"},
            {"internalType":"uint256","name":"amountIn","type":"uint256"},
            {"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"}
        ],
        "name":"quoteExactInputSingle",
        "outputs":[
            {"internalType":"uint256","name":"amountOut","type":"uint256"},
            {"internalType":"uint160","name":"","type":"uint160"},
            {"internalType":"uint32","name":"","type":"uint32"},
            {"internalType":"uint256","name":"","type":"uint256"}
        ],
        "stateMutability":"nonpayable",
        "type":"function"
    },
    {
        "inputs":[
            {"internalType":"bytes","name":"path","type":"bytes"},
            {"internalType":"uint256","name":"amountIn","type":"uint256"}
        ],
        "name":"quoteExactInput",
        "outputs":[
            {"internalType":"uint256","name":"amountOut","type":"uint256"},
            {"internalType":"uint160","name":"","type":"uint160"},
            {"internalType":"uint32","name":"","type":"uint32"},
            {"internalType":"uint256","name":"","type":"uint256"}
        ],
        "stateMutability":"nonpayable",
        "type":"function"
    }
]

_V3_ROUTER_ABI = [
    {
        "inputs":[
            {
                "components":[
                    {"internalType":"address","name":"tokenIn","type":"address"},
                    {"internalType":"address","name":"tokenOut","type":"address"},
                    {"internalType":"uint24","name":"fee","type":"uint24"},
                    {"internalType":"address","name":"recipient","type":"address"},
                    {"internalType":"uint256","name":"deadline","type":"uint256"},
                    {"internalType":"uint256","name":"amountIn","type":"uint256"},
                    {"internalType":"uint256","name":"amountOutMinimum","type":"uint256"},
                    {"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"}
                ],
                "internalType":"struct ISwapRouter.ExactInputSingleParams",
                "name":"params",
                "type":"tuple"
            }
        ],
        "name":"exactInputSingle",
        "outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],
        "stateMutability":"payable",
        "type":"function"
    },
    {
        "inputs":[
            {"internalType":"bytes","name":"path","type":"bytes"},
            {"internalType":"address","name":"recipient","type":"address"},
            {"internalType":"uint256","name":"deadline","type":"uint256"},
            {"internalType":"uint256","name":"amountIn","type":"uint256"},
            {"internalType":"uint256","name":"amountOutMinimum","type":"uint256"}
        ],
        "name":"exactInput",
        "outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],
        "stateMutability":"payable",
        "type":"function"
    }
]

def _v3_encode_path(tokens, fees):
    # tokens: list of EIP-55 addresses (len = len(fees)+1)
    # fees: list of uint24
    assert len(tokens) == len(fees)+1
    b = b""
    for i, fee in enumerate(fees):
        b += bytes.fromhex(tokens[i][2:].zfill(40))
        b += int(fee).to_bytes(3, "big")
    b += bytes.fromhex(tokens[-1][2:].zfill(40))
    return "0x" + b.hex()

def _v3_best_direct(quoter, w3, t_in, t_out, amount):
    best = (0, None)  # (amountOut, fee)
    for fee in (500, 3000, 10000):
        try:
            out, *_ = quoter.functions.quoteExactInputSingle(
                t_in, t_out, fee, int(amount), 0
            ).call()
            if int(out) > int(best[0]):
                best = (int(out), fee)
        except Exception:
            pass
    return best  # e.g., (amountOut, 3000)

def _v3_best_via_weth(quoter, w3, t_in, t_out, weth, amount):
    best = (0, None, None)  # (out, fee_in, fee_out)
    fees = (500, 3000, 10000)
    for f1 in fees:
        for f2 in fees:
            try:
                path = _v3_encode_path([t_in, weth, t_out], [f1, f2])
                out, *_ = quoter.functions.quoteExactInput(path, int(amount)).call()
                if int(out) > int(best[0]):
                    best = (int(out), f1, f2)
            except Exception:
                pass
    return best

def _ultra__get_local_v3_swap_tx(self, chain, sell, buy, amount_in_raw, slippage=None, slippage_bps=50):

    # --- slippage normalization ---
    # Accepts: slippage in fraction (0.005 = 0.5%), percent (0.5 = 0.5%), or bps (50 = 0.5%).
    bps = None
    if slippage is not None:
        try:
            x = float(str(slippage).strip())
            if 0 <= x <= 1.0:
                # fraction -> bps
                bps = int(round(x * 10_000))
            elif 1.0 < x <= 100.0:
                # percent -> bps
                bps = int(round(x * 100))
            else:
                # assume bps
                bps = int(round(x))
        except Exception:
            bps = None
    if bps is None:
        bps = int(slippage_bps)
    # Clamp to sane bounds
    slippage_bps = max(1, min(5_000, bps))
    """
    Local Uniswap v3 pathfinder (Arbitrum only, direct or WETH hop).
    Returns dict compatible with aggregator outputs: to, data, value, allowanceTarget, sellToken, buyToken, buyAmount.
    """
    if Web3 is None:
        raise RuntimeError("web3 is required for LocalV3 swaps")

    ch = (chain or "").lower()
    if ch != "arbitrum":
        raise NotImplementedError("LocalV3 currently implemented for 'arbitrum' only")

    info = _V3_ADDR.get(ch) or {}
    router_addr = Web3.to_checksum_address(info["router"])
    quoter_addr = Web3.to_checksum_address(info["quoter"])
    weth_addr   = Web3.to_checksum_address(info["weth"])

    # Provider
    url = getattr(self, "_alchemy_url")(ch)
    if not url:
        raise RuntimeError(f"No RPC URL for chain={ch}")
    w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 20}))

    # Resolve tokens
    def _norm(a):
        if a is None:
            return None
        s = str(a).strip()
        if s.lower() in ("eth", "native"):
            return "ETH"
        return Web3.to_checksum_address(s)

    token_in  = _norm(sell)
    token_out = _norm(buy)
    if token_in is None or token_out is None:
        raise ValueError("sell/buy must be provided")

    is_native_in  = token_in == "ETH"
    is_native_out = token_out == "ETH"

    # v3 swaps use WETH; map ETH to WETH for path
    t_in_path  = weth_addr if is_native_in  else token_in
    t_out_path = weth_addr if is_native_out else token_out

    quoter = w3.eth.contract(address=quoter_addr, abi=_V3_QUOTER_ABI)
    router = w3.eth.contract(address=router_addr, abi=_V3_ROUTER_ABI)

    amt_in = int(amount_in_raw)

    # 1) direct
    best_out, best_fee = _v3_best_direct(quoter, w3, t_in_path, t_out_path, amt_in)
    route = None
    if best_out > 0:
        route = ("direct", best_fee, None, best_out)

    # 2) via WETH if direct not available and neither side is already WETH
    if route is None and (t_in_path != weth_addr and t_out_path != weth_addr):
        out2, f1, f2 = _v3_best_via_weth(quoter, w3, t_in_path, t_out_path, weth_addr, amt_in)
        if out2 > 0:
            route = ("via", (f1, f2), None, out2)

    if route is None:
        return {"__error__": "LocalV3: no quote (no pool direct or via WETH)"}

    # Compute minOut with slippage
    exp_out = int(route[3])
    min_out = max(0, int(exp_out * (10_000 - int(slippage_bps)) // 10_000))

    # Build tx data
    recipient = self.acct.address  # receive to sender by default
    deadline  = int(w3.eth.get_block("latest").timestamp) + 600

    if route[0] == "direct":
        fee = int(route[1])
        params = (
            t_in_path,
            t_out_path,
            fee,
            recipient,
            deadline,
            amt_in,
            min_out,
            0  # sqrtPriceLimitX96
        )
        data = router.encode_abi("exactInputSingle", args=[params])
        value = amt_in if is_native_in else 0
    else:
        f1, f2 = route[1]
        path = _v3_encode_path([t_in_path, weth_addr, t_out_path], [int(f1), int(f2)])
        data = router.encode_abi("exactInput", args=[path, recipient, deadline, amt_in, min_out])
        value = amt_in if is_native_in else 0

    # Allowance target is always the router for ERC-20 sells
    allowance_target = router.address

    return {
        "to": router.address,
        "data": data,
        "value": hex(int(value)),
        "allowanceTarget": allowance_target,
        "sellToken": token_in if token_in != "ETH" else weth_addr,
        "buyToken":  token_out if token_out  != "ETH" else weth_addr,
        "buyAmount": str(exp_out),
        "aggregator": "LocalV3",
    }

# Attach to class at import time (monkey-patch)
try:
    UltraSwapBridge.get_local_v3_swap_tx = _ultra__get_local_v3_swap_tx
except Exception:
    pass


# --- injected: ERC-20 read helpers (idempotent) ---
try:
    UltraSwapBridge
except NameError:
    pass
else:
    if not hasattr(UltraSwapBridge, "erc20_balance_of"):
        # We attach small helpers without touching the existing class body.
        def _rw__get_w3(self, chain: str):
            try:
                # prefer an existing helper if present
                return self._w3(chain)  # type: ignore[attr-defined]
            except Exception:
                from web3 import Web3
                url = self._alchemy_url(chain)
                return Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 20}))

        def erc20_balance_of(self, chain: str, token: str, owner: str) -> int:
            from web3 import Web3
            w3 = _rw__get_w3(self, chain)
            abi = [{"inputs":[{"name":"owner","type":"address"}],
                    "name":"balanceOf","outputs":[{"name":"","type":"uint256"}],
                    "stateMutability":"view","type":"function"}]
            c = w3.eth.contract(Web3.to_checksum_address(token), abi=abi)
            return int(c.functions.balanceOf(Web3.to_checksum_address(owner)).call())

        def erc20_decimals(self, chain: str, token: str) -> int:
            from web3 import Web3
            w3 = _rw__get_w3(self, chain)
            abi = [{"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],
                    "stateMutability":"view","type":"function"}]
            c = w3.eth.contract(Web3.to_checksum_address(token), abi=abi)
            try:
                return int(c.functions.decimals().call())
            except Exception:
                return 18

        def erc20_symbol(self, chain: str, token: str) -> str:
            from web3 import Web3
            w3 = _rw__get_w3(self, chain)
            abi = [{"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],
                    "stateMutability":"view","type":"function"}]
            c = w3.eth.contract(Web3.to_checksum_address(token), abi=abi)
            try:
                return str(c.functions.symbol().call())
            except Exception:
                return ""

        UltraSwapBridge.erc20_balance_of = erc20_balance_of   # type: ignore[attr-defined]
        UltraSwapBridge.erc20_decimals   = erc20_decimals     # type: ignore[attr-defined]
        UltraSwapBridge.erc20_symbol     = erc20_symbol       # type: ignore[attr-defined]
# --- end injected helpers ---

# === BEGIN: 0x preflight wrapper ===
try:
    _ULTRA_ORIG_SEND_SWAP_0X = UltraSwapBridge.send_swap_via_0x  # preserve original
except Exception:
    _ULTRA_ORIG_SEND_SWAP_0X = None

def _rb__preflight_swap(self, chain: str, quote: dict):
    """Dry-run the 0x tx via eth_call to surface the revert reason before sending."""
    url = getattr(self, "_alchemy_url", lambda ch: None)(chain)
    if not url:
        return False, "missing RPC url for chain", None
    try:
        w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 25}))
    except Exception as e:
        return False, f"rpc init failed: {e!r}", None

    to_raw   = quote.get("to")
    data     = quote.get("data")
    val_raw  = quote.get("value", 0)
    to_addr  = None
    try:
        to_addr = Web3.to_checksum_address(to_raw) if to_raw else None
    except Exception:
        pass
    if not (to_addr and data):
        return False, "quote incomplete (missing to/data)", None

    tx = {
        "from": getattr(self.acct, "address", None),
        "to":   to_addr,
        "data": data,
        "value": _rb__hexint(val_raw, 0),
    }

    # Optional: rough gas estimate (helps some nodes return revert early with reason)
    try:
        est = w3.eth.estimate_gas(tx)
        tx["gas"] = int(est)
    except Exception:
        # leave gas unset; call can still return revert reason
        pass

    # Call at latest state; if it reverts, surface the message
    try:
        _ = w3.eth.call(tx)
        return True, None, tx
    except Exception as e:
        # Try to pull a readable reason string
        reason = ""
        if getattr(e, "args", None):
            reason = str(e.args[0])
        else:
            reason = repr(e)
        return False, reason, tx

def _rb__wrapped_send_swap_via_0x(self, chain: str, quote: dict, *a, **kw):
    ok, reason, pre_tx = _rb__preflight_swap(self, chain, quote)
    if not ok:
        # classify common issues so the CLI can react helpfully
        if "insufficient" in (reason or "").lower() or "too little received" in (reason or "").lower():
            raise ValueError(f"preflight: likely slippage/minOut issue — {reason}")
        if "transfer" in (reason or "").lower():
            raise ValueError(f"preflight: token transfer failed (fee-on-transfer/restricted?) — {reason}")
        raise ValueError(f"preflight: revert — {reason}")
    # delegate to original sender (it already signs and broadcasts)
    if _ULTRA_ORIG_SEND_SWAP_0X is None:
        raise RuntimeError("original send_swap_via_0x missing; cannot delegate")
    return _ULTRA_ORIG_SEND_SWAP_0X(self, chain, quote, *a, **kw)

# bind wrapper
UltraSwapBridge.send_swap_via_0x = _rb__wrapped_send_swap_via_0x
# === END: 0x preflight wrapper ===

# === ULTRA_PATCH_0X_V1: 0x v1 quote + sender with preflight & auto-slippage ===
import os, json
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from web3 import Web3

def _ultra__0x_headers():
    h = {
        "Accept": "application/json",
        "User-Agent": "CoolCryptoUtilities/1.0"
    }
    api_key = os.getenv("ZEROX_API_KEY") or os.getenv("OX_API_KEY") or os.getenv("API_0X")
    if api_key:
        h["0x-api-key"] = api_key
    # no 0x-version header for v1
    return h

def _ultra__http_get_json(url: str, params: dict, timeout=25):
    full = url + "?" + urlencode(params)
    try:
        with urlopen(Request(full, headers=_ultra__0x_headers()), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        raise RuntimeError(f"0x {e.code} {e.reason}: {body}")

def _ultra__chain_id(chain: str) -> int:
    m = {"ethereum":1, "base":8453, "arbitrum":42161, "optimism":10, "polygon":137}
    return m.get(chain)

def _ultra__norm_quote_numbers(q: dict) -> dict:
    if not isinstance(q, dict): return q
    for k in ("gas","estimatedGas","gasPrice","maxFeePerGas","maxPriorityFeePerGas","value","buyAmount","sellAmount"):
        if k in q: q[k] = _rb__hexint(q.get(k), 0)
    return q

def _ultra__get_0x_quote_v1(self, chain: str, sell_token: str, buy_token: str, sell_amount_raw: int, slippage_bps: int | None = None):
    cid = _ultra__chain_id(chain)
    if not cid: raise ValueError(f"unsupported chain for 0x: {chain}")
    slip = slippage_bps if slippage_bps is not None else int(os.getenv("SWAP_SLIPPAGE_BPS","50"))  # 50 bps = 0.5%
    slip_pct = max(0, slip) / 10000.0
    params = {
        "chainId": cid,
        "sellToken": sell_token,
        "buyToken": buy_token,
        "sellAmount": int(sell_amount_raw),
        "takerAddress": self.acct.address,  # important for validation path
        "slippagePercentage": slip_pct,
        # keep validation enabled
    }
    q = _ultra__http_get_json(f"{_rb__0x_base_url(self, chain)}/swap/v1/quote", params)
    return _ultra__norm_quote_numbers(q)

def _ultra__preflight_ok(w3, frm, to, data_bytes, value_wei) -> tuple[bool,str]:
    try:
        w3.eth.call({"from": frm, "to": to, "data": data_bytes, "value": int(value_wei)}, "latest")
        return True, ""
    except Exception as e:
        return False, str(e)

def _ultra__auto_fees(w3, tip_gwei: float | None):
    # EIP-1559 fees
    try:
        base = int(w3.eth.get_block("latest")["baseFeePerGas"])
    except Exception:
        base = int(getattr(w3.eth, "gas_price", 0)) or int(w3.to_wei(2, "gwei"))
    tip = w3.to_wei(float(tip_gwei if tip_gwei is not None else float(os.getenv("SWAP_TIP_GWEI","2"))), "gwei")
    # heuristic: maxFee = base*2 + tip
    max_fee = base * 2 + tip
    return max_fee, tip

def _ultra__send_swap_via_0x_v1(self, chain: str, quote: dict, *, wait=True, slippage_bps: int | None = None):
    # normalize quote numerics and build tx
    q = _ultra__norm_quote_numbers(dict(quote))
    w3 = self._rb__w3(chain)
    to = Web3.to_checksum_address(q["to"])
    data = q["data"]
    if isinstance(data, str) and data.startswith("0x"):
        data_bytes = bytes.fromhex(data[2:])
    elif isinstance(data, (bytes,bytearray)):
        data_bytes = bytes(data)
    else:
        raise ValueError("bad 0x quote: no tx data")
    value = _rb__hexint(q.get("value"), 0)

    # base tx
    tx = {
        "from": self.acct.address,
        "to": to,
        "data": data_bytes,
        "value": int(value),
        "nonce": w3.eth.get_transaction_count(self.acct.address),
        "chainId": int(w3.eth.chain_id),
        "type": 2,
    }

    # gas (estimate first; fallback to quote hint; then hard fallback)
    try:
        gas_est = int(w3.eth.estimate_gas({"from": tx["from"], "to": tx["to"], "data": tx["data"], "value": tx["value"]}))
    except Exception:
        gas_est = int(q.get("estimatedGas") or q.get("gas") or 250000)
    tx["gas"] = max(21000, gas_est)

    # fees
    max_fee, tip = _ultra__auto_fees(w3, None)
    tx["maxFeePerGas"] = int(max_fee)
    tx["maxPriorityFeePerGas"] = int(tip)

    # preflight
    ok, why = _ultra__preflight_ok(w3, tx["from"], tx["to"], tx["data"], tx["value"])
    if not ok:
        # try auto-slippage bumps by re-quoting at higher slippage and retry
        # derive sell/buy/amount from quote we got
        sell_tok = q.get("sellToken") or q.get("from") or ""
        buy_tok  = q.get("buyToken")  or q.get("toToken") or ""
        sell_amt = int(q.get("sellAmount") or 0)
        if not sell_tok or not buy_tok or not sell_amt:
            raise ValueError(f"preflight: {why}")
        bumps = [50, 100, 150, 200]  # bps
        already = set([int(slippage_bps)] if slippage_bps is not None else [])
        for b in bumps:
            if b in already: 
                continue
            try:
                newq = _ultra__get_0x_quote_v1(self, chain, sell_tok, buy_tok, sell_amt, slippage_bps=b)
                # rebuild tx pieces
                to2 = Web3.to_checksum_address(newq["to"])
                data2 = newq["data"]
                data2 = bytes.fromhex(data2[2:]) if isinstance(data2, str) and data2.startswith("0x") else data2
                val2 = _rb__hexint(newq.get("value"), 0)
                # re-estimate & fees
                try:
                    gas2 = int(w3.eth.estimate_gas({"from": tx["from"], "to": to2, "data": data2, "value": int(val2)}))
                except Exception:
                    gas2 = int(newq.get("estimatedGas") or newq.get("gas") or tx["gas"])
                tx.update({"to": to2, "data": data2, "value": int(val2), "gas": max(21000, gas2)})
                ok2, why2 = _ultra__preflight_ok(w3, tx["from"], tx["to"], tx["data"], tx["value"])
                if ok2:
                    q = newq  # adopt for logs
                    break
                why = why2
            except Exception as e:
                why = str(e)
                continue
        else:
            raise ValueError(f"preflight: {why}")

    # sign + send
    signed = w3.eth.account.sign_transaction(tx, self.pk)
    txh = w3.eth.send_raw_transaction(signed.rawTransaction).hex()
    if wait:
        rec = w3.eth.wait_for_transaction_receipt(txh, timeout=180)
        return txh, rec
    return txh, None

# bind overrides
try:
    UltraSwapBridge.get_0x_quote = _ultra__get_0x_quote_v1
    UltraSwapBridge.send_swap_via_0x = _ultra__send_swap_via_0x_v1
except NameError:
    pass

# === ULTRA_PATCH_0X_V1_COMPAT: accept gas/max_fee/priority/nonce/extra kwargs ===
from web3 import Web3

def _ultra__send_swap_via_0x_v1(self, chain: str, quote: dict, *, wait=True, slippage_bps=None,
                                gas=None, max_fee_gwei=None, max_priority_gwei=None, nonce=None, **_extra):
    # normalize numerics
    q = dict(quote)
    try:
        _ = _ultra__norm_quote_numbers  # ensure exists
    except NameError:
        def _ultra__norm_quote_numbers(d): 
            for k in ("gas","estimatedGas","gasPrice","maxFeePerGas","maxPriorityFeePerGas","value","buyAmount","sellAmount"):
                if k in d:
                    d[k] = _rb__hexint(d.get(k), 0)
            return d
    q = _ultra__norm_quote_numbers(q)

    w3 = self._rb__w3(chain)
    to = Web3.to_checksum_address(q["to"])
    data = q["data"]
    if isinstance(data, str) and data.startswith("0x"):
        data_bytes = bytes.fromhex(data[2:])
    elif isinstance(data, (bytes, bytearray)):
        data_bytes = bytes(data)
    else:
        raise ValueError("bad 0x quote: no tx data")
    value = _rb__hexint(q.get("value"), 0)

    tx = {
        "from": self.acct.address,
        "to": to,
        "data": data_bytes,
        "value": int(value),
        "chainId": int(w3.eth.chain_id),
        "type": 2,
        "nonce": (nonce if nonce is not None else w3.eth.get_transaction_count(self.acct.address)),
    }

    # gas handling: use provided gas if valid, else estimate, else quote hint, else fallback
    g = _rb__hexint(gas, 0) if gas is not None else 0
    if g <= 0:
        try:
            g = int(w3.eth.estimate_gas({"from": tx["from"], "to": tx["to"], "data": tx["data"], "value": tx["value"]}))
        except Exception:
            g = int(q.get("estimatedGas") or q.get("gas") or 250000)
    tx["gas"] = max(21000, g)

    # EIP-1559 fees
    try:
        base = int(w3.eth.get_block("latest")["baseFeePerGas"])
    except Exception:
        base = int(getattr(w3.eth, "gas_price", 0)) or int(w3.to_wei(2, "gwei"))
    tip = w3.to_wei(float(max_priority_gwei) if max_priority_gwei is not None else float(os.getenv("SWAP_TIP_GWEI","2")), "gwei")
    if max_fee_gwei is not None:
        max_fee = w3.to_wei(float(max_fee_gwei), "gwei")
    else:
        max_fee = base * 2 + tip  # heuristic
    tx["maxPriorityFeePerGas"] = int(tip)
    tx["maxFeePerGas"] = int(max_fee)

    # preflight; if revert, auto-bump slippage by re-quoting and rebuilding tx
    def _preflight(frm, to, data_bytes, value_wei):
        try:
            w3.eth.call({"from": frm, "to": to, "data": data_bytes, "value": int(value_wei)}, "latest")
            return True, ""
        except Exception as e:
            return False, str(e)

    ok, why = _preflight(tx["from"], tx["to"], tx["data"], tx["value"])
    if not ok:
        # attempt slippage bumps with v1 re-quotes
        sell_tok = q.get("sellToken") or q.get("from") or ""
        buy_tok  = q.get("buyToken")  or q.get("toToken") or ""
        sell_amt = int(q.get("sellAmount") or 0)
        if not (sell_tok and buy_tok and sell_amt):
            raise ValueError(f"preflight: {why}")
        bumps = [50, 100, 150, 200]  # 0.5% → 2.0%
        seen = set([int(slippage_bps)] if slippage_bps is not None else [])
        for b in bumps:
            if b in seen: 
                continue
            seen.add(b)
            try:
                newq = _ultra__get_0x_quote_v1(self, chain, sell_tok, buy_tok, sell_amt, slippage_bps=b)
                newq = _ultra__norm_quote_numbers(newq)
                to2 = Web3.to_checksum_address(newq["to"])
                dat = newq["data"]
                dat = bytes.fromhex(dat[2:]) if isinstance(dat, str) and dat.startswith("0x") else dat
                val = _rb__hexint(newq.get("value"), 0)
                try:
                    gg = int(w3.eth.estimate_gas({"from": tx["from"], "to": to2, "data": dat, "value": int(val)}))
                except Exception:
                    gg = int(newq.get("estimatedGas") or newq.get("gas") or tx["gas"])
                tx.update({"to": to2, "data": dat, "value": int(val), "gas": max(21000, gg)})
                ok2, why2 = _preflight(tx["from"], tx["to"], tx["data"], tx["value"])
                if ok2:
                    q = newq  # for completeness
                    break
                why = why2
            except Exception as e:
                why = str(e)
                continue
        else:
            raise ValueError(f"preflight: revert — {why}")

    signed = w3.eth.account.sign_transaction(tx, self.pk)
    txh = w3.eth.send_raw_transaction(signed.rawTransaction).hex()
    if wait:
        rec = w3.eth.wait_for_transaction_receipt(txh, timeout=180)
        return txh, rec
    return txh, None

# rebind
try:
    UltraSwapBridge.send_swap_via_0x = _ultra__send_swap_via_0x_v1
except NameError:
    pass

# --- compat aliases for web3 getter ---
try:
    UltraSwapBridge._web3 = UltraSwapBridge._rb__w3
    UltraSwapBridge.w3    = UltraSwapBridge._rb__w3
except NameError:
    pass


# ---- Arbitrum constants ----
CAMELOT_V2_ROUTER = {
    "arbitrum": "0xC873FECbd354f5A56E00E710B90EF4201db2448d",
}
WETH_ADDRESS = {
    "arbitrum": "0x82aF49447D8a07e3bd95BDdB56f35241523fBab1",
}


# minimal UniswapV2-like router ABI
_ABI_V2_ROUTER = [
  {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},
             {"internalType":"uint256","name":"amountOutMin","type":"uint256"},
             {"internalType":"address[]","name":"path","type":"address[]"},
             {"internalType":"address","name":"to","type":"address"},
             {"internalType":"address","name":"referrer","type":"address"},
             {"internalType":"uint256","name":"deadline","type":"uint256"}],
   "name":"swapExactTokensForTokensSupportingFeeOnTransferTokens",
   "outputs":[], "stateMutability":"nonpayable","type":"function"},
  {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},
             {"internalType":"address[]","name":"path","type":"address[]"}],
   "name":"getAmountsOut",
   "outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],
   "stateMutability":"view","type":"function"}
]

# minimal ERC20 balance/allow/approve ABI (if missing)

def send_prebuilt_tx(self, chain: str, to: str, data: str, *, value: int = 0, gas: int | None = None):
    w3 = self._rb__w3(chain)
    to_cs = w3.to_checksum_address(to)
    tx = {
        "chainId": w3.eth.chain_id,
        "from": self.acct.address,
        "to": to_cs,
        "nonce": w3.eth.get_transaction_count(self.acct.address, "pending"),
        "data": data if data else "0x",
        "value": int(value or 0),
        **self._rb__fee_fields(w3),
    }
    try:
        if gas is None:
            tx["gas"] = int(w3.eth.estimate_gas(tx) * 1.2)
        else:
            tx["gas"] = int(gas)
    except Exception:
        # fallback
        tx["gas"] = 250000
    signed = self.acct.sign_transaction(tx)
    txh = w3.eth.send_raw_transaction(signed.rawTransaction)
    return w3.to_hex(txh)

def camelot_v2_quote(self, chain: str, token_in: str, token_out: str, amount_in: int, *, slippage_bps: int = 100):
    """Return a dict that mirrors aggregator quote keys for our CLI:
       {'aggregator':'CamelotV2','to','data','value','allowanceTarget','gas','buyAmount','route'}
    """
    chain = chain.lower().strip()
    if chain not in CAMELOT_V2_ROUTER: 
        return {"__error__": f"CamelotV2 unsupported on {chain}"}
    w3 = self._rb__w3(chain)
    router = w3.eth.contract(w3.to_checksum_address(CAMELOT_V2_ROUTER[chain]), abi=_ABI_V2_ROUTER)
    t_in  = w3.to_checksum_address(token_in)
    t_out = w3.to_checksum_address(token_out)
    weth  = w3.to_checksum_address(WETH_ADDRESS.get(chain, WETH_ADDRESS["arbitrum"]))
    path_try = []
    # 1) direct
    try:
        amts = router.functions.getAmountsOut(int(amount_in), [t_in, t_out]).call()
        if len(amts) >= 2 and int(amts[-1]) > 0:
            path_try.append([t_in, t_out])
    except Exception:
        pass
    # 2) via WETH
    try:
        amtsA = router.functions.getAmountsOut(int(amount_in), [t_in, weth]).call()
        amtsB = router.functions.getAmountsOut(int(amtsA[-1]), [weth, t_out]).call()
        if int(amtsA[-1]) > 0 and int(amtsB[-1]) > 0:
            path_try.append([t_in, weth, t_out])
    except Exception:
        pass
    if not path_try:
        return {"__error__": "CamelotV2: no path (direct or via WETH)"}
    # choose the best (largest out)
    best = None
    best_out = -1
    for p in path_try:
        try:
            amts = router.functions.getAmountsOut(int(amount_in), p).call()
            out = int(amts[-1])
            if out > best_out:
                best_out, best = out, p
        except Exception:
            continue
    if best is None or best_out <= 0:
        return {"__error__": "CamelotV2: quoting failed"}
    # build tx data
    out_min = int(best_out * (10_000 - slippage_bps) / 10_000)
    deadline = int(self._rb__w3(chain).eth.get_block("latest")["timestamp"]) + 600
    fn = router.functions.swapExactTokensForTokensSupportingFeeOnTransferTokens(int(amount_in), out_min, best, self.acct.address, "0x0000000000000000000000000000000000000000", deadline)
    data = fn._encode_transaction_data()
    allowance_target = CAMELOT_V2_ROUTER[chain]
    # try gas estimate
    gas_est = 0
    try:
        gas_est = int(self._rb__w3(chain).eth.estimate_gas({
            "from": self.acct.address,
            "to": allowance_target,
            "data": data,
            "value": 0
        }) * 1.2)
    except Exception:
        gas_est = 250000
    return {
        "aggregator": "CamelotV2",
        "to": allowance_target,
        "data": data,
        "value": 0,
        "allowanceTarget": allowance_target,
        "gas": gas_est,
        "buyAmount": best_out,
        "route": best,
    }

UltraSwapBridge.erc20_decimals = erc20_decimals

UltraSwapBridge.send_prebuilt_tx = send_prebuilt_tx

UltraSwapBridge.camelot_v2_quote = camelot_v2_quote


def _ultra__0x_v2_headers():
    import os
    h = {"Accept": "application/json",
         "User-Agent": os.getenv("HTTP_UA", "CoolCryptoUtilities/1.0"),
         "0x-version": "v2"}
    api_key = _ultra__load_0x_api_key()
    if api_key:
        h["0x-api-key"] = api_key
    return h

def _ultra__http_get_json_v2(path: str, params: dict, timeout=25):
    import requests
    url = "https://api.0x.org" + path
    r = requests.get(url, headers=_ultra__0x_v2_headers(), params=params, timeout=timeout)
    if r.status_code >= 400:
        msg = r.text
        if r.status_code == 401:
            msg += "  (Hint: set ZEROX_API_KEY env or ~/.config/coolcrypto/0x_api_key)"
        raise RuntimeError(f"0x v2 {r.status_code}: {msg}")
    return r.json()

def _ultra__get_0x_quote_v2_allowance_holder(self, chain: str, sell_token: str, buy_token: str, sell_amount_raw: int, slippage_bps: int | None = None):
    """
    0x Swap API v2 (Allowance-Holder) quote.
    - Uses single host https://api.0x.org, endpoint /swap/allowance-holder/quote
    - Required params: chainId, sellToken, buyToken, sellAmount (string int), taker, slippageBps
    - Returns normalized dict with transaction fields + sellAmountRaw for re-quotes.
    """
    import os
    cid = _ultra__chain_id(chain)
    if not cid:
        raise ValueError(f"unsupported chain for 0x v2: {chain}")
    slip = int(slippage_bps if slippage_bps is not None else os.getenv("SWAP_SLIPPAGE_BPS", "100"))
    params = {
        "chainId": cid,
        "sellToken": sell_token,
        "buyToken": buy_token,
        # v2 requires integer base units (string)
        "sellAmount": str(int(sell_amount_raw)),
        "taker": self.acct.address,
        "slippageBps": max(0, slip),
    }
    q = _ultra__http_get_json_v2("/swap/allowance-holder/quote", params)

    # Prefer 'transaction' bundle when present
    txo  = q.get("transaction") or {}
    to   = txo.get("to")   or q.get("to")
    data = txo.get("data") or q.get("data")
    value= txo.get("value") or q.get("value") or 0
    gas  = txo.get("gas")  or q.get("estimatedGas") or q.get("gas") or 0
    gasp = txo.get("gasPrice") or q.get("gasPrice") or 0

    # Allowance target (top-level or issues.allowance.spender)
    allowance = q.get("allowanceTarget")
    if not allowance:
        try:
            allowance = (q.get("issues") or {}).get("allowance", {}).get("spender")
        except Exception:
            allowance = None

    out = {
        "aggregator": "0x-v2-allowance-holder",
        "to": to, "data": data, "value": value,
        "gas": gas, "gasPrice": gasp,
        "allowanceTarget": allowance,
        "sellToken": sell_token, "buyToken": buy_token,
        "sellAmountRaw": int(sell_amount_raw),
        "sellAmount": (q.get("sellAmount") or str(int(sell_amount_raw))),
        "buyAmount": q.get("buyAmount"),
        "issues": q.get("issues"), "route": q.get("route"),
    }
    if not out["to"] or not out["data"]:
        raise RuntimeError("0x v2 quote: missing tx fields")
    return out
def _ultra__send_swap_via_0x_v2(self, chain: str, quote: dict, *, wait=True, slippage_bps: int | None = None,
                                gas=None, max_fee_gwei=None, max_priority_gwei=None, nonce=None, **_extra):
    """
    Send a v2 Allowance-Holder transaction.
    - Uses EIP-1559 fee fields from _rb__fee_fields
    - Preflights with eth_call; on revert, bumps slippage and re-quotes using sellAmountRaw.
    """
    import os
    from web3 import Web3

    def _hx(v, d=0):
        try:
            if v is None: return d
            if isinstance(v, int): return v
            sv = str(v).strip().lower()
            return int(sv, 16) if sv.startswith("0x") else int(sv)
        except Exception:
            return d

    w3 = self._rb__w3(chain)
    q  = dict(quote)
    to = Web3.to_checksum_address(q["to"])
    data = q["data"]
    if isinstance(data, str) and data.startswith("0x"):
        data_bytes = bytes.fromhex(data[2:])
    elif isinstance(data, (bytes, bytearray)):
        data_bytes = bytes(data)
    else:
        raise ValueError("bad 0x v2 quote: no tx data")
    value = _hx(q.get("value"), 0)

    tx = {
        "from": self.acct.address,
        "to": to,
        "data": data_bytes,
        "value": int(value),
        "chainId": int(w3.eth.chain_id),
        "type": 2,
        "nonce": (nonce if nonce is not None else w3.eth.get_transaction_count(self.acct.address, "pending")),
    }

    # gas: estimate -> quote hint -> fallback
    g = _hx(gas, 0) if gas is not None else 0
    if g <= 0:
        try:
            g = int(w3.eth.estimate_gas({"from": tx["from"], "to": tx["to"], "data": tx["data"], "value": tx["value"]}))
        except Exception:
            g = _hx(q.get("gas") or q.get("estimatedGas"), 220000)
    tx["gas"] = max(21000, int(g))

    # EIP-1559 fees
    fees = self._rb__fee_fields(w3, max_priority_gwei=max_priority_gwei, max_fee_gwei=max_fee_gwei)
    tx.update(fees)

    dbg = bool(int(os.getenv("DEBUG_SWAP", "0")))
    if dbg:
        try:
            print(f"[v2] preflight tx → to={tx['to']} value={tx['value']} gas≈{tx['gas']}")
        except Exception:
            pass

    # Preflight; if revert, bump slippage and re-quote using sellAmountRaw
    try:
        w3.eth.call({"from": tx["from"], "to": tx["to"], "data": tx["data"], "value": tx["value"]}, "latest")
    except Exception as e:
        sell_tok = q.get("sellToken") or ""
        buy_tok  = q.get("buyToken")  or ""
        sell_amt = q.get("sellAmountRaw")
        if sell_amt is None:
            sell_amt = _hx(q.get("sellAmount"), 0)
        sell_amt = int(sell_amt or 0)

        bumps = [
            int(os.getenv("SWAP_BUMP_BPS1", "100")),
            int(os.getenv("SWAP_BUMP_BPS2", "150")),
            int(os.getenv("SWAP_BUMP_BPS3", "200")),
        ]
        tried = set([int(slippage_bps)] if slippage_bps is not None else [])
        why = str(e)
        for b in bumps:
            if b in tried: 
                continue
            tried.add(b)
            if dbg:
                try:
                    print(f"[v2] re-quote params: chain={chain} sell={sell_tok} buy={buy_tok} sellAmount={int(sell_amt)} bps={b}")
                except Exception:
                    pass
            newq = _ultra__get_0x_quote_v2_allowance_holder(self, chain, sell_tok, buy_tok, int(sell_amt), slippage_bps=b)
            to2 = Web3.to_checksum_address(newq["to"])
            dat = newq["data"]; dat = bytes.fromhex(dat[2:]) if isinstance(dat, str) and dat.startswith("0x") else dat
            val = _hx(newq.get("value"), 0)
            try:
                gg = int(w3.eth.estimate_gas({"from": tx["from"], "to": to2, "data": dat, "value": int(val)}))
            except Exception:
                gg = _hx(newq.get("gas") or newq.get("estimatedGas"), tx["gas"])
            tx.update({"to": to2, "data": dat, "value": int(val), "gas": max(21000, gg)})
            try:
                w3.eth.call({"from": tx["from"], "to": tx["to"], "data": tx["data"], "value": tx["value"]})
                q = newq
                break
            except Exception as ee:
                why = str(ee)
                continue
        else:
            raise ValueError(f"preflight: revert — {why}")

    # Sign & send
    self._rb__ensure_live()
    signed = self.acct.sign_transaction(tx)
    txh = w3.eth.send_raw_transaction(signed.rawTransaction)
    if not wait:
        return w3.to_hex(txh), None
    rc = w3.eth.wait_for_transaction_receipt(txh)
    ok = int(rc.get("status", 0)) == 1
    if not ok:
        raise RuntimeError("0x v2: on-chain revert")
    return w3.to_hex(txh), rc
def _ultra__load_0x_api_key():
    import os
    # 1) env (preferred)
    for k in ("ZEROX_API_KEY","OX_API_KEY","API_0X","API0X","API_OX"):
        v = os.getenv(k)
        if v and v.strip():
            return v.strip()
    # 2) local files (git-ignored), first hit wins
    cand = [
        os.path.expanduser("~/.config/coolcrypto/0x_api_key"),
        os.path.expanduser("~/.coolcrypto/0x_api_key"),
        os.path.join(os.getcwd(), ".secrets/0x_api_key"),
        os.path.join(os.getcwd(), ".env.local"),
        os.path.join(os.getcwd(), ".env"),
    ]
    for fp in cand:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): 
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() in ("ZEROX_API_KEY","OX_API_KEY","API_0X","API0X","API_OX"):
                            return v.strip()
                    else:
                        # raw key file
                        if len(line) > 20:
                            return line
        except Exception:
            pass
    return None
UltraSwapBridge.get_0x_quote = _ultra__get_0x_quote_v2_allowance_holder
UltraSwapBridge.send_swap_via_0x = _ultra__send_swap_via_0x_v2
