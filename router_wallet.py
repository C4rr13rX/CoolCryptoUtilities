
from __future__ import annotations
import os, time, json, certifi, requests, sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable
from pathlib import Path
from dotenv import load_dotenv, find_dotenv, dotenv_values
from filter_scams import FilterScamTokens
from cache import CacheBalances, CacheTransfers


# ---------- .env loader ----------
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
    base = "https://api.0x.org/swap/permit2/price"
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
            res = self._alchemy_post(url, "alchemy_getTokenBalances", params).get("result") or {}
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

