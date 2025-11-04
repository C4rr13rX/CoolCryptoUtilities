
from __future__ import annotations

import os
import time
import math
import json
import hmac
import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import certifi
import requests
from dotenv_fallback import load_dotenv, find_dotenv, dotenv_values

# ---------------------------------------------------------------------
# .env loader (same robust pattern you use elsewhere; safe on PyDroid)
# ---------------------------------------------------------------------
def _load_env_robust() -> None:
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False); return

    cands: List[str] = []
    try: cands.append(os.path.join(os.getcwd(), ".env"))
    except Exception: pass
    try:
        if os.sys.argv and os.sys.argv[0]:
            cands.append(os.path.join(os.path.dirname(os.path.realpath(os.sys.argv[0])), ".env"))
    except Exception: pass
    try:
        if "__file__" in globals():
            cands.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".env"))
    except Exception: pass
    try: cands.append(os.path.join(os.path.expanduser("~"), ".env"))
    except Exception: pass

    for p in cands:
        try:
            if os.path.isfile(p):
                load_dotenv(p, override=False); return
        except Exception:
            pass
    for p in cands:
        try:
            if os.path.isfile(p):
                for k, v in (dotenv_values(p) or {}).items():
                    os.environ.setdefault(k, v or "")
                return
        except Exception:
            pass

_load_env_robust()

# ---------------------------------------------------------------------
# GoPlus API basics
# - Access token: POST /api/v1/token with SHA1(app_key + time + app_secret)
# - Auth header: Authorization: Bearer <access_token>
# - Token Security: GET /api/v1/token_security/{chain_id}?contract_addresses=0x..,0x..
# - Free-tier recommended limit ~30 requests/min
# Docs: (citations provided in chat message)
# ---------------------------------------------------------------------

GOPLUS_BASE = os.getenv("GOPLUS_BASE", "https://api.gopluslabs.io")
GOPLUS_APP_KEY = os.getenv("GOPLUS_APP_KEY", "").strip()
GOPLUS_APP_SECRET = os.getenv("GOPLUS_APP_SECRET", "").strip()

# Knobs (optional; sane defaults)
FILTER_SCAMS = bool(int(os.getenv("FILTER_SCAMS", "1") or "1"))
SCAM_STRICT = bool(int(os.getenv("SCAM_STRICT", "0") or "0"))
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "8") or "8")
PRICE_WORKERS = int(os.getenv("PRICE_WORKERS", "8") or "8") # not used here, kept for symmetry
GOPLUS_RATE_PER_MIN = int(os.getenv("GOPLUS_RATE_PER_MIN", "25") or "25") # stay < 30/min

# Chain name -> chain_id mapping (only the networks you use)
CHAIN_IDS: Dict[str, int] = {
    "ethereum": 1,
    "base": 8453,
    "arbitrum": 42161,
    "optimism": 10,
    "polygon": 137,
}

# Input style support:
# - "portfolio" (UltraSwapBridge default): "chain:0x..." except Base uses "0x...@base"
# - "prefix": "chain:0x..."
# - "suffix": "0x...@chain"
# - "pairs": [("chain","0x..."), ...]
TokenSpec = Union[str, Tuple[str, str]]

@dataclass
class FilterResult:
    tokens: List[TokenSpec] # filtered, same shape as input
    flagged: Dict[str, Dict[str, Any]] # addr(lower) -> raw GoPlus fields
    reasons: Dict[str, List[str]] # addr(lower) -> reasons (["is_honeypot", ...])

class _Bearer:
    """Tiny cached bearer token holder."""
    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._exp: float = 0.0
        self._lock = threading.Lock()

    def get(self, force: bool = False) -> str:
        with self._lock:
            now = time.time()
            if not force and self._token and now < self._exp - 5:
                return self._token
            self._token, ttl = self._fetch_token()
            self._exp = now + max(ttl - 5, 30) # small safety margin; min 30s
            return self._token

    @staticmethod
    def _fetch_token() -> Tuple[str, int]:
        if not GOPLUS_APP_KEY or not GOPLUS_APP_SECRET:
            raise RuntimeError("GOPLUS_APP_KEY / GOPLUS_APP_SECRET missing in environment.")

        t = int(time.time())
        # Per docs: sign = sha1(app_key + time + app_secret)
        to_sign = f"{GOPLUS_APP_KEY}{t}{GOPLUS_APP_SECRET}".encode("utf-8")
        sign = hashlib.sha1(to_sign).hexdigest()
        url = f"{GOPLUS_BASE}/api/v1/token"
        r = requests.post(
            url,
            json={"app_key": GOPLUS_APP_KEY, "time": t, "sign": sign},
            timeout=HTTP_TIMEOUT_SEC,
            verify=certifi.where(),
        )
        r.raise_for_status()
        j = r.json() or {}
        # The API returns either {"access_token": "...", "expires_in": 7200}
        # or wrapped under {"result": {...}} depending on gateway; handle both.
        if "access_token" in j:
            return str(j["access_token"]), int(j.get("expires_in", 3600))
        res = j.get("result") or {}
        return str(res.get("access_token") or ""), int(res.get("expires_in") or 3600)

_bearer = _Bearer()

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"accept": "application/json"})
    return s

def _rate_sleep(last_times: List[float], limit_per_min: int) -> None:
    # Basic leaky-bucket: keep times in last 60s; if over limit, sleep just enough.
    now = time.time()
    one_min_ago = now - 60.0
    while last_times and last_times[0] < one_min_ago:
        last_times.pop(0)
    if len(last_times) >= limit_per_min:
        wait = 60.0 - (now - last_times[0]) + 0.01
        if wait > 0:
            time.sleep(wait)

def _parse_token_item(it: TokenSpec) -> Optional[Tuple[str, str, str]]:
    """
    Returns tuple (shape, chain, address) or None.
      shape: "prefix" | "suffix" | "pairs"
    """
    if isinstance(it, tuple) and len(it) == 2:
        ch = str(it[0]).lower().strip()
        addr = str(it[1]).strip()
        if ch in CHAIN_IDS and addr.startswith("0x") and len(addr) >= 42:
            return ("pairs", ch, addr)
        return None

    if isinstance(it, str):
        s = it.strip()
        if ":" in s:
            ch, addr = s.split(":", 1)
            ch = ch.lower().strip()
            addr = addr.strip()
            if ch in CHAIN_IDS and addr.startswith("0x"):
                return ("prefix", ch, addr)
        if "@" in s:
            addr, ch = s.split("@", 1)
            ch = ch.lower().strip()
            addr = addr.strip()
            if ch in CHAIN_IDS and addr.startswith("0x"):
                return ("suffix", ch, addr)
    return None

def _rebuild(shape: str, ch: str, addr: str) -> TokenSpec:
    if shape == "pairs": return (ch, addr)
    if shape == "suffix": return f"{addr}@{ch}"
    return f"{ch}:{addr}"

def _chunk(lst: Sequence[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), n):
        yield list(lst[i:i+n])

def _must_flag(fields: Dict[str, Any], strict: bool) -> Tuple[bool, List[str]]:
    """
    Only flag hard scams by default (honeypot / cannot sell all).
    In strict mode, also flag if transfer can be paused or black/white list is enforced.
    """
    reasons: List[str] = []
    def is_true(key: str) -> bool:
        v = fields.get(key, None)
        if v is None: return False
        if isinstance(v, (int, float)): return v == 1
        if isinstance(v, str): return v.strip() == "1" or v.strip().lower() == "true"
        return False

    if is_true("is_honeypot"):
        reasons.append("is_honeypot")
    if is_true("cannot_sell_all"):
        reasons.append("cannot_sell_all")

    if strict:
        for k in ("transfer_pausable", "is_blacklisted", "personal_slippage_modifiable"):
            if is_true(k): reasons.append(k)

    return (len(reasons) > 0, reasons)

class FilterScamTokens:
    """
    Usage:
        fst = FilterScamTokens()
        filtered = fst.filter(annotated_tokens_from_ultraswapbridge)
        # filtered.tokens -> same shape as input, minus flagged
        # filtered.flagged -> {addr_lower: raw goplus record}
        # filtered.reasons -> {addr_lower: ["is_honeypot", ...]}

    Env:
        GOPLUS_APP_KEY, GOPLUS_APP_SECRET # required
        FILTER_SCAMS=1|0 # enable/disable (default 1)
        SCAM_STRICT=1|0 # broaden flags (default 0)
        GOPLUS_RATE_PER_MIN # default 25
        HTTP_TIMEOUT_SEC # default 8
    """

    def __init__(self, chunk_size: int = 80, strict: Optional[bool] = None):
        # Safe chunk; GoPlus supports batching via comma-separated list.
        # Keep conservative to avoid large URLs.
        self.chunk_size = int(chunk_size)
        self.strict = SCAM_STRICT if strict is None else bool(strict)
        self._sess = _session()
        self._calls_times: List[float] = []

    def _headers(self) -> Dict[str, str]:
        tok = _bearer.get()
        return {"Authorization": f"Bearer {tok}", "accept": "application/json"}

    def _fetch_chain_batch(self, chain: str, addrs: List[str]) -> Dict[str, Dict[str, Any]]:
        """Return addr_lower -> fields (GoPlus record)."""
        out: Dict[str, Dict[str, Any]] = {}
        if not addrs:
            return out
        chain_id = CHAIN_IDS[chain]
        for group in _chunk(addrs, self.chunk_size):
            # Rate-limit
            _rate_sleep(self._calls_times, GOPLUS_RATE_PER_MIN)
            url = f"{GOPLUS_BASE}/api/v1/token_security/{chain_id}"
            params = {"contract_addresses": ",".join(group)}
            # rotate header each call (token refresh safe)
            hdrs = self._headers()
            r = self._sess.get(url, params=params, headers=hdrs, timeout=HTTP_TIMEOUT_SEC, verify=certifi.where())
            self._calls_times.append(time.time())

            if r.status_code == 401:
                # refresh bearer once and retry
                _bearer.get(force=True)
                hdrs = self._headers()
                r = self._sess.get(url, params=params, headers=hdrs, timeout=HTTP_TIMEOUT_SEC, verify=certifi.where())
                self._calls_times.append(time.time())

            r.raise_for_status()
            j = r.json() or {}
            # Normalize possible shapes:
            # v1/v2 commonly -> {"code":1,"message":"OK","result":{"0x..":{...},"0x..":{...}}}
            # Some gateways -> {"result":[{"address":"0x..", ...}, ...]}
            res = j.get("result", {})
            if isinstance(res, dict):
                for k, v in res.items():
                    if isinstance(v, dict):
                        out[(k or "").lower()] = v
            elif isinstance(res, list):
                for item in res:
                    addr = (item or {}).get("address") or (item or {}).get("contract_address") or ""
                    if addr:
                        out[str(addr).lower()] = item
        return out

    def filter(self, tokens: Sequence[TokenSpec]) -> FilterResult:
        """
        Filter tokens (any of the supported shapes). Returns same shape, minus flagged.
        """
        if not FILTER_SCAMS:
            return FilterResult(tokens=list(tokens), flagged={}, reasons={})

        # Parse & bucket by chain while remembering original order/shape
        parsed: List[Tuple[int, str, str, str]] = [] # (index, shape, chain, addr)
        for idx, it in enumerate(tokens):
            p = _parse_token_item(it)
            if not p:
                continue
            shape, ch, addr = p
            parsed.append((idx, shape, ch, addr))

        by_chain: Dict[str, List[str]] = {}
        for _, _, ch, addr in parsed:
            by_chain.setdefault(ch, []).append(addr)

        # Query GoPlus per chain
        addr_info: Dict[str, Dict[str, Any]] = {}
        reasons: Dict[str, List[str]] = {}
        for ch, addr_list in by_chain.items():
            info = self._fetch_chain_batch(ch, addr_list)
            for a_lower, fields in info.items():
                flag, why = _must_flag(fields, self.strict)
                if flag:
                    addr_info[a_lower] = fields
                    reasons[a_lower] = why

        # Build filtered output in original shape & order
        kept: List[TokenSpec] = []
        for idx, shape, ch, addr in parsed:
            if addr.lower() in addr_info:
                continue # drop
            kept.append(_rebuild(shape, ch, addr))

        return FilterResult(tokens=kept, flagged=addr_info, reasons=reasons)


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage with "portfolio" style list
    sample = [
        "arbitrum:0x0000000000000000000000000000000000000000", # nonsense; expect no result
        "0x111111111117dc0aa78b770fa6a738034120c302@base", # 1inch on Base (should be fine)
    ]
    filt = FilterScamTokens(strict=False)
    try:
        res = filt.filter(sample)
        print("INPUT :", json.dumps(sample, indent=2))
        print("OUTPUT:", json.dumps(res.tokens, indent=2))
        if res.flagged:
            print("\n-- Flagged (dropped) --")
            for a, why in res.reasons.items():
                print(a, "=>", why)
    except Exception as e:
        print("Filter error:", type(e).__name__, str(e))
