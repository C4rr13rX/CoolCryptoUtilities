# router_wallet.py  —  clean OOP version
from __future__ import annotations

import os, sys, time, json, certifi, requests, threading
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, getcontext as _getctx
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

from dotenv_fallback import load_dotenv, find_dotenv, dotenv_values

try:
    from web3 import Web3  # type: ignore
    from eth_account import Account  # type: ignore
    from eth_account.signers.local import LocalAccount  # type: ignore
    _WEB3_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover - optional dependency
    Web3 = None  # type: ignore
    Account = None  # type: ignore
    LocalAccount = Any  # type: ignore
    _WEB3_AVAILABLE = False
    _WEB3_ERROR = _exc

try:
    # web3 v6
    from web3.middleware import ExtraDataToPOAMiddleware as POA_MIDDLEWARE
except ImportError:
    try:
        # web3 v5
        from web3.middleware import geth_poa_middleware as POA_MIDDLEWARE
    except ImportError:
        POA_MIDDLEWARE = None

from cache import CacheBalances, CacheTransfers
from services.token_catalog import get_core_token_map
from services.token_safety import TokenSafetyRegistry, enforce_token_safety
from services.wallet_optimizer import AdaptiveGasOracle
from services.providers.rpc_health import RpcHealthTracker


# =============================================================================
# Environment / constants
# =============================================================================

def load_env_robust() -> None:
    """Best-effort .env loader (cwd, script dir, file dir, home)."""
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)
        return

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
                load_dotenv(p, override=False)
                return
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


def wallet_secrets_available() -> bool:
    """Return True when either MNEMONIC or PRIVATE_KEY env var is present."""
    if os.getenv("MNEMONIC") and os.getenv("MNEMONIC").strip():
        return True
    if os.getenv("PRIVATE_KEY") and os.getenv("PRIVATE_KEY").strip():
        return True
    return False

REQ_KW = {"timeout": 30, "verify": certifi.where()}
JSON_HEADERS = {"accept": "application/json", "content-type": "application/json"}

ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY", "").strip()
ZEROX_API_KEY   = os.getenv("ZEROX_API_KEY", "").strip()  # optional for 0x
ONEINCH_API_KEY = os.getenv("ONEINCH_API_KEY", "").strip()  # required for 1inch v6
LIFI_BASE = os.getenv("LIFI_BASE", "https://li.quest/v1")
LIFI_API_KEY = os.getenv("LIFI_API_KEY", "").strip()

NATIVE = "0x0000000000000000000000000000000000000000"

ALCHEMY_SLUGS: Dict[str, str] = {
    "ethereum": "eth-mainnet",
    "base": "base-mainnet",
    "arbitrum": "arb-mainnet",
    "optimism": "opt-mainnet",
    "polygon": "polygon-mainnet",
    "bsc": "bnb-mainnet",
    "avalanche": "avax-mainnet",
}

ALCHEMY_ENV: Dict[str, str] = {
    "ethereum": "ALCHEMY_ETH_URL",
    "base": "ALCHEMY_BASE_URL",
    "arbitrum": "ALCHEMY_ARB_URL",
    "optimism": "ALCHEMY_OP_URL",
    "polygon": "ALCHEMY_POLY_URL",
    "bsc": "ALCHEMY_BSC_URL",
    "avalanche": "ALCHEMY_AVAX_URL",
}

CHAIN_NATIVE_SYMBOL: Dict[str, str] = {
    "ethereum": "ETH",
    "base": "ETH",
    "arbitrum": "ETH",
    "optimism": "ETH",
    "polygon": "MATIC",
    "bsc": "BNB",
    "avalanche": "AVAX",
    "zksync": "ETH",
}

NATIVE_SENTINELS = {
    NATIVE.lower(),
    "eth",
    "native",
    "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
}

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
    "bsc": {
        "id": 56, "poa": True,
        "rpcs": (
            _env(os.getenv("BSC_RPC_URL"))
            + [
                "https://bsc-dataseed.binance.org",
                "https://bsc.publicnode.com",
                "https://bscrpc.com",
            ]
        ),
    },
    "avalanche": {
        "id": 43114, "poa": False,
        "rpcs": (
            _env(os.getenv("AVAX_RPC_URL"))
            + [
                "https://api.avax.network/ext/bc/C/rpc",
                "https://avalanche-c-chain-rpc.publicnode.com",
                "https://avalanche.drpc.org",
            ]
        ),
    },
    "zksync": {
        "id": 324, "poa": False,
        "rpcs": (
            _env(os.getenv("ZKSYNC_RPC_URL"))
            + [
                "https://mainnet.era.zksync.io",
                "https://zksync.drpc.org",
            ]
        ),
    },
}

CHAIN_BY_ID: Dict[int, str] = {
    cfg["id"]: name for name, cfg in CHAINS.items() if isinstance(cfg.get("id"), int)
}

CORE_TOKEN_DISCOVERY = os.getenv("WALLET_INCLUDE_CORE_TOKENS", "1").strip().lower() not in {"0", "false", "no"}
CORE_TRACKED_TOKENS: Dict[str, Dict[str, str]] = get_core_token_map()

ERC20_ABI = [
    {"constant": True, "inputs": [], "name": "decimals", "outputs":[{"name":"","type":"uint8"}], "type":"function"},
    {"constant": True, "inputs": [{"name":"account","type":"address"}], "name":"balanceOf", "outputs":[{"name":"","type":"uint256"}], "type":"function"},
    {"constant": True, "inputs": [{"name":"owner","type":"address"},{"name":"spender","type":"address"}], "name":"allowance", "outputs":[{"name":"","type":"uint256"}], "type":"function"},
    {"constant": True, "inputs": [], "name": "symbol", "outputs":[{"name":"","type":"string"}], "type":"function"},
    {"constant": True, "inputs": [], "name": "name", "outputs":[{"name":"","type":"string"}], "type":"function"},
    {"constant": False, "inputs": [{"name":"recipient","type":"address"},{"name":"amount","type":"uint256"}], "name":"transfer", "outputs":[{"name":"","type":"bool"}], "type":"function"},
    {"constant": False, "inputs": [{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}], "name":"approve", "outputs":[{"name":"","type":"bool"}], "type":"function"},
]

_getctx().prec = 50


# =============================================================================
# Utility functions (pure; not patching)
# =============================================================================

_session: Optional[requests.Session] = None
def _http() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(JSON_HEADERS)
    return _session

def _gwei(n: float) -> int:
    return int(n * 1_000_000_000)

def _hexint(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    if isinstance(v, int):
        return v
    try:
        s = str(v).strip().lower()
        return int(s, 16) if s.startswith("0x") else int(s)
    except Exception:
        return default

def _normalize_addr(a: str) -> str:
    if not a: return a
    s = a.strip()
    if s.upper() in ("ETH", "NATIVE"): return NATIVE
    if not s.startswith("0x"): s = "0x" + s
    return Web3.to_checksum_address(s)

def _parse_watch_token_blob(blob: Optional[str]) -> Dict[str, List[str]]:
    """Parse WALLET_PINNED_TOKENS style env (chain:token,chain:0x...)."""
    result: Dict[str, List[str]] = {}
    if not blob:
        return result
    entries = [chunk.strip() for chunk in str(blob).split(",") if chunk.strip()]
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

def _to_decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(0)

def _decode_erc20_text(value: Any) -> str:
    """
    Decode ERC-20 string/name/symbol responses which can be bytes32, bytes, or str.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip("\x00")
    if isinstance(value, bytes):
        return value.replace(b"\x00", b"").decode("utf-8", errors="ignore").strip()
    try:
        from hexbytes import HexBytes  # type: ignore
        if isinstance(value, HexBytes):
            return value.replace(b"\x00", b"").decode("utf-8", errors="ignore").strip()
    except Exception:
        pass
    if isinstance(value, (bytearray, memoryview)):
        try:
            return bytes(value).replace(b"\x00", b"").decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""
    if isinstance(value, int):
        try:
            return bytes.fromhex(hex(value)[2:]).replace(b"\x00", b"").decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""
    return str(value)

def _dec_to_str_plain(x: Decimal, max_dp: int) -> str:
    s = format(x, "f")
    if "." in s:
        intp, frac = s.split(".", 1)
        frac = frac[:max_dp].rstrip("0")
        return intp if not frac else f"{intp}.{frac}"
    return s


# =============================================================================
# Dataclasses
# =============================================================================

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


# =============================================================================
# UltraSwapBridge (clean OOP)
# =============================================================================

class UltraSwapBridge:
    """
    - Wallet management (mnemonic/private key)
    - RPC management per chain (Alchemy/Infura/public)
    - Portfolio discovery (Alchemy balances + transfers fallback)
    - ERC-20 helpers
    - Aggregators: 0x v2 Allowance-Holder, 1inch v6, OpenOcean v4
    - Local DEXes: Uniswap V3 (Router02), Camelot V2 (Arbitrum)
    - LI.FI quote and execute
    """

    # ------------------------ Init / Web3 ------------------------

    def __init__(
        self,
        mnemonic: Optional[str] = None,
        private_key: Optional[str] = None,
        derivation_path: str = "m/44'/60'/0'/0/0",
        cache_transfers: Optional[CacheTransfers] = None,
        reorg_safety_blocks: Optional[int] = None,
    ):
        if not _WEB3_AVAILABLE:
            raise ImportError(
                "web3 and eth-account are required for blockchain operations. Install with 'pip install web3 eth-account'."
            ) from _WEB3_ERROR

        mnemonic = mnemonic or os.getenv("MNEMONIC")
        private_key = private_key or os.getenv("PRIVATE_KEY")
        if mnemonic:
            Account.enable_unaudited_hdwallet_features()
            self.acct: LocalAccount = Account.from_mnemonic(mnemonic, account_path=derivation_path)
            self.pk = self.acct.key
        elif private_key:
            self.acct = Account.from_key(private_key)
            self.pk = self.acct.key
        else:
            raise ValueError("Provide MNEMONIC or PRIVATE_KEY (in .env or constructor).")

        self.ct: Optional[CacheTransfers] = cache_transfers
        self.reorg_safety = (
            int(os.getenv("PORTFOLIO_REORG_SAFETY", "12"))
            if reorg_safety_blocks is None else int(reorg_safety_blocks)
        )
        self._rpc_clients: Dict[str, Tuple[Web3, str]] = {}
        self._rpc_latency: Dict[str, float] = {}
        # Adaptive gas oracle keeps EIP-1559 tips reasonable while avoiding stalls.
        ttl_env = os.getenv("GAS_ORACLE_TTL", "15")
        sample_env = os.getenv("GAS_ORACLE_SAMPLE", "8")
        try:
            oracle_ttl = max(3, int(ttl_env))
        except Exception:
            oracle_ttl = 15
        try:
            oracle_sample = max(4, int(sample_env))
        except Exception:
            oracle_sample = 8
        percentile_map = None
        pct_blob = os.getenv("GAS_ORACLE_PERCENTILES")
        if pct_blob:
            try:
                raw = json.loads(pct_blob)
                percentile_map = {str(k).lower(): float(v) for k, v in raw.items()}
            except Exception:
                percentile_map = None
        self._gas_oracle = AdaptiveGasOracle(
            ttl_sec=oracle_ttl,
            sample_size=oracle_sample,
            percentile_map=percentile_map,
            default_strategy=os.getenv("GAS_STRATEGY_DEFAULT", "balanced"),
        )
        safety_flag = os.getenv("TOKEN_SAFETY_ENFORCE", "1").strip().lower() not in {"0", "false", "no"}
        self._token_safety = TokenSafetyRegistry() if safety_flag else None
        self._token_safety_enabled = safety_flag
        self._gas_scope_defaults = self._load_gas_scope_defaults()

    def get_address(self) -> str:
        return self.acct.address

    def _alchemy_url(self, chain: str) -> str:
        env_key = ALCHEMY_ENV.get(chain, "")
        url = (os.getenv(env_key) or "").strip()
        if url:
            return url
        if ALCHEMY_API_KEY and chain in ALCHEMY_SLUGS:
            return f"https://{ALCHEMY_SLUGS[chain]}.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        return ""

    def _w3(self, chain: str) -> Web3:
        """Return a healthy Web3 connection, preferring the fastest known RPC per chain."""
        cfg = CHAINS[chain]
        cached = self._rpc_clients.get(chain)
        if cached:
            w3, url = cached
            try:
                if w3.is_connected():
                    return w3
            except Exception:
                pass
            self._rpc_clients.pop(chain, None)

        urls = [u for u in cfg["rpcs"] if u]
        if not urls:
            raise RuntimeError(f"No RPC URLs configured for {chain}")
        urls.sort(key=lambda u: self._rpc_latency.get(u, float("inf")))

        errs: List[str] = []
        best_choice: Optional[Tuple[float, Web3, str]] = None
        for url in urls:
            start = time.perf_counter()
            try:
                provider = Web3.HTTPProvider(url, request_kwargs=REQ_KW)
                w3 = Web3(provider)
                if not w3.is_connected():
                    raise RuntimeError("not connected")
                elapsed = max(time.perf_counter() - start, 1e-3)
                if cfg.get("poa") and POA_MIDDLEWARE:
                    w3.middleware_onion.inject(POA_MIDDLEWARE, layer=0)
                self._rpc_latency[url] = elapsed
                best_choice = (elapsed, w3, url)
                break  # first healthy URL after sorting is the winner
            except Exception as e:
                self._rpc_latency[url] = float("inf")
                errs.append(f"{url} -> {type(e).__name__}: {e}")
                continue

        if not best_choice:
            raise RuntimeError(f"RPC not reachable for {chain}. Tried:\n" + "\n".join(errs))

        _, w3_obj, good_url = best_choice
        self._rpc_clients[chain] = (w3_obj, good_url)
        return w3_obj

    # ------------------------ Fees / Sending ------------------------

    def _gas_urgency(self, scope: Optional[str] = None) -> Optional[str]:
        if not scope:
            return None
        key = f"GAS_STRATEGY_{scope.upper()}"
        val = os.getenv(key)
        if not val:
            return None
        trimmed = val.strip().lower()
        return trimmed or None

    def _load_gas_scope_defaults(self) -> Dict[str, str]:
        """
        Optional JSON or comma-delimited mapping for default gas urgencies per scope.
        Examples:
            GAS_SCOPE_DEFAULTS='{"send": "eco", "swap": "balanced"}'
            GAS_SCOPE_DEFAULTS='send:eco,swap:balanced'
        """
        blob = os.getenv("GAS_SCOPE_DEFAULTS")
        if not blob:
            return {}
        mapping: Dict[str, str] = {}
        raw = blob.strip()
        try:
            if raw.startswith("{"):
                data = json.loads(raw)
                if isinstance(data, dict):
                    for scope, urgency in data.items():
                        scope_key = str(scope).strip().lower()
                        urg_val = str(urgency).strip().lower()
                        if scope_key and urg_val:
                            mapping[scope_key] = urg_val
                    return mapping
        except Exception:
            mapping.clear()
        parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
        for part in parts:
            if ":" not in part:
                continue
            scope, urgency = part.split(":", 1)
            scope_key = scope.strip().lower()
            urg_val = urgency.strip().lower()
            if scope_key and urg_val:
                mapping[scope_key] = urg_val
        return mapping

    # ------------------------ Safety helpers ------------------------

    def ensure_token_safe(self, chain: str, token: str) -> None:
        """
        Enforce the token safety registry (no-op for non-address inputs).
        """
        enforce_token_safety(chain, token, self._token_safety)

    def _resolve_scope_urgency(self, scope: Optional[str], explicit: Optional[str]) -> Optional[str]:
        """
        Priority:
          1. Explicit urgency argument.
          2. GAS_SCOPE_<SCOPE> env variable.
          3. GAS_SCOPE_DEFAULTS map entry.
        """
        if explicit:
            trimmed = explicit.strip().lower()
            return trimmed or None
        if not scope:
            return None
        env_key = f"GAS_SCOPE_{scope.upper()}"
        env_val = os.getenv(env_key)
        if env_val:
            trimmed = env_val.strip().lower()
            if trimmed:
                return trimmed
        default = self._gas_scope_defaults.get(scope.lower())
        if default:
            trimmed = str(default).strip().lower()
            return trimmed or None
        return None

    def _apply_fee_strategy(
        self,
        chain: str,
        w3,
        tx: MutableMapping[str, Any],
        *,
        scope: Optional[str] = None,
        urgency: Optional[str] = None,
    ) -> Dict[str, int]:
        resolved = self._resolve_scope_urgency(scope, urgency)
        return self._gas_oracle.apply_to_tx(chain, w3, tx, urgency=resolved)

    def _legacy_fee_strategy(self, w3) -> Dict[str, int]:
        gp_override = os.getenv("GAS_PRICE_GWEI")
        if gp_override:
            return {"gasPrice": _gwei(float(gp_override))}

        tip_default = float(os.getenv("GAS_TIP_GWEI", "3"))
        tip_floor = float(os.getenv("GAS_TIP_FLOOR_GWEI", "1"))
        mult = float(os.getenv("GAS_BASE_MULT", "2.0"))

        try:
            hist = w3.eth.fee_history(6, "pending", [25, 50, 75, 90])
            base = int((hist.get("baseFeePerGas") or [0])[-1])
            rewards = [max(r[-2:]) for r in hist.get("reward", []) if r]
            recent_tip = int(sum(rewards) / max(1, len(rewards))) if rewards else 0

            tip = max(_gwei(tip_default), recent_tip, _gwei(tip_floor))
            max_fee = int(base * mult) + tip
            if max_fee < base + tip:
                max_fee = base + tip
            return {"maxFeePerGas": max_fee, "maxPriorityFeePerGas": tip}
        except Exception:
            try:
                gp = int(w3.eth.gas_price)
            except Exception:
                gp = _gwei(30)
            return {"gasPrice": gp}

    def _suggest_fees(
        self,
        chain: str,
        w3,
        *,
        urgency: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> Dict[str, int]:
        tx: Dict[str, Any] = {}
        try:
            return self._apply_fee_strategy(chain, w3, tx, scope=scope, urgency=urgency)
        except Exception:
            return self._legacy_fee_strategy(w3)


    def send_prebuilt_tx(
        self,
        chain: str,
        to: str,
        data: str,
        *,
        value: int = 0,
        gas: int | None = None,
        fee_scope: Optional[str] = None,
        fee_urgency: Optional[str] = None,
    ):
        """
        Generic sender for locally-built transactions.
        - Always ensures a gas value (estimate when not provided).
        - Fills sensible EIP-1559 fees when needed.
        - Prints a brief send log and waits (with timeout) for a receipt.
        """
        import os
        # Support either _w3 or _rb__w3 depending on which you kept
        w3 = self._w3(chain) if hasattr(self, "_w3") else self._rb__w3(chain)
        acct = self.acct
        to_cs = w3.to_checksum_address(to)
        data_hex = data or "0x"
        val = int(value or 0)

        tx = {
            "from": acct.address,
            "to":   to_cs,
            "value": val,
            "data":  data_hex,
            "nonce": w3.eth.get_transaction_count(acct.address, "pending"),
            "chainId": int(w3.eth.chain_id),
        }

        # --- gas (estimate if not provided) ---
        if gas is None:
            try:
                est = int(w3.eth.estimate_gas({"from": tx["from"], "to": tx["to"], "value": tx["value"], "data": tx["data"]}))
            except Exception:
                # WETH deposit is ~23–30k; use a safe fallback. Otherwise a conservative default.
                is_weth_deposit = (data_hex.lower().startswith("0xd0e30db0"))  # deposit()
                est = 60000 if is_weth_deposit else 250000
            tx["gas"] = int(est * 1.2)
        else:
            tx["gas"] = int(gas)

        # --- fees (prefer adaptive oracle; fallback handled inside helper) ---
        tx.update(self._suggest_fees(chain, w3, scope=fee_scope, urgency=fee_urgency))

        # Sign & send
        signed = w3.eth.account.sign_transaction(tx, private_key=acct.key)
        raw = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None) or signed
        txh = w3.eth.send_raw_transaction(raw)

        print(
            f"[send_prebuilt_tx] chainId={tx['chainId']} to={to_cs} val={tx['value']} "
            f"gas={tx.get('gas')} mfpg={tx.get('maxFeePerGas')} "
            f"mpfpg={tx.get('maxPriorityFeePerGas')} gp={tx.get('gasPrice')} dataLen={len(data_hex)}"
        )

        timeout_s = int(os.getenv("TX_TIMEOUT_SEC", "120"))
        try:
            rc = w3.eth.wait_for_transaction_receipt(txh, timeout=timeout_s)
            print(f"[send_prebuilt_tx] mined status={rc.get('status')} gasUsed={rc.get('gasUsed')}")
        except Exception as e:
            print(f"[send_prebuilt_tx] wait timeout/pending: {e!r}")

        return w3.to_hex(txh)


    def send_prebuilt_tx_from_0x(
        self,
        chain: str,
        txobj: dict,
        *,
        fee_scope: Optional[str] = None,
        fee_urgency: Optional[str] = None,
    ):
        """
        Send a tx exactly as 0x v2 returned it (to, data, value, gas?, fee fields?).
        - Ensures 'gas' by estimating if missing.
        - Respects provided fee fields; otherwise fills EIP-1559/legacy as needed.
        """
        import os
        w3 = self._w3(chain) if hasattr(self, "_w3") else self._rb__w3(chain)
        acct = self.acct

        to    = txobj.get("to")
        data  = txobj.get("data") or "0x"
        value = txobj.get("value") or 0
        gas   = txobj.get("gas")
        if not to or not data:
            raise ValueError("0x tx missing 'to' or 'data'")

        to_cs = w3.to_checksum_address(to)
        val   = int(value if not (isinstance(value, str) and value.startswith("0x")) else int(value, 16))

        tx = {
            "from": acct.address,
            "to":   to_cs,
            "value": val,
            "data":  data,
            "nonce": w3.eth.get_transaction_count(acct.address, "pending"),
            "chainId": int(w3.eth.chain_id),
        }

        # --- gas: use provided, else estimate, else quote hint, else fallback ---
        if gas is not None:
            tx["gas"] = int(gas if not (isinstance(gas, str) and gas.startswith("0x")) else int(gas, 16))
        else:
            # try to estimate on-chain
            try:
                est = int(w3.eth.estimate_gas({"from": tx["from"], "to": tx["to"], "value": tx["value"], "data": tx["data"]}))
            except Exception:
                # try hints in 0x payload
                hint = txobj.get("estimatedGas") or txobj.get("gas")
                if isinstance(hint, str) and hint.startswith("0x"):
                    try:
                        hint = int(hint, 16)
                    except Exception:
                        hint = None
                est = int(hint or 250000)
            tx["gas"] = int(est * 1.2)

        # --- fees: honor provided, else fill ---
        mfpg  = txobj.get("maxFeePerGas")
        mpfpg = txobj.get("maxPriorityFeePerGas")
        gp    = txobj.get("gasPrice")
        try:
            if mfpg is not None and mpfpg is not None:
                tx["maxFeePerGas"] = int(mfpg if not (isinstance(mfpg, str) and mfpg.startswith("0x")) else int(mfpg, 16))
                tx["maxPriorityFeePerGas"] = int(
                    mpfpg if not (isinstance(mpfpg, str) and mpfpg.startswith("0x")) else int(mpfpg, 16)
                )
            elif gp is not None:
                tx["gasPrice"] = int(gp if not (isinstance(gp, str) and gp.startswith("0x")) else int(gp, 16))
            else:
                tx.update(self._suggest_fees(chain, w3, scope=fee_scope, urgency=fee_urgency))
        except Exception:
            tx["gasPrice"] = int(getattr(w3.eth, "gas_price", w3.to_wei(10, "gwei")))

        signed = w3.eth.account.sign_transaction(tx, private_key=acct.key)
        raw = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None) or signed
        txh = w3.eth.send_raw_transaction(raw)

        print(
            f"[send_prebuilt_tx_from_0x] chainId={tx['chainId']} to={tx['to']} val={tx['value']} "
            f"gas={tx.get('gas')} mfpg={tx.get('maxFeePerGas')} mpfpg={tx.get('maxPriorityFeePerGas')} "
            f"gp={tx.get('gasPrice')} dataLen={len((data or '0x'))}"
        )

        timeout_s = int(os.getenv("TX_TIMEOUT_SEC", "120"))
        try:
            rc = w3.eth.wait_for_transaction_receipt(txh, timeout=timeout_s)
            print(f"[send_prebuilt_tx_from_0x] mined status={rc.get('status')} gasUsed={rc.get('gasUsed')}")
        except Exception as e:
            print(f"[send_prebuilt_tx_from_0x] wait timeout/pending: {e!r}")

        return w3.to_hex(txh)


    # ------------------------ ERC-20 helpers ------------------------

    def _erc20(self, w3: Web3, addr: str):
        return w3.eth.contract(address=Web3.to_checksum_address(addr), abi=ERC20_ABI)

    def erc20_decimals(self, chain: str, token: str) -> int:
        w3 = self._w3(chain)
        if token.lower() == NATIVE:
            return 18
        try:
            return int(self._erc20(w3, token).functions.decimals().call())
        except Exception:
            return 18

    def erc20_balance_of(self, chain: str, token: str, owner: str) -> int:
        w3 = self._w3(chain)
        return int(self._erc20(w3, token).functions.balanceOf(Web3.to_checksum_address(owner)).call())

    def erc20_allowance(self, chain: str, token: str, owner: str, spender: str) -> int:
        w3 = self._w3(chain)
        return int(self._erc20(w3, token).functions.allowance(
            Web3.to_checksum_address(owner), Web3.to_checksum_address(spender)
        ).call())

    def erc20_symbol(self, chain: str, token: str) -> str:
        w3 = self._w3(chain)
        try:
            raw = self._erc20(w3, token).functions.symbol().call()
            return _decode_erc20_text(raw)
        except Exception:
            return ""

    def erc20_name(self, chain: str, token: str) -> str:
        w3 = self._w3(chain)
        try:
            raw = self._erc20(w3, token).functions.name().call()
            return _decode_erc20_text(raw)
        except Exception:
            return ""

    def approve_erc20(self, chain: str, token: str, spender: str, amount: int) -> str:
        w3 = self._w3(chain)
        token_cs = Web3.to_checksum_address(token)
        spend_cs = Web3.to_checksum_address(spender)
        sender = self.acct.address
        c = self._erc20(w3, token_cs)
        base = {
            "chainId": w3.eth.chain_id,
            "from": sender,
            "nonce": w3.eth.get_transaction_count(sender, "pending"),
            "value": 0,
        }
        base.update(self._suggest_fees(chain, w3, scope="approve"))
        try:
            est = c.functions.approve(spend_cs, int(amount)).estimate_gas({"from": sender})
        except Exception:
            est = 60000
        gas_final = int(est * 1.2)
        tx = c.functions.approve(spend_cs, int(amount)).build_transaction({**base, "gas": gas_final})
        signed = self.acct.sign_transaction(tx)
        raw = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None) or getattr(signed, "raw", None) or signed
        txh = w3.eth.send_raw_transaction(raw)
        return w3.to_hex(txh)

    def send_erc20(self, chain: str, token: str, to: str, amount: int, *, gas: Optional[int] = None) -> str:
        """
        Transfer ERC-20 `amount` (base units) to `to` on `chain`.
        Respects EIP-1559 fees and estimates gas when not provided.
        """
        w3 = self._w3(chain)
        token_cs = Web3.to_checksum_address(token)
        to_cs = Web3.to_checksum_address(to)
        sender = self.acct.address

        fn = self._erc20(w3, token_cs).functions.transfer(to_cs, int(amount))
        base_tx = {
            "chainId": w3.eth.chain_id,
            "from": sender,
            "nonce": w3.eth.get_transaction_count(sender, "pending"),
            "value": 0,
            **self._suggest_fees(chain, w3, scope="send"),
        }

        if gas is None:
            try:
                est = fn.estimate_gas({"from": sender})
            except Exception:
                est = 65000
            gas = int(est * 1.2)

        tx = fn.build_transaction({**base_tx, "gas": int(gas)})
        signed = self.acct.sign_transaction(tx)
        raw = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None) or getattr(signed, "raw", None) or signed
        txh = w3.eth.send_raw_transaction(raw)
        return w3.to_hex(txh)

    # ------------------------ Portfolio discovery ------------------------

    def _alchemy_post(self, url: str, method: str, params: Any) -> Dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        for attempt in range(3):
            r = _http().post(url, json=payload, timeout=60, verify=certifi.where())
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(0.4 * (attempt + 1))
                continue
            r.raise_for_status()
            j = r.json()
            if "error" in j:
                raise requests.HTTPError(str(j["error"]))
            return j
        r.raise_for_status()
        return r.json()

    def _discover_via_balances(self, url: str, min_balance_wei: int = 1) -> List[str]:
        seen: List[str] = []
        page_key: Optional[str] = None
        while True:
            params: List[Any] = [self.acct.address, "erc20"]
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

        addrs: set[str] = set()
        if self.ct is not None:
            try:
                self.ct.merge_new(wallet, chain, new_pages)
                all_transfers = self.ct.get_all(wallet, chain) or []
                for t in all_transfers:
                    rc = (t.get("rawContract") or {}).get("address")
                    if rc:
                        try: addrs.add(Web3.to_checksum_address(rc))
                        except Exception: pass
                if baseline_last >= 0:
                    changed = self.ct.touched_tokens_since(wallet, chain, since_block=baseline_last) or set()
                    if changed:
                        print(f"[discover/transfers] {chain}: tokens changed since {baseline_last}: {len(changed)}")
            except Exception as e:
                print(f"[discover/transfers] {chain}: cache merge error {type(e).__name__}: {e}")
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
        """Return [(chain, checksum_token_address), ...] for non-zero ERC-20 balances."""
        chains = list(chains) if chains else list(CHAINS.keys())
        out: List[Tuple[str, str]] = []

        for ch in chains:
            addr_set: set[str] = set()

            try:
                cb = CacheBalances()
                cached_map = (cb.get_state(self.acct.address, ch) or {}).get("tokens", {}) or {}
                for addr, meta in cached_map.items():
                    raw = meta.get("balance_hex") or meta.get("raw") or "0x0"
                    try:
                        raw_int = int(str(raw), 16) if str(raw).startswith("0x") else int(str(raw))
                    except Exception:
                        raw_int = 0
                    if raw_int < int(min_balance_wei):
                        continue
                    quantity = meta.get("quantity")
                    if quantity is not None:
                        try:
                            if float(quantity) <= 0:
                                continue
                        except Exception:
                            pass
                    try:
                        addr_set.add(Web3.to_checksum_address(addr))
                    except Exception:
                        continue
                if cached_map:
                    print(f"[discover] {ch}: seed {len(addr_set)} tokens from cache")
            except Exception:
                pass

            url = self._alchemy_url(ch)
            if url:
                try:
                    remote = self._discover_via_balances(url, min_balance_wei=min_balance_wei)
                    addr_set.update(remote)
                    print(f"[discover] {ch}: found {len(remote)} tokens via balances")
                except Exception as e:
                    print(f"[discover] {ch}: balances error {type(e).__name__}: {e}")
                    try:
                        remote = self._discover_via_transfers(url, chain=ch)
                        addr_set.update(remote)
                        print(f"[discover] {ch}: found {len(remote)} tokens via transfers (cached)")
                    except Exception as e2:
                        print(f"[discover] {ch}: transfer fallback error {type(e2).__name__}: {e2}")
            else:
                print(f"[discover] {ch}: no Alchemy URL")

            try:
                w3 = self._w3(ch)
                native_bal = w3.eth.get_balance(self.acct.address)
                if native_bal >= int(min_balance_wei):
                    addr_set.add(NATIVE)
            except Exception:
                pass

            if CORE_TOKEN_DISCOVERY:
                extras = self._discover_core_tokens(ch, addr_set, min_balance_wei)
                if extras:
                    print(f"[discover] {ch}: added {len(extras)} core tokens with balance")
                    addr_set.update(extras)

            out.extend((ch, addr) for addr in sorted(addr_set))

        return out

    def _discover_core_tokens(self, chain: str, existing: Iterable[str], min_balance_wei: int) -> List[str]:
        """Ensure common stables/majors are tracked even if not seen in cache yet."""
        candidates = CORE_TRACKED_TOKENS.get(chain)
        if not candidates:
            return []
        have = {addr.lower() for addr in existing}
        additions: List[str] = []
        holder = self.acct.address
        for symbol, addr in candidates.items():
            addr_lower = addr.lower()
            if addr_lower in have:
                continue
            try:
                bal = self.erc20_balance_of(chain, addr, holder)
            except Exception:
                continue
            if bal >= int(min_balance_wei):
                try:
                    additions.append(Web3.to_checksum_address(addr))
                except Exception:
                    continue
        return additions

    def discover_tokens(self, style: str = "portfolio", chains: Optional[Iterable[str]] = None) -> List[str]:
        """
        style='portfolio' -> base uses '0x...@base', others 'chain:0x...'
        style='prefix'    -> 'chain:0x...'
        style='suffix'    -> '0x...@chain'
        style='pairs'     -> returns [('chain','0x...'), ...]
        """
        pairs = self.discover_tokens_pairs(chains=chains)
        if style == "pairs":
            return pairs  # type: ignore[return-value]

        if style == "prefix":
            annotated = [f"{ch}:{addr}" for ch, addr in pairs]
        elif style == "suffix":
            annotated = [f"{addr}@{ch}" for ch, addr in pairs]
        else:
            out: List[str] = []
            for ch, addr in pairs:
                ch_l = (ch or "").lower()
                out.append(f"{addr}@base" if ch_l == "base" else f"{ch_l}:{addr}")
            annotated = out

        return list(dict.fromkeys(annotated))

    # ------------------------ Pricing (optional 0x fill-ins) ------------------------

    def _price_usd_0x_single(self, chain: str, token: str, decimals: int) -> Decimal:
        """Simple 0x price for USD; try sell 1 token → USDC; else invert 1 USDC."""
        base = "https://api.0x.org/swap/v1/price"
        headers = {"accept": "application/json"}
        if ZEROX_API_KEY:
            headers["0x-api-key"] = ZEROX_API_KEY
            headers["0x-version"] = "v2"

        chain_id = CHAINS[chain]["id"]
        addr = _normalize_addr(token)
        price = Decimal(0)

        # sellAmount = 1 token
        try:
            sell_amount = str(10 ** min(int(decimals), 36))
        except Exception:
            sell_amount = str(10 ** 18)

        try:
            r = _http().get(
                base,
                params={"chainId": str(chain_id), "sellToken": addr, "buyToken": "USDC", "sellAmount": sell_amount},
                headers=headers, timeout=20, verify=certifi.where()
            )
            if r.status_code in (429, 500, 502, 503, 504):
                r.raise_for_status()
            j = r.json()
            if "price" in j:
                price = Decimal(str(j["price"]))
            elif "buyAmount" in j:
                usdc = Decimal(j["buyAmount"]) / Decimal(10**6)
                price = usdc
        except Exception:
            price = Decimal(0)

        if price <= 0:
            try:
                r = _http().get(
                    base,
                    params={"chainId": str(chain_id), "sellToken": addr, "buyToken": "USDC", "buyAmount": str(10**6)},
                    headers=headers, timeout=20, verify=certifi.where()
                )
                if r.status_code in (429, 500, 502, 503, 504):
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

    def enrich_portfolio_with_0x(self, snapshot: Dict[str, Dict[str, Any]], addr_to_chain: Dict[str, str], qty_dp: int = 18, usd_dp: int = 8) -> None:
        """Fill missing USD using 0x (non-intrusive)."""
        missing_by_chain: Dict[str, List[str]] = {}
        for addr, info in snapshot.items():
            usd = _to_decimal(info.get("usd_amount", "0"))
            if usd <= 0:
                ch = addr_to_chain.get(addr.lower())
                if ch:
                    missing_by_chain.setdefault(ch, []).append(addr)

        for ch, addrs in missing_by_chain.items():
            if not addrs:
                continue
            try:
                _ = self._w3(ch)  # ensure RPC ok
            except Exception:
                continue
            for a in addrs:
                try:
                    dec = self.erc20_decimals(ch, a)
                    px = self._price_usd_0x_single(ch, a, dec)
                    if px > 0:
                        qty = _to_decimal(snapshot[a].get("quantity", "0"))
                        usd_total = (qty * px)
                        snapshot[a]["usd_amount"] = _dec_to_str_plain(usd_total, usd_dp)
                except Exception:
                    pass

    # ------------------------ LI.FI ------------------------

    def _lifi_quote(self, params: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"accept": "application/json"}
        if LIFI_API_KEY:
            headers["x-lifi-api-key"] = LIFI_API_KEY
        r = _http().get(f"{LIFI_BASE}/quote", params=params, timeout=30, verify=certifi.where(), headers=headers)
        r.raise_for_status()
        return r.json()

    def _lifi_status(self, tx_hash: str) -> Dict[str, Any]:
        r = _http().get(f"{LIFI_BASE}/status", params={"txHash": tx_hash}, timeout=30, verify=certifi.where())
        r.raise_for_status()
        return r.json()

    def quote(self, src_chain: str, dst_chain: str, from_token: str, to_token: str, amount: float, slippage: float = 0.003) -> QuotePreview:
        from_token = _normalize_addr(from_token)
        to_token = _normalize_addr(to_token)
        w3 = self._w3(src_chain)
        dec = 18 if from_token.lower() == NATIVE else self.erc20_decimals(src_chain, from_token)
        from_amt = int(round(amount * (10 ** dec)))
        params = {
            "fromChain": CHAINS[src_chain]["id"], "toChain": CHAINS[dst_chain]["id"],
            "fromToken": from_token, "toToken": to_token,
            "fromAmount": str(from_amt), "fromAddress": self.acct.address,
            "slippage": slippage,
        }
        q = self._lifi_quote(params)
        est, act = q["estimate"], q["action"]
        gas_usd = None
        try:
            if est.get("gasCosts"):
                gas_usd = sum(float(x.get("amountUSD") or 0) for x in est["gasCosts"])
        except Exception:
            pass
        return QuotePreview(
            from_chain=act["fromChainId"], to_chain=act["toChainId"],
            from_symbol=act["fromToken"]["symbol"], to_symbol=act["toToken"]["symbol"],
            from_amount=int(est["fromAmount"]) / float(10 ** act["fromToken"]["decimals"]),
            to_amount=int(est["toAmount"]) / float(10 ** act["toToken"]["decimals"]),
            to_amount_min=int(est.get("toAmountMin", est["toAmount"])) / float(10 ** act["toToken"]["decimals"]),
            gas_usd=gas_usd, tx_request=q.get("transactionRequest") or {},
        )

    def execute(
        self,
        src_chain: str,
        dst_chain: str,
        from_token: str,
        to_token: str,
        amount: float,
        slippage: float = 0.003,
        wait: bool = False,
    ) -> Dict[str, Any]:
        w3 = self._w3(src_chain)
        from_token = _normalize_addr(from_token)
        to_token   = _normalize_addr(to_token)

        # --- quote via LI.FI ---
        pv = self.quote(src_chain, dst_chain, from_token, to_token, amount, slippage)
        txreq = pv.tx_request
        if not txreq:
            raise RuntimeError("LI.FI returned no transactionRequest to execute.")

        # --- approve if selling ERC-20 ---
        if from_token.lower() != NATIVE:
            needed = int(round(amount * (10 ** self.erc20_decimals(src_chain, from_token))))
            spender = Web3.to_checksum_address(txreq["to"])
            current = self.erc20_allowance(src_chain, from_token, self.acct.address, spender)
            if current < needed:
                txh = self.approve_erc20(src_chain, from_token, spender, needed)
                try:
                    w3.eth.wait_for_transaction_receipt(txh, timeout=240)
                except Exception:
                    pass

        # --- build tx (honor EIP-1559, estimate gas if needed) ---
        to_addr = Web3.to_checksum_address(txreq["to"])
        value   = _hexint(txreq.get("value"), 0)

        data = txreq["data"]
        if isinstance(data, str) and data.startswith("0x"):
            data_bytes = bytes.fromhex(data[2:])
        elif isinstance(data, (bytes, bytearray)):
            data_bytes = bytes(data)
        else:
            raise ValueError("transactionRequest.data missing/invalid")

        tx = {
            "to": to_addr,
            "from": self.acct.address,
            "data": data_bytes,
            "value": int(value),
            "chainId": int(w3.eth.chain_id),
            "nonce": w3.eth.get_transaction_count(self.acct.address, "pending"),
        }
        if not any(k in txreq for k in ("maxFeePerGas", "gasPrice")):
            tx.update(self._suggest_fees(src_chain, w3, scope="bridge"))
        else:
            if "gasPrice" in txreq:
                tx["gasPrice"] = _hexint(txreq.get("gasPrice"), 0)
            if "maxFeePerGas" in txreq:
                tx["maxFeePerGas"] = _hexint(txreq.get("maxFeePerGas"), 0)
            if "maxPriorityFeePerGas" in txreq:
                tx["maxPriorityFeePerGas"] = _hexint(txreq.get("maxPriorityFeePerGas"), 0)

        gl = txreq.get("gasLimit")
        if gl:
            tx["gas"] = _hexint(gl, 0)
        else:
            # Estimate without fee fields interfering
            est_tx = dict(tx)
            est_tx.pop("maxFeePerGas", None)
            est_tx.pop("maxPriorityFeePerGas", None)
            est_tx.pop("gasPrice", None)
            try:
                tx["gas"] = int(w3.eth.estimate_gas(est_tx) * 1.12)
            except Exception:
                tx["gas"] = 250000

        # --- preflight (non-fatal) ---
        try:
            w3.eth.call({"from": tx["from"], "to": tx["to"], "data": tx["data"], "value": tx["value"]}, "latest")
        except Exception as e:
            print(f"[lifi preflight] warning: {e!r}")

        # --- sign & send (robust raw tx attr handling) ---
        stx = self.acct.sign_transaction(tx)
        raw = getattr(stx, "rawTransaction", None) or getattr(stx, "raw_transaction", None) or getattr(stx, "raw", None) or stx
        txh = w3.eth.send_raw_transaction(raw)
        tx_hash = txh.hex() if hasattr(txh, "hex") else w3.to_hex(txh)

        out: Dict[str, Any] = {"preview": pv.__dict__, "txHash": tx_hash}

        # --- optional bridge status polling ---
        if pv.from_chain != pv.to_chain and wait:
            while True:
                try:
                    s = self._lifi_status(tx_hash)
                    out["bridgeStatus"] = s
                    if s.get("status") in ("DONE", "FAILED"):
                        break
                except Exception:
                    pass
                time.sleep(20)

        return out


    # ------------------------ 0x v2 (Allowance-Holder) ------------------------

    @staticmethod
    def _0x_base_url(chain: str) -> str:
        m = {
            "ethereum": "https://api.0x.org",
            "base": "https://base.api.0x.org",
            "arbitrum": "https://arbitrum.api.0x.org",
            "optimism": "https://optimism.api.0x.org",
            "polygon": "https://polygon.api.0x.org",
        }
        return m.get((chain or "").lower(), "")

    def get_0x_quote_v2(self, chain: str, sell_token: str, buy_token: str, sell_amount_raw: int, slippage_bps: int = 100) -> dict:
        """
        0x v2 Allowance-Holder quote (returns tx + allowanceTarget).
        Docs: https://docs.0x.org/ (v2 Swap API)
        """
        base = self._0x_base_url(chain)
        if not base:
            raise RuntimeError(f"0x not supported for chain '{chain}'")

        params = {
            "chainId": CHAINS[chain]["id"],
            "sellToken": sell_token,
            "buyToken": buy_token,
            "sellAmount": str(int(sell_amount_raw)),
            "taker": self.acct.address,
            "slippageBps": int(slippage_bps),
        }
        headers = {"Accept": "application/json", "0x-version": "v2"}
        if ZEROX_API_KEY:
            headers["0x-api-key"] = ZEROX_API_KEY

        r = requests.get(f"{base}/swap/allowance-holder/quote", params=params, headers=headers, timeout=25)
        if r.status_code >= 400:
            msg = r.text
            if r.status_code == 401:
                msg += "  (Hint: set ZEROX_API_KEY env)"
            raise RuntimeError(f"0x v2 {r.status_code}: {msg}")
        q = r.json()

        txo  = q.get("transaction") or {}
        to   = txo.get("to") or q.get("to")
        data = txo.get("data") or q.get("data")
        value= txo.get("value") or q.get("value") or 0
        gas  = txo.get("gas") or q.get("estimatedGas") or q.get("gas") or 0

        allowance = q.get("allowanceTarget")
        if not allowance:
            try:
                allowance = (q.get("issues") or {}).get("allowance", {}).get("spender")
            except Exception:
                allowance = None

        if not to or not data:
            raise RuntimeError("0x v2 quote missing tx fields")

        return {
            "aggregator": "0x-v2",
            "to": to, "data": data, "value": value,
            "gas": gas,
            "allowanceTarget": allowance,
            "sellToken": sell_token, "buyToken": buy_token,
            "sellAmountRaw": int(sell_amount_raw),
            "sellAmount": q.get("sellAmount") or str(int(sell_amount_raw)),
            "buyAmount": q.get("buyAmount"),
            "issues": q.get("issues"), "route": q.get("route"),
        }

    def send_swap_via_0x_v2(self, chain: str, quote: dict) -> str:
        """
        Broadcast a 0x v2 Allowance-Holder swap:
        - auto-approval if needed
        - preflight via eth_call
        - EIP-1559 fees
        """
        w3 = self._w3(chain)
        q = dict(quote or {})

        # strict sell amount
        sell_amt = q.get("sellAmountRaw")
        if sell_amt is None:
            sell_amt = _hexint(q.get("sellAmount"), 0)
        sell_amt = int(sell_amt)
        if sell_amt <= 0:
            raise ValueError("send/v2: sellAmount must be > 0")

        # approval if ERC-20
        sell_tok = (q.get("sellToken") or "").lower()
        if sell_tok not in ("eth","native",NATIVE.lower(),"0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"):
            spender = q.get("allowanceTarget") or (q.get("transaction") or {}).get("to") or q.get("to")
            if spender:
                have = self.erc20_allowance(chain, sell_tok, self.acct.address, spender)
                if int(have) < int(sell_amt):
                    mode = os.getenv("APPROVE_MODE", "e").strip().lower()
                    if mode not in ("e","u"): mode = "e"
                    value = int(sell_amt) if mode == "e" else int((1<<256)-1)
                    print(f"[approve] spender={spender} need={sell_amt} have={have} mode={'exact' if mode=='e' else 'unlimited'}")
                    txh = self.approve_erc20(chain, sell_tok, spender, value)
                    try:
                        rc = w3.eth.wait_for_transaction_receipt(txh, timeout=180)
                        print(f"[approve] rc.status={rc.get('status')} gasUsed={rc.get('gasUsed')}")
                    except Exception:
                        pass

        # build tx
        to = Web3.to_checksum_address(q["to"])
        data = q["data"]
        if isinstance(data, str) and data.startswith("0x"):
            data_bytes = bytes.fromhex(data[2:])
        elif isinstance(data, (bytes, bytearray)):
            data_bytes = bytes(data)
        else:
            raise ValueError("bad 0x v2 quote: no tx data")
        value = _hexint(q.get("value"), 0)

        tx = {
            "from": self.acct.address,
            "to": to,
            "data": data_bytes,
            "value": int(value),
            "chainId": int(w3.eth.chain_id),
            "type": 2,
            "nonce": w3.eth.get_transaction_count(self.acct.address, "pending"),
        }

        # gas: estimate → hint → fallback
        try:
            g = int(w3.eth.estimate_gas({"from": tx["from"], "to": tx["to"], "data": tx["data"], "value": tx["value"]}))
        except Exception:
            g = _hexint(q.get("estimatedGas") or q.get("gas"), 250000)
        tx["gas"] = max(21000, g)

        # fees
        tx.update(self._suggest_fees(w3))

        # preflight
        try:
            w3.eth.call({"from": tx["from"], "to": tx["to"], "data": tx["data"], "value": tx["value"]}, "latest")
        except Exception as e:
            raise RuntimeError(f"preflight revert: {e}")

        # sign + send
        stx = w3.eth.account.sign_transaction(tx, private_key=self.pk)
        raw = getattr(stx, "rawTransaction", None) or getattr(stx, "raw_transaction", None)
        txh = w3.eth.send_raw_transaction(raw)
        return w3.to_hex(txh)

    # ------------------------ 1inch v6 ------------------------

    _ONEINCH_NATIVE = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
    _ONEINCH_CHAIN_IDS = {
        "ethereum": 1,
        "base": 8453,
        "arbitrum": 42161,
        "optimism": 10,
        "polygon": 137,
    }

    def get_1inch_swap_tx(self, chain: str, sell_token: str, buy_token: str, sell_amount_wei: int, slippage: float = 0.01) -> dict:
        base_cid = self._ONEINCH_CHAIN_IDS.get(chain.lower())
        if not base_cid:
            raise RuntimeError(f"1inch not supported for '{chain}'")
        if not ONEINCH_API_KEY:
            raise RuntimeError("ONEINCH_API_KEY is not set")

        s_id = self._ONEINCH_NATIVE if (sell_token.lower() in ("eth","native")) else Web3.to_checksum_address(sell_token)
        b_id = self._ONEINCH_NATIVE if (buy_token.lower()  in ("eth","native")) else Web3.to_checksum_address(buy_token)
        q = {
            "src": s_id,
            "dst": b_id,
            "amount": str(int(sell_amount_wei)),
            "from": self.acct.address,
            "slippage": f"{slippage*100:.4f}",
        }
        url = f"https://api.1inch.dev/swap/v6.0/{base_cid}/swap"
        r = requests.get(url, headers={"Authorization": f"Bearer {ONEINCH_API_KEY}", "Accept":"application/json"}, params=q, timeout=25)
        if r.status_code >= 400:
            raise RuntimeError(f"1inch {r.status_code}: {r.text}")
        j = r.json()
        tx = j.get("tx") or {}
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

    # ------------------------ OpenOcean v4 ------------------------

    _OO_NATIVE = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
    _OO_CHAINS = {
        "ethereum": "eth",
        "base": "base",
        "arbitrum": "arbitrum",
        "optimism": "optimism",
        "polygon": "polygon",
    }

    def get_openocean_swap_tx(self, chain: str, sell_token: str, buy_token: str, sell_amount_wei: int, slippage: float = 0.01) -> dict:
        oo_ch = self._OO_CHAINS.get(chain.lower())
        if not oo_ch:
            raise RuntimeError(f"OpenOcean not supported for '{chain}'")
        s_id = self._OO_NATIVE if (sell_token.lower() in ("eth","native")) else Web3.to_checksum_address(sell_token)
        b_id = self._OO_NATIVE if (buy_token.lower()  in ("eth","native")) else Web3.to_checksum_address(buy_token)
        slip_pct = f"{slippage*100:.4f}".rstrip('0').rstrip('.') if slippage else "1"
        q = {
            "inTokenAddress": s_id,
            "outTokenAddress": b_id,
            "amount": str(int(sell_amount_wei)),
            "slippage": slip_pct,
            "account": self.acct.address,
        }
        url = f"https://open-api.openocean.finance/v4/{oo_ch}/swap"
        r = requests.get(url, params=q, headers={"Accept":"application/json"}, timeout=25)
        if r.status_code >= 400:
            raise RuntimeError(f"OpenOcean {r.status_code}: {r.text}")
        j = r.json()
        tx = (j.get("data") or {})
        to_addr = tx.get("to") or j.get("to")
        data_hex = tx.get("data") or j.get("data") or "0x"
        value = int(tx.get("value") or j.get("value") or 0)
        est_gas = int(tx.get("gasLimit") or j.get("gasLimit") or j.get("estimatedGas") or 0)
        out_amt = int(j.get("outAmount") or j.get("toAmount") or 0)
        if not to_addr or not data_hex:
            raise RuntimeError(f"OpenOcean swap response missing tx fields: {j}")
        return {"to": to_addr, "data": data_hex, "value": value, "estimatedGas": est_gas, "buyAmount": out_amt}

    # ------------------------ Uniswap V3 (Router02) ------------------------

    # Canonical addresses (main chains; extend as needed)
    UNI_V3 = {
        "ethereum": {
            "SWAP_ROUTER": "0x68b3465833FB72A70ecDF485E0e4C7bD8665Fc45",  # SwapRouter02
            "QUOTER_V1":   "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
            "WETH":        "0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2",
        },
        "arbitrum": {
            "SWAP_ROUTER": "0x68b3465833FB72A70ecDF485E0e4C7bD8665Fc45",
            "QUOTER_V1":   "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",  # V2 quoter also OK; we use V1 signature
            "WETH":        "0x82aF49447D8a07e3bd95BDdB56f35241523fBab1",
        },
    }

    # QuoterV1: quoteExactInputSingle(address,address,uint24,uint256,uint160) -> uint256
    _ABI_QUOTER_V1 = [{
        "inputs": [
            {"internalType":"address","name":"tokenIn","type":"address"},
            {"internalType":"address","name":"tokenOut","type":"address"},
            {"internalType":"uint24","name":"fee","type":"uint24"},
            {"internalType":"uint256","name":"amountIn","type":"uint256"},
            {"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"}
        ],
        "name":"quoteExactInputSingle","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],
        "stateMutability":"view","type":"function"
    }]

    # Router02 exactInputSingle((address,address,uint24,address,uint256,uint256,uint160)) -> uint256
    _ABI_ROUTER02 = [{
        "inputs":[{"components":[
            {"internalType":"address","name":"tokenIn","type":"address"},
            {"internalType":"address","name":"tokenOut","type":"address"},
            {"internalType":"uint24","name":"fee","type":"uint24"},
            {"internalType":"address","name":"recipient","type":"address"},
            {"internalType":"uint256","name":"amountIn","type":"uint256"},
            {"internalType":"uint256","name":"amountOutMinimum","type":"uint256"},
            {"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"}
        ],"internalType":"struct ISwapRouter.ExactInputSingleParams","name":"params","type":"tuple"}],
        "name":"exactInputSingle",
        "outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],
        "stateMutability":"payable","type":"function"
    },
    # Router02 exactInput(bytes path,address recipient,uint256 amountIn,uint256 amountOutMinimum) -> uint256
    {
        "inputs":[
            {"internalType":"bytes","name":"path","type":"bytes"},
            {"internalType":"address","name":"recipient","type":"address"},
            {"internalType":"uint256","name":"amountIn","type":"uint256"},
            {"internalType":"uint256","name":"amountOutMinimum","type":"uint256"}
        ],
        "name":"exactInput",
        "outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],
        "stateMutability":"payable","type":"function"
    }]

    @staticmethod
    def _v3_encode_path(tokens: List[str], fees: List[int]) -> str:
        if len(tokens) != len(fees) + 1:
            raise ValueError("V3 path: tokens must be n+1 for n fees")
        b = b""
        for i, fee in enumerate(fees):
            b += bytes.fromhex(tokens[i][2:].zfill(40))
            b += int(fee).to_bytes(3, "big")
        b += bytes.fromhex(tokens[-1][2:].zfill(40))
        return "0x" + b.hex()

    def univ3_quote_and_build(self, chain: str, token_in: str, token_out: str, amount_in: int, *, slippage_bps: int = 100, recipient: Optional[str] = None) -> Dict[str, Any]:
        """
        Local UniswapV3 (Router02) builder. Tries direct fees [500,3000,10000], and
        if needed via WETH. Uses QuoterV1 (5 args) — correct per official ABI.
        Returns tx dict compatible with aggregator senders.
        """
        ch = chain.lower().strip()
        if ch not in self.UNI_V3:
            return {"__error__": f"UniswapV3 unsupported on {chain}"}

        w3 = self._w3(ch)
        conf = self.UNI_V3[ch]
        weth = Web3.to_checksum_address(conf["WETH"])
        router = w3.eth.contract(Web3.to_checksum_address(conf["SWAP_ROUTER"]), abi=self._ABI_ROUTER02)
        qv1 = w3.eth.contract(Web3.to_checksum_address(conf["QUOTER_V1"]), abi=self._ABI_QUOTER_V1)

        def _norm(x: str) -> str:
            s = (x or "").strip().lower()
            if s in ("eth", "native", NATIVE.lower(), "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"):
                return weth
            return Web3.to_checksum_address(x)

        t_in  = _norm(token_in)
        t_out = _norm(token_out)
        if int(amount_in) <= 0:
            return {"__error__":"UniswapV3: amount_in must be > 0"}
        if t_in == t_out:
            return {"__error__":"UniswapV3: token_in == token_out"}

        # 1) direct best fee
        best_out, best_fee = 0, 0
        for f in (500, 3000, 10000):
            try:
                out = qv1.functions.quoteExactInputSingle(t_in, t_out, int(f), int(amount_in), 0).call()
                if int(out) > best_out:
                    best_out, best_fee = int(out), int(f)
            except Exception:
                pass

        via = None
        if best_out <= 0 and (t_in != weth and t_out != weth):
            # 2) try via WETH (two-hop path quotes)
            for f1 in (500, 3000, 10000):
                for f2 in (500, 3000, 10000):
                    try:
                        path = self._v3_encode_path([t_in, weth, t_out], [f1, f2])
                        # We could use QuoterV2 quoteExactInput(path, amountIn) but V1 isn't available for path.
                        # If QUOTER_V2 address is handy, you can extend here. We'll rely on direct route if available.
                        pass
                    except Exception:
                        pass
            # If you want robust multi-hop quotes, wire in QuoterV2. For now, prefer direct routes.

        if best_out <= 0:
            return {"__error__":"UniswapV3: no viable pool (direct)"}

        out_min = max(1, best_out * (10_000 - int(slippage_bps)) // 10_000)
        recp = Web3.to_checksum_address(recipient) if recipient else self.acct.address

        # Build Router02 exactInputSingle (no deadline in Router02)
        params = (t_in, t_out, int(best_fee), recp, int(amount_in), int(out_min), 0)
        data = router.encode_abi("exactInputSingle", args=[params])

        to = Web3.to_checksum_address(conf["SWAP_ROUTER"])
        gas = None
        try:
            gas = int(w3.eth.estimate_gas({"from": recp, "to": to, "data": data, "value": 0}) * 1.2)
        except Exception:
            gas = None

        return {
            "aggregator": "UniswapV3",
            "allowanceTarget": to,
            "tx": {"to": to, "data": data, "value": 0, **({"gas": gas} if gas else {})},
            "buyAmount": str(best_out),
            "fee": best_fee,
        }

    # ------------------------ Camelot V2 (Arbitrum) ------------------------

    CAMELOT_V2_ROUTER = {"arbitrum": "0xC873FECbd354f5A56E00E710B90EF4201db2448d"}
    WETH_ADDRESS = {"arbitrum": "0x82aF49447D8a07e3bd95BDdB56f35241523fBab1"}

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

    def camelot_v2_quote(self, chain: str, token_in: str, token_out: str, amount_in: int, *, slippage_bps: int = 100) -> Dict[str, Any]:
        ch = chain.lower().strip()
        if ch not in self.CAMELOT_V2_ROUTER:
            return {"__error__": f"CamelotV2 unsupported on {chain}"}
        w3 = self._w3(chain)
        router = w3.eth.contract(w3.to_checksum_address(self.CAMELOT_V2_ROUTER[chain]), abi=self._ABI_V2_ROUTER)
        t_in  = w3.to_checksum_address(token_in)
        t_out = w3.to_checksum_address(token_out)
        weth  = w3.to_checksum_address(self.WETH_ADDRESS.get(chain, self.WETH_ADDRESS["arbitrum"]))

        path_try: List[List[str]] = []
        # direct
        try:
            amts = router.functions.getAmountsOut(int(amount_in), [t_in, t_out]).call()
            if len(amts) >= 2 and int(amts[-1]) > 0:
                path_try.append([t_in, t_out])
        except Exception:
            pass
        # via WETH
        try:
            amtsA = router.functions.getAmountsOut(int(amount_in), [t_in, weth]).call()
            amtsB = router.functions.getAmountsOut(int(amtsA[-1]), [weth, t_out]).call()
            if int(amtsA[-1]) > 0 and int(amtsB[-1]) > 0:
                path_try.append([t_in, weth, t_out])
        except Exception:
            pass
        if not path_try:
            return {"__error__": "CamelotV2: no path (direct or via WETH)"}

        best, best_out = None, -1
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

        out_min = int(best_out * (10_000 - int(slippage_bps)) / 10_000)
        deadline = int(w3.eth.get_block("latest")["timestamp"]) + 600
        fn = router.functions.swapExactTokensForTokensSupportingFeeOnTransferTokens(
            int(amount_in), out_min, best, self.acct.address,
            "0x0000000000000000000000000000000000000000", deadline
        )
        data = fn._encode_transaction_data()
        allowance_target = self.CAMELOT_V2_ROUTER[chain]

        gas_est = 0
        try:
            gas_est = int(w3.eth.estimate_gas({"from": self.acct.address, "to": allowance_target, "data": data, "value": 0}) * 1.2)
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


# =============================================================================
# Token annotation helpers (pure)
# =============================================================================

def detect_chain_for_token(addr: str, candidates: Iterable[str] = CHAINS.keys()) -> Optional[str]:
    addr = _normalize_addr(addr)
    if addr.lower() == NATIVE:
        return None
    for ch in candidates:
        try:
            # lightweight: try decimals() call
            w3 = UltraSwapBridge(None, private_key=os.getenv("PRIVATE_KEY") or "0x"+"0"*64)._w3(ch)  # dummy for RPC
            w3.eth.contract(address=Web3.to_checksum_address(addr), abi=ERC20_ABI).functions.decimals().call()
            return ch
        except Exception:
            continue
    return None

def annotate_tokens(tokens: Iterable[Any], default_chain: str = "ethereum") -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for it in tokens:
        ch, a = None, None
        if isinstance(it, str):
            s = it.strip()
            if ":" in s and s.split(":", 1)[0].lower() in CHAINS:
                ch, a = s.split(":", 1); ch = ch.lower(); a = a.strip()
            elif "@" in s:
                a, ch = s.split("@", 1); a = a.strip(); ch = ch.lower()
            else:
                a = s
        elif isinstance(it, (tuple, list)) and len(it) == 2:
            ch = str(it[0]).lower(); a = str(it[1])
        elif isinstance(it, dict):
            ch = str(it.get("chain", "") or "").lower() or None
            a = str(it.get("address", "") or "")
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
