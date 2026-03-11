#!/usr/bin/env python3
import os
import sys
import json
import time
import re
import random
import subprocess
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta, UTC
from threading import Thread, Semaphore
from concurrent.futures import ThreadPoolExecutor, as_completed
from web3 import Web3
import requests

# ---------- .env loader ----------
from dotenv_fallback import load_dotenv, find_dotenv, dotenv_values

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

def _configure_io_encoding() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if not stream:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
            continue
        except Exception:
            pass
        try:
            buffer = getattr(stream, "buffer", None)
            if buffer:
                import io
                wrapper = io.TextIOWrapper(buffer, encoding="utf-8", errors="replace", line_buffering=True)
                setattr(sys, stream_name, wrapper)
        except Exception:
            pass

_configure_io_encoding()

# --- CONFIGURATION (read from environment; fall back to previous defaults) ---
def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

ANKR_API_KEY         = os.getenv("ANKR_API_KEY", "").strip()
CHAIN_NAME_ENV       = os.getenv("CHAIN_NAME", "").strip().lower()
PAIR_ASSIGNMENT_FILE = os.getenv(
    "PAIR_ASSIGNMENT_FILE",
    "data/pair_provider_assignment.json",
)
OUTPUT_DIR_ENV       = os.getenv("OUTPUT_DIR", "").strip()
INTERMEDIATE_DIR_ENV = os.getenv("INTERMEDIATE_DIR", "").strip()
YEARS_BACK           = _int_env("YEARS_BACK", 3)
GRANULARITY_SECONDS  = _int_env("GRANULARITY_SECONDS", 60 * 5)   # 5 min
MAX_WORKERS          = _int_env("MAX_WORKERS", 30)

def _effective_workers() -> int:
    """Return dynamic worker count, capped by MAX_WORKERS and system availability."""
    try:
        from services.resource_governor import governor
        if not governor._started:
            governor.start()
        return governor.max_workers(base=MAX_WORKERS, floor=2)
    except Exception:
        return min(MAX_WORKERS, 8)

def _wait_if_pressured(label: str = "download") -> None:
    """Pause if system is under memory/CPU pressure to prevent OS lockup."""
    try:
        from services.resource_governor import governor
        if governor._started:
            governor.wait_if_pressured(label=label, max_wait=60.0)
    except Exception:
        pass
ANKR_RPS_LIMIT       = _int_env("ANKR_RPS_LIMIT", 30)
LOGS_PER_PARSE_BATCH = _int_env("LOGS_PER_PARSE_BATCH", 10)
RPC_CONNECT_TIMEOUT  = _int_env("RPC_CONNECT_TIMEOUT_SEC", 12)
RPC_CONNECT_RETRIES  = _int_env("RPC_CONNECT_RETRIES", 3)
RPC_MAX_RETRIES      = _int_env("RPC_MAX_RETRIES", 3)
PAIR_MAX_FAILURES    = _int_env("PAIR_MAX_FAILURES", 5)
PAIR_BACKOFF_BASE    = _int_env("PAIR_BACKOFF_BASE_SEC", 300)

# Optional full RPC URL override (e.g., self-hosted, Alchemy, Infura, Ankr w/ key)
RPC_URL_OVERRIDE = os.getenv("ANKR_RPC_URL", "").strip()

# Defer Web3 init until after we resolve the chain to avoid cross-chain mismatches
web3 = None  # type: ignore

PUBLIC_RPC_FALLBACKS = {
    "ethereum": "https://eth.llamarpc.com",
    "base": "https://mainnet.base.org",
    "arbitrum": "https://arb1.arbitrum.io/rpc",
    "optimism": "https://mainnet.optimism.io",
    "polygon": "https://polygon-rpc.com",
}

ANKR_CHAIN_SLUGS = {
    "ethereum": "eth",
    "base": "base",
    "arbitrum": "arbitrum",
    "optimism": "optimism",
    "polygon": "polygon",
}

ANKR_PUBLIC_ENDPOINTS = {
    chain: f"https://rpc.ankr.com/{slug}"
    for chain, slug in ANKR_CHAIN_SLUGS.items()
}

EXTRA_PUBLIC_RPC_FALLBACKS = {
    "base": [
        "https://base.llamarpc.com",
        "https://base-rpc.publicnode.com",
        "https://gateway.tenderly.co/public/base",
    ],
}

ACTIVE_CHAIN: Optional[str] = None
ACTIVE_RPC: Optional[str] = None
_rpc_candidates: List[str] = []
_rpc_cursor = 0

def _infer_chain_from_path(path: Path) -> Optional[str]:
    stem = path.stem.lower()
    for suffix in ("_pair_provider_assignment", "_pairs", "_assignment"):
        if stem.endswith(suffix):
            stem = stem.removesuffix(suffix)
    chunks = stem.split("_")
    if chunks:
        candidate = chunks[0].strip()
        if candidate in {"base", "ethereum", "arbitrum", "optimism", "polygon"}:
            return candidate
    return None

def _rpc_candidates_for_chain(chain: str) -> List[str]:
    chain = chain.lower().strip()
    prefer_free = os.getenv("PREFER_FREE_RPC", "1").strip().lower() not in {"0", "false", "no"}
    prefer_ankr = os.getenv("PREFER_ANKR_RPC", "1").strip().lower() not in {"0", "false", "no"}
    candidates: List[str] = []

    if RPC_URL_OVERRIDE:
        candidates.append(RPC_URL_OVERRIDE)

    if prefer_ankr:
        ankr_slug = ANKR_CHAIN_SLUGS.get(chain, chain)
        if ANKR_API_KEY:
            candidates.append(f"https://rpc.ankr.com/{ankr_slug}/{ANKR_API_KEY}".rstrip("/"))
        candidates.append(ANKR_PUBLIC_ENDPOINTS.get(chain, ""))

    alchemy_env_map = {
        "base": "ALCHEMY_BASE_URL",
        "ethereum": "ALCHEMY_ETH_URL",
        "arbitrum": "ALCHEMY_ARB_URL",
        "optimism": "ALCHEMY_OP_URL",
        "polygon": "ALCHEMY_POLY_URL",
    }
    slugs = {
        "base": "base-mainnet",
        "ethereum": "eth-mainnet",
        "arbitrum": "arb-mainnet",
        "optimism": "opt-mainnet",
        "polygon": "polygon-mainnet",
    }
    env_var = alchemy_env_map.get(chain, "")
    candidate = os.getenv(env_var, "").strip()
    if candidate:
        candidates.append(candidate)
    key = os.getenv("ALCHEMY_API_KEY", "").strip()
    if key and chain in slugs:
        candidates.append(f"https://{slugs[chain]}.g.alchemy.com/v2/{key}")

    if not prefer_ankr and ANKR_API_KEY:
        ankr_slug = ANKR_CHAIN_SLUGS.get(chain, chain)
        candidates.append(f"https://rpc.ankr.com/{ankr_slug}/{ANKR_API_KEY}".rstrip("/"))
        candidates.append(ANKR_PUBLIC_ENDPOINTS.get(chain, ""))

    if prefer_free:
        candidates.append(PUBLIC_RPC_FALLBACKS.get(chain, ""))
        candidates.extend(EXTRA_PUBLIC_RPC_FALLBACKS.get(chain, []))
    else:
        candidates.extend(EXTRA_PUBLIC_RPC_FALLBACKS.get(chain, []))
        candidates.append(PUBLIC_RPC_FALLBACKS.get(chain, ""))

    cleaned = []
    seen = set()
    for url in candidates:
        url = (url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        cleaned.append(url)
    if not cleaned:
        raise RuntimeError(
            f"No RPC configured for chain '{chain}'. "
            "Set ANKR_API_KEY or ANKR_RPC_URL, or a chain-specific URL (e.g., ALCHEMY_ETH_URL)."
        )
    return cleaned


def _probe_rpc(url: str) -> Optional[Web3]:
    try:
        w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": RPC_CONNECT_TIMEOUT}))
        if not w3.is_connected():
            return None
        _ = w3.eth.chain_id
        return w3
    except Exception:
        return None


def _connect_web3(chain: str) -> Web3:
    global web3, ACTIVE_RPC, _rpc_candidates, _rpc_cursor, ACTIVE_CHAIN
    ACTIVE_CHAIN = chain
    _rpc_candidates = _rpc_candidates_for_chain(chain)
    if not _rpc_candidates:
        raise RuntimeError(f"No RPC candidates for {chain}")
    start = _rpc_cursor
    for offset in range(len(_rpc_candidates)):
        idx = (start + offset) % len(_rpc_candidates)
        url = _rpc_candidates[idx]
        for _ in range(max(1, RPC_CONNECT_RETRIES)):
            w3 = _probe_rpc(url)
            if w3 is not None:
                web3 = w3
                ACTIVE_RPC = url
                _rpc_cursor = idx
                return w3
            time.sleep(0.2)
    raise RuntimeError(f"[ERROR] Could not connect to any RPC endpoint for {chain}")


def get_web3() -> Web3:
    global web3
    if web3 is None or not web3.is_connected():
        if not ACTIVE_CHAIN:
            raise RuntimeError("Chain not set for RPC initialization.")
        return _connect_web3(ACTIVE_CHAIN)
    return web3

# --- RATE LIMITER ---
bucket = Semaphore(0)
def _refill_bucket():
    while True:
        for _ in range(ANKR_RPS_LIMIT):
            bucket.release()
        time.sleep(1)
Thread(target=_refill_bucket, daemon=True).start()

def limited(fn, *args, **kwargs):
    last_exc = None
    for attempt in range(max(1, RPC_MAX_RETRIES)):
        bucket.acquire()
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < RPC_MAX_RETRIES:
                try:
                    if ACTIVE_CHAIN:
                        _connect_web3(ACTIVE_CHAIN)
                except Exception:
                    pass
                time.sleep(0.15)
            else:
                raise
    if last_exc:
        raise last_exc

# --- ABIs ---
PAIR_ABI = json.loads("""[
  {
    "anonymous": false,
    "inputs":[
      {"indexed":true,  "name":"sender",     "type":"address"},
      {"indexed":false, "name":"amount0In",  "type":"uint256"},
      {"indexed":false, "name":"amount1In",  "type":"uint256"},
      {"indexed":false, "name":"amount0Out", "type":"uint256"},
      {"indexed":false, "name":"amount1Out", "type":"uint256"},
      {"indexed":true,  "name":"to",         "type":"address"}
    ],
    "name":"Swap","type":"event"
  },
  {"inputs":[],"name":"token0","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},
  {"inputs":[],"name":"token1","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"}
]""")

V3_POOL_ABI = json.loads("""[
  {
    "anonymous": false,
    "inputs": [
      {"indexed": true, "name": "sender", "type": "address"},
      {"indexed": true, "name": "recipient", "type": "address"},
      {"indexed": false, "name": "amount0", "type": "int256"},
      {"indexed": false, "name": "amount1", "type": "int256"},
      {"indexed": false, "name": "sqrtPriceX96", "type": "uint160"},
      {"indexed": false, "name": "liquidity", "type": "uint128"},
      {"indexed": false, "name": "tick", "type": "int24"}
    ],
    "name": "Swap",
    "type": "event"
  },
  {"inputs":[],"name":"token0","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},
  {"inputs":[],"name":"token1","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},
  {"inputs":[],"name":"fee","outputs":[{"internalType":"uint24","name":"","type":"uint24"}],"stateMutability":"view","type":"function"}
]""")

# keep your original ABI; add tolerant variants for edge tokens
ERC20_ABI = json.loads("""[
  {"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"stateMutability":"view","type":"function"}
]""")
ERC20_ABI_U256 = json.loads("""[
  {"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]""")
ERC20_ABI_UPPER = json.loads("""[
  {"constant":true,"inputs":[],"name":"DECIMALS","outputs":[{"name":"","type":"uint8"}],"stateMutability":"view","type":"function"}
]""")

# --- HELPERS ---
def parse_delay(msg: str):
    m = re.search(r'retry in (\d+)([smh])', msg)
    if not m: return None
    val, unit = int(m.group(1)), m.group(2)
    return val * {'s':1,'m':60,'h':3600}[unit]

def get_block_by_timestamp(ts: int):
    latest = limited(lambda: get_web3().eth.block_number)
    latest_ts = limited(lambda: get_web3().eth.get_block(latest))["timestamp"]
    return max(1, latest - int((latest_ts - ts) / 13))

def fetch_chunk(ev, b, e_b, sym, idx, total):
    delays = [30, 30, 30, 60]
    for attempt, delay in enumerate(delays, start=1):
        try:
            logs = limited(lambda: ev.get_logs(from_block=b, to_block=e_b))
            print(f"    [{idx+1}/{total}] [DATA] {sym}: {len(logs)} logs [{b}->{e_b}]")
            return logs
        except Exception as exc:
            print(f"    [WARN] fetch error for {sym}[{b}->{e_b}] (attempt {attempt}/{len(delays)}): {exc}")
            if attempt < len(delays):
                sleep_sec = delay + random.uniform(0, 5)
                time.sleep(sleep_sec)
            else:
                print(f"    [ERROR] giving up on chunk [{b}->{e_b}] after {len(delays)} attempts, skipping")
                return []

def get_block_with_retry(block_number, retries=7):
    for attempt in range(retries):
        try:
            return limited(lambda: get_web3().eth.get_block(block_number))
        except Exception as e:
            msg = str(e)
            delay = parse_delay(msg) or (10 + random.uniform(0, 5))
            print(f"      Block fetch rate-limited or failed, retrying in {delay}s (attempt {attempt+1}/{retries})")
            time.sleep(delay)
    raise RuntimeError(f"Failed to fetch block {block_number} after {retries} retries")

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _has_code(addr: str) -> bool:
    try:
        code = limited(lambda: get_web3().eth.get_code(Web3.to_checksum_address(addr)))
    except Exception:
        return False
    try:
        return len(code) > 0
    except Exception:
        return bool(code)


def _should_rebuild_assignment(assignment: dict, chain: str, sample_size: int = 30, threshold: float = 0.75) -> bool:
    pairs = list(assignment.get("pairs", {}).keys())
    if not pairs:
        return False
    sample = random.sample(pairs, min(sample_size, len(pairs)))
    missing = sum(1 for addr in sample if not _has_code(addr))
    ratio = missing / max(1, len(sample))
    if ratio >= threshold:
        print(f"[WARN] {ratio:.0%} of sampled pairs have no code on {chain}. Rebuilding assignment.")
        return True
    return False


def _rebuild_assignment_from_index(chain: str, assignment_path: Path) -> dict:
    index_path = Path("data") / f"pair_index_{chain}.json"
    if not index_path.exists() or _pair_index_invalid(index_path, chain):
        _refresh_pair_index(chain, index_path)
    with index_path.open("r", encoding="utf-8") as fh:
        index = json.load(fh)
    pairs = {}
    for addr, meta in index.items():
        symbol = str(meta.get("symbol") or "").upper()
        if not symbol:
            continue
        pairs[addr] = {
            "symbol": symbol,
            "index": int(meta.get("index", len(pairs))),
            "completed": False,
        }
    assignment = {
        "granularity_seconds": GRANULARITY_SECONDS,
        "years_back": YEARS_BACK,
        "start_date": datetime.now(UTC).isoformat(),
        "chain": chain,
        "pairs": pairs,
    }
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    with assignment_path.open("w", encoding="utf-8") as fh:
        json.dump(assignment, fh, indent=2)
    print(f"[INFO] Rebuilt assignment with {len(pairs)} pairs from {index_path}")
    return assignment


def _pair_index_invalid(index_path: Path, chain: str, sample_size: int = 30, threshold: float = 0.75) -> bool:
    try:
        with index_path.open("r", encoding="utf-8") as fh:
            index = json.load(fh)
    except Exception:
        return True
    addrs = list(index.keys())
    if not addrs:
        return True
    sample = random.sample(addrs, min(sample_size, len(addrs)))
    missing = sum(1 for addr in sample if not _has_code(addr))
    ratio = missing / max(1, len(sample))
    if ratio >= threshold:
        print(f"[WARN] Pair index {index_path} looks wrong for {chain} ({ratio:.0%} empty code).")
        return True
    return False


def _refresh_pair_index(chain: str, index_path: Path) -> None:
    print(f"[INFO] Refreshing pair index for {chain}...")
    env = os.environ.copy()
    env["CHAIN_NAME"] = chain
    env.setdefault("PAIR_INDEX_OUTPUT_PATH", str(index_path))
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "make2000index.py")]
    try:
        subprocess.run(cmd, env=env, check=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to refresh pair index for {chain}: {exc}")

def safe_decimals(addr: str):
    """
    Robust decimals() reader:
      - verifies contract code exists
      - tries uint8, then uint256, then DECIMALS()
      - returns int on success, None on failure
    """
    try:
        a = Web3.to_checksum_address(addr)
    except Exception:
        print(f"   [ERROR] Invalid token address: {addr}")
        return None

    if not _has_code(a):
        print(f"   [ERROR] No contract code at {a} (not an ERC-20).")
        return None

    try:
        c = get_web3().eth.contract(address=a, abi=ERC20_ABI)
        return int(limited(lambda: c.functions.decimals().call()))
    except Exception:
        pass
    try:
        c = get_web3().eth.contract(address=a, abi=ERC20_ABI_U256)
        return int(limited(lambda: c.functions.decimals().call()))
    except Exception:
        pass
    try:
        c = get_web3().eth.contract(address=a, abi=ERC20_ABI_UPPER)
        return int(limited(lambda: c.functions.DECIMALS().call()))
    except Exception:
        pass

    print(f"   [ERROR] Could not read decimals() from {a}.")
    return None

def parse_logs_v2_no_ts(logs, dec0, dec1, granularity_seconds):
    # returns {bar_slot: [parsed records]}, {bar_slot: set(block_numbers)}
    records_by_bar = {}
    blocks_by_bar = {}
    for lg in logs:
        args = lg['args']
        blk_num = lg['blockNumber']
        a1out = int(args['amount1Out']) / 10**dec1
        a1in  = int(args['amount1In'])  / 10**dec1
        if a1out > 0:
            rec = {"block": blk_num, "price": int(args['amount0In'])/10**dec0/a1out,
                   "buy_volume": a1out, "sell_volume": 0.0}
        elif a1in > 0:
            rec = {"block": blk_num, "price": int(args['amount0Out'])/10**dec0/a1in,
                   "buy_volume": 0.0,    "sell_volume": a1in}
        else:
            continue
        bar_slot = blk_num // granularity_seconds
        records_by_bar.setdefault(bar_slot, []).append(rec)
        blocks_by_bar.setdefault(bar_slot, set()).add(blk_num)
    return records_by_bar, blocks_by_bar

def parse_logs_v3_no_ts(logs, dec0, dec1, granularity_seconds):
    records_by_bar = {}
    blocks_by_bar = {}
    for lg in logs:
        args = lg["args"]
        blk_num = lg["blockNumber"]
        amt0 = int(args["amount0"])
        amt1 = int(args["amount1"])
        if amt0 == 0 or amt1 == 0:
            continue
        a0 = abs(amt0) / 10**dec0
        a1 = abs(amt1) / 10**dec1
        if a1 == 0:
            continue
        price = a0 / a1
        if amt1 < 0:
            rec = {"block": blk_num, "price": price, "buy_volume": a1, "sell_volume": 0.0}
        else:
            rec = {"block": blk_num, "price": price, "buy_volume": 0.0, "sell_volume": a1}
        bar_slot = blk_num // granularity_seconds
        records_by_bar.setdefault(bar_slot, []).append(rec)
        blocks_by_bar.setdefault(bar_slot, set()).add(blk_num)
    return records_by_bar, blocks_by_bar

def _is_revert_error(exc: Exception) -> bool:
    """Return True if the exception indicates a permanent contract-level revert
    (method doesn't exist, wrong ABI, etc.) rather than a transient RPC/network error."""
    msg = str(exc).lower()
    revert_indicators = (
        "execution reverted", "revert", "invalid opcode",
        "out of gas", "bad instruction", "invalid jump",
        "could not decode", "missing revert data",
        "returned an empty", "0x", "abi", "decode",
    )
    transient_indicators = (
        "timeout", "timed out", "connection", "connect",
        "429", "rate limit", "too many requests",
        "502", "503", "504", "server error",
        "eof", "broken pipe", "reset by peer",
    )
    if any(t in msg for t in transient_indicators):
        return False
    return any(r in msg for r in revert_indicators)


def resolve_pool_kind(pair_addr):
    """Attempt to read token0/token1 from a pool contract.
    Returns:
      dict   — success: {kind, t0_raw, t1_raw, fee}
      None   — transient error (worth retrying later)
      False  — permanent incompatibility (never retry)
    """
    # Pre-check: does the address even have deployed code?
    if not _has_code(pair_addr):
        return False  # no contract at this address — permanent

    # Try V2 interface
    try:
        pair_v2 = get_web3().eth.contract(address=pair_addr, abi=PAIR_ABI)
        t0_raw = limited(lambda: pair_v2.functions.token0().call())
        t1_raw = limited(lambda: pair_v2.functions.token1().call())
    except Exception as exc_v2:
        v2_permanent = _is_revert_error(exc_v2)
        # Try V3 interface
        try:
            pair_v3 = get_web3().eth.contract(address=pair_addr, abi=V3_POOL_ABI)
            t0_raw = limited(lambda: pair_v3.functions.token0().call())
            t1_raw = limited(lambda: pair_v3.functions.token1().call())
        except Exception as exc_v3:
            v3_permanent = _is_revert_error(exc_v3)
            if v2_permanent and v3_permanent:
                return False  # both interfaces permanently rejected
            return None  # at least one was transient — worth retrying

    # Determine if it's V3 (has fee()) or V2
    pool_kind = "v2"
    fee = None
    try:
        pair_v3 = get_web3().eth.contract(address=pair_addr, abi=V3_POOL_ABI)
        fee = int(limited(lambda: pair_v3.functions.fee().call()))
        pool_kind = "v3"
    except Exception:
        pass
    return {"kind": pool_kind, "t0_raw": t0_raw, "t1_raw": t1_raw, "fee": fee}

def aggregate_ohlcv_by_bar(records, bar_ts):
    if not records:
        return {}
    open_p = records[0]['price']
    close_p = records[-1]['price']
    high_p = max(r['price'] for r in records)
    low_p  = min(r['price'] for r in records)
    buy_vol = sum(r['buy_volume'] for r in records)
    sell_vol = sum(r['sell_volume'] for r in records)
    vol = buy_vol + sell_vol
    vwap = sum(r['price'] * (r['buy_volume'] + r['sell_volume']) for r in records) / vol if vol > 0 else close_p
    return {
        "timestamp": bar_ts,
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close_p,
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "net_volume": buy_vol - sell_vol,
        "vwap": vwap
    }

def load_intermediate_chunks(pair_dir):
    # Returns {bar_slot: [records]}, {bar_slot: set(block_numbers)}
    records_by_bar = {}
    blocks_by_bar = {}
    if not os.path.exists(pair_dir): return records_by_bar, blocks_by_bar
    for fname in os.listdir(pair_dir):
        if not fname.endswith(".json"): continue
        with open(os.path.join(pair_dir, fname), "r") as f:
            chunk = json.load(f)
            for bar_slot, records in chunk.get("records_by_bar", {}).items():
                records_by_bar.setdefault(bar_slot, []).extend(records)
            for bar_slot, blocks in chunk.get("blocks_by_bar", {}).items():
                blocks_by_bar.setdefault(bar_slot, set()).update(blocks)
    blocks_by_bar = {k: set(v) if not isinstance(v, set) else v for k, v in blocks_by_bar.items()}
    return records_by_bar, blocks_by_bar

def save_intermediate_chunk(pair_dir, chunk_idx, records_by_bar, blocks_by_bar):
    os.makedirs(pair_dir, exist_ok=True)
    fname = os.path.join(pair_dir, f"chunk_{chunk_idx:06d}.json")
    chunk = {
        "records_by_bar": {str(k): v for k, v in records_by_bar.items()},
        "blocks_by_bar": {str(k): list(v) for k, v in blocks_by_bar.items()}
    }
    with open(fname, "w") as f:
        json.dump(chunk, f, indent=2)

def save_bars(filename, bars):
    tmp = filename + ".tmp"
    with open(tmp, "w") as f:
        json.dump(bars, f, indent=2)
    os.replace(tmp, filename)

# DEXes that use the standard Uniswap token0()/token1() pool interface.
_UNISWAP_COMPATIBLE_DEXIDS = frozenset({
    "uniswap", "sushiswap", "pancakeswap", "quickswap", "camelot",
    "trader-joe", "spookyswap", "spiritswap", "velodrome", "aerodrome",
    "baseswap", "swapbased", "alienbase", "maverick", "thena",
    "ramses", "chronos", "zyberswap", "arbidex", "woofi",
    "solidly", "equalizer", "retro", "synthswap", "dackie",
})


def _find_alternative_pool(sym: str, chain: str) -> Optional[str]:
    """
    Search DexScreener for an alternative Uniswap-compatible pool for a token pair.
    Returns the new pool address if found, or None.
    """
    parts = sym.split("-")
    if len(parts) < 2:
        return None
    # Search for both tokens to find matching pools
    query = parts[0] if parts[0] != "WETH" else parts[1]
    try:
        resp = requests.get(
            "https://api.dexscreener.com/latest/dex/search",
            params={"q": query},
            timeout=12,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json() or {}
    except Exception:
        return None

    candidates = []
    for entry in payload.get("pairs") or []:
        try:
            chain_id = str(entry.get("chainId") or entry.get("chain") or "").lower()
            if chain_id != chain:
                continue
            dex_id = str(entry.get("dexId") or "").lower().strip()
            if dex_id not in _UNISWAP_COMPATIBLE_DEXIDS:
                continue
            base = entry.get("baseToken") or {}
            quote = entry.get("quoteToken") or {}
            base_sym = (base.get("symbol") or "").upper()
            quote_sym = (quote.get("symbol") or "").upper()
            entry_sym = f"{base_sym}-{quote_sym}"
            # Match the token pair (in either order)
            pair_set = {p.upper() for p in parts}
            entry_set = {base_sym, quote_sym}
            if pair_set != entry_set:
                continue
            pair_addr = str(entry.get("pairAddress") or "").strip()
            if not pair_addr:
                continue
            volume = float(entry.get("volume", {}).get("h24", 0) or 0)
            liquidity = float((entry.get("liquidity") or {}).get("usd", 0) or 0)
            score = volume + liquidity * 0.5
            candidates.append((pair_addr, entry_sym, dex_id, score))
        except Exception:
            continue

    if not candidates:
        return None

    # Pick the best by score
    candidates.sort(key=lambda c: c[3], reverse=True)
    best_addr, best_sym, best_dex, best_score = candidates[0]
    print(f"   [REPLACE] Found alternative pool: {best_addr} ({best_dex}, score={best_score:.0f})")
    return best_addr


def update_assignment(assignment, addr, **fields):
    meta = assignment["pairs"][addr]
    for k, v in fields.items():
        if v is None:
            meta.pop(k, None)
        else:
            meta[k] = v
    tmp = PAIR_ASSIGNMENT_FILE + ".tmp"
    for attempt in range(3):
        try:
            with open(tmp, "w") as wf:
                json.dump(assignment, wf, indent=2)
            os.replace(tmp, PAIR_ASSIGNMENT_FILE)
            return
        except OSError as exc:
            if attempt < 2:
                time.sleep(0.3 * (attempt + 1))
            else:
                raise


def _is_deferred(meta: dict) -> bool:
    until = meta.get("deferred_until")
    if not until:
        return False
    try:
        if isinstance(until, (int, float)):
            ts = float(until)
        else:
            ts = datetime.fromisoformat(str(until).replace("Z", "+00:00")).timestamp()
        return time.time() < ts
    except Exception:
        return False


def defer_pair(assignment: dict, addr: str, error: str) -> None:
    meta = assignment["pairs"].get(addr, {})
    failures = int(meta.get("failures", 0)) + 1
    meta["failures"] = failures
    meta["last_failure"] = datetime.now(UTC).isoformat()
    meta["error"] = error
    if failures >= PAIR_MAX_FAILURES:
        meta["completed"] = True
        meta["skipped"] = True
        meta.pop("deferred_until", None)
    else:
        delay = min(6 * 3600, PAIR_BACKOFF_BASE * (2 ** max(0, failures - 1)))
        meta["deferred_until"] = (datetime.now(UTC) + timedelta(seconds=delay)).isoformat()
    update_assignment(assignment, addr, **meta)

def main():
    assignment_path = Path(PAIR_ASSIGNMENT_FILE)
    if not assignment_path.exists():
        raise RuntimeError(
            f"Assignment file {assignment_path} not found. "
            "Set PAIR_ASSIGNMENT_FILE or place the file in the data/ directory."
        )
    with assignment_path.open() as f:
        assignment = json.load(f)
    skip_counts = {
        "invalid_pair_addr": 0,
        "non_uniswap_pool": 0,
        "invalid_token_address": 0,
        "token_metadata_unreadable": 0,
    }

    inferred_chain = assignment.get("chain") or _infer_chain_from_path(assignment_path)
    explicit_chain = CHAIN_NAME_ENV or None
    chain = (explicit_chain or inferred_chain)
    if not chain:
        raise RuntimeError(
            "Chain is not specified. Set CHAIN_NAME in the environment or add a "
            "'chain' key to the assignment file."
        )
    chain = chain.lower().strip()
    if assignment.get("chain") and assignment.get("chain").lower() != chain:
        raise RuntimeError(
            f"Chain mismatch: assignment file tagged '{assignment.get('chain')}', environment wants '{chain}'. "
            "Align CHAIN_NAME or fix the assignment file."
        )
    assignment.setdefault("chain", chain)

    _connect_web3(chain)
    print(f"[INFO] Chain: {chain} | RPC: {ACTIVE_RPC}")
    if _should_rebuild_assignment(assignment, chain):
        assignment = _rebuild_assignment_from_index(chain, assignment_path)

    output_dir = Path(OUTPUT_DIR_ENV) if OUTPUT_DIR_ENV else Path("data") / "historical_ohlcv" / chain
    intermediate_dir = Path(INTERMEDIATE_DIR_ENV) if INTERMEDIATE_DIR_ENV else Path("data") / "intermediate" / chain
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)

    start_ts = int((datetime.now(UTC) - timedelta(days=YEARS_BACK*365)).timestamp())
    start_blk = get_block_by_timestamp(start_ts)
    end_blk   = limited(lambda: get_web3().eth.block_number)
    print(f"[SCAN] Scanning blocks {start_blk} -> {end_blk}")

    chunks_count = YEARS_BACK*365*24
    total_blocks = end_blk - start_blk
    CHUNK_SIZE_BLOCKS = max(1, total_blocks // chunks_count)
    print(f"[INFO] Dynamic CHUNK_SIZE set to {CHUNK_SIZE_BLOCKS} blocks")

    # --- GAP BACKFILL: retry previously failed block ranges ---
    gap_pairs = [
        (addr, m) for addr, m in assignment["pairs"].items()
        if m.get("gap_ranges") and not m.get("skipped") and not _is_deferred(m)
    ]
    if gap_pairs:
        print(f"[BACKFILL] {len(gap_pairs)} pairs have gaps to retry...")
        for addr, gm in gap_pairs:
            sym = gm["symbol"]
            pair_dir = os.path.join(intermediate_dir, sym)
            try:
                pair_addr = Web3.to_checksum_address(addr)
            except Exception:
                continue
            resolved = resolve_pool_kind(pair_addr)
            if not resolved or resolved is False:
                continue
            pool_kind = resolved["kind"]
            try:
                t0 = Web3.to_checksum_address(resolved["t0_raw"])
                t1 = Web3.to_checksum_address(resolved["t1_raw"])
            except Exception:
                continue
            dec0 = safe_decimals(t0)
            dec1 = safe_decimals(t1)
            if dec0 is None or dec1 is None:
                continue
            if pool_kind == "v3":
                pair = get_web3().eth.contract(address=pair_addr, abi=V3_POOL_ABI)
            else:
                pair = get_web3().eth.contract(address=pair_addr, abi=PAIR_ABI)
            ev = pair.events.Swap()
            remaining_gaps = []
            existing_chunks = sorted(Path(pair_dir).glob("chunk_*.json")) if os.path.exists(pair_dir) else []
            chunk_idx = len(existing_chunks)
            for gap_b, gap_e in gm["gap_ranges"]:
                logs = fetch_chunk(ev, gap_b, gap_e, sym, chunk_idx, 0)
                if logs:
                    log_batches = list(chunked(logs, LOGS_PER_PARSE_BATCH))
                    chunk_records_by_bar = {}
                    chunk_blocks_by_bar = {}
                    with ThreadPoolExecutor(max_workers=_effective_workers()) as ex:
                        if pool_kind == "v3":
                            futs = [ex.submit(parse_logs_v3_no_ts, batch, dec0, dec1, GRANULARITY_SECONDS) for batch in log_batches]
                        else:
                            futs = [ex.submit(parse_logs_v2_no_ts, batch, dec0, dec1, GRANULARITY_SECONDS) for batch in log_batches]
                        for pf in as_completed(futs):
                            try:
                                recs, blks = pf.result()
                                for bs, r in recs.items():
                                    chunk_records_by_bar.setdefault(bs, []).extend(r)
                                for bs, bl in blks.items():
                                    chunk_blocks_by_bar.setdefault(bs, set()).update(bl)
                            except Exception:
                                pass
                    save_intermediate_chunk(pair_dir, chunk_idx, chunk_records_by_bar, chunk_blocks_by_bar)
                    chunk_idx += 1
                    print(f"    [BACKFILL] {sym}: filled gap [{gap_b}->{gap_e}] ({len(logs)} logs)")
                else:
                    remaining_gaps.append([gap_b, gap_e])
            if remaining_gaps:
                update_assignment(assignment, addr, gap_ranges=remaining_gaps)
            else:
                # All gaps filled — clear gap_ranges
                gm.pop("gap_ranges", None)
                update_assignment(assignment, addr, gap_ranges=None)
                print(f"    [BACKFILL] {sym}: all gaps filled!")

    pairs = [
        (addr, meta)
        for addr, meta in assignment["pairs"].items()
        if not meta.get("completed", False) and not _is_deferred(meta)
    ]
    print(f"[START] Will process {len(pairs)} pairs...")

    for addr, meta in pairs:
        _wait_if_pressured("download_pair")
        sym = meta["symbol"]
        idx = meta["index"]
        out = os.path.join(output_dir, f"{idx:04d}_{sym}.json")
        pair_dir = os.path.join(intermediate_dir, sym)
        print(f"\n=== Processing {sym} ===")

        pair_start_blk = meta.get("next_block", None)
        if pair_start_blk is None:
            pair_start_blk = start_blk
        else:
            pair_start_blk = int(pair_start_blk)

        try:
            pair_addr = Web3.to_checksum_address(addr)
        except Exception as e:
            print(f"   [ERROR] Invalid pair address for {sym}: {addr} ({e}). Skipping.")
            defer_pair(assignment, addr, "invalid_pair_address")
            skip_counts["invalid_pair_addr"] += 1
            continue

        resolved = resolve_pool_kind(pair_addr)
        if resolved is False:
            # Permanent: no contract code, or both V2/V3 interfaces reverted.
            # Try to find an alternative Uniswap-compatible pool for the same pair.
            alt_addr = _find_alternative_pool(sym, chain)
            if alt_addr:
                try:
                    alt_cs = Web3.to_checksum_address(alt_addr)
                    alt_resolved = resolve_pool_kind(alt_cs)
                    if alt_resolved and alt_resolved is not False:
                        print(f"   [SWAP] {sym}: replacing {pair_addr} -> {alt_cs}")
                        # Remove old entry, add new one with same metadata
                        old_meta = dict(meta)
                        old_meta.pop("completed", None)
                        old_meta.pop("skipped", None)
                        old_meta.pop("error", None)
                        old_meta["next_block"] = None  # restart from beginning for new pool
                        old_meta["replaced_from"] = addr
                        assignment["pairs"].pop(addr, None)
                        assignment["pairs"][alt_addr] = old_meta
                        update_assignment(assignment, alt_addr)
                        # Re-set loop variables for this iteration
                        addr = alt_addr
                        pair_addr = alt_cs
                        resolved = alt_resolved
                    else:
                        print(f"   [SKIP] {sym}: alternative {alt_cs} also incompatible.")
                        resolved = False
                except Exception as exc:
                    print(f"   [SKIP] {sym}: alternative pool check failed: {exc}")
                    resolved = False
            if resolved is False:
                print(f"   [SKIP] {sym}: {pair_addr} is permanently incompatible (not a Uniswap-style pool). No alternative found.")
                meta_update = {"completed": True, "skipped": True, "error": "permanently_incompatible"}
                update_assignment(assignment, addr, **meta_update)
                skip_counts["non_uniswap_pool"] += 1
                continue
        if resolved is None:
            # Transient: RPC timeout/rate-limit — worth retrying later
            print(f"   [WARN] {sym}: {pair_addr} token0/token1 call failed (transient). Deferring.")
            defer_pair(assignment, addr, "transient_rpc_error")
            skip_counts["non_uniswap_pool"] += 1
            continue
        pool_kind = resolved["kind"]
        t0_raw = resolved["t0_raw"]
        t1_raw = resolved["t1_raw"]
        v3_fee = resolved["fee"]
        if pool_kind == "v3":
            pair = get_web3().eth.contract(address=pair_addr, abi=V3_POOL_ABI)
        else:
            pair = get_web3().eth.contract(address=pair_addr, abi=PAIR_ABI)
        ev = pair.events.Swap()

        try:
            t0 = Web3.to_checksum_address(t0_raw)
            t1 = Web3.to_checksum_address(t1_raw)
        except Exception as e:
            print(f"   [ERROR] {sym}: invalid token addresses {t0_raw}/{t1_raw}: {e}. Deferring.")
            defer_pair(assignment, addr, "invalid_token_address")
            skip_counts["invalid_token_address"] += 1
            continue

        dec0 = safe_decimals(t0)
        dec1 = safe_decimals(t1)
        if dec0 is None or dec1 is None:
            print(f"   [WARN] Deferring {sym}: unreadable token metadata (t0={t0}, t1={t1}).")
            defer_pair(assignment, addr, "token_metadata_unreadable")
            skip_counts["token_metadata_unreadable"] += 1
            continue

        if pool_kind == "v3":
            print(f"   pool: v3 fee={v3_fee} | tokens: {t0}(dec{dec0}) / {t1}(dec{dec1})")
        else:
            print(f"   pool: v2 | tokens: {t0}(dec{dec0}) / {t1}(dec{dec1})")
        update_assignment(assignment, addr, pool_type=pool_kind, fee=v3_fee)

        b = pair_start_blk
        # Resume chunk index from existing intermediate files so we don't
        # misalign block position with chunk numbering after a crash.
        existing_chunks = sorted(Path(pair_dir).glob("chunk_*.json")) if os.path.exists(pair_dir) else []
        chunk_idx = len(existing_chunks)
        total_chunks = ((end_blk - b) + CHUNK_SIZE_BLOCKS - 1) // CHUNK_SIZE_BLOCKS + chunk_idx
        skipped_ranges: list = []
        consecutive_zero_logs = int(meta.get("consecutive_zero_logs", 0))
        max_zero_chunks = _int_env("MAX_ZERO_LOG_CHUNKS", 5)
        total_logs_seen = 0
        dead_pair = False
        while b < end_blk:
            e_b = min(b + CHUNK_SIZE_BLOCKS, end_blk)

            print(f"    [SCAN] Fetching chunk {chunk_idx+1}/{total_chunks} [{b}->{e_b}] for {sym}")
            logs = fetch_chunk(ev, b, e_b, sym, chunk_idx, total_chunks)
            if logs:
                consecutive_zero_logs = 0
                total_logs_seen += len(logs)
                log_batches = list(chunked(logs, LOGS_PER_PARSE_BATCH))
                chunk_records_by_bar = {}
                chunk_blocks_by_bar = {}
                with ThreadPoolExecutor(max_workers=_effective_workers()) as ex:
                    if pool_kind == "v3":
                        futs = [ex.submit(parse_logs_v3_no_ts, batch, dec0, dec1, GRANULARITY_SECONDS) for batch in log_batches]
                    else:
                        futs = [ex.submit(parse_logs_v2_no_ts, batch, dec0, dec1, GRANULARITY_SECONDS) for batch in log_batches]
                    for pf in as_completed(futs):
                        try:
                            recs_by_bar, blocks_by_bar = pf.result()
                            for bar_slot, recs in recs_by_bar.items():
                                chunk_records_by_bar.setdefault(bar_slot, []).extend(recs)
                            for bar_slot, blocks in blocks_by_bar.items():
                                chunk_blocks_by_bar.setdefault(bar_slot, set()).update(blocks)
                        except Exception as e:
                            print(f"      Parse error: {e}. Skipping one batch.")
                save_intermediate_chunk(pair_dir, chunk_idx, chunk_records_by_bar, chunk_blocks_by_bar)
                update_assignment(assignment, addr, next_block=e_b)
            elif logs is not None:
                # fetch_chunk returned [] — zero logs in this block range
                consecutive_zero_logs += 1
                if consecutive_zero_logs >= max_zero_chunks and total_logs_seen == 0:
                    print(f"   [DEAD] {sym}: {consecutive_zero_logs} consecutive chunks with 0 logs. Dead pair — removing.")
                    dead_pair = True
                    break
                skipped_ranges.append([b, e_b])
                update_assignment(assignment, addr, next_block=e_b, consecutive_zero_logs=consecutive_zero_logs)
            else:
                update_assignment(assignment, addr, next_block=e_b)
            b = e_b
            chunk_idx += 1

        if dead_pair:
            # Try to find a replacement pool
            alt_addr = _find_alternative_pool(sym, chain)
            if alt_addr:
                try:
                    alt_cs = Web3.to_checksum_address(alt_addr)
                    alt_resolved = resolve_pool_kind(alt_cs)
                    if alt_resolved and alt_resolved is not False:
                        print(f"   [SWAP] {sym}: replacing dead pool {pair_addr} -> {alt_cs}")
                        old_meta = dict(meta)
                        old_meta.pop("completed", None)
                        old_meta.pop("skipped", None)
                        old_meta.pop("error", None)
                        old_meta.pop("consecutive_zero_logs", None)
                        old_meta["next_block"] = None
                        old_meta["replaced_from"] = addr
                        assignment["pairs"].pop(addr, None)
                        assignment["pairs"][alt_addr] = old_meta
                        update_assignment(assignment, alt_addr)
                        continue  # skip to next pair
                except Exception as exc:
                    print(f"   [WARN] {sym}: alternative pool check failed: {exc}")
            # No alternative found — mark completed/skipped
            update_assignment(assignment, addr, completed=True, skipped=True, error="persistent_zero_logs")
            skip_counts.setdefault("zero_logs", 0)
            skip_counts["zero_logs"] += 1
            continue

        print("    [SCAN] Merging intermediate data and fetching timestamps...")
        all_records_by_bar, min_block_per_bar = load_intermediate_chunks(pair_dir)
        min_block_per_bar = {bar_slot: min(blocks) for bar_slot, blocks in min_block_per_bar.items() if blocks}
        bar_slot_to_ts = {}
        with ThreadPoolExecutor(max_workers=_effective_workers()) as ex:
            futs = {ex.submit(get_block_with_retry, block): bar_slot for bar_slot, block in min_block_per_bar.items()}
            for pf in as_completed(futs):
                bar_slot = futs[pf]
                try:
                    blk = pf.result()
                    bar_slot_to_ts[bar_slot] = blk["timestamp"]
                except Exception as e:
                    print(f"      Failed to fetch timestamp for bar {bar_slot}: {e}")

        print("    [AGG] Aggregating OHLCV bars...")
        bars = []
        for bar_slot, records in all_records_by_bar.items():
            ts = bar_slot_to_ts.get(bar_slot)
            if ts is not None:
                bars.append(aggregate_ohlcv_by_bar(records, ts))
        bars.sort(key=lambda x: x["timestamp"])
        save_bars(out, bars)
        # Persist any gap ranges so they can be backfilled on a future run
        if skipped_ranges:
            prev_gaps = meta.get("gap_ranges") or []
            all_gaps = prev_gaps + skipped_ranges
            print(f"    [GAP] {sym}: {len(skipped_ranges)} block ranges could not be fetched (total gaps: {len(all_gaps)})")
            update_assignment(assignment, addr, completed=True, gap_ranges=all_gaps)
        else:
            update_assignment(assignment, addr, completed=True)
        print(f"    [DONE] Pair {sym} complete and saved.")

    # --- UPDATE-TO-PRESENT pass: extend previously completed pairs to current block ---
    completed_pairs = [
        (addr, m) for addr, m in assignment["pairs"].items()
        if m.get("completed") and not m.get("skipped")
        and m.get("next_block") and int(m.get("next_block", 0)) < end_blk
    ]
    if completed_pairs:
        print(f"\n[UPDATE] {len(completed_pairs)} completed pairs have new blocks to catch up on...")
        for addr, meta in completed_pairs:
            if _is_deferred(meta):
                continue
            sym = meta["symbol"]
            idx_num = meta["index"]
            catch_up_start = int(meta["next_block"])
            blocks_behind = end_blk - catch_up_start
            if blocks_behind < CHUNK_SIZE_BLOCKS:
                continue  # not enough new blocks to bother
            print(f"\n=== Updating {sym} ({blocks_behind} blocks behind) ===")
            out = os.path.join(output_dir, f"{idx_num:04d}_{sym}.json")
            pair_dir = os.path.join(intermediate_dir, sym)
            try:
                pair_addr = Web3.to_checksum_address(addr)
            except Exception:
                continue
            resolved = resolve_pool_kind(pair_addr)
            if not resolved or resolved is False:
                continue
            pool_kind = resolved["kind"]
            t0_raw, t1_raw, v3_fee = resolved["t0_raw"], resolved["t1_raw"], resolved["fee"]
            try:
                t0 = Web3.to_checksum_address(t0_raw)
                t1 = Web3.to_checksum_address(t1_raw)
            except Exception:
                continue
            dec0 = safe_decimals(t0)
            dec1 = safe_decimals(t1)
            if dec0 is None or dec1 is None:
                continue
            if pool_kind == "v3":
                pair = get_web3().eth.contract(address=pair_addr, abi=V3_POOL_ABI)
            else:
                pair = get_web3().eth.contract(address=pair_addr, abi=PAIR_ABI)
            ev = pair.events.Swap()
            b = catch_up_start
            existing_chunks = sorted(Path(pair_dir).glob("chunk_*.json")) if os.path.exists(pair_dir) else []
            chunk_idx = len(existing_chunks)
            while b < end_blk:
                e_b = min(b + CHUNK_SIZE_BLOCKS, end_blk)
                logs = fetch_chunk(ev, b, e_b, sym, chunk_idx, 0)
                if logs:
                    log_batches = list(chunked(logs, LOGS_PER_PARSE_BATCH))
                    chunk_records_by_bar = {}
                    chunk_blocks_by_bar = {}
                    with ThreadPoolExecutor(max_workers=_effective_workers()) as ex:
                        if pool_kind == "v3":
                            futs = [ex.submit(parse_logs_v3_no_ts, batch, dec0, dec1, GRANULARITY_SECONDS) for batch in log_batches]
                        else:
                            futs = [ex.submit(parse_logs_v2_no_ts, batch, dec0, dec1, GRANULARITY_SECONDS) for batch in log_batches]
                        for pf in as_completed(futs):
                            try:
                                recs_by_bar, blocks_by_bar = pf.result()
                                for bar_slot, recs in recs_by_bar.items():
                                    chunk_records_by_bar.setdefault(bar_slot, []).extend(recs)
                                for bar_slot, blocks in blocks_by_bar.items():
                                    chunk_blocks_by_bar.setdefault(bar_slot, set()).update(blocks)
                            except Exception:
                                pass
                    save_intermediate_chunk(pair_dir, chunk_idx, chunk_records_by_bar, chunk_blocks_by_bar)
                update_assignment(assignment, addr, next_block=e_b)
                b = e_b
                chunk_idx += 1
            # Re-merge all intermediate data including new chunks
            all_records_by_bar, min_block_per_bar = load_intermediate_chunks(pair_dir)
            min_block_per_bar = {bs: min(bl) for bs, bl in min_block_per_bar.items() if bl}
            bar_slot_to_ts = {}
            with ThreadPoolExecutor(max_workers=_effective_workers()) as ex:
                futs = {ex.submit(get_block_with_retry, block): bs for bs, block in min_block_per_bar.items()}
                for pf in as_completed(futs):
                    bs = futs[pf]
                    try:
                        blk = pf.result()
                        bar_slot_to_ts[bs] = blk["timestamp"]
                    except Exception:
                        pass
            bars = []
            for bs, records in all_records_by_bar.items():
                ts = bar_slot_to_ts.get(bs)
                if ts is not None:
                    bars.append(aggregate_ohlcv_by_bar(records, ts))
            bars.sort(key=lambda x: x["timestamp"])
            save_bars(out, bars)
            update_assignment(assignment, addr, completed=True, next_block=end_blk)
            print(f"    [DONE] {sym} updated to block {end_blk}")

    print("\n[DONE] All pairs complete.")
    print(
        "Deferred/Skipped summary: "
        f"invalid_pair_addr={skip_counts['invalid_pair_addr']}, "
        f"non_uniswap_pool={skip_counts['non_uniswap_pool']}, "
        f"invalid_token_address={skip_counts['invalid_token_address']}, "
        f"token_metadata_unreadable={skip_counts['token_metadata_unreadable']}"
    )

if __name__=="__main__":
    main()
