#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path
from typing import List
import requests
import time
from operator import itemgetter

# -----------------------------------------------------------------------
# .env loader (robust, PyDroid/CLI friendly)
# -----------------------------------------------------------------------
from dotenv_fallback import load_dotenv, find_dotenv, dotenv_values
try:
    from services.env_loader import EnvLoader
except Exception:
    EnvLoader = None  # type: ignore

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
# Try to hydrate secure settings (if running inside the Django repo)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")
os.environ.setdefault("DJANGO_PREFER_SQLITE_FALLBACK", "1")
os.environ.setdefault("ALLOW_SQLITE_FALLBACK", "1")
repo_root = Path(__file__).resolve().parent
web_dir = repo_root / "web"
if str(web_dir) not in sys.path:
    sys.path.insert(0, str(web_dir))
if EnvLoader is not None:
    try:
        EnvLoader.load()
    except Exception:
        pass
try:
    import django
    django.setup()
    import importlib
    import services.secure_settings as secure_settings
    secure_settings = importlib.reload(secure_settings)
    os.environ.update(secure_settings.build_process_env())
except Exception:
    pass

# -----------------------------------------------------------------------
# CONFIGURATION — pulled from environment (no hard-coded secrets)
# -----------------------------------------------------------------------
# Required: your The Graph API key (Studio/Hosted Gateway)
THEGRAPH_API_KEY = os.getenv("THEGRAPH_API_KEY", "").strip()

# Resolve target chain before configuring subgraphs/paths
CHAIN_NAME = os.getenv("CHAIN_NAME", "base").strip().lower() or "base"

# Optional: override the Uniswap V2 subgraph ID via env.
# Default is ONLY for Ethereum; other chains require explicit subgraph IDs.
default_subgraph = ""
if CHAIN_NAME == "ethereum":
    default_subgraph = os.getenv("ETHEREUM_UNISWAP_SUBGRAPH_ID", "").strip() or "EYCKATKGBKLWvSfwvBjzfCBmGwYNdVkduYXVivCsLRFu"
elif CHAIN_NAME == "base":
    default_subgraph = os.getenv("BASE_UNISWAP_SUBGRAPH_ID", "").strip()
elif CHAIN_NAME == "arbitrum":
    default_subgraph = os.getenv("ARBITRUM_UNISWAP_SUBGRAPH_ID", "").strip()
elif CHAIN_NAME == "optimism":
    default_subgraph = os.getenv("OPTIMISM_UNISWAP_SUBGRAPH_ID", "").strip()
elif CHAIN_NAME == "polygon":
    default_subgraph = os.getenv("POLYGON_UNISWAP_SUBGRAPH_ID", "").strip()

UNISWAP_V2_SUBGRAPH_ID = os.getenv("UNISWAP_V2_SUBGRAPH_ID", "").strip() or default_subgraph

# Optional: allow full URL override (e.g., self-hosted/alt gateway)
if UNISWAP_V2_SUBGRAPH_ID:
    THEGRAPH_SUBGRAPH_URL = os.getenv(
        "THEGRAPH_SUBGRAPH_URL",
        f"https://gateway.thegraph.com/api/{THEGRAPH_API_KEY}/subgraphs/id/{UNISWAP_V2_SUBGRAPH_ID}"
    ).strip()
else:
    THEGRAPH_SUBGRAPH_URL = ""

# Output path can be overridden via env if desired
OUTPUT_PATH = os.getenv(
    "PAIR_INDEX_OUTPUT_PATH",
    os.path.join("data", "pair_index_%s.json" % CHAIN_NAME)
)

_GRAPH_REQUIRED = "gateway.thegraph.com" in THEGRAPH_SUBGRAPH_URL
_GRAPH_ENABLED = bool(THEGRAPH_SUBGRAPH_URL) and (not _GRAPH_REQUIRED or bool(THEGRAPH_API_KEY))
_FALLBACK_PAIR_LIMIT = int(os.getenv("PAIR_INDEX_FALLBACK_LIMIT", "1200"))
_FALLBACK_TOP_SYMBOLS = int(os.getenv("PAIR_INDEX_FALLBACK_TOP_SYMBOLS", "200"))
_DEXSCREENER_BACKOFF = float(os.getenv("DEXSCREENER_BACKOFF_SEC", "0.6"))
_DEXSCREENER_TIMEOUT = float(os.getenv("DEXSCREENER_TIMEOUT_SEC", "12"))

# -----------------------------------------------------------------------
# Fetch pairs ordered by a given field (1000 + 1000 for paging)
# -----------------------------------------------------------------------
def fetch_pairs(order_by_field: str, batch_size: int = 1000):
    # Use proper enum types for The Graph; orderBy varies by entity.
    # For Uniswap V2 'pairs', the enum is Pair_orderBy, orderDirection is OrderDirection.
    query = '''
    query($first: Int!, $skip: Int!, $orderBy: Pair_orderBy!, $orderDirection: OrderDirection!) {
      pairs(first: $first, skip: $skip, orderBy: $orderBy, orderDirection: $orderDirection) {
        id
        token0 { symbol }
        token1 { symbol }
      }
    }
    '''
    results = []
    session = requests.Session()
    for skip in (0, batch_size):
        resp = session.post(
            THEGRAPH_SUBGRAPH_URL,
            json={
                "query": query,
                "variables": {
                    "first": batch_size,
                    "skip": skip,
                    "orderBy": order_by_field,
                    "orderDirection": "desc",
                },
            },
            headers={"Content-Type": "application/json"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data and data["errors"]:
            raise RuntimeError(json.dumps(data["errors"], indent=2))
        batch = data.get("data", {}).get("pairs", []) or []
        print(f"Fetched {len(batch)} pairs by {order_by_field} (skip={skip})")
        results.extend(batch)
    return results

# -----------------------------------------------------------------------
# Fallback: Dexscreener + public market snapshots (no API keys)
# -----------------------------------------------------------------------

def _fetch_top_symbols(limit: int) -> List[str]:
    symbols: List[str] = []
    try:
        from services.public_api_clients import aggregate_market_data

        snapshots = aggregate_market_data(top_n=limit)
        for snap in snapshots:
            sym = (snap.symbol or "").upper().strip()
            if sym and sym not in symbols:
                symbols.append(sym)
    except Exception:
        symbols = []
    if not symbols:
        symbols = [
            "BTC",
            "ETH",
            "USDT",
            "USDC",
            "SOL",
            "BNB",
            "AVAX",
            "MATIC",
            "ARB",
            "OP",
            "DOGE",
            "LINK",
            "UNI",
            "AAVE",
        ]
    return symbols[:limit]


# DEXes that use the standard Uniswap token0()/token1() pool interface.
# Pools from other DEXes (Curve, Balancer, DODO, etc.) use different
# interfaces and will always fail resolve_pool_kind() — skip them at index time.
_UNISWAP_COMPATIBLE_DEXIDS = frozenset({
    "uniswap", "sushiswap", "pancakeswap", "quickswap", "camelot",
    "trader-joe", "spookyswap", "spiritswap", "velodrome", "aerodrome",
    "baseswap", "swapbased", "alienbase", "maverick", "thena",
    "ramses", "chronos", "zyberswap", "arbidex", "woofi",
    "solidly", "equalizer", "retro", "synthswap", "dackie",
})


# Minimum thresholds to filter out scam/dead tokens
_MIN_LIQUIDITY_USD = float(os.getenv("PAIR_INDEX_MIN_LIQUIDITY", "10000"))
_MIN_VOLUME_24H_USD = float(os.getenv("PAIR_INDEX_MIN_VOLUME_24H", "1000"))
_MIN_TX_COUNT_24H = int(os.getenv("PAIR_INDEX_MIN_TX_24H", "10"))

# Scam token heuristics
_SCAM_KEYWORDS = frozenset({
    "honeypot", "scam", "rug", "test", "fake", "airdrop", "giveaway",
    "elon", "trump2", "free", "moon1000x", "baby", "safemoon2", "rebase",
})


def _is_likely_scam(entry: dict) -> bool:
    """Heuristic scam filter based on DexScreener metadata."""
    base = entry.get("baseToken") or {}
    quote = entry.get("quoteToken") or {}
    base_sym = (base.get("symbol") or "").lower()
    base_name = (base.get("name") or "").lower()

    # Keyword check
    for kw in _SCAM_KEYWORDS:
        if kw in base_sym or kw in base_name:
            return True

    # Very low liquidity + high volume = likely wash trading
    liq = float((entry.get("liquidity") or {}).get("usd", 0) or 0)
    vol = float(entry.get("volume", {}).get("h24", 0) or 0)
    if liq > 0 and vol > 0 and vol / liq > 50:
        return True  # Volume is 50x liquidity — suspicious

    # Token created in last 24h with very high volume
    pair_created = entry.get("pairCreatedAt")
    if pair_created:
        try:
            age_hours = (time.time() * 1000 - float(pair_created)) / (1000 * 3600)
            if age_hours < 24 and vol > 100000:
                return True  # brand new + huge volume = likely scam
        except Exception:
            pass

    return False


def _get_wallet_symbols(chain: str) -> List[str]:
    """Read token symbols from the wallet's on-chain balances in the database."""
    wallet_symbols: List[str] = []
    try:
        from db import DB
        db = DB()
        # Get all wallets from secure vault
        rows = db.fetch_balances_flat(chains=[chain], include_zero=False)
        for row in rows:
            sym = (row["symbol"] if isinstance(row, dict) else row[5] or "").upper().strip()
            if sym and sym not in wallet_symbols and len(sym) <= 10:
                wallet_symbols.append(sym)
    except Exception as exc:
        print(f"[INFO] Could not read wallet balances for seeding: {exc}")
    return wallet_symbols


def build_index_from_dexscreener(chain: str) -> dict:
    chain = chain.lower().strip()
    target = int(os.getenv("PAIR_INDEX_TARGET", str(_FALLBACK_PAIR_LIMIT)))

    # Seed symbols: wallet holdings first (highest priority), then top market symbols
    wallet_syms = _get_wallet_symbols(chain)
    market_syms = _fetch_top_symbols(_FALLBACK_TOP_SYMBOLS)
    # Deduplicate while preserving wallet-first ordering
    seen = set()
    symbols: List[str] = []
    for sym in wallet_syms + market_syms:
        if sym not in seen:
            seen.add(sym)
            symbols.append(sym)
    if not symbols:
        return {}

    if wallet_syms:
        print(f"[INFO] Seeded with {len(wallet_syms)} wallet token(s): {wallet_syms[:20]}")

    session = requests.Session()
    results: dict = {}
    skipped_dexes: dict = {}
    scam_filtered = 0

    def _score(entry: dict, is_wallet_token: bool = False) -> float:
        volume = float(entry.get("volume", {}).get("h24", 0)
                       or entry.get("volumeUsd24h", 0)
                       or entry.get("volumeUsd", 0)
                       or 0.0)
        liquidity = 0.0
        liq = entry.get("liquidity") or {}
        try:
            liquidity = float(liq.get("usd") or 0.0)
        except Exception:
            liquidity = 0.0
        base_score = volume + (liquidity * 0.5)
        # Wallet tokens get a 3x priority boost
        if is_wallet_token:
            base_score *= 3.0
        return base_score

    wallet_sym_set = frozenset(s.upper() for s in wallet_syms)

    for symbol in symbols:
        try:
            resp = session.get(
                "https://api.dexscreener.com/latest/dex/search",
                params={"q": symbol},
                timeout=_DEXSCREENER_TIMEOUT,
            )
            if resp.status_code != 200:
                continue
            payload = resp.json() or {}
        except Exception:
            continue
        pairs = payload.get("pairs") or []
        for entry in pairs:
            try:
                chain_id = str(entry.get("chainId") or entry.get("chain") or "").lower()
                if chain_id != chain:
                    continue
                dex_id = str(entry.get("dexId") or "").lower().strip()
                if dex_id not in _UNISWAP_COMPATIBLE_DEXIDS:
                    skipped_dexes[dex_id] = skipped_dexes.get(dex_id, 0) + 1
                    continue

                # Scam filter
                if _is_likely_scam(entry):
                    scam_filtered += 1
                    continue

                # Minimum thresholds
                liq_usd = float((entry.get("liquidity") or {}).get("usd", 0) or 0)
                vol_24h = float(entry.get("volume", {}).get("h24", 0)
                               or entry.get("volumeUsd24h", 0)
                               or entry.get("volumeUsd", 0)
                               or 0)
                txns = entry.get("txns") or {}
                tx_24h = int((txns.get("h24") or {}).get("buys", 0) or 0) + int((txns.get("h24") or {}).get("sells", 0) or 0)

                base = entry.get("baseToken") or {}
                quote = entry.get("quoteToken") or {}
                base_sym = (base.get("symbol") or "").upper()
                is_wallet = base_sym in wallet_sym_set

                # Relax thresholds for wallet tokens (we want our own holdings)
                if not is_wallet:
                    if liq_usd < _MIN_LIQUIDITY_USD:
                        continue
                    if vol_24h < _MIN_VOLUME_24H_USD:
                        continue
                    if tx_24h < _MIN_TX_COUNT_24H:
                        continue

                pair_addr = str(entry.get("pairAddress") or "").strip()
                if not pair_addr:
                    continue
                sym = f"{base.get('symbol','')}-{quote.get('symbol','')}".strip("-")
                entry_score = _score(entry, is_wallet_token=is_wallet)
                existing = results.get(pair_addr)
                if existing and existing.get("score", 0) >= entry_score:
                    continue
                results[pair_addr] = {
                    "symbol": sym or "UNKNOWN",
                    "score": entry_score,
                    "source": "wallet" if is_wallet else "market",
                }
            except Exception:
                continue
        if len(results) >= target:
            break
        time.sleep(_DEXSCREENER_BACKOFF + (0.1 * (symbol.__len__() % 3)))

    if skipped_dexes:
        print(f"[INFO] Skipped {sum(skipped_dexes.values())} non-Uniswap pools: {dict(sorted(skipped_dexes.items(), key=lambda kv: -kv[1]))}")
    if scam_filtered:
        print(f"[INFO] Filtered {scam_filtered} likely-scam tokens")

    ranked = sorted(results.items(), key=lambda kv: kv[1].get("score", 0), reverse=True)
    trimmed = ranked[:target]
    index = {}
    wallet_count = 0
    for idx, (addr, meta) in enumerate(trimmed):
        source = meta.get("source", "market")
        index[addr] = {"index": idx, "symbol": meta.get("symbol", "UNKNOWN"), "source": source}
        if source == "wallet":
            wallet_count += 1

    print(f"[INFO] Index: {len(index)} pairs ({wallet_count} from wallet, {len(index) - wallet_count} from market)")
    return index

# -----------------------------------------------------------------------
# Main: combine rankings by volumeUSD and reserveUSD to decide "true" top
# -----------------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if os.path.exists(OUTPUT_PATH):
        print(f"Output already exists, skipping fetch: {OUTPUT_PATH}")
        return

    if _GRAPH_ENABLED:
        try:
            # 1) fetch top by cumulative swap volume (volumeUSD)
            vol_pairs = fetch_pairs("volumeUSD")
            # 2) fetch top by liquidity (reserveUSD)
            reserve_pairs = fetch_pairs("reserveUSD")
        except Exception as exc:
            print(f"Graph fetch failed: {exc}. Falling back to free sources.")
            vol_pairs = []
            reserve_pairs = []
    else:
        if _GRAPH_REQUIRED:
            print("Missing THEGRAPH_API_KEY for gateway; using free-source fallback.")
        vol_pairs = []
        reserve_pairs = []

    if not vol_pairs and not reserve_pairs:
        fallback_index = build_index_from_dexscreener(chain=CHAIN_NAME)
        if not fallback_index:
            raise RuntimeError("Unable to build pair index: graph unavailable and fallback returned empty.")
        with open(OUTPUT_PATH, "w") as f:
            json.dump(fallback_index, f, indent=2)
        print(f"Saved {len(fallback_index)} pairs to {OUTPUT_PATH} (fallback)")
        return

    # build ranking dicts
    vol_rank = {p["id"]: idx for idx, p in enumerate(vol_pairs)}
    res_rank = {p["id"]: idx for idx, p in enumerate(reserve_pairs)}

    # union of all pair IDs
    all_ids = set(vol_rank.keys()) | set(res_rank.keys())

    # compute combined rank (sum of two ranks)
    combined = []
    for pid in all_ids:
        vr = vol_rank.get(pid, len(vol_pairs))
        rr = res_rank.get(pid, len(reserve_pairs))
        combined.append((pid, vr + rr))
    combined.sort(key=itemgetter(1))

    # take top 2000
    top2000 = combined[:2000]

    # map symbols
    symbol_map = {p["id"]: f"{p['token0']['symbol']}-{p['token1']['symbol']}" for p in (vol_pairs + reserve_pairs)}

    # build final index
    index = {}
    for idx, (pid, _) in enumerate(top2000):
        index[pid] = {"index": idx, "symbol": symbol_map.get(pid, "UNKNOWN")}

    # save to file
    with open(OUTPUT_PATH, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Saved {len(index)} pairs to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
