#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path
from typing import List
import requests
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

# Optional: override the Uniswap V2 subgraph ID via env,
# falls back to the given default if not provided.
default_subgraph = "EYCKATKGBKLWvSfwvBjzfCBmGwYNdVkduYXVivCsLRFu"
if CHAIN_NAME == "base":
    default_subgraph = os.getenv("BASE_UNISWAP_SUBGRAPH_ID", "").strip() or default_subgraph

UNISWAP_V2_SUBGRAPH_ID = os.getenv(
    "UNISWAP_V2_SUBGRAPH_ID",
    default_subgraph
).strip()

# Optional: allow full URL override (e.g., self-hosted/alt gateway)
THEGRAPH_SUBGRAPH_URL = os.getenv(
    "THEGRAPH_SUBGRAPH_URL",
    f"https://gateway.thegraph.com/api/{THEGRAPH_API_KEY}/subgraphs/id/{UNISWAP_V2_SUBGRAPH_ID}"
).strip()

# Output path can be overridden via env if desired
OUTPUT_PATH = os.getenv(
    "PAIR_INDEX_OUTPUT_PATH",
    os.path.join("data", "pair_index_%s.json" % CHAIN_NAME)
)

if "gateway.thegraph.com" in THEGRAPH_SUBGRAPH_URL and not THEGRAPH_API_KEY:
    raise RuntimeError(
        "Missing THEGRAPH_API_KEY in environment for gateway.thegraph.com.\n"
        "Set THEGRAPH_API_KEY in your .env or provide THEGRAPH_SUBGRAPH_URL to bypass."
    )

if CHAIN_NAME == "base" and not THEGRAPH_SUBGRAPH_URL:
    raise RuntimeError("Provide THEGRAPH_SUBGRAPH_URL for Base network pairs.")

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
            headers={"Content-Type": "application/json"}
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
# Main: combine rankings by volumeUSD and reserveUSD to decide "true" top
# -----------------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if os.path.exists(OUTPUT_PATH):
        print(f"Output already exists, skipping fetch: {OUTPUT_PATH}")
        return

    # 1) fetch top by cumulative swap volume (volumeUSD)
    vol_pairs = fetch_pairs("volumeUSD")
    # 2) fetch top by liquidity (reserveUSD)
    reserve_pairs = fetch_pairs("reserveUSD")

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
