#!/usr/bin/env python3
import os
import requests
import json
from operator import itemgetter

# -----------------------------------------------------------------------
# 🧠 CONFIGURATION — replace with your actual values
# -----------------------------------------------------------------------
API_KEY = "YOUR_THEGRAPH_API_KEY" # https://thegraph.com/studio/
SUBGRAPH_ID = "EYCKATKGBKLWvSfwvBjzfCBmGwYNdVkduYXVivCsLRFu" #UNISWAP V2
URL = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}"
OUTPUT_PATH = os.path.join("data", "pair_index_top2000.json")

# -----------------------------------------------------------------------
# Fetch pairs ordered by a given field (1000 + 1000 for paging)
# -----------------------------------------------------------------------
def fetch_pairs(order_by_field: str, batch_size: int = 1000):
    query = '''
    query($first: Int!, $skip: Int!, $orderBy: String!) {
      pairs(first: $first, skip: $skip, orderBy: $orderBy, orderDirection: desc) {
        id
        token0 { symbol }
        token1 { symbol }
      }
    }
    '''
    results = []
    for skip in (0, batch_size):
        resp = requests.post(
            URL,
            json={"query": query, "variables": {"first": batch_size, "skip": skip, "orderBy": order_by_field}}
        )
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(data["errors"])
        batch = data["data"]["pairs"]
        print(f"Fetched {len(batch)} pairs by {order_by_field} (skip={skip})")
        results.extend(batch)
    return results

# -----------------------------------------------------------------------
# Main: combine rankings by volumeUSD and reserveUSD to decide "true" top
# -----------------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if os.path.exists(OUTPUT_PATH):
        print(f"⚠️  {OUTPUT_PATH} already exists, skipping fetch")
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
    symbol_map = {p["id"]: f"{p['token0']['symbol']}-{p['token1']['symbol']}" for p in vol_pairs + reserve_pairs}

    # build final index
    index = {}
    for idx, (pid, _) in enumerate(top2000):
        index[pid] = {"index": idx, "symbol": symbol_map.get(pid, "UNKNOWN")}

    # save to file
    with open(OUTPUT_PATH, "w") as f:
        json.dump(index, f, indent=2)
    print(f"✅ Saved {len(index)} pairs to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
