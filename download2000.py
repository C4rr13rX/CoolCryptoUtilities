#!/usr/bin/env python3
import os
import json
import time
import re
import random
from datetime import datetime, timedelta
from threading import Thread, Semaphore, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from web3 import Web3

# --- CONFIGURATION ---
ANKR_API_KEY         = "YOUR_API_KEY"
PAIR_ASSIGNMENT_FILE = "data/pair_provider_assignment.json"
OUTPUT_DIR           = "data/historical_ohlcv"
INTERMEDIATE_DIR     = "data/intermediate"
YEARS_BACK           = 3
GRANULARITY_SECONDS  = 60 * 5  # 5 min
MAX_WORKERS          = 30
ANKR_RPS_LIMIT       = 30
LOGS_PER_PARSE_BATCH = 10

# --- RATE LIMITER ---
bucket = Semaphore(0)
def _refill_bucket():
    while True:
        for _ in range(ANKR_RPS_LIMIT):
            bucket.release()
        time.sleep(1)
Thread(target=_refill_bucket, daemon=True).start()

def limited(fn, *args, **kwargs):
    bucket.acquire()
    return fn(*args, **kwargs)

# --- WEB3 SETUP ---
rpc_url = f"https://rpc.ankr.com/eth/{ANKR_API_KEY}"
web3    = Web3(Web3.HTTPProvider(rpc_url))
assert web3.is_connected(), "❌ Could not connect to Ankr RPC"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

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

ERC20_ABI = json.loads("""[
  {"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"stateMutability":"view","type":"function"}
]""")

# --- HELPERS ---
def parse_delay(msg: str):
    m = re.search(r'retry in (\d+)([smh])', msg)
    if not m: return None
    val, unit = int(m.group(1)), m.group(2)
    return val * {'s':1,'m':60,'h':3600}[unit]

def get_block_by_timestamp(ts: int):
    latest = limited(lambda: web3.eth.block_number)
    latest_ts = limited(web3.eth.get_block, latest)["timestamp"]
    return max(1, latest - int((latest_ts - ts) / 13))

def fetch_chunk(ev, b, e_b, sym, idx, total):
    delays = [30, 30, 30, 60]
    for attempt, delay in enumerate(delays, start=1):
        try:
            logs = limited(ev.get_logs, from_block=b, to_block=e_b)
            print(f"    [{idx+1}/{total}] 📦 {sym}: {len(logs)} logs [{b}→{e_b}]")
            return logs
        except Exception as exc:
            print(f"    ⚠️ fetch error for {sym}[{b}→{e_b}] (attempt {attempt}/{len(delays)}): {exc}")
            if attempt < len(delays):
                sleep_sec = delay + random.uniform(0, 5)
                time.sleep(sleep_sec)
            else:
                print(f"    ❌ giving up on chunk [{b}→{e_b}] after {len(delays)} attempts, skipping")
                return []

def get_block_with_retry(block_number, retries=7):
    for attempt in range(retries):
        try:
            return limited(web3.eth.get_block, block_number)
        except Exception as e:
            msg = str(e)
            delay = parse_delay(msg) or (10 + random.uniform(0, 5))
            print(f"      Block fetch rate-limited or failed, retrying in {delay}s (attempt {attempt+1}/{retries})")
            time.sleep(delay)
    raise RuntimeError(f"Failed to fetch block {block_number} after {retries} retries")

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_logs_no_ts(logs, dec0, dec1, granularity_seconds):
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
        bar_slot = blk_num // granularity_seconds  # still just for grouping
        records_by_bar.setdefault(bar_slot, []).append(rec)
        blocks_by_bar.setdefault(bar_slot, set()).add(blk_num)
    return records_by_bar, blocks_by_bar

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
    # Fix up set-serialization
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

def update_assignment(assignment, addr, **fields):
    assignment["pairs"][addr].update(fields)
    with open(PAIR_ASSIGNMENT_FILE, "w") as wf:
        json.dump(assignment, wf, indent=2)

def main():
    with open(PAIR_ASSIGNMENT_FILE) as f:
        assignment = json.load(f)

    start_ts = int((datetime.utcnow() - timedelta(days=YEARS_BACK*365)).timestamp())
    start_blk = get_block_by_timestamp(start_ts)
    end_blk   = limited(lambda: web3.eth.block_number)
    print(f"⏳ Scanning blocks {start_blk} → {end_blk}")

    chunks_count = YEARS_BACK*365*24
    total_blocks = end_blk - start_blk
    CHUNK_SIZE_BLOCKS = max(1, total_blocks // chunks_count)
    print(f"ℹ️ Dynamic CHUNK_SIZE set to {CHUNK_SIZE_BLOCKS} blocks")

    pairs = [(addr,meta) for addr,meta in assignment["pairs"].items() if not meta.get("completed",False)]
    print(f"🚀 Will process {len(pairs)} pairs…")

    for addr, meta in pairs:
        sym = meta["symbol"]
        idx = meta["index"]
        out = os.path.join(OUTPUT_DIR, f"{idx:04d}_{sym}.json")
        pair_dir = os.path.join(INTERMEDIATE_DIR, sym)
        print(f"\n=== Processing {sym} ===")

        pair_start_blk = meta.get("next_block", None)
        if pair_start_blk is None:
            pair_start_blk = start_blk
        else:
            pair_start_blk = int(pair_start_blk)

        pair = web3.eth.contract(address=Web3.to_checksum_address(addr), abi=PAIR_ABI)
        ev   = pair.events.Swap()
        t0   = pair.functions.token0().call()
        t1   = pair.functions.token1().call()
        dec0 = web3.eth.contract(address=t0,   abi=ERC20_ABI).functions.decimals().call()
        dec1 = web3.eth.contract(address=t1,   abi=ERC20_ABI).functions.decimals().call()
        print(f"   tokens: {t0}(dec{dec0}) / {t1}(dec{dec1})")

        b = pair_start_blk
        chunk_idx = 0
        total_chunks = ((end_blk - b) + CHUNK_SIZE_BLOCKS - 1) // CHUNK_SIZE_BLOCKS
        while b < end_blk:
            e_b = min(b + CHUNK_SIZE_BLOCKS, end_blk)
            if os.path.exists(os.path.join(pair_dir, f"chunk_{chunk_idx:06d}.json")):
                print(f"    Skipping previously processed chunk {chunk_idx+1}/{total_chunks}")
                b = e_b
                chunk_idx += 1
                continue

            print(f"    ⏳ Fetching chunk {chunk_idx+1}/{total_chunks} [{b}→{e_b}] for {sym}")
            logs = fetch_chunk(ev, b, e_b, sym, chunk_idx, total_chunks)
            if logs:
                log_batches = list(chunked(logs, LOGS_PER_PARSE_BATCH))
                # Note: We group by bar_slot/block here for speed, merge later.
                chunk_records_by_bar = {}
                chunk_blocks_by_bar = {}
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                    futs = [ex.submit(parse_logs_no_ts, batch, dec0, dec1, GRANULARITY_SECONDS) for batch in log_batches]
                    for pf in as_completed(futs):
                        try:
                            recs_by_bar, blocks_by_bar = pf.result()
                            for bar_slot, recs in recs_by_bar.items():
                                chunk_records_by_bar.setdefault(bar_slot, []).extend(recs)
                            for bar_slot, blocks in blocks_by_bar.items():
                                chunk_blocks_by_bar.setdefault(bar_slot, set()).update(blocks)
                        except Exception as e:
                            print(f"      Parse error: {e}. Skipping one batch.")
                # Save intermediate chunk for restartability
                save_intermediate_chunk(pair_dir, chunk_idx, chunk_records_by_bar, chunk_blocks_by_bar)
                update_assignment(assignment, addr, next_block=e_b)
            else:
                update_assignment(assignment, addr, next_block=e_b)
            b = e_b
            chunk_idx += 1

        # Now all intermediate chunks exist for this pair; merge, timestamp, aggregate, save
        print("    ⏳ Merging intermediate data and fetching timestamps…")
        all_records_by_bar, min_block_per_bar = load_intermediate_chunks(pair_dir)
        min_block_per_bar = {bar_slot: min(blocks) for bar_slot, blocks in min_block_per_bar.items() if blocks}
        bar_slot_to_ts = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(get_block_with_retry, block): bar_slot for bar_slot, block in min_block_per_bar.items()}
            for pf in as_completed(futs):
                bar_slot = futs[pf]
                try:
                    blk = pf.result()
                    bar_slot_to_ts[bar_slot] = blk["timestamp"]
                except Exception as e:
                    print(f"      Failed to fetch timestamp for bar {bar_slot}: {e}")

        print("    🛠 Aggregating OHLCV bars…")
        bars = []
        for bar_slot, records in all_records_by_bar.items():
            ts = bar_slot_to_ts.get(bar_slot)
            if ts is not None:
                bars.append(aggregate_ohlcv_by_bar(records, ts))
        bars.sort(key=lambda x: x["timestamp"])
        save_bars(out, bars)
        update_assignment(assignment, addr, completed=True)
        print(f"    ✅ Pair {sym} complete and saved.")

    print("\n🎉 All pairs complete.")

if __name__=="__main__":
    main()
