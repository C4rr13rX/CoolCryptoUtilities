#!/usr/bin/env python3
import os
import json
import time
import re
from datetime import datetime, timedelta
from threading import Thread, Semaphore
from concurrent.futures import ThreadPoolExecutor, as_completed
from web3 import Web3

# ---------------------------- CONFIGURATION ----------------------------
ANKR_API_KEY         = "YOUR_ANKR_API_KEY" # https://www.ankr.com/web3-api/
PAIR_ASSIGNMENT_FILE = "data/pair_provider_assignment.json"
OUTPUT_DIR           = "data/historical_ohlcv"
YEARS_BACK           = 5
GRANULARITY_SECONDS  = 60
MAX_WORKERS          = 60     # threads _per pair_
ANKR_RPS_LIMIT       = 30     # free tier: 30 req/sec

# ---------------------------- RATE LIMITER ----------------------------
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

# ---------------------------- WEB3 SETUP ----------------------------
rpc_url = f"https://rpc.ankr.com/eth/{ANKR_API_KEY}"
web3    = Web3(Web3.HTTPProvider(rpc_url))
assert web3.is_connected(), "❌ Could not connect to Ankr RPC"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------- ABI SNIPPETS ----------------------------
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

# ---------------------------- HELPERS ----------------------------
_retry_re = re.compile(r'retry in (\d+)([smh])')
def parse_delay(msg: str):
    m = _retry_re.search(msg)
    if not m:
        return None
    val, unit = int(m.group(1)), m.group(2)
    return val * {'s':1,'m':60,'h':3600}[unit]

def get_block_by_timestamp(ts: int):
    latest = limited(lambda: web3.eth.block_number)
    latest_ts = limited(web3.eth.get_block, latest)["timestamp"]
    return max(1, latest - int((latest_ts - ts) / 13))

# ---------------------------- AGGREGATION ----------------------------
def aggregate_ohlcv(records):
    print("🔨 Aggregating OHLCV bars...")
    bars = {}
    for r in records:
        t   = r["timestamp"] - (r["timestamp"] % GRANULARITY_SECONDS)
        p   = r["price"]
        bv  = r["buy_volume"]
        sv  = r["sell_volume"]
        nv  = bv - sv
        vol = bv + sv
        if t not in bars:
            bars[t] = {"timestamp":t, "open":p, "high":p, "low":p, "close":p,
                       "buy_volume":bv, "sell_volume":sv, "net_volume":nv,
                       "_sum_pv":p*vol, "_sum_v":vol}
        else:
            b = bars[t]
            b["high"]        = max(b["high"], p)
            b["low"]         = min(b["low"], p)
            b["close"]       = p
            b["buy_volume"]  += bv
            b["sell_volume"] += sv
            b["net_volume"]  += nv
            b["_sum_pv"]     += p * vol
            b["_sum_v"]      += vol
    output=[]
    for t in sorted(bars):
        b = bars[t]
        b["vwap"] = b["_sum_pv"]/b["_sum_v"] if b["_sum_v"]>0 else b["close"]
        del b["_sum_pv"]; del b["_sum_v"]
        output.append(b)
    return output

# ---------------------------- FETCH CHUNK ----------------------------
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
                time.sleep(delay)
            else:
                print(f"    ❌ giving up on chunk [{b}→{e_b}] after {len(delays)} attempts, skipping")
                return []

# ---------------------------- PARSING WORKER ----------------------------
def parse_logs(logs, dec0, dec1):
    recs=[]
    for lg in logs:
        args = lg['args']
        blk = limited(web3.eth.get_block, lg['blockNumber'])
        ts  = blk['timestamp']
        a1out = int(args['amount1Out']) / 10**dec1
        a1in  = int(args['amount1In'])  / 10**dec1
        if a1out > 0:
            recs.append({"timestamp": ts, "price": int(args['amount0In'])/10**dec0/a1out,
                         "buy_volume": a1out, "sell_volume": 0.0})
        if a1in > 0:
            recs.append({"timestamp": ts, "price": int(args['amount0Out'])/10**dec0/a1in,
                         "buy_volume": 0.0,    "sell_volume": a1in})
    print(f"    🛠 Parsed {len(logs)} logs in parallel")
    return recs

# ---------------------------- MAIN ----------------------------
def main():
    with open(PAIR_ASSIGNMENT_FILE) as f:
        assignment = json.load(f)

    start_ts = int((datetime.utcnow() - timedelta(days=YEARS_BACK*365)).timestamp())
    start_blk= get_block_by_timestamp(start_ts)
    end_blk  = limited(lambda: web3.eth.block_number)
    print(f"⏳ Scanning blocks {start_blk} → {end_blk}")

    total_blocks = end_blk - start_blk
    chunks_count = YEARS_BACK*365*24
    CHUNK_SIZE_BLOCKS = max(1, total_blocks // chunks_count)
    print(f"ℹ️ Dynamic CHUNK_SIZE set to {CHUNK_SIZE_BLOCKS} blocks (~{(CHUNK_SIZE_BLOCKS*13)/60:.1f} min)")

    pairs = [(addr,meta) for addr,meta in assignment.items() if not meta.get("completed",False)]
    print(f"🚀 Will process {len(pairs)} pairs…")

    for addr,meta in pairs:
        sym = meta["symbol"]; idx = meta["index"]
        out = os.path.join(OUTPUT_DIR, f"{idx:04d}_{sym}.json")
        if os.path.exists(out):
            assignment[addr]["completed"] = True
            continue

        pair = web3.eth.contract(address=Web3.to_checksum_address(addr), abi=PAIR_ABI)
        ev   = pair.events.Swap()
        t0   = pair.functions.token0().call()
        t1   = pair.functions.token1().call()
        dec0 = web3.eth.contract(address=t0,   abi=ERC20_ABI).functions.decimals().call()
        dec1 = web3.eth.contract(address=t1,   abi=ERC20_ABI).functions.decimals().call()
        print(f"   tokens: {t0}(dec{dec0}) / {t1}(dec{dec1})")

        # build and fetch chunks
        chunks = []
        b = start_blk
        while b < end_blk:
            e_b = min(b + CHUNK_SIZE_BLOCKS, end_blk)
            chunks.append((b, e_b))
            b = e_b
        total = len(chunks)
        print(f"    → {total} chunks to fetch")

        all_recs = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            fetch_futs = {ex.submit(fetch_chunk, ev, b, e_b, sym, i, total): (b, e_b)
                          for i, (b, e_b) in enumerate(chunks)}
            parse_futs = []
            completed = 0
            start_time= time.time()
            for fut in as_completed(fetch_futs):
                completed += 1
                elapsed   = int(time.time() - start_time)
                print(f"    ⏳ Fetched {completed}/{total} chunks (elapsed {elapsed}s)")
                logs = fut.result()
                # offload parsing
                parse_futs.append(ex.submit(parse_logs, logs, dec0, dec1))

            # collect parsed records
            for pf in as_completed(parse_futs):
                all_recs.extend(pf.result())

        # aggregate and save
        bars = aggregate_ohlcv(all_recs)
        with open(out, 'w') as f:
            json.dump(bars, f, indent=2)
        assignment[addr]["completed"] = True
        with open(PAIR_ASSIGNMENT_FILE, 'w') as wf:
            json.dump(assignment, wf, indent=2)

    print("\n🎉 All pairs complete.")

if __name__=="__main__":
    main()
