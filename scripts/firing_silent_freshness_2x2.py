"""p705 Echo: rebuild the p696 n=53 FIRING/SILENT join, then 2x2 it on feed freshness.

Pre-registered kill test: if FIRING's win-rate advantage vanishes inside the
live-feed subset, the "filter is real" half of [3e2a8552] is a freshness artifact.
Recipe is from the item: trade_fills entry rows (details.mode ends _entry),
tied to non-annulled trade_outcomes on trade_id, wallet-filtered, NaN-guarded.
Freshness from market_stream (NOT prices -- p704: prices returns no rows for
AERO and COMP, both provably live).
"""
import json
import math
import sqlite3
import sys
from collections import defaultdict

DB = "storage/trading_cache.db"
SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "AERO-USDC"
LOOKBACK_H = float(sys.argv[2]) if len(sys.argv) > 2 else 24.0
BAR_H = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row


def finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


NOW = conn.execute("SELECT MAX(ts) FROM market_stream").fetchone()[0]

# ---- entry rows -------------------------------------------------------------
entries = {}  # trade_id -> (ts, price, symbol)
for r in conn.execute("SELECT ts, symbol, executed_price, expected_price, details FROM trade_fills"):
    try:
        d = json.loads(r["details"]) if r["details"] else {}
    except Exception:
        continue
    if not str(d.get("mode", "")).endswith("_entry"):
        continue
    tid = d.get("trade_id")
    if not tid:
        continue
    px = finite(r["executed_price"]) or finite(r["expected_price"])
    ts = finite(r["ts"])
    if px is None or ts is None or px <= 0:
        continue
    if SYMBOL != "ALL" and r["symbol"] != SYMBOL:
        continue
    entries[tid] = (ts, px, r["symbol"])

# ---- outcomes --------------------------------------------------------------
out_rows = defaultdict(list)
wallets = defaultdict(int)
q = "SELECT trade_id, wallet, symbol, net_profit, status FROM trade_outcomes WHERE status IS NOT 'annulled'"
for r in conn.execute(q):
    if str(r["status"] or "").lower() == "annulled":
        continue
    if r["trade_id"] not in entries:
        continue
    np_ = finite(r["net_profit"])
    if np_ is None:
        continue
    out_rows[r["trade_id"]].append((r["wallet"], r["symbol"], np_))
    wallets[r["wallet"]] += 1

print(f"symbol={SYMBOL}  entry rows w/ trade_id={len(entries)}  joined trade_ids={len(out_rows)}")
print(f"wallets seen: {dict(wallets)}")

# ---- freshness per symbol (hours since last tick) ---------------------------
fresh = {}
for r in conn.execute("SELECT symbol, MAX(ts) m, COUNT(*) n FROM market_stream GROUP BY symbol"):
    fresh[r["symbol"]] = ((NOW - r["m"]) / 3600.0, r["n"])

# ---- donchian state at entry ------------------------------------------------
LB = LOOKBACK_H * BAR_H * 3600.0
rows = []
dark = 0
for tid, (ts, px, sym) in sorted(entries.items(), key=lambda kv: kv[1][0]):
    if tid not in out_rows:
        continue
    hi = conn.execute(
        "SELECT MAX(price) FROM market_stream WHERE symbol=? AND ts < ? AND ts >= ? AND price > 0",
        (sym, ts, ts - LB),
    ).fetchone()[0]
    hi = finite(hi)
    if hi is None:
        dark += 1
        continue
    firing = px >= hi
    age_h, ticks = fresh.get(sym, (float("inf"), 0))
    for (wal, osym, np_) in out_rows[tid]:
        rows.append(dict(tid=tid, sym=sym, ts=ts, firing=firing, net=np_, age_h=age_h, wallet=wal))

print(f"dark-at-entry dropped: {dark}   final rows: {len(rows)}")


def stat(label, rs):
    if not rs:
        print(f"  {label:<22} n=0")
        return
    nets = sorted(r["net"] for r in rs)
    n = len(nets)
    s = sum(nets)
    med = nets[n // 2] if n % 2 else (nets[n // 2 - 1] + nets[n // 2]) / 2
    wins = sum(1 for v in nets if v > 0)
    print(f"  {label:<22} n={n:<4} sum={s:+.5f}  median={med:+.6f}  win={wins}/{n} = {100.0*wins/n:.0f}%")
    if n >= 4:
        d1 = sum(nets[1:])
        d2 = sum(nets[2:])
        print(f"      drop-worst-1 sum={d1:+.5f}   drop-worst-2 sum={d2:+.5f}   "
              f"worst3={nets[0]:+.5f},{nets[1]:+.5f},{nets[2]:+.5f}")


print("\n=== REPRODUCTION (no freshness split) ===")
stat("FIRING", [r for r in rows if r["firing"]])
stat("SILENT", [r for r in rows if not r["firing"]])

for cut in (24.0, 72.0):
    print(f"\n=== 2x2 on freshness, LIVE = last tick < {cut:.0f}h ago ===")
    stat(f"FIRING + LIVE", [r for r in rows if r["firing"] and r["age_h"] < cut])
    stat(f"FIRING + DEAD", [r for r in rows if r["firing"] and r["age_h"] >= cut])
    stat(f"SILENT + LIVE", [r for r in rows if not r["firing"] and r["age_h"] < cut])
    stat(f"SILENT + DEAD", [r for r in rows if not r["firing"] and r["age_h"] >= cut])

print("\n=== symbols in the joined population (age_h since last tick) ===")
bysym = defaultdict(lambda: [0, 0, 0.0])
for r in rows:
    b = bysym[r["sym"]]
    b[0] += 1
    b[1] += 1 if r["firing"] else 0
    b[2] = r["age_h"]
for sym, (n, nf, age) in sorted(bysym.items(), key=lambda kv: -kv[1][0]):
    print(f"  {sym:<18} rows={n:<4} firing={nf:<4} last_tick={age:.1f}h ago")
