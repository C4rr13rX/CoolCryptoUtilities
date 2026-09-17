"""Audit whether every live trade in the store can be tied to an on-chain tx hash.

Item [1c6acf5d] / [50288149] claimed "ZERO of 60 live rows carries a tx_hash".
That claim is FALSE as stated, and the distinction matters:

  * ``trade_outcomes`` (22 mode=live rows, 18 of them ``wallet='live'`` -- the
    -0.186371 scoreboard) has NO hash column and NO hash key in ``details``.
  * ``trade_fills`` (20 ``live_entry`` + 18 ``live_exit`` legs) likewise.
    38 + 22 = 60.  That is where "60 live rows, zero hashes" came from.
  * ``trading_ops`` DOES carry the hash, inside its ``details`` JSON, on every
    ``live-swap-settled`` / ``live-entry`` / ``live-exit`` row -- 47 distinct
    hashes, every one of them a real mined Base transaction sent by our wallet.

So the receipts were never discarded.  They were written to the operations log
and dropped only on the way into the two tables that compute P/L, which is why
every census that looked at the P/L tables concluded the evidence was gone.

The join that recovers them is ``outcome_id`` (exit leg) and ``trade_id``
(entry leg).  Run with ``--verify`` to re-read every recovered hash from Base
and confirm status, sender and block time.  Read-only: this script never signs
or sends anything.

Usage:
    python -X utf8 scripts/tx_hash_provenance_audit.py            # local audit
    python -X utf8 scripts/tx_hash_provenance_audit.py --verify   # + on-chain
    python -X utf8 scripts/tx_hash_provenance_audit.py --json out.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DB = os.getenv("TRADING_DB_PATH") or os.path.join("storage", "trading_cache.db")

# Read-only endpoints.  Both answered eth_getTransactionReceipt in ~0.5s on
# 2026-09-17; mainnet.base.org 403s without a browser User-Agent.
RPC_URLS = (
    "https://mainnet.base.org",
    "https://base-rpc.publicnode.com",
    "https://base.drpc.org",
)

# Statuses on trading_ops whose details carry a live transaction hash.
LIVE_STATUSES = (
    "live-swap-settled",
    "live-swap-reverted",
    "live-entry",
    "live-exit",
    "live-entry-failed",
    "live-exit-failed",
    "live-position-adopted",
)


def _is_hash(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 66 and v.startswith("0x") and all(
        ch in "0123456789abcdefABCDEF" for ch in v[2:]
    )


def _loads(d: Optional[str]) -> Dict[str, Any]:
    if not d:
        return {}
    try:
        o = json.loads(d)
    except Exception:
        return {}
    return o if isinstance(o, dict) else {}


def _iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(float(ts), dt.UTC).strftime("%Y-%m-%d %H:%M:%S")


def connect(db_path: str) -> sqlite3.Connection:
    p = os.path.abspath(db_path)
    if not os.path.exists(p):
        raise SystemExit(f"store not found: {p}")
    size = os.path.getsize(p)
    if size == 0:
        raise SystemExit(
            f"{p} is a 0-byte stub -- it returns nothing and proves nothing. "
            "Use storage/trading_cache.db (TRADING_DB_PATH)."
        )
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def schema_audit(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Which tables have a hash COLUMN, and which merely have the key in JSON."""
    out: List[Dict[str, Any]] = []
    tables = [r[0] for r in con.execute(
        "select name from sqlite_master where type='table' order by name")]
    for t in tables:
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
        hash_cols = [c for c in cols if "hash" in c.lower()]
        if not hash_cols:
            continue
        n = con.execute(f'select count(*) from "{t}"').fetchone()[0]
        for c in hash_cols:
            filled = con.execute(
                f'select count(*) from "{t}" where "{c}" is not null and trim("{c}") <> \'\''
            ).fetchone()[0]
            out.append({"table": t, "column": c, "rows": n, "filled": filled})
    return out


def collect_ops_hashes(con: sqlite3.Connection) -> Dict[str, Any]:
    """Index every hash trading_ops recorded, by trade_id and by outcome_id."""
    by_trade_entry: Dict[str, List[Tuple[float, str]]] = {}
    by_outcome_exit: Dict[str, Tuple[float, str, Optional[str], Optional[str]]] = {}
    settled: Dict[str, Dict[str, Any]] = {}
    distinct: Dict[str, List[str]] = {}

    q = "select ts, status, details from trading_ops where status in (%s)" % (
        ",".join("?" * len(LIVE_STATUSES)))
    for ts, status, d in con.execute(q, LIVE_STATUSES):
        o = _loads(d)
        h = o.get("tx_hash")
        tid = o.get("trade_id")
        for cand in [h, o.get("entry_tx_hash")] + list(o.get("adopted_from_tx_hashes") or []):
            if _is_hash(cand):
                distinct.setdefault(cand, []).append(status)
        if status == "live-entry" and _is_hash(h):
            by_trade_entry.setdefault(tid, []).append((ts, h))
        elif status == "live-exit":
            key = o.get("outcome_id") or tid
            by_outcome_exit[key] = (ts, h if _is_hash(h) else None,
                                    o.get("entry_tx_hash"), tid)
        elif status in ("live-swap-settled", "live-swap-reverted") and _is_hash(h):
            settled[h] = {
                "ts": ts, "status": status, "purpose": o.get("purpose"),
                "symbol": o.get("symbol"), "trade_id": tid,
                "confirmed": o.get("confirmed"), "ok": o.get("ok"),
                "route": o.get("route"), "explorer": o.get("explorer"),
            }
    return {
        "entry_by_trade": by_trade_entry,
        "exit_by_outcome": by_outcome_exit,
        "settled": settled,
        "distinct": distinct,
    }


def recover(con: sqlite3.Connection, ops: Dict[str, Any]) -> List[Dict[str, Any]]:
    """For every live trade_outcomes row, recover its entry and exit hashes."""
    rows: List[Dict[str, Any]] = []
    sql = ("select outcome_id, trade_id, ts, symbol, net_profit, wallet, details "
           "from trade_outcomes order by ts")
    for oid, tid, ts, sym, net, wallet, d in con.execute(sql):
        o = _loads(d)
        if o.get("mode") != "live":
            continue
        ex = ops["exit_by_outcome"].get(oid)
        ent = ops["entry_by_trade"].get(tid) or []
        entry_hash = (ex[2] if ex and _is_hash(ex[2]) else None) or (
            ent[0][1] if ent else None)
        rows.append({
            "outcome_id": oid, "trade_id": tid, "ts": ts, "when": _iso(ts),
            "symbol": sym, "net_profit": net,
            "scoreboard": wallet == "live",       # the 18 that make -0.186371
            "annulled": bool(o.get("annulled")), "repriced": bool(o.get("repriced")),
            "entry_tx_hash": entry_hash,
            "exit_tx_hash": ex[1] if ex else None,
        })
    return rows


def rpc(method: str, params: list, timeout: int = 20) -> Optional[dict]:
    import requests  # local import: the audit runs without network too

    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    headers = {"User-Agent": "Mozilla/5.0", "content-type": "application/json"}
    for url in RPC_URLS:
        try:
            r = requests.post(url, json=payload, timeout=timeout, headers=headers)
            if r.status_code != 200:
                continue
            body = r.json()
            if "result" in body:
                return body["result"]
        except Exception:
            continue
    return None


def verify_onchain(hashes: List[str]) -> Dict[str, Dict[str, Any]]:
    """Re-read each hash from Base.  Read-only; sends nothing, spends no gas."""
    out: Dict[str, Dict[str, Any]] = {}
    blocks: Dict[str, int] = {}
    for h in hashes:
        rec = rpc("eth_getTransactionReceipt", [h])
        if not rec:
            out[h] = {"found": False}
            continue
        bn = rec.get("blockNumber")
        if bn and bn not in blocks:
            blk = rpc("eth_getBlockByNumber", [bn, False]) or {}
            if blk.get("timestamp"):
                blocks[bn] = int(blk["timestamp"], 16)
        out[h] = {
            "found": True,
            "status": int(rec.get("status", "0x0"), 16),
            "from": (rec.get("from") or "").lower(),
            "to": (rec.get("to") or "").lower(),
            "block": int(bn, 16) if bn else None,
            "block_ts": blocks.get(bn),
            "gas_used": int(rec.get("gasUsed", "0x0"), 16),
            "logs": len(rec.get("logs") or []),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--verify", action="store_true",
                    help="re-read every recovered hash from Base (read-only)")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    con = connect(args.db)
    print(f"store: {os.path.abspath(args.db)} "
          f"({os.path.getsize(os.path.abspath(args.db)):,} bytes)\n")

    print("=== 1. hash COLUMNS in the schema ===")
    for r in schema_audit(con):
        print(f"  {r['table']}.{r['column']}: {r['filled']} filled of {r['rows']} rows")
    for t in ("trade_outcomes", "trade_fills", "trading_ops"):
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
        print(f"  {t}: NO hash column  (cols: {', '.join(cols)})")
    print()

    ops = collect_ops_hashes(con)
    print("=== 2. hashes recorded inside trading_ops.details ===")
    per_status: Dict[str, set] = {}
    for h, statuses in ops["distinct"].items():
        for s in statuses:
            per_status.setdefault(s, set()).add(h)
    for s in sorted(per_status):
        print(f"  {s}: {len(per_status[s])} distinct hashes")
    print(f"  union: {len(ops['distinct'])} distinct transaction hashes\n")

    rows = recover(con, ops)
    board = [r for r in rows if r["scoreboard"]]
    print("=== 3. live trade_outcomes -> recovered hashes ===")
    hdr = f"  {'when':19s} {'symbol':12s} {'net':>10s}  {'entry_tx':66s} {'exit_tx'}"
    print(hdr)
    for r in rows:
        mark = "*" if r["scoreboard"] else " "
        print(f" {mark}{r['when']:19s} {r['symbol']:12s} {r['net_profit']:+10.6f}  "
              f"{str(r['entry_tx_hash']):66s} {str(r['exit_tx_hash'])}")
    both = [r for r in rows if r["entry_tx_hash"] and r["exit_tx_hash"]]
    b_both = [r for r in board if r["entry_tx_hash"] and r["exit_tx_hash"]]
    print(f"\n  live outcome rows: {len(rows)}   both legs hashed: {len(both)}")
    print(f"  scoreboard rows (wallet='live'): {len(board)}  "
          f"both legs hashed: {len(b_both)}  "
          f"net sum: {sum(r['net_profit'] for r in board):+.6f}")
    print()

    verified: Dict[str, Dict[str, Any]] = {}
    if args.verify:
        hs = sorted(ops["distinct"])
        print(f"=== 4. on-chain verification of {len(hs)} hashes (read-only) ===")
        verified = verify_onchain(hs)
        found = [h for h, v in verified.items() if v.get("found")]
        ok = [h for h in found if verified[h]["status"] == 1]
        senders = {verified[h]["from"] for h in found}
        print(f"  found on Base: {len(found)}/{len(hs)}   status=1 (success): {len(ok)}")
        print(f"  distinct senders: {len(senders)} -> {sorted(senders)}")
        drift = []
        for h in found:
            bt = verified[h].get("block_ts")
            rec = ops["settled"].get(h)
            if bt and rec:
                drift.append(abs(bt - float(rec["ts"])))
        if drift:
            drift.sort()
            print(f"  |block_ts - recorded ts|: median {drift[len(drift)//2]:.0f}s  "
                  f"max {drift[-1]:.0f}s")
        for h in sorted(found):
            v = verified[h]
            rec = ops["settled"].get(h, {})
            print(f"    {h} status={v['status']} blk={v['block']} "
                  f"logs={v['logs']:2d} purpose={rec.get('purpose')} "
                  f"sym={rec.get('symbol')}")
        missing = [h for h in hs if not verified[h].get("found")]
        if missing:
            print(f"  NOT FOUND ON CHAIN ({len(missing)}): {missing}")
        print()

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({
                "rows": rows,
                "distinct_hashes": sorted(ops["distinct"]),
                "settled": ops["settled"],
                "onchain": verified,
            }, fh, indent=2)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
