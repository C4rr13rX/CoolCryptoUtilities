#!/usr/bin/env python3
"""Identify and re-price (or annul) ghost take-profit overshoots by target ratio.

5504769 stopped the ghost path from booking a limit exit at the tick that
crossed its target (`trading.bot.limit_exit_fill_price`), forward-only. Rows
already recorded in `trade_outcomes` before 5504769 booked the full sampling
gap past their own `target_price`.

CRITICAL INVARIANT (pinned by
`tests/test_an_overshoot_row_is_named_by_its_target_ratio_not_its_size.py`):
  * An overshoot row MUST be named from its own `exit_price / target_price`
    ratio (`exit_price > target_price * (1.0 + fee_rate)` on a limit exit
    `take_profit_limit` / `target_hit`), NEVER from the size of the gain
    (`exit_price / entry_price - 1`).
  * A large real winner whose strategy set a wide target (+25% target, +25.1%
    fill -> ratio 1.0008 <= 1 + fee_rate) is NOT an overshoot and MUST NOT be
    touched.
  * A smaller gain that crossed a tight limit on a sparse tick (+1.0% target,
    +6.0% fill -> ratio 1.0495 > 1 + fee_rate) IS an overshoot and MUST be
    caught.
  * Rows whose `action='enter'` op has no recoverable `target_price` are
    reported as `UNKNOWN` and NEVER touched on a guessed target.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "storage" / "trading_cache.db"
LEDGER_PATH = ROOT / "data" / "strategy_ledger.json"

LIMIT_TAKE_PROFIT_REASONS = ("take_profit_limit", "target_hit")
ALL_TAKE_PROFIT_REASONS = ("take_profit_limit", "target_hit", "time_take_profit")
DEFAULT_FEE_RATE = 0.003187
LOOKBACK_OPS_WINDOW_SEC = 14 * 86400.0


def position_hash(trade_id: Any) -> str:
    """Extract the per-position hash suffix shared by trading_ops and trade_outcomes."""
    raw = str(trade_id or "").strip()
    if not raw:
        return ""
    parts = raw.split(":")
    if parts and parts[-1] in ("exit", "enter") and len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def recover_entry_metadata(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    trade_id: Any,
    outcome_ts: float,
) -> Dict[str, Any]:
    """Recover `target_price` and `strategy_id` from `trading_ops` via position hash.

    Uses the `(symbol, ts)` / `ts` window around `outcome_ts` first so a 700k-row
    `trading_ops` table scans in milliseconds rather than minutes, falling back
    to an unbounded scan only on small test databases or older holdings.
    """
    hsh = position_hash(trade_id)
    if not hsh:
        return {"target_price": None, "strategy_id": None}

    target_price: Optional[float] = None
    strategy_id: Optional[str] = None

    queries = [
        (
            "SELECT action, details FROM trading_ops "
            "WHERE symbol = ? AND ts BETWEEN ? AND ? ORDER BY ts",
            (symbol, float(outcome_ts) - LOOKBACK_OPS_WINDOW_SEC, float(outcome_ts) + 300.0),
        ),
    ]

    for sql, params in queries:
        matched_any = False
        for row in conn.execute(sql, params):
            raw_det = row[1] or ""
            if hsh not in raw_det:
                continue
            matched_any = True
            try:
                det = json.loads(raw_det)
            except (TypeError, ValueError):
                continue
            if not isinstance(det, dict):
                continue
            if not strategy_id and det.get("strategy_id"):
                strategy_id = str(det["strategy_id"])
            if str(row[0] or "") == "enter":
                try:
                    cand_target = float(det.get("target_price") or 0.0)
                except (TypeError, ValueError):
                    cand_target = 0.0
                if cand_target > 0.0:
                    target_price = cand_target
        if target_price is not None or matched_any:
            break

    return {"target_price": target_price, "strategy_id": strategy_id}


def judge_overshoot_row(
    *,
    entry_price: float,
    exit_price: float,
    quantity: float,
    gross_profit: float,
    net_profit: float,
    fee_cost: float,
    reason: str,
    target_price: Optional[float],
    fee_rate: float = DEFAULT_FEE_RATE,
    min_overshoot_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """Judge whether a single closed ghost take-profit row overshot its limit.

    Named strictly from `exit_price / target_price` vs `limit_ratio`, NEVER from
    `exit_price / entry_price` or dollar `gross_profit`.
    """
    base_reason = str(reason or "").split(":")[0].strip()
    is_limit_tp = base_reason in LIMIT_TAKE_PROFIT_REASONS
    threshold_ratio = (
        float(min_overshoot_ratio)
        if min_overshoot_ratio is not None
        else (1.0 + max(0.0, float(fee_rate)))
    )
    return_pct = ((exit_price / entry_price) - 1.0) * 100.0 if entry_price > 0 and exit_price > 0 else None

    if not is_limit_tp:
        return {
            "is_limit_tp": False,
            "known_target": target_price is not None and target_price > 0,
            "is_overshoot": False,
            "verdict": "not_limit_exit",
            "target_price": target_price,
            "limit_fill_price": exit_price,
            "target_ratio": (exit_price / target_price) if (target_price and target_price > 0) else None,
            "threshold_ratio": threshold_ratio,
            "return_pct": return_pct,
            "repriced_gross": float(gross_profit),
            "repriced_net": float(net_profit),
        }

    if target_price is None or not (target_price > 0.0) or not (exit_price > 0.0):
        return {
            "is_limit_tp": True,
            "known_target": False,
            "is_overshoot": False,
            "verdict": "unknown_target",
            "target_price": None,
            "limit_fill_price": exit_price,
            "target_ratio": None,
            "threshold_ratio": threshold_ratio,
            "return_pct": return_pct,
            "repriced_gross": float(gross_profit),
            "repriced_net": float(net_profit),
        }

    target_ratio = float(exit_price) / float(target_price)
    limit_fill_price = float(target_price) * (1.0 + max(0.0, float(fee_rate)))
    is_overshoot = bool(target_ratio > threshold_ratio + 1e-12 and exit_price > limit_fill_price + 1e-12)
    qty = abs(float(quantity or 0.0))
    if is_overshoot and entry_price > 0.0 and qty > 0.0:
        repriced_gross = (limit_fill_price - float(entry_price)) * qty
        repriced_net = repriced_gross - float(fee_cost or 0.0)
        verdict = "overshoot"
    else:
        repriced_gross = float(gross_profit)
        repriced_net = float(net_profit)
        verdict = "at_or_within_limit"

    return {
        "is_limit_tp": True,
        "known_target": True,
        "is_overshoot": is_overshoot,
        "verdict": verdict,
        "target_price": float(target_price),
        "limit_fill_price": float(limit_fill_price),
        "target_ratio": float(target_ratio),
        "threshold_ratio": float(threshold_ratio),
        "return_pct": return_pct,
        "repriced_gross": float(repriced_gross),
        "repriced_net": float(repriced_net),
    }


def scan_overshoot_rows(
    conn: sqlite3.Connection,
    *,
    fee_rate: float = DEFAULT_FEE_RATE,
    min_overshoot_ratio: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Inspect all closed ghost take-profit rows and attach target-ratio verdicts."""
    conn.row_factory = sqlite3.Row
    results: List[Dict[str, Any]] = []

    for row in conn.execute(
        "SELECT * FROM trade_outcomes WHERE status = 'closed' ORDER BY ts"
    ):
        try:
            det = json.loads(row["details"] or "{}")
        except (TypeError, ValueError):
            det = {}
        if not isinstance(det, dict):
            det = {}
        mode = str(det.get("mode") or row["wallet"] or "").lower()
        if mode == "live" or str(row["wallet"] or "").lower() == "live":
            continue
        if det.get("annulled") or det.get("repriced_at_limit"):
            continue

        reason = str(det.get("reason") or "")
        base_reason = reason.split(":")[0].strip()
        if base_reason not in ALL_TAKE_PROFIT_REASONS:
            continue

        explicit_target = None
        try:
            if float(det.get("target_price") or 0.0) > 0.0:
                explicit_target = float(det["target_price"])
        except (TypeError, ValueError):
            explicit_target = None

        meta = recover_entry_metadata(
            conn,
            symbol=str(row["symbol"] or ""),
            trade_id=row["trade_id"],
            outcome_ts=float(row["ts"] or 0.0),
        )
        target_price = explicit_target if explicit_target is not None else meta["target_price"]
        owner = str(det.get("strategy_id") or meta["strategy_id"] or "") or None

        verdict = judge_overshoot_row(
            entry_price=float(row["entry_price"] or 0.0),
            exit_price=float(row["exit_price"] or 0.0),
            quantity=abs(float(row["quantity"] or 0.0)),
            gross_profit=float(row["gross_profit"] or 0.0),
            net_profit=float(row["net_profit"] or 0.0),
            fee_cost=float(row["fee_cost"] or 0.0),
            reason=reason,
            target_price=target_price,
            fee_rate=fee_rate,
            min_overshoot_ratio=min_overshoot_ratio,
        )
        verdict.update(
            {
                "outcome_id": str(row["outcome_id"]),
                "trade_id": str(row["trade_id"] or ""),
                "symbol": str(row["symbol"] or ""),
                "strategy_id": owner,
                "ts": float(row["ts"] or 0.0),
                "entry_price": float(row["entry_price"] or 0.0),
                "exit_price": float(row["exit_price"] or 0.0),
                "quantity": abs(float(row["quantity"] or 0.0)),
                "gross_profit": float(row["gross_profit"] or 0.0),
                "net_profit": float(row["net_profit"] or 0.0),
                "fee_cost": float(row["fee_cost"] or 0.0),
                "reason": reason,
            }
        )
        results.append(verdict)

    return results


def apply_corrections(
    db_path: Path,
    *,
    action: str = "reprice",
    fee_rate: float = DEFAULT_FEE_RATE,
    min_overshoot_ratio: Optional[float] = None,
    ledger_path: Optional[Path] = None,
    backup: bool = True,
) -> Dict[str, Any]:
    """Re-price or annul identified overshoot rows in `trade_outcomes` and sync state."""
    if action not in ("reprice", "annul"):
        raise ValueError(f"unsupported action: {action}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        scanned = scan_overshoot_rows(
            conn, fee_rate=fee_rate, min_overshoot_ratio=min_overshoot_ratio
        )
        targets = [r for r in scanned if r["is_overshoot"]]
        if not targets:
            return {"scanned": len(scanned), "modified": 0, "net_delta": 0.0, "gross_delta": 0.0, "rows": []}

        stamp = time.strftime("%Y%m%d-%H%M%S")
        if backup:
            bck_db_path = db_path.with_name(f"{db_path.name}.bak-overshoot-{stamp}")
            bck_conn = sqlite3.connect(str(bck_db_path))
            try:
                conn.backup(bck_conn)
            finally:
                bck_conn.close()
            if ledger_path and ledger_path.exists():
                shutil.copy2(ledger_path, ledger_path.with_name(f"{ledger_path.name}.bak-overshoot-{stamp}"))

        now_ts = time.time()
        total_net_delta = 0.0
        total_gross_delta = 0.0
        per_strategy_net_delta: Dict[str, float] = {}

        for item in targets:
            oid = item["outcome_id"]
            cur_row = conn.execute(
                "SELECT details FROM trade_outcomes WHERE outcome_id = ?", (oid,)
            ).fetchone()
            try:
                det = json.loads((cur_row["details"] if cur_row else "") or "{}")
            except (TypeError, ValueError):
                det = {}
            if not isinstance(det, dict):
                det = {}

            if action == "annul":
                det["annulled"] = {
                    "ts": now_ts,
                    "reason": "limit_exit_overshot_target_price",
                    "target_price": item["target_price"],
                    "target_ratio": item["target_ratio"],
                    "was_exit_price": item["exit_price"],
                    "was_gross_profit": item["gross_profit"],
                    "was_net_profit": item["net_profit"],
                }
                conn.execute(
                    "UPDATE trade_outcomes SET status = 'annulled', details = ? WHERE outcome_id = ?",
                    (json.dumps(det), oid),
                )
                net_delta = -float(item["net_profit"])
                gross_delta = -float(item["gross_profit"])
            else:
                det["repriced_at_limit"] = {
                    "ts": now_ts,
                    "reason": "limit_exit_overshot_target_price",
                    "target_price": item["target_price"],
                    "limit_fill_price": item["limit_fill_price"],
                    "target_ratio": item["target_ratio"],
                    "was_exit_price": item["exit_price"],
                    "was_gross_profit": item["gross_profit"],
                    "was_net_profit": item["net_profit"],
                }
                conn.execute(
                    "UPDATE trade_outcomes SET exit_price = ?, gross_profit = ?, net_profit = ?, details = ? "
                    "WHERE outcome_id = ?",
                    (
                        float(item["limit_fill_price"]),
                        float(item["repriced_gross"]),
                        float(item["repriced_net"]),
                        json.dumps(det),
                        oid,
                    ),
                )
                net_delta = float(item["repriced_net"]) - float(item["net_profit"])
                gross_delta = float(item["repriced_gross"]) - float(item["gross_profit"])

            total_net_delta += net_delta
            total_gross_delta += gross_delta
            sid = item.get("strategy_id")
            if sid:
                per_strategy_net_delta[sid] = per_strategy_net_delta.get(sid, 0.0) + net_delta

        # Keep kv_store ghost_trading state arithmetically reconciled with trade_outcomes
        try:
            kv_row = conn.execute("SELECT value FROM kv_store WHERE key = 'state'").fetchone()
            if kv_row and kv_row[0]:
                state_obj = json.loads(kv_row[0])
                gt = state_obj.get("ghost_trading") if isinstance(state_obj, dict) else None
                if isinstance(gt, dict):
                    closed_sum = conn.execute(
                        "SELECT COUNT(*), COALESCE(SUM(CASE WHEN net_profit > 0 THEN 1 ELSE 0 END), 0), "
                        "COALESCE(SUM(net_profit), 0.0) "
                        "FROM trade_outcomes WHERE wallet = 'ghost' AND status = 'closed'"
                    ).fetchone()
                    if closed_sum is not None:
                        gt["total_trades"] = int(closed_sum[0] or 0)
                        gt["wins"] = int(closed_sum[1] or 0)
                        gt["total_profit"] = float(closed_sum[2] or 0.0)
                        gt["realized_profit"] = float(closed_sum[2] or 0.0)
                        gt["stable_bank"] = 0.0
                        sqb = gt.get("sim_quote_balances")
                        if isinstance(sqb, dict) and "base:USDC" in sqb:
                            open_exp = sum(
                                float(v or 0.0) for v in (gt.get("active_exposure") or {}).values()
                            )
                            sqb["base:USDC"] = round(25.0 + float(closed_sum[2] or 0.0) - open_exp, 8)
                        conn.execute(
                            "UPDATE kv_store SET value = ?, updated_at = ? WHERE key = 'state'",
                            (json.dumps(state_obj), now_ts),
                        )
        except sqlite3.OperationalError:
            pass

        conn.commit()
    finally:
        conn.close()

    if ledger_path and ledger_path.exists() and per_strategy_net_delta:
        from services.atomic_json import file_lock, read_json, write_json

        with file_lock(ledger_path):
            ldata, ok = read_json(ledger_path, default=None)
            if ok and isinstance(ldata, dict):
                changed = False
                for sid, delta in per_strategy_net_delta.items():
                    entry = ldata.get(sid)
                    if isinstance(entry, dict) and isinstance(entry.get("ghost"), dict):
                        g = entry["ghost"]
                        g["total_profit"] = float(g.get("total_profit", 0.0) or 0.0) + float(delta)
                        changed = True
                if changed:
                    write_json(ledger_path, ldata)

    return {
        "scanned": len(scanned),
        "modified": len(targets),
        "action": action,
        "gross_delta": total_gross_delta,
        "net_delta": total_net_delta,
        "rows": targets,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--ledger", default=str(LEDGER_PATH))
    ap.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    ap.add_argument("--min-overshoot-ratio", type=float, default=None)
    ap.add_argument("--mode", choices=("reprice", "annul"), default="reprice")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()

    db_path = Path(args.db)
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        rows = scan_overshoot_rows(
            conn,
            fee_rate=args.fee_rate,
            min_overshoot_ratio=args.min_overshoot_ratio,
        )
    finally:
        conn.close()

    overshoots = [r for r in rows if r["is_overshoot"]]
    unknown = [r for r in rows if r["verdict"] == "unknown_target"]

    print(f"database: {db_path}")
    print(
        f"take-profit rows scanned: {len(rows)} | "
        f"overshoots (exit/target > threshold): {len(overshoots)} | "
        f"unknown target (untouched): {len(unknown)}"
    )
    for r in sorted(overshoots, key=lambda x: -(x["target_ratio"] or 0.0)):
        print(
            f"  {r['symbol']:14s} sid={str(r['strategy_id']):22s} "
            f"ret={r['return_pct']:+7.2f}% exit/target={r['target_ratio']:.4f} "
            f"gross {r['gross_profit']:+.4f} -> {r['repriced_gross']:+.4f} "
            f"net {r['net_profit']:+.4f} -> {r['repriced_net']:+.4f}"
        )

    if not args.apply:
        print("\ndry run -- pass --apply to write changes")
        return 0

    res = apply_corrections(
        db_path,
        action=args.mode,
        fee_rate=args.fee_rate,
        min_overshoot_ratio=args.min_overshoot_ratio,
        ledger_path=Path(args.ledger) if args.ledger else None,
    )
    print(
        f"\napplied ({res['action']}): {res['modified']} rows updated | "
        f"gross_delta={res['gross_delta']:+.6f} | net_delta={res['net_delta']:+.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
