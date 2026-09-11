"""Strike four exits that the chain says never happened.

On 2026-09-03 four positions were "closed" by simulation while holding real
tokens. ``_interpret_predictions`` computes::

    pos_is_live = pos["mode"] == "live" and self.live_trading_enabled

so when the bot-level flag went false, a position opened with real money fell
through to the simulated branch and was marked out against the feed price: no
swap, no tx hash, no proceeds. The row was written with ``wallet="ghost"`` but
``StrategyLedger.record`` was called with ``mode=pos_mode`` -- i.e. **live** --
so a fiction landed in the book that gates real money. trading/bot.py now
refuses that path (``live_position_cannot_exit_in_simulation``); this repairs
the four records it already wrote.

EVIDENCE. eth_getLogs over every ERC-20 Transfer touching
0x291c854811e92906a658Fb94Aa511bF919f968ad for the day. Fifteen swaps settled;
each of the four exits below has no settling transfer in that set:

    17:40:03  CBETH-USDC   -0.000005   sold 0.00000023   no transfer
    19:12:15  BSTONK-USDC  -0.142865   sold 360.264243   no transfer; the
              wallet still holds all 360.2642432254 BSTONK, bought for 0.75
              USDC at 17:02:15 in
              0xcd6fb05c92af5077f9be707727c1d57e0ac1dfedd54d9f87e860376b96ea560b
              and never sold
    20:03:27  CBBTC-USDC   +0.000267   sold 0.00000928   no transfer; the
              wallet holds all 0.0000370900 cbBTC from four buys
    20:38:53  CBETH-USDC   -0.003011   sold 0.00015239   no transfer, and the
              wallet had held only 0.0000003733 cbETH since 16:46 -- this exit
              sold tokens that did not exist

The three cbETH sells that DID settle are, in full:

    0x927834717d12395c1eb3d9148609a2b8142a59403205d1caa4ffd04e68e0e005
    0x166c461a1f6b75da3ad77e8a4fd78080cc0dcb35391b11187c38950e35a377a0
    0x9ffdd1cfe3f17fbeddb2b3091b49f8f6524b5a0538ae9a2f3aadb56610264e6c

and the one AERO sell:

    0x77f0075e0e6b79e71ba63667efe4aec9a80bb2b6e858400f63305d90ec7a6ed1

WHAT THIS CHANGES. atf_static's live book goes from 7 trades / 3W-4L /
-0.14045951 to the three outcomes that actually settled on chain: 2W / 1L /
+0.00515508. The BSTONK line alone was -0.142865 against +0.011281 of wins --
102% of the live P/L, and the tail_risk 0.1429 the live gate refuses on.

WHAT THIS DOES NOT DO. It does not add anything. One real settled sell --
0x166c461a1f6b75da3ad77e8a4fd78080cc0dcb35391b11187c38950e35a377a0, 0.000264
cbETH for 0.745515 USDC at 16:42:23 -- has no outcome row at all and is NOT
inserted here. Inventing an entry basis for it would fabricate a record, and
leaving a likely WIN out is the conservative direction. It is reported instead.

The database rows are not deleted; ``status`` becomes ``annulled`` (which
``trade_outcome_summary``'s ``status='closed'`` filter already excludes) and
``details.annulled`` carries the reason, so the audit trail survives.

Run:  .venv/Scripts/python.exe -X utf8 scripts/annul_unsettled_live_exits.py [--apply]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.atomic_json import file_lock, read_json, write_json  # noqa: E402
from trading.strategies.ledger import StrategyLedger  # noqa: E402

DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "trading_cache.db"
STRATEGY_ID = "atf_static"

ANNUL_REASON = (
    "no settling ERC-20 Transfer on base for this quantity; the position was "
    "marked out against the feed price while live execution was disarmed "
    "(bot.py: live_position_cannot_exit_in_simulation)"
)

#: The defect's exact signature, and the reason this is a query rather than a
#: hardcoded list of ids: a row written by the simulated branch carries
#: ``wallet='ghost'`` (because ``pos_is_live`` was false) while
#: ``details.mode`` carries ``'live'`` (because ``pos_mode`` was live). No
#: correct path can produce that pair -- the two are computed from the same
#: ``pos_mode`` and diverge only when ``live_trading_enabled`` is false.
#:
#: The set it selects is checked against the numbers measured from the chain
#: below, so a row that is not one of the four measured fictions cannot be
#: annulled by accident.
EXPECTED_UNSETTLED = 4
EXPECTED_STRUCK_NET = -0.14561459047918154

#: What atf_static's live book must read before this runs. If it has moved on,
#: something has traded and the record must be re-judged, not rewritten.
EXPECTED_LIVE = {"trades": 7, "wins": 3, "losses": 4, "total_profit": -0.14045951062203274}

#: ...and what it must read afterwards: the three on-chain-settled outcomes.
CORRECTED_LIVE = {
    "trades": 3,
    "wins": 2,
    "losses": 1,
    "total_profit": 0.005155079857143799,
    # +0.0012394, then +0.0097748 (peak +0.0110142), then -0.0058591.
    "peak_profit": 0.0110141631446139,
    "max_drawdown": 0.005859083287470101,
    # The last settled outcome (16:38:15 CBETH) was a loss.
    "consecutive_losses": 1,
}


def _find_rows(conn: sqlite3.Connection) -> list:
    """Every outcome booked live but executed in simulation."""
    conn.row_factory = sqlite3.Row
    found = []
    for row in conn.execute("SELECT * FROM trade_outcomes ORDER BY ts"):
        det = json.loads(row["details"] or "{}")
        if str(det.get("mode")) == "live" and str(row["wallet"]) == "ghost":
            found.append(dict(row))
    if len(found) != EXPECTED_UNSETTLED:
        raise SystemExit(
            f"expected {EXPECTED_UNSETTLED} rows with wallet=ghost + details.mode=live, "
            f"found {len(found)}. The set has changed since it was measured against "
            "the chain -- re-measure before rewriting anything."
        )
    struck = sum(float(r["net_profit"]) for r in found if str(r["status"]) == "closed")
    if abs(struck - EXPECTED_STRUCK_NET) > 1e-12:
        raise SystemExit(
            f"the four rows sum to {struck!r}, expected {EXPECTED_STRUCK_NET!r}; "
            "these are not the outcomes that were checked against the chain."
        )
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write the change (default: dry run)")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    rows = _find_rows(conn)

    print(f"database : {DB_PATH}")
    print("rows the chain does not support:")
    struck = 0.0
    for r in rows:
        det = json.loads(r["details"] or "{}")
        print(
            "  %s  %-13s wallet=%-5s details.mode=%-5s net=%+.9f qty=%.12f status=%s"
            % (
                time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(r["ts"])),
                r["symbol"], r["wallet"], det.get("mode"), r["net_profit"],
                r["quantity"], r["status"],
            )
        )
        if str(det.get("mode")) != "live":
            raise SystemExit(f"{r['outcome_id']} is not a live-booked outcome; refusing")
        if str(r["status"]) != "closed":
            print(f"    already {r['status']} -- nothing to do for this row")
            continue
        struck += float(r["net_profit"])
    print(f"\nnet P/L these four fictions contributed: {struck:+.9f}")

    path = StrategyLedger.DEFAULT_PATH
    with file_lock(path):
        data, ok = read_json(path, default=None)
        if not ok or not isinstance(data, dict):
            print(f"cannot read ledger at {path}", file=sys.stderr)
            return 1
        ent = data.get(STRATEGY_ID)
        if not isinstance(ent, dict):
            print(f"{STRATEGY_ID} is not in the ledger", file=sys.stderr)
            return 1
        live = ent.get("live") or {}

        print(f"\nledger   : {path}")
        print(
            "  live now : %d trades %dW/%dL  net %+.9f"
            % (live.get("trades", 0), live.get("wins", 0), live.get("losses", 0),
               live.get("total_profit", 0.0))
        )
        problems = []
        for key, want in EXPECTED_LIVE.items():
            got = live.get(key)
            if isinstance(want, float):
                if got is None or abs(float(got) - want) > 1e-12:
                    problems.append(f"live.{key} is {got!r}, expected {want!r}")
            elif int(got or 0) != want:
                problems.append(f"live.{key} is {got!r}, expected {want!r}")
        if problems:
            print("\nREFUSING: atf_static's live book is not the one measured against the chain:")
            for p in problems:
                print(f"  - {p}")
            print("Something has traded since. Re-measure against the chain before rewriting.")
            return 2

        # The struck rows must account for exactly the difference.
        delta = EXPECTED_LIVE["total_profit"] - CORRECTED_LIVE["total_profit"]
        if abs(delta - struck) > 1e-12:
            print(
                f"\nREFUSING: the four rows sum to {struck!r} but the ledger "
                f"difference is {delta!r}; they do not describe the same set.",
                file=sys.stderr,
            )
            return 2
        print(
            "  live after: %d trades %dW/%dL  net %+.9f"
            % (CORRECTED_LIVE["trades"], CORRECTED_LIVE["wins"], CORRECTED_LIVE["losses"],
               CORRECTED_LIVE["total_profit"])
        )
        gross_win = 0.0012394 + 0.0097748
        gross_loss = 0.0058591
        print(f"  live profit factor after: {gross_win / gross_loss:.4f}")

        if not args.apply:
            print("\ndry run -- re-run with --apply")
            return 0

        backup_db = DB_PATH.with_name(
            DB_PATH.name + f".bak-unsettled-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        shutil.copy2(DB_PATH, backup_db)
        print(f"\nbacked up db     : {backup_db}")
        backup_ledger = path.with_suffix(
            f".json.bak-unsettled-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        shutil.copy2(path, backup_ledger)
        print(f"backed up ledger : {backup_ledger}")

        for r in rows:
            if str(r["status"]) != "closed":
                continue
            det = json.loads(r["details"] or "{}")
            det["annulled"] = {
                "ts": time.time(),
                "reason": ANNUL_REASON,
                "evidence": "eth_getLogs, base, 2026-09-03: no Transfer of this quantity",
                "was_status": r["status"],
                "was_net_profit": float(r["net_profit"]),
            }
            conn.execute(
                "UPDATE trade_outcomes SET status=?, details=? WHERE outcome_id=?",
                ("annulled", json.dumps(det), r["outcome_id"]),
            )
        conn.commit()
        print(f"annulled {sum(1 for r in rows if str(r['status']) == 'closed')} outcome rows")

        live.update(CORRECTED_LIVE)
        ent["live"] = live
        ent.setdefault("corrections", []).append(
            {
                "ts": time.time(),
                "what": "removed 4 live outcomes with no settling on-chain transfer",
                "removed_net": struck,
                "reason": ANNUL_REASON,
                "not_added": (
                    "the settled sell 0x166c461a1f6b75da3ad77e8a4fd78080cc0dcb35391b11187c38950e35a377a0 "
                    "(0.000264 cbETH -> 0.745515 USDC, 16:42:23Z) has no outcome row and was NOT "
                    "inserted; inventing its entry basis would fabricate a record"
                ),
            }
        )
        if not write_json(path, data):
            print("ledger write failed", file=sys.stderr)
            return 1

    ledger = StrategyLedger()
    stats = ledger.stats(STRATEGY_ID)
    print(f"\nledger now: live={stats.get('live')}")
    print(f"live_approved={stats.get('live_approved')}  approved_ids={ledger.approved_ids()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
