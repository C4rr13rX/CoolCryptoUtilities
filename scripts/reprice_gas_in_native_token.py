"""Re-price the gas on five settled live round trips: ETH, not the pair.

THE DEFECT. ``TradingBot._estimate_native_price`` seeded its answer from
``price`` -- the traded pair's price -- and then tried to overwrite it from the
price book. The overwrite could never happen:

  * ``db.fetch_price`` returns a ``sqlite3.Row``, which has no ``.get``, so
    ``row.get("usd")`` raised AttributeError straight into a bare
    ``except Exception: pass``; and
  * every row in ``prices`` is stored under chain ``global`` (37 rows,
    measured 2026-09-04), so ``fetch_price("base", "ETH")`` returns None
    regardless of how it is read.

So ``fee_cost = total_gas_native_realized * native_price_usd`` valued ETH gas
at the price of whatever was being traded.

THE PROOF IS IN THE ROWS THEMSELVES. Dividing each recorded ``fee_cost`` by the
gas actually burned -- read from the receipts with ``eth_getTransactionReceipt``,
``gasUsed * effectiveGasPrice`` -- recovers the price the fee was charged at:

    AERO-USDC   fee 8.430420735204225e-07 / 1.728577435e-06 ETH = $0.4877
    CBETH-USDC  fee 0.005209389852366538  / 1.836843733e-06 ETH = $2836.06
    CBETH-USDC  fee 0.003461948410600778  / 1.218798508e-06 ETH = $2840.46
    AERO-USDC   fee 7.775206287420001e-07 / 1.54946319e-06  ETH = $0.5018
    CBBTC-USDC  fee 0.4135511037032909    / 5.112829501e-06 ETH = $80884.98

Every one is that pair's own price. None is ETH's ($2498.78, chain ``global``,
source ``consensus``). The worst case charged $0.4136 of gas on a $3.00
position -- 13.8% -- and booked a -$0.41765310 loss where the true figure is
-$0.01687782. That single fiction was 38x the sum of every live win, which the
standing orders name as the signature of a broken mechanism.

THE RECEIPTS. All five transactions returned status 0x1. Gas, in full:

  0x62cafa4ccb0c7a34c7c8a767ac569832f69601f3fb71d396b4813f104acfb0c2  AERO in
  0x77f0075e0e6b79e71ba63667efe4aec9a80bb2b6e858400f63305d90ec7a6ed1  AERO out
  0x076978740803789cd40564cb150753bc075a5f6600d8422add58a4720822b82b  cbETH in
  0x927834717d12395c1eb3d9148609a2b8142a59403205d1caa4ffd04e68e0e005  cbETH out
  0xee36a71d5b0ff542a66a33782085f33df52252892b1c18b7c92538b4bb89c388  cbETH in
  0x9ffdd1cfe3f17fbeddb2b3091b49f8f6524b5a0538ae9a2f3aadb56610264e6c  cbETH out
  0xfce0036c5c9e2e73a5985a5cfb849ed745e55a7b07c2c27bca00cca85d0888b5  AERO in
  0x121349488665874b2259c283d268cda7220663d11b146d2651983a0566316bb5  AERO out
  0x53f7303248255d6923b42c5920a58413c8b4c85b57cb54df01be0e14899b07ad  cbBTC in
  0x3c992f2709bb54a39dc59d832aa3cc6c4dd0f006c40579f36e3e69cd6f295666  cbBTC in
  0xd9f7c07aba10d585a02426c23e771e73b3babc8b927fdd3cf8d230103f826d3e  cbBTC in
  0x0dfedf3a7c4f77a35706ddca6813f9ec93455c0a5d7e5d3c7b9d2760ab8f0959  cbBTC in
  0x340034967fb6145db692f039f7a95108295249aa2f0f2f987d17aa69e061705e  cbBTC out

The cbBTC exit is the adopted-orphan position: it sold the tokens of all four
buys at once, so it carries all four entries' gas. The two cbETH round trips
were partial exits, so each carries its entry's gas x (quantity sold / entry
size) -- 0.591780 and 0.421056 -- exactly as ``allocation_ratio`` computed it.

THIS IS NOT A FLATTERING CORRECTION. Gas was UNDER-charged on both AERO round
trips (valued at $0.49 instead of $2498.78), so the first AERO trade turns from
a +0.0012394 win into a -0.0030791 loss. The live book goes from 2W/3L to
1W/4L. What it stops being is fictional:

    was   5 trades  2W/3L  net -0.41354380  worst -0.41765310
    now   5 trades  1W/4L  net -0.01992203  worst -0.01687782

Profit factor 0.3429. The live gate still refuses -- correctly, and now on
numbers the chain agrees with. This script does not open any gate.

WHAT ELSE THIS MEASURES. Gas on base is about 1.7e-06 ETH per round trip,
roughly $0.0043. On a $0.75 clip that is 0.57%, on top of the 0.65% fee gate:
a $0.75 round trip must move more than 1.2% just to break even. That cost floor
is real and it is why both AERO round trips lose.

BOTH BOOKS. ``data/strategy_ledger.json`` gates graduation and
``data/strategy_registry.json`` is what link 10 reads; a correction that
reaches only one of them is how -0.1405 survived for seven hours after the
chain said +0.0052 (commit fbe24d2). Both are replayed here, in order, because
peak_profit / max_drawdown / consecutive_losses are path-dependent and cannot
be obtained by subtracting a delta.

Run:  .venv/Scripts/python.exe -X utf8 scripts/reprice_gas_in_native_token.py [--apply]

STOP PRODUCTION FIRST. A running bot holds these books open and will clobber
the rewrite.
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

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "storage" / "trading_cache.db"
REGISTRY_PATH = ROOT / "data" / "strategy_registry.json"
STRATEGY_ID = "atf_static"

#: chain=global, token=eth, source=consensus. The only ETH price the system
#: holds; ~5h stale at the time of the correction, which moves a $0.0043 fee by
#: well under a hundredth of a cent.
ETH_USD = 2498.77748000803

REASON = (
    "gas is paid in ETH but fee_cost valued it at the traded pair's price: "
    "_estimate_native_price seeded price_candidate from `price` and its lookup "
    "could never overwrite it (sqlite3.Row has no .get, and prices are stored "
    "only under chain 'global'). Re-priced at ETH $%.8f using gas read from "
    "the transaction receipts (gasUsed * effectiveGasPrice)." % ETH_USD
)
EVIDENCE = "eth_getTransactionReceipt, base, 2026-09-04: 13 receipts, all status 0x1"

#: trade_id -> (gas ETH actually attributed to this outcome, gross_profit,
#: fee_cost and net_profit as recorded). The gas figures are the receipts'
#: own numbers, scaled by the allocation_ratio the exit used.
REPRICE = {
    "2:AERO-USDC:aec1a3366848462180392dcbb7abe891": {
        "gas_eth": 1.728577435e-06,
        "was_gross": 0.0012402430184282887,
        "was_fee": 8.430420735204225e-07,
        "was_net": 0.0012393999763547683,
    },
    "2:CBETH-USDC:bfa397e608c743e7a18c1f4450590080": {
        "gas_eth": 1.836843733e-06,
        "was_gross": 0.014984153020625668,
        "was_fee": 0.005209389852366538,
        "was_net": 0.009774763168259131,
    },
    "2:CBETH-USDC:34c513c44ff64ee581fe8d673f2d867a": {
        "gas_eth": 1.218798508e-06,
        "was_gross": -0.0023971348768693224,
        "was_fee": 0.003461948410600778,
        "was_net": -0.005859083287470101,
    },
    "2:AERO-USDC:2269251a5116425b8ce9745cdf189dc8": {
        "gas_eth": 1.54946319e-06,
        "was_gross": -0.0010449999999999626,
        "was_fee": 7.775206287420001e-07,
        "was_net": -0.0010457775206287045,
    },
    "2:CBBTC-USDC:f235437c56b646618cbf19ab5ba152f6": {
        "gas_eth": 5.112829501e-06,
        "was_gross": -0.00410200000000005,
        "was_fee": 0.4135511037032909,
        "was_net": -0.41765310370329095,
    },
}

#: What both books must read BEFORE this runs. If either has moved, something
#: has traded and the record must be re-measured, not rewritten.
EXPECTED_LIVE = {
    "trades": 5,
    "wins": 2,
    "losses": 3,
    "total_profit": -0.4135438013667759,
}

#: Every recorded fee divided by the gas actually burned must recover that
#: pair's own price -- the defect's signature. A row that does not show it is
#: not one of the rows this was measured on.
SIGNATURE_RANGES = {
    "AERO-USDC": (0.4, 0.6),
    "CBETH-USDC": (2700.0, 3000.0),
    "CBBTC-USDC": (78000.0, 84000.0),
}


def _replay(rows: list) -> dict:
    """Path-dependent live stats, recomputed in time order. Never subtracted."""
    run = peak = mdd = 0.0
    wins = losses = streak = max_streak = 0
    gross_win = gross_loss = 0.0
    best, worst = None, None
    symbols: dict = {}
    for row in rows:
        net = row["net"]
        run += net
        peak = max(peak, run)
        mdd = max(mdd, peak - run)
        if net > 0:
            wins += 1
            gross_win += net
            streak = 0
        else:
            losses += 1
            gross_loss += -net
            streak += 1
            max_streak = max(max_streak, streak)
        best = net if best is None else max(best, net)
        worst = net if worst is None else min(worst, net)
        symbols[row["symbol"]] = symbols.get(row["symbol"], 0) + 1
    return {
        "trades": len(rows),
        "wins": wins,
        "losses": losses,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "total_profit": run,
        "best": best,
        "worst": worst,
        "peak_profit": peak,
        "max_drawdown": mdd,
        "consecutive_losses": streak,
        "max_consecutive_losses": max_streak,
        "symbols": symbols,
    }


def _load(conn: sqlite3.Connection) -> list:
    conn.row_factory = sqlite3.Row
    found = []
    for row in conn.execute("SELECT * FROM trade_outcomes ORDER BY ts"):
        det = json.loads(row["details"] or "{}")
        if det.get("mode") != "live" or row["status"] != "closed":
            continue
        found.append(dict(row))
    if len(found) != len(REPRICE):
        raise SystemExit(
            f"expected {len(REPRICE)} closed live outcomes, found {len(found)}. "
            "The set has changed since it was measured against the chain -- "
            "re-measure before rewriting anything."
        )
    for row in found:
        tid = row["trade_id"]
        want = REPRICE.get(tid)
        if want is None:
            raise SystemExit(f"{tid} was not measured against the chain; refusing")
        for field, key in (("gross_profit", "was_gross"), ("fee_cost", "was_fee"),
                           ("net_profit", "was_net")):
            if abs(float(row[field]) - want[key]) > 1e-15:
                raise SystemExit(
                    f"{tid}.{field} is {row[field]!r}, measured {want[key]!r}; refusing"
                )
        implied = want["was_fee"] / want["gas_eth"]
        lo, hi = SIGNATURE_RANGES[row["symbol"]]
        if not lo <= implied <= hi:
            raise SystemExit(
                f"{tid}: fee/gas implies ${implied:.4f}, which is not "
                f"{row['symbol']}'s price range {lo}-{hi}; this row does not "
                "carry the defect's signature and must not be repriced."
            )
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-price live gas in ETH.")
    ap.add_argument("--apply", action="store_true", help="write the change (default: dry run)")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    rows = _load(conn)

    print(f"database : {DB_PATH}")
    print(f"ETH price: ${ETH_USD} (chain=global, source=consensus)\n")

    replay_rows = []
    for row in rows:
        want = REPRICE[row["trade_id"]]
        fee = want["gas_eth"] * ETH_USD
        net = float(row["gross_profit"]) - fee
        implied = want["was_fee"] / want["gas_eth"]
        print(
            "  %s %-12s gas %.9g ETH\n"
            "      fee %.9f (charged at $%.4f) -> %.9f (at $%.4f)\n"
            "      net %+.9f -> %+.9f"
            % (time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(row["ts"])), row["symbol"],
               want["gas_eth"], want["was_fee"], implied, fee, ETH_USD,
               float(row["net_profit"]), net)
        )
        replay_rows.append({"symbol": row["symbol"], "net": net, "ts": float(row["ts"]),
                            "fee": fee, "outcome_id": row["outcome_id"],
                            "trade_id": row["trade_id"]})

    corrected = _replay(replay_rows)
    corrected["first_ts"] = min(r["ts"] for r in replay_rows)
    corrected["last_ts"] = max(r["ts"] for r in replay_rows)

    print("\n  was : %d trades %dW/%dL  net %+.9f  worst %+.9f"
          % (EXPECTED_LIVE["trades"], EXPECTED_LIVE["wins"], EXPECTED_LIVE["losses"],
             EXPECTED_LIVE["total_profit"], REPRICE["2:CBBTC-USDC:f235437c56b646618cbf19ab5ba152f6"]["was_net"]))
    print("  now : %d trades %dW/%dL  net %+.9f  worst %+.9f"
          % (corrected["trades"], corrected["wins"], corrected["losses"],
             corrected["total_profit"], corrected["worst"]))
    print("  profit factor: %.4f  (still a losing book; this opens no gate)"
          % (corrected["gross_win"] / max(corrected["gross_loss"], 1e-12)))

    # --- the ledger (gates graduation) --------------------------------------
    ledger_path = StrategyLedger.DEFAULT_PATH
    reg_data, reg_ok = read_json(REGISTRY_PATH, default=None)
    if not reg_ok or not isinstance(reg_data, dict):
        print(f"cannot read registry at {REGISTRY_PATH}", file=sys.stderr)
        return 1
    reg_entry = (reg_data.get("strategies") or {}).get(STRATEGY_ID)
    if not isinstance(reg_entry, dict):
        print(f"{STRATEGY_ID} is not in the registry", file=sys.stderr)
        return 1

    with file_lock(ledger_path):
        led_data, led_ok = read_json(ledger_path, default=None)
        if not led_ok or not isinstance(led_data, dict):
            print(f"cannot read ledger at {ledger_path}", file=sys.stderr)
            return 1
        led_entry = led_data.get(STRATEGY_ID)
        if not isinstance(led_entry, dict):
            print(f"{STRATEGY_ID} is not in the ledger", file=sys.stderr)
            return 1

        problems = []
        for label, live in (("ledger", led_entry.get("live") or {}),
                            ("registry", (reg_entry.get("lifetime") or {}).get("live") or {})):
            for key, want in EXPECTED_LIVE.items():
                got = live.get(key)
                if isinstance(want, float):
                    if got is None or abs(float(got) - want) > 1e-12:
                        problems.append(f"{label}.live.{key} is {got!r}, expected {want!r}")
                elif int(got or 0) != want:
                    problems.append(f"{label}.live.{key} is {got!r}, expected {want!r}")
        if problems:
            print("\nREFUSING: the live books are not the ones measured against the chain:")
            for p in problems:
                print(f"  - {p}")
            print("Something has traded since. Stop production and re-measure.")
            return 2

        if not args.apply:
            print("\ndry run -- re-run with --apply (stop production first)")
            return 0

        stamp = time.strftime("%Y%m%d-%H%M%S")
        for src in (DB_PATH, ledger_path, REGISTRY_PATH):
            dst = src.with_name(src.name + f".bak-gasreprice-{stamp}")
            shutil.copy2(src, dst)
            print(f"backed up {src.name} -> {dst.name}")

        for r in replay_rows:
            row = next(x for x in rows if x["outcome_id"] == r["outcome_id"])
            det = json.loads(row["details"] or "{}")
            det["repriced"] = {
                "ts": time.time(),
                "reason": REASON,
                "evidence": EVIDENCE,
                "gas_eth": REPRICE[r["trade_id"]]["gas_eth"],
                "eth_usd": ETH_USD,
                "was_fee_cost": REPRICE[r["trade_id"]]["was_fee"],
                "was_net_profit": REPRICE[r["trade_id"]]["was_net"],
            }
            conn.execute(
                "UPDATE trade_outcomes SET fee_cost=?, net_profit=?, details=? WHERE outcome_id=?",
                (r["fee"], r["net"], json.dumps(det), r["outcome_id"]),
            )
        conn.commit()
        print(f"repriced {len(replay_rows)} outcome rows")

        correction = {
            "ts": time.time(),
            "mode": "live",
            "what": "re-priced gas in ETH instead of in the traded pair",
            "reason": REASON,
            "evidence": EVIDENCE,
            "was": dict(EXPECTED_LIVE),
            "now": {k: corrected[k] for k in ("trades", "wins", "losses", "total_profit")},
        }

        live = dict(led_entry.get("live") or {})
        for key in ("trades", "wins", "losses", "total_profit", "peak_profit",
                    "max_drawdown", "consecutive_losses"):
            live[key] = corrected[key]
        live["last_ts"] = corrected["last_ts"]
        led_entry["live"] = live
        led_entry.setdefault("corrections", []).append(correction)
        if not write_json(ledger_path, led_data):
            print("ledger write failed", file=sys.stderr)
            return 1
        print(f"rewrote {ledger_path.name}")

    reg_live = dict((reg_entry.get("lifetime") or {}).get("live") or {})
    for key in ("trades", "wins", "losses", "gross_win", "gross_loss", "total_profit",
                "best", "worst", "peak_profit", "max_drawdown", "consecutive_losses",
                "max_consecutive_losses"):
        reg_live[key] = corrected[key]
    reg_live["last_ts"] = corrected["last_ts"]
    reg_entry.setdefault("lifetime", {})["live"] = reg_live
    reg_entry.setdefault("corrections", []).append(correction)
    with file_lock(REGISTRY_PATH):
        if not write_json(REGISTRY_PATH, reg_data):
            print("registry write failed", file=sys.stderr)
            return 1
    print(f"rewrote {REGISTRY_PATH.name}")

    ledger = StrategyLedger()
    print(f"\nledger now  : live={ledger.stats(STRATEGY_ID).get('live')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
