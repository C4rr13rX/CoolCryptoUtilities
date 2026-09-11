"""Which path bot.py's entries take, and which cost test looked at them.

Produces every number in ``data/directive_path_gate_census.md``. Read-only: it
opens the trading cache with ``mode=ro`` so it cannot touch a running lane.

    python -X utf8 scripts/directive_path_census.py [--days 7]

Two things this deliberately does NOT do:

* It does not count rows. ``trading_ops`` logs ~3.3 rows per tick, so entries
  are deduped on ``(symbol, ts, reason)`` -- or on ``trade_id`` when one is
  present -- before any ratio is quoted.
* It does not count the c0d3rv2 ATF scout's entries as decision-chain entries.
  That lane never runs the chain being measured, and folding its 109 ghost
  entries in would have reported the directive share as 99.2% instead of 98.4%
  for a reason that has nothing to do with the branch ordering.
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.roundtrip_cost import roundtrip_cost_rate  # noqa: E402
from trading.bot import horizon_seconds  # noqa: E402

ENTRY_STATUSES = ("ghost-entry", "live-entry")


def _entries(conn: sqlite3.Connection, cut: float):
    """Deduped entries on bot.py's decision chain, newest last."""
    seen: set = set()
    out = []
    query = (
        "select ts, symbol, status, details from trading_ops "
        "where ts >= ? and status in (?, ?) order by ts"
    )
    for ts, symbol, status, raw in conn.execute(query, (cut, *ENTRY_STATUSES)):
        try:
            details = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if str(details.get("source") or "").startswith("c0d3rv2"):
            continue                      # the scout's own lane, not this chain
        reason = str(details.get("reason") or "")
        trade_id = str(details.get("trade_id") or "")
        key = trade_id or (symbol, round(float(ts), 1), reason[:40])
        if key in seen:
            continue
        seen.add(key)
        out.append((float(ts), symbol, status, details, reason))
    return out


def _path_of(reason: str) -> str:
    if reason == "model-long":
        return "model"
    if reason.startswith("ghost-explore"):
        return "ghost-explore"
    return "directive"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=7.0)
    parser.add_argument(
        "--db", default=str(PROJECT_ROOT / "storage" / "trading_cache.db")
    )
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=120)
    cut = time.time() - args.days * 86400.0
    rows = _entries(conn, cut)

    paths = collections.Counter(
        (status, _path_of(reason)) for _, _, status, _, reason in rows
    )
    print(f"== ENTRIES ON bot.py's DECISION CHAIN, last {args.days:g} days ==")
    print(f"  deduped entries: {len(rows)}")
    for (status, path), n in sorted(paths.items()):
        print(f"    {status:<12} {path:<14} {n}")

    directives = [r for r in rows if _path_of(r[4]) == "directive"]
    if not directives:
        print("  no directive entries in the window")
        return 0

    # -- the move-size conjunct, applied to the directive's own forecast ------
    refused = passed = unpriceable = 0
    expected_returns = []
    vol_refused = vol_absent = 0
    for _, _, _, details, _ in directives:
        plan = details.get("bus_plan") or {}
        expected = plan.get("expected_return")
        size = plan.get("size")
        price = details.get("entry_price") or details.get("price")
        if expected is None or size is None or not price:
            unpriceable += 1
            continue
        expected_returns.append(float(expected))
        notional = float(size) * float(price)
        cost = roundtrip_cost_rate(notional if notional > 0 else 0.75)
        if float(expected) < cost:
            refused += 1
        else:
            passed += 1
        vol = (details.get("brain") or {}).get("volatility_rel")
        if vol is None:
            vol_absent += 1
        elif float(vol) < cost:
            vol_refused += 1

    print("\n== THE MOVE-SIZE CONJUNCT, APPLIED TO THE DIRECTIVE'S OWN FORECAST ==")
    print(f"  priceable directive entries : {refused + passed}")
    print(f"  would be REFUSED            : {refused}")
    print(f"  would pass                  : {passed}")
    print(f"  unpriceable (no bus_plan)   : {unpriceable}")
    if expected_returns:
        expected_returns.sort()
        print(f"  expected_return  median {statistics.median(expected_returns):.6f}"
              f"  min {expected_returns[0]:.6f}  max {expected_returns[-1]:.6f}")
    print(f"  volatility_rel absent on {vol_absent} of {len(directives)};"
          f" below cost on {vol_refused} of the rest")

    # -- which lattice exit each entry that got through took -----------------
    exits = collections.Counter()
    horizons = collections.Counter()
    for _, _, _, details, _ in directives:
        plan = details.get("bus_plan") or {}
        label = str(plan.get("horizon") or "")
        horizons[label] += 1
        expected = float(plan.get("expected_return") or 0.0)
        if expected <= 0:
            exits["expected_return_not_positive"] += 1
        elif horizon_seconds(label) <= 0:
            exits[f"unreadable_horizon:{label}"] += 1
        else:
            exits["reached_the_window_test"] += 1

    print("\n== LATTICE EXIT TAKEN, ON THE ENTRIES THAT GOT THROUGH ==")
    print("  (evaluated with the CURRENT horizon_seconds parser, so an exit")
    print("   listed here is one the shipped fix does not close)")
    for name, n in exits.most_common():
        print(f"    {name:<36} {n}")
    print("  horizons emitted: " + ", ".join(
        f"{k or '(none)'}={v}" for k, v in horizons.most_common()))

    answered = conn.execute(
        "select count(*) from trading_ops "
        "where ts >= ? and status = 'entry-refused-lattice'",
        (cut,),
    ).fetchone()[0]
    layers = collections.Counter()
    for (raw,) in conn.execute(
        "select details from trading_ops "
        "where ts >= ? and status = 'entry-refused-lattice'",
        (cut,),
    ):
        try:
            layers[str(json.loads(raw).get("detail") or "").split(":")[0]] += 1
        except (TypeError, ValueError):
            layers["unparsed"] += 1
    print(f"\n  lattice ANSWERED and refused: {answered}")
    for name, n in layers.most_common():
        print(f"    {name:<36} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
