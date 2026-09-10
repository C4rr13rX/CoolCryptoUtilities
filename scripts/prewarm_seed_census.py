"""Census: which live symbols would be prewarmed from a FOREIGN price series?

Resolves the prewarm candidate exactly the way
``trading/bot.py::_prewarm_buffer_from_history`` resolves it, then compares the
median of that file's last ``--window`` closes against the median of the
freshest live ``market_stream`` ticks, and asks
``services.prewarm_seed_guard.seed_verdict`` for the verdict the bot would
reach.

    python -X utf8 scripts/prewarm_seed_census.py
    python -X utf8 scripts/prewarm_seed_census.py --raw     # no threshold, just the numbers

Exits nonzero when any live symbol would be seeded beyond tolerance, so it can
be used as an acceptance check rather than read by eye.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_db  # noqa: E402
from services.prewarm_seed_guard import (  # noqa: E402
    max_age_days,
    max_log_ratio,
    median_price,
    seed_verdict,
)


def resolve_candidate(root: Path, symbol: str):
    """The same two-step resolution the bot uses: exact symbol, then a loose
    base-symbol glob, newest mtime wins."""
    sym_u = symbol.upper().replace("/", "-")
    candidates = sorted(root.glob(f"*_{sym_u}.json"))
    if not candidates:
        base = sym_u.split("-", 1)[0]
        candidates = sorted(root.glob(f"*_{base}-*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="base")
    ap.add_argument("--window", type=int, default=60, help="bars the bot seeds")
    ap.add_argument("--hours", type=float, default=6.0, help="how recent a symbol must tick to count as live")
    ap.add_argument("--raw", action="store_true", help="print every log ratio, do not gate")
    args = ap.parse_args()

    now = time.time()
    db = get_db()
    pairs = [
        (s, c)
        for (s, c) in db.list_market_pairs_since(now - args.hours * 3600.0)
        if c.lower() == args.chain.lower()
    ]
    root = Path("data/historical_ohlcv") / args.chain.lower()

    rows = []
    for symbol, chain in sorted(pairs):
        live = [p for (p, _ts) in db.recent_market_prices(symbol, chain, limit=25)]
        live_med = median_price(live)
        chosen = resolve_candidate(root, symbol)
        if chosen is None:
            rows.append({"symbol": symbol, "state": "no_prewarm_file", "live_median": live_med})
            continue
        try:
            with chosen.open("r", encoding="utf-8") as fh:
                bars = json.load(fh)
        except Exception as exc:  # pragma: no cover - filesystem accident
            rows.append({"symbol": symbol, "state": f"unreadable ({exc})", "file": chosen.name})
            continue
        if not isinstance(bars, list) or not bars:
            rows.append({"symbol": symbol, "state": "empty_file", "file": chosen.name})
            continue
        tail = bars[-args.window :]
        closes = []
        newest_ts = 0.0
        for b in tail:
            try:
                px = float(b.get("close", 0) or b.get("price", 0))
                if px > 0:
                    closes.append(px)
                newest_ts = max(newest_ts, float(b.get("timestamp", 0) or 0))
            except Exception:
                continue
        seed_med = median_price(closes)
        v = seed_verdict(
            seed_median=seed_med,
            live_median=live_med,
            newest_bar_ts=newest_ts or None,
            now=now,
        )
        rows.append(
            {
                "symbol": symbol,
                "state": ("SEEDS" if v.ok else "REFUSED"),
                "reason": v.reason,
                "file": chosen.name,
                "seed_median": seed_med,
                "live_median": live_med,
                "log_ratio": v.detail.get("log_ratio"),
                "age_days": v.detail.get("age_days"),
            }
        )

    print(f"prewarm seed census -- chain={args.chain} window={args.window} live_within={args.hours}h")
    print(f"  thresholds: |log ratio| <= {max_log_ratio():.4f}, age <= {max_age_days():.2f} days")
    print()
    hdr = f"  {'symbol':<16} {'state':<15} {'log_ratio':>10} {'age_d':>8}  reason / file"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    beyond = 0
    for r in sorted(rows, key=lambda x: (x.get("log_ratio") is None, -abs(x.get("log_ratio") or 0))):
        lr = r.get("log_ratio")
        ad = r.get("age_days")
        lr_s = f"{lr:+10.4f}" if isinstance(lr, float) else " " * 10
        ad_s = f"{ad:8.2f}" if isinstance(ad, float) else " " * 8
        tail = r.get("reason", "") or ""
        if r.get("file"):
            tail = f"{tail}  {r['file']}"
        print(f"  {r['symbol']:<16} {r['state']:<15} {lr_s} {ad_s}  {tail}")
        if r["state"] == "REFUSED":
            beyond += 1

    seeds = sum(1 for r in rows if r["state"] == "SEEDS")
    nofile = sum(1 for r in rows if r["state"] == "no_prewarm_file")
    print()
    print(f"  {len(rows)} live symbols: {seeds} seed, {beyond} REFUSED, {nofile} have no prewarm file")
    if args.raw:
        return 0
    if beyond:
        print(f"  {beyond} symbol(s) would be seeded from a foreign series -- the guard refuses them.")
    # The guard REFUSING is the fixed state: the criterion is that none is
    # SEEDED beyond tolerance, which is what a REFUSED row proves.
    leaked = [
        r
        for r in rows
        if r["state"] == "SEEDS"
        and isinstance(r.get("log_ratio"), float)
        and abs(r["log_ratio"]) > max_log_ratio()
    ]
    if leaked:
        print("  FAIL: seeded beyond tolerance -> " + ", ".join(r["symbol"] for r in leaked))
        return 1
    print("  OK: 0 symbols seeded beyond tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
