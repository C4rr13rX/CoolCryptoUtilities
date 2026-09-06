"""Drive and inspect the SmartStorageService from the command line.

    python scripts/smart_storage_ctl.py status
    python scripts/smart_storage_ctl.py sync --domain news --dry-run
    python scripts/smart_storage_ctl.py sync --domain news --limit 500
    python scripts/smart_storage_ctl.py reclaim --domain market --apply
    python scripts/smart_storage_ctl.py plan

``status`` and ``plan`` are read-only and safe to run at any time. ``sync``
uploads; ``reclaim`` deletes local copies and therefore defaults to a dry run
that shows what would go before anything goes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.smart_storage import get_storage  # noqa: E402


def _fmt_mb(value: float) -> str:
    return f"{value:,.1f} MB" if value < 1024 else f"{value / 1024:,.2f} GB"


def cmd_status(_args) -> int:
    svc = get_storage()
    stats = svc.stats()
    print("SMART STORAGE")
    print(f"  remote      : {'ON  -> ' + str(stats['bucket']) if stats['remote_enabled'] else 'OFF (local only)'}")
    print(f"  region      : {stats['region']}")
    print(f"  local budget: {stats['local_budget_mb']} MB")
    print(f"  hot window  : {stats['hot_window_bars']} bars/symbol")
    print()
    print("LOCAL FOOTPRINT")
    total = 0.0
    for domain in svc.roots:
        size = svc.local_footprint_mb(domain)
        total += size
        print(f"  {domain:14s} {_fmt_mb(size):>12s}   {svc.roots[domain]}")
    print(f"  {'TOTAL':14s} {_fmt_mb(total):>12s}")
    print()
    print("COUNTERS")
    for key in ("local_hits", "remote_hits", "misses", "uploads",
                "downloads", "evictions", "errors"):
        print(f"  {key:14s} {stats[key]}")
    if not stats["remote_enabled"]:
        print()
        print("  Remote is off. Nothing is ever evicted in this mode: every")
        print("  local file is the only copy. See .env.smartstorage.example.")
    return 0


def cmd_sync(args) -> int:
    svc = get_storage()
    if not svc.config.remote_enabled:
        print("remote is disabled -- set MARKET_DATA_S3_ENABLED=1 and a bucket")
        return 1
    result = svc.sync_up(args.domain, limit=args.limit, dry_run=args.dry_run)
    verb = "would upload" if args.dry_run else "uploaded"
    print(f"{verb} {result['uploaded']} objects "
          f"({_fmt_mb(result['bytes'] / 1e6)}), "
          f"skipped {result['skipped']} already durable, "
          f"{result['errors']} errors")
    if result.get("reason"):
        print(f"  {result['reason']}")
    return 0


def cmd_reclaim(args) -> int:
    svc = get_storage()
    result = svc.reclaim(args.domain, dry_run=not args.apply)
    if result.get("reason"):
        print(result["reason"])
        return 0
    verb = "evicted" if args.apply else "would evict"
    print(f"{verb} {len(result['evicted'])} of {result['candidates']} candidates, "
          f"freeing {_fmt_mb(result['freed_mb'])}")
    if result["skipped_not_durable"]:
        print(f"  {result['skipped_not_durable']} skipped: not confirmed in S3 "
              f"(they stay local -- a cache must never delete the only copy)")
    return 0


def cmd_plan(_args) -> int:
    """What the placement policy would do, and why, without doing it."""
    svc = get_storage()
    print("PLACEMENT PLAN")
    print()
    print("Q1  what must be local for zero latency?")
    print(f"    the hot set: {svc.config.hot_window_bars} bars per active symbol.")
    print("    Historical bars are immutable, so only the tail is ever re-read.")
    print()
    print("Q2  what is the cheapest way to hold the rest?")
    print("    S3 with segment-per-symbol-per-period objects. Per-REQUEST cost")
    print("    dominates for small objects: bar-by-bar reads would be ~449k")
    print("    GETs/day; one object per symbol-day is ~780 GETs/month.")
    print("    Ingress is free, so syncing up costs nothing but storage.")
    print()
    print("Q3  how small can local go without risking data?")
    print("    Eviction is permitted only for keys confirmed present in S3,")
    print("    so the worst case is a slow read, never a missing one.")
    print()
    total = sum(svc.local_footprint_mb(d) for d in svc.roots)
    print(f"    local now: {_fmt_mb(total)}   budget: {svc.config.local_budget_mb} MB")
    if not svc.config.remote_enabled:
        print("    remote OFF -- nothing would be evicted.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="where data lives right now").set_defaults(func=cmd_status)
    sub.add_parser("plan", help="what the policy would do, and why").set_defaults(func=cmd_plan)

    p_sync = sub.add_parser("sync", help="upload local objects not yet in S3")
    p_sync.add_argument("--domain", default="news")
    p_sync.add_argument("--limit", type=int, default=None)
    p_sync.add_argument("--dry-run", action="store_true")
    p_sync.set_defaults(func=cmd_sync)

    p_rec = sub.add_parser("reclaim", help="evict cold local copies that are safe in S3")
    p_rec.add_argument("--domain", default="market")
    p_rec.add_argument("--apply", action="store_true",
                       help="actually delete (default is a dry run)")
    p_rec.set_defaults(func=cmd_reclaim)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
