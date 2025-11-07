#!/usr/bin/env python3
"""
Utility to stress-test the lab training pipeline across many OHLCV datasets.

Examples:
    python tools/lab_regression.py --limit 5
    python tools/lab_regression.py --files 0000_WETH-USDT.json 0001_USDC-WETH.json --group-size 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.model_lab import get_model_lab_runner
from trading.pipeline import TrainingPipeline


def chunked(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    if size <= 0:
        size = 1
    for idx in range(0, len(seq), size):
        yield list(seq[idx : idx + size])


def resolve_paths(entries: Sequence[str]) -> List[str]:
    runner = get_model_lab_runner()
    resolved = runner.resolve_paths(entries)
    if resolved:
        return resolved
    # fallback: assume absolute paths
    paths: List[str] = []
    for entry in entries:
        path = Path(entry).expanduser().resolve()
        if path.is_file():
            paths.append(str(path))
    return paths


def gather_default_files(limit: int | None = None) -> List[str]:
    base_dir = Path("data") / "historical_ohlcv"
    files = sorted(base_dir.rglob("*.json"))
    if limit:
        files = files[:limit]
    return [str(path) for path in files]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run lab training against multiple datasets sequentially.")
    parser.add_argument("--files", nargs="*", default=[], help="Specific dataset files (relative to data/historical_ohlcv).")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of datasets when using the default file list.")
    parser.add_argument("--group-size", type=int, default=1, help="Number of files per training batch.")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per training invocation.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    args = parser.parse_args(argv)

    selected = args.files or gather_default_files(args.limit if args.limit > 0 else None)
    if not selected:
        print("No datasets found. Populate data/historical_ohlcv first.", file=sys.stderr)
        return 1

    resolved = resolve_paths(selected)
    if not resolved:
        print("Unable to resolve any dataset paths.", file=sys.stderr)
        return 1

    pipeline = TrainingPipeline()
    summary = {
        "total_batches": 0,
        "success": 0,
        "failed": 0,
        "details": [],
    }

    for chunk in chunked(resolved, args.group_size):
        summary["total_batches"] += 1
        start = time.time()
        try:
            _, metrics, info = pipeline.lab_train_on_files(
                chunk,
                epochs=max(1, args.epochs),
                batch_size=max(8, args.batch_size),
            )
            elapsed = time.time() - start
            summary["success"] += 1
            detail = {
                "status": "ok",
                "files": chunk,
                "samples": info.get("samples"),
                "metrics": metrics,
                "duration_sec": elapsed,
            }
            print(f"[OK] {len(chunk)} file(s) in {elapsed:0.2f}s :: samples={info.get('samples')}")
        except Exception as exc:  # pragma: no cover - manual utility
            elapsed = time.time() - start
            summary["failed"] += 1
            detail = {
                "status": "error",
                "files": chunk,
                "error": str(exc),
                "duration_sec": elapsed,
            }
            print(f"[ERROR] {chunk} failed after {elapsed:0.2f}s :: {exc}", file=sys.stderr)
        summary["details"].append(detail)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
