"""How many DIFFERENT instruments is data/historical_ohlcv actually made of?

Every brain experiment in this repo picks a corpus file and asks for a horizon
in BARS. A bar is not a unit: this corpus mixes cadences, so ``--horizon 12``
asked about 33 minutes on one file and 48 DAYS on another, and both reports
wrote the same string. That is not a tidiness problem -- whether a prediction
target can pay for itself is a function of the horizon in MINUTES, because the
cost floor is charged in percent of notional and the realised move that has to
clear it grows with wall-clock time, not with bar count.

This script reads every file in the corpus once and answers three things:

  1. how many files sit at each cadence, and what a 12-bar horizon MEANS there
  2. which files are too coarse to be a timeframe anybody chose -- a 4-day
     median gap is a sparse or broken download, and it is currently eligible
     for selection like any other file
  3. which files change cadence mid-file, where even the per-file number lies

The bound it applies is ``omen_experiment.MAX_CORPUS_BAR_SECONDS``, named
there with its reason, so the census and the experiments cannot drift apart.

Cadence is measured two ways on purpose. HEAD cadence is the median of the
first 400 gaps, which is exactly what ``omen_experiment.bar_seconds`` uses to
convert a horizon; FULL cadence is the median over the whole file. Where they
disagree the file is not one instrument and the conversion is wrong somewhere
inside it, so the census names those files rather than averaging over them.

Usage
-----
  python -X utf8 scripts/omen_corpus_cadence_census.py
  python -X utf8 scripts/omen_corpus_cadence_census.py --root data/historical_ohlcv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.omen_experiment import (  # noqa: E402
    DEFAULT_HORIZON_MINUTES, MAX_CADENCE_DRIFT_RATIO, MAX_CORPUS_BAR_SECONDS,
    horizon_bars,
)

#: Pull timestamps out of the raw bytes rather than parsing 3 GB of JSON. The
#: census only needs the time axis, and a full parse of the corpus costs
#: minutes on a box that is also serving the price feed.
TIMESTAMP = re.compile(rb'"timestamp"\s*:\s*(\d+)')

#: Below this a "corpus" is a handful of bars and its median gap is noise.
#: This is the rule that catches the three 23-bar stubs whose "cadence" reads
#: 345600s -- four days is not a timeframe anybody downloaded, it is what the
#: median gap of a 23-row stub spanning three years comes out at.
MIN_BARS_FOR_A_CADENCE = 40

#: MAX_CADENCE_DRIFT_RATIO is imported, not redefined: the census and the
#: experiments must apply the SAME bound or the census stops describing what
#: the selector does.


def _median(values: List[int]) -> int:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def census_file(path: Path) -> Dict[str, Any]:
    """Cadence, span and a mid-file-change flag for one corpus file."""
    stamps = [int(m) for m in TIMESTAMP.findall(path.read_bytes())]
    stamps.sort()
    gaps = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
    row: Dict[str, Any] = {
        # Repo-relative when it lives here, absolute otherwise: a test censuses
        # a synthetic corpus in a tmpdir, and a census that CRASHES on a corpus
        # outside the repo cannot be exercised by the test that proves it.
        "path": (path.relative_to(ROOT).as_posix()
                 if path.is_relative_to(ROOT) else path.as_posix()),
        "chain": path.parent.name,
        "symbol": path.stem.split("_", 1)[-1],
        "bars": len(stamps),
        "megabytes": round(path.stat().st_size / 1e6, 2),
    }
    if len(gaps) < MIN_BARS_FOR_A_CADENCE:
        row.update({"head_seconds": None, "full_seconds": None,
                    "span_days": None, "eligible": False,
                    "reason_class": "too_few_bars",
                    "reason": f"only {len(stamps)} bars, too few to measure "
                              f"a cadence"})
        return row
    head = _median(gaps[:399])
    full = _median(gaps)
    drift = max(head, full) / max(1, min(head, full))
    row.update({
        "head_seconds": head,
        "full_seconds": full,
        "span_days": round((stamps[-1] - stamps[0]) / 86400.0, 2),
        # The head is what the experiment converts a horizon with, so a file
        # whose full-file median differs is one the conversion is wrong in.
        "cadence_drift_ratio": round(drift, 3),
        "cadence_stable": drift <= MAX_CADENCE_DRIFT_RATIO,
        "horizon_12bars_minutes": round(head * 12 / 60.0, 1),
        "default_horizon_bars": horizon_bars(DEFAULT_HORIZON_MINUTES, head),
    })
    if head > MAX_CORPUS_BAR_SECONDS:
        row["eligible"] = False
        row["reason_class"] = "coarser_than_bound"
        row["reason"] = (f"median gap {head}s is coarser than the "
                         f"{MAX_CORPUS_BAR_SECONDS}s bound")
    elif not row["cadence_stable"]:
        row["eligible"] = False
        row["reason_class"] = "cadence_changes_mid_file"
        row["reason"] = (f"cadence changes mid-file by {drift:.2f}x: head "
                         f"{head}s, full-file median {full}s")
    else:
        row["eligible"] = True
        row["reason_class"] = ""
        row["reason"] = ""
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/historical_ohlcv")
    parser.add_argument("--report-dir", default="data/brain_experiments")
    parser.add_argument("--tag", default="p113_corpus_cadence_census")
    args = parser.parse_args()

    root = ROOT / args.root
    files = sorted(root.rglob("*.json"))
    if not files:
        print(f"no corpus under {root}")
        return 2

    started = time.time()
    rows = [census_file(p) for p in files]
    took = time.time() - started

    by_cadence = Counter(r["head_seconds"] for r in rows
                         if r["head_seconds"] is not None)
    eligible = [r for r in rows if r["eligible"]]
    excluded = [r for r in rows if not r["eligible"]]

    print(f"{len(rows)} corpus files under {args.root}, read in {took:.0f}s")
    print(f"\n{'cadence':>10} {'files':>6} {'h12 means':>12} "
          f"{'{:.0f}min means'.format(DEFAULT_HORIZON_MINUTES):>14}  note")
    for cadence, count in sorted(by_cadence.items()):
        h12 = cadence * 12 / 60.0
        note = "" if cadence <= MAX_CORPUS_BAR_SECONDS else "EXCLUDED: coarser than bound"
        print(f"{cadence:>9}s {count:>6} {h12:>10.0f}m "
              f"{horizon_bars(DEFAULT_HORIZON_MINUTES, cadence):>12} bars  {note}")

    print(f"\nSELECTION BOUND: median gap <= {MAX_CORPUS_BAR_SECONDS}s "
          f"({MAX_CORPUS_BAR_SECONDS / 3600:.0f}h)")
    print(f"  eligible {len(eligible)} / {len(rows)}   excluded {len(excluded)}")
    reasons = Counter(r["reason_class"] for r in excluded)
    for reason, count in reasons.most_common():
        print(f"    {count:>4}  {reason}")
    for row in sorted(excluded, key=lambda r: -(r["head_seconds"] or 0))[:12]:
        print(f"      {row['path']}: {row['reason']}")

    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": args.root,
        "files": len(rows),
        "read_seconds": round(took, 1),
        "selection_bound_bar_seconds": MAX_CORPUS_BAR_SECONDS,
        "min_bars_for_a_cadence": MIN_BARS_FOR_A_CADENCE,
        "max_cadence_drift_ratio": MAX_CADENCE_DRIFT_RATIO,
        "excluded_by_reason": dict(Counter(r["reason_class"]
                                           for r in excluded)),
        "default_horizon_minutes": DEFAULT_HORIZON_MINUTES,
        "files_by_cadence_seconds": {str(k): v for k, v in
                                     sorted(by_cadence.items())},
        "horizon_12_bars_minutes_by_cadence": {
            str(k): round(k * 12 / 60.0, 1) for k in sorted(by_cadence)},
        "eligible": len(eligible),
        "excluded": len(excluded),
        "excluded_files": [
            {"path": r["path"], "head_seconds": r["head_seconds"],
             "full_seconds": r["full_seconds"], "bars": r["bars"],
             "reason": r["reason"]}
            for r in sorted(excluded, key=lambda r: -(r["head_seconds"] or 0))],
        "rows": rows,
    }
    out_dir = ROOT / args.report_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.tag}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
