"""Which of the four entry conjuncts is refusing the ghost lane, and by how much.

``BusScheduler.evaluate`` opens a position only when all four of these hold at
once (``trading/scheduler.py``, the enter branch)::

    expected      >  min_profit_floor
    direction_prob >= SCHEDULER_MIN_DIRECTION_PROB   (default 0.6)
    confidence     >= SCHEDULER_MIN_CONFIDENCE       (default 0.6)
    net_margin     >= SCHEDULER_MIN_NET_MARGIN       (default = min_profit)

A four-way AND fails silently: the tick ends at ``no_candidates (thresholds not
met)`` and, before pass 103, wrote nothing to ``trading_ops`` at all. Three
passes of funnel census consequently attributed the missing candidates to the
symbol edge ban, which had refused the worst-affected symbol exactly zero times.

The inputs are already recorded. ``organism_snapshots.payload['prediction']``
carries ``direction_prob`` and ``net_margin`` on every tick, so the question
"can this conjunct EVER be true on the current feed?" is answerable from stored
data without instrumenting anything.

Measured 2026-09-10 over 600 snapshots in 6h, this printed::

    direction_prob      max 0.5000   floor 0.6000   reachable   0/600
    net_margin          max 0.0000   floor 0.0000   reachable   1/600

Two conjuncts that a live feed never satisfies are not a strict gate, they are
a closed lane. THE FLOORS ARE NOT THE DEFECT: ``net_margin`` equals
``price_mu - 0.0065`` and ``model_definition.py`` documents ``price_mu`` as a
fractional return valued on the 0.01-0.1 scale, so a p50 of -1.34 is a head
predicting a -134% price move. Lowering a floor to admit that is deleting the
test that would have caught it. Read this to find out WHICH number to go and
fix, not which floor to move.

Usage::

    python -X utf8 scripts/entry_conjunct_census.py [--hours 6] [--limit 2000]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_DB = os.path.join("storage", "trading_cache.db")

#: (payload key, env var naming its floor, default floor, comparison is >=)
CONJUNCTS: Tuple[Tuple[str, str, float, bool], ...] = (
    ("direction_prob", "SCHEDULER_MIN_DIRECTION_PROB", 0.6, True),
    ("confidence", "SCHEDULER_MIN_CONFIDENCE", 0.6, True),
    ("net_margin", "SCHEDULER_MIN_NET_MARGIN", 0.0, True),
)


def _floor(env_var: str, default: float) -> float:
    """The floor as production reads it, not as the docstring remembers it."""
    raw = os.getenv(env_var)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def read_predictions(
    db_path: str = DEFAULT_DB,
    *,
    hours: float = 6.0,
    limit: int = 2000,
    now: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Every ``prediction`` block written in the window, newest first."""
    cutoff = (now if now is not None else time.time()) - float(hours) * 3600.0
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT payload FROM organism_snapshots WHERE ts > ? "
            "ORDER BY ts DESC LIMIT ?",
            (float(cutoff), int(limit)),
        ).fetchall()
    finally:
        conn.close()
    out: List[Dict[str, Any]] = []
    for (payload,) in rows:
        try:
            snap = json.loads(payload)
        except (TypeError, ValueError):
            continue
        pred = snap.get("prediction")
        if isinstance(pred, dict):
            out.append(pred)
    return out


def census(predictions: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per conjunct: how far its input ever got, and how often it cleared.

    ``reachable`` is the count of ticks on which the conjunct was satisfiable at
    all. Zero over a full window is the finding: no threshold tuning, no
    strategy and no symbol choice can open a position through a conjunct whose
    input never reaches its floor.

    A conjunct whose input is absent from the payload reports ``observed: 0``
    rather than a fabricated zero -- an unmeasured condition must not read as a
    failing one.
    """
    preds = list(predictions)
    result: Dict[str, Dict[str, Any]] = {}
    for key, env_var, default, inclusive in CONJUNCTS:
        floor = _floor(env_var, default)
        values = [
            float(p[key]) for p in preds
            if isinstance(p.get(key), (int, float)) and not isinstance(p.get(key), bool)
        ]
        reachable = sum(
            1 for v in values if (v >= floor if inclusive else v > floor)
        )
        entry: Dict[str, Any] = {
            "floor": floor,
            "floor_env": env_var,
            "observed": len(values),
            "reachable": reachable,
        }
        if values:
            ordered = sorted(values)
            entry.update(
                {
                    "min": ordered[0],
                    "p50": ordered[len(ordered) // 2],
                    "max": ordered[-1],
                    "headroom": ordered[-1] - floor,
                }
            )
        result[key] = entry
    return result


def unsatisfiable(report: Dict[str, Dict[str, Any]]) -> List[str]:
    """Conjuncts that were observed and never once cleared their floor."""
    return [
        key for key, entry in report.items()
        if entry.get("observed", 0) > 0 and entry.get("reachable", 0) == 0
    ]


def render(report: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        f"{'conjunct':<18}{'min':>10}{'p50':>10}{'max':>10}"
        f"{'floor':>10}{'reachable':>16}",
    ]
    for key, entry in report.items():
        if not entry.get("observed"):
            lines.append(f"{key:<18}{'not recorded in the snapshot payload':>56}")
            continue
        lines.append(
            f"{key:<18}{entry['min']:>10.4f}{entry['p50']:>10.4f}"
            f"{entry['max']:>10.4f}{entry['floor']:>10.4f}"
            f"{entry['reachable']:>8}/{entry['observed']:<7}"
        )
    dead = unsatisfiable(report)
    lines.append("")
    if dead:
        lines.append(
            "UNSATISFIABLE ON THIS FEED: " + ", ".join(dead)
            + " -- never cleared its floor once in the window."
        )
        lines.append(
            "The entry test is an AND, so the ghost lane cannot open while that"
            " holds. Fix the NUMBER, not the floor: a floor lowered to admit an"
            " input this far out of range deletes the check that caught it."
        )
    else:
        lines.append("Every observed conjunct cleared its floor at least once.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    preds = read_predictions(args.db, hours=args.hours, limit=args.limit)
    report = census(preds)
    if args.json:
        print(json.dumps({"snapshots": len(preds), "conjuncts": report}, indent=2))
    else:
        print(f"{len(preds)} snapshots with a prediction block in the last "
              f"{args.hours:g}h\n")
        print(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
