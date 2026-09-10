#!/usr/bin/env python3
"""
The graduation funnel, in one screen, for the refinement loop to open each pass.

The loop's goal is a strategy ARMED for live trading. That is not one number,
it is a funnel: evidence has to arrive (ghost trades), it has to be good enough
(win rate, ghost P/L), the ledger has to stamp an approval, and the live gate
chain has to let the approval reach a bot. A pass that only reads "live trades
today: 0" cannot tell which of those four is the wall it is standing at.

So this composes three views that already exist and are already trusted:

  scripts/loop_status.py       is the pipeline alive at all
  scripts/readiness_report.py  what the ghost record supports, per strategy
  scripts/live_path_check.py   the ten links between a tick and a paid trade

and puts the graduation question at the top, because that is the goal.

Run:  python scripts/graduation_status.py
      python scripts/graduation_status.py --json     (metrics for the gate)

Every section is best-effort: a subsystem that raises prints its error and the
rest of the report still renders. A status command that dies because one table
is locked tells the next pass nothing, which is the failure mode this exists to
avoid.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The report reads books; nothing here should ever place an order.
os.environ.setdefault("REVENIR_READ_ONLY", "1")


def _quiet(fn, *args, **kwargs):
    """Call fn, swallowing whatever it prints. Returns (value, error_string)."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(io.StringIO()):
            return fn(*args, **kwargs), None
    except Exception:  # noqa: BLE001 -- a broken section must not kill the report
        return None, traceback.format_exc(limit=3).strip().splitlines()[-1]


def collect() -> dict:
    out: dict = {"generated_at": time.time(), "errors": {}}

    # --- graduation: the goal ------------------------------------------------
    try:
        from scripts.readiness_report import collect as readiness_collect
    except Exception:  # noqa: BLE001
        sys.path.insert(0, str(ROOT / "scripts"))
        from readiness_report import collect as readiness_collect  # type: ignore

    rep, err = _quiet(readiness_collect)
    if err:
        out["errors"]["readiness"] = err
        rep = {"criteria": {}, "totals": {}, "strategies": []}
    out["criteria"] = rep.get("criteria", {})
    out["totals"] = rep.get("totals", {})

    strategies = rep.get("strategies", []) or []
    crit = out["criteria"]
    min_trades = int(crit.get("min_trades", 20) or 20)

    # Ranked by how close each is to the bar, so a pass can see what to feed.
    def _distance(s: dict) -> tuple:
        short = max(0, min_trades - int(s.get("ghost_trades", 0)))
        return (0 if s.get("ready") else 1, short, -float(s.get("ghost_profit", 0.0)))

    ranked = sorted(strategies, key=_distance)
    out["approved"] = [s["id"] for s in strategies if s.get("live_approved")]
    out["ready_not_approved"] = [
        s["id"] for s in strategies if s.get("ready") and not s.get("live_approved")
    ]
    out["closest"] = [
        {
            "id": s["id"],
            "ghost_trades": s["ghost_trades"],
            "win_rate": s["win_rate"],
            "win_rate_lower_95": s.get("win_rate_lower_95", 0.0),
            "ghost_profit": s["ghost_profit"],
            "blockers": s["blockers"],
        }
        for s in ranked[:12]
    ]

    # Which wall is the funnel actually standing at? One answer, not five.
    out["wall"] = classify_wall(
        approved=out["approved"],
        ready_not_approved=out["ready_not_approved"],
        ranked=ranked,
        min_trades=min_trades,
    )

    # --- the ten links -------------------------------------------------------
    links, err = _quiet(_live_path_links)
    if err:
        out["errors"]["live_path"] = err
    out["live_path"] = links or []
    # ok is tri-state; None (unmeasurable) is a wall too, not a pass.
    out["first_failing_link"] = next(
        (l for l in (links or []) if l.get("ok") is not True), None
    )

    # --- is anything alive at all -------------------------------------------
    pipe, err = _quiet(_pipeline_metrics)
    if err:
        out["errors"]["pipeline"] = err
    out["pipeline"] = pipe or {}

    return out


def classify_wall(approved, ready_not_approved, ranked, min_trades) -> str:
    """Name the ONE thing standing between the ledger and a live-armed strategy.

    ``live_approved == 0`` is the same number whether nothing has enough
    evidence yet or two strategies clear the bar and were never stamped. Those
    are opposite jobs -- generate ghost trades, versus fix the stamping -- so
    the classification, not the count, is what a pass acts on.
    """
    if approved:
        return "APPROVED -- graduation is done; the wall is downstream (gate/executor)"
    if ready_not_approved:
        return (
            "READY BUT UNSTAMPED -- %s clears the bar and carries no approval; "
            "the ledger is not stamping graduated_ts"
            % ", ".join(list(ready_not_approved)[:3])
        )
    near = ranked[0] if ranked else None
    if near is None:
        return "NO STRATEGIES IN THE LEDGER -- nothing can graduate"
    if int(near.get("ghost_trades", 0)) < min_trades:
        return (
            "EVIDENCE -- the closest strategy (%s) has %d/%d ghost trades; "
            "the funnel needs closed ghost round trips, not more analysis"
            % (near["id"], near.get("ghost_trades", 0), min_trades)
        )
    return "QUALITY -- strategies have the trades but not the win rate/P-L: %s" % (
        "; ".join(near.get("blockers", []))[:200]
    )


def _live_path_links() -> list:
    """The ten links, as data. Falls back to parsing if the module has no API.

    ``ok`` is tri-state upstream: True, False, or None for a link that could not
    be measured. An unmeasurable link is NOT a passing one, so it is carried
    through as None and counted as failing when picking the first wall.
    """
    try:
        from scripts.live_path_check import run as run_links  # type: ignore
    except Exception:  # noqa: BLE001
        try:
            sys.path.insert(0, str(ROOT / "scripts"))
            from live_path_check import run as run_links  # type: ignore
        except Exception:  # noqa: BLE001
            return _live_path_links_by_parsing()
    out = []
    for link in run_links():
        d = link.to_dict()
        out.append(
            {
                "step": d.get("step"),
                "name": d.get("name"),
                "ok": d.get("ok"),
                "detail": (d.get("detail") or "")[:200],
                "fix": (d.get("fix") or "")[:200],
            }
        )
    return out


def _live_path_links_by_parsing() -> list:
    """Run the checker as a subprocess and read its table. Slow but honest."""
    import re
    import subprocess

    exe = os.environ.get("REVENIR_PYTHON") or sys.executable
    proc = subprocess.run(
        [exe, "-X", "utf8", str(ROOT / "scripts" / "live_path_check.py")],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=str(ROOT),
    )
    rows = []
    pat = re.compile(r"\[(PASS|FAIL)\s*\]\s+(\d+)\s+(\S+)\s*(.*)")
    for line in (proc.stdout or "").splitlines():
        m = pat.search(line)
        if m:
            rows.append(
                {
                    "step": int(m.group(2)),
                    "name": m.group(3),
                    "ok": m.group(1) == "PASS",
                    "detail": m.group(4).strip()[:160],
                }
            )
    return rows


def _pipeline_metrics() -> dict:
    """Whatever loop_status already measures, as JSON."""
    import subprocess

    exe = os.environ.get("REVENIR_PYTHON") or sys.executable
    proc = subprocess.run(
        [exe, "-X", "utf8", str(ROOT / "scripts" / "loop_status.py"), "--json"],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(ROOT),
    )
    text = (proc.stdout or "").strip()
    start = text.find("{")
    if start < 0:
        return {}
    return json.loads(text[start:])


def render(r: dict) -> str:
    o = []
    A = o.append
    crit = r.get("criteria", {})
    tot = r.get("totals", {})

    A("=" * 74)
    A("GRADUATION TO LIVE TRADING -- the only scoreboard for this loop")
    A("=" * 74)
    A("")
    A("  LIVE-APPROVED STRATEGIES: %d        <- this is the number to move"
      % int(tot.get("live_approved", 0) or 0))
    A("")
    A("  bar: %s ghost trades, %s win rate, ghost P/L above %s   (enforced=%s)" % (
        crit.get("min_trades", "?"),
        _pct(crit.get("min_winrate")),
        crit.get("min_profit", "?"),
        crit.get("enforced", "?"),
    ))
    A("  ledger: %s strategies, %s ghost trades, %s win, P/L %+.4f over %.1f days"
      % (
          tot.get("strategies", "?"),
          tot.get("ghost_trades", "?"),
          _pct(tot.get("win_rate")),
          float(tot.get("ghost_profit", 0.0) or 0.0),
          float(tot.get("ledger_span_days", 0.0) or 0.0),
      ))
    eta = tot.get("eta_days_to_first_graduation")
    A("  evidence rate: %.1f ghost trades/day; nearest graduation ~%s"
      % (
          float(tot.get("trades_per_day", 0.0) or 0.0),
          ("%.1f days" % eta) if eta else "never at this rate",
      ))
    A("")
    A("  THE WALL: %s" % r.get("wall", "unknown"))
    A("")

    A("  closest to the bar:")
    A("    %-34s %6s %6s %6s  %s" % ("strategy", "ghost", "win", "P/L", "blockers"))
    for s in r.get("closest", []):
        A("    %-34s %6d %5s %+7.4f  %s" % (
            s["id"][:34],
            s["ghost_trades"],
            _pct(s["win_rate"]),
            s["ghost_profit"],
            "; ".join(s["blockers"])[:70],
        ))
    if r.get("approved"):
        A("")
        A("  APPROVED: %s" % ", ".join(r["approved"]))
    A("")

    links = r.get("live_path") or []
    if links:
        A("-" * 74)
        A("  PATH TO A PAID LIVE TRADE")
        for l in links:
            A("    [%s] %-2s %-11s %s" % (
                {True: "PASS", False: "FAIL"}.get(l["ok"], " -- "),
                l.get("step", "?"),
                (l.get("name") or "")[:11],
                (l.get("detail") or "")[:52],
            ))
        first = r.get("first_failing_link")
        if first:
            A("    NEXT: step %s (%s) -- everything after it is unknowable."
              % (first.get("step"), first.get("name")))
        else:
            A("    Every link passes.")
        A("")

    pipe = r.get("pipeline") or {}
    if pipe:
        A("-" * 74)
        A("  PIPELINE: %s" % json.dumps(pipe)[:600])
        A("")

    for where, err in (r.get("errors") or {}).items():
        A("  !! %s section failed: %s" % (where, err))

    A("=" * 74)
    return "\n".join(o)


def _pct(v) -> str:
    try:
        return "%.0f%%" % (float(v) * 100)
    except (TypeError, ValueError):
        return "?"


def metrics(r: dict) -> dict:
    """One flat line of JSON for the gate to diff before and after a pass."""
    tot = r.get("totals", {})
    links = r.get("live_path") or []
    first = r.get("first_failing_link") or {}
    return {
        "live_approved": int(tot.get("live_approved", 0) or 0),
        "ready_not_approved": len(r.get("ready_not_approved") or []),
        "ghost_trades": int(tot.get("ghost_trades", 0) or 0),
        "ghost_win_rate": round(float(tot.get("win_rate", 0.0) or 0.0), 4),
        "ghost_profit": round(float(tot.get("ghost_profit", 0.0) or 0.0), 4),
        "evidence_per_day": round(float(tot.get("trades_per_day", 0.0) or 0.0), 2),
        "eta_days": tot.get("eta_days_to_first_graduation"),
        "links_passing": sum(1 for l in links if l.get("ok") is True),
        "links_total": len(links),
        "first_failing_link": first.get("name"),
        "closest_shortfall": min(
            [
                max(0, int(crit_min(r)) - s["ghost_trades"])
                for s in (r.get("closest") or [])
            ]
            or [None]
        )
        if r.get("closest")
        else None,
    }


def crit_min(r: dict) -> int:
    try:
        return int(r.get("criteria", {}).get("min_trades", 20) or 20)
    except (TypeError, ValueError):
        return 20


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="flat metrics for the gate")
    ap.add_argument("--full-json", action="store_true", help="the whole report")
    args = ap.parse_args()

    report = collect()
    if args.full_json:
        print(json.dumps(report, indent=2, default=str))
    elif args.json:
        print(json.dumps(metrics(report)))
    else:
        print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
