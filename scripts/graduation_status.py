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
    # Read the SAME population the graduation bar reads. readiness_report
    # prints the pooled ghost book, and graduation judges _tradeable_of(ghost)
    # -- the round trips the live lane could actually have placed. Measured
    # 2026-09-10 the two differ by more than a factor of ten and by SIGN:
    # atf_static_scout is 236 trades / +6.4818 pooled and 3 trades / -0.0777
    # tradeable. A status built on the pooled number reports a strategy as
    # ready that the ledger will never approve, and sends the pass to fix a
    # stamp that is working correctly.
    rep, err = _quiet(_ledger_view)
    if err:
        out["errors"]["ledger"] = err
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
    out["blocked"] = [
        {"id": s["id"], "why": s.get("blocked_reason", "")}
        for s in strategies
        if s.get("structurally_blocked")
    ]
    # A structural block also writes demote_reason, so a strategy that was
    # never actually demoted would otherwise appear here as "x0" -- a demotion
    # that never happened, next to the real ones.
    out["demoted"] = [
        {"id": s["id"], "times": s.get("demotions", 0), "why": s.get("demote_reason", "")}
        for s in strategies
        if s.get("demote_reason") and int(s.get("demotions", 0) or 0) > 0
    ]
    out["closest"] = [
        {
            "id": s["id"],
            "ghost_trades": s["ghost_trades"],
            # readiness_report names this `pooled_ghost_trades`, matching its
            # own `ghost_trades`/`ghost_wins`/`ghost_profit` convention; this
            # module's internal collector names it `pooled_trades`. Both feed
            # this dict, so read both -- a bare `pooled_trades` lookup silently
            # returned 0 and printed "out of 0 pooled" beside a 236-trade book.
            "pooled_trades": s.get("pooled_trades", s.get("pooled_ghost_trades", 0)),
            "win_rate": s["win_rate"],
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
        blocked=out["blocked"],
        demoted=out["demoted"],
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


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _ledger_view() -> dict:
    """The graduation funnel as the LEDGER sees it, not as a report renders it.

    Three things this reads that the pooled view does not:

      * ``_tradeable_of(ghost)`` -- graduation judges only round trips the live
        lane could actually have placed. The pooled book includes symbols the
        live lane refuses on sight, and that evidence can never be spent.
      * ``graduation_blocked`` / ``_ghost_only_ids()`` -- a structural bar that
        outranks any record. A ghost-only executor has no live branch, so no
        amount of evidence can promote it.
      * ``demote_reason`` -- a demoted strategy does not re-graduate; it goes
        through the re-arm rule, which judges evidence gathered SINCE the
        demotion. Counting its lifetime book toward the bar is the thrash the
        ledger was fixed to stop.
    """
    from trading.strategies.ledger import (  # noqa: PLC0415
        StrategyLedger,
        _ghost_only_ids,
        _tradeable_of,
    )

    ledger = StrategyLedger()
    data = dict(getattr(ledger, "_data", {}))
    ghost_only = set(_ghost_only_ids())

    min_trades = _env_int("STRATEGY_GRADUATION_MIN_TRADES", 20)
    min_winrate = _env_float("STRATEGY_GRADUATION_MIN_WINRATE", 0.55)
    min_profit = _env_float("STRATEGY_GRADUATION_MIN_PROFIT", 0.0)

    now = time.time()
    strategies = []
    stamps = []

    for sid, entry in sorted(data.items()):
        ghost = entry.get("ghost", {}) or {}
        live = entry.get("live", {}) or {}
        sub = _tradeable_of(ghost)

        trades = int(sub.get("trades", 0) or 0)
        wins = int(sub.get("wins", 0) or 0)
        profit = float(sub.get("total_profit", 0.0) or 0.0)
        pooled = int(ghost.get("trades", 0) or 0)

        last_ts = float(ghost.get("last_ts", 0.0) or 0.0)
        if last_ts:
            stamps.append(last_ts)

        win_rate = wins / trades if trades else 0.0
        blocked = bool(entry.get("graduation_blocked")) or sid in ghost_only
        demote_reason = entry.get("demote_reason") or ""

        blockers = []
        if blocked:
            blockers.append("structurally blocked: no live branch exists")
        demotions = int(entry.get("demotions", 0) or 0)
        if demote_reason and demotions > 0:
            blockers.append("demoted x%d, judged by the re-arm rule" % demotions)
        if trades < min_trades:
            blockers.append("needs %d more TRADEABLE ghost trades" % (min_trades - trades))
        if trades and win_rate < min_winrate:
            blockers.append(
                "tradeable win rate %.0f%% < %.0f%%" % (win_rate * 100, min_winrate * 100)
            )
        if profit <= min_profit:
            blockers.append("tradeable P/L %+.4f not above %s" % (profit, min_profit))

        strategies.append({
            "id": sid,
            "ghost_trades": trades,           # tradeable -- the bar's population
            "pooled_trades": pooled,          # the whole book, for contrast
            "ghost_wins": wins,
            "win_rate": win_rate,
            "ghost_profit": profit,
            "pooled_profit": float(ghost.get("total_profit", 0.0) or 0.0),
            "live_approved": bool(entry.get("live_approved")),
            "live_trades": int(live.get("trades", 0) or 0),
            "graduated_ts": entry.get("graduated_ts"),
            "demotions": int(entry.get("demotions", 0) or 0),
            "demote_reason": demote_reason,
            "structurally_blocked": blocked,
            "blocked_reason": demote_reason if blocked else "",
            "last_trade_age_days": (now - last_ts) / 86400 if last_ts else None,
            "blockers": blockers,
            "ready": not blockers,
        })

    total_trades = sum(s["ghost_trades"] for s in strategies)
    total_wins = sum(s["ghost_wins"] for s in strategies)
    total_pooled = sum(s["pooled_trades"] for s in strategies)
    span_days = ((max(stamps) - min(stamps)) / 86400) if len(stamps) > 1 else 0.0
    rate = (total_trades / span_days) if span_days > 0 else 0.0

    eta = None
    if rate > 0 and strategies:
        active = max(1, sum(1 for s in strategies if s["ghost_trades"] > 0))
        per_strategy = rate / active
        shortfalls = [
            (min_trades - s["ghost_trades"]) / per_strategy
            for s in strategies
            if s["ghost_trades"] < min_trades
            and not s["live_approved"]
            and not s["structurally_blocked"]
        ]
        if shortfalls:
            eta = min(shortfalls)

    return {
        "criteria": {
            "min_trades": min_trades,
            "min_winrate": min_winrate,
            "min_profit": min_profit,
            "enforced": os.getenv("STRATEGY_GRADUATION_ENFORCED", "1"),
            "population": "_tradeable_of(ghost) -- what the live lane could have placed",
        },
        "totals": {
            "strategies": len(strategies),
            "ghost_trades": total_trades,
            "pooled_trades": total_pooled,
            "ghost_wins": total_wins,
            "win_rate": total_wins / total_trades if total_trades else 0.0,
            "ghost_profit": sum(s["ghost_profit"] for s in strategies),
            "pooled_profit": sum(s["pooled_profit"] for s in strategies),
            "live_approved": sum(1 for s in strategies if s["live_approved"]),
            "ledger_span_days": span_days,
            "trades_per_day": rate,
            "eta_days_to_first_graduation": eta,
        },
        "strategies": strategies,
    }


def classify_wall(approved, ready_not_approved, ranked, min_trades,
                  blocked=(), demoted=()) -> str:
    """Name the ONE thing standing between the ledger and a live-armed strategy.

    ``live_approved == 0`` is the same number in five different situations and
    they need different work. The order below is the order the ledger itself
    applies: a structural bar outranks any record, a demotion routes to the
    re-arm rule rather than to graduation, and only then does the bar apply.
    """
    if approved:
        return "APPROVED -- graduation is done; the wall is downstream (gate/executor)"

    if ready_not_approved:
        return (
            "READY BUT UNSTAMPED -- %s clears the bar on TRADEABLE evidence and "
            "carries no approval; the ledger is not stamping graduated_ts"
            % ", ".join(list(ready_not_approved)[:3])
        )

    near = ranked[0] if ranked else None
    if near is None:
        return "NO STRATEGIES IN THE LEDGER -- nothing can graduate"

    # A strategy holding the biggest book while being structurally unable to
    # spend it is the most misleading state available: the evidence reads as
    # progress and can never become a trade.
    best_blocked = None
    by_id = {s["id"]: s for s in ranked}
    for b in blocked or ():
        s = by_id.get(b["id"])
        if s and int(s.get("pooled_trades", 0)) >= min_trades:
            best_blocked = b
            break
    if best_blocked and int(near.get("ghost_trades", 0)) < min_trades:
        return (
            "STRUCTURALLY BLOCKED -- %s holds the biggest ghost book and can "
            "NEVER spend it (%s). Its evidence is not progress: either give it a "
            "live branch, or run the same edge from a strategy that has one"
            % (best_blocked["id"], (best_blocked.get("why") or "no live branch")[:90])
        )

    if int(near.get("ghost_trades", 0)) < min_trades:
        return (
            "EVIDENCE (TRADEABLE) -- the closest strategy (%s) has %d/%d ghost "
            "trades THE LIVE LANE COULD HAVE PLACED, out of %d pooled. The funnel "
            "needs closed round trips in tradeable symbols; pooled ghost volume "
            "is not progress"
            % (
                near["id"],
                near.get("ghost_trades", 0),
                min_trades,
                near.get("pooled_trades", 0),
            )
        )

    if demoted and any(d["id"] == near["id"] for d in demoted):
        return (
            "DEMOTED -- %s has the trades and is judged by the re-arm rule on "
            "evidence gathered SINCE its demotion, not on its lifetime book"
            % near["id"]
        )

    return "QUALITY -- strategies have the tradeable trades but not the win rate/P-L: %s" % (
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
    A("  bar: %s TRADEABLE ghost trades, %s win rate, P/L above %s   (enforced=%s)" % (
        crit.get("min_trades", "?"),
        _pct(crit.get("min_winrate")),
        crit.get("min_profit", "?"),
        crit.get("enforced", "?"),
    ))
    A("  the bar counts ONLY round trips the live lane could have placed --")
    A("  pooled ghost volume in symbols it refuses is not progress.")
    A("")
    A("  ledger: %s strategies over %.1f days" % (
        tot.get("strategies", "?"), float(tot.get("ledger_span_days", 0.0) or 0.0)))
    A("    tradeable : %s trades, %s win, P/L %+.4f    <- what graduation reads"
      % (
          tot.get("ghost_trades", "?"),
          _pct(tot.get("win_rate")),
          float(tot.get("ghost_profit", 0.0) or 0.0),
      ))
    A("    pooled    : %s trades,          P/L %+.4f    <- what the old report showed"
      % (
          tot.get("pooled_trades", "?"),
          float(tot.get("pooled_profit", 0.0) or 0.0),
      ))
    eta = tot.get("eta_days_to_first_graduation")
    A("  tradeable evidence rate: %.1f/day; nearest graduation ~%s"
      % (
          float(tot.get("trades_per_day", 0.0) or 0.0),
          ("%.1f days" % eta) if eta else "never at this rate",
      ))
    A("")
    A("  THE WALL: %s" % r.get("wall", "unknown"))
    A("")

    A("  closest to the bar (ranked on TRADEABLE evidence):")
    A("    %-30s %6s %6s %5s %8s  %s"
      % ("strategy", "trade", "pooled", "win", "P/L", "blockers"))
    for s_ in r.get("closest", []):
        A("    %-30s %6d %6d %5s %+8.4f  %s" % (
            s_["id"][:30],
            s_["ghost_trades"],
            s_.get("pooled_trades", 0),
            _pct(s_["win_rate"]),
            s_["ghost_profit"],
            "; ".join(s_["blockers"])[:60],
        ))
    if r.get("approved"):
        A("")
        A("  APPROVED: %s" % ", ".join(r["approved"]))
    if r.get("blocked"):
        A("")
        A("  STRUCTURALLY BLOCKED (evidence that can never be spent):")
        for b in r["blocked"]:
            A("    %-30s %s" % (b["id"][:30], (b.get("why") or "no live branch")[:60]))
    if r.get("demoted"):
        A("")
        A("  DEMOTED (judged by the re-arm rule on evidence since demotion):")
        for d in r["demoted"]:
            A("    %-30s x%-3d %s" % (d["id"][:30], d.get("times", 0), (d.get("why") or "")[:52]))
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
    """One flat line of JSON for the gate to diff before and after a pass.

    ``ghost_trades`` is the TRADEABLE count, because that is the population the
    graduation bar reads. ``pooled_trades`` rides along so a pass can see the
    gap it is closing -- on 2026-09-10 those were 24 and 410.
    """
    tot = r.get("totals", {})
    links = r.get("live_path") or []
    first = r.get("first_failing_link") or {}
    closest = r.get("closest") or []
    min_trades = crit_min(r)
    shortfalls = [
        max(0, min_trades - int(s.get("ghost_trades", 0))) for s in closest
    ]
    return {
        "live_approved": int(tot.get("live_approved", 0) or 0),
        "ready_not_approved": len(r.get("ready_not_approved") or []),
        "structurally_blocked": len(r.get("blocked") or []),
        "demoted": len(r.get("demoted") or []),
        "ghost_trades": int(tot.get("ghost_trades", 0) or 0),
        "pooled_trades": int(tot.get("pooled_trades", 0) or 0),
        "ghost_win_rate": round(float(tot.get("win_rate", 0.0) or 0.0), 4),
        "ghost_profit": round(float(tot.get("ghost_profit", 0.0) or 0.0), 4),
        "pooled_profit": round(float(tot.get("pooled_profit", 0.0) or 0.0), 4),
        "evidence_per_day": round(float(tot.get("trades_per_day", 0.0) or 0.0), 2),
        "eta_days": tot.get("eta_days_to_first_graduation"),
        "links_passing": sum(1 for l in links if l.get("ok") is True),
        "links_total": len(links),
        "first_failing_link": first.get("name"),
        "closest_shortfall": min(shortfalls) if shortfalls else None,
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
