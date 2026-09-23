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


def _populations_view() -> dict:
    """The three strategy populations. See services.strategy_population."""
    from services.strategy_population import populations

    return populations()


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

    # The denominator, carried on the payload rather than left to the reader.
    # `totals.strategies` is the EVIDENCED count and reads like "the
    # population"; it is not, and quoting it as one has already produced three
    # incompatible fractions in a single pass. Degraded to an empty dict on
    # failure so the status still renders -- a missing denominator block is
    # visibly missing, whereas a fabricated one is not.
    pops, pop_err = _quiet(_populations_view)
    if pop_err:
        out["errors"]["populations"] = pop_err
        pops = {}
    out["populations"] = pops

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


def _tradeable_median_from_store():
    """Median net_profit of the live-tradeable closed rows, or None.

    A SUM CAN BE CARRIED BY ONE ROW AND A MEDIAN CANNOT, and this report has
    only ever printed sums. Pass 687 measured the pooled tradeable book at
    +6.1425 over 182 rows with a MEDIAN of -0.009095; the top row was 56% of the
    sum and the top five were 146%, the top two landing on one day under no
    ``strategy_id``. Per symbol it is worse and more general -- six positive
    sums, five negative medians, one row carrying 44-382% of each.

    Returning None rather than a zero is the same choice made throughout this
    file: a report that printed 0.0 when it could not measure would be quoted
    as a measurement. Every failure path here is silent BY DESIGN -- the median
    is a corroboration line, and an unreadable store must not take the ladder
    down with it.

    THIS DELEGATES THE PREDICATE. ``ledger._live_tradeable`` is imported, never
    reimplemented; ``scripts/tradeable_book.py:73`` records what happened the
    last time a second copy of it existed. Note that the predicate is READ-TIME
    and 98% of its refusals come from ``services.symbol_edge_gate``, which
    computes its verdict FROM these same rows -- so this book moves as the gate
    learns. That is why the caller labels the line with its source.
    """
    import json as _json
    import sqlite3 as _sqlite3

    try:
        from trading.strategies.ledger import _live_tradeable
    except Exception:  # noqa: BLE001
        return None
    # THE CENSOR IS NOT OPTIONAL ON THIS LINE, AND IT CARRIES THE SIGN.
    #
    # ``trade_outcomes`` is append-only and has no plausibility guard, so every
    # row the ledger rejected on the way in is still in the money table and
    # every all-time read of it -- including this one -- counts them as fills.
    # Measured pass 688 over the population this function selects:
    #
    #     UNCENSORED   n=182  median -0.009095  total +6.1425
    #     CENSORED     n=177  median -0.009991  total -0.3336
    #
    # FIVE ROWS OF 182 CARRY THE SIGN OF THE WHOLE LINE. Four are unclassified
    # and one is rsi_reversal@1w; the AERO pair is the documented chain, where
    # a fake +161% win exits at a price the tape never printed and the NEXT
    # position opens at that fictional basis and books a real-looking -55%:
    #
    #     AERO-USDC     unclassified     +3.2066   0.436805 -> 1.14
    #     AERO-USDC     unclassified     -1.1115   1.14     -> 0.513839
    #     AAVE-USDC     unclassified     +3.4700   129.485  -> 354.99
    #     BASELINE-USDC unclassified     +0.1999   0.000233 -> 0.000368
    #     UNI-USDC      rsi_reversal@1w  +0.7112   2.859    -> 6.372307
    #
    # THE CONCLUSION IS ROBUST TO THE ONE ROW THE CENSOR GETS WRONG.
    # BASELINE-USDC was flagged by the ratio arm and then EXONERATED by the
    # tape -- ``market_stream`` corroborates BOTH ends within 30 min, a real
    # microcap move. Keeping it would put this line at -0.1337, which is still
    # negative. The claim "the store-derived tradeable book is negative" does
    # not depend on that row, so this delegates the shared rule unchanged
    # rather than hardcoding a per-symbol exemption into a report. A second
    # copy of a predicate is a second answer to one question -- the same reason
    # ``_live_tradeable`` above is imported and never reimplemented.
    #
    # WHAT THIS IS NOT: a P/L improver. It makes the headline WORSE, which is
    # the check working. A filter that flattered the record it judges would be
    # laundering it.
    try:
        from services.outcome_plausibility import IMPLAUSIBLE_RET
    except Exception:  # noqa: BLE001
        return None
    db = os.getenv("TRADING_DB_PATH") or str(ROOT / "storage" / "trading_cache.db")
    try:
        con = _sqlite3.connect("file:%s?mode=ro" % Path(db).as_posix(), uri=True)
    except Exception:  # noqa: BLE001
        return None
    nets = []
    censored = 0
    try:
        for sym, status, net, details, entry_px, exit_px in con.execute(
            "SELECT symbol, status, net_profit, details, entry_price, exit_price "
            "FROM trade_outcomes"
        ):
            if str(status or "").lower() != "closed":
                continue
            try:
                det = _json.loads(details or "{}")
            except Exception:  # noqa: BLE001
                det = {}
            if not isinstance(det, dict):
                det = {}
            if str(det.get("annulled") or "").lower() in ("1", "true", "yes"):
                continue
            sid = str(det.get("strategy_id") or "") or "unclassified"
            try:
                if not _live_tradeable(str(sym or ""), sid):
                    continue
            except Exception:  # noqa: BLE001
                continue
            # The ratio arm of the shared censor, applied to the rows this
            # line is about to take a median of. Only rows with BOTH prices
            # can be judged; one without them is kept, because dropping a row
            # for being unmeasurable would censor on absence rather than on
            # evidence.
            try:
                e_px = float(entry_px or 0.0)
                x_px = float(exit_px or 0.0)
                if e_px > 0.0 and x_px > 0.0 and abs(x_px / e_px - 1.0) > IMPLAUSIBLE_RET:
                    censored += 1
                    continue
            except Exception:  # noqa: BLE001
                pass
            nets.append(float(net or 0.0))
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            con.close()
        except Exception:  # noqa: BLE001
            pass
    if len(nets) < 5:
        return None
    nets.sort()
    total = sum(nets)
    # ``top_share`` IS ONLY MEANINGFUL ON A POSITIVE SUM. It exists to answer
    # "is this sum one row?", and that question presupposes a sum worth
    # carrying. Divided by a negative total it returns a negative percentage --
    # censoring takes this book to -0.3336 and the naive formula printed
    # "top row -252% of the sum", which reads as a measurement and is not one.
    # None means "not applicable" and the caller omits the clause.
    return {
        "n": len(nets),
        "censored": censored,
        "median": nets[len(nets) // 2],
        "total": total,
        "top_share": (100.0 * nets[-1] / total) if total > 0.0 else None,
    }


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
    #
    # RANKED ON TRADEABLE TRADES, NOT POOLED. This test read pooled_trades and
    # that is the one number this loop has already ruled is not evidence --
    # "pooled ghost volume in symbols the live lane refuses is not progress",
    # which the EVIDENCE branch eight lines below says in as many words.
    #
    # Measured 2026-09-10, atf_static_scout, the strategy this branch has named
    # as the system's top wall for at least six consecutive passes:
    #
    #     ledger ghost book        237 trades, 186 wins, +6.4498
    #     ledger TRADEABLE subset    4 trades,   1 win,  -0.1097
    #     rows in trade_outcomes     0          <- all-time, of 210 closed
    #     trading_ops naming it    374          <- 7d, so it IS running
    #
    # So the book that outranked every other wall is 4 tradeable round trips
    # that lose money, and it has no receipts at all: every de-contamination
    # instrument in this repo (the implausibility filter, the take-profit
    # clamp, the tradeable predicate in scripts/tradeable_book.py) reads
    # trade_outcomes, and not one of them can see a single scout trip. A book
    # that cannot be audited must not outrank one that can.
    #
    # On the tradeable count the scout scores 4/20 and this branch no longer
    # fires, so the header falls through to EVIDENCE (TRADEABLE) -- which is
    # the wall the ledger actually has. Porting an edge is still the right
    # answer for a blocked strategy that has PROVEN one; this only stops an
    # unmeasured pooled book from claiming the top of the precedence order.
    best_blocked = None
    by_id = {s["id"]: s for s in ranked}
    for b in blocked or ():
        s = by_id.get(b["id"])
        if s and int(s.get("ghost_trades", 0)) >= min_trades:
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
    # THE DENOMINATOR, NAMED. Three counts of "the strategies" were quoted in
    # one pass -- 72, 43 and 39 -- and a criterion phrased as a share cannot be
    # passed or failed while it is ambiguous which one it divides by. Worse,
    # the ledger count is SELF-REFERENTIAL as a denominator: a strategy enters
    # the ledger by producing the numerator, so "11 of 38" can be satisfied by
    # strategies leaving. Printed here so nobody has to go and find it.
    pop = r.get("populations") or {}
    counts = pop.get("counts") or {}
    if counts:
        A("    ^ that %s is the EVIDENCED population (has recorded an "
          "outcome). It is not the population." % counts.get("evidenced", "?"))
        A("      offered a tick %s  |  commissioned %s  |  evidenced %s  "
          "|  known %s" % (
              counts.get("offered", "?"), counts.get("commissioned", "?"),
              counts.get("evidenced", "?"), counts.get("known", "?")))
        A("      any share of 'the population' uses KNOWN %s; a share of "
          "'strategies that have traded' uses EVIDENCED %s. Say which." % (
              counts.get("known", "?"), counts.get("evidenced", "?")))
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
    # THE MEDIAN, BECAUSE A SUM CAN BE CARRIED BY ONE ROW AND A MEDIAN CANNOT.
    # Measured pass 687: six symbol buckets with a positive SUM, five with a
    # NEGATIVE median, one row carrying 44-382% of each. The pooled tradeable
    # book read +6.1425 over 182 rows with a median of -0.009095, its top row
    # 56% of the sum and its top five 146%. The win rate above does NOT catch
    # this -- AERO reads 54% win on a -0.003381 median -- so the two are
    # printed together and fail differently.
    #
    # SOURCE IS DELIBERATELY NAMED ON THE LINE. The sums above come from the
    # ledger's accumulated counter, which stores no rows, so no median of THOSE
    # rows exists to print. This median is recomputed from trade_outcomes
    # through the ledger's own _live_tradeable, which is a READ-TIME predicate
    # over a WRITE-TIME counter's population -- a different book, and labelling
    # it as anything else would be the drift this report exists to expose.
    med = _tradeable_median_from_store()
    if med is not None:
        line = ("    median row (store, read-time predicate): %+.6f over %d rows"
                % (med["median"], med["n"]))
        if med.get("top_share") is not None:
            line += "; top row %.0f%% of the sum" % med["top_share"]
        A(line)
        # THE CENSOR'S COST IS PRINTED, NOT HIDDEN. A report that silently
        # dropped rows would be quoted as a count of the whole book. Pass 688
        # measured these five rows carrying the SIGN of this line: +6.1425
        # uncensored over 182 rows, -0.3336 censored over 177.
        if med.get("censored"):
            A("      (%d implausible row(s) censored via "
              "services.outcome_plausibility; sum here %+.4f)"
              % (med["censored"], med["total"]))
        if med["median"] <= 0.0 and med["total"] > 0.0:
            A("      ^ POSITIVE SUM, NON-POSITIVE MEDIAN: the sign is carried by "
              "the tail, not by the typical trade. Do not read it as an edge.")
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
        for line in _live_refusal_arithmetic():
            A(line)

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


def _live_refusal_arithmetic() -> list:
    """Print the ARITHMETIC behind step 6, not the label 'ghost_validation_block'.

    The ladder renders each link's detail truncated to 52 characters, so step 6
    has been printing "blocked: ghost_validation_block" for weeks. That string
    names a category, not a cause: it is identical whether the gate refused on
    tail risk, on a loss streak, on drawdown, or -- as measured on 2026-09-22 --
    on a strategy that has never traded.

    The cause is one subtraction and the report never showed it. Production
    publishes the whole verdict in data/reports/confusion_matrices.json under
    ``transition_plan``; every number below is read from there, so this cannot
    drift away from what the gate actually decided. The only hard-coded thing is
    the source line, which is a pointer for the reader, not a value.

    Best-effort like every other section: a missing or malformed report returns
    no lines rather than taking the report down.
    """
    try:
        path = ROOT / "data" / "reports" / "confusion_matrices.json"
        if not path.exists():
            return []
        blob = json.loads(path.read_text(encoding="utf-8"))
        plan = blob.get("transition_plan") or {}
        flags = plan.get("risk_flags") or {}
        if not flags.get("halt_live"):
            return []
        ghost = plan.get("ghost_validation") or {}
        age_min = (time.time() - float(blob.get("updated_at", 0) or 0)) / 60.0

        sid = ghost.get("strategy_id") or "(unnamed)"
        samples = int(ghost.get("samples", 0) or 0)
        need = int(ghost.get("min_trades", 0) or 0)
        reason = str(ghost.get("reason") or flags.get("live_blocked_reason") or "")

        out = [
            "  WHY STEP 6 REFUSES -- the arithmetic, not the label",
            "    (read from the transition plan production published %.0f min ago)"
            % age_min,
            "    gate        : _ghost_validation_for_live  trading/pipeline.py:4749-4752",
            "    reason      : %s" % reason,
            "    judging     : %s -- %d ghost trades of %d required"
            % (sid, samples, need),
            "    consequence : halt_live=%s  live_mode=%s  recommended_live_usd=$%.2f"
            % (
                flags.get("halt_live"),
                flags.get("live_mode", "?"),
                float(flags.get("recommended_live_usd", 0.0) or 0.0),
            ),
        ]
        if "cold_start_bootstrap" in reason:
            out += [
                "    NOT A BUG, A DEADLOCK. An empty book returns ready=True with",
                "    reason='cold_start_bootstrap' (pipeline.py:5159-5165) -- the bypass that",
                "    lets a fresh system start collecting. Line 4750 refuses to spend that as",
                "    evidence, correctly: per-strategy it would hand a live-money 'ready' to",
                "    every strategy that has never been tried. So the ONLY live-approved",
                "    strategy must produce %d ghost trades to be allowed live, and the live" % need,
                "    gate is not what stops it trading -- it simply does not fire.",
                "    The fix is upstream of this gate: either %s must enter more often," % sid,
                "    or a strategy that DOES enter must be the one approved.",
            ]
        out.append("")
        return out
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never take the report down
        return ["  (live refusal arithmetic unavailable: %r)" % (exc,), ""]


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
