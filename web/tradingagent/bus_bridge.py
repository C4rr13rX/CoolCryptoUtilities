"""Know what the bus scheduler has already promised before trading against it.

The scheduler runs multi-horizon routes: a directive carries a horizon, and
the position it opens is expected to resolve inside that window. That is the
bus -- capital committed to be somewhere by a certain time.

The reading agent trades the same wallet. If it buys a token the scheduler is
mid-route on, or sells one the scheduler is holding to a horizon, it does not
merely take a separate position: it moves the shared balance out from under a
plan already in flight. The scheduler then either cannot size its next leg or
exits into a book that changed for reasons it never saw.

So before the agent acts on a symbol it is told what is already scheduled on
it, and what returning that capital in time would require -- the plan to get
the people back on the bus.

Nothing here blocks a trade. The agent may still decide the opportunity is
worth breaking a route for; it simply may not do so unknowingly, and the plan
it must satisfy travels with the decision.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Horizon labels the scheduler uses, in seconds. A directive's horizon is the
#: deadline the capital is expected back by.
HORIZON_SECONDS = {
    "5m": 300, "10m": 600, "15m": 900, "30m": 1800,
    "1h": 3600, "5h": 18000, "12h": 43200,
    "1d": 86400, "3d": 259200, "5d": 432000, "1w": 604800,
}


def _horizon_seconds(label: str) -> Optional[int]:
    return HORIZON_SECONDS.get(str(label or "").strip().lower())


def scheduled_commitments(window_sec: float = 86400) -> List[Dict[str, Any]]:
    """Positions the scheduler has opened and is still expecting to resolve.

    Read from the recorded directives rather than from the live scheduler
    object, because the agent runs in a different process: asking the
    scheduler directly would either need it in-process or return whatever a
    second instance happened to reconstruct.
    """
    out: List[Dict[str, Any]] = []
    try:
        import sqlite3

        conn = sqlite3.connect(
            f"file:{ROOT / 'storage' / 'trading_cache.db'}?mode=ro", uri=True)
        now = time.time()

        # Entries within the window, and the exits that closed them. An entry
        # with no later exit on the same symbol is still riding.
        entries: Dict[str, Dict[str, Any]] = {}
        for ts, symbol, statusstr, details in conn.execute(
                "SELECT ts, symbol, status, details FROM trading_ops "
                "WHERE ts > ? AND (status LIKE '%-entry' OR status LIKE '%-exit') "
                "ORDER BY ts", (now - window_sec,)):
            symbol = str(symbol or "")
            try:
                payload = json.loads(details) if details else {}
            except Exception:
                payload = {}

            if str(statusstr).endswith("-entry"):
                horizon = str(payload.get("horizon") or "")
                entries[symbol] = {
                    "symbol": symbol,
                    "opened_at": float(ts or 0.0),
                    "horizon": horizon,
                    "horizon_sec": _horizon_seconds(horizon),
                    "strategy_id": str(payload.get("strategy_id") or ""),
                    "size": payload.get("size"),
                    "target_price": payload.get("target_price"),
                    "live": str(statusstr).startswith("live"),
                }
            else:
                entries.pop(symbol, None)

        for record in entries.values():
            horizon_sec = record.get("horizon_sec")
            age = time.time() - record["opened_at"]
            record["age_sec"] = round(age, 1)
            if horizon_sec:
                record["due_in_sec"] = round(horizon_sec - age, 1)
                record["overdue"] = age > horizon_sec
            else:
                # No horizon recorded means no deadline can be checked. Say so
                # rather than implying the position is on schedule.
                record["due_in_sec"] = None
                record["overdue"] = None
            out.append(record)
    except Exception:
        pass

    out.sort(key=lambda r: (r.get("due_in_sec") is None, r.get("due_in_sec") or 0))
    return out


def conflicts_for(symbols: List[str], window_sec: float = 86400
                  ) -> Dict[str, Dict[str, Any]]:
    """Which of these symbols the scheduler is already committed on."""
    wanted = {str(s or "").upper() for s in symbols if s}
    return {c["symbol"].upper(): c for c in scheduled_commitments(window_sec)
            if c["symbol"].upper() in wanted}


def return_plan(commitment: Dict[str, Any], clip_usd: float) -> Dict[str, Any]:
    """What it would take to get this capital back on the bus in time.

    Answers the question the agent actually has to satisfy: if I use this
    capital now, by when must it be back, and is that achievable at the size I
    am trading?
    """
    due_in = commitment.get("due_in_sec")
    horizon = commitment.get("horizon") or "unknown"

    if due_in is None:
        return {
            "feasible": None,
            "detail": f"{commitment.get('symbol')} has no recorded horizon, so "
                      f"no return deadline can be checked. Treat it as "
                      f"committed until it exits.",
        }

    if due_in <= 0:
        return {
            "feasible": False,
            "detail": f"{commitment.get('symbol')} is ALREADY OVERDUE on its "
                      f"{horizon} horizon by {abs(due_in) / 60:.1f} min. The "
                      f"scheduler is waiting on capital that has not come back; "
                      f"closing this is worth more than opening anything new.",
        }

    # A round trip needs an entry and an exit to both settle. Settlement here
    # has taken minutes, not seconds, so a deadline inside a few minutes is not
    # a window a trade can fit through.
    minimum_round_trip_sec = 300.0
    feasible = due_in > minimum_round_trip_sec

    return {
        "feasible": feasible,
        "due_in_min": round(due_in / 60.0, 1),
        "detail": (
            f"{commitment.get('symbol')} is due back in {due_in / 60:.1f} min "
            f"on its {horizon} horizon."
            + ("" if feasible else
               " That is inside the ~5 min a round trip needs to settle, so "
               "using this capital now will make the scheduler miss its "
               "window.")),
    }


def bus_briefing(clip_usd: float = 0.0, window_sec: float = 86400) -> List[str]:
    """The scheduler's commitments, as lines the agent can act on."""
    commitments = scheduled_commitments(window_sec)
    if not commitments:
        return ["The bus scheduler has no open commitments; nothing is riding."]

    lines = [f"The scheduler is holding {len(commitments)} position(s). "
             f"Trading these symbols moves capital a plan is already using."]

    overdue = [c for c in commitments if c.get("overdue")]
    if overdue:
        lines.append(
            f"{len(overdue)} are ALREADY OVERDUE: "
            + ", ".join(f"{c['symbol']} ({c.get('horizon') or '?'})"
                        for c in overdue[:6])
            + ". Closing an overdue position returns capital the scheduler is "
              "waiting on, which is usually worth more than a new entry.")

    for commitment in commitments[:8]:
        plan = return_plan(commitment, clip_usd)
        lines.append("  " + plan["detail"])
    return lines
