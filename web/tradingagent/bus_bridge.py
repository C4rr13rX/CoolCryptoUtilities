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


def _position_book(conn) -> Optional[set]:
    """Symbols the bot actually holds, or None if the book cannot be read.

    None is not "empty". An unreadable book must not be taken as proof that
    nothing is held -- that would let the agent open a position on a symbol it
    is already carrying. Only a book that parses is allowed to filter.
    """
    try:
        row = conn.execute(
            "SELECT value FROM kv_store WHERE key = 'state'").fetchone()
    except Exception:  # noqa: BLE001
        return None
    if not row or not row[0]:
        return None
    try:
        state = json.loads(row[0])
    except Exception:  # noqa: BLE001
        return None
    positions = state.get("positions")
    if not isinstance(positions, dict):
        return None
    return {str(sym) for sym in positions}


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

        # AN ENTRY IS ALSO CLEARED BY THINGS THAT ARE NOT AN EXIT.
        #
        # Reconstructing "still open" from entry-minus-exit rows misses every
        # other way a position leaves the book, and those are the MAJORITY:
        # the dark-feed sweep abandons a position without booking an exit (466
        # abandonments in 24h, by design -- marking out against an 11-day-old
        # price would fabricate the outcome), and slot releases and phantom
        # drops do the same.
        #
        # Measured 2026-09-05: trading_ops showed 1130 of 1745 entries with no
        # matching exit, and 26 symbols "held" for 218-344 hours, while the
        # persisted position book held ZERO positions. The agent believed it
        # was carrying PEPE 487 minutes past a 30-minute horizon and spent
        # every one of 33 runs sweeping positions that did not exist -- so it
        # considered only PEPE and CBZEC and never looked at a new token,
        # while WALDO, ZZZ, KEYCAT, JACKET and CRUX all appeared on the feed.
        for release_status in ("position-released", "position-abandoned-dark-feed",
                               "live-position-dropped-phantom"):
            for (symbol,) in conn.execute(
                    "SELECT symbol FROM trading_ops WHERE ts > ? AND status = ?",
                    (now - window_sec, release_status)):
                entries.pop(str(symbol or ""), None)

        # THE BOOK IS THE TRUTH; THE LOG IS A NARRATIVE OF IT.
        #
        # Even with every release status accounted for, a reconstruction can
        # only ever be as complete as the list of statuses someone remembered
        # to enumerate. The persisted book is what the bot actually holds, so
        # anything absent from it is not a commitment however its log rows
        # read. Used as a FILTER rather than a source, because the book
        # carries no horizon and this function's whole job is deadlines.
        book = _position_book(conn)
        if book is not None:
            entries = {sym: rec for sym, rec in entries.items() if sym in book}

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


def moonlight_window(commitment: Dict[str, Any],
                     *, round_trip_sec: float = 300.0) -> Dict[str, Any]:
    """The slack inside a commitment -- time that is free to earn in.

    A commitment is a DEADLINE, not a prohibition. The teenager who has to be
    home by curfew is not thereby forbidden to leave; they are forbidden to be
    late. The hours in between are theirs, and what they earn in them is
    theirs too.

    That is the difference between this and treating a scheduled position as
    a blocker. A 3-day horizon does not mean the capital is busy for three
    days -- it means it must be back in three days. If a round trip resolves
    in ten minutes, that window holds roughly 430 of them, and refusing all
    of them to protect a deadline three days out is not prudence, it is
    leaving the whole night unused.

    Returns how many round trips fit, and how much slack is left after
    reserving one full trip as the margin for getting home.
    """
    due_in = commitment.get("due_in_sec")
    if due_in is None:
        # No deadline recorded means no window can be computed. Silence is not
        # permission: treat it as fully committed until it exits.
        return {"trips": 0, "slack_sec": 0.0, "reason":
                f"{commitment.get('symbol')} has no recorded horizon, so no "
                f"return window can be measured"}

    due_in = float(due_in)
    trip = max(60.0, float(round_trip_sec))

    # RESERVE THE TRIP HOME. Never plan into the last round trip's worth of
    # time -- being late is the one failure that is not recoverable, because
    # the scheduler is sizing its next leg against capital it expects back.
    usable = due_in - trip
    if usable <= 0:
        return {"trips": 0, "slack_sec": 0.0, "reason":
                f"{commitment.get('symbol')} is due back in {due_in / 60:.1f} "
                f"min, which is inside the {trip / 60:.1f} min a round trip "
                f"needs -- there is no night left to go out in"}

    trips = int(usable // trip)
    return {
        "trips": trips,
        "slack_sec": round(usable, 1),
        "reason": (
            f"{commitment.get('symbol')} is due back in {due_in / 60:.1f} min. "
            f"After reserving {trip / 60:.1f} min to get home, {usable / 60:.1f} "
            f"min are free -- about {trips} round trip(s) of earning inside a "
            f"commitment that is not otherwise doing anything."),
    }


def compounding_plan(realised_profit_usd: float, clip_usd: float,
                     *, reinvest_fraction: float = 0.5) -> Dict[str, Any]:
    """What the earnings should become: a bigger operation, not a bigger spend.

    The soda money does not get spent on sodas. It buys a second cooler, then
    a van. Profit that is simply re-risked at the same clip grows nothing --
    the operation stays the size it was and only the variance grows.

    So realised profit raises the CLIP, which is the size of the next trade,
    and does so on the half that is kept working while the other half stays
    banked. That is the difference between a business that compounds and one
    that merely churns.
    """
    profit = float(realised_profit_usd or 0.0)
    clip = max(0.01, float(clip_usd or 0.0))

    if profit <= 0:
        return {"new_clip_usd": clip, "reinvested_usd": 0.0,
                "reason": "nothing realised yet; the operation stays its "
                          "current size"}

    reinvest = profit * max(0.0, min(1.0, float(reinvest_fraction)))
    new_clip = clip + reinvest
    return {
        "new_clip_usd": round(new_clip, 6),
        "reinvested_usd": round(reinvest, 6),
        "banked_usd": round(profit - reinvest, 6),
        "reason": (
            f"${profit:.4f} realised: ${reinvest:.4f} back into the clip "
            f"(${clip:.4f} -> ${new_clip:.4f}), ${profit - reinvest:.4f} "
            f"banked. The next trade is bigger because the last one worked, "
            f"which is the only way small money becomes large money."),
    }


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
        # ...and what the commitment leaves FREE. A deadline three days out is
        # not three days of idleness; it is three days minus one trip home.
        window = moonlight_window(commitment)
        if window.get("trips", 0) > 0:
            lines.append("    " + window["reason"])
    return lines
