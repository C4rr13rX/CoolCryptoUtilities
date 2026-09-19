"""[c9e248aa] The candidate-to-entry census: who is refused, by whom, and why.

Criterion 1 of the item demands a per-status, per-strategy census of
trading_ops action=enter and action=hold, because `action='enter'` includes
CANDIDATE rows and a census that does not split by status reports hundreds of
entries where there are tens.

Criterion 2 demands the refuser be NAMED WITH A NUMBER, and asks what fraction
of the candidates each refuser accounts for. That second half is built on a
false premise and this script says so rather than answering it: refusal rows
are emitted per tick per strategy, not per candidate, so there are ~4.5x more
holds than candidates in any window. A percentage-of-candidates would be a
number with the wrong denominator. What IS answerable, and is answered here, is
the composition of each refuser.

The largest refuser, `entry-arbitration`, carries the top-level reason
`one_tick_was_offered_to_many_and_spent_by_one`. Pass 483 established that this
label is a COVER: the real per-strategy reason is inside the row's `skipped{}`
dict, and it has now lied twice (see also the confidence_floor case, where 126
of 126 rsi_reversal refusals hid under the same label). Pass 483 opened
skipped{} for money_button only. This opens it for EVERY strategy, and also
reads `chosen` to name who actually wins the tick.

Read-only. Touches no gate, no threshold and no production file.
"""

from __future__ import annotations

import collections
import json
import os
import sqlite3
import sys
import time

DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
WINDOW_H = float(os.environ.get("CENSUS_WINDOW_H", "24"))

CANDIDATE_STATUSES = {"ghost_candidate", "ghost_candidate_quote_ok"}
ENTRY_STATUSES = {"ghost-entry", "live-entry"}


def _details(raw):
    try:
        return json.loads(raw or "{}")
    except (ValueError, TypeError):
        return {}


def main():
    db = sqlite3.connect(DB)
    cur = db.cursor()
    cutoff = time.time() - WINDOW_H * 3600.0

    print("=" * 76)
    print("[c9e248aa] candidate -> entry conversion, last %.0fh" % WINDOW_H)
    print("=" * 76)

    # ---- criterion 1: action x status, never pooled ---------------------
    cur.execute(
        "SELECT action, status, COUNT(*) FROM trading_ops "
        "WHERE ts>=? AND action IN ('enter','hold') GROUP BY action, status "
        "ORDER BY COUNT(*) DESC",
        (cutoff,),
    )
    rows = cur.fetchall()
    cands = sum(n for a, s, n in rows if a == "enter" and s in CANDIDATE_STATUSES)
    entries = sum(n for a, s, n in rows if a == "enter" and s in ENTRY_STATUSES)
    holds = sum(n for a, s, n in rows if a == "hold")

    print("\n--- 1. ACTION x STATUS (action='enter' is NOT an entry) ---")
    for a, s, n in rows:
        kind = ""
        if a == "enter" and s in CANDIDATE_STATUSES:
            kind = "  <- CANDIDATE, not an entry"
        elif a == "enter" and s in ENTRY_STATUSES:
            kind = "  <- a real entry"
        print("  %-6s %-30s %6d%s" % (a, s, n, kind))
    print("\n  candidates %d   entries %d   holds %d" % (cands, entries, holds))
    print("  conversion: %d/%d = %.2f%%" % (entries, cands, 100.0 * entries / cands if cands else 0.0))

    # ---- the denominator warning the item's criterion 2 needs -----------
    print("\n--- 2. WHY 'FRACTION OF CANDIDATES' IS THE WRONG DENOMINATOR ---")
    print("  holds %d vs candidates %d = %.1fx. Refusals are emitted per tick" % (holds, cands, holds / cands if cands else 0.0))
    print("  per strategy, not per candidate, so they do NOT partition the")
    print("  candidate set and no refuser can be given a %-of-candidates share.")
    print("  Composition of the refusers, which IS well defined:")
    for a, s, n in rows:
        if a == "hold":
            print("    %-30s %6d  %5.1f%% of holds" % (s, n, 100.0 * n / holds))

    # ---- criterion 1, per strategy: who actually enters ------------------
    print("\n--- 3. WHO ACTUALLY ENTERS (strategy_id on the entry rows) ---")
    cur.execute(
        "SELECT status, details FROM trading_ops WHERE ts>=? AND action='enter' "
        "AND status IN ('ghost-entry','live-entry')",
        (cutoff,),
    )
    by_strat = collections.Counter()
    for status, raw in cur.fetchall():
        d = _details(raw)
        by_strat[(status, d.get("strategy_id") or "(none)")] += 1
    if not by_strat:
        print("  no entry rows in the window.")
    for (status, sid), n in by_strat.most_common():
        print("  %-12s %-34s %5d" % (status, sid, n))

    # ---- the interior of the largest refuser -----------------------------
    print("\n--- 4. INSIDE entry-arbitration: the top-level label is a COVER ---")
    print("  reason= at the top of the row is a ROW-LEVEL label. The real")
    print("  per-strategy verdict is in skipped{}. Opening it for ALL strategies:")
    cur.execute(
        "SELECT details FROM trading_ops WHERE ts>=? AND action='hold' "
        "AND status='entry-arbitration'",
        (cutoff,),
    )
    arb = cur.fetchall()
    winners = collections.Counter()
    skipped_by_strat = collections.defaultdict(collections.Counter)
    offered_total = 0
    for (raw,) in arb:
        d = _details(raw)
        chosen = d.get("chosen")
        if chosen:
            winners[chosen if isinstance(chosen, str) else json.dumps(chosen)[:40]] += 1
        skipped = d.get("skipped") or {}
        if isinstance(skipped, dict):
            offered_total += 1
            for sid, why in skipped.items():
                key = why if isinstance(why, str) else json.dumps(why)[:40]
                skipped_by_strat[sid][key] += 1

    print("\n  rows: %d   with a skipped{} dict: %d" % (len(arb), offered_total))
    print("\n  WHO WINS THE TICK (chosen):")
    for name, n in winners.most_common(10):
        print("    %-40s %5d  %5.1f%%" % (name, n, 100.0 * n / len(arb) if arb else 0.0))
    if not winners:
        print("    (no 'chosen' field populated)")

    # The shape that matters is not which strategy refuses most -- they are
    # all offered the same ticks and tie exactly. It is that almost every one
    # of them declines with a content-free label.
    mute, substantive = [], []
    for sid, reasons in skipped_by_strat.items():
        tot = sum(reasons.values())
        if tot and reasons.get("no_signal", 0) / tot >= 0.95:
            mute.append(sid)
        else:
            substantive.append((sid, reasons))
    print("\n  WHO REFUSES, AND ON WHAT (skipped{}):")
    print("    strategies appearing in skipped{}: %d" % len(skipped_by_strat))
    print("    of those, >=95%% 'no_signal': %d   <- a content-free refusal" % len(mute))
    print("    giving a SUBSTANTIVE reason: %d" % len(substantive))
    print("\n    the substantive ones, in full:")
    for sid, reasons in sorted(substantive, key=lambda kv: -sum(kv[1].values())):
        tot = sum(reasons.values())
        top = "  ".join("%s %d (%d%%)" % (r, n, round(100.0 * n / tot))
                        for r, n in reasons.most_common(6))
        print("      %-24s n=%-5d %s" % (sid, tot, top))
    print("\n    NOT RESOLVED HERE: 'no_signal' may mean 'I evaluated and found")
    print("    nothing' or 'I was never really run'. Those are different bugs")
    print("    and this census cannot tell them apart. Do not assume either.")

    # ---- what happens to the remainder -----------------------------------
    print("\n--- 5. THE REMAINDER: candidates that are neither entered nor held ---")
    print("  %d candidates - %d entries = %d unaccounted at the entry row." % (cands, entries, cands - entries))
    print("  They do not vanish: each tick re-emits a candidate, so the SAME")
    print("  symbol appears many times. Distinct symbols is the honest count:")
    cur.execute(
        "SELECT status, COUNT(DISTINCT symbol) FROM trading_ops WHERE ts>=? "
        "AND action='enter' GROUP BY status",
        (cutoff,),
    )
    per_status = dict(cur.fetchall())
    for s, n in sorted(per_status.items(), key=lambda kv: -kv[1]):
        print("    %-30s %4d distinct symbols" % (s, n))
    plain = per_status.get("ghost_candidate", 0)
    ok = per_status.get("ghost_candidate_quote_ok", 0)
    ent = per_status.get("ghost-entry", 0)
    if plain:
        print("\n  THE FUNNEL IN SYMBOLS, which is the honest unit:")
        print("    %d proposed -> %d quotable (%.0f%% lost) -> %d entered"
              % (plain, ok, 100.0 * (1 - ok / plain), ent))
        print("  The QUOTE step is the large cut, not the entry gate.")

    db.close()


if __name__ == "__main__":
    sys.exit(main())
