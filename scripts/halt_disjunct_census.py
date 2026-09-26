"""Recompute which of halt_live's nine disjuncts actually fired, from the logged
booleans rather than from the halt_reason label.

WHY THIS EXISTS, and why it is not just a nicer print of halt_reason.

`pipeline.py` publishes `risk_flags.halt_live` as a nine-way disjunction, and
publishes `risk_flags.halt_reason` immediately below it as a separate or-chain.
Those two are not the same question. The or-chain has a label for only four of
the nine disjuncts (the ghost check's own reason, wallet_sparse, capital_deficit,
native_starved) plus an externally-supplied block_reason that wins ahead of all
of them. tail_block, drawdown_breach, loss_rate_block, loss_streak_block and
recommended_ratio <= 0 have no label at all, so a plan halted by one of those
reports whichever labelled disjunct happens to also be true -- or nothing.

So reading halt_reason and concluding "that is why the lane halted" is reading
the first LABELLED cause, not the firing one. Pass 720 checked whether that
distinction pays: on the stored tape it is nearly free on the reason string (1
halt row in 1050 carries an empty reason) and decisive on the attribution.

WHAT IT MEASURED, at pass 720 over 1350 rows carrying risk_flags.halt_live,
1050 of them with halt_live true:

    recommended_ratio <= 0    1050   100.0%   <- universal
    not ghost_ready           1028    97.9%
    tail_block                 406     38.7%
    loss_rate_block             51      4.9%
    loss_streak_block           29      2.8%
    wallet_sparse               16      1.5%
    capital_deficit > 0          5      0.5%
    native_starved               5      0.5%
    drawdown_breach              0      0.0%

`risk_budget_cap` is the literal float 0.0 on all 1050 -- not a missing key
coalesced to zero, not None: the key is present on every row and takes exactly
one distinct value. 22 rows have ghost_ready TRUE and halt_live TRUE anyway.

The consequence is the reason this script is in the tree: repairing ghost_ready
alone would move halt_live on ZERO of the 1050 rows, because
`recommended_ratio <= 0` is still true underneath it. ghost_ready is a 97.9%
co-traveller of the halt, not its cause. Anyone proposing to fix the entry-lane
halt by way of the ghost evidence bar should run this first and see the 100.0%
line sitting above their 97.9% one.

Those 22 ghost_ready-true halts also bound an earlier pass-718 claim of mine
that the only route to a zero multiplier is the `(mini_ready or ghost_ready)`
else-branch in pipeline.py: with ghost_ready true and the cap still 0.0, there
is a second route, and that claim is incomplete.

Read-only. Touches nothing but the trading store.

    python scripts/halt_disjunct_census.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import Counter

DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")

# The nine disjuncts of pipeline.py's halt_live, each paired with how to recover
# it from a published plan. They do not all live under risk_flags: ghost_ready is
# hoisted to live_transition and native_starved sits in wallet_state, which is
# part of why reading this off one dict has been getting it wrong.
#
# recommended_ratio is republished as risk_flags.risk_budget_cap; that rename is
# the reason the raw string "recommended_ratio" finds zero rows in the store and
# an earlier census concluded the term was never logged.


def disjuncts(live_transition: dict) -> dict:
    flags = live_transition.get("plan", {}).get("risk_flags", {})
    wallet = live_transition.get("wallet_state", {})
    cap = flags.get("risk_budget_cap")
    return {
        "tail_block": bool(flags.get("tail_block")),
        "drawdown_breach": bool(flags.get("drawdown_breach")),
        "not ghost_ready": live_transition.get("ghost_ready") is not True,
        "loss_rate_block": bool(flags.get("loss_rate_block")),
        "loss_streak_block": bool(flags.get("loss_streak_block")),
        "wallet_sparse": bool(flags.get("wallet_sparse")),
        "capital_deficit > 0": (flags.get("capital_deficit") or 0) > 0,
        "native_starved": bool(wallet.get("native_starved")),
        # No default. A missing cap is reported as its own bucket below rather
        # than silently counted as halting or not halting -- that substitution is
        # exactly how a census talks itself into a number.
        "recommended_ratio <= 0": cap is not None and cap <= 0,
    }


def main() -> int:
    if not os.path.exists(DB):
        print(f"no trading store at {DB}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    # The control first. A filtered count that returns something plausible is not
    # evidence the filter matched what you meant -- this store has burned several
    # passes on exactly that, so print the denominator before the numerator.
    total = conn.execute("select count(*) from trading_ops").fetchone()[0]

    carried = 0
    halted = 0
    fired = Counter()
    labels = Counter()
    caps = Counter()
    cap_missing = 0
    ghost_ready_but_halted = []

    rows = conn.execute(
        "select ts, details from trading_ops where details like '%halt_live%'"
    )
    for ts, raw in rows:
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            continue
        transition = payload.get("brain", {}).get("live_transition", {})
        flags = transition.get("plan", {}).get("risk_flags", {})
        if "halt_live" not in flags:
            continue
        carried += 1
        if not flags["halt_live"]:
            continue
        halted += 1

        labels[flags.get("halt_reason") or ""] += 1
        if "risk_budget_cap" not in flags:
            cap_missing += 1
        else:
            caps[repr(flags["risk_budget_cap"])] += 1

        on = disjuncts(transition)
        for name, is_on in on.items():
            if is_on:
                fired[name] += 1
        if transition.get("ghost_ready") is True:
            ghost_ready_but_halted.append((ts, flags.get("halt_reason") or ""))

    print(f"trading_ops rows, no filter        : {total}")
    print(f"rows carrying risk_flags.halt_live : {carried}")
    print(f"rows with halt_live TRUE           : {halted}")
    if not halted:
        print("\nno halt rows -- nothing to attribute")
        return 0

    print(f"\nFIRING DISJUNCTS, recomputed from the booleans (n={halted}):")
    for name in disjuncts({}):
        n = fired[name]
        print(f"  {name:<24} {n:6d}  {100.0 * n / halted:5.1f}%")

    print("\nhalt_reason LABEL, for contrast -- this is what the or-chain reported:")
    for label, n in labels.most_common():
        print(f"  {n:6d}  {label!r}")

    print(f"\nrisk_budget_cap values: {dict(caps.most_common(8))}")
    print(f"risk_budget_cap key absent on: {cap_missing}")

    print(f"\nghost_ready TRUE yet halt_live TRUE: {len(ghost_ready_but_halted)}")
    for ts, reason in ghost_ready_but_halted[:10]:
        print(f"  ts={ts:.0f}  reason={reason!r}")

    # The line the script exists to print. A disjunction is only explained by a
    # term that is true whenever it is.
    universal = [n for n in disjuncts({}) if fired[n] == halted]
    print(f"\ndisjuncts true on EVERY halt row: {universal or 'none'}")
    if universal and "not ghost_ready" not in universal:
        print(
            "  => repairing ghost_ready alone leaves halt_live TRUE on all "
            f"{halted} rows; it is a co-traveller, not the cause."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
