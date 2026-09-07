"""Stamp the licence P/L reference on entries demoted before the field existed.

``StrategyLedger._licence_net`` measures live P/L earned under the CURRENT
licence to trade, from ``live["pl_ref"]``, and falls back to the LIFETIME sum
when that field is absent so an old ledger is judged exactly as it was before.
``_demote_locked`` now stamps the reference when it revokes a licence, and
``_grant_live_licence`` when it grants one -- but every entry demoted before
that code shipped carries neither, so the fallback keeps them on the lifetime
sum and the ratchet the fix removes is still closed on them.

That is not a hypothetical. Measured 2026-09-06 on data/strategy_ledger.json,
with LIVE TRADES TODAY: 0 and ``approved_ids()`` empty all day:

    atf_static   live 18 trades, 5W/13L, total_profit -0.186371
                 pl_ref absent, trades_ref absent
                 demote_reason "live P/L -0.1585 over 17 trades is not
                               profitable"
                 demotions 7, graduation_blocked False

so ``_maybe_rearm_locked`` reads net=-0.186371 <= 0 and returns before it looks
at any ghost evidence -- forever, because a demoted strategy takes no further
live trades and the sum therefore cannot move. atf_static is the only entry in
the ledger with a live execution branch, so that one frozen number is what
empties ``approved_ids()``, which is what makes ``_live_gate_candidates()``
empty, which is what leaves the live gate judging the pooled book of 36
strategies instead of the one that would spend the money.

WHAT THIS DOES AND DOES NOT DO. It writes two fields:

    live["pl_ref"]     = live["total_profit"]   (the sum at the demotion)
    live["trades_ref"] = live["trades"]         (the count at the demotion)

It does NOT set ``live_approved``, clear ``demote_reason``, or touch
``total_profit``, ``trades``, ``peak_profit`` or the ghost book. Nothing is
forgiven and nothing is re-approved: the lifetime record stays exactly where it
is, still quoted by the demote reason, still on every dashboard. What changes is
that ``_maybe_rearm_locked`` can now reach the test that matters, and that test
is unchanged -- a full graduation-grade ghost book gathered AFTER the demotion.
atf_static has 6 such trades against a bar of 20, so it does not re-arm today;
it becomes able to.

Entries that are currently APPROVED are skipped. Their licence is live and its
reference belongs to whenever it was granted, which this script cannot know --
guessing "now" would silently forgive whatever the current licence has already
lost, and that is the one direction this must not fail in.

    python scripts/stamp_licence_pl_reference.py            report
    python scripts/stamp_licence_pl_reference.py --apply    write
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "strategy_ledger.json"


def main() -> int:
    apply = "--apply" in sys.argv[1:]
    if not LEDGER.exists():
        print("ledger not found: %s" % LEDGER)
        return 1

    data = json.loads(LEDGER.read_text(encoding="utf-8"))
    pending = []
    for sid, ent in sorted(data.items()):
        if not isinstance(ent, dict):
            continue
        live = ent.get("live")
        if not isinstance(live, dict):
            continue
        if ent.get("live_approved"):
            print("%-34s SKIP  currently approved; its licence is open" % sid)
            continue
        if live.get("pl_ref") is not None and live.get("trades_ref") is not None:
            continue
        trades = int(live.get("trades", 0) or 0)
        net = float(live.get("total_profit", 0.0) or 0.0)
        if trades <= 0 and net == 0.0:
            # Never traded live. The fallback and the stamp agree at zero, so
            # writing one would be noise in a file people read by hand.
            continue
        print("%-34s STAMP live %d trades, net %+.6f  (demote_reason: %s)"
              % (sid, trades, net, str(ent.get("demote_reason") or "-")[:60]))
        pending.append((sid, trades, net))

    if not pending:
        print("\nnothing to stamp.")
        return 0

    if not apply:
        print("\nreport only. re-run with --apply to stamp %d entr%s."
              % (len(pending), "y" if len(pending) == 1 else "ies"))
        return 0

    backup = LEDGER.with_suffix(".json.bak-plref-%s"
                                % time.strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(LEDGER, backup)
    print("\nbacked up -> %s" % backup.name)

    # Through the ledger's own lock, and re-read inside it.
    #
    # Production records a ghost outcome into this file every few minutes, so a
    # bare read-modify-write here would write back a copy loaded seconds ago and
    # silently revert every outcome recorded in between -- the ledger's own
    # `demote()` carries the same warning, and this repo has already lost 95% of
    # a run's outcomes to exactly that shape. The values stamped are read fresh
    # from inside the lock rather than reused from the report above, so a trade
    # that lands between the two does not get stamped out of the record.
    sys.path.insert(0, str(ROOT))
    from trading.strategies.ledger import StrategyLedger

    ledger = StrategyLedger(path=str(LEDGER))
    stamped = 0
    with ledger._lock, ledger._file_lock():
        ledger._load()
        for sid, _reported_trades, _reported_net in pending:
            ent = ledger._data.get(sid)
            if not isinstance(ent, dict) or ent.get("live_approved"):
                print("%-34s skipped: approved since the report was taken" % sid)
                continue
            live = ent.setdefault("live", {})
            if live.get("pl_ref") is not None and live.get("trades_ref") is not None:
                continue
            net = float(live.get("total_profit", 0.0) or 0.0)
            trades = int(live.get("trades", 0) or 0)
            live["pl_ref"] = net
            live["trades_ref"] = trades
            ent.setdefault("corrections", []).append({
                "ts": time.time(),
                "what": "stamped the licence P/L reference at the demotion boundary",
                "reason": (
                    "_licence_net falls back to the LIFETIME total when pl_ref "
                    "is absent, and a demotion freezes that total at the value "
                    "which caused it -- so _maybe_rearm_locked returned before "
                    "reading any ghost evidence and the strategy could never be "
                    "re-armed. Neither total_profit nor trades is altered; no "
                    "approval is granted; re-arming still requires a full fresh "
                    "ghost book."
                ),
                "pl_ref": net,
                "trades_ref": trades,
            })
            print("%-34s stamped pl_ref=%+.6f trades_ref=%d" % (sid, net, trades))
            stamped += 1
        if stamped:
            ledger._save()

    print("stamped %d entr%s." % (stamped, "y" if stamped == 1 else "ies"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
