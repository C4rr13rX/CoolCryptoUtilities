"""Reverse the erroneous 2026-09-03 demotion of atf_static.

atf_static is the only strategy that has ever spent real money on this account.
It made three live trades, all three settled on-chain from wallet
0x291c854811e92906a658Fb94Aa511bF919f968ad:

    0x77f0075e0e6b79e71ba63667efe4aec9a80bb2b6e858400f63305d90ec7a6ed1  +0.0012394
    0x927834717d12395c1eb3d9148609a2b8142a59403205d1caa4ffd04e68e0e005  +0.0097748
    0x9ffdd1cfe3f17fbeddb2b3091b49f8f6524b5a0538ae9a2f3aadb56610264e6c  -0.0058591
                                                                        ----------
                                                                        +0.0051551

Two wins, one loss, net positive -- and it was demoted, with
``demote_reason = "live drawdown: +0.0052 from peak +0.0110"``. The give-back
brake in trading/strategies/ledger.py had no minimum-sample gate, so it judged a
ratio against a running maximum built from three points. Under the configured
25% that tolerated a loss of 0.00275 at trade three, while a typical trade on
this feed is 0.00562: any ordinary losing trade demoted it.

The brake is now gated behind STRATEGY_DEMOTE_MIN_DRAWDOWN_TRADES (default 8).
Re-evaluated under the corrected rules, this record is NOT a demotion:

    consecutive losses  1  <  2   (STRATEGY_DEMOTE_MAX_LIVE_LOSSES)
    net live P/L   +0.0052  >  0  (judged from trade 3, and it is positive)
    give-back           n/a       (3 trades < 8)

So this script restores the flag the defect removed. It does NOT grant approval
on new evidence -- it re-applies the corrected rules to the record that was
already there, which is why it is a one-off script and not a ledger method: a
general "reinstate" API would be a way to hand out spending permission without
evidence, and this repo has been burned by exactly that.

The demotion also blanked the ghost book (``_demote_locked`` resets it), which
destroyed 22 trades of accumulated evidence. That is NOT reconstructed here --
the pre-demotion ledger ghost figures were not snapshotted anywhere, and
inventing them would be fabricating a record. The ghost book refills from live
ghost trading; the audit trail below records that it was lost.

Run:  .venv/Scripts/python.exe -X utf8 scripts/restore_atf_static_approval.py [--apply]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.atomic_json import file_lock, read_json, write_json  # noqa: E402
from trading.strategies.ledger import StrategyLedger  # noqa: E402

STRATEGY_ID = "atf_static"

#: The demotion this reverses. Restoring is only correct if the entry on disk is
#: still the exact one the defect produced -- if anything has traded since, the
#: record has moved on and must be re-judged by the live rules, not by this.
EXPECTED_REASON_PREFIX = "live drawdown:"
EXPECTED_LIVE_TRADES = 3
EXPECTED_PROFIT = 0.0051550798571437986


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write the change (default: dry run)")
    args = ap.parse_args()

    path = StrategyLedger.DEFAULT_PATH
    with file_lock(path):
        data, ok = read_json(path, default=None)
        if not ok or not isinstance(data, dict):
            print(f"cannot read ledger at {path}", file=sys.stderr)
            return 1

        ent = data.get(STRATEGY_ID)
        if not isinstance(ent, dict):
            print(f"{STRATEGY_ID} is not in the ledger", file=sys.stderr)
            return 1

        live = ent.get("live", {})
        reason = str(ent.get("demote_reason") or "")
        trades = int(live.get("trades", 0))
        profit = float(live.get("total_profit", 0.0))

        print(f"ledger        : {path}")
        print(f"live_approved : {ent.get('live_approved')}")
        print(f"live trades   : {trades} ({live.get('wins')}W / {live.get('losses')}L)")
        print(f"live P/L      : {profit:+.7f}")
        print(f"consec losses : {live.get('consecutive_losses')}")
        print(f"demote_reason : {reason}")

        if ent.get("live_approved"):
            print("\nalready approved -- nothing to do")
            return 0

        # Refuse to act on a record that has moved on since the defect fired.
        problems = []
        if not reason.startswith(EXPECTED_REASON_PREFIX):
            problems.append(f"demote_reason is not the give-back defect ({reason!r})")
        if trades != EXPECTED_LIVE_TRADES:
            problems.append(f"live trades is {trades}, expected {EXPECTED_LIVE_TRADES}")
        if abs(profit - EXPECTED_PROFIT) > 1e-12:
            problems.append(f"live P/L is {profit!r}, expected {EXPECTED_PROFIT!r}")
        if profit <= 0:
            problems.append("live P/L is not positive -- this demotion was EARNED")
        if problems:
            print("\nREFUSING to restore. The record is not the one the defect demoted:")
            for p in problems:
                print(f"  - {p}")
            print("Re-judge it with the live rules instead.")
            return 2

        if not args.apply:
            print("\ndry run -- would restore live_approved=True. Re-run with --apply")
            return 0

        backup = path.with_suffix(f".json.bak-drawdownfix-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(path, backup)
        print(f"\nbacked up to  : {backup}")

        # Through the ledger's own licence-granting helper, so this script
        # cannot hand out a licence on a contract the ledger has moved past.
        # It re-bases `dd_ref`: without that, restoring approval only buys one
        # live outcome before the give-back brake re-demotes against the same
        # stale `peak_profit` (measured 2026-09-04 -- see _grant_live_licence).
        StrategyLedger._grant_live_licence(ent, ts_key="graduated_ts")
        ent["demote_reason"] = None
        # Keep the audit trail: the demotion happened, and it was wrong.
        ent["reinstated_ts"] = time.time()
        ent["reinstated_reason"] = (
            "reversed: demoted 2026-09-03 by the give-back brake with no minimum "
            "sample, on a 3-trade record that was 2W/1L and net +0.0052. Under the "
            "corrected rule (STRATEGY_DEMOTE_MIN_DRAWDOWN_TRADES=8) this record is "
            "not a demotion. The demotion also blanked a 22-trade ghost book, which "
            "is NOT reconstructed -- those figures were never snapshotted and "
            "inventing them would fabricate a record."
        )

        if not write_json(path, data):
            print("write failed", file=sys.stderr)
            return 1

    ledger = StrategyLedger()
    print(f"restored      : is_live_approved={ledger.is_live_approved(STRATEGY_ID)}")
    print(f"approved_ids  : {ledger.approved_ids()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
