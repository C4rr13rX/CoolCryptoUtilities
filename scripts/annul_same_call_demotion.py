"""Annul a demotion that fired in the same ``record()`` call as the graduation.

Measured 2026-09-04. ``atf_static`` -- the only strategy that has ever spent
real money on this account, and net POSITIVE at +0.14230439571137737 over 9 live
round trips -- reads ``live_approved=False``. Link 5 of the path check has said
"no strategy approved for live" ever since, and no strategy has traded live for
2h43m.

The ledger recorded the mechanism in its own timestamps:

    graduated_ts  1788529132.2898452   (2026-09-04T13:38:52.289845Z)
    demoted_ts    1788529132.2898726   (2026-09-04T13:38:52.289873Z)

27 microseconds apart. ``_evaluate_graduation_locked`` approved it and
``_evaluate_demotion_locked``, running next in the same ``record()``, revoked
the approval before a single live decision could be taken against it.

The cause is fixed in trading/strategies/ledger.py: graduation never re-based
``dd_ref``, so the give-back brake judged the new licence against
``peak_profit`` = +0.2221 -- a high-water mark set under a licence that had
already been revoked. The 25% bar was +0.1666 against a current +0.1423.
``_grant_live_licence`` now owns the re-base for both ways in, and
tests/test_graduation_rebases_the_drawdown_peak.py pins it.

Fixing the code does not fix the record. The entry still carries the demotion,
and a demoted strategy is routed to ``_maybe_rearm_locked``, which requires 20
FRESH ghost trades gathered since the demotion. atf_static has 1, and it was a
loss. At the measured ghost yield (~3 exits an hour across every strategy
combined) that is days away -- so the bug would keep costing trading time long
after it stopped being reachable.

This re-applies the corrected rule to the record that was already there. It
grants NOTHING on new evidence: the entry graduated on its own ghost book
(22/36 = 61% against a 55% bar, +0.9127) moments before it was demoted, and it
is that graduation being restored.

Deliberately narrow. It refuses unless ALL of these hold, because each one is
part of the fingerprint of this specific defect:

  * currently demoted, with a reason starting "live drawdown:", AND
  * ``graduated_ts`` and ``demoted_ts`` are within 1 second of each other
    (the same-call signature -- a demotion earned over real trading is not),
    AND
  * ``live.dd_ref`` is absent (the field whose absence IS the defect), AND
  * live P/L is strictly positive (the brake only ever fires on strategies
    that are still up; a strategy that lost real money keeps its demotion),
    AND
  * the ghost book independently clears the graduation bar.

A strategy demoted for a genuine net loss, a losing streak, or a give-back
made under its current licence is left exactly where it is.

Run:  .venv/Scripts/python.exe -X utf8 scripts/annul_same_call_demotion.py [--apply]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.atomic_json import file_lock, read_json, write_json  # noqa: E402
from trading.strategies.ledger import StrategyLedger  # noqa: E402

#: How close the two timestamps must be to call it "the same call". The real
#: gap was 2.7e-05s; a second is four orders of magnitude of headroom and still
#: nothing like a demotion earned across real trades.
SAME_CALL_SECONDS = 1.0


def _bar() -> tuple:
    return (
        int(os.getenv("STRATEGY_GRADUATION_MIN_TRADES", "20")),
        float(os.getenv("STRATEGY_GRADUATION_MIN_WINRATE", "0.55")),
        float(os.getenv("STRATEGY_GRADUATION_MIN_PROFIT", "0.0")),
    )


def _judge(sid: str, ent: dict) -> tuple:
    """Return (qualifies, [reasons it does not])."""
    problems = []
    live = ent.get("live") or {}
    ghost = ent.get("ghost") or {}
    reason = str(ent.get("demote_reason") or "")

    if ent.get("live_approved"):
        problems.append("already approved")
    if not reason.startswith("live drawdown:"):
        problems.append(f"demote_reason is not the give-back brake ({reason!r})")

    grad = ent.get("graduated_ts")
    demo = ent.get("demoted_ts")
    if not isinstance(grad, (int, float)) or not isinstance(demo, (int, float)):
        problems.append("no graduated_ts/demoted_ts pair to compare")
    else:
        gap = abs(float(demo) - float(grad))
        if gap > SAME_CALL_SECONDS:
            problems.append(
                f"graduation and demotion are {gap:.3f}s apart -- not the same call"
            )

    if "dd_ref" in live:
        problems.append("live.dd_ref is present -- this entry was judged on a re-based peak")

    profit = float(live.get("total_profit", 0.0) or 0.0)
    if profit <= 0.0:
        problems.append(f"live P/L {profit:+.7f} is not positive -- the demotion was EARNED")

    min_trades, min_winrate, min_profit = _bar()
    trades = int(ghost.get("trades", 0) or 0)
    wins = int(ghost.get("wins", 0) or 0)
    gprofit = float(ghost.get("total_profit", 0.0) or 0.0)
    rate = wins / max(trades, 1)
    if not (trades >= min_trades and rate >= min_winrate and gprofit > min_profit):
        problems.append(
            f"ghost book {wins}/{trades} ({rate:.0%}) {gprofit:+.4f} does not clear "
            f"the graduation bar ({min_trades}, {min_winrate:.0%}, >{min_profit})"
        )
    return (not problems, problems)


def main() -> int:
    ap = argparse.ArgumentParser(description="Annul same-call drawdown demotions.")
    ap.add_argument("--apply", action="store_true", help="write the change (default: dry run)")
    args = ap.parse_args()

    path = StrategyLedger.DEFAULT_PATH
    repaired = []
    with file_lock(path):
        data, ok = read_json(path, default=None)
        if not ok or not isinstance(data, dict):
            print(f"cannot read ledger at {path}", file=sys.stderr)
            return 1

        print(f"ledger: {path}")
        print(f"graduation bar: {_bar()}")
        print()

        for sid, ent in data.items():
            if not isinstance(ent, dict) or ent.get("live_approved"):
                continue
            if not str(ent.get("demote_reason") or ""):
                continue
            qualifies, problems = _judge(sid, ent)
            live = ent.get("live") or {}
            print(f"{sid}")
            print(f"    demote_reason : {ent.get('demote_reason')}")
            print(f"    live          : {live.get('trades')} trades, "
                  f"P/L {float(live.get('total_profit', 0.0) or 0.0):+.7f}, "
                  f"peak {float(live.get('peak_profit', 0.0) or 0.0):+.7f}, "
                  f"dd_ref {live.get('dd_ref')!r}")
            print(f"    qualifies     : {qualifies}")
            for p in problems:
                print(f"      - {p}")
            print()
            if qualifies:
                repaired.append(sid)

        if not repaired:
            print("nothing to annul.")
            return 0

        if not args.apply:
            print(f"dry run -- would annul {len(repaired)}: {', '.join(repaired)}")
            print("re-run with --apply to write.")
            return 0

        backup = Path(str(path) + f".bak-samecalldemotion-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(path, backup)
        print(f"backed up -> {backup.name}")

        for sid in repaired:
            ent = data[sid]
            was = str(ent.get("demote_reason"))
            # Through the ledger's own helper: it re-bases dd_ref, which is the
            # entire point -- restoring the flag without it buys one live
            # outcome before the same brake fires on the same stale peak.
            StrategyLedger._grant_live_licence(ent, ts_key="graduated_ts")
            ent["demote_reason"] = None
            ent["live"]["consecutive_losses"] = 0
            # Keep `demotions`: it happened, and zeroing it would hide that.
            ent.setdefault("corrections", []).append({
                "ts": time.time(),
                "mode": "live",
                "what": "annulled a demotion that fired in the same record() call as the graduation",
                "was": was,
                "reason": (
                    "_evaluate_graduation_locked did not re-base live.dd_ref, so the "
                    "give-back brake judged the new licence against peak_profit "
                    f"{float(ent['live'].get('peak_profit', 0.0)):+.7f} -- a high-water "
                    "mark set under a licence already revoked. Fixed in "
                    "trading/strategies/ledger.py::_grant_live_licence, which now owns "
                    "the re-base for both graduation and re-arm."
                ),
                "evidence": (
                    f"graduated_ts {ent.get('graduated_ts')!r} vs demoted_ts "
                    f"{ent.get('demoted_ts')!r} in the pre-repair ledger: 2.7e-05s apart"
                ),
                "dd_ref_now": ent["live"]["dd_ref"],
            })
            print(f"  annulled {sid} -> live_approved=True, dd_ref={ent['live']['dd_ref']:+.7f}")

        if not write_json(path, data):
            print("write failed", file=sys.stderr)
            return 1

    ledger = StrategyLedger()
    print()
    print(f"approved_ids: {ledger.approved_ids()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
