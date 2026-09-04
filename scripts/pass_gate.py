"""Refuse to accept a pass that broke something, and hold it to a clock.

Two rules were being asked for in prose and ignored in practice.

BREAKAGE. "Never let a fix break something else" was an instruction, and an
instruction is a request. Between 2026-09-02 and 2026-09-03 17:00 the loop
landed 69 commits with no such rule in force, and the damage was exactly the
kind it describes: an exit that sized from a stored quantity instead of the
on-chain balance sold 40-60% of two positions, which booked losses the market
never produced, which demoted the only live strategy, which left every strategy
at live_approved=False so nothing could trade at all.

CLOCK. The sprint said "timeboxed" and named no time. A pass could spend an
hour understanding and still call itself compliant.

This makes both mechanical. It runs the tests that guard the money path and
compares the result to the snapshot taken before the pass started:

  * a test that PASSED before and FAILS now is a regression, and the pass is
    REJECTED regardless of what else it achieved;
  * a pass that ran past its budget without producing a settled swap is
    reported as OVER BUDGET, so the next pass inherits a shorter leash rather
    than the same open-ended one.

It cannot stop a bad commit from being written -- only the agent can do that
-- but it makes the breakage impossible to overlook, which is the part that
kept failing.

Usage:
    python scripts/pass_gate.py --snapshot   # before a pass
    python scripts/pass_gate.py --check      # after a pass
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "pass_gate_snapshot.json"

#: How long a pass may run before it is called over budget, when it has not
#: produced a settled swap. The user's target: a strategy adjusted or built to
#: reach live profitable trades inside ten minutes.
SPRINT_BUDGET_SEC = float(os.getenv("SPRINT_BUDGET_SEC", "600"))

#: The files that pin the money path. Kept narrow so the gate runs in seconds
#: and can be applied to every pass rather than skipped for being slow.
# Named explicitly rather than matched by keyword. A keyword sweep pulled in
# 47 files, some of which import TensorFlow and take minutes, so the gate could
# not finish and reported "0 passed" -- which would have waved every pass
# through. These are the files that pin the money path and run in seconds.
GATE_TESTS = (
    "test_demotion_needs_a_net_loss.py",
    "test_token_contract_guard.py",
    "test_ledger_rejects_artifacts.py",
    "test_dust_is_not_a_sparse_wallet.py",
    "test_atf_feed_corroboration.py",
    "test_feed_density_gate.py",
    "test_swap_never_requires_0x.py",
    "test_settled_swap_is_always_recorded.py",
    "test_token_resolution_unblocks_live.py",
    "test_money_path_records_tx_hash.py",
    "test_boundary_contracts.py",
)


def _targets() -> list:
    out = []
    for name in GATE_TESTS:
        path = ROOT / "tests" / name
        if path.exists():
            out.append(str(path))
    return out


def _run_tests() -> dict:
    """Per-test outcomes, so a regression can be named rather than counted."""
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    exe = str(py) if py.exists() else sys.executable
    targets = _targets()
    if not targets:
        return {"ran": False, "outcomes": {}}

    # No timeout: a cut-off run reports "did not run", which would read as a
    # clean pass and defeat the gate.
    try:
        out = subprocess.run(
            [exe, "-m", "pytest", *targets, "-q", "--no-header", "-rf",
             "--tb=no"],
            cwd=str(ROOT), capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001
        return {"ran": False, "outcomes": {}, "error": str(exc)}

    text = (out.stdout or "") + (out.stderr or "")
    outcomes = {}
    for line in text.splitlines():
        m = re.match(r"^(?:FAILED|ERROR)\s+(\S+)", line.strip())
        if m:
            outcomes[m.group(1)] = "fail"
    passed = re.search(r"(\d+) passed", text)
    failed = re.search(r"(\d+) (?:failed|error)", text)
    return {
        "ran": bool(passed or failed),
        "passed": int(passed.group(1)) if passed else 0,
        "failed": int(failed.group(1)) if failed else 0,
        "outcomes": outcomes,
    }


def _profit_numbers() -> dict:
    """The numbers constraint 3 says every change must justify itself against.

    Reported before and after each pass so "this raises profitability" is a
    measurement rather than a claim. A pass that moved none of them has not
    shown its work, whatever it built.
    """
    out = {"live_trades": 0, "net_pl": 0.0, "profit_factor": 0.0,
           "stranded_positions": 0}
    try:
        sys.path.insert(0, str(ROOT))
        from services import strategy_registry

        gross_win = gross_loss = 0.0
        for row in strategy_registry.list_strategies():
            live = ((row.get("lifetime") or {}).get("live")) or {}
            out["live_trades"] += int(live.get("trades") or 0)
            out["net_pl"] += float(live.get("total_profit") or 0.0)
            gross_win += abs(float(live.get("gross_win") or 0.0))
            gross_loss += abs(float(live.get("gross_loss") or 0.0))
        if gross_loss > 0:
            out["profit_factor"] = round(gross_win / gross_loss, 4)
        elif gross_win > 0:
            out["profit_factor"] = 999.0
        out["net_pl"] = round(out["net_pl"], 6)
    except Exception:
        pass

    # Capital that entered a position and never came back out.
    try:
        import sqlite3

        c = sqlite3.connect(
            "file:%s?mode=ro" % (ROOT / "storage" / "trading_cache.db"), uri=True)
        entries = list(c.execute(
            "SELECT COUNT(*) FROM trading_ops WHERE status='live-entry'"))[0][0]
        exits = list(c.execute(
            "SELECT COUNT(*) FROM trading_ops WHERE status='live-exit'"))[0][0]
        out["stranded_positions"] = max(0, entries - exits)
    except Exception:
        pass
    return out


def _settled_swaps() -> int:
    import sqlite3

    try:
        c = sqlite3.connect(
            "file:%s?mode=ro" % (ROOT / "storage" / "trading_cache.db"), uri=True)
        return list(c.execute(
            "SELECT COUNT(*) FROM trading_ops WHERE status='live-swap-settled'"))[0][0]
    except Exception:
        return 0


def snapshot() -> int:
    res = _run_tests()
    SNAP.parent.mkdir(parents=True, exist_ok=True)
    SNAP.write_text(json.dumps({
        "ts": time.time(),
        "tests": res,
        "settled": _settled_swaps(),
        "profit": _profit_numbers(),
    }, indent=2), encoding="utf-8")
    print("snapshot: %d passed, %d failed, %d settled swaps"
          % (res.get("passed", 0), res.get("failed", 0), _settled_swaps()))
    return 0


def check() -> int:
    try:
        before = json.loads(SNAP.read_text(encoding="utf-8"))
    except Exception:
        print("no snapshot to compare against; run --snapshot before the pass")
        return 0

    after = _run_tests()
    elapsed = time.time() - float(before.get("ts") or time.time())
    settled_now = _settled_swaps()
    settled_before = int(before.get("settled") or 0)
    gained = settled_now - settled_before

    was = (before.get("tests") or {}).get("outcomes") or {}
    now = after.get("outcomes") or {}
    regressions = sorted(t for t in now if t not in was)

    print("=" * 68)
    print("PASS GATE")
    print("=" * 68)
    print("tests   : %d passed, %d failed  (was %d passed, %d failed)"
          % (after.get("passed", 0), after.get("failed", 0),
             (before.get("tests") or {}).get("passed", 0),
             (before.get("tests") or {}).get("failed", 0)))
    print("swaps   : %+d settled this pass (total %d)" % (gained, settled_now))
    print("elapsed : %.1f min (budget %.0f min)"
          % (elapsed / 60.0, SPRINT_BUDGET_SEC / 60.0))
    print()

    verdict = 0
    if regressions:
        print("REJECTED -- this pass BROKE tests that were passing before it:")
        for t in regressions:
            print("    %s" % t)
        print()
        print("Fix these before anything else. A change that trades one broken")
        print("link for another is not progress, whatever else the pass did.")
        verdict = 1
    elif not after.get("ran"):
        print("INCONCLUSIVE -- the test run produced no counts. Treat as unproven.")
    else:
        print("OK -- nothing that was passing is broken.")

    # Constraint 3: did the numbers this work claims to move actually move?
    was_p = before.get("profit") or {}
    now_p = _profit_numbers()
    print("profit numbers (constraint 3):")
    for key, label in (("live_trades", "live trades"),
                       ("net_pl", "net P/L"),
                       ("profit_factor", "profit factor"),
                       ("stranded_positions", "stranded positions")):
        b = was_p.get(key, 0)
        a = now_p.get(key, 0)
        arrow = "->" if a != b else "=="
        flag = ""
        if key == "stranded_positions" and a > b:
            flag = "   WORSE: capital entered and did not come back"
        elif key in ("net_pl", "profit_factor") and a < b:
            flag = "   WORSE"
        print("    %-20s %-12s %s %-12s%s" % (label, b, arrow, a, flag))
    print()

    if gained <= 0 and elapsed > SPRINT_BUDGET_SEC:
        print()
        print("OVER BUDGET -- %.1f minutes with no settled swap. The objective is"
              % (elapsed / 60.0))
        print("live profitable trades, not analysis. Next pass: pick the shortest")
        print("path to one settled round trip and take it.")

    return verdict


def main() -> int:
    args = sys.argv[1:]
    if "--snapshot" in args:
        return snapshot()
    if "--check" in args:
        return check()
    sys.stderr.write(__doc__ or "")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
