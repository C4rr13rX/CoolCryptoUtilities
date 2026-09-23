#!/usr/bin/env python
"""End-of-pass census: is this pass about to leave shipped work on the floor?

WHY THIS EXISTS. A test that never reached git is a rumour. It is invisible to
the pass gate, invisible to every concurrent agent sharing this one worktree,
and one `git clean` from gone. The defect has now recurred three times:

  * pass 114 -- seven money-path regression tests sat untracked for a week, and
    65 of 92 brain reports were gitignored, so every "commit
    data/brain_experiments/" instruction was a silent no-op.
  * pass 458 -- the whole `load_quote_census` / `--census` cost-table seam
    existed only as uncommitted working-tree changes from an earlier pass.
  * pass 680 -- `scripts/feed_rate_power.py` and
    `tests/test_a_ten_minute_window_cannot_see_a_two_tick_change.py`, both
    written 2026-09-11, still untracked twelve days later.

THE MECHANISM, named rather than guessed. It is NOT a failing hook and it is
NOT `git add` being mis-scoped. The standing order is to scope every commit
with `git commit -- <paths>`, because `git add -A` and `git stash` swallow a
concurrent agent's work. That order is correct and it has this exact blind
spot: **a path you do not name is a path that does not get committed, and git
says nothing about it.** `git commit -- a.py b.py` succeeds, prints a clean
numstat for the two files it was given, and never mentions the third file you
wrote. The agent reads a successful commit and reports the work shipped.

So the fix cannot be "add everything" -- that would reintroduce the problem the
scoping rule exists to prevent. The fix is a CHECK: before a pass ends, say out
loud which code files it is about to leave behind.

TWO FAILURE MODES, both reported, because they look identical in a report and
have different repairs:

  * UNTRACKED -- a new file git has never seen. This is the dangerous one: no
    other agent can see it, and it is lost to `git clean`.
  * MODIFIED -- a tracked file with uncommitted edits. Visible to others via
    `git status`, recoverable, but still not shipped.

AGE IS WHAT MAKES THIS ACTIONABLE. An untracked file written minutes ago is
normal work in progress. One older than a day is work a previous pass believed
it had shipped. Only the latter fails the census.

DELIBERATELY STANDALONE. This does not touch `scripts/pass_gate.py`, the most
contended file in the system (37 claims, 12 collisions). Wiring it in is one
line that one agent should add afterwards, announced on the board.

    python -X utf8 scripts/uncommitted_work_census.py
    python -X utf8 scripts/uncommitted_work_census.py --max-age-days 0 --json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Sequence

# Where shipped WORK lives. Data and backup artifacts are noise here -- this
# census is about code that a pass claimed it had landed.
CODE_DIRS: Sequence[str] = ("scripts", "tests", "trading", "services", "web")
CODE_SUFFIXES: Sequence[str] = (".py",)


def _git(args: Sequence[str], root: str) -> str:
    out = subprocess.run(
        ["git", *args],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {root}: {out.stderr.strip()}")
    return out.stdout


def _is_code(path: str) -> bool:
    if not path.endswith(tuple(CODE_SUFFIXES)):
        return False
    head = path.split("/", 1)[0]
    # Top-level modules (main.py, router_wallet.py) count too -- the 2026-09
    # money-path losses included them.
    return head in CODE_DIRS or "/" not in path


def _age_days(root: str, path: str, now: float) -> float:
    full = os.path.join(root, path.replace("/", os.sep))
    try:
        return max(0.0, (now - os.path.getmtime(full)) / 86400.0)
    except OSError:
        return 0.0


def census(root: str = ".", now: float | None = None) -> Dict[str, List[Dict]]:
    """Untracked and modified code files, each with its age in days."""
    now = time.time() if now is None else now

    untracked = [
        p for p in _git(
            ["ls-files", "--others", "--exclude-standard"], root
        ).splitlines() if p and _is_code(p)
    ]
    # HEAD may not exist in a fresh repo with no commits; an empty diff is the
    # honest answer there rather than a crash.
    try:
        modified = [
            p for p in _git(
                ["diff", "--name-only", "HEAD"], root
            ).splitlines() if p and _is_code(p)
        ]
    except RuntimeError:
        modified = []

    def rows(paths: List[str]) -> List[Dict]:
        return sorted(
            ({"path": p, "age_days": round(_age_days(root, p, now), 2)}
             for p in paths),
            key=lambda r: -r["age_days"],
        )

    return {"untracked": rows(untracked), "modified": rows(modified)}


def verdict(result: Dict[str, List[Dict]], max_age_days: float) -> List[Dict]:
    """The untracked files old enough to be work a past pass thought it shipped."""
    return [r for r in result["untracked"] if r["age_days"] > max_age_days]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument(
        "--max-age-days", type=float, default=1.0,
        help="untracked code older than this is work a previous pass believed "
             "it had shipped; 1 day lets the current pass's own work pass")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    result = census(args.root)
    stale = verdict(result, args.max_age_days)

    if args.json:
        print(json.dumps({**result, "stale": stale}, indent=2))
        return 1 if stale else 0

    print("=" * 72)
    print("UNCOMMITTED WORK CENSUS -- a test that never reached git is a rumour")
    print("=" * 72)
    print()
    if not result["untracked"]:
        print("  UNTRACKED code files: none")
    else:
        print(f"  UNTRACKED code files: {len(result['untracked'])}"
              "   <- invisible to every other agent, lost to `git clean`")
        for r in result["untracked"]:
            flag = "  STALE" if r["age_days"] > args.max_age_days else ""
            print(f"    {r['age_days']:8.2f}d  {r['path']}{flag}")
    print()
    if not result["modified"]:
        print("  MODIFIED tracked code files: none")
    else:
        print(f"  MODIFIED tracked code files: {len(result['modified'])}"
              "   <- visible and recoverable, but not shipped")
        for r in result["modified"]:
            print(f"    {r['age_days']:8.2f}d  {r['path']}")
    print()
    if stale:
        print(f"  FAIL: {len(stale)} untracked code file(s) older than "
              f"{args.max_age_days}d.")
        print("  This is work a previous pass believed it had shipped.")
        print("  Commit them SCOPED -- `git commit -F - -- <exact paths>`.")
        print("  NEVER `git add -A` and NEVER `git stash`: concurrent agents")
        print("  share this worktree and both swallow their work.")
        return 1
    print("  OK: nothing older than the current pass is left untracked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
