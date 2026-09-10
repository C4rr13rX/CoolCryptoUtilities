"""Which tests take their verdict from LIVE MARKET DATA?

THE DEFECT THIS MEASURES. Three predicates decide at runtime whether a symbol
is tradeable, and all three answer from ``storage/trading_cache.db`` -- the
production tape:

    services.symbol_edge_gate.refusal_reason         the closed round-trip book
    services.symbol_motion_gate.refusal_reason       the price feed
    services.stop_survivability_gate.refusal_reason  the price feed
    trading.strategies.ledger._live_tradeable        asks the first and the third

A test that reaches one of them without patching it is not asserting on the
code: it is asserting on this week's market. Three such failures were found in
pass 109 by their damage rather than by reading, and the pass gate is the only
thing standing between a money-path regression and a commit.

TWO MODES, AND THE SECOND IS THE ONE THAT PROVES ANYTHING.

  --list   STATIC. Grep-with-a-parser: which test files name a predicate, and
           which of those install a patch over it. Cheap, and a LOWER BOUND --
           a test that calls ``ledger.record()`` reaches ``_live_tradeable``
           without naming it anywhere, so a file can be tape-dependent and
           invisible here.

  --prove  EMPIRICAL. Runs the suite twice on one tree -- once against the
           production database, once with every gate repointed at an EMPTY one
           -- and diffs the pass/fail SET. A file whose set moves is deciding
           on the tape, whatever its imports say. This is the criterion; the
           static list is the map that tells you where to look.

The empty arm is not a second fixture. All three gates FAIL OPEN by design, so
an empty database has a KNOWN correct verdict for every symbol (allow), which
makes any difference between the arms attributable to the data rather than to
the choice of replacement data. See ``scripts/_live_data_isolation.py``.

Exit code is 1 when --prove finds any test whose outcome moved, so this can be
wired into a gate.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

#: The names that mean "ask the production tape". Module basenames and the
#: ledger helper, matched against source text and against parsed imports.
PREDICATES = (
    "symbol_edge_gate",
    "symbol_motion_gate",
    "stop_survivability_gate",
    "_live_tradeable",
    "stop_is_unenforceable",
)

#: What counts as installing a patch: monkeypatch.setattr, mock.patch(...),
#: patch.object(...), or a plain module-attribute assignment in a fixture.
_PATCH_CALL = re.compile(
    r"(monkeypatch\.setattr|mock\.patch|patch\.object|patch\()", re.I
)


def _test_files() -> List[Path]:
    return sorted(p for p in TESTS.glob("test_*.py") if p.is_file())


def _patched_targets(source: str) -> Set[str]:
    """Every predicate name that appears inside a patching construct."""
    found: Set[str] = set()
    for line_no, line in enumerate(source.splitlines()):
        if not _PATCH_CALL.search(line):
            continue
        # A patch target is frequently split across lines; take the call and
        # the two lines after it, which covers every shape used in tests/.
        window = "\n".join(source.splitlines()[line_no:line_no + 3])
        for name in PREDICATES:
            if name in window:
                found.add(name)
    return found


def _referenced(source: str) -> Set[str]:
    return {name for name in PREDICATES if name in source}


def classify(path: Path) -> Dict[str, object] | None:
    source = path.read_text(encoding="utf-8", errors="replace")
    referenced = _referenced(source)
    if not referenced:
        return None
    patched = _patched_targets(source)
    unpatched = sorted(referenced - patched)
    try:
        ast.parse(source)
        parsed = True
    except SyntaxError:
        parsed = False
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "referenced": sorted(referenced),
        "patched": sorted(patched),
        "unpatched": unpatched,
        "reaches_live_data": bool(unpatched),
        "parsed": parsed,
    }


def run_list(as_json: bool) -> int:
    rows = [r for r in (classify(p) for p in _test_files()) if r]
    exposed = [r for r in rows if r["reaches_live_data"]]
    if as_json:
        print(json.dumps({"rows": rows, "exposed": len(exposed)}, indent=2))
        return 0
    print(f"{len(_test_files())} test files scanned")
    print(f"{len(rows)} reference a live-data predicate")
    print(f"{len(exposed)} reference one WITHOUT patching it\n")
    print(f"{'file':70s}  unpatched")
    for row in sorted(exposed, key=lambda r: r["file"]):
        print(f"{row['file']:70s}  {','.join(row['unpatched'])}")
    print("\nPATCHED (the shape to copy):")
    for row in sorted(rows, key=lambda r: r["file"]):
        if not row["reaches_live_data"]:
            print(f"  {row['file']:68s}  {','.join(row['patched'])}")
    print("\nThis list is a LOWER BOUND -- run --prove for the real answer.")
    return 0


def _outcomes(paths: List[str], isolate: bool) -> Tuple[Dict[str, str], str]:
    """node id -> outcome, by parsing pytest's own report lines."""
    cmd = [sys.executable, "-X", "utf8", "-m", "pytest", "-q", "--no-header",
           "-p", "no:randomly", "-rA", *paths]
    if isolate:
        cmd[6:6] = ["-p", "scripts._live_data_isolation"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=1800)
    out = (proc.stdout or "") + (proc.stderr or "")
    outcomes: Dict[str, str] = {}
    for line in out.splitlines():
        m = re.match(r"^(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\s+(\S+)", line.strip())
        if m:
            outcomes[m.group(2)] = m.group(1)
    return outcomes, out


def run_prove(paths: List[str], as_json: bool) -> int:
    targets = paths or [
        r["file"] for r in (classify(p) for p in _test_files())
        if r and r["reaches_live_data"]
    ]
    print(f"proving {len(targets)} file(s) -- two runs on one tree\n")
    live, live_out = _outcomes(targets, isolate=False)
    print(f"  live database : {len(live)} tests, "
          f"{sum(1 for v in live.values() if v in ('FAILED', 'ERROR'))} failed")
    empty, empty_out = _outcomes(targets, isolate=True)
    print(f"  empty database: {len(empty)} tests, "
          f"{sum(1 for v in empty.values() if v in ('FAILED', 'ERROR'))} failed")

    moved = {
        node: (live.get(node, "ABSENT"), empty.get(node, "ABSENT"))
        for node in sorted(set(live) | set(empty))
        if live.get(node, "ABSENT") != empty.get(node, "ABSENT")
    }
    if as_json:
        print(json.dumps({"moved": moved, "n_live": len(live),
                          "n_empty": len(empty)}, indent=2))
    else:
        print(f"\n{len(moved)} test(s) changed outcome with the tape:")
        for node, (a, b) in moved.items():
            print(f"  {node}\n      live={a}  empty={b}")
        if not moved:
            print("  (none -- the suite's verdict does not depend on the market)")
    if not live and not empty:
        print("\nNEITHER RUN REPORTED A TEST -- treating as a failure, not a pass.")
        print(live_out[-2000:])
        return 1
    return 1 if moved else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="static enumeration (lower bound)")
    ap.add_argument("--prove", action="store_true", help="two runs, diff the pass/fail set")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("paths", nargs="*", help="restrict --prove to these files")
    args = ap.parse_args()
    if args.prove:
        return run_prove(args.paths, args.json)
    return run_list(args.json)


if __name__ == "__main__":
    raise SystemExit(main())
