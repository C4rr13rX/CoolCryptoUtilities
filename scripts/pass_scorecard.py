"""Score what a pass actually achieved, and keep the behavioural prompt that
produced the best results.

Why this is measured rather than self-reported: an agent asked "how did you
do?" will say "well". Every number here is computed from the repo and the
chain -- commits that landed, tests that pass, whether the wallet nonce moved
-- so a pass cannot score itself well by claiming success. This repo has
already shipped four strategies whose entire records were fabricated; a
self-graded scorecard would be the same failure in a new place.

The score has four parts, each 0-25:

  PROGRESS   did the goal get closer? (settled trades, live P/L, ledger
             coverage, ghost throughput)
  CORRECTNESS do the tests pass, and did this pass break any that were
             passing before it?
  EVIDENCE   did it produce verifiable artefacts -- a real tx hash, a new
             test -- rather than assertions?
  EFFICIENCY did it commit anything at all, and did it avoid churn (a
             commit that only reverts or re-fixes the previous pass)?

Each pass appends one row to data/pass_scores.json. The behavioural prompt
in data/behavior_prompt.md is what the loop prepends to the next pass; when
a variant scores better over a window than the incumbent, it is promoted and
the change is recorded. That is the whole refinement loop: vary, measure,
keep what wins.

Usage:
    python scripts/pass_scorecard.py --score        # score the pass that just ended
    python scripts/pass_scorecard.py --report       # history and current prompt
    python scripts/pass_scorecard.py --prompt       # print the active prompt only
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
SCORES = ROOT / "data" / "pass_scores.json"
PROMPT = ROOT / "data" / "behavior_prompt.md"
DB = ROOT / "storage" / "trading_cache.db"

# The starting behavioural prompt. Deliberately short: it is a lever the loop
# tunes, not a place to restate the task.
SEED_PROMPT = """Work in this order: reproduce the problem with a measurement,
make the smallest change that fixes the root cause, prove it with a test, then
stop. Prefer one finished thing over three started ones."""

VARIANTS = [
    SEED_PROMPT,
    """Before writing code, state in one line what evidence would prove this
pass succeeded. Then go get that evidence first, and only afterwards make the
code tidy. If you cannot name the evidence, you do not yet understand the
problem.""",
    """Spend at most a third of this pass understanding and the rest changing
and proving. If you are still reading code halfway through, you are stuck --
pick the smallest reversible change that could work and try it.""",
    """Every claim in your report must carry the command that produced it. If
you cannot paste the output, do not make the claim. Numbers without a command
behind them are the failure mode that wasted weeks here.""",
    """Fix the thing that unblocks the most other things. When two problems
compete, choose the one whose fix makes the other measurable, and say why you
chose it.""",
]


def _run(cmd, timeout=None):
    """Run to completion. timeout=None means NO LIMIT, and that is the default.

    Nothing here should be killed on a clock. We cannot predict that a test
    run or a git query needs less than N seconds, and cutting one off does
    not fail safe -- it reports "did not run", which the scorer then has to
    treat as unknown, so the whole pass is scored on missing data. Waiting is
    always cheaper than discarding the work.
    """
    try:
        out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                             text=True, timeout=timeout, shell=False)
        return out.returncode, (out.stdout or "") + (out.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return 1, "%s: %s" % (type(exc).__name__, exc)


def _load(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _chain_state() -> dict:
    """Nonce and USDC, straight from the chain. The only unfalsifiable input."""
    import urllib.request

    w = "0x291c854811e92906a658Fb94Aa511bF919f968ad"
    usdc = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
    hdrs = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    for url in ("https://base-rpc.publicnode.com", "https://1rpc.io/base",
                "https://base.llamarpc.com"):
        try:
            def rpc(method, params):
                req = urllib.request.Request(
                    url,
                    data=json.dumps({"jsonrpc": "2.0", "id": 1,
                                     "method": method, "params": params}).encode(),
                    headers=hdrs)
                # A third-party RPC that never answers is not our work being
                # cut off: we fall through to the next endpoint below.
                with urllib.request.urlopen(req, timeout=15) as r:
                    return json.loads(r.read())["result"]

            nonce = int(rpc("eth_getTransactionCount", [w, "latest"]), 16)
            data = "0x70a08231" + "0" * 24 + w[2:].lower()
            bal = int(rpc("eth_call", [{"to": usdc, "data": data}, "latest"]), 16) / 1e6
            return {"nonce": nonce, "usdc": round(bal, 6)}
        except Exception:
            continue
    return {}


def _db_state() -> dict:
    import sqlite3

    out = {"tx_hashes": 0, "settled": 0, "ghost_1h": 0}
    try:
        c = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
        now = time.time()
        rows = list(c.execute(
            "SELECT details FROM trading_ops WHERE status LIKE 'live%' LIMIT 500"))
        pat = re.compile(r"0x[0-9a-fA-F]{64}")
        out["tx_hashes"] = sum(1 for (d,) in rows if pat.search(str(d)))
        out["settled"] = list(c.execute(
            "SELECT COUNT(*) FROM trading_ops WHERE status LIKE 'live%' "
            "AND status NOT LIKE '%blocked%' AND status NOT LIKE '%dry-run%'"))[0][0]
        out["ghost_1h"] = list(c.execute(
            "SELECT COUNT(*) FROM trading_ops WHERE status='ghost-entry' AND ts>?",
            (now - 3600,)))[0][0]
    except Exception:
        pass
    return out


def _tests() -> dict:
    """Whole-suite pass/fail. A pass that breaks a green test is a bad pass."""
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    exe = str(py) if py.exists() else sys.executable
    # No --timeout: pytest-timeout is not installed here, and passing it makes
    # pytest exit with a usage error that reads as a failing suite -- scoring a
    # green repo as broken. The subprocess timeout below is the real guard.
    # No -x either: we want the true failure COUNT, not the first failure.
    # Scope: the tests that guard the money path, not the whole suite.
    #
    # The full suite takes >2 minutes here and pulls in TensorFlow, so running
    # it after every pass both stalls the loop and times out the subprocess,
    # which then reads as "did not run". These are the files that actually
    # pin the behaviour a pass can break -- swap routing, the density gate,
    # the ledger, graduation -- and they run in seconds.
    targets = [str(t) for t in sorted((ROOT / "tests").glob("test_*.py"))
               if any(k in t.name for k in (
                   "swap", "density", "ledger", "ghost", "graduat",
                   "money_button", "artifact", "corrobor", "gas"))]
    if not targets:
        targets = [str(ROOT / "tests")]
    rc, out = _run([exe, "-m", "pytest", *targets, "-q", "--no-header"])
    m = re.search(r"(\d+) passed", out)
    f = re.search(r"(\d+) failed", out)
    e = re.search(r"(\d+) error", out)
    passed = int(m.group(1)) if m else 0
    failed = (int(f.group(1)) if f else 0) + (int(e.group(1)) if e else 0)
    # A run that produced no counts at all did not run -- treat that as unknown
    # rather than as a clean pass, so a broken invocation cannot score well.
    if not m and not f and not e:
        return {"passed": 0, "failed": 0, "ran": False}
    return {"passed": passed, "failed": failed, "ran": True}


def _commits_since(ts: float) -> list:
    rc, out = _run(["git", "log", "--since=@%d" % int(ts), "--format=%h|%s"])
    if rc != 0:
        return []
    return [l for l in out.strip().split("\n") if l.strip()]


def score_pass() -> dict:
    history = _load(SCORES, [])
    prev = history[-1] if history else {}
    since = float(prev.get("ts") or (time.time() - 3600))

    chain = _chain_state()
    db = _db_state()
    tests = _tests()
    commits = _commits_since(since)

    p_chain = prev.get("chain") or {}
    p_db = prev.get("db") or {}
    p_tests = prev.get("tests") or {}

    # ---- PROGRESS: did the goal move? -----------------------------------
    progress = 0
    if chain.get("nonce") and p_chain.get("nonce"):
        moved = chain["nonce"] - p_chain["nonce"]
        progress += min(15, moved * 5)          # settled txs are the goal
    if db.get("settled", 0) > p_db.get("settled", 0):
        progress += 6
    if db.get("tx_hashes", 0) > p_db.get("tx_hashes", 0):
        progress += 4                            # recording caught up
    progress = max(0, min(25, progress))

    # ---- CORRECTNESS: did it break anything? ----------------------------
    correctness = 25
    if not tests.get("ran", True):
        correctness = 10          # unknown is not the same as good
    if tests["failed"]:
        correctness -= min(25, tests["failed"] * 8)
    # Regressing a previously-green suite is the worst outcome here.
    if p_tests and tests["failed"] > p_tests.get("failed", 0):
        correctness -= 10
    correctness = max(0, min(25, correctness))

    # ---- EVIDENCE: verifiable artefacts ---------------------------------
    evidence = 0
    if db.get("tx_hashes", 0):
        evidence += 12
    if tests["passed"] > p_tests.get("passed", 0):
        evidence += 8                            # a new test was added
    if commits:
        evidence += 5
    evidence = max(0, min(25, evidence))

    # ---- EFFICIENCY: did it land work without churn ---------------------
    efficiency = 0
    if commits:
        efficiency += 15
        churn = sum(1 for c in commits
                    if re.search(r"revert|undo|re-?fix|actually", c, re.I))
        efficiency -= min(15, churn * 5)
        efficiency += 10 if len(commits) <= 4 else 4   # focus beats volume
    efficiency = max(0, min(25, efficiency))

    total = progress + correctness + evidence + efficiency
    row = {
        "ts": time.time(),
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": (PROMPT.read_text(encoding="utf-8").strip()
                   if PROMPT.exists() else SEED_PROMPT),
        "score": total,
        "parts": {"progress": progress, "correctness": correctness,
                  "evidence": evidence, "efficiency": efficiency},
        "chain": chain, "db": db, "tests": tests,
        "commits": len(commits),
    }
    history.append(row)
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    SCORES.write_text(json.dumps(history[-200:], indent=2), encoding="utf-8")
    return row


def refine() -> str:
    """Keep the prompt with the best recent average; try a new one otherwise.

    Deliberately simple: each variant must earn at least MIN_TRIALS passes
    before it can be judged, so a single lucky pass cannot lock in a prompt.
    """
    MIN_TRIALS = 3
    history = _load(SCORES, [])
    if not PROMPT.exists():
        PROMPT.parent.mkdir(parents=True, exist_ok=True)
        PROMPT.write_text(SEED_PROMPT, encoding="utf-8")
        return SEED_PROMPT

    current = PROMPT.read_text(encoding="utf-8").strip()
    by_prompt = {}
    for row in history:
        by_prompt.setdefault(row.get("prompt", ""), []).append(row.get("score", 0))

    tried = by_prompt.get(current, [])
    if len(tried) < MIN_TRIALS:
        return current                       # not enough evidence yet

    scored = {p: sum(v) / len(v) for p, v in by_prompt.items() if len(v) >= MIN_TRIALS}
    untried = [v for v in VARIANTS if len(by_prompt.get(v, [])) < MIN_TRIALS]

    if untried:
        nxt = untried[0]                     # explore
    else:
        nxt = max(scored, key=scored.get)    # exploit the best measured

    if nxt.strip() != current:
        PROMPT.write_text(nxt.strip(), encoding="utf-8")
    return nxt.strip()


def report() -> None:
    history = _load(SCORES, [])
    if not history:
        print("no passes scored yet")
        return
    print("=== last 12 passes ===")
    print("%-20s %5s  %-4s %-4s %-4s %-4s  %s" %
          ("when", "score", "prog", "corr", "evid", "effi", "commits"))
    for r in history[-12:]:
        p = r.get("parts", {})
        print("%-20s %5d  %-4d %-4d %-4d %-4d  %d" %
              (r.get("when", "?"), r.get("score", 0), p.get("progress", 0),
               p.get("correctness", 0), p.get("evidence", 0),
               p.get("efficiency", 0), r.get("commits", 0)))
    by = {}
    for r in history:
        by.setdefault(r.get("prompt", ""), []).append(r.get("score", 0))
    print()
    print("=== behavioural prompts, by measured average ===")
    for p, v in sorted(by.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print("  avg %5.1f over %2d pass(es): %s" %
              (sum(v) / len(v), len(v), p.replace("\n", " ")[:80]))


def main() -> int:
    args = sys.argv[1:]
    if "--prompt" in args:
        if not PROMPT.exists():
            PROMPT.parent.mkdir(parents=True, exist_ok=True)
            PROMPT.write_text(SEED_PROMPT, encoding="utf-8")
        print(PROMPT.read_text(encoding="utf-8").strip())
        return 0
    if "--report" in args:
        report()
        return 0
    if "--score" in args:
        row = score_pass()
        nxt = refine()
        p = row["parts"]
        print("pass scored %d/100  (progress %d, correctness %d, evidence %d, efficiency %d)"
              % (row["score"], p["progress"], p["correctness"], p["evidence"], p["efficiency"]))
        print("chain: %s   db: %s   tests: %s" % (row["chain"], row["db"], row["tests"]))
        print()
        print("behavioural prompt for the next pass:")
        print("  " + nxt.replace("\n", "\n  "))
        return 0
    sys.stderr.write(__doc__ or "")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
