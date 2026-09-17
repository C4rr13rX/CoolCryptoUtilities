"""Find backlog items that are the SAME defect filed under different ids.

WHY THIS EXISTS. Pass 438 measured the backlog at 107 todo items, 53 of them
priority 1, and then read them: five separate ids described one defect
(``roundtrip_cost_rate`` takes no symbol), and four described another (the test
suite writes into ``logs/system.log``). One of those five was filed by the
senior in the same hour he rejected three others for being duplicates.

The cause is structural, not careless. Every pass re-derives the same finding
from the same data, and filing reads as output while searching does not. The
consequence is worse than a long list: a fix gets split across ids so no single
one is finishable, the diagnosis is orphaned on whichever id is rejected, and an
agent handed "the top p1 item" is usually handed a duplicate rather than the
most important thing.

So: grep before ``scrum.py add``. This does the grep, over every open item at
once, and it scores on the tokens that actually identify a defect here --
function names, file paths, dotted call sites -- not on prose, because two items
can describe one bug in completely different English (``the entry gate grades
every candidate with a cost model blind to the token`` and ``services/
roundtrip_cost.py takes no symbol`` are the same item).

READ-ONLY. It never writes to the backlog. It prints clusters; a human decides
which id keeps the work, moves the criteria there with ``scrum.py accept``, and
collapses the rest with ``scrum.py reject --note "duplicate of <id>"``.

    python scripts/backlog_duplicates.py
    python scripts/backlog_duplicates.py --min-score 0.25 --include-done
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

DEFAULT_BACKLOG = Path("D:/Projects/ContinuousRefinement/data/backlog-revenir.json")

# Tokens that identify a DEFECT rather than a sentence. A defect here is almost
# always named by the thing it lives in: a function, a module path, a call site.
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
PATHY = re.compile(r"[A-Za-z_][\w./\\-]*\.py\b")
# Backlog ids are 8 hex digits. An item quoting one is referring to it.
ID_TOKEN = re.compile(r"^[0-9a-f]{8}$")

# Words that appear in nearly every item on this board and therefore separate
# nothing. Without this, "the", "trade" and "ghost" alone would cluster the
# whole backlog into one blob.
STOP = {
    "this", "that", "with", "from", "have", "been", "which", "when", "what",
    "does", "them", "then", "than", "into", "over", "only", "every", "there",
    "because", "would", "could", "should", "still", "must", "same", "other",
    "more", "most", "less", "than", "item", "items", "pass", "passes", "test",
    "tests", "measure", "measured", "report", "reports", "reported", "number",
    "numbers", "code", "file", "files", "line", "lines", "name", "named",
    "trade", "trades", "trading", "ghost", "live", "strategy", "strategies",
    "data", "value", "values", "check", "checks", "board", "work", "done",
    "fix", "fixed", "make", "made", "take", "takes", "taken", "says", "said",
    "real", "here", "none", "whether", "before", "after", "each", "both",
}


def tokens(item: dict) -> set[str]:
    """Distinctive tokens for one item: identifiers and file paths, not prose."""
    text = " ".join(
        str(item.get(k) or "")
        for k in ("title", "description")
    )
    text += " " + " ".join(str(a) for a in (item.get("acceptance") or []))
    text += " " + " ".join(str(f) for f in (item.get("files") or []))

    out: set[str] = set()
    for m in PATHY.findall(text):
        out.add(m.replace("\\", "/").lower())
    for m in IDENT.findall(text):
        low = m.lower()
        if low in STOP or low.isdigit():
            continue
        # An item that CITES another item's id is a cross-reference, not a
        # duplicate of it -- and a well-written backlog is full of them. Found
        # live: [2a87591b] scored against [c4f16946] largely because its text
        # names "c4f16946". Drop backlog-id-shaped tokens.
        if ID_TOKEN.match(low):
            continue
        # A bare English word is weak evidence; a snake_case or dotted name is
        # strong, because nobody writes `roundtrip_cost_rate` by coincidence.
        out.add(low)
    return out


def weight(tok: str, df: dict[str, int], n: int) -> float:
    """Rarer tokens carry more. A token in half the backlog identifies nothing."""
    d = df.get(tok, 1)
    if d >= max(3, n // 8):
        return 0.0
    strong = ("_" in tok) or ("." in tok) or ("/" in tok)
    return (2.0 if strong else 1.0) / d


def score(a: set[str], b: set[str], df, n) -> tuple[float, list[str]]:
    shared = a & b
    if not shared:
        return 0.0, []
    num = sum(weight(t, df, n) for t in shared)
    den = sum(weight(t, df, n) for t in (a | b)) or 1.0
    ranked = sorted(shared, key=lambda t: -weight(t, df, n))
    return num / den, [t for t in ranked if weight(t, df, n) > 0][:6]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backlog", type=Path, default=DEFAULT_BACKLOG)
    # 0.12 was CALIBRATED, not guessed: at this threshold the finder
    # independently recovers 3 of the 4 log-contamination ids and 3 of the 5
    # roundtrip_cost_rate ids that pass 438 collapsed by hand, without being
    # told which they were. 0.18 found one cluster in the whole open backlog
    # and missed both known families. Recall is partial by design -- this is
    # triage that makes a human read two items, not a verdict.
    ap.add_argument("--min-score", type=float, default=0.12)
    ap.add_argument("--include-done", action="store_true",
                    help="also consider done/rejected items (finds work already closed)")
    args = ap.parse_args()

    if not args.backlog.exists():
        print("no backlog at %s" % args.backlog)
        return 2

    data = json.loads(args.backlog.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data

    open_states = {"todo", "doing", "in_progress", "blocked", "reopened"}
    pool = [
        it for it in items
        if args.include_done or str(it.get("status", "")).lower() in open_states
    ]
    if len(pool) < 2:
        print("fewer than two candidate items; nothing to compare")
        return 0

    toks = {it["id"]: tokens(it) for it in pool}
    df: dict[str, int] = {}
    for s in toks.values():
        for t in s:
            df[t] = df.get(t, 0) + 1
    n = len(pool)

    by_id = {it["id"]: it for it in pool}
    pairs = []
    for x, y in itertools.combinations(sorted(toks), 2):
        s, why = score(toks[x], toks[y], df, n)
        if s >= args.min_score and why:
            pairs.append((s, x, y, why))
    pairs.sort(reverse=True)

    # Transitive clusters: A~B and B~C puts all three in one group, which is how
    # the roundtrip_cost_rate family actually presented -- no single pair held
    # all five.
    parent: dict[str, str] = {}

    def find(k: str) -> str:
        parent.setdefault(k, k)
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for _s, x, y, _w in pairs:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    groups: dict[str, list[str]] = {}
    for k in list(parent):
        groups.setdefault(find(k), []).append(k)
    groups = {k: v for k, v in groups.items() if len(v) > 1}

    print("=" * 74)
    print("PROBABLE DUPLICATE CLUSTERS -- %d open items, %d clusters, min-score %.2f"
          % (n, len(groups), args.min_score))
    print("=" * 74)
    if not groups:
        print("\nnone above threshold. Lower --min-score to look harder.")
        return 0

    why_of = {(x, y): w for _s, x, y, w in pairs}
    for gid, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        members.sort(key=lambda i: (by_id[i].get("priority", 9),
                                    -int(by_id[i].get("points", 0) or 0)))
        print("\n--- %d items ---" % len(members))
        for i in members:
            it = by_id[i]
            print("  [%s] p%s %spt %-8s %s"
                  % (i, it.get("priority", "?"), it.get("points", "?"),
                     it.get("status", "?"), str(it.get("title", ""))[:64]))
        shared = set.intersection(*(toks[i] for i in members))
        strong = sorted((t for t in shared if weight(t, df, n) > 0),
                        key=lambda t: -weight(t, df, n))[:8]
        if strong:
            print("  shared: %s" % ", ".join(strong))
        else:
            for (x, y), w in why_of.items():
                if x in members and y in members:
                    print("  linked by: %s" % ", ".join(w))
                    break
        keep = members[0]
        print("  SUGGEST: keep [%s] (highest priority), move criteria there with" % keep)
        print("           scrum.py accept %s --set ... , then for each other id:" % keep)
        print("           scrum.py reject <id> --note \"duplicate of %s, criteria moved there\""
              % keep)

    print("\nThis is a SUGGESTION, not a verdict -- two items can share a function")
    print("and still be different defects. Read them before collapsing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
