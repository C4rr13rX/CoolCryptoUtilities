#!/usr/bin/env python3
"""
Make the attempts ledger READABLE, because right now it is not.

``data/attempts-revenir.md`` exists so that a pass can check whether its
hypothesis has already been tried and failed. Measured 2026-09-10 it cannot
serve that purpose: 4,242 lines carrying 25 date-anchored entries, one of them
spanning more than 1,700 lines. Every observed read is a ``tail`` -- ``tail
-60``, ``tail -30``, ``tail -4`` -- which reaches the last ~1% of the file and
usually lands inside the middle of a single entry.

So the ledger is write-mostly: 62 appends against ~33 reads in one window, and
the reads cannot see the writes. The predicted consequence is visible in the
file itself -- "shape" appears in 13 separate entries across passes 93-107,
"held-out" in 6, "confidence" in 8 -- and pass 107 records two passes reaching
OPPOSITE conclusions about the same pair without either citing the other.

This does not rewrite the ledger. It reads it and prints ONE LINE PER ENTRY:
date, author, and the first sentence of the hypothesis, plus whichever RESULT
line the entry recorded. That is the thing a pass actually needs before
forming a hypothesis, and it fits in a prompt.

    python scripts/attempts_index.py                  # every entry, one line
    python scripts/attempts_index.py --grep shape     # entries mentioning it
    python scripts/attempts_index.py --recent 12      # the last 12
    python scripts/attempts_index.py --stats          # topic recurrence

Read this BEFORE forming a hypothesis. It is cheap; the file is not.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data" / "attempts-revenir.md"

#: A new entry starts on a line beginning with a date. Everything until the
#: next such line belongs to it, however many lines that is.
ENTRY_RE = re.compile(r"^(20\d\d-\d\d-\d\d)\s*(.*)$")

#: Lines that carry the entry's outcome rather than its narrative. Matched
#: case-insensitively at the start of a stripped line.
RESULT_PREFIXES = ("RESULT", "VERDICT", "MEASURED", "next:", "NEXT")


def parse(path: Path) -> list:
    """Every entry as {date, head, body, result}. Never raises on a bad file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print("attempts_index: cannot read %s: %s" % (path, exc), file=sys.stderr)
        return []

    entries, current = [], None
    for line in text.splitlines():
        m = ENTRY_RE.match(line)
        if m:
            if current:
                entries.append(current)
            current = {"date": m.group(1), "head": m.group(2).strip(), "body": []}
        elif current is not None:
            current["body"].append(line)
    if current:
        entries.append(current)

    for e in entries:
        e["lines"] = len(e["body"]) + 1
        e["result"] = _first_result(e["body"])
        e["author"] = _author(e["head"])
        e["text"] = (e["head"] + " " + " ".join(e["body"])).lower()
    return entries


def _author(head: str) -> str:
    """'Cove pass 106 -- BRAIN...' -> 'Cove'. Best effort, never wrong loudly."""
    m = re.match(r"[\s|]*([A-Z][a-z]{2,9})\b", head)
    return m.group(1) if m else "?"


def _first_result(body: list) -> str:
    for line in body:
        s = line.strip()
        for prefix in RESULT_PREFIXES:
            if s.upper().startswith(prefix.upper()):
                return s[:160]
    return ""


def _summary(entry: dict, width: int = 96) -> str:
    """The first sentence of the hypothesis, which is what identifies it."""
    head = entry["head"]
    # Strip the author and any pass marker; keep what the entry is ABOUT.
    body = re.sub(r"^[\s|]*[A-Z][a-z]{2,9}\b[^|]*\|?", "", head).strip(" |-")
    if not body:
        for line in entry["body"]:
            if line.strip():
                body = line.strip()
                break
    body = re.sub(r"\s+", " ", body)
    return body[:width]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(DEFAULT))
    ap.add_argument("--grep", help="only entries whose text contains this (case-insensitive)")
    ap.add_argument("--recent", type=int, help="only the last N entries")
    ap.add_argument("--stats", action="store_true", help="topic recurrence across entries")
    ap.add_argument("--full", help="print the whole body of the entry at this index")
    args = ap.parse_args()

    entries = parse(Path(args.file))
    if not entries:
        print("attempts_index: no dated entries found. The ledger's entries must "
              "START a line with a YYYY-MM-DD date, or nothing can index them.")
        return 1

    if args.full is not None:
        try:
            e = entries[int(args.full)]
        except (ValueError, IndexError):
            print("attempts_index: --full takes an index from the listing")
            return 1
        print("%s %s" % (e["date"], e["head"]))
        print("\n".join(e["body"]))
        return 0

    if args.stats:
        return _stats(entries)

    shown = entries
    if args.grep:
        needle = args.grep.lower()
        shown = [e for e in shown if needle in e["text"]]
    if args.recent:
        shown = shown[-args.recent:]

    total_lines = sum(e["lines"] for e in entries)
    print("%d entries over %d lines in %s"
          % (len(entries), total_lines, Path(args.file).name))
    if args.grep:
        print("  %d mention %r" % (len(shown), args.grep))
    print()
    print("  %-4s %-11s %-8s %5s  %s" % ("#", "date", "who", "lines", "hypothesis"))
    for i, e in enumerate(entries):
        if e not in shown:
            continue
        print("  %-4d %-11s %-8s %5d  %s"
              % (i, e["date"], e["author"][:8], e["lines"], _summary(e)))
        if e["result"]:
            print("       %s" % e["result"][:110])
    print()
    print("  --full N prints one entry whole. Read the INDEX before forming a")
    print("  hypothesis; a tail of this file lands inside one entry's middle.")
    return 0


def _stats(entries: list) -> int:
    """Which topics keep coming back? Recurrence is the re-exploration signal."""
    topics = ("shape", "held-out", "confidence", "regime", "distinctness",
              "calibrat", "graduat", "tradeable", "stale", "cost", "direction",
              "pool", "metacognit", "temporal", "overshoot", "starv")
    print("TOPIC RECURRENCE -- an entry counted once however often it says the word")
    print("A topic in many entries has been re-explored; check those before repeating it.")
    print()
    rows = []
    for t in topics:
        hits = [i for i, e in enumerate(entries) if t in e["text"]]
        if hits:
            rows.append((len(hits), t, hits))
    for n, t, hits in sorted(rows, reverse=True):
        print("  %-14s %2d entries  #%s" % (t, n, ",".join(str(h) for h in hits[:12])))
    print()
    print("  --grep <topic> lists them; --full N reads one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
