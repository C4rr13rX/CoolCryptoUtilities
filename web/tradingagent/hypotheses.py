"""Hypotheses generated from what the system refused, not from what it traded.

WHY REFUSALS ARE THE RICHER SOURCE
----------------------------------
Every learning path in this package reads closed round trips: what we entered,
what it returned. That sample is selected by the very gates being evaluated,
so it can only ever say how good our accepted trades were. It cannot say
whether the gates are right, because the trades they refused are not in it.

The refusals are. ``trading_ops`` carries a row for every blocked entry with
the reason attached -- entry-refused-lattice naming the layer that stopped it,
entry-refused-symbol-edge, guard-blocked, hold-negative. Those rows are a
record of what the system believes, and beliefs are testable:

  * If the lattice refuses long horizons on a symbol, and that symbol's
    long-horizon trades elsewhere in the book did fine, the horizon rule is
    costing us.
  * If a guard fires constantly on one symbol and that symbol's realised
    returns are ordinary, the guard is mis-calibrated for it.
  * If a refusal reason correlates with nothing, it is noise wearing the
    costume of a rule.

WHAT THIS GENERATES
-------------------
Falsifiable statements in the same ``Theorem`` shape everything else uses, so
a generated hypothesis goes through the identical fit/holdout machinery as a
hand-written one and earns no special standing. The generator proposes; the
holdout disposes.

WHAT IT WILL NOT DO
-------------------
It will not propose a hypothesis it cannot test. A claim about a refusal
reason with four occurrences is not a claim, it is an anecdote, and generating
it would fill the report with untestable noise that crowds out the real ones.
"""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "storage" / "trading_cache.db"

#: Occurrences a refusal reason needs before a hypothesis about it is worth
#: stating. Below this the sample cannot distinguish a rule from a run of luck.
MIN_REFUSALS = 12

#: Refusals whose direction is REVERSED by construction, and which therefore
#: cannot be judged by comparing returns.
#:
#: These fire BECAUSE a symbol is already being traded successfully, so the
#: symbols they guard look profitable no matter how good or bad the rule is.
#: The correlation is real and the causal reading is backwards.
#:
#: Found the hard way, 2026-09-05: the generator flagged
#: entry-refused-live-held as "guarding symbols that perform BETTER"
#: (+0.03817 against +0.02149) and it was one step from being acted on. But
#: that guard refuses an entry only when a LIVE position is ALREADY OPEN on
#: the symbol -- and live positions exist on our better symbols, because
#: those are the ones that graduated. The guard does not select profitable
#: symbols; their profitability is why they are held. Removing it would let
#: the bot double-buy a token it already owns, which is precisely the failure
#: it was written to stop.
#:
#: A hypothesis generator that cannot tell "this rule picks winners" from
#: "this rule fires on winners" will eventually talk someone into deleting a
#: guard that was working.
CONFOUNDED_BY_SELECTION = {
    # Fires only when a live position is already open: selection is the cause,
    # not the effect.
    "entry-refused-live-held",
    # Fires when the same strategy already holds the symbol -- same shape.
    "entry-refused-duplicate",
    # Fires when another strategy holds the slot; the slot is occupied
    # BECAUSE something found the symbol worth trading.
    "entry-refused-slot-busy",
}

#: Closed trades needed on the other side of the comparison.
MIN_COMPARISON = 12


def refusal_census(since_sec: float = 86400 * 7) -> Dict[str, Dict[str, Any]]:
    """What the system refused, and on which symbols.

    Grouped by reason, because a reason is a belief and that is the unit
    worth testing. A count alone would say how loud a rule is, not whether
    it is right.
    """
    out: Dict[str, Dict[str, Any]] = {}
    try:
        conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    except Exception:  # noqa: BLE001
        return out

    try:
        cutoff = time.time() - since_sec
        for status, symbol, details in conn.execute(
                "SELECT status, symbol, details FROM trading_ops "
                "WHERE ts > ? AND (status LIKE '%refused%' "
                "OR status LIKE '%blocked%' OR status LIKE 'hold-%')",
                (cutoff,)):
            reason = str(status or "")
            try:
                payload = json.loads(details) if details else {}
            except Exception:  # noqa: BLE001
                payload = {}
            # The detail is more specific than the status: entry-refused-
            # lattice says which LAYER refused, and the layers are different
            # beliefs that deserve separate tests.
            detail = str((payload or {}).get("detail") or "")
            layer = detail.split(":", 1)[0].strip() if detail else ""
            key = f"{reason}:{layer}" if layer else reason

            bucket = out.setdefault(key, {
                "reason": reason, "layer": layer, "count": 0,
                "symbols": {}, "examples": []})
            bucket["count"] += 1
            symbol = str(symbol or "")
            bucket["symbols"][symbol] = bucket["symbols"].get(symbol, 0) + 1
            if len(bucket["examples"]) < 3 and detail:
                bucket["examples"].append(detail[:160])
    except Exception:  # noqa: BLE001
        return out
    finally:
        conn.close()

    return out


def _symbol_returns(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[float]]:
    by_symbol: Dict[str, List[float]] = {}
    for row in rows:
        value = row.get("return")
        if isinstance(value, (int, float)):
            by_symbol.setdefault(str(row.get("symbol") or ""), []).append(float(value))
    return by_symbol


def generate(rows: Sequence[Dict[str, Any]],
             census: Optional[Dict[str, Dict[str, Any]]] = None,
             ) -> List[Dict[str, Any]]:
    """Hypotheses worth testing, each with the predicate that tests it.

    Returns dicts rather than Theorem objects so this module does not depend
    on the theorem layer -- the caller wraps them. Each carries the evidence
    that provoked it, so a reader can see WHY the question was asked.
    """
    census = census if census is not None else refusal_census()
    by_symbol = _symbol_returns(rows)
    proposals: List[Dict[str, Any]] = []

    for key, bucket in sorted(census.items(),
                              key=lambda kv: kv[1]["count"], reverse=True):
        count = int(bucket["count"])
        if count < MIN_REFUSALS:
            # Not a rule yet, just a few events. Saying so is more useful
            # than a hypothesis nobody can test.
            continue

        # The symbols this belief fires on most.
        hot = sorted(bucket["symbols"].items(), key=lambda kv: kv[1],
                     reverse=True)[:3]
        hot_names = [name for name, _ in hot if name]
        if not hot_names:
            continue

        # THE QUESTION: on the symbols this rule refuses, did the trades we
        # DID take there do worse than the book as a whole? If they did, the
        # rule is finding something real. If they did better, it is costing us.
        refused_returns: List[float] = []
        for name in hot_names:
            refused_returns.extend(by_symbol.get(name, []))
        others: List[float] = []
        for name, values in by_symbol.items():
            if name not in hot_names:
                others.extend(values)

        if len(refused_returns) < MIN_COMPARISON or len(others) < MIN_COMPARISON:
            proposals.append({
                "id": f"untestable:{key}",
                "statement": (
                    f"{key} fired {count} times, mostly on "
                    f"{', '.join(hot_names)} -- but those symbols have "
                    f"{len(refused_returns)} closed trades, too few to say "
                    f"whether the refusals were right."),
                "testable": False,
                "evidence": {"count": count, "symbols": hot_names,
                             "examples": bucket["examples"]},
            })
            continue

        mean_refused = statistics.mean(refused_returns)
        mean_others = statistics.mean(others)
        direction = "worse" if mean_refused < mean_others else "BETTER"

        # A refusal that fires BECAUSE the symbol is already being traded
        # cannot be judged this way: the comparison is confounded by the
        # selection that produced it. Report the numbers, refuse the verdict.
        if bucket["reason"] in CONFOUNDED_BY_SELECTION:
            proposals.append({
                "id": f"confounded:{key}",
                "statement": (
                    f"{key} fired {count} times on {', '.join(hot_names)}, "
                    f"which returned {mean_refused:+.5f} against "
                    f"{mean_others:+.5f} elsewhere -- but this rule fires "
                    f"only when the symbol is ALREADY being traded, so it "
                    f"appears on our better symbols by construction. The "
                    f"comparison cannot say whether the rule is right."),
                "testable": False,
                "confounded": True,
                "evidence": {
                    "count": count, "symbols": hot_names,
                    "mean_return_on_guarded": round(mean_refused, 6),
                    "mean_return_elsewhere": round(mean_others, 6),
                    "examples": bucket["examples"],
                },
            })
            continue

        proposals.append({
            "id": key,
            "statement": (
                f"{key} concentrates on {', '.join(hot_names)}. Trades taken "
                f"on those symbols returned {mean_refused:+.5f} against "
                f"{mean_others:+.5f} elsewhere -- the rule is guarding "
                f"symbols that perform {direction}."),
            "testable": True,
            # The predicate a Theorem needs: is this trade on one of the
            # symbols the rule concentrates on?
            "symbols": hot_names,
            "evidence": {
                "count": count,
                "mean_return_on_guarded": round(mean_refused, 6),
                "mean_return_elsewhere": round(mean_others, 6),
                "n_guarded": len(refused_returns),
                "n_elsewhere": len(others),
                "examples": bucket["examples"],
            },
            # A rule guarding symbols that do BETTER than the book is a rule
            # costing money, and that is the finding worth surfacing loudly.
            "suspicion": ("this guard may be refusing profitable trades"
                          if mean_refused > mean_others else
                          "the guard looks justified"),
        })

    return proposals


def as_theorems(proposals: Sequence[Dict[str, Any]]) -> List[Any]:
    """Wrap testable proposals in the Theorem the falsifier already uses.

    Untestable proposals are dropped here rather than earlier: they are worth
    REPORTING (they say where the book is too thin to judge a rule) and not
    worth testing, and those are different jobs.
    """
    from .theorems import Theorem

    out: List[Any] = []
    for proposal in proposals:
        if not proposal.get("testable"):
            continue
        symbols = set(proposal.get("symbols") or ())
        if not symbols:
            continue
        out.append(Theorem(
            name=proposal["id"][:60],
            statement=proposal["statement"],
            predicate=(lambda t, s=symbols: str(t.get("symbol") or "") in s),
            rationale=(
                "Generated from the refusal census: this asks whether a rule "
                "that fires constantly on these symbols is protecting us or "
                "costing us. " + str(proposal.get("suspicion") or "")),
        ))
    return out
