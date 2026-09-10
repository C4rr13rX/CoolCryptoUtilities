"""
The graduation status must name WHICH wall the funnel stands at, and must judge
on the population the LEDGER judges on.

Two bugs are pinned here, and the second one shipped.

1. A status command that reports "live-approved: 0" and stops. Zero approvals is
   true whether nothing has enough evidence yet, or strategies clear the bar and
   were never stamped, or the best book belongs to a strategy that structurally
   cannot spend it. Those need opposite work, and a loop that cannot tell them
   apart burns passes on the wrong one.

2. Reading the POOLED ghost book instead of ``_tradeable_of(ghost)``. Graduation
   counts only round trips the live lane could actually have placed; the pooled
   book includes symbols it refuses on sight. Measured 2026-09-10 on the real
   ledger the two disagree by a factor of ten AND by sign:

       atf_static_scout   pooled 236 trades / 79% / +6.4818
                       tradeable   3 trades / 33% / -0.0777
       ledger totals      pooled 410 trades / +6.8134
                       tradeable  23 trades / -0.8940

   A status built on the pooled number called two strategies "ready" and named
   the wall UNSTAMPED, sending the pass to fix a stamp that was working exactly
   as designed. The bar was never met.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import graduation_status as gs  # noqa: E402


def _report(strategies, links=None, min_trades=20):
    """Build the shape collect() produces, without touching the real ledger."""
    ranked = sorted(
        strategies,
        key=lambda s: (
            0 if s.get("ready") else 1,
            max(0, min_trades - int(s.get("ghost_trades", 0))),
            -float(s.get("ghost_profit", 0.0)),
        ),
    )
    approved = [s["id"] for s in strategies if s.get("live_approved")]
    ready_unstamped = [
        s["id"] for s in strategies if s.get("ready") and not s.get("live_approved")
    ]
    return {
        "criteria": {"min_trades": min_trades, "min_winrate": 0.55, "min_profit": 0.0},
        "totals": {"live_approved": len(approved)},
        "approved": approved,
        "ready_not_approved": ready_unstamped,
        "blocked": [
            {"id": s["id"], "why": s.get("blocked_reason", "no live branch")}
            for s in strategies
            if s.get("structurally_blocked")
        ],
        "demoted": [
            {"id": s["id"], "times": s.get("demotions", 0), "why": s.get("demote_reason", "")}
            for s in strategies
            if s.get("demote_reason") and int(s.get("demotions", 0) or 0) > 0
        ],
        "closest": [
            {
                "id": s["id"],
                "ghost_trades": s["ghost_trades"],
                "pooled_trades": s.get("pooled_trades", s["ghost_trades"]),
                "win_rate": s.get("win_rate", 0.0),
                "ghost_profit": s.get("ghost_profit", 0.0),
                "blockers": s.get("blockers", []),
            }
            for s in ranked
        ],
        "live_path": links or [],
        "first_failing_link": next(
            (l for l in (links or []) if l.get("ok") is not True), None
        ),
        "errors": {},
    }


def _strategy(sid, trades, pooled=None, win=0.7, profit=1.0, ready=None,
              approved=False, blockers=(), blocked=False, demotions=0,
              demote_reason=""):
    return {
        "id": sid,
        "ghost_trades": trades,
        "pooled_trades": pooled if pooled is not None else trades,
        "win_rate": win,
        "ghost_profit": profit,
        "ready": (not blockers) if ready is None else ready,
        "live_approved": approved,
        "blockers": list(blockers),
        "structurally_blocked": blocked,
        "blocked_reason": demote_reason if blocked else "",
        "demotions": demotions,
        "demote_reason": demote_reason,
    }


def _wall(report, min_trades=20):
    return gs.classify_wall(
        approved=report["approved"],
        ready_not_approved=report["ready_not_approved"],
        ranked=report["closest"],
        min_trades=min_trades,
        blocked=report["blocked"],
        demoted=report["demoted"],
    )


def test_a_big_pooled_book_on_a_blocked_strategy_is_not_progress():
    """The real 2026-09-10 state: the biggest book can never be spent.

    atf_static_scout had 236 pooled ghost trades at 79% and +6.4818, and a
    ghost-only executor -- no live branch exists, so none of it can become a
    trade. Its TRADEABLE book is 3 trades at -0.0777.

    Against the old behaviour (pooled book, no structural check) this read as
    "READY BUT UNSTAMPED" and sent the pass to fix the graduation stamp. The
    stamp was correct; the evidence was unspendable.
    """
    strategies = [
        _strategy("atf_static_scout", 3, pooled=236, win=0.33, profit=-0.0777,
                  blocked=True, demote_reason="ghost-only executor: no live branch exists",
                  blockers=["structurally blocked: no live branch exists"]),
        _strategy("rsi_reversal", 6, pooled=19, win=0.0, profit=-0.4453,
                  blockers=["needs 14 more TRADEABLE ghost trades"]),
    ]
    report = _report(strategies)

    assert report["ready_not_approved"] == [], "nothing clears the bar on tradeable evidence"

    wall = _wall(report)
    # A pooled book does NOT buy the top of the precedence order. The scout's
    # 236 pooled trades sit behind 3 TRADEABLE ones and zero rows in
    # trade_outcomes, so no de-contamination instrument in this repo can see
    # them; classify_wall ranks the blocked branch on tradeable trades for
    # exactly that reason ("a book that cannot be audited must not outrank one
    # that can"). At 3 against min_trades the branch must not fire, and the
    # wall falls through to the one the ledger actually has.
    assert wall.startswith("EVIDENCE (TRADEABLE)"), wall
    assert "rsi_reversal" in wall
    # Both original failures being pinned, and they still bite: never send the
    # pass at the graduation stamp, and never let the pooled book read as the
    # top wall on the strength of its size alone.
    assert "UNSTAMPED" not in wall
    assert not wall.startswith("STRUCTURALLY BLOCKED"), (
        "236 unauditable pooled trades must not outrank a spendable record")


# NOT A TEST YET, DELIBERATELY -- see [db03ebf6] in the backlog.
#
# The other side of this rule is unpinned: a blocked strategy that has EARNED
# >= min_trades TRADEABLE trades and still cannot spend them. Jet probed it at
# pass 107 and classify_wall reports "QUALITY -- strategies have the tradeable
# trades but not the win rate/P-L", because the STRUCTURALLY BLOCKED branch is
# guarded by `near.ghost_trades < min_trades` and `near` is ranked[0], which is
# the blocked strategy itself. Calling that a QUALITY wall looks wrong -- the
# record is real and auditable, and the reason it cannot be spent is structural,
# not a win rate -- and it would route a pass at the strategy's P/L instead of
# at its missing live branch. It is left unasserted rather than pinned because
# fixing it means changing the precedence order in production code, which needs
# its own pass and its own before/after numbers. Do not add an assertion here
# that simply records the current QUALITY answer: that would enshrine the
# behaviour the item exists to question.


def test_the_bar_is_read_on_tradeable_trades_not_pooled_ones():
    """A strategy past the bar on pooled volume and short on tradeable is NOT ready."""
    strategies = [
        _strategy("atf_static", 4, pooled=52, win=0.50, profit=-0.0187,
                  blockers=["needs 16 more TRADEABLE ghost trades"]),
    ]
    report = _report(strategies)
    assert report["ready_not_approved"] == []

    wall = _wall(report)
    assert wall.startswith("EVIDENCE (TRADEABLE)"), wall
    # Both numbers must appear, because the GAP is the finding: a pass that
    # sees only "4/20" may go hunting for a dead ghost harness that is in fact
    # producing 52 trades, none of them in symbols the live lane will take.
    assert "4/20" in wall
    assert "52 pooled" in wall


def test_ready_but_unstamped_still_reported_when_tradeable_evidence_clears():
    """The UNSTAMPED wall is real -- it just needs TRADEABLE evidence to trigger."""
    strategies = [
        _strategy("some_strategy", 25, pooled=30, win=0.72, profit=1.20),
    ]
    report = _report(strategies)
    assert report["ready_not_approved"] == ["some_strategy"]

    wall = _wall(report)
    assert wall.startswith("READY BUT UNSTAMPED"), wall
    assert "TRADEABLE" in wall


def test_a_structural_block_does_not_count_as_a_demotion():
    """graduation_blocked writes demote_reason, but nothing was demoted.

    Listing it among demotions shows "x0" -- a demotion that never happened,
    next to real ones -- and would have a pass reading the re-arm rule for a
    strategy that never reached it.
    """
    strategies = [
        _strategy("atf_static_scout", 3, pooled=236, blocked=True, demotions=0,
                  demote_reason="ghost-only executor: no live branch exists",
                  blockers=["structurally blocked"]),
        _strategy("atf_static", 4, pooled=52, demotions=7,
                  demote_reason="live P/L -0.1585 over 17 trades is not profitable",
                  blockers=["demoted x7, judged by the re-arm rule"]),
    ]
    report = _report(strategies)

    assert [d["id"] for d in report["demoted"]] == ["atf_static"]
    assert [b["id"] for b in report["blocked"]] == ["atf_static_scout"]


def test_an_unmeasurable_link_is_not_counted_as_passing():
    """ok=None means "could not measure", and must not advance the marker.

    live_path_check's Link.unknown() sets ok to None. Identity-testing ``is
    False`` would treat that as a pass and point the next pass at a wall
    further down the path than the one actually blocking.
    """
    links = [
        {"step": 1, "name": "FEED", "ok": True, "detail": "ticking"},
        {"step": 2, "name": "SIGNALS", "ok": None, "detail": "could not measure"},
        {"step": 3, "name": "GHOST", "ok": False, "detail": "no ghost activity"},
    ]
    report = _report([_strategy("x", 1, blockers=["needs 19 more"])], links=links)

    assert report["first_failing_link"]["name"] == "SIGNALS"
    m = gs.metrics(report)
    assert m["links_passing"] == 1, "only FEED passed; None is not a pass"
    assert m["first_failing_link"] == "SIGNALS"


def test_metrics_carries_both_populations_and_stays_flat():
    """The gate diffs a single JSON object; nesting would break the comparison.

    Both counts ride along because the GAP between them is what a pass is
    closing -- 23 tradeable against 410 pooled on the day this was written.
    """
    import json

    report = _report(
        [_strategy("atf_static_scout", 3, pooled=236, win=0.33, profit=-0.0777)],
        links=[{"step": 1, "name": "FEED", "ok": True, "detail": ""}],
    )
    report["totals"] = {
        "live_approved": 0, "ghost_trades": 23, "pooled_trades": 410,
        "win_rate": 0.17, "ghost_profit": -0.894, "pooled_profit": 6.8134,
        "trades_per_day": 3.2,
    }
    m = gs.metrics(report)

    assert m["ghost_trades"] == 23 and m["pooled_trades"] == 410
    assert m["ghost_profit"] == -0.894 and m["pooled_profit"] == 6.8134
    line = json.dumps(m)
    assert "\n" not in line
    for key, value in m.items():
        assert not isinstance(value, (dict, list)), f"{key} must be scalar, got {value!r}"


def test_render_survives_a_section_that_failed():
    """A locked DB in one section must not blank the whole report.

    A status command that dies tells the next pass nothing, which is worse than
    a partial one -- the loop opens the pass with no facts and the agent guesses.
    """
    report = _report([_strategy("atf_static", 4, pooled=52, win=0.5, profit=-0.0187)])
    report["totals"] = {"live_approved": 0, "ghost_trades": 4, "pooled_trades": 52}
    report["errors"] = {"pipeline": "sqlite3.OperationalError: database is locked"}
    report["live_path"] = []
    text = gs.render(report)
    assert "GRADUATION TO LIVE TRADING" in text
    assert "atf_static" in text
    assert "pipeline section failed" in text
