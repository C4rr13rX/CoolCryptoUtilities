"""
The graduation status must name WHICH wall the funnel stands at, not just that
it is stuck.

The bug this prevents: a status command that reports "live-approved: 0" and
stops. Zero approvals is true whether nothing has enough evidence yet, or two
strategies clear the bar and the ledger simply never stamped them. Those are
opposite jobs -- go generate ghost trades, versus go fix the stamping code --
and a loop that cannot tell them apart burns passes on the wrong one. The
previous run spent 86 passes asking "which gate is closed?" because the status
it opened with could not distinguish them.

Also pinned: ``ok`` on a live-path link is tri-state, and None means the link
could not be measured. An unmeasurable link counted as a pass would move the
"first failing link" marker PAST the real wall, which is the same class of
error -- a report that is confidently wrong about where the work is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
        "closest": [
            {
                "id": s["id"],
                "ghost_trades": s["ghost_trades"],
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


def _strategy(sid, trades, win=0.7, profit=1.0, ready=None, approved=False, blockers=()):
    return {
        "id": sid,
        "ghost_trades": trades,
        "win_rate": win,
        "ghost_profit": profit,
        "ready": (not blockers) if ready is None else ready,
        "live_approved": approved,
        "blockers": list(blockers),
    }


def test_ready_but_unstamped_is_not_reported_as_missing_evidence():
    """Two strategies past the bar with no approval is a STAMPING failure.

    Against the old behaviour -- "live_approved == 0, therefore go get more
    evidence" -- this is the case that sends a pass to generate ghost trades
    that already exist. The wall must name the strategies and say the approval
    is missing, not that the evidence is.
    """
    strategies = [
        _strategy("atf_static_scout", 236, win=0.79, profit=6.48),
        _strategy("atf_static", 52, win=0.56, profit=1.54),
        _strategy("ema_cross", 3, win=0.33, profit=-0.02, blockers=["needs 17 more"]),
    ]
    report = _report(strategies)
    assert report["ready_not_approved"] == ["atf_static_scout", "atf_static"]
    assert report["approved"] == []

    wall = gs.classify_wall(
        approved=report["approved"],
        ready_not_approved=report["ready_not_approved"],
        ranked=report["closest"],
        min_trades=20,
    )
    assert wall.startswith("READY BUT UNSTAMPED"), wall
    assert "atf_static_scout" in wall
    # The failure mode being pinned: never send this pass to collect evidence.
    assert "EVIDENCE" not in wall
    assert "more ghost trades" not in wall

    m = gs.metrics(report)
    assert m["ready_not_approved"] == 2
    assert m["live_approved"] == 0


def test_no_evidence_is_distinguishable_from_unstamped():
    """When nothing clears the bar, ready_not_approved must be empty.

    This is the other side of the discrimination: same live_approved == 0, and
    the metrics must still separate the two situations for the gate to diff.
    """
    strategies = [
        _strategy("ema_cross", 3, win=0.33, profit=-0.02, blockers=["needs 17 more"]),
        _strategy("bus_schedule", 4, win=0.0, profit=-0.09, blockers=["needs 16 more"]),
    ]
    report = _report(strategies)
    assert report["ready_not_approved"] == []

    wall = gs.classify_wall(
        approved=report["approved"],
        ready_not_approved=report["ready_not_approved"],
        ranked=report["closest"],
        min_trades=20,
    )
    assert wall.startswith("EVIDENCE"), wall
    assert "UNSTAMPED" not in wall

    m = gs.metrics(report)
    assert m["live_approved"] == 0
    assert m["ready_not_approved"] == 0
    # The shortfall is what a pass acts on: how many ghost trades short the
    # closest strategy is. 20 - 4 = 16, not zero.
    assert m["closest_shortfall"] == 16


def test_an_unmeasurable_link_is_not_counted_as_passing():
    """ok=None means "could not measure", and must not advance the marker.

    live_path_check's Link.unknown() sets ok to None. Truthiness testing would
    treat that as a failure by accident and identity-testing ``is False`` would
    treat it as a pass -- the second is the dangerous one, because it points the
    next pass at a wall further down the path than the one actually blocking.
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


def test_metrics_is_one_flat_json_line_for_the_gate():
    """The gate diffs a single JSON object; nesting would break the comparison."""
    import json

    report = _report(
        [_strategy("atf_static_scout", 236, win=0.79, profit=6.48)],
        links=[{"step": 1, "name": "FEED", "ok": True, "detail": ""}],
    )
    m = gs.metrics(report)
    line = json.dumps(m)
    assert "\n" not in line
    for key, value in m.items():
        assert not isinstance(value, (dict, list)), f"{key} must be scalar, got {value!r}"


def test_render_survives_a_section_that_failed():
    """A locked DB in one section must not blank the whole report.

    A status command that dies tells the next pass nothing, which is worse than
    a partial one -- the loop opens the pass with no facts and the agent guesses.
    """
    report = _report([_strategy("atf_static", 52, win=0.56, profit=1.54)])
    report["errors"] = {"pipeline": "sqlite3.OperationalError: database is locked"}
    report["live_path"] = []
    text = gs.render(report)
    assert "GRADUATION TO LIVE TRADING" in text
    assert "atf_static" in text
    assert "pipeline section failed" in text
