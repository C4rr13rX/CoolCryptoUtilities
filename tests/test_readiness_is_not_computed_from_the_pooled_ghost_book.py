"""The readiness report must measure what the graduation gate measures.

The defect this pins, measured 2026-09-10 on the real ledger:

    atf_static        POOLED 52 trades  29 wins  +1.5407  -> report said ready
                   TRADEABLE  4 trades   2 wins  -0.0187  -> gate said no
    atf_static_scout  POOLED 236 trades 186 wins +6.4818  -> report said ready
                   TRADEABLE   3 trades   1 win  -0.0778  -> gate said no,
                                                             permanently

``scripts/readiness_report.collect`` computed ``ready`` from ``entry["ghost"]``
-- the pooled book -- while ``StrategyLedger._evaluate_graduation_locked``
judges ``_tradeable_of(ghost)`` and returns early on three structural bars the
report never read: ``graduation_blocked``, ``GHOST_ONLY_STRATEGY_IDS``, and
``demote_reason``.

The cost was not a wrong number on a page. ``classify_wall`` consumes
``ready``, so the wall printed at the top of every pass read "READY BUT
UNSTAMPED -- the ledger is not stamping graduated_ts" while both strategies
already carried ``graduated_ts`` and the gate was correctly refusing them.
Passes were steered at the stamping code, which was not broken.

Every test here fails against the pre-fix ``collect`` and passes after it.
"""

from __future__ import annotations

import copy

import pytest

from scripts import readiness_report


def _entry(pooled_trades, pooled_wins, pooled_profit,
           tradeable_trades, tradeable_wins, tradeable_profit, **extra):
    """A ledger entry whose pooled book clears the bar and whose tradeable
    sub-book does not. That gap is the whole bug."""
    ent = {
        "ghost": {
            "trades": pooled_trades,
            "wins": pooled_wins,
            "losses": pooled_trades - pooled_wins,
            "total_profit": pooled_profit,
            "last_ts": 1789000000.0,
            "tradeable": {
                "trades": tradeable_trades,
                "wins": tradeable_wins,
                "losses": tradeable_trades - tradeable_wins,
                "total_profit": tradeable_profit,
            },
        },
        "live": {"trades": 0, "wins": 0, "losses": 0, "total_profit": 0.0},
        "live_approved": False,
    }
    ent.update(extra)
    return ent


class _FakeLedger:
    """Stands in for StrategyLedger so the test never touches the real file."""

    def __init__(self, data):
        self._data = data

    def _licence_net(self, live):
        return float(live.get("total_profit", 0.0))

    def _licence_trades(self, live):
        return int(live.get("trades", 0))


@pytest.fixture
def collect_with(monkeypatch):
    """Run ``collect()`` against an in-memory ledger."""

    def _run(data, env=None):
        for key, val in (env or {}).items():
            monkeypatch.setenv(key, val)
        fake = _FakeLedger(copy.deepcopy(data))
        monkeypatch.setattr(
            "trading.strategies.ledger.StrategyLedger", lambda *a, **k: fake
        )
        return readiness_report.collect()

    return _run


def _by_id(report, sid):
    return next(s for s in report["strategies"] if s["id"] == sid)


def test_a_pooled_book_that_clears_the_bar_is_not_ready(collect_with):
    """atf_static's exact shape: 52/29/+1.5407 pooled, 4/2/-0.0187 tradeable.

    The pooled book clears 20 trades, 55% and positive P/L on every count. The
    tradeable book clears none of them. The gate reads the tradeable book, so
    the report must too. Pre-fix this asserted ready is True.
    """
    rep = collect_with({
        "atf_static": _entry(52, 29, 1.5406913, 4, 2, -0.01868877),
    })
    s = _by_id(rep, "atf_static")

    assert s["ready"] is False, (
        "a strategy whose LIVE-TRADEABLE record is 4 trades at -0.0187 was "
        "reported ready because its pooled book was 52 trades at +1.5407"
    )
    assert s["ghost_trades"] == 4
    assert s["ghost_profit"] == pytest.approx(-0.01868877)
    assert s["population"] == "tradeable"
    # The pooled numbers are reported, not deleted -- throughput stays visible.
    assert s["pooled_ghost_trades"] == 52
    assert s["pooled_ghost_profit"] == pytest.approx(1.5406913)
    assert "needs 16 more ghost trades" in s["blockers"]


def test_a_ghost_only_strategy_is_never_reported_ready(collect_with):
    """atf_static_scout carries graduation_blocked=True: no live branch exists.

    ``_evaluate_graduation_locked`` returns before reading a single number for
    such a strategy, and no quantity of ghost evidence retires that. The report
    must say so as a PERMANENT blocker, not as a countdown -- "needs N more
    trades" on a strategy that can never graduate is an invitation to go feed
    it, which is exactly the wasted pass this whole fix is about.
    """
    rep = collect_with({
        "atf_static_scout": _entry(
            236, 186, 6.4818, 25, 20, 3.0, graduation_blocked=True
        ),
    })
    s = _by_id(rep, "atf_static_scout")

    assert s["ready"] is False, (
        "a ghost-only executor cleared the bar on its tradeable book and was "
        "reported ready; it has no live branch and can never spend money"
    )
    assert s["permanently_blocked"] is True
    assert any("can never graduate" in b for b in s["blockers"])
    assert rep["totals"]["permanently_blocked"] == 1


def test_the_default_ghost_only_id_is_blocked_without_the_flag(collect_with):
    """GHOST_ONLY_STRATEGY_IDS is the other half of the structural bar.

    An entry can carry the id and not the ``graduation_blocked`` field -- the
    ledger consults both, so the report must consult both.
    """
    rep = collect_with(
        {"scout_x": _entry(236, 186, 6.4818, 25, 20, 3.0)},
        env={"GHOST_ONLY_STRATEGY_IDS": "scout_x"},
    )
    assert _by_id(rep, "scout_x")["permanently_blocked"] is True
    assert _by_id(rep, "scout_x")["ready"] is False


def test_a_demoted_strategy_is_judged_on_fresh_evidence_only(collect_with):
    """A demotion sends the strategy to the re-arm rule, which reads the
    tradeable delta since ``ghost_at_demotion`` -- not the lifetime book.

    Reading the lifetime book here is the "a ghost trade undid a live
    demotion" shape: evidence the strategy already had when it was demoted
    cannot be what un-demotes it.
    """
    ent = _entry(60, 40, 2.0, 30, 21, 1.5, demote_reason="live drawdown")
    # It had 28 of those 30 tradeable trades BEFORE the demotion.
    ent["ghost_at_demotion"] = {
        "trades": 56, "wins": 38, "total_profit": 1.9,
        "tradeable": {"trades": 28, "wins": 20, "total_profit": 1.45},
    }
    rep = collect_with({"demoted": ent})
    s = _by_id(rep, "demoted")

    assert s["population"] == "fresh tradeable since demotion"
    assert s["ghost_trades"] == 2, (
        "the lifetime tradeable book (30 trades) was read as if it were "
        "evidence gathered since the demotion (2 trades)"
    )
    assert s["ready"] is False
    assert "needs 18 more ghost trades" in s["blockers"]


def test_a_demoted_strategy_with_no_snapshot_starts_its_window_empty(collect_with):
    """Fail CLOSED, exactly as ``_maybe_rearm_locked`` does.

    Demoted before the snapshot existed, or by a hand-edit: the gate baselines
    from now, so the fresh window is empty until the next close. Reporting the
    lifetime book here would say "ready" for a strategy the gate will refuse.
    """
    ent = _entry(60, 40, 2.0, 30, 21, 1.5, demote_reason="live drawdown")
    ent.pop("ghost_at_demotion", None)
    s = _by_id(collect_with({"demoted": ent}), "demoted")

    assert s["ghost_trades"] == 0
    assert s["ready"] is False


def test_a_demoted_strategy_that_lost_real_money_carries_that_blocker(collect_with):
    """The re-arm rule refuses on a losing live licence over a big enough
    sample, whatever the ghost record says. atf_static's real state: demoted on
    "live P/L -0.1585 over 17 trades". A report that omits this reads as though
    fresh ghost trades alone would re-arm it.
    """
    ent = _entry(60, 40, 2.0, 60, 45, 3.0, demote_reason="live P/L -0.1585 over 17 trades")
    ent["live"] = {"trades": 17, "wins": 5, "losses": 12, "total_profit": -0.1585}
    ent["ghost_at_demotion"] = {"trades": 0, "wins": 0, "total_profit": 0.0,
                                "tradeable": {"trades": 0, "wins": 0, "total_profit": 0.0}}
    s = _by_id(collect_with({"atf_static": ent}), "atf_static")

    # Fresh tradeable evidence alone would clear the bar here...
    assert s["ghost_trades"] == 60
    # ...and it still is not ready, because the live licence lost money.
    assert s["ready"] is False
    assert any("cannot excuse lost money" in b for b in s["blockers"])


def test_the_eta_ignores_strategies_that_can_never_graduate(collect_with):
    """A permanently blocked strategy must not set the countdown.

    It is nearest the bar by trade count and it never arrives, so letting it
    hold the minimum reports an ETA for a graduation that cannot happen. The
    real report printed "nearest graduation ~0.7 days" for months.
    """
    blocked_ent = _entry(300, 250, 9.0, 19, 15, 4.0, graduation_blocked=True)
    far_ent = _entry(30, 20, 1.0, 2, 1, 0.1)
    # The rate is trades/span, so the two stamps must differ or span_days is 0
    # and there is no ETA to test. Ten days apart.
    far_ent["ghost"]["last_ts"] = blocked_ent["ghost"]["last_ts"] + 10 * 86400

    rep = collect_with({
        "blocked_and_close": blocked_ent,
        "real_and_far": far_ent,
    })
    eta = rep["totals"]["eta_days_to_first_graduation"]
    blocked = _by_id(rep, "blocked_and_close")

    assert blocked["ghost_trades"] == 19  # one trade off the bar...
    assert blocked["permanently_blocked"] is True  # ...and it never gets there
    # The ETA must describe "real_and_far" (18 short), not the blocked one (1
    # short). Pre-fix the blocked strategy set it, so any ETA proving the fix
    # must be strictly larger than the one a 1-trade shortfall would give.
    assert eta is not None
    assert eta > 1.0


def test_ready_and_the_ledger_gate_agree_on_the_real_ledger():
    """The end-to-end invariant, against whatever the ledger actually holds.

    Nothing may be reported ready-and-unapproved while the ledger's own gate
    would refuse it. This is the assertion that would have caught the bug
    without anyone knowing which field was wrong: the report and the gate
    disagreeing about a single strategy is the defect, whatever its cause.
    """
    from trading.strategies.ledger import StrategyLedger

    rep = readiness_report.collect()
    ledger = StrategyLedger()
    data = dict(getattr(ledger, "_data", {}))

    for s in rep["strategies"]:
        if not s["ready"] or s["live_approved"]:
            continue
        ent = data.get(s["id"], {})
        assert not ent.get("graduation_blocked"), (
            f"{s['id']} is reported ready and is structurally barred from "
            f"ever graduating"
        )
        assert not ent.get("demote_reason"), (
            f"{s['id']} is reported ready while demoted for "
            f"{ent.get('demote_reason')!r}; the re-arm rule owns it"
        )
