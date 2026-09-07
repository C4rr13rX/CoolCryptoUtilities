"""Progress toward live is measured on the population the bar actually reads.

``scripts/readiness_report.py`` -- which is what ``/api/telemetry/readiness/``
serves and what the dashboard has been showing -- reports ``atf_static`` as::

    {"id": "atf_static", "ghost_trades": 49, "ready": true, "blockers": []}

``atf_static`` cannot graduate, and it is not close. It carries a
``demote_reason``, so ``_evaluate_graduation_locked`` hands it to
``_maybe_rearm_locked``, which reads the FRESH LIVE-TRADEABLE delta gathered
since ``ghost_at_demotion``. Measured 2026-09-07 that is 1 round trip against a
bar of 20. The readiness row is wrong by a factor of ~49 because it counted the
pooled ghost book, which is not a population any bar consults:

    pooled ghost              49 trades   <- what the page showed, "ready"
    live-tradeable subset      1 trade
    fresh since demotion       1 trade    <- what the re-arm rule reads

Two independent errors stack there, and each on its own is enough to invert the
verdict:

  * pooled vs live-tradeable. Across the whole 41-strategy population the two
    are 394 and 7. A licence granted on the pooled number is a licence granted
    on evidence the live lane could never have placed -- the same shape
    ``ledger._live_tradeable`` was written to stop.
  * lifetime vs fresh-since-demotion. A demotion is a demand for NEW evidence.
    Counting the book that convicted the strategy as proof of its recovery
    re-approves it on the very trades that demoted it.

So the rule pinned here: a row's ``progress`` reports the basis the ledger
would use for THAT strategy -- ``re-arm`` for a demoted one, ``first-licence``
otherwise -- and a demoted strategy with a large pooled book and no fresh
tradeable evidence reports a blocker rather than readiness.
"""
from __future__ import annotations

import json

import pytest

from services import strategy_registry
from services.strategy_population import collect


def _files(tmp_path, monkeypatch, ledger_rows, registry_rows=None):
    reg = tmp_path / "registry.json"
    reg.write_text(
        json.dumps({"strategies": registry_rows or {}}), encoding="utf-8"
    )
    monkeypatch.setattr(strategy_registry, "REGISTRY_PATH", reg)
    led = tmp_path / "ledger.json"
    led.write_text(json.dumps(ledger_rows), encoding="utf-8")
    return led


@pytest.fixture(autouse=True)
def _bar(monkeypatch):
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_TRADES", "20")
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_WINRATE", "0.55")
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_PROFIT", "0.0")
    monkeypatch.setenv("GHOST_ONLY_STRATEGY_IDS", "")


def _atf_static_shaped():
    """The real ledger's shape for a demoted strategy with a big pooled book."""
    return {
        "demoted": {
            "ghost": {
                "trades": 49,
                "wins": 28,
                "total_profit": 1.5777,
                # Only ONE of those 49 is live-tradeable...
                "tradeable": {
                    "trades": 1,
                    "wins": 1,
                    "losses": 0,
                    "total_profit": 0.0183,
                },
            },
            # ...and zero tradeable rows predate the demotion, so the fresh
            # delta is that same single trade.
            "ghost_at_demotion": {
                "trades": 40,
                "wins": 22,
                "total_profit": 1.2,
                "tradeable": {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_profit": 0.0,
                },
            },
            "live": {"trades": 18, "wins": 5, "total_profit": -0.1863},
            "live_approved": False,
            "demote_reason": "live P/L -0.1585 over 17 trades is not profitable",
            "demoted_ts": 1000.0,
            "demotions": 7,
        }
    }


def test_a_demoted_strategy_is_judged_on_fresh_tradeable_evidence(
    tmp_path, monkeypatch
):
    """49 pooled ghost trades must not read as 49 trades of progress."""
    led = _files(tmp_path, monkeypatch, _atf_static_shaped())

    row = collect(now=2000.0, ledger_path=led)["strategies"][0]
    prog = row["progress"]

    assert prog["basis"] == "re-arm"
    # The bug: this was 49.
    assert prog["trades_have"] == 1
    assert prog["trades_need"] == 20
    assert prog["meets_trades"] is False
    assert prog["sample_fraction"] == pytest.approx(0.05)

    # ...and it must SAY it is blocked rather than report an empty list.
    assert row["blockers"], "a strategy 1/20 of the way to a licence reported no blocker"
    assert "1/20" in row["blockers"][0]
    assert "re-arm" in row["blockers"][0]

    # The pooled book is still reported -- it is real -- but beside the
    # tradeable subset, never instead of it.
    assert row["ghost"]["trades"] == 49
    assert row["tradeable"]["trades"] == 1

    assert row["stage"] == "ghost"
    assert row["status"] == "demoted-rearming"
    assert row["status_reason"].startswith("live P/L")
    assert row["stage_age_sec"] == pytest.approx(1000.0)


def test_an_undemoted_strategy_is_judged_on_its_tradeable_subset(
    tmp_path, monkeypatch
):
    """The first licence reads the tradeable subset, not the pooled book."""
    led = _files(
        tmp_path,
        monkeypatch,
        {
            "fresh": {
                "ghost": {
                    "trades": 30,
                    "wins": 25,
                    "total_profit": 2.0,
                    "tradeable": {
                        "trades": 3,
                        "wins": 2,
                        "losses": 1,
                        "total_profit": 0.05,
                    },
                }
            }
        },
    )

    prog = collect(now=2000.0, ledger_path=led)["strategies"][0]["progress"]

    assert prog["basis"] == "first-licence"
    # 30 pooled trades clearing the 20-trade bar would have read as ready.
    assert prog["trades_have"] == 3
    assert prog["meets_trades"] is False


def test_a_permanently_blocked_strategy_reports_the_block_and_nothing_else(
    tmp_path, monkeypatch
):
    """A structural bar is never re-litigated, so no sample shortfall is listed.

    ``atf_static_scout`` holds 235 ghost round trips at a 79% win rate and can
    never spend one of them: it hardcodes ``wallet="ghost"``. Listing
    "2/20 round trips" under that block would imply a route to live that
    ``_evaluate_graduation_locked`` returns before ever reaching.
    """
    led = _files(
        tmp_path,
        monkeypatch,
        {
            "scout": {
                "ghost": {
                    "trades": 235,
                    "wins": 186,
                    "total_profit": 6.5595,
                    "tradeable": {"trades": 2, "wins": 2, "total_profit": 0.9},
                },
                "graduation_blocked": True,
                "graduation_blocked_reason": "ghost-only executor: no live branch exists",
                "demote_reason": "ghost-only",
                "demoted_ts": 1500.0,
            }
        },
    )

    row = collect(now=2000.0, ledger_path=led)["strategies"][0]

    assert row["stage"] == "rejected"
    assert row["status"] == "blocked-permanently"
    assert "no live branch" in row["status_reason"]
    assert row["blockers"] == ["ghost-only executor: no live branch exists"]
    # Its record is still shown. A rejected strategy with a 79% book is
    # exactly the row worth reading.
    assert row["ghost"]["trades"] == 235
    assert row["ghost"]["win_rate"] == pytest.approx(186 / 235)


def test_a_ghost_only_id_is_barred_whether_or_not_it_has_a_ledger_row(
    tmp_path, monkeypatch
):
    """GHOST_ONLY_STRATEGY_IDS bars a strategy the ledger has never written.

    The ledger stamps the block itself on load, but only onto entries that
    EXIST -- ``_revoke_ghost_only_approval`` iterates the ids and skips any
    with no ``self._data`` row. A ghost-only strategy that is registered and
    has never recorded an outcome therefore reaches this page unstamped, and
    without the mirror here it would render as an ordinary candidate on its
    way to live.

    Both halves are asserted because they take different code paths, and the
    reason string is deliberately NOT pinned: when the ledger has a row it
    supplies its own canonical wording, and asserting on the mirror's fallback
    would pin a string that the shipped path never produces.
    """
    monkeypatch.setenv("GHOST_ONLY_STRATEGY_IDS", "scout_x,never_ran_scout")
    led = _files(
        tmp_path,
        monkeypatch,
        {"scout_x": {"ghost": {"trades": 50, "wins": 40, "total_profit": 3.0}}},
        registry_rows={
            "never_ran_scout": {
                "strategy_id": "never_ran_scout",
                "name": "never_ran_scout",
                "created_at": 1.0,
            }
        },
    )

    by_id = {r["id"]: r for r in collect(now=2000.0, ledger_path=led)["strategies"]}

    # Has a ledger row: the ledger stamped it on load.
    assert by_id["scout_x"]["stage"] == "rejected"
    assert by_id["scout_x"]["status"] == "blocked-permanently"
    assert by_id["scout_x"]["blockers"], "a barred strategy reported no reason"

    # No ledger row at all: only the mirror in this module can catch it, and
    # without it this row reads as `candidate` -- promotable.
    assert by_id["never_ran_scout"]["stage"] == "rejected"
    assert by_id["never_ran_scout"]["status"] == "blocked-permanently"
    assert "GHOST_ONLY_STRATEGY_IDS" in by_id["never_ran_scout"]["blockers"][0]


def test_a_live_approved_strategy_reports_no_blockers(tmp_path, monkeypatch):
    """The live stage must be reachable and must render clean when reached."""
    led = _files(
        tmp_path,
        monkeypatch,
        {
            "armed": {
                "ghost": {
                    "trades": 25,
                    "wins": 20,
                    "tradeable": {"trades": 22, "wins": 15, "total_profit": 1.0},
                },
                "live": {"trades": 4, "wins": 3, "total_profit": 0.4},
                "live_approved": True,
                "graduated_ts": 1200.0,
            }
        },
    )

    row = collect(now=2000.0, ledger_path=led)["strategies"][0]

    assert row["stage"] == "live"
    assert row["status"] == "live-armed"
    assert row["blockers"] == []
    assert row["stage_age_sec"] == pytest.approx(800.0)
    assert collect(now=2000.0, ledger_path=led)["totals"]["live_approved"] == 1
