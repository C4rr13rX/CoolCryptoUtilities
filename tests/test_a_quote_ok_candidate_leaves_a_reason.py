"""A candidate that quoted OK and never entered must say why, in ``trading_ops``.

``services/atf_static_strategy`` probes a real quote for every candidate it
publishes and logs ``ghost_candidate_quote_ok`` when the quote succeeds. That
row is the funnel's entry point. Every refusal downstream of it writes its own
row -- ``entry-predropped-edge-ban`` in the scheduler, and
``entry-refused-lattice`` / ``entry-refused-slot-busy`` /
``entry-refused-stop-survivability`` in ``trading/bot.py`` -- so a census of
``trading_ops`` can normally attribute each candidate to the gate that stopped
it.

One path wrote nothing at all: ``BusScheduler.evaluate`` returning ``None`` at
``no_candidates (thresholds not met)``. The reason lived only in memory, on
``state.last_filter_reason``, where a snapshot could show it and no query could
count it.

Measured 2026-09-10 from ``storage/trading_cache.db``:

    organism_snapshots, 6h        1749 snapshots, 1646 route evaluations
      no_candidates               1083  (66%)   <- not one trading_ops row
      directive issued             563

    trading_ops, same 6h, DRB-USDC
      ghost_candidate_quote_ok      34
      every other status             0   <- 34 candidates, no answer anywhere

DRB-USDC was evaluated 36 times in that window and privately recorded
``no_candidates (thresholds not met)`` every time. From ``trading_ops`` -- the
table every funnel measurement in this repo is built on -- those 34 candidates
simply evaporated, and three passes of census work attributed them to the edge
ban, which had refused DRB exactly zero times.

This test pins the row and the condition it names. It is about ATTRIBUTION, not
admission: the candidate is still refused, and the four entry conjuncts are
untouched. What changes is that the refusal is countable.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pytest

import trading.scheduler as scheduler_module
from trading.scheduler import BusScheduler


SYMBOL = "DRB-USDC"


class StubPortfolio:
    def __init__(self, quote_balance: float, native_balance: float) -> None:
        self.quote_balance = quote_balance
        self.native_balance = native_balance

    def get_quantity(self, symbol: str, chain: str = "base") -> float:
        return self.quote_balance if symbol.upper() == "USDC" else 0.0

    def get_native_balance(self, chain: str = "base") -> float:
        return self.native_balance


class RecordingDb:
    """Captures ``log_trade`` the way ``trading_ops`` would store it."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.rows: List[Dict[str, Any]] = []

    def log_trade(self, **kwargs: Any) -> None:
        self.rows.append(dict(kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _pump(sched: BusScheduler, steps: int = 32) -> None:
    ts = time.time() - steps * 60.0
    for i in range(steps):
        sched._update_state(
            {
                "symbol": SYMBOL,
                "ts": ts + i * 60.0,
                "price": 1.0,
                "volume": 1_000.0,
                "chain": "base",
            }
        )


def _scheduler(monkeypatch) -> BusScheduler:
    sched = BusScheduler(horizons=[("5m", 300)], min_profit=0.01, prefill=False)
    _pump(sched)

    # No edge ban: this candidate must reach `no_candidates`, not the
    # pre-drop, or the test would be re-proving `entry-predropped-edge-ban`.
    monkeypatch.setattr(scheduler_module, "_strategy_edge_refusal", lambda sid: None)
    monkeypatch.setattr(scheduler_module, "_symbol_edge_refusal", lambda sym, sid=None: None)

    # No forecast signal and no strategy candidate: the tick survives only
    # because the ATF scout candidate below keeps it alive, which is exactly
    # the shape DRB-USDC is in on the live feed.
    monkeypatch.setattr(sched, "_forecast", lambda state: [])
    if sched.strategy_registry is not None:
        monkeypatch.setattr(sched.strategy_registry, "evaluate_all", lambda state, ctx: [])

    # A live, quote-probed atf_static candidate for this symbol -- the row
    # that is logged as `ghost_candidate_quote_ok`. Its expected_return is
    # deliberately below any plausible cost, so the entry test REFUSES it and
    # the refusal is what we are measuring.
    import services.atf_static_strategy as atf

    monkeypatch.setattr(
        atf,
        "latest_signals",
        lambda max_age_sec=1800.0: [
            {
                "symbol": SYMBOL,
                "quote_probe": {"ok": True},
                "expected_return": 0.001,
            }
        ],
    )
    monkeypatch.setenv("ATF_STATIC_GHOST_SCOUT_ENABLED", "1")

    recorder = RecordingDb(sched.db)
    monkeypatch.setattr(sched, "db", recorder)
    return sched


def _evaluate(sched: BusScheduler):
    return sched.evaluate(
        {
            "symbol": SYMBOL,
            "ts": time.time(),
            "price": 1.0,
            "volume": 1_000.0,
            "chain": "base",
        },
        # Direction and confidence are healthy on purpose: the conjunct that
        # binds must be the margin, so the recorded `failed` list is a real
        # discrimination and not "everything failed".
        {"direction_prob": 0.75, "exit_conf": 0.72, "net_margin": 0.02},
        StubPortfolio(quote_balance=1_000.0, native_balance=1.0),
    )


def test_a_quote_ok_candidate_that_never_enters_writes_a_refusal_row(monkeypatch) -> None:
    """34 DRB-USDC candidates in 6h wrote zero rows. One refusal, one row."""
    sched = _scheduler(monkeypatch)

    directive = _evaluate(sched)

    assert directive is None, (
        "the fixture is meant to be REFUSED -- if it entered, the conjuncts "
        "moved and this test is measuring the wrong path"
    )
    assert sched.routes[SYMBOL].last_filter_reason.startswith("no_candidates"), (
        f"expected the no_candidates path, got {sched.routes[SYMBOL].last_filter_reason!r}"
    )

    rows = [r for r in sched.db.rows if r.get("status") == "entry-refused-no-candidates"]
    assert rows, (
        "a candidate that quoted OK was refused and left NO row in trading_ops: "
        f"statuses written were {[r.get('status') for r in sched.db.rows]!r}"
    )

    row = rows[-1]
    assert row.get("symbol") == SYMBOL
    assert row.get("action") == "hold"
    detail = (row.get("details") or {}).get("detail") or {}
    assert detail.get("failed"), (
        "the row names no condition -- a reason string that identifies nothing "
        "is why the previous census could not tell a refusal from a loss"
    )
    assert "expected_vs_min_profit" in detail["failed"], (
        "the binding conjunct is the margin against the profit floor; the row "
        f"blamed {detail['failed']!r}"
    )
    got = detail["values"]["expected_vs_min_profit"]
    assert got["got"] <= got["need"], (
        "the row claims a conjunct failed while its own numbers say it passed: "
        f"{got!r}"
    )


def test_a_refusal_row_is_not_written_when_no_candidate_was_quoted(monkeypatch) -> None:
    """No quote-OK candidate means nothing was lost, so nothing is logged.

    The row exists to attribute candidates that entered the funnel. Writing one
    on every idle tick would add volume without adding attribution -- and the
    scheduler evaluates every streamed symbol on every tick whether or not any
    strategy is interested in it.
    """
    sched = _scheduler(monkeypatch)
    monkeypatch.setenv("ATF_STATIC_GHOST_SCOUT_ENABLED", "0")

    directive = _evaluate(sched)

    assert directive is None
    assert not [r for r in sched.db.rows if r.get("status") == "entry-refused-no-candidates"], (
        "a tick with no candidate at all logged a lost candidate"
    )
