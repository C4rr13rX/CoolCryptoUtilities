"""A round trip whose price did not move says nothing about the strategy.

WHY THIS FILE EXISTS
--------------------
Measured 2026-09-23, pass 689, cross-verifying backlog item ``9d4d29c7``. The
claim under audit was that ``services/strategy_edge_gate`` was banning
``rsi_reversal@5d`` on "three rows that are not trades" with return_pct
exactly 0.0 and a zero-variance sample. The three rows tie by outcome_id and
the t reproduces at ``-92,555``, but BOTH stated causes are wrong:

  * ``details.return_pct`` is not 0.0, it is ABSENT. The key exists in 6 of
    284 closed rows, so any reader that defaults a missing key to 0.0 will
    manufacture that observation on 278 rows.
  * The sample is not zero-variance. Its stdev is 1.734e-07 -- small, but
    finite and positive -- so the ``stdev <= 0`` branch of ``_t_statistic``
    never runs and there is no guard hole there to fix.

The real cause is that ENTRY PRICE EQUALS EXIT PRICE on all three rows
(0.0001663 on both legs), because DRB-USDC's feed stalled: one distinct price
across the six ticks spanning the exits, against 2097 distinct prices over
2388 ticks all-time. The round trip earned the negative of its fee and nothing
else. Three receipts for the same fee differ only by quantity dust, so the
dispersion collapses and the t-statistic reports a confidence that three
observations cannot carry.

WHAT THESE TESTS PIN
--------------------
  * Flat round trips are excluded from the sample the gate judges on.
  * That exclusion is not an unbanning: a strategy whose non-flat evidence
    still fails cost is still refused, at the same ``MAX_T``.
  * A strategy left under ``MIN_SAMPLES`` by the exclusion becomes UNJUDGED,
    which is the correct state for a strategy with no usable evidence.
  * Bans resting on real dispersion survive -- the fix must not switch the
    gate off, which is the failure mode this repo has shipped before.
  * An unreadable or missing exit price KEEPS the row. This module fails open
    on everything it cannot read, and silently shrinking its own sample is how
    a gate stops firing without anyone noticing.
"""
from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Well above the measured round trip, so "did it clear cost" is never the
#: thing under test here -- these tests are about which rows enter the sample.
COST = 0.0046544285973921596


def _book(path: Path, rows) -> None:
    """``rows`` are ``(strategy, net_profit, entry_price, exit_price, qty)``."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE trade_outcomes ("
        " outcome_id INTEGER PRIMARY KEY, ts REAL, status TEXT,"
        " net_profit REAL, entry_price REAL, exit_price REAL,"
        " quantity REAL, details TEXT)"
    )
    for index, (strategy, net, entry, exit_, qty) in enumerate(rows):
        conn.execute(
            "INSERT INTO trade_outcomes"
            " (ts, status, net_profit, entry_price, exit_price, quantity, details)"
            " VALUES (?, 'closed', ?, ?, ?, ?, ?)",
            (1000.0 + index, net, entry, exit_, qty,
             json.dumps({"strategy_id": strategy})),
        )
    conn.commit()
    conn.close()


@pytest.fixture()
def gate(tmp_path, monkeypatch):
    import services.strategy_edge_gate as module
    importlib.reload(module)
    db_path = tmp_path / "book.db"
    monkeypatch.setattr(module, "DB_PATH", db_path)
    monkeypatch.setattr(module, "_cost", lambda: COST)

    def _load(rows):
        if db_path.exists():
            db_path.unlink()
        _book(db_path, rows)
        module._cache.clear()
        module._cache_built_at = 0.0
        return module

    return _load


def _the_three_drb_rows(strategy="rsi_reversal@5d"):
    """The exact shape of the rows that produced t=-92,555, to scale.

    Entry and exit identical at 0.0001663; only the quantity differs, which is
    the dust that survives a fee-only round trip.
    """
    return [
        (strategy, -0.025333812874984853, 0.0001663, 0.0001663, 33020.17703995217),
        (strategy, -0.025338705145480818, 0.0001663, 0.0001663, 33027.795411914136),
        (strategy, -0.025343598360734320, 0.0001663, 0.0001663, 33035.415255077314),
    ]


def test_three_fee_receipts_on_a_stalled_feed_are_not_a_verdict(gate):
    """The regression this file was written for.

    Before the fix these three rows banned the strategy at t=-92,555 -- a
    confidence no sample of three can carry, manufactured entirely by the
    price not moving.
    """
    module = gate(_the_three_drb_rows())
    assert module.refusal_reason("rsi_reversal@5d") is None, (
        "a round trip that opened and closed at 0.0001663 earned the negative "
        "of its fee and nothing else; it is not evidence about the strategy"
    )


def test_the_absurd_t_is_gone_rather_than_merely_capped(gate):
    """No verdict at all, not a softened one.

    Capping the t would keep the rows in the sample and keep charging a fee to
    a strategy's edge. The rows have to LEAVE.
    """
    module = gate(_the_three_drb_rows())
    assert "rsi_reversal@5d" not in module.banned_strategies()


def test_a_strategy_that_really_loses_is_still_refused(gate):
    """The fix must not be an unbanning.

    Same count, same MAX_T, prices that actually moved: the verdict stands.
    """
    rows = [("the_loser", -0.02, 1.0, 0.98 + index * 0.001, 1.0) for index in range(4)]
    module = gate(rows)
    reason = module.refusal_reason("the_loser")
    assert reason, "a real loss on a moving tape must still be refused"
    assert "4 closed round trips" in reason


def test_flat_rows_do_not_pad_a_real_strategys_sample(gate):
    """A mixed book keeps only the rows that carry a return.

    ``obv_accumulation@1w`` is the largest standing ban in the real book at
    n=9, of which 2 are flat. It must be judged on the 7, and say 7.
    """
    moving = [("mixed", -0.02, 1.0, 0.98 + index * 0.001, 1.0) for index in range(5)]
    flat = [("mixed", -0.005, 1.0, 1.0, 1.0) for _ in range(3)]
    module = gate(moving + flat)
    reason = module.refusal_reason("mixed")
    assert reason, "five real losses are still a verdict"
    assert "5 closed round trips" in reason, (
        f"the three flat rows must not pad the sample: {reason}"
    )


def test_a_strategy_left_under_the_floor_becomes_unjudged_not_cleared(gate):
    """Dropping below MIN_SAMPLES is 'no evidence', not 'good evidence'.

    The strategy is allowed to propose again and must re-earn a verdict on
    round trips where the price moved. Nothing about it has been endorsed.
    """
    module = gate(_the_three_drb_rows("starved") +
                  [("starved", -0.02, 1.0, 0.98, 1.0)])
    assert module.refusal_reason("starved") is None
    assert "starved" not in module.banned_strategies()


def test_a_missing_exit_price_keeps_the_row(gate):
    """Fails open on what it cannot read, like the rest of this module.

    A NULL exit price means the flatness is unknown. Discarding the row would
    let an unreadable column quietly switch the gate off.
    """
    rows = [("unreadable", -0.02, 1.0, None, 1.0) for _ in range(4)]
    module = gate(rows)
    reason = module.refusal_reason("unreadable")
    assert reason, "an unreadable exit price must not remove the evidence"
    assert "4 closed round trips" in reason


def test_a_flat_winner_is_dropped_too(gate):
    """The exclusion is about the tape, not about the sign.

    A flat round trip cannot be a winner -- but if one is ever booked as such,
    it is the same category error and must not enter the sample either. This
    pins that the rule is 'the price did not move', never 'the row lost'.
    """
    import services.strategy_edge_gate as module_ref  # noqa: F401
    module = gate([("flat_winner", +0.02, 1.0, 1.0, 1.0) for _ in range(4)])
    assert module._load_book().get("flat_winner") is None


def test_it_is_the_price_pair_that_decides_not_the_profit(gate):
    """Zero net profit on a MOVING tape is still an observation.

    A strategy that exactly broke even after fees observed something real. The
    rule must not degrade into "drop rows whose profit is small".
    """
    module = gate([("break_even", 0.0, 1.0, 1.02, 1.0) for _ in range(4)])
    assert module._load_book().get("break_even") == [0.0, 0.0, 0.0, 0.0]
