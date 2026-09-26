"""Hand-approval cannot clear the live refusal, because the refusal reads LIFETIME.

[137d3c5a] THE SAME ONE-WAY RATCHET, ONE METHOD AWAY, ON THE HALF THAT SPENDS.

Two predicates in ``trading/strategies/ledger.py`` judge the same strategy on
two different books, and only one of them was fixed.

The DEMOTION predicate reads the licence. ``ledger.py:1149-1160`` calls
``self._licence_trades(live)`` and ``self._licence_net(live)``, and its own
comment at ``:1142-1148`` says exactly why, verbatim::

    Measured over the CURRENT licence, not over the strategy's lifetime.
    Read lifetime it is a one-way ratchet: a demoted strategy takes no
    further live trades, so its lifetime sum cannot move, so the rule that
    demoted it re-demotes it on every subsequent record() call forever. See
    `_licence_net` for the measurement on atf_static, whose -0.186371 over
    18 lifetime trades had produced seven demotions and an empty
    `approved_ids()`.

The REFUSAL predicate reads lifetime. ``_losing_book_locked`` at
``ledger.py:1663-1664`` takes ``book.get("trades")`` and
``book.get("total_profit")`` straight off the ledger row. There is no licence
scoping anywhere in that method. ``live_order_refusal`` (``:1679``) calls it,
``may_place_live_order`` (``:1809``) conjoins it, and
``bot.py:2050 _strategy_live_approved`` returns that -- which is the single seam
behind every path in ``bot.py`` that spends real money (``:7282``, ``:7558``,
``:7735``, ``:9261``).

So the ratchet the demotion comment describes as fixed is still live on the
money path, and it is still quoting the very number that comment names as the
bug it fixed. Measured 2026-09-26 against the real store, read-only::

    live_order_refusal('atf_static')
      -> 'measured negative expectancy: atf_static live book n=18
          net=-0.1864 mean=-0.010354/round-trip'

against ``data/strategy_ledger.json`` ``demote_reason`` for the same row,
``'live P/L -0.1585 over 17 trades is not profitable'`` -- a different n and a
different net, from a different book, in the same file.

WHY THIS IS THE ROUTE-(a) WALL AND NOT A TIDINESS COMPLAINT. The standing
instruction offers "hand-approve ONE strategy onto the live lane at a minimum
clip and let real fills be the evidence". ``atf_static`` is the only strategy
that fills -- 953 of 2494 all-time ``ghost-entry``/``ghost-exit`` rows in
``trading_ops``, against 7 for the one id that actually holds a licence, and all
7 of those predate its 2026-09-16 approval by twelve days. Hand-approving
``atf_static`` sets ``live_approved`` and nothing else: ``:1762-1764`` narrows
``own_modes`` from ``("live", "ghost")`` to ``("live",)``, and then reads the
same lifetime live book, finds the same ``-0.186371``, and refuses the same
order. ``test_approval_does_not_clear_the_refusal`` is that claim, executable.

The lockout is closed by construction: ``total_profit`` moves only when a live
trade settles, a live trade settles only if the refusal passes, and the refusal
fails while ``total_profit`` is negative. No input breaks the cycle.
``test_a_positive_book_is_admitted`` pins the other side of it -- the gate is a
MEASUREMENT, not a constant, so the only thing that could ever clear it is the
live trade it forbids.

WHAT THESE TESTS DO NOT CLAIM. They do not say the lifetime read is the bug.
The complement is live and unresolved: the lifetime read at ``:1663`` may be
deliberate and the licence read at ``:1149`` may be the thing that is too loose.
The two comments in this one file do not agree and neither cites the other.
These tests characterise HEAD so that whoever settles that question has the
behaviour pinned before they change it -- they are written to pass on
``9155c2b``, and a fix to either predicate should make the marked assertion
fail loudly rather than silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from trading.strategies.ledger import StrategyLedger

SID = "atf_static"

#: ``atf_static``'s real lifetime live book, read 2026-09-26 from
#: ``storage/trading_cache.db`` (``select count(*), sum(net_profit) from
#: trade_outcomes where wallet='live'``) and agreeing with the scoreboard the
#: loop reports. Eighteen round trips, net -0.186371. This is the book the next
#: production boot judges, so it is the book these tests use rather than a tidy
#: invention.
REAL_LIVE_TRADES = 18
REAL_LIVE_PROFIT = -0.186371

#: The same 18 rows split by whether production still trades that symbol, from
#: the same query grouped by ``symbol``, against the ``SYMBOL EDGE GATE``
#: refusals production logged at 2026-09-26 09:26:08. Fourteen rows sit on
#: AERO-USDC (9), CBBTC-USDC (3), CBXRP-USDC (1) and BASECAT-USDC (1), all four
#: refused today, and they carry -0.178426 -- 95.7% of the entire loss. The
#: residual is recorded here because it is the number an operator needs and it
#: is NOT a licence to subset: it is 4 rows at -0.007946, which is flat rather
#: than positive, and carries no information either way.
FOSSIL_ROWS = 14
FOSSIL_PROFIT = -0.178426
RESIDUAL_ROWS = 4
RESIDUAL_PROFIT = -0.007946


def _mode(trades: int = 0, wins: int = 0, total_profit: float = 0.0) -> Dict[str, Any]:
    """A book in the shape the real store holds it.

    Same helper as ``tests/test_money_button_dip_live_gate.py``; the fields are
    the ones ``_losing_book_locked`` and ``_licence_net`` actually read.
    """
    return {
        "trades": int(trades),
        "wins": int(wins),
        "losses": max(int(trades) - int(wins), 0),
        "total_profit": float(total_profit),
        "conf_ema": 0.0,
        "peak_profit": 0.0,
        "max_drawdown": 0.0,
        "consecutive_losses": 0,
        "last_ts": 0.0,
    }


def _row(
    *,
    live_approved: bool = False,
    demotions: int = 7,
    live: Optional[Dict[str, Any]] = None,
    ghost: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """``atf_static``'s row, defaulting to the fields the real store carries.

    ``demotions=7`` is the recorded value, not a guess: the real row reads
    ``demotions: 7, rearms: 1, live_approved: false``.
    """
    return {
        "live_approved": live_approved,
        "demotions": demotions,
        "demote_reason": None,
        "graduation_blocked": False,
        "graduation_blocked_reason": None,
        "ghost": ghost if ghost is not None else _mode(),
        "live": live if live is not None else _mode(),
    }


def _ledger(tmp_path: Path, rows: Dict[str, Dict[str, Any]]) -> StrategyLedger:
    """A ledger on an explicit tmp path -- never the real store.

    The real store has concurrent writers and a live operator approval in it, so
    a test must neither read its way to a pass nor write to it.
    """
    path = tmp_path / "strategy_ledger.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return StrategyLedger(path=str(path))


def _real_book(**overrides: Any) -> Dict[str, Any]:
    book = _mode(trades=REAL_LIVE_TRADES, wins=7, total_profit=REAL_LIVE_PROFIT)
    book.update(overrides)
    return book


def test_the_recorded_book_refuses_a_live_order(tmp_path: Path) -> None:
    """Baseline: the real lifetime book refuses, and names itself in the reason."""
    ledger = _ledger(tmp_path, {SID: _row(live=_real_book())})

    assert ledger.may_place_live_order(SID) is False
    reason = ledger.live_order_refusal(SID)
    assert reason is not None
    # The reason must be a MEASUREMENT of the row, not a constant: both the
    # count and the net have to appear in it, or no census could ever attribute
    # the refusal.
    assert "n=18" in reason
    assert "-0.1864" in reason


def test_approval_does_not_clear_the_refusal(tmp_path: Path) -> None:
    """THE ROUTE-(a) WALL. Hand-approving the only strategy that fills changes nothing.

    This is the assertion a fix to either predicate should break. It passes on
    ``9155c2b`` and it is the reason editing ``live_approved`` in
    ``data/strategy_ledger.json`` would look like a successful arm and then
    place no order.
    """
    approved = _ledger(tmp_path, {SID: _row(live_approved=True, live=_real_book())})

    # The LICENCE is granted -- approval did what approval does.
    assert approved.is_live_approved(SID) is True

    # The ORDER is still refused, on the same lifetime book, because
    # ``_losing_book_locked`` reads ``total_profit`` off the row and approval
    # cannot touch that field.
    assert approved.may_place_live_order(SID) is False
    reason = approved.live_order_refusal(SID)
    assert reason is not None and "n=18" in reason


def test_a_positive_book_is_admitted(tmp_path: Path) -> None:
    """The other side of the ratchet: the gate is a measurement, so it can clear.

    Seed the identical 18 round trips with a positive net and the same code
    permits the same order. That is what makes the lockout structural rather
    than a hard-coded veto -- the ONLY thing that can clear it is a settled live
    trade, which is the thing it forbids.
    """
    winner = _ledger(
        tmp_path,
        {SID: _row(live_approved=True, live=_real_book(total_profit=+0.05))},
    )

    assert winner.live_order_refusal(SID) is None
    assert winner.may_place_live_order(SID) is True


def test_the_fossil_split_is_recorded_and_does_not_rescue_the_book(tmp_path: Path) -> None:
    """The subset an operator will reach for, pinned so nobody reaches for it blind.

    95.7% of the loss is on symbols production refused at 09:26:08 today, which
    is the honest argument that the refusal measures a fossil. It is NOT an
    argument that the survivor is profitable: with BPAD-USDC still in the
    selection universe (``deferred OHLCV backfill for BPAD-USDC`` at
    2026-09-24 12:51:35, so it cannot be dropped), the residual is 4 rows at
    -0.007946. Flat. Seeded here, the residual book STILL refuses -- so a fix
    that merely drops the refused symbols does not open the lane either.
    """
    assert FOSSIL_ROWS + RESIDUAL_ROWS == REAL_LIVE_TRADES
    # 2e-6, not 1e-6: the two parts are each quoted to six places off the
    # per-symbol query output, so their sum carries the rounding of both and
    # lands on -0.186372 against the total's -0.186371. Tightening this would be
    # asserting that a rounded sum is exact.
    assert FOSSIL_PROFIT + RESIDUAL_PROFIT == pytest.approx(REAL_LIVE_PROFIT, abs=2e-6)

    residual = _ledger(
        tmp_path,
        {
            SID: _row(
                live_approved=True,
                live=_real_book(trades=RESIDUAL_ROWS, wins=2, total_profit=RESIDUAL_PROFIT),
            )
        },
    )
    assert residual.may_place_live_order(SID) is False
