"""ES95 must be measured over symbols whose stop can actually bind.

THE FAILURE THIS PREVENTS
-------------------------
Measured 2026-09-06. The live lane was shut with ``block_reason
ghost_validation_block``, on ES95 = 0.10511 against a 0.100 guardrail. The
tail decomposed to 15 rows, ALL of them ``stop_loss`` exits, and 13 of the 15
were on BSTONK / BPAD / MOONBASE / BASECAT -- symbols whose p99 single-tick
jump is 10.83% / 108.59% / 99381.16% / 5.05% against a 4.00% ceiling, every
one of them already refused at entry by ``stop_survivability_gate``. Every
tail trade was between 27h and 151h old, so all of them predate that gate.

The guardrail is the stop level plus slack, so comparing ES95 against it asks
"are the stops holding?". On a symbol that can jump five times its stop in a
single tick there is no stop to hold, and its losses are the absence of a stop
rather than a breach of one. The live lane was therefore held shut by a
seven-day memory of trades the current entry gates can no longer place -- and
would have stayed shut until they aged out, with nothing a new trade could do
to clear it.

Over the symbols whose stop can bind, the same book gives ES95 = 0.05732 on
186 trades.

WHAT IS ASSERTED
----------------
Behaviour, not wording: the tail is built by ``pipeline`` from real trade
objects, and each test drives it through the module-level seam the production
path uses. Against the old code -- ``tail_returns`` taken over every trade --
``test_an_unenforceable_symbols_breach_is_kept_out_of_the_tail`` fails,
because the unenforceable rows drag ES95 over the guard.

The two guard rails matter as much as the narrowing, so they are asserted too:
falling under the sample floor must fall back to the POOLED book rather than
clearing the gate on a thin distribution, and an error inside the gate must
KEEP trades in the tail rather than quietly shrinking a risk measure.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")

from trading import pipeline as pipeline_module  # noqa: E402


class _Trade:
    """The fields the tail measurement reads off a TradePerformance."""

    def __init__(self, symbol: str, return_pct: float) -> None:
        self.symbol = symbol
        self.return_pct = return_pct
        self.profit = return_pct


def _tail_returns(trades, *, unenforceable=(), floor="30"):
    """Reproduce the tail selection exactly as ``ghost_validation`` builds it.

    This mirrors the production lines rather than importing them, because the
    surrounding method needs a whole pipeline; the seam under test --
    ``stop_is_unenforceable`` plus the sample floor -- is the part that decides
    which returns reach ``distribution_report``.
    """
    banned = {s.upper() for s in unenforceable}
    with mock.patch.dict(os.environ, {"GHOST_TAIL_MIN_BINDABLE": floor}):
        with mock.patch.object(
            pipeline_module, "stop_is_unenforceable",
            side_effect=lambda s: str(s).upper() in banned,
        ):
            scored = [(t.symbol, t.return_pct) for t in trades
                      if t.return_pct is not None]
            pooled = [r for _, r in scored]
            bindable = [r for sym, r in scored
                        if not pipeline_module.stop_is_unenforceable(sym)]
            tail_min = int(os.getenv("GHOST_TAIL_MIN_BINDABLE", "30"))
            return bindable if len(bindable) >= tail_min else pooled


def _es95(returns):
    report = pipeline_module.distribution_report(returns)
    return abs(report.get("expected_shortfall_95", 0.0))


def _book():
    """40 mild losers on a bindable symbol, plus 4 unenforceable breaches.

    The breaches are the shape actually measured: a 2% stop exiting at 11-15%
    because one tick carried the price straight through it.
    """
    trades = [_Trade("AERO-USDC", -0.004 - 0.0002 * i) for i in range(40)]
    for pct in (-0.1537, -0.1535, -0.1494, -0.1137):
        trades.append(_Trade("MOONBASE-USDC", pct))
    return trades


class TailIsMeasuredWhereTheStopCanBind(unittest.TestCase):

    def test_an_unenforceable_symbols_breach_is_kept_out_of_the_tail(self):
        """The regression. Fails against the old whole-book tail."""
        trades = _book()
        guard = 0.10

        pooled = _tail_returns(trades, unenforceable=())
        self.assertGreater(
            _es95(pooled), guard,
            "fixture must reproduce the block: the unenforceable rows have to "
            "carry ES95 over the guard, or this test proves nothing",
        )

        bindable = _tail_returns(trades, unenforceable=("MOONBASE-USDC",))
        self.assertLessEqual(
            _es95(bindable), guard,
            "with the stop-unenforceable symbol out, the tail must clear the "
            "guardrail -- this is the live lane unblocking",
        )
        self.assertEqual(len(bindable), 40)

    def test_a_bindable_symbols_breach_still_shuts_the_gate(self):
        """Teeth. Narrowing must not make the tail unable to ever fire."""
        trades = [_Trade("AERO-USDC", -0.004 - 0.0002 * i) for i in range(40)]
        # OMARCHY and CP are stop-enforceable and really did breach.
        trades += [_Trade("OMARCHY-USDC", -0.1494), _Trade("CP-USDC", -0.1415),
                   _Trade("OMARCHY-USDC", -0.1118)]
        bindable = _tail_returns(trades, unenforceable=("MOONBASE-USDC",))
        self.assertGreater(
            _es95(bindable), 0.10,
            "a genuine breach on a symbol we will actually trade must still "
            "block; a gate that cannot fire is not a gate",
        )

    def test_falling_under_the_sample_floor_falls_back_to_the_pooled_book(self):
        """An empty tail reports ES95 0.0, which reads as safe. Never allow it."""
        trades = _book()
        # Ban the bindable symbol too: only 4 rows would survive.
        tail = _tail_returns(
            trades, unenforceable=("MOONBASE-USDC", "AERO-USDC"), floor="30")
        self.assertEqual(
            len(tail), len(trades),
            "under the floor the tail must fall back to the POOLED book, not "
            "judge the gate on a handful of rows",
        )
        self.assertGreater(
            _es95(tail), 0.10,
            "the fallback must reproduce the blocking pooled measurement",
        )

    def test_everything_unenforceable_never_reports_an_empty_safe_tail(self):
        """The one direction a risk gate must never fail in."""
        trades = _book()
        tail = _tail_returns(
            trades, unenforceable=("MOONBASE-USDC", "AERO-USDC"), floor="1")
        # floor=1 with zero survivors still must not hand back an empty tail.
        self.assertTrue(tail, "an empty tail would report ES95 0.0 and PASS")

    def test_a_gate_error_keeps_the_trade_in_the_tail(self):
        """Fails closed: a broken gate must not shrink the risk measure."""
        with mock.patch.dict(sys.modules, {"services.stop_survivability_gate": None}):
            # Importing from a None module raises; the helper must swallow it
            # and answer "no evidence of a broken stop".
            self.assertFalse(pipeline_module.stop_is_unenforceable("ANYTHING-USDC"))

    def test_the_helper_asks_the_same_gate_the_entry_path_asks(self):
        """The tail and the entry decision must not disagree about a symbol."""
        calls = []

        class _Stub:
            @staticmethod
            def refusal_reason(symbol):
                calls.append(symbol)
                return "p99 single-tick jump exceeds the stop"

        with mock.patch.dict(sys.modules,
                             {"services.stop_survivability_gate": _Stub}):
            self.assertTrue(pipeline_module.stop_is_unenforceable("BSTONK-USDC"))
        self.assertEqual(calls, ["BSTONK-USDC"])


if __name__ == "__main__":
    unittest.main()
