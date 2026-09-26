"""The scout must BOOK its round-trip fee per symbol, not just GATE on it.

THE DEFECT THIS PINS, measured 2026-09-25 (Dune, pass 708)
----------------------------------------------------------
``services/atf_static_strategy.py`` asks ``services.roundtrip_cost`` what a
round trip costs in four places. On 2026-09-17 the symbol was threaded into
ONE of them -- the exit GATE, ``cost_rate = _roundtrip_cost_rate(clip_usd,
symbol)`` -- and the module's own comment there explains exactly why it must
be symbol-aware ("2.01bp on WETH and 3301.53bp on COMP ... a factor of
1,642").

The other three were left asking blind:

    profit_usd = _net_profit_usd(profit, clip_usd)          # the NET P/L
    "roundtrip_cost_usd": _roundtrip_cost_usd(clip_usd)     # the details blob
    fee_cost=_roundtrip_cost_usd(clip_usd)                  # the ledger row

So the scout DECIDED to exit on this symbol's real cost and then WROTE the
trade to the book charged at a token-blind constant. Every closed
``atf_static_scout`` row in ``trade_outcomes`` carries a ``fee_cost`` of
exactly $0.023169 -- which is $0.004047 + $0.003187 x the $6.00 ghost clip,
the blind default to the last digit -- regardless of what it traded.

WHICH DIRECTION IT WENT, because this is the part that matters: the blind
constant is not conservative. Over the ten closed scout rows (5 AERO-USDC,
3 SR-USDC, 2 VIRTUAL-USDC) the book charged $0.231690 and the symbol-aware
cost of those same ten round trips is $0.345770. The scout's ghost P/L was
FLATTERED by $0.114080, driven by VIRTUAL-USDC at $0.112574 a round trip
(187.6bp at a $6 clip) booked as if it were $0.023169 (38.6bp). AERO ran the
other way -- $0.010223 real against $0.023169 booked -- and did not cover it.

That is a strategy in the graduation table being helped by a fee it never
paid, which is the failure mode this repo has been bitten by before.

WHY THIS TEST IS STRUCTURAL. The three sites live inside the exit loop, which
writes to the trading store. Asserting on the CALL rather than on a round
trip through the database keeps the test honest about what it checks: it
pins the argument, not the arithmetic. ``test_roundtrip_cost_rate_is_per_
symbol.py`` already covers the arithmetic, and the invariant
``net_profit == gross_profit - fee_cost`` holds by construction in
``services/roundtrip_cost.py`` only when both sides are asked the same way --
which is precisely what regressed here.
"""

from __future__ import annotations

import ast
import os
import unittest

_MODULE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "services",
    "atf_static_strategy.py",
)

#: Every costing helper the scout imports from ``services.roundtrip_cost``
#: whose FIRST argument is a notional and whose LAST is the symbol. The
#: symbol is keyword-or-positional in all three, so accept either.
_COSTING_CALLS = {
    "_roundtrip_cost_usd": 2,
    "_roundtrip_cost_rate": 2,
    "_net_profit_usd": 3,
}


def _calls_outside_the_fallback_shims(tree: ast.Module) -> list[ast.Call]:
    """Every call to a costing helper that is not inside its own shim def.

    The ``except ImportError`` block at the top of the module defines
    fallback shims with these same names, and ``_roundtrip_cost_rate``'s
    shim legitimately calls ``_roundtrip_cost_usd(n)`` with one argument --
    it has no census to consult, which is the whole point of a shim. Those
    definitions are not booking sites and must not be pinned here.
    """
    shim_bodies: list[ast.AST] = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in _COSTING_CALLS
    ]
    inside_a_shim = {id(n) for body in shim_bodies for n in ast.walk(body)}
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _COSTING_CALLS
        and id(node) not in inside_a_shim
    ]


class TestTheScoutPricesEveryCostCallBySymbol(unittest.TestCase):
    def setUp(self) -> None:
        with open(_MODULE, "r", encoding="utf-8") as handle:
            self.source = handle.read()
        self.tree = ast.parse(self.source)

    def test_there_is_something_to_check(self) -> None:
        """Guard against the test passing because it found nothing.

        An AST test that silently matches zero nodes is worse than no test.
        """
        calls = _calls_outside_the_fallback_shims(self.tree)
        self.assertGreaterEqual(
            len(calls),
            4,
            "expected at least the four costing call sites (one gate, three "
            f"booking) outside the fallback shims; found {len(calls)}. If the "
            "module was restructured, fix this test rather than deleting it.",
        )

    def test_no_costing_call_asks_without_the_symbol(self) -> None:
        blind: list[str] = []
        for node in _calls_outside_the_fallback_shims(self.tree):
            name = node.func.id  # type: ignore[union-attr]
            supplied = len(node.args) + len(node.keywords)
            if supplied < _COSTING_CALLS[name]:
                blind.append(
                    f"{os.path.basename(_MODULE)}:{node.lineno} "
                    f"{name}(...) got {supplied} arg(s), needs "
                    f"{_COSTING_CALLS[name]} -- the symbol is missing"
                )
        self.assertEqual(
            [],
            blind,
            "A costing call in the scout is asking token-blind. That is not a "
            "conservative default: measured over the ten closed scout rows it "
            "UNDER-charged the book by $0.114080, because VIRTUAL-USDC costs "
            "$0.112574 a round trip at the $6 ghost clip and the blind "
            "constant is $0.023169. Pass the symbol.\n  " + "\n  ".join(blind),
        )

    def test_the_gate_and_the_booking_agree(self) -> None:
        """net == gross - fee only holds if both sides ask the same question.

        This is the invariant the module's own comment at the ledger write
        claims holds "by construction". It does -- but only when the fee in
        ``fee_cost`` and the fee subtracted inside ``_net_profit_usd`` are
        priced for the same symbol. Booking one blind and the other aware is
        exactly how that construction breaks.
        """
        from services.roundtrip_cost import net_profit_usd, roundtrip_cost_usd

        clip = 6.0
        for symbol in ("AERO-USDC", "VIRTUAL-USDC", "WETH-USDC", "SR-USDC"):
            for return_pct in (0.01, -0.004, 0.0):
                gross = return_pct * clip
                fee = roundtrip_cost_usd(clip, symbol)
                net = net_profit_usd(return_pct, clip, symbol)
                self.assertAlmostEqual(
                    net,
                    gross - fee,
                    places=12,
                    msg=f"{symbol} at {return_pct}: net {net} != gross {gross} - fee {fee}",
                )

    def test_the_blind_default_is_not_conservative_for_what_the_scout_trades(
        self,
    ) -> None:
        """The reason the missing symbol is a bug and not a rounding choice.

        If the token-blind constant were an over-estimate for every symbol
        the scout actually trades, booking blind would merely be pessimistic
        and this would be a tidy-up. It is not: VIRTUAL-USDC, which the scout
        has closed rows in, costs nearly 5x the constant.
        """
        from services.roundtrip_cost import roundtrip_cost_usd

        clip = 6.0
        blind = roundtrip_cost_usd(clip, None)
        virtual = roundtrip_cost_usd(clip, "VIRTUAL-USDC")
        self.assertGreater(
            virtual,
            blind,
            "VIRTUAL-USDC was measured at $0.112574 a round trip against the "
            f"$0.023169 blind default; got virtual={virtual} blind={blind}. "
            "If the census has been re-cut and this no longer holds, re-run "
            "the ten-row reconciliation before relaxing the claim.",
        )


if __name__ == "__main__":
    unittest.main()
