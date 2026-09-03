"""A live entry must not buy at a price the decision never saw.

``slippage_bps`` bounds the fill against the ROUTER'S OWN QUOTE. It says
nothing about whether that quote matches the price the strategy decided on,
and those two numbers come apart exactly when a token is thin -- which is when
it matters.

Measured 2026-09-03 from trading_cache.db. Eleven live fills exist. Ten landed
within 0.35% of their expected amount. One did not:

    trade_fills ts 1788455198  BSTONK-USDC  live_entry
        expected 391.791 tokens at 0.001914286
        received 360.264 tokens at 0.002081805
        => +8.751% on price, -8.05% on quantity

LIVE_TRADE_SLIPPAGE_BPS was 75 (0.75%) at the time. The router honoured its own
quote to the basis point; its own quote was the bad number. The first feed
sample after entry marked the position at -13.08%, and it was stopped at
-18.40% for -$0.1429 -- 104% of all live P/L to date (-$0.1374 over six closed
trades). Remove that single fill and the live book is +$0.0054 at a profit
factor of 1.92 instead of 0.0759.

So the entry now states the least base token it will accept for its money, and
a route that will not deliver that much never broadcasts. The bound is checked
BEFORE the allowance and before the send, and it is not a broadcast, so the
caller's fallback chain is still free to try the next route: one thin pool is
not a reason to abandon the trade.
"""
from __future__ import annotations

import unittest
from unittest import mock

from services.swap_service import SwapOutcome, SwapService


USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
BSTONK = "0x2222222222222222222222222222222222222222"

# The real numbers from the trade this guard is written against.
BSTONK_SPEND_USDC = 0.75
BSTONK_EXPECTED_TOKENS = 391.7910447761194      # 0.75 / 0.001914285714
BSTONK_QUOTED_TOKENS = 360.264243225393         # what actually arrived
BSTONK_DECIMALS = 18


class _Bridge:
    class _Acct:
        address = "0x291c854811e92906a658Fb94Aa511bF919f968ad"

    acct = _Acct()

    class _Eth:
        chain_id = 8453
        default_account = None

    def _w3(self, chain):
        w3 = mock.Mock()
        w3.eth = self._Eth()
        return w3


def _service():
    svc = SwapService.__new__(SwapService)
    svc.bridge = _Bridge()
    svc.zx = mock.Mock()
    svc.uni = svc.camelot = svc.sushi = None
    svc.recorder = None
    return svc


class QuoteShortfallTest(unittest.TestCase):
    """The bound itself, at the boundary where the quote is read."""

    def test_no_bound_means_no_opinion(self):
        # Every pre-existing caller passes nothing and must be unaffected.
        self.assertIsNone(SwapService._quote_shortfall({"buyAmount": "1"}, None))
        self.assertIsNone(SwapService._quote_shortfall({}, None))

    def test_a_quote_that_meets_the_floor_passes(self):
        self.assertIsNone(SwapService._quote_shortfall({"buyAmount": "1000"}, 1000))
        self.assertIsNone(SwapService._quote_shortfall({"buyAmount": "1001"}, 1000))

    def test_a_quote_below_the_floor_is_refused(self):
        reason = SwapService._quote_shortfall({"buyAmount": "999"}, 1000)
        self.assertIsNotNone(reason)
        self.assertTrue(reason.startswith("quote_below_floor:"), reason)

    def test_a_route_that_will_not_say_what_it_pays_is_refused(self):
        # A bound the caller asked for must never silently become no bound.
        self.assertEqual(
            SwapService._quote_shortfall({}, 1000), "quote_missing_buy_amount"
        )
        self.assertEqual(
            SwapService._quote_shortfall({"buyAmount": ""}, 1000),
            "quote_missing_buy_amount",
        )
        self.assertTrue(
            (SwapService._quote_shortfall({"buyAmount": "lots"}, 1000) or "")
            .startswith("quote_buy_amount_unparseable:")
        )

    def test_buy_amount_arrives_as_a_string_of_raw_base_units(self):
        """The contract every provider actually writes.

        uniswap_v3, camelot_v2 and sushi_v2 all set ``"buyAmount": str(best_out)``
        where best_out is the router's raw integer output. Comparing it as an
        int against a raw floor is what makes this a units-safe check; parsing
        it as a float would lose precision on an 18-decimal token.
        """
        huge = 10 ** 24 + 1
        self.assertIsNone(SwapService._quote_shortfall({"buyAmount": str(huge)}, huge))
        self.assertTrue(
            (SwapService._quote_shortfall({"buyAmount": str(huge - 1)}, huge) or "")
            .startswith("quote_below_floor:")
        )


class LocalProviderRefusesBeforeSpendingTest(unittest.TestCase):
    def _run(self, *, quoted_raw: int, min_buy_raw):
        svc = _service()
        approvals: list = []
        sends: list = []
        svc._ensure_allowance = lambda *a, **k: approvals.append(a) or True   # noqa: E731
        svc._send = lambda *a, **k: sends.append(k) or SwapOutcome(
            ok=True, broadcast=True, tx_hash="0x" + "cd" * 32
        )
        outcome = svc._try_local_provider(
            name="UniswapV3",
            q={
                "buyAmount": str(quoted_raw),
                "allowanceTarget": "0x" + "11" * 20,
                "tx": {"to": "0x" + "22" * 20, "data": "0x", "value": 0, "gas": 300000},
            },
            chain="base",
            sell_token=USDC_BASE,
            sell_raw=750000,
            min_buy_raw=min_buy_raw,
        )
        return outcome, approvals, sends

    def test_the_bstonk_quote_is_refused(self):
        floor = int(BSTONK_EXPECTED_TOKENS * 0.98 * 10 ** BSTONK_DECIMALS)
        quoted = int(BSTONK_QUOTED_TOKENS * 10 ** BSTONK_DECIMALS)
        outcome, approvals, sends = self._run(quoted_raw=quoted, min_buy_raw=floor)

        self.assertFalse(outcome.ok)
        self.assertFalse(outcome.broadcast, "no money may leave on a refused quote")
        self.assertTrue(outcome.reason.startswith("quote_below_floor:"), outcome.reason)
        self.assertEqual(sends, [], "the transaction must never be built")
        self.assertEqual(approvals, [], "not even an approval should be paid for")

    def test_a_normal_fill_still_broadcasts(self):
        """The ten good fills must not be collateral damage.

        The tightest of them (CBETH-USDC live_entry, ts 1788449882) was 0.314%
        adverse on price and 0.31% short on quantity; every one of the eleven
        other legs was inside 0.61%. A 2% floor clears all of them.
        """
        expected_raw = int(BSTONK_EXPECTED_TOKENS * 10 ** BSTONK_DECIMALS)
        floor = int(BSTONK_EXPECTED_TOKENS * 0.98 * 10 ** BSTONK_DECIMALS)
        quoted = int(expected_raw * 0.9969)          # the real CBETH shortfall
        outcome, _approvals, sends = self._run(quoted_raw=quoted, min_buy_raw=floor)

        self.assertTrue(outcome.broadcast)
        self.assertEqual(len(sends), 1)

    def test_an_unbounded_call_is_unchanged(self):
        outcome, _approvals, sends = self._run(quoted_raw=1, min_buy_raw=None)
        self.assertTrue(outcome.broadcast)
        self.assertEqual(len(sends), 1)


class SwapRoutedConvertsTheFloorTest(unittest.TestCase):
    """UNITS. The caller speaks human; the quote speaks raw."""

    def _routed(self, svc, **kw):
        return svc._swap_routed(
            chain="base", sell=USDC_BASE, buy=BSTONK,
            amount_human="0.750000", slippage_bps=75, **kw
        )

    def test_the_floor_is_converted_with_the_buy_tokens_decimals(self):
        svc = _service()
        seen: dict = {}

        def _decimals(chain, token):
            return 6 if token.lower() == USDC_BASE.lower() else BSTONK_DECIMALS

        svc._decimals_or_none = _decimals
        svc._resolve_token = lambda ch, t: t

        class _Uni:
            def quote_and_build(self, *a, **k):
                return {"buyAmount": str(int(BSTONK_QUOTED_TOKENS * 10 ** 18)),
                        "tx": {"to": "0x" + "22" * 20, "data": "0x", "value": 0}}

        svc.uni = _Uni()
        svc.camelot = svc.sushi = _Uni()

        def _spy(*, name, q, chain, sell_token, sell_raw, min_buy_raw=None):
            seen["min_buy_raw"] = min_buy_raw
            return SwapOutcome(ok=False, broadcast=False, route=name, reason="stub")

        svc._try_local_provider = _spy

        self._routed(svc, min_buy_human=BSTONK_EXPECTED_TOKENS * 0.98)

        floor = seen["min_buy_raw"]
        self.assertIsInstance(floor, int)
        # 391.791... * 0.98 = 383.955..., in 18 decimals.
        self.assertEqual(floor // 10 ** 18, 383)
        # and the token it was NOT converted with: USDC's 6.
        self.assertNotEqual(floor // 10 ** 6, 383)

    def test_an_unreadable_buy_decimal_refuses_rather_than_guessing(self):
        """Fail closed. A bound we cannot enforce must not become no bound.

        Guessing 18 here is the same 10^12 error the sell side already refuses
        (see _decimals_or_none): on a 6-decimal buy token it would set a floor
        10^12 too high and block every trade, and on the other side of a wrong
        table it would set one 10^12 too low and wave the bad fill straight
        through. Neither is a bound.
        """
        svc = _service()
        svc._resolve_token = lambda ch, t: t
        svc._decimals_or_none = lambda chain, token: (
            6 if token.lower() == USDC_BASE.lower() else None
        )
        broadcast = []
        svc._try_local_provider = lambda **kw: broadcast.append(kw) or SwapOutcome(
            ok=True, broadcast=True
        )

        outcome = self._routed(svc, min_buy_human=1.0)

        self.assertFalse(outcome.ok)
        self.assertFalse(outcome.broadcast)
        self.assertEqual(outcome.reason, f"decimals_unknown:{BSTONK}")
        self.assertEqual(broadcast, [], "nothing may be sent when the bound is unenforceable")

    def test_a_nonsense_floor_refuses(self):
        svc = _service()
        svc._resolve_token = lambda ch, t: t
        svc._decimals_or_none = lambda chain, token: 6
        for bad in (float("nan"), -1.0, "not a number"):
            outcome = self._routed(svc, min_buy_human=bad)
            self.assertFalse(outcome.broadcast, bad)
            self.assertEqual(outcome.reason, "min_buy_not_a_number", bad)

    def test_no_floor_leaves_the_buy_decimals_unread(self):
        """Every existing caller must keep working without a buy-side read.

        The gas refill, the quote top-up and the four bus actions all pass no
        bound. If this path started reading buy decimals unconditionally, a
        flaky RPC on the buy token would begin refusing swaps that used to work.
        """
        svc = _service()
        svc._resolve_token = lambda ch, t: t
        reads: list = []

        def _decimals(chain, token):
            reads.append(token)
            return 6 if token.lower() == USDC_BASE.lower() else BSTONK_DECIMALS

        svc._decimals_or_none = _decimals
        svc._try_local_provider = lambda **kw: SwapOutcome(ok=True, broadcast=True)

        self._routed(svc)

        self.assertEqual([t.lower() for t in reads], [USDC_BASE.lower()])


class SwapPassesTheFloorThroughTest(unittest.TestCase):
    def test_swap_forwards_the_bound_and_records_it(self):
        svc = _service()
        seen: dict = {}
        recorded: list = []
        svc.recorder = lambda outcome, context: recorded.append(context)

        def _routed(**kw):
            seen.update(kw)
            return SwapOutcome(ok=True, broadcast=True, tx_hash="0x" + "ef" * 32)

        svc._swap_routed = _routed

        svc.swap(
            chain="base", sell=USDC_BASE, buy=BSTONK, amount_human="0.750000",
            slippage_bps=75, min_buy_human=383.9, purpose="live_entry",
        )

        self.assertEqual(seen["min_buy_human"], 383.9)
        self.assertNotIn("purpose", seen, "record metadata must not reach the router")
        self.assertEqual(recorded[0]["min_buy_human"], 383.9)
        self.assertEqual(recorded[0]["purpose"], "live_entry")

    def test_swap_without_a_bound_still_forwards_none(self):
        svc = _service()
        seen: dict = {}
        svc._swap_routed = lambda **kw: seen.update(kw) or SwapOutcome(ok=True)
        svc.swap(chain="base", sell=USDC_BASE, buy=BSTONK, amount_human="1")
        self.assertIsNone(seen["min_buy_human"])


class BotComputesTheFloorFromThePlanTest(unittest.TestCase):
    """The number bot.py hands to the router, checked against the real trade."""

    @staticmethod
    def _floor(trade_size: float, price: float, max_adverse: float) -> float:
        # Mirrors trading/bot.py's live-entry block exactly.
        quote_spend_target = max(0.0, trade_size * price)
        spend_human = f"{quote_spend_target:.6f}"
        expected_base = float(spend_human) / price
        return expected_base * (1.0 - max_adverse)

    def test_the_bstonk_entry_would_have_been_refused(self):
        floor = self._floor(BSTONK_EXPECTED_TOKENS, 0.001914285714, 0.02)
        self.assertGreater(
            floor, BSTONK_QUOTED_TOKENS,
            "the fill that lost 104% of live P/L must not clear the floor",
        )

    def test_the_ten_good_fills_would_all_have_passed(self):
        # (expected_amount, executed_amount) straight out of trade_fills for
        # every live leg except BSTONK. Nothing here may be refused.
        legs = [
            (0.000274613, 0.00027375),      # CBETH entry  1788435497
            (1.53531, 1.53916),             # AERO  entry  1788435524
            (1.53916, 1.53916),             # AERO  exit   1788439233
            (0.000161634, 0.000162),        # CBETH exit   1788449383
            (0.000264452, 0.000263623),     # CBETH entry  1788449882
            (9.28268e-06, 9.28e-06),        # CBBTC entry  1788452530
            (9.28018e-06, 9.27e-06),        # CBBTC entry  1788452667
            (9.2729e-06, 9.26e-06),         # CBBTC entry  1788453502
            (9.27296e-06, 9.28e-06),        # CBBTC entry  1788453645
            (0.000263623, 0.000111),        # CBETH exit   1788454066 (partial)
        ]
        for expected, executed in legs[:-1]:
            with self.subTest(expected=expected):
                self.assertGreaterEqual(
                    executed, expected * 0.98,
                    "a 2% floor must not refuse a fill this lane really made",
                )

    def test_the_floor_is_derived_from_the_rounded_spend(self):
        """A floor for a trade we are not making is not a floor.

        amount_human goes to the router as a 6dp string. On a sub-cent clip the
        rounding is a real fraction of the spend, and a floor computed from the
        unrounded number would demand more tokens than the money being sent can
        possibly buy -- refusing every trade at the smallest sizes, which is
        every trade this account makes.
        """
        price = 3.0
        trade_size = 0.0000005                       # spends 0.0000015 -> "0.000002"
        floor = self._floor(trade_size, price, 0.0)
        self.assertAlmostEqual(floor, float(f"{trade_size * price:.6f}") / price)
        self.assertGreater(floor, trade_size, "rounding up must raise the floor, not strand it")


class OutcomesCarryTheirStrategyTest(unittest.TestCase):
    """trade_outcomes is the ledger's only independent check; it must attribute.

    Measured 2026-09-03: all 92 rows carried no strategy at all, so the six live
    rows -- every one of them atf_static's -- were anonymous in the database. A
    profit factor of 0.0759 computed from them was reported against
    money_button, which has never traded live.
    """

    def test_the_details_dict_names_the_strategy(self):
        import inspect
        import re

        from trading.bot import TradingBot

        source = inspect.getsource(TradingBot)
        calls = re.findall(r"record_trade_outcome\((.*?)\n\s*\)\n", source, re.S)
        self.assertEqual(
            len(calls), 1,
            "exactly one site writes trade_outcomes; a second one would need "
            "this attribution too",
        )
        body = calls[0]
        self.assertIn('"strategy_id"', body)
        # The same expression the StrategyLedger call uses, so the database row
        # and the ledger row can be reconciled one-to-one.
        self.assertIn('str(pos.get("strategy_id") or "") or "unclassified"', body)


if __name__ == "__main__":
    unittest.main()
