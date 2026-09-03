"""A guessed decimals is a 10^12 error booked as a price.

Measured 2026-09-03. Two real swaps settled on base a minute apart, both
selling USDC:

    11:36  CBETH-USDC  tx 0x076978740803789cd40564cb150753bc075a5f6600d8422add58a4720822b82b  entry price 2.739721277650459e-09
    11:37  AERO-USDC   tx 0x62cafa4ccb0c7a34c7c8a767ac569832f69601f3fb71d396b4813f104acfb0c2  entry price 0.48727988389842986

The feed carried CBETH-USDC at $2731.12 in that same minute and AERO-USDC at
$0.48747. AERO is right. cbETH is wrong by exactly 10^12::

    0.75 USDC / 0.000273750474589586 cbETH = 2739.72      the real price
    2739.72e-12                             = 2.7397e-09   what was booked
    10^-12 = 10^-(18 - 6)                                  USDC read as 18

``executed_entry_price = quote_spent / base_received`` in trading/bot.py, and
the quantity was right, so the corruption is in ``quote_spent``: the receipt
parser was handed 18 decimals for a 6-decimal token and read 0.75 USDC as
7.5e-13.

Two layers each turned a failed read into the number 18 --
``router_wallet.erc20_decimals`` (``except: return 18``) and
``SwapService._decimals`` (``except: return 18``) -- so by the time the value
reached ``parse_fill_from_receipt`` there was nothing left to distinguish "the
contract says 18" from "nobody answered". Base RPC flakiness is established
here; all five configured endpoints once refused a receipt read.

The damage lands after the money has moved. An entry price 12 orders of
magnitude low turns a $0.75 position into a ~1e12x return when it exits, and
that record goes to the ledger that decides graduation. This repo has already
purged four strategies for fabricated records.

token_decimals.py exists for exactly this and names the failure in its own
docstring; the swap path simply never consulted it.
"""

from __future__ import annotations

import unittest
import unittest.mock

from services.fill_receipt import ReceiptFill
from services.swap_service import SwapService
from token_decimals import known_token_decimals

USDC_BASE = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
CBETH_BASE = "0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22"
AERO_BASE = "0x940181a94A35A4569E4529A3CDfB74e38FD98631"

#: The real trade, from trading_ops row 64482.
CBETH_FILL_QTY = 0.000273750474589586
CBETH_USD = 0.75
#: market_stream, base, 2026-09-03 11:36.
CBETH_FEED_PRICE = 2731.12


class _DeadRPCBridge:
    """Every contract read fails. This is the state that corrupted the trade."""

    def __init__(self):
        self.calls = 0

    def _w3(self, chain):
        return object()

    def _erc20(self, w3, token):
        self.calls += 1
        raise RuntimeError("base RPC refused")

    def erc20_decimals(self, chain, token):
        # The real bridge swallows the failure and answers 18.
        return 18


class _LiveBridge:
    """A working contract read, for the tokens not in the table."""

    def __init__(self, decimals):
        self._decimals_by_token = decimals
        self.calls = 0

    def _w3(self, chain):
        return object()

    def _erc20(self, w3, token):
        self.calls += 1
        value = self._decimals_by_token[token.lower()]

        class _Fn:
            def call(self_inner):
                return value

        class _Fns:
            def decimals(self_inner):
                return _Fn()

        class _C:
            functions = _Fns()

        return _C()

    def erc20_decimals(self, chain, token):
        return self._decimals_by_token[token.lower()]


def _service(bridge, *, rpc_urls=()):
    """A SwapService with no network of its own.

    ``_decimals_or_none`` asks every configured RPC when the bridge's own
    contract read fails, so the endpoint list is stubbed here: an empty one
    means "no endpoint answered", which is what these tests are asserting
    about, and it keeps them off the network. Tests that want the RPC rescue
    pass their own fake list and stub ``requests.post``.
    """
    service = SwapService.__new__(SwapService)
    service.bridge = bridge
    service._rpc_urls = lambda chain: list(rpc_urls)
    return service


def _clear_decimals_cache():
    """Measured decimals are cached per process; tests must not inherit them."""
    from services import swap_service as _mod

    _mod._DECIMALS_MEASURED.clear()


class TheCorruptedTradeIsArithmeticTest(unittest.TestCase):
    """Pin the numbers, so the size of the error cannot be argued about."""

    def test_the_booked_price_is_the_real_price_over_ten_to_the_twelve(self):
        real_price = CBETH_USD / CBETH_FILL_QTY
        self.assertAlmostEqual(real_price, 2739.72, places=1)
        booked = 2.739721277650459e-09
        self.assertAlmostEqual(booked * 1e12, real_price, places=6)

    def test_the_real_price_agrees_with_the_feed_and_the_booked_one_does_not(self):
        real_price = CBETH_USD / CBETH_FILL_QTY
        self.assertLess(abs(real_price - CBETH_FEED_PRICE) / CBETH_FEED_PRICE, 0.01)
        booked = 2.739721277650459e-09
        self.assertGreater(CBETH_FEED_PRICE / booked, 1e11)

    def test_the_factor_is_the_decimals_difference(self):
        """18 - 6, i.e. USDC measured as an 18-decimal token."""
        self.assertEqual(10 ** (18 - 6), 10**12)


class DecimalsComeFromTheTableFirstTest(unittest.TestCase):
    """The corrupting token never needs an RPC call again."""

    def setUp(self):
        _clear_decimals_cache()

    def test_usdc_on_base_resolves_without_touching_the_chain(self):
        bridge = _DeadRPCBridge()
        service = _service(bridge)
        self.assertEqual(service._decimals_or_none("base", USDC_BASE), 6)
        self.assertEqual(
            bridge.calls, 0, "USDC decimals must not depend on an RPC read"
        )

    def test_the_two_tokens_actually_held_are_in_the_table(self):
        """Both were confirmed on-chain 2026-09-03 via base-rpc.publicnode.com."""
        self.assertEqual(known_token_decimals("base", CBETH_BASE), 18)
        self.assertEqual(known_token_decimals("base", AERO_BASE), 18)

    def test_case_does_not_matter(self):
        self.assertEqual(known_token_decimals("base", CBETH_BASE.lower()), 18)
        self.assertEqual(known_token_decimals("base", CBETH_BASE.upper()), 18)


class UnreadableDecimalsAreUnknownNotEighteenTest(unittest.TestCase):
    """The half that matters: a failed read must not arrive as a number."""

    def setUp(self):
        _clear_decimals_cache()

    def test_an_unlisted_token_with_a_dead_rpc_reads_unknown(self):
        service = _service(_DeadRPCBridge())
        self.assertIsNone(service._decimals_or_none("base", "0xdeadbeef" + "0" * 32))

    def test_a_working_read_is_still_used_for_unlisted_tokens(self):
        token = "0xfeed" + "0" * 36
        service = _service(_LiveBridge({token: 8}))
        self.assertEqual(service._decimals_or_none("base", token), 8)

    def test_a_genuine_eighteen_is_not_mistaken_for_a_failure(self):
        """The availability half: legitimate 18-decimal tokens must still trade."""
        token = "0xfeed" + "0" * 36
        bridge = _LiveBridge({token: 18})
        service = _service(bridge)
        self.assertEqual(service._decimals_or_none("base", token), 18)
        self.assertEqual(
            bridge.calls, 1, "one RPC call, the same as before the fix"
        )

    def test_native_is_still_eighteen(self):
        service = _service(_DeadRPCBridge())
        self.assertEqual(service._decimals_or_none("base", "native"), 18)


class ReadFillRefusesRatherThanBookAGuessTest(unittest.TestCase):
    """ok=False falls back to the wallet delta; ok=True with a guess does not."""

    def setUp(self):
        _clear_decimals_cache()

    def test_unknown_decimals_produce_an_unreadable_fill(self):
        service = _service(_DeadRPCBridge())
        unlisted = "0xdeadbeef" + "0" * 32
        fill = service.read_fill(
            "base",
            "0xabc",
            sell=unlisted,
            buy=unlisted,
            wallet="0x291c854811e92906a658Fb94Aa511bF919f968ad",
            receipt={"logs": [], "status": 1},
        )
        self.assertIsInstance(fill, ReceiptFill)
        self.assertFalse(fill.ok)
        self.assertTrue(
            fill.reason.startswith("decimals_unknown"),
            f"expected decimals_unknown, got {fill.reason!r}",
        )

    def test_a_readable_pair_is_not_refused(self):
        """The fix must not turn every fill unreadable."""
        service = _service(_DeadRPCBridge())
        fill = service.read_fill(
            "base",
            "0xabc",
            sell=USDC_BASE,
            buy=CBETH_BASE,
            wallet="0x291c854811e92906a658Fb94Aa511bF919f968ad",
            receipt={"logs": [], "status": 1},
        )
        self.assertIsInstance(fill, ReceiptFill)
        self.assertNotIn("decimals_unknown", fill.reason or "")


class OneFlakyEndpointMustNotCostAPositionTest(unittest.TestCase):
    """Refusing is right; refusing when the chain WOULD have answered is not.

    2026-09-03 11:36:33, from data/production.log::

        live-swap: entry fill unreadable from receipt 0x9088fe4e0c8041d7210
        81cbb83822821bafaaba7ac9664c15b163436150aa6c7
        (decimals_unknown:0xB2000000000000000000004c27f6523082f41D01)

    That swap had SETTLED. Read back from its own receipt: 0.75 USDC out,
    18609003629119603875 raw BASECAT in, status 0x1. It was booked
    `live-entry-failed / no_fill_detected` and $0.75 of BASECAT was left
    on-chain with no position pointing at it.

    The contract was never the problem -- base-rpc.publicnode.com and
    mainnet.base.org both answered 0x12 for that exact address minutes later.
    A single endpoint was flaky, and `_decimals_or_none` asked exactly one.
    """

    def setUp(self):
        _clear_decimals_cache()

    def _post(self, answers):
        """A fake requests.post: `answers` maps url -> result string or None."""
        class _Resp:
            def __init__(self, status, body):
                self.status_code = status
                self._body = body

            def json(self):
                return self._body

        def post(url, json=None, timeout=None, verify=None):
            answer = answers.get(url)
            if answer is None:
                raise RuntimeError("endpoint down")
            return _Resp(200, {"jsonrpc": "2.0", "id": 1, "result": answer})

        return post

    def _patched(self, service, answers):
        import requests

        return unittest.mock.patch.object(
            requests, "post", self._post(answers)
        )

    def test_a_second_endpoint_rescues_the_read(self):
        token = "0xb2000000000000000000004c27f6523082f41d01"
        service = _service(_DeadRPCBridge(), rpc_urls=("http://down", "http://up"))
        answers = {"http://up": "0x" + "0" * 62 + "12"}  # 0x12 = 18
        with self._patched(service, answers):
            self.assertEqual(service._decimals_or_none("base", token), 18)

    def test_no_endpoint_answering_is_still_unknown(self):
        """The half that must not regress: unknown stays unknown."""
        token = "0xdeadbeef" + "0" * 32
        service = _service(_DeadRPCBridge(), rpc_urls=("http://down",))
        with self._patched(service, {}):
            self.assertIsNone(service._decimals_or_none("base", token))

    def test_an_address_with_no_code_is_unknown_not_zero(self):
        """`0x` is an empty return, not a 0-decimal token. Booking a fill with
        0 decimals is the same class of error as booking it with 18."""
        token = "0xdeadbeef" + "0" * 32
        service = _service(_DeadRPCBridge(), rpc_urls=("http://up",))
        with self._patched(service, {"http://up": "0x"}):
            self.assertIsNone(service._decimals_or_none("base", token))

    def test_an_out_of_range_answer_is_refused(self):
        """ERC-20 decimals is a uint8; 2^160 is a malformed reply."""
        token = "0xdeadbeef" + "0" * 32
        service = _service(_DeadRPCBridge(), rpc_urls=("http://up",))
        with self._patched(service, {"http://up": "0x" + "f" * 64}):
            self.assertIsNone(service._decimals_or_none("base", token))

    def test_a_measured_answer_is_cached_and_a_failure_is_not(self):
        token = "0xb2000000000000000000004c27f6523082f41d01"
        service = _service(_DeadRPCBridge(), rpc_urls=("http://up",))
        with self._patched(service, {"http://up": "0x" + "0" * 62 + "12"}):
            self.assertEqual(service._decimals_or_none("base", token), 18)
        # Second call answers from the cache with every endpoint gone.
        dead = _service(_DeadRPCBridge(), rpc_urls=())
        self.assertEqual(dead._decimals_or_none("base", token), 18)

        _clear_decimals_cache()
        other = "0xdeadbeef" + "0" * 32
        with self._patched(service, {}):
            self.assertIsNone(service._decimals_or_none("base", other))
        # A failed read says nothing about the token, so nothing was learned.
        with self._patched(service, {"http://up": "0x" + "0" * 62 + "08"}):
            self.assertEqual(service._decimals_or_none("base", other), 8)

    def test_the_table_still_wins_and_costs_no_request(self):
        service = _service(_DeadRPCBridge(), rpc_urls=("http://up",))
        called = []

        import requests

        def post(*args, **kwargs):
            called.append(args)
            raise AssertionError("USDC must never need an RPC read")

        with unittest.mock.patch.object(requests, "post", post):
            self.assertEqual(service._decimals_or_none("base", USDC_BASE), 6)
        self.assertEqual(called, [])


class SwapSizingRefusesOnUnknownDecimalsTest(unittest.TestCase):
    """The same guess in the direction that spends money."""

    def test_the_sizing_path_reads_the_refusing_accessor(self):
        import inspect

        source = inspect.getsource(SwapService._swap_routed)
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertIn("dec = self._decimals_or_none(ch, sell)", code)
        self.assertIn('reason=f"decimals_unknown:{sell}"', code)


if __name__ == "__main__":
    unittest.main()
