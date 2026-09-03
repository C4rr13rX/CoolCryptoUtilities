"""A guessed decimals told the live-sizing plan the wallet held $0.00.

Measured 2026-09-03. The wallet held 3.687393 USDC on base -- confirmed from
two independent endpoints (mainnet.base.org, base-rpc.publicnode.com), both
returning ``balanceOf = 3687393`` and ``decimals() = 6``. The ``balances``
table held::

    balance_hex 0x3843e1   (= 3687393, correct)
    decimals    18         (WRONG -- USDC is 6)
    quantity    3.687393E-12

3687393 / 10^18 instead of / 10^6. $3.69 of stables read as
$0.0000000000037, and the consequence was not cosmetic::

    stable_usd            0.00      (should be 3.687393)
    deployable_stable     0.00      = stable_usd - deficit - native_buffer_gap
    recommended_live_usd  0.00      = ratio * deployable_stable
    block_reason          min_clip

``_build_transition_plan`` refused every live trade as "below the $0.75
minimum clip" against a wallet that could afford that clip five times over.
Every risk gate upstream had already passed -- tail_risk, profit_factor,
payoff_ratio, net_expectancy, loss_rate -- so this one wrong integer was the
whole distance between a ready strategy and a live trade.

The writer was ``RealtimeBalanceRefresher._fetch_erc20``, which sized the
stored quantity from ``bridge.erc20_decimals``. That accessor answers 18 both
when a token really has 18 decimals and when the RPC call failed
(``except: return 18``), so a transient base RPC outage was indistinguishable
from a measurement. Base RPC flakiness is established here: all five
configured endpoints once refused a receipt read.

``balances.py`` had already been hardened against this exact bug (its own
``_erc20_decimals`` returns None on failure, and it applies the canonical
override from token_decimals.py). The fast-refresh path was a second writer to
the same table that never got the fix, so it silently re-poisoned the rows the
repair script had cleaned.

Two rules, pinned below:
  1. The authoritative table answers BEFORE any RPC, so the corruptible tokens
     have no RPC in their path at all.
  2. When nothing can answer, the row is REFUSED, not written from a guess. A
     stale balance is recoverable; one deflated by 10^12 spends real money.
"""

from __future__ import annotations

import unittest
from decimal import Decimal

from services.wallet_optimizer import RealtimeBalanceRefresher, _to_decimal_string
from token_decimals import known_token_decimals

USDC_BASE = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
USDT_BASE = "0xfde4c96c8593536e31f229ea8f37b2ada2699bb2"

#: eth_call balanceOf, base, 2026-09-03. Agreed by mainnet.base.org and
#: base-rpc.publicnode.com.
USDC_RAW_ON_CHAIN = 3687393
USDC_HUMAN_ON_CHAIN = Decimal("3.687393")

#: The clip the plan must be able to fund from that balance.
MIN_CLIP_USD = 0.75


ASOF_BLOCK = 50830866


class _Eth:
    block_number = ASOF_BLOCK


class _W3:
    eth = _Eth()


class _RefusingDecimals:
    """``decimals()`` reverts; everything else on the node still works.

    This is the shape of the real failure. The endpoint was reachable enough
    to answer ``balanceOf`` and ``eth_blockNumber`` -- the stored row carried
    the correct ``balance_hex 0x3843e1`` and a live ``asof_block`` -- while the
    ``decimals()`` read came back empty. A bridge that is dead in every
    direction would never have written a row at all.
    """

    def __init__(self, outer):
        self._outer = outer

    def call(self):
        self._outer.raw_decimals_calls += 1
        raise RuntimeError("base RPC refused decimals()")


class _Functions:
    def __init__(self, outer):
        self._outer = outer

    def decimals(self):
        return _RefusingDecimals(self._outer)


class _Contract:
    def __init__(self, outer):
        self.functions = _Functions(outer)


class _DeadRPCBridge:
    """The node answers balances and blocks, but not ``decimals()``."""

    def __init__(self, raw=USDC_RAW_ON_CHAIN, symbol="USDC"):
        self._raw = raw
        self._symbol = symbol
        self.decimals_calls = []
        self.raw_decimals_calls = 0

    def _w3(self, chain):
        return _W3()

    def _erc20(self, w3, token):
        return _Contract(self)

    def erc20_balance_of(self, chain, token, owner):
        return self._raw

    def erc20_decimals(self, chain, token):
        # The real bridge swallows the failure and answers 18. If anything
        # reaches this, the guess is back.
        self.decimals_calls.append(token)
        return 18

    def erc20_symbol(self, chain, token):
        return self._symbol

    def erc20_name(self, chain, token):
        return self._symbol


def _refresher(bridge):
    r = RealtimeBalanceRefresher.__new__(RealtimeBalanceRefresher)
    r.bridge = bridge
    return r


class KnownStablesNeverNeedAnRPCTest(unittest.TestCase):
    """Rule 1: the table answers first, so the RPC cannot corrupt these."""

    def test_usdc_on_base_resolves_to_six_with_every_endpoint_down(self):
        bridge = _DeadRPCBridge()
        self.assertEqual(_refresher(bridge)._erc20_decimals("base", USDC_BASE, "USDC"), 6)
        # ...and it never fell through to the accessor that guesses 18.
        self.assertEqual(bridge.decimals_calls, [])

    def test_the_quantity_that_reaches_the_table_matches_the_chain(self):
        bridge = _DeadRPCBridge()
        row = _refresher(bridge)._fetch_erc20("base", USDC_BASE, "0xwallet", {})
        self.assertIsNotNone(row, "a known stable must never be refused")
        self.assertEqual(row["decimals"], 6)
        self.assertEqual(Decimal(row["quantity"]), USDC_HUMAN_ON_CHAIN)
        # The exact corruption, spelled out: the same raw at 18 decimals.
        self.assertNotEqual(Decimal(row["quantity"]), Decimal("3.687393E-12"))

    def test_base_usdt_was_stamped_eighteen_in_the_same_table(self):
        # 0xfde4c9... sat at decimals=18 beside the USDC row. Zero balance
        # today, so it cost nothing yet; it is the next one to have cost.
        self.assertEqual(known_token_decimals("base", USDT_BASE, "USDT"), 6)
        bridge = _DeadRPCBridge()
        self.assertEqual(_refresher(bridge)._erc20_decimals("base", USDT_BASE, None), 6)
        self.assertEqual(bridge.decimals_calls, [])

    def test_router_wallet_accessor_prefers_the_table_over_a_dead_rpc(self):
        # The shared accessor, used by send_service, bridge_service and the
        # swap/bridge sizing in router_wallet. Its contract is unchanged --
        # always an int, never a raise -- so those callers only improve.
        import router_wallet

        class Boom(router_wallet.UltraSwapBridge):
            def __init__(self):
                pass

            def _w3(self, chain):
                raise RuntimeError("base RPC refused")

        answer = Boom().erc20_decimals("base", USDC_BASE)
        self.assertEqual(answer, 6)
        self.assertIsInstance(answer, int)


class TheDoubleReallyReproducesTheBugTest(unittest.TestCase):
    """A regression test is worthless if it passes against the old code too."""

    def test_the_old_one_line_path_still_produces_the_corrupted_quantity(self):
        # The line that wrote the bad row was:
        #     decimals = int(self.bridge.erc20_decimals(chain, token))
        #     qty = _to_decimal_string(bal, decimals)
        # Run exactly that against this double and the 10^12 error reappears,
        # which is what makes the assertions above meaningful.
        bridge = _DeadRPCBridge()
        legacy_decimals = int(bridge.erc20_decimals("base", USDC_BASE))
        legacy_qty = _to_decimal_string(bridge.erc20_balance_of("base", USDC_BASE, "0x"), legacy_decimals)

        self.assertEqual(legacy_decimals, 18)
        self.assertEqual(Decimal(legacy_qty), Decimal("3.687393E-12"))
        self.assertLess(float(legacy_qty), MIN_CLIP_USD)

        # The fixed path, same bridge, same call.
        row = _refresher(_DeadRPCBridge())._fetch_erc20("base", USDC_BASE, "0xw", {})
        self.assertEqual(Decimal(row["quantity"]), USDC_HUMAN_ON_CHAIN)


class UnknownDecimalsRefuseTheRowTest(unittest.TestCase):
    """Rule 2: nobody guesses. An unanswerable token is skipped, not deflated."""

    def test_a_token_no_source_can_size_is_refused(self):
        # Neither the address nor the symbol is in the table, and the contract
        # will not answer. Note the symbol fallback is deliberate and load
        # bearing -- a bridged USDC deployment at an unlisted address still
        # resolves to 6 -- so the unknown case needs an unknown symbol too.
        unknown = "0x00000000000000000000000000000000deadbeef"
        bridge = _DeadRPCBridge(symbol="MYSTERY")
        self.assertIsNone(known_token_decimals("base", unknown, "MYSTERY"))
        row = _refresher(bridge)._fetch_erc20("base", unknown, "0xwallet", {})
        self.assertIsNone(
            row,
            "an unsizable balance must be refused, not written from a guessed 18",
        )

    def test_refresh_skips_a_refused_row_leaving_the_last_good_one(self):
        # `_fetch` is what `refresh` submits, and `refresh` does
        # `if not result: continue`. The None must survive that far rather
        # than raising AttributeError on .update().
        import inspect

        source = inspect.getsource(RealtimeBalanceRefresher._fetch)
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertIn("if data is None:", code)

        unknown = "0x00000000000000000000000000000000deadbeef"

        class _Plan:
            chain = "base"
            token = unknown
            meta = {}

        r = _refresher(_DeadRPCBridge(symbol="MYSTERY"))
        r.native_token = "0x0000000000000000000000000000000000000000"
        self.assertIsNone(r._fetch(_Plan(), "0xwallet"))


class MinClipCanBeFundedFromTheRealBalanceTest(unittest.TestCase):
    """The consequence the two rules exist to prevent."""

    def test_the_deflated_quantity_cannot_fund_the_clip_and_the_real_one_can(self):
        deflated = float(_to_decimal_string(USDC_RAW_ON_CHAIN, 18))
        correct = float(_to_decimal_string(USDC_RAW_ON_CHAIN, 6))

        # This is the arithmetic _build_transition_plan runs. Stables peg to
        # $1, deficit and native_buffer_gap were both measured 0.0.
        self.assertLess(deflated, MIN_CLIP_USD)   # -> block_reason="min_clip"
        self.assertGreaterEqual(correct, MIN_CLIP_USD)
        # Not marginal: the real balance funds the clip nearly five times.
        self.assertGreater(correct / MIN_CLIP_USD, 4.0)


if __name__ == "__main__":
    unittest.main()
