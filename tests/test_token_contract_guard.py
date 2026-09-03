"""A token address must earn its way to a swap by answering on chain.

Measured 2026-09-03: ``data/token_addresses.json`` held eight addresses shaped
one shared address shape (BASECAT, BLUECHIP, NVDAC, BASEJUICE, AAPL,
GOOGLC, METAC, RAWR). ``eth_getCode`` returns ONE byte for each, so none is an
ERC-20 -- but ``decimals()`` answered 18, so every downstream check passed.
1.50 USDC was spent entering BASECAT across two swaps that settled on chain and
can never be sold back, and the ``no_fill_detected`` retries that followed
ended the only burst of rapid profitable trading this system has produced.

These tests pin the guard's behaviour, not a blocklist. Blocking eight known
addresses would stop those eight; the ninth arrives tomorrow from a discovery
feed under a different prefix and costs the same money. So the guard asks the
chain about each address, and the tests check that it refuses anything that
answers wrong and allows anything that answers like a token.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import token_contract_guard as guard  # noqa: E402


def _fake_chain(*, code_bytes: int, decimals=18, supply=1_000_000,
                balance=0, unreachable=False):
    """An RPC that answers the way a chain would for one hypothetical token."""

    def _rpc(chain, method, params):
        if unreachable:
            return None, False
        if method == "eth_getCode":
            return ("0x" + "ab" * code_bytes) if code_bytes else "0x", True
        data = (params[0] or {}).get("data", "")
        if data.startswith(guard._SEL_DECIMALS):
            return (None, True) if decimals is None else (hex(decimals), True)
        if data.startswith(guard._SEL_TOTAL_SUPPLY):
            return (None, True) if supply is None else (hex(supply), True)
        if data.startswith(guard._SEL_BALANCE_OF):
            return (None, True) if balance is None else (hex(balance), True)
        return None, True

    return _rpc


ADDR = "0x" + "1" * 40


class TokenContractGuardTest(unittest.TestCase):
    def setUp(self):
        guard.clear_cache()
        self.addCleanup(guard.clear_cache)

    # ---------------------------------------------------------- refusals --

    def test_the_measured_stub_is_refused(self):
        """One byte of code is the exact shape that cost 1.50 USDC."""
        with mock.patch.object(guard, "_rpc", _fake_chain(code_bytes=1)):
            ok, reason = guard.verify("base", ADDR)
        self.assertFalse(ok)
        self.assertEqual(reason, "code_1_bytes")

    def test_an_address_with_no_code_is_refused(self):
        with mock.patch.object(guard, "_rpc", _fake_chain(code_bytes=0)):
            ok, _ = guard.verify("base", ADDR)
        self.assertFalse(ok)

    def test_a_contract_that_is_not_a_token_is_refused(self):
        """Plenty of code, but decimals() does not answer."""
        with mock.patch.object(guard, "_rpc",
                               _fake_chain(code_bytes=4000, decimals=None)):
            ok, reason = guard.verify("base", ADDR)
        self.assertFalse(ok)
        self.assertEqual(reason, "decimals_unanswered")

    def test_absurd_decimals_are_refused(self):
        with mock.patch.object(guard, "_rpc",
                               _fake_chain(code_bytes=4000, decimals=200)):
            ok, _ = guard.verify("base", ADDR)
        self.assertFalse(ok)

    def test_a_token_with_zero_supply_is_refused(self):
        """Nobody holds it, so it can be bought into and never sold out of."""
        with mock.patch.object(guard, "_rpc",
                               _fake_chain(code_bytes=4000, supply=0)):
            ok, reason = guard.verify("base", ADDR)
        self.assertFalse(ok)
        self.assertEqual(reason, "total_supply_zero")

    def test_balance_of_must_work_because_every_swap_needs_it(self):
        with mock.patch.object(guard, "_rpc",
                               _fake_chain(code_bytes=4000, balance=None)):
            ok, reason = guard.verify("base", ADDR)
        self.assertFalse(ok)
        self.assertEqual(reason, "balance_of_unanswered")

    def test_a_malformed_address_is_refused_without_a_call(self):
        ok, reason = guard.verify("base", "0xdeadbeef")
        self.assertFalse(ok)
        self.assertEqual(reason, "not_an_address")

    # --------------------------------------------------------- approvals --

    def test_a_real_token_is_allowed(self):
        with mock.patch.object(guard, "_rpc", _fake_chain(code_bytes=1852,
                                                          decimals=6)):
            ok, reason = guard.verify("base", ADDR)
        self.assertTrue(ok)
        self.assertIn("ok_code_1852", reason)

    def test_native_sentinels_are_not_contracts_and_are_allowed(self):
        for sentinel in ("native", "", "0x" + "e" * 40):
            ok, _ = guard.verify("base", sentinel)
            self.assertTrue(ok, f"{sentinel} must pass through")

    # ------------------------------------------------- failure direction --

    def test_an_rpc_outage_allows_rather_than_halting_trading(self):
        """Our outage is not the token's fault; downstream guards still apply."""
        with mock.patch.object(guard, "_rpc", _fake_chain(code_bytes=0,
                                                          unreachable=True)):
            ok, reason = guard.verify("base", ADDR)
        self.assertTrue(ok)
        self.assertEqual(reason, "rpc_unreachable")

    def test_an_outage_verdict_is_never_cached(self):
        """Otherwise one blip would whitelist a stub for the whole run."""
        with mock.patch.object(guard, "_rpc", _fake_chain(code_bytes=0,
                                                          unreachable=True)):
            self.assertTrue(guard.verify("base", ADDR)[0])
        # The chain comes back and tells the truth.
        with mock.patch.object(guard, "_rpc", _fake_chain(code_bytes=1)):
            ok, reason = guard.verify("base", ADDR)
        self.assertFalse(ok, "the outage must not have been remembered as a pass")
        self.assertEqual(reason, "code_1_bytes")

    def test_a_refusal_is_remembered(self):
        """A stub stays refused without re-asking on every trade."""
        with mock.patch.object(guard, "_rpc", _fake_chain(code_bytes=1)):
            self.assertFalse(guard.verify("base", ADDR)[0])

        def _explode(*a, **k):  # pragma: no cover - must never be called
            raise AssertionError("cached refusal should not hit the chain again")

        with mock.patch.object(guard, "_rpc", _explode):
            self.assertFalse(guard.verify("base", ADDR)[0])

    def test_an_approval_is_rechecked_after_the_window(self):
        """A token that dies mid-run must stop being trusted."""
        with mock.patch.object(guard, "_rpc", _fake_chain(code_bytes=4000)):
            self.assertTrue(guard.verify("base", ADDR)[0])
        with mock.patch.object(guard, "REVERIFY_AFTER_SEC", -1):
            with mock.patch.object(guard, "_rpc", _fake_chain(code_bytes=1)):
                ok, _ = guard.verify("base", ADDR)
        self.assertFalse(ok, "a stale approval must be re-verified, not trusted")


if __name__ == "__main__":
    unittest.main()
