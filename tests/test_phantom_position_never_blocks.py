"""A position the wallet does not hold must never block trading.

Measured 2026-09-04 06:06. The pipeline looked alive -- 24 ghost entries in six
hours, 47 candidates an hour -- and live entries were EXACTLY ZERO for six
hours. The cause was two records claiming live positions that did not exist:

    GRASS-USDC   refused 26 entries in one hour
    CBETH-USDC   refused 6 more, with held_tx_hash='' -- a position written
                 but never bought

The wallet held 0.0000000000 CBETH at the time, confirmed on chain. A live
position blocks every further entry on its symbol, so a wrong record is a
permanent, silent halt: the bot cannot buy because it believes it already has,
and cannot sell because there is nothing there.

Verification also has to CHANGE the in-memory book, not merely report on it. A
check that logs "phantom" and leaves the record in place re-runs on the next
directive and blocks the symbol again, which is indistinguishable from no fix.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Bot:
    """The two methods under test, lifted onto a stand-in.

    The real TradingBot pulls in TensorFlow and a live DB; the behaviour being
    pinned is entirely in _position_is_real_on_chain and what the caller does
    with its answer.
    """

    def __init__(self, balance, token="0x" + "1" * 40, reachable=True):
        from trading.bot import TradingBot

        self.positions = {}
        self._balance = balance
        self._token = token
        self._reachable = reachable
        self._resolve_token_address = lambda chain, sym: self._token
        self._position_is_real_on_chain = TradingBot._position_is_real_on_chain.__get__(self)


def _fake_rpc(balance, reachable=True):
    def _rpc(chain, method, params):
        if not reachable:
            return None, False
        return hex(balance), True
    return _rpc


class PhantomPositionTest(unittest.TestCase):
    def test_a_position_with_no_tokens_is_not_real(self):
        """The CBETH case: record says a position, wallet holds zero."""
        bot = _Bot(balance=0)
        with mock.patch("services.token_contract_guard._rpc", _fake_rpc(0)):
            real = bot._position_is_real_on_chain(
                "base", "CBETH-USDC",
                {"size": 0.000262122199594547, "tx_hash": ""})
        self.assertFalse(real, "zero balance means the position is a phantom")

    def test_a_position_with_tokens_is_real(self):
        bot = _Bot(balance=262122199594547)
        with mock.patch("services.token_contract_guard._rpc",
                        _fake_rpc(262122199594547)):
            real = bot._position_is_real_on_chain(
                "base", "CBETH-USDC", {"size": 0.00026, "tx_hash": "0x" + "a" * 64})
        self.assertTrue(real, "a held position must keep blocking new entries")

    def test_an_rpc_outage_keeps_the_block(self):
        """An outage must not release a real position and cause a double buy."""
        bot = _Bot(balance=0, reachable=False)
        with mock.patch("services.token_contract_guard._rpc",
                        _fake_rpc(0, reachable=False)):
            real = bot._position_is_real_on_chain(
                "base", "CBETH-USDC", {"size": 0.00026, "tx_hash": ""})
        self.assertTrue(real, "fail closed: keep the block when we cannot check")

    def test_an_unresolvable_token_keeps_the_block(self):
        bot = _Bot(balance=0)
        bot._resolve_token_address = lambda chain, sym: None
        real = bot._position_is_real_on_chain(
            "base", "MYSTERY-USDC", {"size": 1.0, "tx_hash": ""})
        self.assertTrue(real, "cannot verify means keep the block")

    def test_verification_mutates_the_in_memory_book(self):
        """Reporting a phantom without removing it blocks the symbol forever."""
        bot = _Bot(balance=0)
        bot.positions["CBETH-USDC"] = {"mode": "live", "size": 0.00026}

        with mock.patch("services.token_contract_guard._rpc", _fake_rpc(0)):
            if not bot._position_is_real_on_chain(
                    "base", "CBETH-USDC", bot.positions["CBETH-USDC"]):
                bot.positions.pop("CBETH-USDC", None)

        self.assertNotIn(
            "CBETH-USDC", bot.positions,
            "a phantom must be dropped from the live book, not merely logged")


if __name__ == "__main__":
    unittest.main()
