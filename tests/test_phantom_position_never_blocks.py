"""A position the wallet does not hold must never block trading.

Measured 2026-09-04 06:06. The pipeline looked alive -- 24 ghost entries in six
hours, 47 candidates an hour -- and live entries were EXACTLY ZERO for six
hours. The cause was a record claiming a live position that no longer existed:
a live position blocks every further entry on its symbol, so a wrong record is
a permanent, silent halt.

WHAT THE CHAIN SAID, decoded 2026-09-04 06:40 from the transaction inputs (the
base node in use serves tx bodies but returns NULL receipts, so the calldata is
the evidence). CBETH-USDC was NOT "written but never bought" as first
diagnosed -- BOTH legs settled:

    buy  0x5de159efe0d946b683c08f00773c43fbd7e0803f45953f5e5ab670f8af7d5077
         exactInput USDC->WETH->CBETH, amountIn 750000 = 0.75 USDC
    sell 0x4ca1a606eb33ef24df951f15177532a2d9803554082ddb4c8677cc5d9bbc7e2d
         exactInput CBETH->WETH->USDC, amountIn 262495452605958,
         minOut 744363 = 0.744363 USDC

``live-swap-settled`` was written at 00:17:32 and no ``live-exit`` ever
followed. The round trip completed on chain and the book never heard, so a
sold-out position refused every atf_static entry on the symbol for seven hours
and 18 exits ran into ``live_position_cannot_exit_in_simulation``.

Two things this pins, both of which the first fix got wrong:

1. WIRING. The check hung off ONE branch (the ``entry-refused-live-held``
   one). Two predicates several hundred lines earlier read the same ``pos``
   and refuse first, so the check never ran for the symbol it was written for
   -- CBETH-USDC was still refused as ``entry-refused-duplicate`` at 06:23:11,
   eight minutes AFTER that fix was committed, and GRASS-USDC 41 times in two
   hours. Reconciliation now happens once, where ``pos`` is established.

2. PERSISTENCE. The book is not in-memory only, whatever the first comment
   claimed: it was read back out of
   ``kv_store['state']['ghost_trading']['positions']``, and ``_load_state``
   feeds it straight back. ``_save_state`` removes a symbol only if it is in
   ``_owned_symbols``, so an unclaimed pop survives the merge and is
   resurrected on restart. Claim, pop, save.
"""

from __future__ import annotations

import inspect
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Bot:
    """The methods under test, lifted onto a stand-in.

    The real TradingBot pulls in TensorFlow and a live DB; the behaviour being
    pinned is entirely in the two reconciliation methods and in what the caller
    does with the answer.
    """

    PHANTOM_RECHECK_INTERVAL_SEC = 60.0

    def __init__(self, token="0x" + "1" * 40):
        from trading.bot import TradingBot

        self.positions = {}
        self._token = token
        self._owned_position_symbols = set()
        self.saved = 0
        self.logged = []
        self._resolve_token_address = lambda chain, sym: self._token
        self._position_is_real_on_chain = (
            TradingBot._position_is_real_on_chain.__get__(self)
        )
        self._drop_phantom_live_position = (
            TradingBot._drop_phantom_live_position.__get__(self)
        )
        self._claim_position_symbol = TradingBot._claim_position_symbol.__get__(self)
        self._owned_symbols = TradingBot._owned_symbols.fget(self)
        # Both of these are lazy properties on the real class, for bots built
        # through __new__; the stand-in has to bind them explicitly.
        self._phantom_checked_at = TradingBot._phantom_checked_at.fget(self)

        class _DB:
            def __init__(self, outer):
                self._outer = outer

            def log_trade(self, **kw):
                self._outer.logged.append(kw)

        class _Metrics:
            def feedback(self, *a, **kw):
                pass

        self.db = _DB(self)
        self.metrics = _Metrics()

    def _save_state(self):
        self.saved += 1


def _fake_rpc(raw, reachable=True):
    calls = []

    def _rpc(chain, method, params):
        calls.append((chain, method))
        if not reachable:
            return None, False
        return raw, True

    _rpc.calls = calls
    return _rpc


_ZERO = "0x" + "0" * 64
_HELD = "0x" + "0" * 49 + "ee5c5f2b3a33"  # a nonzero balance


class PositionIsRealOnChainTest(unittest.TestCase):
    def test_a_position_with_no_tokens_is_not_real(self):
        """The CBETH case: the book says a position, the wallet holds zero."""
        bot = _Bot()
        with mock.patch("services.token_contract_guard._rpc", _fake_rpc(_ZERO)):
            real = bot._position_is_real_on_chain(
                "base", "CBETH-USDC",
                {"size": 0.000262122199594547, "tx_hash": ""})
        self.assertFalse(real, "zero balance means the position is a phantom")

    def test_a_position_with_tokens_is_real(self):
        bot = _Bot()
        with mock.patch("services.token_contract_guard._rpc", _fake_rpc(_HELD)):
            real = bot._position_is_real_on_chain(
                "base", "CBETH-USDC",
                {"size": 0.00026, "tx_hash": "0x" + "a" * 64})
        self.assertTrue(real, "a held position must keep blocking new entries")

    def test_an_rpc_outage_keeps_the_block(self):
        """Fail CLOSED: an outage must not release a real position."""
        bot = _Bot()
        with mock.patch("services.token_contract_guard._rpc",
                        _fake_rpc(None, reachable=False)):
            real = bot._position_is_real_on_chain(
                "base", "CBETH-USDC", {"size": 0.00026, "tx_hash": ""})
        self.assertTrue(real, "keep the block when we cannot check")

    def test_an_empty_answer_is_unreadable_not_zero(self):
        """A node that answers '0x' has nothing to say; that is not a zero balance.

        int('0x', 16) raises, and the bare except would have turned an
        unreadable balance into "keep the block" through an exception path.
        Same outcome, but named -- and it must never read as a phantom.
        """
        bot = _Bot()
        with mock.patch("services.token_contract_guard._rpc", _fake_rpc("0x")):
            real = bot._position_is_real_on_chain(
                "base", "CBETH-USDC", {"size": 0.00026, "tx_hash": ""})
        self.assertTrue(real, "an empty answer must keep the block")

    def test_an_unresolvable_token_keeps_the_block(self):
        bot = _Bot()
        bot._resolve_token_address = lambda chain, sym: None
        real = bot._position_is_real_on_chain(
            "base", "MYSTERY-USDC", {"size": 1.0, "tx_hash": ""})
        self.assertTrue(real, "cannot verify means keep the block")


class DropPhantomLivePositionTest(unittest.TestCase):
    def _phantom(self):
        return {
            "mode": "live",
            "size": 0.000262122199594547,
            "entry_price": 2861.2608972460434,
            "entry_ts": 1788491703.8893654,
            "trade_id": "2:CBETH-USDC:44c3665bffdf4861a56112da28ead2c3",
            "strategy_id": "atf_static",
            "entry_tx_hash":
                "0x5de159efe0d946b683c08f00773c43fbd7e0803f45953f5e5ab670f8af7d5077",
        }

    def test_a_phantom_is_dropped_from_the_book(self):
        bot = _Bot()
        bot.positions["CBETH-USDC"] = self._phantom()
        with mock.patch("services.token_contract_guard._rpc", _fake_rpc(_ZERO)):
            out = bot._drop_phantom_live_position(
                "CBETH-USDC", chain="base", pos=bot.positions["CBETH-USDC"])
        self.assertIsNone(out, "the caller must be told the position is gone")
        self.assertNotIn(
            "CBETH-USDC", bot.positions,
            "a phantom must be dropped from the live book, not merely logged")

    def test_the_drop_is_persisted_not_just_in_memory(self):
        """The bug the first fix shipped: an unclaimed pop is resurrected.

        _save_state removes a symbol from the persisted book only when it is in
        _owned_symbols, so popping without claiming leaves the row in
        kv_store['state']['ghost_trading']['positions'] for _load_state to feed
        straight back on the next restart.
        """
        bot = _Bot()
        bot.positions["CBETH-USDC"] = self._phantom()
        with mock.patch("services.token_contract_guard._rpc", _fake_rpc(_ZERO)):
            bot._drop_phantom_live_position(
                "CBETH-USDC", chain="base", pos=bot.positions["CBETH-USDC"])
        self.assertIn(
            "CBETH-USDC", bot._owned_symbols,
            "the symbol must be CLAIMED or _save_state's merge keeps the row")
        self.assertEqual(bot.saved, 1, "the drop must be persisted")

    def test_the_drop_is_recorded_with_a_full_length_hash(self):
        """An unlogged correction is indistinguishable from no block at all."""
        bot = _Bot()
        bot.positions["CBETH-USDC"] = self._phantom()
        with mock.patch("services.token_contract_guard._rpc", _fake_rpc(_ZERO)):
            bot._drop_phantom_live_position(
                "CBETH-USDC", chain="base", pos=bot.positions["CBETH-USDC"])
        rows = [r for r in bot.logged
                if r.get("status") == "live-position-dropped-phantom"]
        self.assertEqual(len(rows), 1, "the drop must leave a trading_ops row")
        tx = rows[0]["details"]["dropped_entry_tx_hash"]
        self.assertEqual(len(tx), 66, "a truncated hash is not verifiable evidence")

    def test_a_real_position_is_kept(self):
        bot = _Bot()
        pos = dict(self._phantom())
        bot.positions["CBETH-USDC"] = pos
        with mock.patch("services.token_contract_guard._rpc", _fake_rpc(_HELD)):
            out = bot._drop_phantom_live_position(
                "CBETH-USDC", chain="base", pos=pos)
        self.assertIs(out, pos, "a held position must be returned unchanged")
        self.assertIn("CBETH-USDC", bot.positions)
        self.assertEqual(bot.saved, 0, "nothing changed, nothing to save")

    def test_an_outage_never_releases_a_position(self):
        """Fail CLOSED: releasing a real position would cause a double buy."""
        bot = _Bot()
        pos = dict(self._phantom())
        bot.positions["CBETH-USDC"] = pos
        with mock.patch("services.token_contract_guard._rpc",
                        _fake_rpc(None, reachable=False)):
            out = bot._drop_phantom_live_position(
                "CBETH-USDC", chain="base", pos=pos)
        self.assertIs(out, pos, "an outage must keep the block")
        self.assertIn("CBETH-USDC", bot.positions)

    def test_a_ghost_position_is_never_chain_checked(self):
        """Ghost positions hold no tokens; checking one is pure RPC spend."""
        bot = _Bot()
        pos = {"mode": "ghost", "size": 1.0}
        rpc = _fake_rpc(_ZERO)
        with mock.patch("services.token_contract_guard._rpc", rpc):
            out = bot._drop_phantom_live_position(
                "GRASS-USDC", chain="base", pos=pos)
        self.assertIs(out, pos, "a ghost position is not a phantom")
        self.assertEqual(rpc.calls, [], "no chain read for a ghost position")

    def test_no_position_is_passed_through(self):
        bot = _Bot()
        self.assertIsNone(
            bot._drop_phantom_live_position("X-USDC", chain="base", pos=None))

    def test_the_chain_read_is_rate_limited(self):
        """This runs on every sample; it must not cost an RPC call each time."""
        bot = _Bot()
        pos = dict(self._phantom())
        rpc = _fake_rpc(_HELD)
        with mock.patch("services.token_contract_guard._rpc", rpc):
            for _ in range(5):
                bot._drop_phantom_live_position(
                    "CBETH-USDC", chain="base", pos=pos)
        self.assertEqual(
            len(rpc.calls), 1,
            "five samples inside the interval must cost one balanceOf call")


class ReconciliationIsWiredBeforeEveryRefusalTest(unittest.TestCase):
    """The gap that let the first fix ship looking done.

    The check existed, was correct, was tested -- and was wired into ONE of the
    three places that refuse an entry on the strength of ``pos``. The two that
    actually fired for CBETH-USDC and GRASS-USDC ran hundreds of lines earlier
    and never called it. These pin the wiring, not the logic.
    """

    def _source(self):
        from trading.bot import TradingBot

        return inspect.getsource(TradingBot._interpret_predictions)

    def test_reconciliation_runs_before_the_duplicate_refusal(self):
        src = self._source()
        reconcile = src.index("_drop_phantom_live_position")
        duplicate = src.index("entry_duplicates_held_position = bool(")
        self.assertLess(
            reconcile, duplicate,
            "entry-refused-duplicate refused CBETH-USDC with held_mode=live "
            "because it read `pos` before anything checked it against the chain")

    def test_reconciliation_runs_before_the_live_slot_refusal(self):
        src = self._source()
        reconcile = src.index("_drop_phantom_live_position")
        live_slot = src.index("entry_refused_by_live_slot = bool(")
        self.assertLess(
            reconcile, live_slot,
            "entry_refused_by_live_slot must not refuse on an unverified book")

    def test_reconciliation_runs_where_the_position_is_established(self):
        """Directly after the book is read, beside its mirror (adoption)."""
        src = self._source()
        established = src.index("pos = self.positions.get(symbol)")
        reconcile = src.index("_drop_phantom_live_position")
        adopt = src.index("_adopt_orphaned_live_holding")
        self.assertLess(established, reconcile)
        self.assertLess(
            reconcile, adopt,
            "drop a sold-out position BEFORE trying to adopt an unbooked one")

    def test_there_is_only_one_implementation_of_the_rule(self):
        """Two copies of a rule is two contracts to keep in sync.

        The inline pop this replaced could not persist -- it never claimed the
        symbol -- so the two copies did not even agree.
        """
        from trading import bot as bot_module

        src = inspect.getsource(bot_module)
        callers = src.count("self._position_is_real_on_chain(")
        self.assertEqual(
            callers, 1,
            "_position_is_real_on_chain must have exactly one caller "
            "(_drop_phantom_live_position); found %d" % callers)


if __name__ == "__main__":
    unittest.main()
