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

#: The account live swaps are signed from. balanceOf must be asked about this
#: address -- never about the token contract. See
#: BalanceIsAskedAboutTheWalletTest.
WALLET = "0x291c854811e92906a658Fb94Aa511bF919f968ad"


class _Bot:
    """The methods under test, lifted onto a stand-in.

    The real TradingBot pulls in TensorFlow and a live DB; the behaviour being
    pinned is entirely in the two reconciliation methods and in what the caller
    does with the answer.
    """

    PHANTOM_RECHECK_INTERVAL_SEC = 60.0

    def __init__(self, token="0x" + "1" * 40, wallet=WALLET):
        from trading.bot import TradingBot

        self.positions = {}
        self._token = token
        # The account live swaps are signed from. balanceOf must be asked
        # about THIS, never about the token contract.
        self._wallet = wallet
        self._bridge = object()
        self._init_bridge = lambda: self._bridge
        self._live_wallet_address = lambda: self._wallet
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
        # ``params`` is recorded because the verdict is only half the
        # contract -- the other half is WHICH ADDRESS was asked about, and
        # the bug below lived entirely in that half.
        calls.append((chain, method, params))
        if not reachable:
            return None, False
        return raw, True

    _rpc.calls = calls
    return _rpc


def _balanceof_arg(params):
    """The address argument out of an eth_call to balanceOf(address).

    calldata is '0x' + 8 selector chars + 64 chars of a left-padded address,
    so the address is the last 40 of those 64.
    """
    data = params[0]["data"]
    assert data[:10] == "0x70a08231", data[:10]
    arg = data[10:]
    assert len(arg) == 64, len(arg)
    return "0x" + arg[24:]


_ZERO = "0x" + "0" * 64
_HELD = "0x" + "0" * 49 + "ee5c5f2b3a33"  # a nonzero balance


class BalanceIsAskedAboutTheWalletTest(unittest.TestCase):
    """balanceOf takes the HOLDER, and the holder is our wallet.

    The shipped call built its calldata from ``str(token)[2:]`` -- the token's
    OWN address -- so it asked every contract how much of itself it held and
    never once asked about the wallet. Every existing test above mocked the
    RPC and asserted the VERDICT, so none of them noticed the question was
    wrong; that is exactly how this shipped.

    Measured 2026-09-04 07:12 on base, wallet
    0x291c854811e92906a658Fb94Aa511bF919f968ad, through the bot's own
    ``services.token_contract_guard._rpc``:

        symbol   balanceOf(TOKEN)            balanceOf(WALLET)
        CBETH    2055717985610008168         0
        AERO     187805394408695762999133    3019286196837921898
        CBBTC    0                           1078

    It fails in both directions, and both directions cost money:

      * CBETH's phantom was immortal. cbETH holds cbETH, so the check said
        "real" forever. The round trip had settled on chain at 00:17:32 and
        the stale row refused every atf_static entry for 7.9 hours;
        ``live-position-dropped-phantom`` was never written once, ever.
      * CBBTC's REAL position looked dead. The cbBTC contract holds no cbBTC,
        so the check said "phantom" while the wallet held 1078 raw. Dropping
        that un-books tokens we are still holding, and nothing ever sells
        them -- the "11 buys against 4 sells" failure this check exists to
        end.
    """

    def test_balanceof_is_asked_about_the_wallet_not_the_token(self):
        token = "0x" + "ab" * 20
        bot = _Bot(token=token)
        pos = {"mode": "live", "size": 1.0}
        rpc = _fake_rpc(_HELD)
        with mock.patch("services.token_contract_guard._rpc", rpc):
            bot._position_is_real_on_chain("base", "CBETH-USDC", pos)

        self.assertEqual(len(rpc.calls), 1, "expected exactly one balanceOf")
        chain, method, params = rpc.calls[0]
        self.assertEqual(method, "eth_call")
        self.assertEqual(
            params[0]["to"].lower(), token.lower(),
            "the call must be sent TO the token contract")
        self.assertEqual(
            _balanceof_arg(params), WALLET.lower(),
            "balanceOf must be asked about the WALLET; asking about the token "
            "made CBETH's phantom immortal for 7.9 hours")
        self.assertNotEqual(
            _balanceof_arg(params), token.lower(),
            "asking the token about itself is the bug, not the check")

    def test_a_token_that_holds_itself_is_still_a_phantom(self):
        """CBETH, exactly: the contract holds 2055717985610008168 of itself
        while the wallet holds zero. The shipped code read the former."""
        bot = _Bot(token="0x2ae3f1ec7f1f5012cfeab0185bfc7aa3cf0dec22")
        pos = {"mode": "live", "size": 0.000262122199594547}

        def _rpc(chain, method, params):
            # The chain, faithfully: self-balance nonzero, wallet balance zero.
            if _balanceof_arg(params) == bot._token.lower():
                return "0x" + format(2055717985610008168, "064x"), True
            return _ZERO, True

        with mock.patch("services.token_contract_guard._rpc", _rpc):
            self.assertFalse(
                bot._position_is_real_on_chain("base", "CBETH-USDC", pos),
                "the wallet holds no CBETH, so the position is a phantom "
                "however much cbETH the cbETH contract holds of itself")

    def test_a_token_holding_none_of_itself_keeps_a_real_position(self):
        """CBBTC, exactly: the contract holds 0 of itself while the wallet
        holds 1078 raw. The shipped code would have dropped a REAL position
        and orphaned the tokens behind it."""
        bot = _Bot(token="0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf")
        pos = {"mode": "live", "size": 1.078e-05}

        def _rpc(chain, method, params):
            if _balanceof_arg(params) == bot._token.lower():
                return _ZERO, True
            return "0x" + format(1078, "064x"), True

        with mock.patch("services.token_contract_guard._rpc", _rpc):
            self.assertTrue(
                bot._position_is_real_on_chain("base", "CBBTC-USDC", pos),
                "the wallet holds 1078 raw cbBTC -- dropping this position "
                "would un-book tokens nothing would ever sell")

    def test_an_unnameable_wallet_keeps_the_block(self):
        """Fail CLOSED. An empty address would pad into somebody else's slot,
        and 'we cannot name the holder' is unreadable, not zero."""
        for bad in ("", "0x", "0xdeadbeef"):
            bot = _Bot(wallet=bad)
            rpc = _fake_rpc(_ZERO)
            with mock.patch("services.token_contract_guard._rpc", rpc):
                self.assertTrue(
                    bot._position_is_real_on_chain(
                        "base", "CBETH-USDC", {"mode": "live", "size": 1.0}),
                    f"wallet {bad!r} must keep the block, not release it")
            self.assertEqual(
                rpc.calls, [],
                "a wallet we cannot name must not reach the chain at all")


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


class DropAndAdoptionMustAskAboutTheSameContractTest(unittest.TestCase):
    """The two reconciliation rules must not answer differently.

    Measured 2026-09-04 from ``trading_ops``, one symbol, ninety minutes:

        11:57:37  CBETH-USDC  dropped-phantom   wallet_holds_none_of_this_token
        11:57:41  CBETH-USDC  adopted           onchain_holding_had_no_position
        11:59:12  CBETH-USDC  dropped-phantom
        12:02:44  CBETH-USDC  adopted
        12:02:54  CBETH-USDC  dropped-phantom
        ...  13 adoptions against 11 phantom-drops in 24h

    One rule says the wallet holds it, the other says it holds none of it, on
    the same symbol four seconds apart. They are not both reading the chain
    wrong -- they are reading DIFFERENT CONTRACTS. ``_adopt_orphaned_live_holding``
    resolves the contract the settled BUY actually bought and stores it on the
    position as ``base_token_address``; ``_execute_decision`` sizes the sell
    from that same field (``base_address_hint``); and this check resolved the
    TICKER instead.

    While the position stands it refuses every entry on the symbol; when it is
    dropped the next directive enters again. That is the churn behind
    ``stop_loss:-0.0203`` and ``stop_loss:-0.0278`` -- positions entered and
    stopped out inside 25 minutes.

    A ticker is not a token here: 131 of 408 discovered base symbols map to
    more than one contract, and BASECAT resolves to nothing at all now that the
    stub is purged from the address book.
    """

    #: cbBTC, the contract a settled buy actually bought.
    CBBTC = "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf"
    #: What resolving the ticker happened to return instead.
    OTHER = "0x" + "cd" * 20

    def test_the_position_contract_is_asked_about_not_the_ticker(self):
        bot = _Bot(token=self.OTHER)
        pos = {"mode": "live", "size": 1.078e-05,
               "base_token_address": self.CBBTC}
        rpc = _fake_rpc(_ZERO)
        with mock.patch("services.token_contract_guard._rpc", rpc):
            bot._position_is_real_on_chain("base", "CBBTC-USDC", pos)

        self.assertEqual(len(rpc.calls), 1)
        _chain, _method, params = rpc.calls[0]
        self.assertEqual(
            params[0]["to"].lower(), self.CBBTC,
            "the balance must be read at the contract the position was opened "
            "in -- the one adoption booked and the one the exit will sell")
        self.assertNotEqual(
            params[0]["to"].lower(), self.OTHER.lower(),
            "resolving the ticker is what made the two rules disagree")

    def test_a_held_position_is_not_dropped_when_the_ticker_diverges(self):
        """The expensive direction: dropping a position we really hold.

        The ticker resolves to a contract holding nothing; the contract we
        actually bought holds 1078 raw. Reading the ticker un-books tokens the
        wallet is still holding, and nothing ever sells them.
        """
        bot = _Bot(token=self.OTHER)
        pos = {"mode": "live", "size": 1.078e-05,
               "base_token_address": self.CBBTC}

        def _rpc(chain, method, params):
            if params[0]["to"].lower() == self.CBBTC:
                return "0x" + format(1078, "064x"), True
            return _ZERO, True

        with mock.patch("services.token_contract_guard._rpc", _rpc):
            self.assertTrue(
                bot._position_is_real_on_chain("base", "CBBTC-USDC", pos),
                "the contract this position holds has 1078 raw in the wallet")

    def test_an_unresolvable_ticker_no_longer_blocks_forever(self):
        """BASECAT, exactly. The stub was purged, so the ticker resolves to
        None and the check returned True -- 'cannot check, keep the block' --
        for a position adoption could still book. An immortal block."""
        bot = _Bot()
        bot._resolve_token_address = lambda chain, sym: None
        basecat = "0x" + "b2" + "0" * 38
        pos = {"mode": "live", "size": 38.09, "base_token_address": basecat}
        rpc = _fake_rpc(_ZERO)
        with mock.patch("services.token_contract_guard._rpc", rpc):
            self.assertFalse(
                bot._position_is_real_on_chain("base", "BASECAT-USDC", pos),
                "the position names its own contract; an unresolvable ticker "
                "is no reason to keep a block the wallet does not back")
        self.assertEqual(len(rpc.calls), 1, "it must reach the chain at all")

    def test_a_pool_id_on_the_position_falls_back_to_the_ticker(self):
        """A 32-byte Uniswap v4 pool id is 66 chars and is not a token.

        Discovery stores pool ids for v4 pairs, and this repo has already
        handed one to a swap as though it were the token being bought.
        """
        ticker_token = "0x" + "ab" * 20
        bot = _Bot(token=ticker_token)
        pool_id = "0x" + "9" * 64
        pos = {"mode": "live", "size": 1.0, "base_token_address": pool_id}
        rpc = _fake_rpc(_ZERO)
        with mock.patch("services.token_contract_guard._rpc", rpc):
            bot._position_is_real_on_chain("base", "X-USDC", pos)
        self.assertEqual(
            rpc.calls[0][2][0]["to"].lower(), ticker_token.lower(),
            "a pool id is not a contract to read a balance from")

    def test_a_position_without_the_field_still_uses_the_ticker(self):
        """Positions booked before the field existed must keep working."""
        ticker_token = "0x" + "ab" * 20
        bot = _Bot(token=ticker_token)
        rpc = _fake_rpc(_HELD)
        with mock.patch("services.token_contract_guard._rpc", rpc):
            self.assertTrue(
                bot._position_is_real_on_chain(
                    "base", "AERO-USDC", {"mode": "live", "size": 1.0}))
        self.assertEqual(rpc.calls[0][2][0]["to"].lower(), ticker_token.lower())

    def test_all_three_rules_name_the_same_field(self):
        """The agreement is only real if the three sites share one key.

        Adoption WRITES ``base_token_address``, the exit SIZES from it, and the
        phantom check now READS it. A rename in any one of them puts the two
        reconciliation rules back on different contracts, which is this bug.
        """
        from trading.bot import TradingBot

        adopt = inspect.getsource(TradingBot._adopt_orphaned_live_holding)
        check = inspect.getsource(TradingBot._position_is_real_on_chain)
        execute = inspect.getsource(TradingBot._interpret_predictions)

        self.assertIn('"base_token_address": str(swap_token or "")', adopt,
                      "adoption must book the contract the buy bought")
        self.assertIn('pos.get("base_token_address")', check,
                      "the phantom check must read the position's contract")
        self.assertIn('base_address_hint = str(pos.get("base_token_address")',
                      execute,
                      "the exit must size from the position's contract")


if __name__ == "__main__":
    unittest.main()
