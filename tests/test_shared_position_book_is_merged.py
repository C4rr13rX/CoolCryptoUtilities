"""One bot's saved state must not erase another bot's open position.

GhostSupervisor runs one TradingBot per symbol, all sharing a single state
blob. ``_save_state`` wrote ``ghost_trading.positions`` from only the calling
bot's ``self.positions``, so the last writer erased every other bot's open
position. ``TradingBot.__init__`` seeds a new bot from that blob, so a bot
built for a symbol whose row had just been clobbered started with no position,
re-entered it, and the position it had been holding closed for nobody: no exit
row, no outcome, no ledger entry.

Measured 2026-09-02 against trading_ops:

    ghost-entry rows with no matching exit    152
    distinct symbols among them                16   (the book is keyed by symbol,
                                                     so >=136 were destroyed)
    VIRTUAL-USDC entries / exits            66 / 1   one entry per ~90s for 17h
    positions in the persisted book              5   one of them already exited

That is the graduation blocker underneath link 5. Promotion is scored on CLOSED
ghost trades; atf_static -- the only executor that can spend real money and the
only one with a positive record -- opened 26 and closed 11, and sits at 11 of
the 20 trades it needs.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.bot import ACCOUNTING_VERSION, TradingBot


class _SharedStateDB:
    """The one state blob every bot in the pool reads and writes."""

    def __init__(self) -> None:
        self.state = {}

    def load_state(self):
        return self.state

    def save_state(self, state):
        self.state = state

    def trade_outcome_summary(self, _mode):
        return {"checkpoint": 0.0, "net_profit": 0.0, "closed": 0, "profitable": 0}

    def fetch_trade_outcomes(self, limit=0):
        return []


def _bot(db, symbol):
    """A TradingBot with only the attributes the state path touches.

    Built without __init__ on purpose: constructing a real one starts streams,
    models and a scheduler, none of which this behaviour depends on.
    """
    bot = object.__new__(TradingBot)
    bot.db = db
    bot.primary_symbol = symbol
    bot.primary_chain = "base"
    bot.positions = {}
    bot.bus_routes = {}
    bot.stable_bank = 0.0
    bot.total_profit = 0.0
    bot.realized_profit = 0.0
    bot.total_trades = 0
    bot.wins = 0
    bot.sim_quote_balances = {}
    bot.sim_native_balances = {}
    bot.ghost_session_id = 2
    bot.active_exposure = {}
    bot._auto_execute_approved = False
    bot._owned_position_symbols = set()
    bot._load_state()
    return bot


def _position(symbol, price):
    return {
        "mode": "ghost",
        "strategy_id": "obv_accumulation@5d",
        "entry_price": price,
        "size": 0.13,
        "ts": 1788396291.0,
        "entry_ts": 1788396291.0,
        "trade_id": f"2:{symbol}:deadbeef",
        "route": [symbol.split("-")[0], "USDC"],
    }


def _book(db):
    return db.state.get("ghost_trading", {}).get("positions", {})


class SharedPositionBookTest(unittest.TestCase):
    """The bots are built BEFORE either opens a position, because that is what
    GhostSupervisor does -- it constructs the whole pool at startup and keeps
    the objects. Ordering is the entire bug: build the second bot afterwards
    and it inherits the first bot's row through _load_state and writes it back,
    so even the unfixed code looks correct.
    """

    def setUp(self):
        self.db = _SharedStateDB()
        self.virtual = _bot(self.db, "VIRTUAL-USDC")
        self.arb = _bot(self.db, "ARB-USDC")

    def _open(self, bot, symbol, price):
        bot._claim_position_symbol(symbol)
        bot.positions[symbol] = _position(symbol, price)
        bot._save_state()

    def test_second_bot_save_keeps_the_first_bots_position(self):
        """The exact clobber: VIRTUAL survives a save by the ARB bot."""
        self._open(self.virtual, "VIRTUAL-USDC", 0.6738)
        self.assertEqual(set(_book(self.db)), {"VIRTUAL-USDC"})

        self._open(self.arb, "ARB-USDC", 0.31)

        self.assertEqual(set(_book(self.db)), {"VIRTUAL-USDC", "ARB-USDC"})

    def test_a_bot_built_after_the_save_still_sees_its_position(self):
        """The consequence: no position, so the bot re-enters and the one it
        was holding closes for nobody. This is what produced 66 VIRTUAL-USDC
        entries against a single exit."""
        self._open(self.virtual, "VIRTUAL-USDC", 0.6938)
        self._open(self.arb, "ARB-USDC", 0.31)

        replacement = _bot(self.db, "VIRTUAL-USDC")
        self.assertIn("VIRTUAL-USDC", replacement.positions)
        self.assertEqual(
            replacement.positions["VIRTUAL-USDC"]["entry_price"], 0.6938
        )

    def test_closing_a_position_removes_it_from_the_shared_book(self):
        """Merging must not turn into never deleting: a closed position that
        stayed in the book would be reloaded as open by the next bot."""
        self._open(self.virtual, "VIRTUAL-USDC", 0.6938)
        self._open(self.arb, "ARB-USDC", 0.31)

        del self.virtual.positions["VIRTUAL-USDC"]
        self.virtual._save_state()

        self.assertEqual(set(_book(self.db)), {"ARB-USDC"})

    def test_an_inherited_position_is_not_resurrected(self):
        """A bot loads the whole book but only streams its own symbol, so every
        other row is a copy it will hold stale forever. Writing that copy back
        would undo the owner's close."""
        self._open(self.virtual, "VIRTUAL-USDC", 0.6938)

        # Built while VIRTUAL was open: it inherits the row without owning it.
        latecomer = _bot(self.db, "ARB-USDC")
        self.assertIn("VIRTUAL-USDC", latecomer.positions)
        self.assertNotIn("VIRTUAL-USDC", latecomer._owned_position_symbols)

        del self.virtual.positions["VIRTUAL-USDC"]
        self.virtual._save_state()
        self.assertEqual(set(_book(self.db)), set())

        self._open(latecomer, "ARB-USDC", 0.31)

        self.assertEqual(set(_book(self.db)), {"ARB-USDC"})

    def test_routes_survive_another_bots_save(self):
        """bus_routes is fed back by _load_state, so a lost route is a position
        whose swap path is forgotten."""
        self.virtual.bus_routes["VIRTUAL-USDC"] = ["VIRTUAL", "USDC"]
        self.virtual._save_state()

        self.arb.bus_routes["ARB-USDC"] = ["ARB", "USDC"]
        self.arb._save_state()

        routes = self.db.state["ghost_trading"]["routes"]
        self.assertEqual(
            routes, {"VIRTUAL-USDC": ["VIRTUAL", "USDC"], "ARB-USDC": ["ARB", "USDC"]}
        )

    def test_book_shape_is_unchanged(self):
        """Consumers read len(positions) and reload the rows; the merged book
        must stay dict[str, dict] with the same fields."""
        self._open(self.virtual, "VIRTUAL-USDC", 0.6938)

        book = _book(self.db)
        self.assertIsInstance(book, dict)
        row = book["VIRTUAL-USDC"]
        self.assertIsInstance(row, dict)
        self.assertIsInstance(row["entry_ts"], float)
        self.assertIsInstance(row["size"], float)
        self.assertEqual(
            self.db.state["ghost_trading"]["accounting_version"], ACCOUNTING_VERSION
        )


if __name__ == "__main__":
    unittest.main()
