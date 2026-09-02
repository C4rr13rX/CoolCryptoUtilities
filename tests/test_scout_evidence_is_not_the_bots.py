"""A strategy may only be graduated on trades its own executor took.

Link 9 (LIVE) reported "no live trades yet" while every bot-level gate stood
open. Measured 2026-09-02 against the database:

  * the bot reached its live entry gate 264 times in 24h and the swap guard
    PASSED 94 of them, yet not one became a live trade;
  * every directive arriving there came from a strategy the ledger had not
    graduated (obv_accumulation@5d, rsi_reversal@5d, ...), so
    ``_strategy_live_approved`` downgraded all 94 to ghost;
  * the ONLY strategy holding ``live_approved`` was ``atf_static`` -- and 368
    of its 376 closed trades (97.9%) were executed by
    ``services/atf_static_strategy.py``'s ghost scout, which hardcodes
    ``wallet="ghost"`` / ``mode="ghost"`` and publishes
    ``live_execution_enabled: False``.

The strategy permitted to spend money had no code path that could spend it;
the ones with such a path were never permitted. Two executors reporting into
one ledger id is what produced that, and it is the same defect class as the
fabricated records this project has purged twice: a record that is not a
measurement of the thing being promoted. Here it was not invented, it was
misattributed -- which spends money just as readily.

These tests pin the split so the two executors can never be re-merged by
accident.
"""

from __future__ import annotations

import unittest
from unittest import mock

from services.atf_static_strategy import (
    SCOUT_STRATEGY_ID,
    SIGNAL_STRATEGY_ID,
    _run_ghost_quote_scout,
)
from trading.strategies.ledger import StrategyLedger


class _FakeDB:
    """Minimal stand-in for TradingDatabase across the scout's ghost cycle."""

    def __init__(self, positions, price):
        self._json = {"atf_static_strategy:ghost_positions": positions}
        self._price = price
        self.trades = []

    # -- kv ----------------------------------------------------------------
    def get_json(self, key, default=None):
        return self._json.get(key, default)

    def set_json(self, key, value):
        self._json[key] = value

    # -- market feed -------------------------------------------------------
    def recent_market_prices(self, symbol, chain, *, since_ts=None, limit=25):
        import time

        now = time.time()
        # Dense enough to satisfy the stop-enforceability gate.
        return [(self._price, now - i * 60.0) for i in range(30)]

    def get_market_price(self, symbol, chain, *, ts=None, after=True):
        return (self._price, 0.0, None)

    # -- audit -------------------------------------------------------------
    def log_trade(self, **kwargs):
        self.trades.append(kwargs)


def _closing_position(entry_price):
    """One open position whose target is already met, so the cycle exits it."""
    import time

    return {
        "TOAD-USDC": {
            "source": "c0d3rv2_atf_static",
            "strategy_id": SCOUT_STRATEGY_ID,
            "symbol": "TOAD-USDC",
            "chain": "base",
            "entry_ts": time.time() - 600.0,
            "entry_price": entry_price,
            "last_price": entry_price,
            "target_return": 0.01,
        }
    }


class ScoutOutcomeAttributionTest(unittest.TestCase):
    def test_scout_exit_is_recorded_under_the_scout_id(self):
        """The id the LIVE gate reads must never be credited by this loop."""
        db = _FakeDB(_closing_position(1.0), price=1.10)   # +10%, clears target
        with mock.patch(
            "services.atf_static_strategy._record_ghost_outcome"
        ) as record:
            result = _run_ghost_quote_scout(
                db=db, signals=[], chain="base", quote_token="USDC",
                max_positions=3,
            )

        self.assertEqual([e["action"] for e in result["events"]], ["exit"])
        record.assert_called_once()
        self.assertEqual(record.call_args.args[0], SCOUT_STRATEGY_ID)
        self.assertNotEqual(record.call_args.args[0], SIGNAL_STRATEGY_ID)

    def test_logged_exit_row_names_the_scout_as_the_strategy(self):
        """``details.strategy_id`` is what every later attribution reads."""
        db = _FakeDB(_closing_position(1.0), price=1.10)
        with mock.patch("services.atf_static_strategy._record_ghost_outcome"):
            _run_ghost_quote_scout(
                db=db, signals=[], chain="base", quote_token="USDC",
                max_positions=3,
            )
        exits = [t for t in db.trades if t.get("status") == "ghost-exit"]
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0]["details"]["strategy_id"], SCOUT_STRATEGY_ID)

    def test_the_two_ids_are_distinct(self):
        self.assertNotEqual(SCOUT_STRATEGY_ID, SIGNAL_STRATEGY_ID)


class ScoutGraduationDoesNotAuthoriseTheBotTest(unittest.TestCase):
    """A graduated ghost-only executor must not open the live gate."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = Path(self._dir.name) / "ledger.json"

    def _ledger(self):
        return StrategyLedger(path=self.path)

    def test_scout_graduation_does_not_graduate_the_bot(self):
        ledger = self._ledger()
        for _ in range(25):
            ledger.record(SCOUT_STRATEGY_ID, profit=0.05, mode="ghost",
                          confidence=0.7, symbol="TOAD-USDC")
        self.assertTrue(ledger.is_live_approved(SCOUT_STRATEGY_ID))
        # ...and the id the bot's live path consults is untouched.
        self.assertFalse(ledger.is_live_approved(SIGNAL_STRATEGY_ID))

    def test_bot_graduates_on_its_own_record(self):
        """The split must not make graduation unreachable, only earned."""
        ledger = self._ledger()
        for _ in range(25):
            ledger.record(SIGNAL_STRATEGY_ID, profit=0.05, mode="ghost",
                          confidence=0.7, symbol="CBBTC-USDC")
        self.assertTrue(ledger.is_live_approved(SIGNAL_STRATEGY_ID))


if __name__ == "__main__":
    unittest.main()
