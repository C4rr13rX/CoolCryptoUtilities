"""
One weak strategy must not stop every other strategy from collecting evidence.

`halt_ghost` is raised from a single aggregate accuracy number and zeroes the
risk budget, which halts the scheduler for every strategy at once. That defeats
per-strategy graduation: each strategy keeps its own ledger and is promoted on
its own record, but one strategy dragging the average down froze collection for
all of them -- including strategies with good records, and including any new
strategy that had not yet placed a single trade.

It is also circular. Ghost trading is how a strategy *earns* the evidence the
accuracy gate measures, so halting collection because accuracy is low
guarantees accuracy stays low. Observed 2026-08-26: precision 0.416 against a
0.6 target with ghost collection halted, and `atf_static`'s 209 trades at 33.5%
dominating an aggregate that then blocked `money_button` from ever recording
its first trade.

**What is deliberately NOT relaxed**, because this touches real money:

  * `halt_live` still applies globally.
  * `_strategy_live_approved` still requires a strategy to have graduated on
    its own ledger (20 trades, 55% win rate) before it can trade live.
  * when live trading is enabled, the global ghost halt applies as before.

Only ghost simulation keeps running, which risks nothing and is the only way
out of the deadlock.
"""

from __future__ import annotations

import os
import types
import unittest
from unittest import mock

from trading.bot import TradingBot


def _bot(*, live_trading: bool) -> types.SimpleNamespace:
    """A stand-in exposing just what the halt decision reads."""
    bot = types.SimpleNamespace()
    bot.live_trading_enabled = live_trading
    bot._ghost_halt_is_per_strategy = types.MethodType(
        TradingBot._ghost_halt_is_per_strategy, bot
    )
    return bot


class GhostHaltIsPerStrategy(unittest.TestCase):
    def test_ghost_collection_continues_when_not_trading_live(self):
        """
        The deadlock breaker.

        In ghost mode nothing is at risk, so a global accuracy halt must not
        stop strategies from gathering the very evidence it is measuring.
        """
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GHOST_HALT_GLOBAL", None)
            self.assertTrue(_bot(live_trading=False)._ghost_halt_is_per_strategy())

    def test_the_global_halt_still_applies_when_live_trading_is_on(self):
        """
        Real money keeps the conservative behaviour.

        Relaxing a halt is only safe while trades are simulated.
        """
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GHOST_HALT_GLOBAL", None)
            self.assertFalse(_bot(live_trading=True)._ghost_halt_is_per_strategy())

    def test_the_old_behaviour_can_be_restored_by_env(self):
        """An operator must be able to put the all-or-nothing halt back."""
        with mock.patch.dict(os.environ, {"GHOST_HALT_GLOBAL": "1"}):
            self.assertFalse(_bot(live_trading=False)._ghost_halt_is_per_strategy())

    def test_the_live_gate_is_untouched(self):
        """
        Ghost collection running does NOT imply anything may trade live.

        `_strategy_live_approved` remains the only door to real money, and it
        consults the per-strategy ledger.
        """
        source = TradingBot._strategy_live_approved.__doc__ or ""
        self.assertIn("ghost-graduated", source)

        bot = types.SimpleNamespace()
        bot.strategy_ledger = mock.Mock()
        bot.strategy_ledger.is_live_approved.return_value = False
        bot._strategy_live_approved = types.MethodType(
            TradingBot._strategy_live_approved, bot
        )
        directive = types.SimpleNamespace(strategy_id="money_button")
        with mock.patch.dict(os.environ, {"STRATEGY_GRADUATION_ENFORCED": "1"}):
            self.assertFalse(bot._strategy_live_approved(directive))
        bot.strategy_ledger.is_live_approved.assert_called_with("money_button")


if __name__ == "__main__":
    unittest.main()
