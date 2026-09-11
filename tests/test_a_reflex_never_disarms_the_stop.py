"""A volatility reflex must not disarm the stop on money already at risk.

``_interpret_predictions`` short-circuits on ``self._reflex_blocked_until``:
when a reflex rule has fired (a volatility spike or a drawdown), a live bot
returned the decision immediately. That return sits ~800 lines ABOVE the
held-position branch, which is the only place take-profit, stop-loss and the
timed exit are evaluated. So during exactly the spike the reflex fired on, an
open LIVE position could not be closed.

Measured 2026-09-04 07:05 over one hour of organism_snapshots: 25
reflex-blocked decisions across 7 symbols, 6 of them on symbols carrying a
live position (CBETH-USDC 3, CBBTC-USDC 3). Every one of those six was a tick
on which the stop-loss was unreachable.

This is the THIRD time this shape has cost money here, and that is why it is
worth a test rather than a comment:

  * "a refused entry swallowed the stop" -- 72 of 73 samples evaluated no
    trigger at all, and the single stop that did fire realised -18.4%, which
    was 92% of all live P/L at the time.
  * ``pos_is_live`` required ``live_trading_enabled``, so a demoted bot could
    not sell what it had already bought: 18 straight refused exits.

Each was a guard written about ENTRIES that also swallowed the EXIT.

The rule this file pins: a reflex may refuse to OPEN risk, never to CLOSE it.
Letting a live holding through opens nothing new -- an entry landing on a
live-held slot is refused unconditionally by the ``entry-refused-live-held``
branch, whoever is asking -- so the only decision such a sample can still
reach is the exit.
"""

from __future__ import annotations

import inspect
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_REFLEX_OPEN = "if time.time() < self._reflex_blocked_until:"
#: the first statement after the reflex block, and so its end
_REFLEX_CLOSE = "gas_required = self._estimate_gas_cost("


def _source() -> str:
    from trading.bot import TradingBot

    return inspect.getsource(TradingBot._interpret_predictions)


def _reflex_block(src: str) -> str:
    """Just the reflex short-circuit, not the whole 3000-line method.

    Scoping matters. A first draft of this file searched the ENTIRE method for
    ``if self.live_trading_enabled and not <flag>:`` and matched
    ``not live_approved`` -- a pre-existing line ~1200 lines further down that
    has nothing to do with the reflex. The test passed against the unfixed
    source for the wrong reason, which is the failure mode this whole suite
    exists to catch.
    """
    start = src.index(_REFLEX_OPEN)
    end = src.index(_REFLEX_CLOSE, start)
    return src[start:end]


class ReflexNeverDisarmsTheStopTest(unittest.TestCase):
    def test_the_reflex_return_is_guarded_by_the_live_holding(self):
        """The short-circuit must not fire when a live position is open."""
        block = _reflex_block(_source())
        self.assertNotRegex(
            block, r"if self\.live_trading_enabled:\s*\n\s*decision\.update",
            "an UNCONDITIONAL reflex return disarms the stop on a live "
            "position: it skips the held-position branch ~800 lines below, "
            "which is the only place stop-loss is evaluated")
        gate = re.search(
            r"if self\.live_trading_enabled and not (\w+):", block)
        self.assertIsNotNone(
            gate,
            "the reflex short-circuit must exempt a live holding; without the "
            "exemption a volatility spike disarms the stop on real money")
        flag = gate.group(1)

        # ...and the flag has to actually mean "we hold this live", not just
        # be a name that reads that way.
        decl = re.search(
            re.escape(flag) + r"\s*=\s*\(?(.+?)\)?\s*\n\s*if self\.live",
            block, re.S)
        self.assertIsNotNone(decl, f"{flag} must be assigned inside the block")
        body = decl.group(1)
        self.assertIn("pos", body, f"{flag} must be derived from `pos`")
        self.assertIn(
            '"live"', body,
            f"{flag} must test the position MODE -- a ghost holding must still "
            "short-circuit, because there the reflex is doing its real job of "
            "not opening new positions into a spike")

    def test_the_position_is_known_before_the_reflex_is_consulted(self):
        """`pos` must be established above the reflex gate.

        The gate now reads `pos`. If the reflex check ran first, that would be
        a NameError on the first spike -- the same latent shape as the
        deleted inline phantom check, which read a variable named `chain` that
        the function never assigns and survived only because `and` short
        circuits.
        """
        src = _source()
        established = src.index("pos = self.positions.get(symbol)")
        reflex = src.index("if time.time() < self._reflex_blocked_until:")
        self.assertLess(
            established, reflex,
            "`pos` must be assigned before the reflex gate reads it")

    def test_reconciliation_runs_before_the_reflex_gate(self):
        """A phantom must not decide whether the reflex is bypassed.

        The gate keys on `pos.mode == "live"`. A stale live row that the chain
        no longer backs would otherwise buy itself a reflex bypass on top of
        the entry block it already holds.
        """
        src = _source()
        reconcile = src.index("_drop_phantom_live_position")
        reflex = src.index("if time.time() < self._reflex_blocked_until:")
        self.assertLess(
            reconcile, reflex,
            "the book must agree with the wallet before the reflex gate "
            "reads the position mode")

    def test_the_exit_branch_is_below_the_reflex_gate(self):
        """The premise of the whole file.

        If the held-position exit logic ever moves ABOVE the reflex gate this
        test should fail loudly rather than the guard quietly becoming
        pointless -- at which point the exemption can go.
        """
        src = _source()
        reflex = src.index("if time.time() < self._reflex_blocked_until:")
        exit_branch = src.index("# Held-position exit logic.")
        self.assertLess(
            reflex, exit_branch,
            "the reflex gate returns before the exit branch -- that ordering "
            "is why the exemption is needed")

    def test_a_ghost_holding_still_short_circuits(self):
        """Only a LIVE holding is exempt.

        A ghost position costs nothing to hold and carries no stop worth
        protecting, so a reflex must still stop the bot opening into a spike.
        """
        block = _reflex_block(_source())
        gate = re.search(r"if self\.live_trading_enabled and not (\w+):", block)
        flag = gate.group(1)
        decl = re.search(
            re.escape(flag) + r"\s*=\s*\(?(.+?)\)?\s*\n\s*if self\.live",
            block, re.S)
        self.assertNotIn(
            '"ghost"', decl.group(1),
            "a ghost holding must not buy a reflex bypass")


if __name__ == "__main__":
    unittest.main()
