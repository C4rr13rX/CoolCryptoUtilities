"""A gas refill must never consume the capital it exists to enable.

Observed three times on a $14 wallet 2026-08-27: the refill converted the
ENTIRE stable balance into ETH chasing a native buffer, leaving $0 deployable.
Live trading then blocked on capital_deficit / wallet_sparse while the wallet
still held $14 of value, and two manual rebalances back into USDC were each
undone by the next refill.

The first version of this cap called ``.get()`` on ``swap_plan`` -- which is a
LIST, not a dict -- so it raised, was swallowed by a bare except, and the cap
silently did nothing. Hence these tests operate on the real list shape.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock


def _capped(swap_plan, max_share="0.25"):
    """Mirror of the guard in TradingBot._rebalance_for_gas."""
    share = float(max_share)
    if share <= 0:
        return False
    entries = swap_plan if isinstance(swap_plan, list) else [swap_plan]
    spend = held = 0.0
    for item in entries:
        if not isinstance(item, dict) or not item.get("is_stable"):
            continue
        try:
            spend += float(item.get("spend_usd") or 0.0)
            held += float(item.get("usd_value") or 0.0)
        except (TypeError, ValueError):
            continue
    return held > 0 and spend > held * share


def _leg(spend_usd, usd_value, is_stable=True):
    return {
        "symbol": "USDC", "token": "0xUSDC",
        "spend_amount": spend_usd, "spend_usd": spend_usd,
        "obtain_native": 0.0, "available": usd_value,
        "usd_value": usd_value, "kind": "stable" if is_stable else "volatile",
        "is_stable": is_stable,
    }


class GasRefillStableCapTest(unittest.TestCase):
    def test_draining_the_whole_stable_balance_is_refused(self):
        """The exact observed failure: spend all $6.99 of $6.99."""
        self.assertTrue(_capped([_leg(6.99, 6.99)]))

    def test_small_top_up_is_allowed(self):
        """A refill within the cap must still work -- gas is necessary."""
        self.assertFalse(_capped([_leg(1.00, 6.99)]))

    def test_boundary_at_the_cap_is_allowed(self):
        self.assertFalse(_capped([_leg(1.7475, 6.99)]))

    def test_just_over_the_cap_is_refused(self):
        self.assertTrue(_capped([_leg(1.80, 6.99)]))

    def test_multiple_stable_legs_are_summed(self):
        """Splitting the drain across legs must not evade the cap."""
        self.assertTrue(_capped([_leg(1.0, 3.0), _leg(1.0, 3.0), _leg(1.0, 3.0)]))

    def test_volatile_legs_are_ignored(self):
        """Selling a volatile holding for gas is not the failure mode."""
        self.assertFalse(_capped([_leg(9.0, 10.0, is_stable=False)]))

    def test_dict_shape_does_not_raise(self):
        """The original bug: a non-list plan must not blow up the guard."""
        try:
            _capped(_leg(6.99, 6.99))
        except Exception as exc:  # noqa: BLE001
            self.fail("guard raised on dict-shaped plan: %r" % (exc,))

    def test_zero_cap_disables_the_check(self):
        self.assertFalse(_capped([_leg(6.99, 6.99)], max_share="0"))

    def test_empty_plan_is_safe(self):
        self.assertFalse(_capped([]))

    def test_malformed_entries_do_not_raise(self):
        self.assertFalse(_capped([{"is_stable": True, "spend_usd": None,
                                   "usd_value": "not-a-number"}]))


if __name__ == "__main__":
    unittest.main()
