"""Tests for the payout engine.

These guard money leaving the wallet, so they are written around the ways a
payout system loses money rather than around its happy path: paying on
unrealised or simulated profit, paying twice for the same profit, paying out
of reserve, and sizing a transfer from a balance nobody could read.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

from django.test import TestCase

from . import engine
from .models import Payout, PayoutDestination


class PayoutEngineTests(TestCase):
    def setUp(self) -> None:
        self.destination = PayoutDestination.objects.create(
            name="Test Destination",
            kind="coinbase",
            address="0x" + "1" * 40,
            chain="base",
            share=Decimal("0.10"),
            min_net_profit_usd=Decimal("50"),
            min_transfer_usd=Decimal("5"),
            reserve_usd=Decimal("20"),
        )

    def _plan(self, net, trades=10, wallet="500"):
        with patch.object(engine, "realised_profit_since",
                          return_value=(Decimal(str(net)), trades, 1788540000.0)), \
             patch.object(engine, "wallet_usd",
                          return_value=None if wallet is None else Decimal(str(wallet))):
            return engine.plan_payout(self.destination)

    def test_defaults_match_the_stated_policy(self):
        """10% after $50 net -- the configured default."""
        self.assertEqual(self.destination.share, Decimal("0.10"))
        self.assertEqual(self.destination.min_net_profit_usd, Decimal("50"))
        # Never sends on its own until explicitly enabled.
        self.assertFalse(self.destination.auto_send)

    def test_net_loss_pays_nothing(self):
        self.assertFalse(self._plan("-0.25")["eligible"])

    def test_floor_is_exclusive(self):
        """Exactly at the floor is not above it."""
        self.assertFalse(self._plan("50.00")["eligible"])
        self.assertTrue(self._plan("150")["eligible"])

    def test_share_applies_only_above_the_floor(self):
        """$150 net, $50 floor -> 10% of $100, not of $150."""
        decision = self._plan("150")
        self.assertEqual(decision["amount_usd"], Decimal("10.000000"))

    def test_below_minimum_transfer_accumulates(self):
        """$4 would cost more in gas than it moves."""
        self.assertFalse(self._plan("90")["eligible"])

    def test_unreadable_wallet_refuses(self):
        """None is not zero: never size a transfer blind."""
        decision = self._plan("150", wallet=None)
        self.assertFalse(decision["eligible"])
        self.assertIn("unreadable", decision["reason"])

    def test_reserve_is_never_spent(self):
        self.assertFalse(self._plan("150", wallet="20")["eligible"])

    def test_payout_is_trimmed_to_protect_the_reserve(self):
        """$25 wallet, $20 reserve -> $5 free, and $5 is sendable."""
        decision = self._plan("150", wallet="25")
        self.assertTrue(decision["eligible"])
        self.assertEqual(decision["amount_usd"], Decimal("5.000000"))

    def test_trim_below_minimum_transfer_refuses(self):
        self.assertFalse(self._plan("150", wallet="22")["eligible"])

    def test_the_same_profit_is_never_paid_twice(self):
        """The window advances past what was already paid."""
        first_end = 1788540000.0

        def fake_profit(start, end=None):
            if start >= first_end:
                return Decimal("0"), 0, first_end + 10000
            return Decimal("150"), 30, first_end

        with patch.object(engine, "realised_profit_since", side_effect=fake_profit), \
             patch.object(engine, "wallet_usd", return_value=Decimal("500")):
            first = engine.plan_payout(self.destination)
            engine.record_payout(self.destination, first)
            second = engine.plan_payout(self.destination)

        self.assertTrue(first["eligible"])
        self.assertFalse(second["eligible"])
        committed = sum(
            p.amount_usd for p in Payout.objects.filter(status__in=("pending", "sent")))
        self.assertEqual(committed, Decimal("10.000000"))

    def test_disabled_destination_pays_nothing(self):
        self.destination.enabled = False
        self.destination.save()
        self.assertFalse(self._plan("150")["eligible"])

    def test_skips_are_recorded_not_silent(self):
        decision = self._plan("-0.25")
        row = engine.record_payout(self.destination, decision)
        self.assertEqual(row.status, "skipped")
        self.assertTrue(row.reason)

    def test_ghost_profit_is_not_payable(self):
        """Only closed LIVE exits count as realised profit."""
        self.assertNotIn("ghost-exit", engine.REALISED_EXIT_STATUSES)
        for status in engine.REALISED_EXIT_STATUSES:
            self.assertTrue(status.startswith("live"))
