import sys
from pathlib import Path
from unittest.mock import patch

from django.test import TestCase

sys.path.append(str(Path(__file__).resolve().parents[2]))

from .models import DiscoveredToken, DiscoveryEvent, HoneypotCheck, SwapProbe
from services.discovery.coordinator import DiscoveryCoordinator
from services.discovery.trending_fetcher import TrendingToken


class DiscoveryModelsTest(TestCase):
    def test_discovery_event_creation(self):
        event = DiscoveryEvent.objects.create(
            symbol="FXS-FRAX",
            chain="base",
            source="dexscreener",
            bull_score=0.8,
            bear_score=0.2,
            metadata={"trigger": "bullish"},
        )
        self.assertIsNotNone(event.id)
        self.assertAlmostEqual(event.bull_score, 0.8)

    def test_honeypot_check(self):
        check = HoneypotCheck.objects.create(
            symbol="FXS-FRAX",
            chain="base",
            verdict="safe",
            confidence=0.9,
            details={"tax": 0.01},
        )
        self.assertEqual(check.verdict, "safe")
        self.assertAlmostEqual(check.confidence, 0.9)

    def test_swap_probe(self):
        probe = SwapProbe.objects.create(
            symbol="FXS-FRAX",
            chain="base",
            buy_tx_hash="0x123",
            sell_tx_hash="0x456",
            buy_amount=10.0,
            sell_amount=9.8,
            success=True,
            gas_cost_native=0.002,
        )
        self.assertTrue(probe.success)
        self.assertGreater(probe.buy_amount, probe.sell_amount)

    def test_discovered_token_status(self):
        token = DiscoveredToken.objects.create(symbol="FXS-FRAX", chain="base")
        self.assertEqual(token.status, "pending")
        token.status = "promoted"
        token.save()
        refreshed = DiscoveredToken.objects.get(symbol="FXS-FRAX")
        self.assertEqual(refreshed.status, "promoted")


class DiscoveryCoordinatorTest(TestCase):
    def test_coordinator_creates_records(self):
        trending_sample = TrendingToken(
            symbol="NEW-USDC",
            chain="base",
            pair_address="0xabc",
            dex_id="dexscreener",
            price_usd=1.0,
            volume_24h_usd=100000.0,
            liquidity_usd=50000.0,
            price_change_1h=5.0,
            price_change_6h=10.0,
            price_change_24h=20.0,
            metadata={"fdv": 1000000},
        )

        def fake_fetch(**kwargs):
            return [trending_sample]

        def fake_goplus(address: str, chain_id: str):
            class Report:
                verdict = "safe"
                confidence = 0.7
                details = {"buy_tax": 0.02, "sell_tax": 0.02}

            return Report()

        coordinator = DiscoveryCoordinator(chains=["base"], limit=1)
        with patch("services.discovery.coordinator.fetch_trending_tokens", fake_fetch), patch(
            "services.discovery.coordinator.goplus_token_security", fake_goplus
        ):
            coordinator.run()

        token = DiscoveredToken.objects.get(symbol="NEW-USDC")
        self.assertEqual(token.status, "validated")
        self.assertTrue(DiscoveryEvent.objects.filter(symbol="NEW-USDC").exists())
        self.assertGreaterEqual(HoneypotCheck.objects.filter(symbol="NEW-USDC").count(), 2)
        probe = SwapProbe.objects.get(symbol="NEW-USDC")
        self.assertEqual(probe.failure_reason, "probe_pending")
