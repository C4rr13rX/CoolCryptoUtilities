"""
A websocket DNS failure must not silence a working REST feed.

`NETWORK_OUTAGE_BLOCK_REST_ON_DNS` defaulted to on, so a DNS failure on ANY
websocket host blocked REST polling for that stream, with backoff growing to
600 seconds. The original reasoning -- if DNS is broken, REST will fail too --
does not hold when the two use different hosts.

This deployment's Base-chain symbols are REST-only on dexscreener while the
websocket pool still tries venues like binance and coinbase. One transient
resolution failure on a host that is not even in the endpoint allowlist
therefore froze a healthy feed. Observed 2026-08-26: 24 streams started, each
wrote exactly one tick, then went quiet for 15+ minutes while DNS itself was
fine (all four hosts resolved in under 70ms when checked directly).

That starves every short-horizon strategy, which needs ~20 samples in its
window before the registry will even evaluate it.

The case the flag was written for -- REST endpoints sharing the failing
websocket's host -- is still handled, by `_dns_outage_blocks_rest`.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from trading.data_stream import MarketDataStream


def _stream() -> MarketDataStream:
    return MarketDataStream(symbol="BASECAT-USDC", chain="base")


class WebsocketDnsDoesNotBlockRest(unittest.TestCase):
    def test_the_flag_defaults_to_off(self):
        """
        The default matters more than the setting.

        `.env` is gitignored, so a deployment that never sets this must still
        get the safe behaviour.
        """
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NETWORK_OUTAGE_BLOCK_REST_ON_DNS", None)
            self.assertFalse(_stream()._block_rest_on_dns)

    def test_a_websocket_dns_outage_leaves_rest_polling_alone(self):
        """The exact production stall: WS DNS fails, REST must keep polling."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NETWORK_OUTAGE_BLOCK_REST_ON_DNS", None)
            stream = _stream()
            stream._network_outage_until = 9e18       # outage active
            stream._network_outage_source = "websocket"
            stream._network_outage_reason = "dns"
            self.assertFalse(stream._network_outage_blocks_rest())

    def test_a_rest_sourced_outage_still_blocks_rest(self):
        """
        When REST itself is what failed, backing off is correct.

        Relaxing that would hammer an endpoint that is actually down.
        """
        stream = _stream()
        stream._network_outage_until = 9e18
        stream._network_outage_source = "rest"
        stream._network_outage_reason = "dns"
        stream._network_outage_block_rest = True
        self.assertTrue(stream._network_outage_blocks_rest())

    def test_the_old_behaviour_can_be_restored(self):
        """An operator must be able to put the strict coupling back."""
        with mock.patch.dict(os.environ, {"NETWORK_OUTAGE_BLOCK_REST_ON_DNS": "1"}):
            stream = _stream()
            self.assertTrue(stream._block_rest_on_dns)
            stream._network_outage_until = 9e18
            stream._network_outage_source = "websocket"
            stream._network_outage_reason = "dns"
            stream._network_outage_block_rest = True
            self.assertTrue(stream._network_outage_blocks_rest())

    def test_no_outage_means_no_block(self):
        stream = _stream()
        stream._network_outage_until = 0.0
        self.assertFalse(stream._network_outage_blocks_rest())


if __name__ == "__main__":
    unittest.main()
