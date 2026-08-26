"""
A restart must not re-probe every tradeable pair before trading can start.

Pair selection asks `_has_live_price` about every candidate. Suppressions were
already persisted to the database, but *positive* results lived only in an
in-memory dict, so each restart re-probed every tradeable pair from scratch:
one live HTTP call each, serially, on the main thread, before a single market
stream could start.

Measured at 0.53s per probe across ~320 candidates -- roughly 2.8 minutes of
dead time per restart, paid again on the next one. That matters because a
strategy needs hours of uninterrupted uptime to accumulate ghost trades, and
every code change costs a restart.

With the disk cache: 2.88s cold, 0.003s after a simulated restart, identical
answers -- about a 1000x saving on the repeat.

The TTL is what keeps this honest. A pair that stops trading is re-probed
within the hour rather than trusted indefinitely, so a stale positive cannot
keep a dead pair in rotation for long.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import trading.selector as selector


class LivePairCache(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "live_pairs.json"
        self._saved = (selector._LIVE_CACHE_PATH, selector._LIVE_CACHE_MEM,
                       selector._LIVE_CACHE_TTL)
        selector._LIVE_CACHE_PATH = self.path
        selector._LIVE_CACHE_MEM = None
        selector._LIVE_CACHE_TTL = 3600.0

    def tearDown(self):
        (selector._LIVE_CACHE_PATH, selector._LIVE_CACHE_MEM,
         selector._LIVE_CACHE_TTL) = self._saved

    def test_a_confirmed_pair_survives_a_restart(self):
        """The whole point: no second probe for a pair we already verified."""
        selector._live_cache_put("base::AERO-USDC")
        selector._LIVE_CACHE_MEM = None          # simulate a fresh process
        self.assertTrue(selector._live_cache_get("base::AERO-USDC"))

    def test_an_unknown_pair_is_not_assumed_live(self):
        self.assertFalse(selector._live_cache_get("base::NEVER-SEEN"))

    def test_an_expired_entry_is_re_probed(self):
        """
        The TTL is what stops a stale positive keeping a dead pair alive.

        Trusting a confirmation forever would be the same class of mistake as
        trusting a cached price forever.
        """
        selector._live_cache_put("base::AERO-USDC")
        cache = selector._live_cache_load()
        cache["base::AERO-USDC"] = time.time() - 7200      # 2h old, TTL is 1h
        self.assertFalse(selector._live_cache_get("base::AERO-USDC"))

    def test_writes_are_durable_and_readable(self):
        selector._live_cache_put("base::AERO-USDC")
        self.assertTrue(self.path.exists())
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIn("base::AERO-USDC", data)

    def test_stale_entries_are_pruned_so_the_file_cannot_grow_forever(self):
        selector._live_cache_put("base::FRESH-USDC")
        cache = selector._live_cache_load()
        cache["base::ANCIENT-USDC"] = time.time() - selector._LIVE_CACHE_TTL * 10
        selector._live_cache_put("base::ANOTHER-USDC")     # triggers the prune
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertNotIn("base::ANCIENT-USDC", data)
        self.assertIn("base::FRESH-USDC", data)

    def test_a_broken_cache_file_never_breaks_selection(self):
        """
        Best effort by design.

        Pair selection failing because a cache file is corrupt would be a far
        worse outcome than simply probing again.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{ not json", encoding="utf-8")
        selector._LIVE_CACHE_MEM = None
        self.assertFalse(selector._live_cache_get("base::AERO-USDC"))
        selector._live_cache_put("base::AERO-USDC")        # must not raise

    def test_an_unwritable_path_is_survived(self):
        selector._LIVE_CACHE_PATH = Path("/nonexistent-root/x/live.json")
        selector._LIVE_CACHE_MEM = None
        selector._live_cache_put("base::AERO-USDC")        # must not raise

    def test_the_cache_short_circuits_the_network_probe(self):
        """A cache hit must not touch the network at all."""
        selector._live_cache_put("base::AERO-USDC")
        selector._LIVE_PAIR_CACHE.pop("base::AERO-USDC", None)
        selector._LIVE_CACHE_MEM = None
        with mock.patch.object(
            selector, "_probe_dexscreener",
            side_effect=AssertionError("probed despite a cache hit"),
        ):
            with mock.patch.object(selector._db, "is_pair_suppressed", return_value=False), \
                 mock.patch.object(selector._db, "get_pair_suppression", return_value=None):
                self.assertTrue(selector._has_live_price("AERO-USDC", chain="base"))


if __name__ == "__main__":
    unittest.main()
