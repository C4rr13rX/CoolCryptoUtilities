"""
A fabricated price must never reach the tick table or a strategy.

When the feed could not obtain a real quote, several paths published an
invented one so downstream consumers would "keep moving". That is worse than
stalling: nothing downstream can distinguish a made-up tick from an observed
one, so strategies trade on it with full confidence and the ledger fills with
evidence about data that never existed.

Measured on 2026-08-26 over a two-hour window: **211 ticks, of which 173 (82%)
were `fallback` and 36 (17%) `offline` -- two came from a live venue.** Across
all history 2,024 of 2,228 rows (91%) were synthetic.

Concrete damage found:

  * `BASEJUICE-USDC` alternated between 0.00111 and 1.00004 -- a 900x swing --
    because the offline store served USDC's dollar price under an alias match.
  * five symbols reported exactly +0.00% movement, which is one cached value
    repeating rather than a stable market.
  * 47% of ghost exits closed at exactly zero P/L, so the strategy ledger was
    largely a record of trading against frozen fiction. It was reset.

A feed that cannot price a symbol must say so and let an operator fix it.
Silence is recoverable; fiction is not.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
import unittest
from pathlib import Path
from unittest import mock

from trading.data_stream import MarketDataStream

ROOT = Path(__file__).resolve().parents[1]

#: Sources that are not observed market prices.
SYNTHETIC = ("fallback", "offline", "bootstrap", "reference", "snapshot")
REAL = ("dexscreener", "geckoterminal", "kucoin", "mexc", "coingecko")


def _dispatch(source: str, *, allow: bool = False):
    """Run one sample through _dispatch; return the prices actually written."""
    async def scenario():
        stream = MarketDataStream(symbol="TESTGUARD-USDC", chain="base")
        written = []
        stream._db.insert_market_sample = lambda **kw: written.append(kw.get("price"))
        sample = {
            "ts": time.time(),
            "symbol": "TESTGUARD-USDC",
            "chain": "base",
            "price": 1.23,
            "volume": 1.0,
            "rest": source,
        }
        env = {"ALLOW_SYNTHETIC_TICKS": "1"} if allow else {}
        with mock.patch.dict(os.environ, env, clear=False):
            if not allow:
                os.environ.pop("ALLOW_SYNTHETIC_TICKS", None)
            await stream._dispatch(sample)
        await stream.stop()
        return written

    return asyncio.run(scenario())


class SyntheticTicksAreRefused(unittest.TestCase):
    def test_every_synthetic_source_is_dropped(self):
        for source in SYNTHETIC:
            with self.subTest(source):
                self.assertEqual(
                    _dispatch(source), [],
                    f"a {source!r} price is invented, not observed, and must "
                    f"never be recorded",
                )

    def test_real_venues_still_write(self):
        """The guard must not cost us actual market data."""
        for source in REAL:
            with self.subTest(source):
                self.assertEqual(_dispatch(source), [1.23])

    def test_dropping_is_counted_so_the_outage_is_visible(self):
        """
        Refusing to invent a price only helps if someone can see it happening.

        Silently dropping would swap one invisible failure for another.
        """
        async def scenario():
            stream = MarketDataStream(symbol="TESTGUARD-USDC", chain="base")
            stream._db.insert_market_sample = lambda **kw: None
            os.environ.pop("ALLOW_SYNTHETIC_TICKS", None)
            for _ in range(3):
                await stream._dispatch({
                    "ts": time.time(), "symbol": "TESTGUARD-USDC", "chain": "base",
                    "price": 1.0, "volume": 1.0, "rest": "fallback",
                })
            count = stream._synthetic_drops
            await stream.stop()
            return count

        self.assertEqual(asyncio.run(scenario()), 3)

    def test_the_behaviour_can_be_restored_deliberately(self):
        """An operator may re-enable it, but never by accident."""
        self.assertEqual(_dispatch("fallback", allow=True), [1.23])


class TheTickTableIsClean(unittest.TestCase):
    """The stored history must contain no fabricated prices."""

    def test_no_synthetic_rows_remain_in_market_stream(self):
        db = ROOT / "storage" / "trading_cache.db"
        if not db.exists():
            self.skipTest("no local trading database")
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        offenders = {}
        for (raw,) in conn.execute("SELECT raw FROM market_stream"):
            try:
                detail = json.loads(raw or "{}")
            except (TypeError, ValueError):
                continue
            source = str(detail.get("rest") or detail.get("source") or "").lower()
            if source in SYNTHETIC:
                offenders[source] = offenders.get(source, 0) + 1
        self.assertFalse(
            offenders,
            f"fabricated prices are still stored as market data: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
