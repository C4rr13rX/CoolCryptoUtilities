"""
A slow consumer must not slow down price acquisition.

`_dispatch` used to `await` async callbacks inline, so the feed ran at the
speed of its slowest consumer. `TradingBot._handle_sample` runs the whole
TF/strategy pipeline per sample, so in production every stream recorded
roughly ONE tick instead of one per second -- and no symbol ever accumulated
the ~20 samples over 12+ minutes that the short-horizon strategies need to
evaluate at all. Ghost trading was starved of data by its own consumer.

Measured on AERO-USDC over a 30-second poll window, before the fix:

    no callback          29 writes
    2s async callback    10 writes
    10s async callback    3 writes

and after: 29 writes in all three cases.

This was invisible in isolation because a bare stream registers no callbacks,
so testing the stream alone showed a perfectly healthy 1s cadence. Only the
bot-plus-stream combination was slow, which is why this test drives dispatch
with a deliberately slow consumer rather than testing the stream by itself.
"""

from __future__ import annotations

import asyncio
import time
import unittest

from trading.data_stream import MarketDataStream


class ConsumersDoNotThrottleTheFeed(unittest.TestCase):
    def _stream(self):
        stream = MarketDataStream(symbol="TEST-USDC", chain="base")
        writes = []
        stream._db.insert_market_sample = lambda **kw: writes.append(time.time())
        return stream, writes

    def _sample(self, price: float, ts: float) -> dict:
        return {
            "ts": ts,
            "symbol": "TEST-USDC",
            "chain": "base",
            "price": price,
            "volume": 1000.0,
            "rest": "dexscreener",
        }

    def test_a_slow_async_consumer_does_not_delay_dispatch(self):
        """
        Dispatch must return promptly no matter how slow the consumer is.

        The consumer still runs -- it is scheduled, not dropped -- but the
        acquisition loop does not wait for it.
        """
        async def scenario():
            stream, writes = self._stream()
            started = []

            async def slow_consumer(sample):
                started.append(time.time())
                await asyncio.sleep(5.0)

            stream.register(slow_consumer)

            begin = time.time()
            for i in range(5):
                await stream._dispatch(self._sample(1.0 + i * 0.01, begin + i))
            elapsed = time.time() - begin

            # Let the scheduled consumers actually begin.
            await asyncio.sleep(0.05)
            await stream.stop()
            return elapsed, len(writes), len(started)

        elapsed, writes, started = asyncio.run(scenario())
        self.assertEqual(writes, 5, "every sample must still be recorded")
        self.assertLess(
            elapsed, 1.0,
            f"5 dispatches took {elapsed:.1f}s; a slow consumer is blocking "
            "acquisition again (inline await regression)",
        )
        self.assertEqual(started, 5, "consumers must still be invoked")

    def test_a_sync_consumer_still_runs_inline(self):
        """
        Sync callbacks keep their ordering guarantees.

        They are cheap by construction, and scheduling them would reorder side
        effects that consumers may depend on.
        """
        async def scenario():
            stream, _writes = self._stream()
            seen = []
            stream.register(lambda sample: seen.append(sample["price"]))
            await stream._dispatch(self._sample(1.23, time.time()))
            # Already recorded by the time dispatch returns -- no await needed.
            result = list(seen)
            await stream.stop()
            return result

        self.assertEqual(asyncio.run(scenario()), [1.23])

    def test_a_failing_consumer_does_not_break_the_feed(self):
        """A raising consumer must be reported, not allowed to stop ingestion."""
        async def scenario():
            stream, writes = self._stream()

            async def broken(sample):
                raise RuntimeError("consumer exploded")

            stream.register(broken)
            for i in range(3):
                await stream._dispatch(self._sample(1.0 + i * 0.01, time.time() + i))
            await asyncio.sleep(0.05)
            await stream.stop()
            return len(writes)

        self.assertEqual(asyncio.run(scenario()), 3)

    def test_stopping_cancels_consumer_work_still_in_flight(self):
        """A stopped stream must not leave consumer tasks running."""
        async def scenario():
            stream, _writes = self._stream()

            async def slow_consumer(sample):
                await asyncio.sleep(30.0)

            stream.register(slow_consumer)
            await stream._dispatch(self._sample(1.0, time.time()))
            await asyncio.sleep(0.05)
            in_flight = len(stream._callback_tasks)
            await stream.stop()
            return in_flight, len(stream._callback_tasks)

        before, after = asyncio.run(scenario())
        self.assertEqual(before, 1)
        self.assertEqual(after, 0, "stop() must cancel in-flight consumer tasks")


if __name__ == "__main__":
    unittest.main()
