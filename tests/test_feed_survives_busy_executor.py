"""A healthy endpoint must not be reported as a timeout because WE were busy.

The REST feed asks a token-bucket limiter for permission before every fetch.
That wait used to happen in `asyncio.to_thread(...)`, which runs on the event
loop's DEFAULT ThreadPoolExecutor -- 10 threads on this 6-CPU box, the same
pool aiohttp resolves hostnames in (aiodns is not installed) and the same pool
candidate training runs in. Every fetch parked a thread for up to 5s before it
issued a request, so with ~30 streams the pool stayed full and requests expired
against their 10s ceiling before they were ever sent.

The failure is indistinguishable from a dead upstream, and it was diagnosed as
one for weeks. Measured 2026-09-01 across 6h of production: 1736 dexscreener +
1732 geckoterminal "timeouts" while a direct probe of both hosts answered in
20-100ms at 12 requests in flight. market_stream fell from ~300 ticks/hour to
3, which starved every strategy downstream of it.

Reproduced against a LOCAL server answering in microseconds -- 32 streams, the
production 10s timeout:

    threaded limiter wait   22 ok, 10 timed out, worst request 10.03s
    awaited limiter wait    32 ok,  0 timed out, worst request  0.31s

So: the limiter wait must be awaited, never handed to the default executor.
These tests hold every thread in that executor and require a fetch to complete
anyway.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import unittest

import aiohttp
from aiohttp import web

from services.adaptive_control import APIRateLimiter
from trading.data_stream import Endpoint, MarketDataStream


def _default_executor_size() -> int:
    return min(32, (os.cpu_count() or 1) + 4)


class RestFetchSurvivesABusyExecutor(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        async def handler(_request):
            return web.json_response({"last": "1.5"})

        app = web.Application()
        app.router.add_get("/", handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        # Port 0 -> the OS picks a free one, so the test cannot collide with
        # whatever else is listening on this machine.
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self._port = site._server.sockets[0].getsockname()[1]
        self._release = threading.Event()
        self._occupied = threading.Semaphore(0)

    async def asyncTearDown(self) -> None:
        self._release.set()
        await self._runner.cleanup()

    async def _saturate_default_executor(self) -> None:
        """Hold every thread in the loop's default executor until teardown."""
        def hold() -> None:
            self._occupied.release()
            self._release.wait(30.0)

        for _ in range(_default_executor_size()):
            asyncio.ensure_future(asyncio.to_thread(hold))
        # Wait until the threads have actually been handed out, otherwise the
        # test races the executor's lazy thread creation and proves nothing.
        deadline = time.time() + 10.0
        for _ in range(_default_executor_size()):
            while not self._occupied.acquire(blocking=False):
                if time.time() > deadline:
                    self.fail("default executor never picked up the blocking work")
                await asyncio.sleep(0.05)

    async def test_fetch_completes_while_the_default_executor_is_full(self):
        stream = MarketDataStream(symbol="TESTFEED-USDC", chain="base")
        # A literal IP so aiohttp skips the resolver: this test is about the
        # limiter wait, and mixing DNS in would blame the wrong thread pool.
        endpoint = Endpoint(
            name="bitstamp",
            ws_template=None,
            subscribe_template=None,
            rest_template=f"http://127.0.0.1:{self._port}/",
        )
        stream._http_session = aiohttp.ClientSession()
        try:
            await self._saturate_default_executor()
            started = time.time()
            result = await asyncio.wait_for(
                stream._fetch_rest_price(endpoint, "TESTFEED", "USDC"),
                timeout=15.0,
            )
            elapsed = time.time() - started
        finally:
            await stream._http_session.close()
            await stream.stop()

        self.assertIsNone(
            result.error,
            f"fetch failed with {result.error!r} against a local server that "
            f"answers instantly -- the pressure was ours, not the endpoint's",
        )
        self.assertLess(
            elapsed, 5.0,
            f"fetch took {elapsed:.1f}s behind a busy executor; the limiter "
            f"wait is back on the default thread pool",
        )


class AwaitedLimiterKeepsTheSameBudget(unittest.IsolatedAsyncioTestCase):
    """The async wait is a scheduling change, not a loosening of the limit."""

    async def test_tokens_are_granted_when_the_bucket_has_them(self):
        limiter = APIRateLimiter(default_capacity=2.0, default_refill_rate=1.0)
        await limiter.acquire_async("host", tokens=1.0, timeout=0.5)
        await limiter.acquire_async("host", tokens=1.0, timeout=0.5)

    async def test_an_empty_bucket_still_times_out(self):
        limiter = APIRateLimiter(default_capacity=1.0, default_refill_rate=0.1)
        await limiter.acquire_async("host", tokens=1.0, timeout=0.5)
        with self.assertRaises(TimeoutError):
            await limiter.acquire_async("host", tokens=1.0, timeout=0.5)

    async def test_waiting_does_not_park_a_thread(self):
        """The wait must yield to the loop, not consume executor capacity."""
        limiter = APIRateLimiter(default_capacity=1.0, default_refill_rate=4.0)
        limiter.acquire("host", tokens=1.0, timeout=0.5)

        progressed = 0

        async def ticker():
            nonlocal progressed
            for _ in range(5):
                await asyncio.sleep(0.01)
                progressed += 1

        task = asyncio.ensure_future(ticker())
        await limiter.acquire_async("host", tokens=1.0, timeout=2.0)
        await task
        self.assertEqual(progressed, 5, "the loop stalled during the wait")

    def test_blocking_acquire_still_works_for_sync_callers(self):
        limiter = APIRateLimiter(default_capacity=1.0, default_refill_rate=0.1)
        limiter.acquire("host", tokens=1.0, timeout=0.5)
        with self.assertRaises(TimeoutError):
            limiter.acquire("host", tokens=1.0, timeout=0.5)


if __name__ == "__main__":
    unittest.main()
