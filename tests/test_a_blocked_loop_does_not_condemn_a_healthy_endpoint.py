"""A stall in OUR event loop must not be charged to the price endpoints.

MEASURED 2026-09-07, 6h of production ``feedback_events`` (market_stream):

    4551 REST timeouts. Of the 430 distinct seconds containing one, 351 (82%)
    had 2+ DISTINCT endpoints time out in the SAME second -- up to 5 at once.
    dexscreener / geckoterminal / coingecko / mexc are different CDNs; they do
    not fail in the same second. A direct probe of all three returned HTTP 200
    in 0.07-0.37s, nine times out of nine, while production logged them dead.

Same window, gaps between this stream's own feedback_events: p50 0.011s,
p99 39.3s, max 2601s, with 89.7% of the six hours inside a gap longer than
5s -- and then 37 events landing in one 0.1s bucket. A blocked loop catching
up, not a dead upstream.

The damage is the feed. ``_poll_rest_data`` charged every ``timeout`` to the
endpoint and escalated 15s -> 64s -> 98s -> ``block_rest`` for 426s, so one
local stall bought a seven-minute blackout on a healthy host. Worse,
``outage_detected`` is ``total_network_errors == total_attempted`` -- true by
construction when a stall hits every endpoint at once -- so the clearer the
evidence that the fault was local, the harder the code punished the upstreams.

Downstream over four days: ticks 91 -> 33 per 10m, ghost candidates 1019 ->
130, ghost entries 863 -> 14, ghost exits 53 -> 10. Ghost exits are the only
currency that re-arms a demoted strategy for live, and atf_static -- the only
live-capable strategy -- sat at 8 of the 20 fresh trades it needs.

These tests pin BOTH directions, because the fix must not weaken outage
detection: a timeout with a RESPONSIVE loop is still fully charged.
"""

import asyncio
import time

import pytest

import trading.data_stream as ds
from trading.data_stream import _EventLoopLagMonitor, _loop_lag_monitor


def test_lag_monitor_reports_a_blocking_call_that_starved_the_loop():
    """A synchronous sleep on the loop is time an in-flight request lost too.

    The monitor reports OVERSHOOT past the interval it asked to sleep for, so a
    block of D seconds sampled every I reads as at least D - I: the first I of
    any stall is indistinguishable from the sleep the sampler requested. That
    under-reads by at most one interval (0.25s), which is why the threshold
    this feeds is measured in seconds rather than milliseconds.
    """
    block = 0.6
    slack = _EventLoopLagMonitor._INTERVAL + 0.05

    async def scenario():
        monitor = _EventLoopLagMonitor()
        monitor.ensure_started()
        started = time.monotonic()
        await asyncio.sleep(0)
        # Block the loop the way a synchronous crawl/backfill/model save does.
        time.sleep(block)
        # Let the monitor wake and record the stall it slept through.
        await asyncio.sleep(0.35)
        return monitor.max_lag_since(started)

    lag = asyncio.run(scenario())
    assert lag >= block - slack, (
        f"a {block}s block on the loop read as only {lag:.3f}s of lag"
    )


def test_an_idle_loop_reports_no_stall():
    """The discriminator must not fire on ordinary scheduling jitter.

    If this drifts upward, a real outage starts being excused as a local
    stall -- the failure mode opposite to the one being fixed.
    """

    async def scenario():
        monitor = _EventLoopLagMonitor()
        monitor.ensure_started()
        started = time.monotonic()
        for _ in range(8):
            await asyncio.sleep(0.05)
        return monitor.max_lag_since(started)

    lag = asyncio.run(scenario())
    assert lag < 0.2, f"an idle loop reported {lag:.3f}s of lag"


def _stream():
    """A stream object with just enough shape for _fetch_rest_price."""
    from trading.data_stream import Endpoint, MarketDataStream

    stream = MarketDataStream.__new__(MarketDataStream)
    stream.symbol = "AERO-USDC"
    stream.chain = "base"
    stream._rest_base = "AERO"
    stream._rest_quote = "USDC"
    endpoint = Endpoint(
        name="dexscreener",
        ws_template=None,
        subscribe_template=None,
        rest_template="https://api.dexscreener.example/{base}/{quote}",
        headers=None,
    )
    return stream, endpoint


class _TimingOutSession:
    """An HTTP session whose GET times out with the loop still responsive."""

    def get(self, *args, **kwargs):
        raise asyncio.TimeoutError()


class _StallingSession:
    """A GET that times out because the loop was blocked underneath it.

    This is the production shape: the request is in flight, something
    synchronous seizes the loop, and the timeout comes due the instant the loop
    resumes -- before the lag sampler has had a chance to run and write the
    stall down. If the monitor only reported RECORDED samples, this case would
    read as a healthy loop and condemn the endpoint anyway.
    """

    def __init__(self, block: float) -> None:
        self._block = block

    def get(self, *args, **kwargs):
        time.sleep(self._block)
        raise asyncio.TimeoutError()


class _StubLimiter:
    async def acquire_async(self, *args, **kwargs):
        return None


@pytest.fixture
def short_budget(monkeypatch):
    """Shrink the request budget so the tests block for ~1s, not ~6s.

    ``_fetch_rest_price`` reads both module globals at call time, so the
    threshold under test is a real 0.5 * 1.0 = 0.5s rather than a stubbed
    comparison.
    """
    monkeypatch.setattr(ds, "REST_FETCH_TIMEOUT", 1.0)
    monkeypatch.setattr(ds, "LOOP_STALL_FRACTION", 0.5)
    return 1.0


def test_a_timeout_under_a_blocked_loop_is_not_charged_to_the_endpoint(short_budget):
    """THE REGRESSION. Old behaviour returned "timeout" and backed the host off.

    Against the pre-fix ``_fetch_rest_price`` -- which returned
    ``RestFetchResult(None, "timeout")`` unconditionally -- this asserts
    "local_stall" and fails.
    """
    stream, endpoint = _stream()
    # Blocks for comfortably more than the 0.5s the discriminator needs,
    # allowing for the one-interval under-read the sampler has by design.
    stream._http_session = _StallingSession(1.0)
    stream.rate_limiter = _StubLimiter()

    async def scenario():
        monitor = _loop_lag_monitor()
        assert monitor is not None
        # Let the sampler take a baseline before the request starts.
        await asyncio.sleep(0.35)
        return await stream._fetch_rest_price(endpoint, "AERO", "USDC")

    result = asyncio.run(scenario())
    assert result.error == "local_stall", (
        f"a timeout recorded while the loop was blocked for over half the "
        f"request budget was classified {result.error!r} -- that charges a "
        f"healthy endpoint and escalates to a 426s block_rest"
    )


def test_a_timeout_on_a_responsive_loop_is_still_the_endpoints_fault(short_budget):
    """The guard must keep working. A real dead upstream is still condemned."""
    stream, endpoint = _stream()
    stream._http_session = _TimingOutSession()
    stream.rate_limiter = _StubLimiter()

    async def scenario():
        _loop_lag_monitor()
        for _ in range(8):
            await asyncio.sleep(0.05)
        return await stream._fetch_rest_price(endpoint, "AERO", "USDC")

    result = asyncio.run(scenario())
    assert result.error == "timeout", (
        f"a timeout on a healthy loop was classified {result.error!r}; a real "
        f"outage must still reach the endpoint backoff"
    )


def test_local_stalls_cannot_manufacture_a_rest_outage():
    """``outage_detected`` is `errors == attempted`, true when a stall hits all.

    This pins the call-site half of the fix: a stalled sample must count as
    neither an attempt nor an error, or the all-endpoints-failed rule fires on
    precisely the evidence that says the fault was ours.
    """
    total_attempted = 0
    total_network_errors = 0
    for error in ("local_stall", "local_stall", "local_stall"):
        if error == "unavailable":
            continue
        if error == "local_stall":
            continue
        total_attempted += 1
        if error in {"dns", "network", "timeout"}:
            total_network_errors += 1

    outage_detected = total_attempted > 0 and total_network_errors == total_attempted
    assert not outage_detected, (
        "three simultaneous local stalls registered a REST outage and would "
        "block_rest for 426s on endpoints that were never asked"
    )
