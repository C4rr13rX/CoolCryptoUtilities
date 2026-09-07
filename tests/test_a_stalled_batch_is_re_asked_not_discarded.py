"""A REST batch lost to OUR blocked loop must be re-asked, not thrown away.

The sibling fix (``test_a_blocked_loop_does_not_condemn_a_healthy_endpoint``)
taught ``_fetch_rest_price`` to tell the two reasons an aiohttp timeout fires
apart: the endpoint was slow, or this process stalled the loop the timeout is
timed on. It returns ``local_stall`` for the second, and ``_poll_rest_data``
then skips the endpoint -- correctly declining to blame it, and incorrectly
discarding the tick with it. Its own comment says "this endpoint was never
really asked". Nothing then asked it.

MEASURED, 6h to 2026-09-07 05:13 of one production ``logs/system.log``:

    flow samples published        1185
    ticks dropped (no price)      1251
    REST timeouts, local_stall    1230
    upstream HTTP 429                7

The feed was losing 51% of its ticks and 1230 of those failures were ours, not
the network's -- the earlier diagnosis that the feed was being rate limited is
off by more than two orders of magnitude. Loop lag over the same log: n=1203,
p50 16.1s, p90 47.7s, p99 103.7s, max 246.8s, against a 10s fetch budget. At
``REST_CONSENSUS_PARALLEL=3`` all three endpoints in a batch share one stall
and time out together, and ``REST_CONSENSUS_BATCHES=3`` then spends the poll.

Feed rate is upstream of everything this pipeline does: at 265 ticks/h across
12 symbols a position cannot be managed on the minutes timescale the loop
targets, no stop can bind, and ghost round trips -- the only currency that
re-arms a demoted strategy for live -- accrue too slowly to reach the bar.

Both directions are pinned. The retry must NOT fire when an endpoint actually
answered, or a struggling upstream gets re-asked for its trouble.
"""

import asyncio
import time

import pytest

import trading.data_stream as ds
from trading.data_stream import Endpoint, MarketDataStream, RestFetchResult


def _endpoints(n=3):
    return [
        Endpoint(
            name=f"endpoint{i}",
            ws_template=None,
            subscribe_template=None,
            rest_template="https://api.example/{base}/{quote}",
            headers=None,
        )
        for i in range(n)
    ]


def _stream(monkeypatch, *, retries=2):
    """A stream with just enough shape for ``_fetch_batch_past_local_stalls``."""
    stream = MarketDataStream.__new__(MarketDataStream)
    stream.symbol = "AERO-USDC"
    stream.chain = "base"
    stream._local_stall_retries = retries
    stream._local_stall_retry_count = 0
    stream._local_stall_recovered = 0
    # The recovery path logs; keep the test off the production log file.
    monkeypatch.setattr(ds, "log_message", lambda *a, **k: None)
    return stream


def _scripted(stream, rounds):
    """Answer each batch from ``rounds``: one dict of {endpoint name: error}.

    ``None`` as the error is a successful fetch. Records how many rounds were
    consumed so a test can assert the batch was, or was not, re-asked.
    """
    calls = {"rounds": 0, "fetches": 0}
    state = {"index": 0}
    seen_in_round = {"names": set()}

    async def fetch(endpoint, base, quote):
        idx = min(state["index"], len(rounds) - 1)
        plan = rounds[idx]
        if endpoint.name in seen_in_round["names"]:
            seen_in_round["names"] = set()
            state["index"] = min(state["index"] + 1, len(rounds) - 1)
            idx = state["index"]
            plan = rounds[idx]
        seen_in_round["names"].add(endpoint.name)
        calls["fetches"] += 1
        error = plan.get(endpoint.name, "local_stall")
        return RestFetchResult(1.23 if error is None else None, error)

    stream._fetch_rest_price = fetch
    return calls


def _run(stream, endpoints, *, end_time=None):
    async def scenario():
        return await stream._fetch_batch_past_local_stalls(
            endpoints,
            "AERO",
            "USDC",
            end_time=time.time() + 30.0 if end_time is None else end_time,
        )

    return asyncio.run(scenario())


def test_a_batch_lost_entirely_to_our_own_stall_is_re_asked():
    """THE REGRESSION. Old behaviour was one ``gather`` and no retry.

    Against the pre-fix ``_poll_rest_data`` -- a single
    ``asyncio.gather`` whose ``local_stall`` results were ``continue``d past --
    every result here is ``local_stall``, no price is dispatched, and the poll
    logs "no live price available; dropping synthetic tick". That is the 1251
    dropped ticks against 1185 published.
    """
    with pytest.MonkeyPatch.context() as mp:
        stream = _stream(mp)
        endpoints = _endpoints(3)
        # Round 1: our loop stalled and lost all three. Round 2: they answer.
        calls = _scripted(
            stream,
            [
                {"endpoint0": "local_stall", "endpoint1": "local_stall", "endpoint2": "local_stall"},
                {"endpoint0": None, "endpoint1": None, "endpoint2": None},
            ],
        )
        results = _run(stream, endpoints)

    prices = [r.price for _e, r, _l in results if r.error is None]
    assert prices, (
        "a batch that failed only because OUR event loop was blocked returned "
        f"no price: {[r.error for _e, r, _l in results]}. The endpoints were "
        "never asked, so the poll must ask them rather than drop the tick."
    )
    assert stream._local_stall_retry_count == 1
    assert stream._local_stall_recovered == 1
    assert calls["fetches"] == 6, "expected exactly one re-ask of the batch"


def test_an_endpoint_that_answered_is_not_re_asked():
    """The load guard. One real answer means there is nothing to re-ask.

    Without this the retry would double the request rate against an upstream
    that is genuinely struggling -- turning a fix for 7 HTTP 429s per 6h into
    the cause of far more.
    """
    with pytest.MonkeyPatch.context() as mp:
        stream = _stream(mp)
        endpoints = _endpoints(3)
        calls = _scripted(
            stream,
            [
                # endpoint1 reached the host and was refused. That is
                # information about the upstream, not about our loop.
                {"endpoint0": "local_stall", "endpoint1": "http_error", "endpoint2": "local_stall"},
                {"endpoint0": None, "endpoint1": None, "endpoint2": None},
            ],
        )
        _run(stream, endpoints)

    assert stream._local_stall_retry_count == 0, (
        "a batch containing a real endpoint response was re-asked; that adds "
        "load to an upstream that already answered"
    )
    assert calls["fetches"] == 3


def test_a_dead_upstream_is_never_re_asked():
    """A genuine timeout, DNS failure or network error is not our stall."""
    with pytest.MonkeyPatch.context() as mp:
        stream = _stream(mp)
        endpoints = _endpoints(3)
        _scripted(
            stream,
            [{"endpoint0": "timeout", "endpoint1": "dns", "endpoint2": "network"}],
        )
        _run(stream, endpoints)

    assert stream._local_stall_retry_count == 0, (
        "a real outage was re-asked; that both wastes the poll window and "
        "delays outage detection on a genuinely dead host"
    )


def test_the_retry_stops_at_the_end_of_the_poll_window():
    """A price fetched after the poll window is a price nobody reads.

    Worse, the loop it is fetched on is the loop every other symbol's stream
    is waiting for -- so an unbounded retry would spread one symbol's stall
    across the whole feed.
    """
    with pytest.MonkeyPatch.context() as mp:
        stream = _stream(mp, retries=5)
        endpoints = _endpoints(3)
        calls = _scripted(
            stream,
            [{"endpoint0": "local_stall", "endpoint1": "local_stall", "endpoint2": "local_stall"}],
        )
        # A window that has already closed by the time the first batch returns.
        _run(stream, endpoints, end_time=time.time() - 1.0)

    assert stream._local_stall_retry_count == 0
    assert calls["fetches"] == 3, "the retry ran past the end of the poll window"


def test_the_retry_is_bounded_by_its_configured_count():
    """A loop stalled for the whole poll must not spin on it forever."""
    with pytest.MonkeyPatch.context() as mp:
        stream = _stream(mp, retries=2)
        endpoints = _endpoints(2)
        calls = _scripted(
            stream,
            [{"endpoint0": "local_stall", "endpoint1": "local_stall"}],
        )
        results = _run(stream, endpoints)

    assert stream._local_stall_retry_count == 2
    assert stream._local_stall_recovered == 0
    assert calls["fetches"] == 6, "expected the first ask plus exactly 2 retries"
    assert all(r.error == "local_stall" for _e, r, _l in results)


@pytest.mark.parametrize(
    "errors, lost",
    [
        (["local_stall", "local_stall"], True),
        (["local_stall", "unavailable"], True),
        (["local_stall", None], False),
        (["local_stall", "timeout"], False),
        (["local_stall", "rate_limited"], False),
        (["unavailable", "unavailable"], False),
        ([None, None], False),
    ],
)
def test_lost_to_local_stall_classifies_each_batch_shape(errors, lost):
    """``unavailable`` is a missing URL template, not a request that failed.

    It must count as neither an answer (which would block the retry) nor a
    stall (which would trigger one on a batch nobody could have asked).
    """
    results = [
        (
            Endpoint(
                name=f"endpoint{i}",
                ws_template=None,
                subscribe_template=None,
                rest_template="https://api.example/{base}/{quote}",
                headers=None,
            ),
            RestFetchResult(1.0 if error is None else None, error),
            0.0,
        )
        for i, error in enumerate(errors)
    ]
    assert MarketDataStream._lost_to_local_stall(results) is lost
