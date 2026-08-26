"""
The brain must never stall the price feed.

`BrainBridge._post` is a synchronous `http.client` call with a 30-second
timeout, and it is reached from `bot.py::_handle_sample` -- the callback every
market stream feeds. A py-spy dump of the live production process caught the
event loop parked inside it:

    readinto (socket.py:719)            <- blocking socket read
    getresponse (http/client.py:1430)
    _post (trading/brain_bridge.py)
    query_confidence -> _brain_record_entry -> _interpret_predictions
    _handle_sample (trading/bot.py)
    run_forever (asyncio/base_events.py:683)

Every stream shares that loop, so one slow brain query froze price collection
for ALL symbols. Measured: writes arrived in bursts roughly 12 minutes apart --
seven symbols sharing an identical timestamp, then nothing -- while an isolated
stream sustained one write per second.

Measured directly: **36.78 seconds** for a single `query_confidence` in a sync
context against the live node. That is the stall, per call.

The brain is an advisory signal; the price feed is the product. Under a running
loop the query is skipped and the caller treats the `None` answer as "no
opinion", which it already handled. `BRAIN_BRIDGE_ALLOW_BLOCKING=1` restores the
old behaviour for anyone who needs it deliberately.
"""

from __future__ import annotations

import asyncio
import os
import time
import unittest
from unittest import mock

from trading.brain_bridge import (
    BrainBridge,
    _brain_blocking_allowed,
    _event_loop_is_running,
)


class EventLoopDetection(unittest.TestCase):
    def test_false_outside_a_loop(self):
        self.assertFalse(_event_loop_is_running())

    def test_true_inside_a_loop(self):
        async def scenario():
            return _event_loop_is_running()

        self.assertTrue(asyncio.run(scenario()))


class BridgeSkipsWhenALoopIsRunning(unittest.TestCase):
    def setUp(self):
        os.environ.pop("BRAIN_BRIDGE_ALLOW_BLOCKING", None)

    def test_post_returns_immediately_under_a_loop(self):
        """
        The property that protects ingestion.

        No socket is touched, so this cannot be slow even if the node is.
        """
        async def scenario():
            bridge = BrainBridge()
            with mock.patch.object(
                bridge, "_ensure", side_effect=AssertionError("opened a socket")
            ):
                started = time.time()
                result = bridge._post("/brain/integrate", b"{}")
                return result, time.time() - started, bridge._skipped_in_loop

        result, elapsed, skipped = asyncio.run(scenario())
        self.assertIsNone(result)
        self.assertLess(elapsed, 0.05)
        self.assertEqual(skipped, 1)

    def test_query_confidence_degrades_to_no_opinion(self):
        """A skipped query must look like 'no opinion', not an error."""
        async def scenario():
            bridge = BrainBridge()
            return bridge.query_confidence("features")

        answer, confidence = asyncio.run(scenario())
        self.assertIsNone(answer)
        self.assertEqual(confidence, 0.0)

    def test_skips_are_counted(self):
        """Silently dropping would trade one invisible failure for another."""
        async def scenario():
            bridge = BrainBridge()
            for _ in range(3):
                bridge._post("/brain/observe", b"{}")
            return bridge._skipped_in_loop

        self.assertEqual(asyncio.run(scenario()), 3)

    def test_outside_a_loop_the_call_proceeds(self):
        """
        Synchronous callers -- supervisors, training pushes -- are unaffected.

        Only the event loop is protected, because only it is shared.
        """
        bridge = BrainBridge()
        with mock.patch.object(bridge, "_ensure", return_value=False) as ensure:
            self.assertIsNone(bridge._post("/brain/observe", b"{}"))
            ensure.assert_called_once()
        self.assertEqual(bridge._skipped_in_loop, 0)

    def test_the_old_behaviour_can_be_restored(self):
        async def scenario():
            bridge = BrainBridge()
            with mock.patch.dict(os.environ, {"BRAIN_BRIDGE_ALLOW_BLOCKING": "1"}):
                self.assertTrue(_brain_blocking_allowed())
                with mock.patch.object(bridge, "_ensure", return_value=False) as ensure:
                    bridge._post("/brain/observe", b"{}")
                    return ensure.call_count, bridge._skipped_in_loop

        calls, skipped = asyncio.run(scenario())
        self.assertEqual(calls, 1, "the blocking path should have been taken")
        self.assertEqual(skipped, 0)


if __name__ == "__main__":
    unittest.main()
