"""
The pair-readiness check must not construct a stream to answer a boolean.

`trading/selector.py::_has_streaming_feed` runs for every candidate pair on
every selection pass -- roughly 25 times a minute in production. It used to
answer by building a full `MarketDataStream` and discarding it, which
allocates a MetricsCollector, a 360-slot price deque and endpoint health
tables, runs endpoint selection, and writes a debug log line.

Measured cost: ~13.7ms per construction versus ~0.06ms for the endpoint check
it stands in for -- about 218x. The visible symptom was a production process
at 77 threads and 871MB that kept *configuring* streams (60-180 init events
per 5 minutes) while *starting* almost none (1-18), so the real streams were
starved of scheduler time and each symbol recorded roughly one tick instead of
one per second.

That made it the binding constraint on trade frequency once the feed itself
was fixed, which is why it is guarded rather than merely tidied.
"""

from __future__ import annotations

import time
import unittest
from unittest import mock

import trading.data_stream as ds
from trading.selector import _has_streaming_feed


class ReadinessCheckIsCheap(unittest.TestCase):
    def test_it_does_not_construct_a_market_data_stream(self):
        """
        The property that actually matters, asserted structurally.

        A timing test alone would be flaky on a loaded machine; this fails
        deterministically if anyone reintroduces the construction.
        """
        with mock.patch.object(
            ds, "MarketDataStream", side_effect=AssertionError("constructed a stream")
        ):
            # Patch the name the selector module resolved at import time too.
            import trading.selector as sel

            with mock.patch.object(
                sel, "MarketDataStream", side_effect=AssertionError("constructed a stream")
            ):
                self.assertTrue(_has_streaming_feed("BASECAT-USDC", chain="base"))

    def test_it_agrees_with_what_a_real_stream_would_build(self):
        """
        Speed is worthless if the answer drifts.

        `has_price_endpoints` reuses `_build_endpoints`, so the check and a
        real stream cannot disagree about which pairs are priceable.
        """
        for symbol in ("BASECAT-USDC", "ETH-USDC", "AERO-USDC", "CBXRP-USDC"):
            probe = ds.has_price_endpoints(symbol)
            real = bool(ds.MarketDataStream(symbol=symbol, chain="base").endpoints)
            self.assertEqual(probe, real, f"{symbol}: probe={probe} real={real}")

    def test_many_checks_stay_fast(self):
        """
        A loose upper bound -- generous enough not to flake, tight enough to
        catch a return to per-call stream construction (which would need
        ~6800ms for this loop).
        """
        start = time.time()
        for _ in range(200):
            _has_streaming_feed("BASECAT-USDC", chain="base")
        elapsed_ms = (time.time() - start) * 1000
        self.assertLess(
            elapsed_ms, 1000.0,
            f"200 readiness checks took {elapsed_ms:.0f}ms; a stream is likely "
            "being constructed per call again",
        )

    def test_an_unknown_symbol_does_not_exclude_the_pair(self):
        """
        Fail open: an error answering the question must not drop the pair.

        The caller has its own historical-data fallback; silently excluding
        pairs here would shrink the traded universe for a parsing quirk.
        """
        self.assertTrue(ds.has_price_endpoints("!!!not-a-symbol!!!"))


if __name__ == "__main__":
    unittest.main()
