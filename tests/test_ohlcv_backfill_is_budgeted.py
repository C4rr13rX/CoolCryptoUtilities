"""OHLCV backfill must never hold the bootstrap thread without a budget.

No market stream exists until `selector.build()` returns, so every second spent
inside pair selection is a hole in `market_stream`. Two earlier passes fixed one
venue at a time and the stall simply moved to the next one:

    2026-09-02  the 142-source news harvest inside `_run_download`
    2026-09-03  Binance answering HTTP 451 from this host

Measured 2026-09-04 with py-spy against production pid 416880 -- fourteen
minutes after start, zero rows written to market_stream in that whole window --
the MainThread was parked one venue further along:

    select_pairs -> try_add_candidate -> _ensure_ohlcv -> download_pair
      -> download_pair_coinbase -> requests.get(api.exchange.coinbase.com)

Timed directly against the live API on the same host, same minute:

    MOONBASE-USDC (absent from Coinbase)   0.9s
    CBETH-USDC    (13645 candles)         77.6s

77.6s is 90 days of 5-minute candles paginated 300 at a time: 46 serial
requests with a 0.35s delay between each. `try_add_candidate` pays that once per
candidate lacking candles, and `_run_download` on the same path waits on a
subprocess for up to DOWNLOAD_SUBPROCESS_TIMEOUT_SEC (300s) more.

So the budget is placed at the mechanism instead of the venue. The properties:

  1. past the deadline `_ensure_ohlcv` performs NO network I/O and returns False
  2. the deferred symbol is queued and backfilled off the selection thread
  3. deadline=None -- every pre-existing caller, and the background worker --
     keeps the old unbounded behaviour, so this is a latency fix and not a
     quiet tightening of which pairs are eligible
  4. a whole selection pass is bounded by ONE budget, not one per candidate
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from unittest import mock

import trading.selector as selector


class BackfillRespectsTheDeadline(unittest.TestCase):
    def setUp(self):
        with selector._OHLCV_DEFER_LOCK:
            selector._OHLCV_DEFERRED.clear()

    def test_past_deadline_does_no_io_and_reports_missing(self):
        with mock.patch.object(selector, "_ohlcv_exists", return_value=False), \
                mock.patch.object(selector, "_ensure_assignment_template") as template, \
                mock.patch.object(selector, "_run_download") as run_download, \
                mock.patch.object(selector, "_defer_ohlcv_backfill") as defer, \
                mock.patch("services.cex_ohlcv_fallback.download_pair") as download:
            started = time.monotonic()
            result = selector._ensure_ohlcv(
                "base", "CBETH-USDC", deadline=time.monotonic() - 1.0
            )
            elapsed = time.monotonic() - started

        # Range/type checked by measurement, not inferred from the signature.
        self.assertIs(result, False)
        self.assertLess(elapsed, 0.5, "a deferred backfill still blocked")
        template.assert_not_called()
        run_download.assert_not_called()
        download.assert_not_called()
        defer.assert_called_once_with("base", "CBETH-USDC", None)

    def test_before_deadline_still_downloads(self):
        """The budget must not stop backfills that fit inside it."""
        with mock.patch.object(selector, "_ohlcv_exists", return_value=False), \
                mock.patch.object(selector, "_ensure_assignment_template",
                                  return_value={"pairs": {}}), \
                mock.patch.object(selector, "_update_assignment"), \
                mock.patch.object(selector, "_run_download"), \
                mock.patch.object(selector, "_defer_ohlcv_backfill") as defer, \
                mock.patch("services.cex_ohlcv_fallback.download_pair",
                           return_value=("stub", [])) as download:
            selector._ensure_ohlcv(
                "base", "CBETH-USDC", deadline=time.monotonic() + 600.0
            )

        defer.assert_not_called()
        download.assert_called_once()

    def test_no_deadline_keeps_the_old_unbounded_path(self):
        """deadline=None is the default, so no existing caller changes shape."""
        with mock.patch.object(selector, "_ohlcv_exists", return_value=False), \
                mock.patch.object(selector, "_ensure_assignment_template",
                                  return_value={"pairs": {}}), \
                mock.patch.object(selector, "_update_assignment"), \
                mock.patch.object(selector, "_run_download"), \
                mock.patch.object(selector, "_defer_ohlcv_backfill") as defer, \
                mock.patch("services.cex_ohlcv_fallback.download_pair",
                           return_value=("stub", [])) as download:
            selector._ensure_ohlcv("base", "CBETH-USDC")

        defer.assert_not_called()
        download.assert_called_once()


class DeferredSymbolsAreActuallyBackfilled(unittest.TestCase):
    """Deferring is a handoff, not a drop. If it drops, pairs never join."""

    def setUp(self):
        with selector._OHLCV_DEFER_LOCK:
            selector._OHLCV_DEFERRED.clear()

    def test_worker_retries_the_symbol_without_a_deadline(self):
        seen = []

        def record(chain, symbol, data_root=None, *, deadline=None):
            seen.append((chain, symbol, deadline))
            return True

        with mock.patch.object(selector, "_ensure_ohlcv", side_effect=record):
            selector._defer_ohlcv_backfill("base", "cbeth-usdc", Path("data/x"))
            worker = selector._OHLCV_DEFER_THREAD
            self.assertIsNotNone(worker)
            worker.join(timeout=5.0)
            self.assertFalse(worker.is_alive(), "backfill worker did not finish")

        self.assertEqual(seen, [("base", "CBETH-USDC", None)])

    def test_the_same_symbol_is_not_queued_twice(self):
        with mock.patch.object(selector, "_drain_ohlcv_backfill"):
            selector._defer_ohlcv_backfill("base", "CBETH-USDC", None)
            selector._defer_ohlcv_backfill("BASE", "cbeth-usdc", None)
        with selector._OHLCV_DEFER_LOCK:
            self.assertEqual(list(selector._OHLCV_DEFERRED), [("base", "CBETH-USDC")])


class OneBudgetCoversTheWholePass(unittest.TestCase):
    """Ten candidates at 78s each is thirteen minutes of dark feed.

    The budget is per selection pass, so N slow candidates cost the budget once
    rather than N times.
    """

    SLOW_SEC = 0.6
    BUDGET_SEC = 1.0

    def test_selection_is_bounded_by_the_budget_not_by_candidate_count(self):
        symbols = [f"SLOW{i}-USDC" for i in range(10)]
        candidates = [
            selector.PairCandidate(
                symbol=sym,
                tokens=sym.split("-"),
                avg_volume=1e6,
                volatility=0.05,
                score=1.0,
                datapath=Path("."),
            )
            for sym in symbols
        ]
        downloads = []

        def slow_download(symbol, **_kwargs):
            downloads.append(symbol)
            time.sleep(self.SLOW_SEC)
            return ("stub", [])

        with mock.patch.object(selector, "_OHLCV_BACKFILL_BUDGET_SEC", self.BUDGET_SEC), \
                mock.patch.object(selector, "analyse_historical_pairs", return_value=candidates), \
                mock.patch.object(selector, "load_watchlists", return_value={}), \
                mock.patch.object(selector, "_load_top_symbols", return_value=[]), \
                mock.patch.object(selector, "PortfolioState") as portfolio, \
                mock.patch.object(selector, "_ohlcv_exists", return_value=False), \
                mock.patch.object(selector, "_has_live_price", return_value=True), \
                mock.patch.object(selector, "_has_streaming_feed", return_value=True), \
                mock.patch.object(selector, "_ensure_assignment_template",
                                  return_value={"pairs": {}}), \
                mock.patch.object(selector, "_update_assignment"), \
                mock.patch.object(selector, "_run_download"), \
                mock.patch.object(selector, "_defer_ohlcv_backfill"), \
                mock.patch.object(selector, "DEFAULT_LIVE_PAIRS", []), \
                mock.patch("services.cex_ohlcv_fallback.download_pair",
                           side_effect=slow_download):
            portfolio.return_value.holdings = {}
            started = time.monotonic()
            selector.select_pairs(limit=10)
            elapsed = time.monotonic() - started

        # Unbudgeted this is 10 * SLOW_SEC. Budgeted it stops after the first
        # download that crosses the deadline, so at most a couple get through.
        self.assertLess(
            elapsed,
            self.BUDGET_SEC + self.SLOW_SEC * 2,
            f"selection took {elapsed:.2f}s over {len(downloads)} downloads; "
            "the OHLCV budget is not bounding the pass",
        )
        self.assertLess(len(downloads), len(symbols))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
