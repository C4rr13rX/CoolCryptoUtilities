"""Pair selection must not hold the thread the market feed runs on.

Measured 2026-09-02 with py-spy against production pid 10420, thirteen minutes
after start and before a single market stream existed:

    select_pairs -> try_add_candidate -> _ensure_ohlcv -> _run_download
      -> _trigger_news_for_symbols -> collect_news_for_terms -> harvest
      -> _fetch_reddit -> requests.get("https://www.reddit.com/r/.../new/.rss")

`_ensure_ohlcv` answers one question -- does this pair have candles -- and it
ran a 142-source serial news crawl with a 720h lookback and up to 200 crawler
pages to answer it. Once per candidate. data/../download-worker.log holds 3727
of those harvests, re-fetching the same 24 symbols.

`reconcile_pairs` then runs the same selection every few minutes, and it ran it
INLINE on the event loop that every MarketDataStream shares. That is what the
holes in market_stream are -- 20.9 to 44.7 minutes each, all through
2026-09-01, while production was up the whole time and both price endpoints
answered a direct probe in under 300ms.

So the two properties this pins:

  1. ensuring OHLCV never harvests news
  2. reconcile_pairs never runs selection on the event loop
"""

from __future__ import annotations

import asyncio
import time
import unittest
from pathlib import Path
from unittest import mock

import services.background_workers as bw


class EnsureOhlcvSkipsNewsHarvest(unittest.TestCase):
    """The OHLCV path must not reach the news pipeline at all."""

    def _run_download_with(self, *, collect_news, incomplete):
        """Drive _run_download past its I/O with everything else stubbed out."""
        assignment = {"pairs": dict(incomplete)}
        with mock.patch.object(bw, "_load_assignment", return_value=assignment), \
                mock.patch.object(bw, "_try_cex_fallback"), \
                mock.patch.object(bw, "_collect_completed_symbols", return_value=["ETH"]), \
                mock.patch.object(bw, "_trigger_news_for_symbols") as news, \
                mock.patch.object(bw.subprocess, "Popen") as popen:
            popen.return_value.wait.return_value = 0
            bw._run_download("base", Path("data/base.json"), collect_news=collect_news)
        return news

    def test_no_news_when_every_pair_is_already_complete(self):
        news = self._run_download_with(collect_news=False, incomplete={})
        news.assert_not_called()

    def test_no_news_after_an_actual_download(self):
        news = self._run_download_with(
            collect_news=False,
            incomplete={"0xabc": {"symbol": "AERO-USDC", "completed": False}},
        )
        news.assert_not_called()

    def test_background_worker_still_collects_news(self):
        """The fix is scoped to the latency-critical caller, not a removal.

        The download worker owns a thread and news is the point of that pass;
        if this stops firing the brain quietly stops learning from headlines.
        """
        news = self._run_download_with(collect_news=True, incomplete={})
        news.assert_called_once()

    def test_selector_passes_collect_news_false(self):
        """The caller on the feed's thread must ask for the cheap path.

        Asserted at the call site rather than by reading the default, because
        the default is deliberately True for everyone else.
        """
        import trading.selector as selector

        with mock.patch.object(selector, "_ohlcv_exists", return_value=False), \
                mock.patch.object(selector, "_ensure_assignment_template",
                                  return_value={"pairs": {}}), \
                mock.patch.object(selector, "_update_assignment"), \
                mock.patch.object(selector, "_run_download") as run_download, \
                mock.patch("services.cex_ohlcv_fallback.download_pair",
                           return_value=("stub", [])):
            index = Path("data") / "pair_index_base.json"
            if not index.exists():  # pragma: no cover - depends on checkout
                self.skipTest("pair index not present")
            selector._ensure_ohlcv("base", "AERO-USDC", data_root=Path("data/historical_ohlcv"))

        if not run_download.called:
            self.skipTest("symbol not in the local pair index; no download path taken")
        self.assertEqual(run_download.call_args.kwargs.get("collect_news"), False)


class ReconcileDoesNotBlockTheLoop(unittest.TestCase):
    """A slow selection must not stop the loop the streams live on."""

    BLOCK_SEC = 1.0

    def test_reconcile_yields_while_selection_runs(self):
        import trading.selector as selector

        supervisor = selector.GhostTradingSupervisor.__new__(
            selector.GhostTradingSupervisor
        )
        supervisor.bots = []
        supervisor.data_streams = []
        supervisor.pair_limit = 2
        supervisor.stream_total = 4
        supervisor.pipeline = mock.Mock()
        supervisor.pipeline.system_profile = None

        def slow_select(*_args, **_kwargs):
            time.sleep(self.BLOCK_SEC)
            return []

        ticks = 0

        async def heartbeat():
            """Stands in for a market stream: it must keep getting the loop."""
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        async def scenario():
            beat = asyncio.create_task(heartbeat())
            try:
                await supervisor.reconcile_pairs()
            finally:
                beat.cancel()

        with mock.patch.object(selector, "select_pairs", side_effect=slow_select), \
                mock.patch.object(selector, "resolve_pair_limit", return_value=(2, {})), \
                mock.patch("services.atf_static_strategy.latest_signals", return_value=[]):
            asyncio.run(scenario())

        # Inline, the loop is frozen for BLOCK_SEC and the heartbeat records
        # roughly nothing. Off-loop it keeps its ~50/second cadence.
        self.assertGreater(
            ticks,
            10,
            f"heartbeat only advanced {ticks} times while selection ran; "
            "reconcile_pairs is holding the event loop",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
