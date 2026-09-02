from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from db import TradingDatabase, get_db
from trading.metrics import FeedbackSeverity, MetricStage, MetricsCollector


class SwapValidator:
    """
    Lightweight guard that scores a proposed swap against recent liquidity,
    execution quality, and volatility before allowing it to proceed.

    The liquidity clause compares the trade against observed per-sample volume.
    That comparison is only meaningful when the feed reports volume at all.
    Every tick this stack records carries ``volume=0`` -- the DexScreener REST
    consensus path publishes price only -- so dividing by the observed mean
    turned "we never measured volume" into "this pair has no liquidity", the
    most extreme violation the clause can express. A $0.35 trade on AERO
    (1.26M pooled, 386k/h traded, in USD) scored a liquidity ratio of 348,862
    against a limit of 0.35 and was refused, as was every other live entry on
    every pair, forever. Absent volume is now carried as *unmeasured* and
    answered from an independent per-symbol reading (DexScreener pair volume,
    via the ATF signal feed or a discovery swap probe); when even that is
    missing the ratio stays undefined and an absolute notional cap stands in
    for it, because a number you never measured cannot be a violation.
    """

    def __init__(
        self,
        *,
        db: Optional[TradingDatabase] = None,
        lookback_sec: Optional[int] = None,
        max_liquidity_ratio: Optional[float] = None,
        min_execution_ratio: Optional[float] = None,
        max_slippage: Optional[float] = None,
        max_volatility: Optional[float] = None,
    ) -> None:
        self.db = db or get_db()
        self.metrics = MetricsCollector(self.db)
        self.lookback_sec = lookback_sec or int(os.getenv("SWAP_GUARD_LOOKBACK_SEC", "7200"))
        self.max_liquidity_ratio = max_liquidity_ratio or float(os.getenv("SWAP_GUARD_LIQUIDITY_RATIO", "0.35"))
        self.min_execution_ratio = min_execution_ratio or float(os.getenv("SWAP_GUARD_MIN_EXEC_RATIO", "0.82"))
        self.max_slippage = max_slippage or float(os.getenv("SWAP_GUARD_MAX_SLIPPAGE", "0.045"))
        self.max_volatility = max_volatility or float(os.getenv("SWAP_GUARD_MAX_VOLATILITY", "0.18"))
        # Horizon the volatility number is expressed over. See
        # ``_estimate_volatility``: the threshold above only means something
        # once the measurement is pinned to a span of time.
        self.volatility_horizon_sec = float(
            os.getenv("SWAP_GUARD_VOL_HORIZON_SEC", "3600")
        )
        # A tick this far from the window median is a denomination artifact,
        # not a price move. 50x in one tick is not something a tradeable pair
        # does; it is the feed reporting a different asset under the same name.
        self.volatility_outlier_factor = float(
            os.getenv("SWAP_GUARD_VOL_OUTLIER_FACTOR", "50")
        )
        # Returns spanning a hole this many times the median cadence are not
        # adjacent ticks and must not be differenced as though they were.
        self.volatility_max_gap_factor = float(
            os.getenv("SWAP_GUARD_VOL_MAX_GAP_FACTOR", "4")
        )
        # Above this share of discarded ticks the window is not a price series
        # for one asset and no volatility can be read off it.
        self.volatility_max_contamination = float(
            os.getenv("SWAP_GUARD_VOL_MAX_CONTAMINATION", "0.25")
        )
        # A window that never moved is a stuck feed, not a calm market. Both
        # bounds exist so the verdict needs an actual observation behind it:
        # enough returns to have seen movement if there were any, over enough
        # time that a pair being traded on a 5-30 minute horizon should have
        # printed a different number at least once.
        self.volatility_frozen_min_returns = float(
            os.getenv("SWAP_GUARD_VOL_FROZEN_MIN_RETURNS", "4")
        )
        self.volatility_frozen_min_span_sec = float(
            os.getenv("SWAP_GUARD_VOL_FROZEN_MIN_SPAN_SEC", "900")
        )
        # How far the price we are about to trade at may sit from the pair's
        # own trailing median. Chosen from the measured distribution, not from
        # taste: over 11,249 ticks scored against their trailing 2h median,
        # p99 is 1.61x and p99.5 is 2.89x, and the distribution is EMPTY
        # between 3x and 5x -- 0.489% of ticks exceed 3x and the same 0.489%
        # exceed 5x. Real moves stop near 2.9x; artifacts resume at 5x and run
        # to 7e13x. 3.0 sits in that gap.
        self.max_price_scale = float(os.getenv("SWAP_GUARD_MAX_PRICE_SCALE", "3.0"))
        # Ceiling on a single trade when no volume basis exists at all. Small
        # enough that no realistically tradeable pair could be moved by it, so
        # the guard still refuses to size blind into an unknown book.
        self.unknown_liquidity_max_usd = float(
            os.getenv("SWAP_GUARD_UNKNOWN_LIQUIDITY_MAX_USD", "25")
        )

    def validate(
        self,
        *,
        symbol: str,
        route: Sequence[str],
        trade_size: float,
        price: float,
        volume: float,
        prediction: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, Dict[str, float], List[str]]:
        symbol_u = symbol.upper()
        samples = self.db.fetch_market_samples_for(symbol_u, limit=360)
        trade_usd = abs(trade_size * price)

        observed_volume_usd = self._average_volume_usd(samples)
        volume_basis = "observed"
        avg_volume_usd = observed_volume_usd
        if avg_volume_usd is None:
            interval = self._sample_interval_sec(samples)
            avg_volume_usd = self._reference_volume_usd(symbol_u, interval)
            volume_basis = "reference" if avg_volume_usd is not None else "unmeasured"
        liquidity_ratio: Optional[float] = None
        if avg_volume_usd is not None and avg_volume_usd > 0:
            liquidity_ratio = trade_usd / avg_volume_usd

        fills = self.db.fetch_trade_fills(limit=100)
        exec_ratio, avg_slippage = self._execution_stats(fills)

        volatility, volatility_measurable, volatility_diag = self._estimate_volatility(samples)

        metrics = {
            "trade_value_usd": trade_usd,
            "avg_volume_usd": float(avg_volume_usd) if avg_volume_usd is not None else -1.0,
            "liquidity_ratio": float(liquidity_ratio) if liquidity_ratio is not None else -1.0,
            "liquidity_measured": 1.0 if liquidity_ratio is not None else 0.0,
            "execution_ratio": exec_ratio,
            "avg_slippage": avg_slippage,
            "volatility": volatility,
            "volatility_measurable": 1.0 if volatility_measurable else 0.0,
        }
        metrics.update(volatility_diag)
        if prediction:
            metrics.update(
                {
                    "pred_direction_prob": float(prediction.get("direction_prob", 0.0)),
                    "pred_margin": float(prediction.get("net_margin", 0.0)),
                }
            )

        price_offset = self._price_scale_offset(samples, price)
        if price_offset is not None:
            metrics["price_scale_offset"] = price_offset

        allowed = True
        reasons: List[str] = []
        if price_offset is not None and price_offset > self.max_price_scale:
            # The price we are about to trade AT is quoted at a different scale
            # from the rest of this pair's recent history. Entering here books
            # an entry the exit cannot be compared against: four ghost outcomes
            # on 2026-08-26 did exactly this (AERO exiting at 1.14 when AERO is
            # $0.478; COMP entering at 42.82 when COMP is $19), producing
            # +174% and +161% "wins" that were denomination changes.
            #
            # This clause exists because the volatility fix opened the gate.
            # Every live entry used to be refused, so a bad entry price could
            # never reach a swap; now one can.
            #
            # Known blind spot, stated rather than papered over: COMP-USDC's
            # artifacts sit at 2.2-2.9x, below this bound. That band is where a
            # genuine two-hour move on a microcap also lives, and price alone
            # cannot separate the two. Catching it needs a second source for
            # the quote, not a lower number here -- lowering it would refuse
            # real moves and still not be a measurement.
            allowed = False
            reasons.append("price_off_scale")
        if liquidity_ratio is not None:
            if liquidity_ratio > self.max_liquidity_ratio:
                allowed = False
                reasons.append("liquidity")
        elif trade_usd > self.unknown_liquidity_max_usd:
            # No volume basis anywhere: the ratio is undefined, so fall back to
            # the only bound that survives without a measurement -- an absolute
            # notional the book cannot plausibly notice.
            allowed = False
            reasons.append("liquidity_unmeasured")
        if exec_ratio < self.min_execution_ratio:
            allowed = False
            reasons.append("execution")
        if avg_slippage > self.max_slippage:
            allowed = False
            reasons.append("slippage")
        if not volatility_measurable:
            # Refuse, but say the true thing: either the window held more than
            # one price scale, or it held one price and never left it. Neither
            # is a volatility that can be compared to the limit, and they are
            # different faults, so they get different names.
            allowed = False
            if volatility_diag.get("vol_frozen"):
                reasons.append("feed_frozen")
            else:
                reasons.append("volatility_unmeasurable")
        elif volatility > self.max_volatility:
            allowed = False
            reasons.append("volatility")

        metrics["allowed"] = 1.0 if allowed else 0.0
        self.metrics.record(
            MetricStage.LIVE_TRADING,
            metrics,
            category="swap_guard",
            meta={
                "symbol": symbol_u,
                "route": list(route),
                "volume_basis": volume_basis,
                "reasons": reasons,
            },
        )
        if not allowed:
            self.metrics.feedback(
                "swap_guard",
                severity=FeedbackSeverity.WARNING,
                label="swap_blocked",
                details={
                    "symbol": symbol_u,
                    "route": list(route),
                    "volume_basis": volume_basis,
                    "reasons": reasons,
                    "metrics": metrics,
                },
            )
        return allowed, metrics, reasons

    def _average_volume_usd(self, samples: Sequence[Dict[str, float]]) -> Optional[float]:
        """Mean per-sample USD volume, or None when the feed reported none.

        None means *unmeasured*, not zero. Returning 0.0 here is what made the
        liquidity ratio unbounded on a feed that never carries volume.
        """
        now = time.time()
        window = []
        for sample in samples:
            ts = float(sample.get("ts") or 0.0)
            if now - ts > self.lookback_sec:
                continue
            price = float(sample.get("price") or 0.0)
            volume = float(sample.get("volume") or 0.0)
            if price > 0 and volume > 0:
                window.append(price * volume)
        if not window:
            return None
        return float(np.mean(window))

    def _sample_interval_sec(self, samples: Sequence[Dict[str, float]]) -> float:
        """Median spacing between samples, so an hourly volume can be scaled
        onto the same per-sample basis the ratio is defined against."""
        stamps = sorted(
            float(s.get("ts") or 0.0)
            for s in samples
            if float(s.get("ts") or 0.0) > 0
        )
        gaps = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
        if not gaps:
            return 300.0
        return float(min(3600.0, max(30.0, float(np.median(gaps)))))

    def _reference_volume_usd(self, symbol: str, interval_sec: float) -> Optional[float]:
        """Per-sample USD volume from an independent per-symbol reading.

        The tick feed publishes price only, but DexScreener pair volume reaches
        this stack two other ways: the ATF signal rows (``volume_h1``) and the
        discovery swap probes (``volume_24h_usd``). Either is scaled down to the
        sample cadence so it means the same thing as the observed mean.
        """
        scale = max(30.0, float(interval_sec)) / 3600.0
        try:
            from services.atf_static_strategy import latest_signals

            best: Optional[float] = None
            for row in latest_signals(float(os.getenv("SWAP_GUARD_REFERENCE_MAX_AGE_SEC", "3600"))):
                if not isinstance(row, dict):
                    continue
                if str(row.get("symbol") or "").upper() != symbol:
                    continue
                hourly = float(row.get("volume_h1") or 0.0)
                if hourly > 0 and (best is None or hourly < best):
                    best = hourly
            if best is not None:
                return best * scale
        except Exception:
            pass
        try:
            hourly = self.db.reference_hourly_volume_usd(symbol)
        except Exception:
            hourly = None
        if hourly and hourly > 0:
            return float(hourly) * scale
        return None

    def _execution_stats(self, fills: Sequence[Dict[str, float]]) -> Tuple[float, float]:
        ratios: List[float] = []
        slippages: List[float] = []
        now = time.time()
        for fill in fills:
            ts = float(fill.get("ts") or 0.0)
            if now - ts > self.lookback_sec:
                continue
            expected_amount = float(fill.get("expected_amount") or 0.0)
            executed_amount = float(fill.get("executed_amount") or 0.0)
            expected_price = float(fill.get("expected_price") or 0.0)
            executed_price = float(fill.get("executed_price") or 0.0)
            if expected_amount > 0:
                ratios.append(executed_amount / expected_amount)
            if expected_price > 0 and executed_price > 0:
                slippages.append(abs(executed_price - expected_price) / expected_price)
        exec_ratio = float(np.mean(ratios)) if ratios else 1.0
        avg_slippage = float(np.mean(slippages)) if slippages else 0.0
        return exec_ratio, avg_slippage

    def _price_scale_offset(
        self, samples: Sequence[Dict[str, float]], price: float
    ) -> Optional[float]:
        """How far the trade price sits from this pair's own recent median.

        Returned as a symmetric factor (2.0 means twice or half), or None when
        there is not enough history to say -- unmeasured is not a violation.

        Measured 2026-09-02: 15 of 163 streamed symbols publish prices more
        than 50x from their own median. For some the contaminated scale is the
        MAJORITY reading -- ARB-USDC's median is 5.3e-7 while ARB trades near
        $0.65 -- so this is deliberately a check on the trade price against the
        recent window, not an attempt to decide which scale is the true one.
        Either way the two cannot be compared, and a position whose entry and
        exit are quoted in different units has no P&L.
        """
        if price <= 0:
            return None
        now = time.time()
        recent = [
            float(s.get("price") or 0.0)
            for s in samples
            if float(s.get("price") or 0.0) > 0
            and now - float(s.get("ts") or 0.0) <= self.lookback_sec
        ]
        if len(recent) < 3:
            return None
        median = float(np.median(recent))
        if median <= 0:
            return None
        return float(max(price / median, median / price))

    def _estimate_volatility(
        self, samples: Sequence[Dict[str, float]]
    ) -> Tuple[float, bool, Dict[str, float]]:
        """Volatility of the pair we are about to trade, over ``lookback_sec``.

        Returns ``(volatility, measurable, diagnostics)``.

        Every live entry for six days was refused with ``swap_guard:volatility``
        and the reason was that this clause was not measuring the trade being
        made. Four independent defects, all found by reading the series the
        guard was actually scoring:

        * **No time window.** The liquidity clause honours ``lookback_sec``;
          this one took whatever 360 rows the table held. Measured 2026-09-02
          those 360 rows spanned **147 hours** for SPACEX-USDC and 144 for
          BSTONK-USDC. The guard was refusing a ten-minute trade because of
          what the pair did six days ago.
        * **Reverse chronological order.** ``fetch_market_samples_for`` is
          ``ORDER BY ts DESC``, so ``np.diff`` walked backwards through time.
          A backwards difference is not a return.
        * **Denomination contamination.** SPACEX-USDC's window held 233 ticks
          near 1.5e-9 and one at 524.37 -- two price scales eleven orders of
          magnitude apart, which is a different asset wearing the same symbol.
          That single pair of rows produced a return of 1.29e11 and set the
          whole reading: the guard reported ``volatility=6.6e10`` against a
          limit of 0.18. A number that large is never a market; it is always
          an artifact, and scoring it as risk hides the real defect.
        * **Scaling by sample count.** ``std * sqrt(min(n, 60))`` expressed the
          answer over "sixty samples", so the same market read calmer whenever
          the feed happened to deliver fewer rows. The horizon is now a span of
          time, which is what the 0.18 threshold can actually be judged against
          and is close to what the old scaling meant in steady state (60
          samples at the observed ~95s cadence is ~1.6h).

        With the series fixed and nothing else changed, BSTONK-USDC reads
        **0.155 against the same 0.18 limit** -- it was inside the operator's
        stated risk appetite all along -- and SPACEX-USDC reads 0.0, because it
        did not move at all in the two hours before it was refused.

        Discarding ticks is not the same as smoothing them away: when too much
        of the window has to be thrown out the series is not one asset's price
        history and the honest answer is that volatility is *unmeasurable*, so
        the guard still refuses -- under its own reason, not as "too volatile".

        **A fifth defect, opened by the four fixes above.** Correcting the
        window turned the reading on a *stuck* feed into 0.0 -- the safest
        score the guard can produce -- so the pairs that are most obviously
        broken became the ones most likely to be traded. Measured 2026-09-02
        over the trailing two hours, 7 of the 20 streamed symbols with enough
        ticks to score printed **one single price for the whole window**:

            1KTO100M-USDC  17 ticks / 4054s   2.70676e-13
            ARB-USDC       46 ticks / 6134s   5.33391e-07
            BASED10-USDC   45 ticks / 5924s   5.52889e-07
            MTGA-USDC      19 ticks / 3389s   8.55727e-06
            SPACEX-USDC    35 ticks / 6018s   1.52589e-09
            SPCX-USDC      34 ticks / 6582s   2.75805e-11
            WOJAK-USDC     12 ticks /  507s   8.26609e-07

        None of those is a price. ARB-USDC settles it: the frozen 5.33e-7 is
        printed 79 times while ARB's real quote, 0.6491, appears 5 times -- the
        wrong number by six orders of magnitude is the *majority* reading. And
        SPACEX-USDC was one of only two symbols the live path was proposing
        entries on at the time.

        So a window that never moved is not low risk; it is a feed that is not
        reporting, and the honest answer is again *unmeasurable*. It refuses
        under ``feed_frozen`` rather than ``volatility_unmeasurable`` because a
        stuck feed and a contaminated one are repaired in different places.

        **A sixth defect, and the same one wearing a disguise.** Testing
        ``max == min`` over the window asks whether the feed held one price,
        but the number the threshold judges is the standard deviation of the
        *filtered* returns, and those two disagree whenever the only price
        change is discarded by the gap clause. Measured 2026-09-02 over the
        trailing two hours, 5 of the 17 streamed symbols with enough ticks to
        score produced a volatility of exactly 0.0 while reading measurable.
        Four were short or sparse enough that the frozen bounds correctly
        withhold a verdict (2 to 6 returns, spans of 163-800s). MTGA-USDC was
        not: 26 ticks over 5694s, two distinct prices, and 23 scored returns
        every one of which was exactly zero. The frozen test is therefore
        applied a second time to the returns actually used, under the same two
        bounds, so "never moved" is judged on the series the guard scored.
        """
        now = time.time()
        rows = sorted(
            (
                (float(s.get("ts") or 0.0), float(s.get("price") or 0.0))
                for s in samples
            ),
            key=lambda row: row[0],
        )
        window = [
            (ts, price)
            for ts, price in rows
            if price > 0 and ts > 0 and now - ts <= self.lookback_sec
        ]
        diag: Dict[str, float] = {
            "vol_window_samples": float(len(window)),
            "vol_dropped_outliers": 0.0,
            "vol_dropped_gaps": 0.0,
            "vol_horizon_sec": self.volatility_horizon_sec,
        }
        if len(window) < 3:
            # Nothing measured is not a violation, the same way an unmeasured
            # volume is not zero liquidity.
            return 0.0, True, diag

        prices = np.array([price for _, price in window], dtype=float)
        median_price = float(np.median(prices))
        keep = np.ones(prices.size, dtype=bool)
        if median_price > 0 and self.volatility_outlier_factor > 1:
            ratio = np.maximum(prices / median_price, median_price / prices)
            keep = ratio <= self.volatility_outlier_factor
        dropped = int(prices.size - int(keep.sum()))
        diag["vol_dropped_outliers"] = float(dropped)
        if dropped and dropped / float(prices.size) > self.volatility_max_contamination:
            diag["vol_contamination"] = dropped / float(prices.size)
            return 0.0, False, diag

        stamps = np.array([ts for ts, _ in window], dtype=float)[keep]
        prices = prices[keep]
        if prices.size < 3:
            return 0.0, True, diag

        span = float(stamps[-1] - stamps[0])
        diag["vol_window_span_sec"] = span
        if (
            float(prices.size - 1) >= self.volatility_frozen_min_returns
            and span >= self.volatility_frozen_min_span_sec
            and float(prices.max()) == float(prices.min())
        ):
            # Not a calm market: the same number, repeated, for long enough
            # that any live pair would have moved. Scoring it 0.0 would hand
            # the guard's best grade to its worst input.
            diag["vol_frozen"] = 1.0
            return 0.0, False, diag

        gaps = np.diff(stamps)
        median_gap = float(np.median(gaps)) if gaps.size else 0.0
        returns = np.diff(prices) / prices[:-1]
        adjacent = np.isfinite(returns)
        if median_gap > 0 and self.volatility_max_gap_factor > 0:
            # A fourteen-hour hole between two prints is not one tick's move.
            adjacent &= gaps <= median_gap * self.volatility_max_gap_factor
        diag["vol_dropped_gaps"] = float(int(returns.size - int(adjacent.sum())))
        returns = returns[adjacent]
        if returns.size == 0 or median_gap <= 0:
            return 0.0, True, diag

        if (
            float(returns.size) >= self.volatility_frozen_min_returns
            and span >= self.volatility_frozen_min_span_sec
            and not np.any(returns)
        ):
            # Frozen again, but only visible *after* filtering. The check above
            # asks whether the window held one price; this one asks whether the
            # series actually scored ever moved, which is the number the
            # threshold is compared against. They come apart whenever the only
            # price change sits across a dropped gap.
            #
            # MTGA-USDC, measured 2026-09-02: 26 ticks over 5694s holding two
            # distinct prices, so ``max != min`` and the window looked alive.
            # The single transition spanned a 1378s hole against a 128s median
            # cadence, so the gap clause discarded it -- correctly, it is not
            # one tick's move -- leaving 23 returns of exactly 0.0. The guard
            # read volatility 0.0, measurable, and handed its best possible
            # grade to a feed that never reported a move.
            diag["vol_frozen"] = 1.0
            return 0.0, False, diag

        # Express the per-tick dispersion over a fixed horizon. Never scale up
        # by more than the window actually contains -- extrapolating an hour of
        # risk from four ticks would be inventing the number, not reading it.
        steps = min(self.volatility_horizon_sec / median_gap, float(returns.size))
        diag["vol_median_gap_sec"] = median_gap
        diag["vol_returns"] = float(returns.size)
        return float(np.std(returns) * np.sqrt(max(steps, 1.0))), True, diag

    def plan_transition(
        self,
        *,
        positions: Dict[str, Dict[str, Any]],
        exposure: Dict[str, float],
        readiness: Dict[str, Any],
        risk_budget: float,
        pending_decision: Optional[Dict[str, Any]] = None,
        wallet_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        readiness = readiness or {}
        wallet_state = wallet_state or readiness.get("wallet_state") or {}
        ghost_meta = readiness.get("ghost_validation") or {}
        ghost_ready = bool(
            readiness.get(
                "ghost_ready",
                ghost_meta.get("ready", True) if isinstance(readiness, dict) else True,
            )
        )
        ghost_reason = str(readiness.get("ghost_reason") or ghost_meta.get("reason") or "")
        tail_risk = float(ghost_meta.get("tail_risk", readiness.get("ghost_tail_risk", 0.0)))
        tail_guard = float(
            ghost_meta.get(
                "tail_guardrail",
                ghost_meta.get("tail_guard", float(os.getenv("GHOST_TAIL_GUARDRAIL", os.getenv("GHOST_TAIL_GUARD", "0.0")))),
            )
        )
        ghost_samples = int(readiness.get("ghost_samples", ghost_meta.get("samples", 0)))
        ghost_min_trades = int(
            ghost_meta.get(
                "min_trades",
                int(os.getenv("MIN_GHOST_TRADES_FOR_PROMOTION", os.getenv("MIN_GHOST_TRADES_OVERRIDE", "0"))),
            )
        )
        if ghost_samples < ghost_min_trades and ghost_min_trades > 0:
            ghost_ready = False
            ghost_reason = ghost_reason or "ghost_sample_gap"
        tail_limit_hit = tail_guard > 0 and tail_risk > tail_guard
        if tail_limit_hit:
            ghost_ready = False
            ghost_reason = ghost_reason or "tail_risk"
        gross_exposure = float(sum(abs(val) for val in exposure.values()))
        max_pending = max(1, int(os.getenv("LIVE_MAX_PENDING_POSITIONS", "4")))
        precision = float(readiness.get("precision", 0.0))
        recall = float(readiness.get("recall", 0.0))
        confidence_floor = float(os.getenv("LIVE_CONFIDENCE_FLOOR", "0.65"))
        confidence_margin = min(precision, recall) - confidence_floor
        budget_scale = max(0.2, min(1.0, 0.6 + confidence_margin))
        tail_headroom = 1.0
        if tail_guard > 0:
            tail_headroom = max(0.25, min(1.0, (tail_guard - tail_risk) / max(tail_guard, 1e-9)))
        adjusted_budget = max(0.05, risk_budget * budget_scale * tail_headroom)
        capital_deficit = max(
            0.0,
            float(wallet_state.get("min_capital_usd", 0.0)) - float(wallet_state.get("stable_usd", 0.0)),
        )
        sparse_wallet = bool(wallet_state.get("sparse"))
        fragmented_wallet = bool(wallet_state.get("fragmented"))
        fragment_ratio = float(wallet_state.get("fragment_ratio", 0.0))
        native_starved = bool(wallet_state.get("native_starved", False))
        native_gap = float(wallet_state.get("native_buffer_gap_usd", 0.0))
        reasons: List[str] = []
        if not ghost_ready:
            reasons.append("ghost_not_ready")
        if ghost_samples < ghost_min_trades and ghost_min_trades > 0:
            reasons.append("ghost_sample_gap")
        if tail_limit_hit and "tail_risk" not in reasons:
            reasons.append("tail_risk")
        if sparse_wallet:
            reasons.append("sparse_wallet")
        if fragmented_wallet:
            reasons.append("fragmented_wallet")
        if capital_deficit > 0:
            reasons.append("capital_deficit")
        if native_starved:
            reasons.append("native_starved")
        if gross_exposure > adjusted_budget:
            reasons.append("exposure_limit")
        if len(positions) > max_pending:
            reasons.append("pending_limit")
        if not ghost_ready:
            adjusted_budget = 0.0
        allowed = (
            ghost_ready
            and not sparse_wallet
            and not fragmented_wallet
            and not native_starved
            and capital_deficit <= 0
            and gross_exposure <= adjusted_budget
            and len(positions) <= max_pending
        )
        snapshot = {
            "allowed": allowed,
            "gross_exposure": gross_exposure,
            "risk_budget": risk_budget,
            "adjusted_risk_budget": adjusted_budget,
            "confidence_margin": confidence_margin,
            "pending_positions": len(positions),
            "readiness": readiness,
            "wallet_state": {
                "sparse": sparse_wallet,
                "capital_deficit": capital_deficit,
                "stable_usd": float(wallet_state.get("stable_usd", 0.0)),
                "fragmented": fragmented_wallet,
                "fragment_ratio": fragment_ratio,
                "native_starved": native_starved,
                "native_buffer_gap_usd": native_gap,
            },
            "block_reasons": reasons,
        }
        bus_swap_plan = None
        if sparse_wallet and capital_deficit > 0:
            bus_swap_plan = {
                "action": "swap_to_stable",
                "reduce_position": float(capital_deficit),
                "reason": "sparse_wallet",
            }
        if native_starved and bus_swap_plan is None:
            target_usd = native_gap if native_gap > 0 else float(os.getenv("GAS_MIN_REFILL_USD", "5"))
            if float(wallet_state.get("stable_usd", 0.0)) > 0:
                bus_swap_plan = {
                    "action": "swap_stable_to_native",
                    "reason": "native_starved",
                    "target_usd": target_usd,
                }
            else:
                bus_swap_plan = {"action": "freeze_live", "reason": "native_starved", "target_usd": target_usd}
        if fragmented_wallet and bus_swap_plan is None:
            bus_swap_plan = {
                "action": "consolidate_fragments",
                "reason": "fragmented_wallet",
                "dust_tokens": list(wallet_state.get("dust_tokens", []))[:8],
            }
        if not ghost_ready and bus_swap_plan is None:
            bus_swap_plan = {"action": "freeze_live", "reason": ghost_reason or "ghost_not_ready"}
        if bus_swap_plan is None and tail_guard > 0 and tail_risk >= tail_guard * 0.9:
            bus_swap_plan = {
                "action": "freeze_live",
                "reason": "tail_risk_headroom" if tail_risk < tail_guard else "tail_risk",
                "tail_risk": tail_risk,
                "tail_guardrail": tail_guard,
            }
        if not allowed and exposure and bus_swap_plan is None:
            try:
                symbol, value = max(exposure.items(), key=lambda kv: abs(kv[1]))
                bus_swap_plan = {
                    "symbol": symbol,
                    "action": "rebalance_to_stable",
                    "reduce_position": float(abs(value) * 0.5),
                    "reason": "exposure_above_budget",
                }
            except Exception:
                bus_swap_plan = {"reason": "exposure_above_budget"}
        snapshot["bus_swap_plan"] = bus_swap_plan
        if pending_decision:
            snapshot["pending_decision"] = {
                "action": pending_decision.get("action"),
                "symbol": pending_decision.get("symbol"),
                "size": pending_decision.get("size"),
            }
        snapshot["risk_flags"] = {
            "ghost_ready": ghost_ready,
            "ghost_reason": ghost_reason,
            "capital_deficit": capital_deficit,
            "sparse_wallet": sparse_wallet,
            "ghost_samples": ghost_samples,
            "ghost_min_trades": ghost_min_trades,
            "tail_risk": tail_risk,
            "tail_guardrail": tail_guard,
            "fragmented_wallet": fragmented_wallet,
            "fragment_ratio": fragment_ratio,
            "native_starved": native_starved,
            "native_buffer_gap_usd": native_gap,
        }
        return snapshot
