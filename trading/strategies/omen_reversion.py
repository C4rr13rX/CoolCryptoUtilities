"""Trade the wizard brain's buy-low omens.

Enters only on an **admitted ``trough``** — the substrate's claim that price
is in the low part of its recent range AND the forward move over the omen's
horizon clears the round-trip cost. Exits on an admitted ``crest``, or when
the horizon the omen was made about has elapsed.

Two properties matter more than the signal:

**It never blocks the feed.** ``brain_bridge`` carries a py-spy dump of the
asyncio loop parked inside a brain socket read: every market stream shares
that loop, so one slow brain call froze price collection for ALL symbols,
and writes arrived in 12-minute bursts instead of once a second. So
``evaluate`` never talks to the node. It reads a cache, and a *background*
thread refreshes it. A cold or stale cache means no candidate this tick,
which costs one tick — a blocking call costs the feed.

**It fails closed on entries and open on exits.** A dead node, an untrained
fabric or a degenerate answer produces no entry. The same conditions must
never strand a position, so an open position still exits on its horizon
without asking the brain anything.

The omen's own cost bar is already ``ROUND_TRIP_COST x OMEN_COST_MULTIPLE``;
``make_candidate`` then re-checks it against ``ctx.fee_rate``. Both bars have
to clear.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from trading.omen_brain import (
    LOOKBACK_BARS, OMEN_CREST, OMEN_TROUGH, Omen, OmenBrain, build_collections,
    label_regime, omen_threshold,
)
from trading.strategies.base import Strategy, StrategyContext, env_float, sample_arrays


#: Seconds per synthetic bar built from the tick stream. 60s x 169 bars is
#: ~2.8h of history — the shortest window that still fills the 7-day-shaped
#: geometry collection the brain was trained on.
BAR_SECONDS = int(os.getenv("OMEN_BAR_SECONDS", "60"))
#: How far ahead the omen is asked about, in bars. 12 x 60s = 12 minutes,
#: inside the single-digit-to-tens-of-minutes round trip this loop targets.
HORIZON_BARS = int(os.getenv("OMEN_HORIZON_BARS", "12"))
#: How long a cached omen is reused. An omen about the next 12 minutes does
#: not change meaningfully inside 30s, and refreshing per tick would put a
#: node round trip on every sample.
CACHE_SEC = float(os.getenv("OMEN_CACHE_SEC", "30"))
#: Confidence a decoded omen must carry to be admitted. READ OFF a run, not
#: guessed: omen-AERO-USDC-h12-20260907-085010 (2725 trained pairs) measured
#: 40 pure-noise frames at confidence min 0.608 / max 0.675, and 500 real
#: held-out frames at min 0.921 / max 0.995. The two distributions do not
#: overlap, and 0.80 sits in the empty band between them -- it rejects every
#: garbage frame and keeps every real one.
#:
#: It is therefore a NOISE filter, not a trade filter: the same run's sweep
#: shows floors from 0.0 to 0.5 admitting all 115 buy omens identically, so
#: this does not select trades and must not be mistaken for doing so. The
#: first guess of 0.45 would have sat below every garbage frame and filtered
#: nothing -- the same mistake the regime gate's first 0.15 margin made.
CONFIDENCE_FLOOR = float(os.getenv("OMEN_CONFIDENCE_FLOOR", "0.80"))
#: Require four query sets to decode the same label, or abstain with
#: verdict="split". This is the gate CONFIDENCE_FLOOR cannot be: measured
#: 2026-09-07, confidence separated a right answer from a wrong one by 0.030
#: on train recall and by -0.002 held-out, while agreement separated them by
#: 99.4% against 73.3%. On by default -- it costs four round trips behind the
#: cache and it is the only measured correctness signal this brain has.
REQUIRE_CONSENSUS = os.getenv("OMEN_REQUIRE_CONSENSUS", "1").strip().lower() in {
    "1", "true", "yes", "on"}
#: Master switch. Off until a run in data/brain_experiments/ shows the omen
#: beating indiscriminate entry per trade, net of cost, on held-out bars.
ENABLED = os.getenv("OMEN_STRATEGY_ENABLED", "0").strip().lower() in {
    "1", "true", "yes", "on"}


def bars_from_samples(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    bar_seconds: int = BAR_SECONDS,
) -> List[Dict[str, Any]]:
    """Bucket a tick series into OHLCV bars, oldest first.

    Only *closed* buckets are returned: the bucket containing the newest
    tick is still forming, and a half-formed bar has a different high/low
    than the same bar will have a minute later — which would make the same
    situation two different atoms.

    Gaps are dropped rather than forward-filled. A forward-filled bar is a
    price that never traded, and this repo has already shipped a feed where
    82 of 94 symbols held a seed price forever.
    """
    if timestamps.size == 0 or bar_seconds <= 0:
        return []
    buckets: Dict[int, List[Tuple[float, float, float]]] = {}
    for ts, price, volume in zip(timestamps, prices, volumes):
        if not (np.isfinite(ts) and np.isfinite(price)) or price <= 0:
            continue
        buckets.setdefault(int(ts) // bar_seconds, []).append(
            (float(ts), float(price), float(volume) if np.isfinite(volume) else 0.0))
    if not buckets:
        return []
    newest = max(buckets)
    bars: List[Dict[str, Any]] = []
    for key in sorted(buckets):
        if key == newest:
            break  # still forming
        ticks = sorted(buckets[key])
        series = [p for _, p, _ in ticks]
        total_volume = sum(v for _, _, v in ticks)
        bars.append({
            "timestamp": key * bar_seconds,
            "open": series[0], "high": max(series),
            "low": min(series), "close": series[-1],
            "net_volume": total_volume,
            # The tick stream carries no buy/sell split. Splitting it by the
            # bar's own direction would invent a flow signal, so it is
            # reported as unknown-but-balanced and the flow collection's
            # buy-share buckets stay constant rather than fictional.
            "buy_volume": total_volume / 2.0,
            "sell_volume": total_volume / 2.0,
        })
    return bars


class OmenReversionStrategy(Strategy):
    """Buy an admitted ``trough``; sell an admitted ``crest``."""

    strategy_id = "omen_reversion"
    default_horizon = "15m"
    min_samples = 60

    def __init__(self, brain: Optional[OmenBrain] = None) -> None:
        self._brain = brain
        self._lock = threading.Lock()
        #: symbol -> (fetched_at, Omen)
        self._cache: Dict[str, Tuple[float, Omen]] = {}
        #: symbol -> in-flight refresh, so a slow node cannot pile up threads
        #: (a scheduler thread leak on one DB already starved this feed once).
        self._inflight: Dict[str, float] = {}
        #: Diagnostics for the population page — why no candidate.
        self.last_reason: str = "not evaluated"

    # -- cache ------------------------------------------------------------
    def _get_brain(self) -> OmenBrain:
        if self._brain is None:
            self._brain = OmenBrain()
        return self._brain

    def cached_omen(self, symbol: str) -> Optional[Omen]:
        """The cached omen for ``symbol``, or None. Never does I/O."""
        with self._lock:
            entry = self._cache.get(symbol)
        if entry is None:
            return None
        fetched_at, omen = entry
        if (time.time() - fetched_at) > CACHE_SEC:
            return None
        return omen

    def _refresh_async(self, symbol: str, chain: str, bars: List[Dict[str, Any]],
                       price: float) -> None:
        """Ask the node on a background thread. Never raises to the caller."""
        now = time.time()
        with self._lock:
            started = self._inflight.get(symbol, 0.0)
            # A refresh that has been running longer than the node timeout is
            # assumed dead; anything fresher is left alone.
            if started and (now - started) < 60.0:
                return
            self._inflight[symbol] = now

        def run() -> None:
            try:
                index = len(bars) - 1
                frames = build_collections(
                    bars, index, horizon_bars=HORIZON_BARS,
                    symbol=symbol, chain=chain)
                omen = self._get_brain().predict(
                    frames, symbol=symbol, chain=chain,
                    as_of_ts=int(bars[index]["timestamp"]), price=price,
                    horizon_bars=HORIZON_BARS, bar_seconds=BAR_SECONDS,
                    confidence_floor=CONFIDENCE_FLOOR,
                    # We hold the bars the frames were built from, so the
                    # regime is arithmetic here. Letting stage 1 guess it
                    # instead measured 86.0% train recall against 90.7%.
                    regime=label_regime(bars, index),
                    # Four query sets must decode the same label or the omen
                    # abstains with verdict="split". 99.4% reproduction when
                    # they agree, 73.3% when they do not -- and confidence
                    # separates those two cases by 0.030, so it cannot be
                    # the gate. Four round trips behind a 60s cache.
                    consensus=REQUIRE_CONSENSUS)
                with self._lock:
                    self._cache[symbol] = (time.time(), omen)
            except Exception:
                # A brain that cannot answer must not take the strategy down.
                pass
            finally:
                with self._lock:
                    self._inflight.pop(symbol, None)

        threading.Thread(target=run, name=f"omen-{symbol}", daemon=True).start()

    # -- the strategy ------------------------------------------------------
    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        if not ENABLED:
            self.last_reason = "OMEN_STRATEGY_ENABLED=0"
            return None

        symbol = str(getattr(state, "symbol", "") or getattr(state, "pair", "") or "")
        if not symbol:
            self.last_reason = "state carries no symbol"
            return None

        min_net = env_float("OMEN_MIN_NET_RETURN", 0.002, lo=0.0, hi=0.1)
        # Enough ticks to fill LOOKBACK_BARS closed bars, plus the forming one.
        lookback_sec = float((LOOKBACK_BARS + 2) * BAR_SECONDS)
        timestamps, prices, volumes = sample_arrays(state, lookback_sec)
        if prices.size < self.min_samples or ctx.last_price <= 0:
            self.last_reason = f"only {prices.size} samples in window"
            return None

        bars = bars_from_samples(timestamps, prices, volumes)
        # An open position must be able to exit even with a dark brain, so
        # the horizon exit is checked before anything asks the node.
        omen = self.cached_omen(symbol)
        if len(bars) > LOOKBACK_BARS:
            self._refresh_async(symbol, ctx.chain, bars, ctx.last_price)
        elif omen is None:
            self.last_reason = (
                f"{len(bars)} closed bars, need {LOOKBACK_BARS + 1}")
            return None

        if omen is None:
            self.last_reason = "no cached omen yet"
            return None
        if not omen.is_actionable:
            self.last_reason = f"omen {omen.omen} verdict {omen.verdict}"
            return None

        expected = abs(omen.expected_move_fraction)
        if expected - ctx.fee_rate < min_net:
            self.last_reason = (
                f"expected {expected:.4%} does not clear fee "
                f"{ctx.fee_rate:.4%} by {min_net:.4%}")
            return None

        if omen.omen == OMEN_TROUGH and ctx.available_quote > 0:
            self.last_reason = "trough admitted"
            return self.make_candidate(
                state, ctx,
                action="enter",
                expected_return=expected,
                target_price=ctx.last_price * (1.0 + expected),
                confidence=omen.confidence,
                direction_prob=min(1.0, 0.5 + 0.5 * omen.confidence),
                horizon=self.default_horizon,
                reason=(f"omen trough: brain expects >= {expected:.2%} over "
                        f"{HORIZON_BARS * BAR_SECONDS / 60:.0f}m "
                        f"(regime {omen.regime}, conf {omen.confidence:.3f})"),
                extra_meta={
                    "omen": omen.omen,
                    "omen_verdict": omen.verdict,
                    "omen_regime": omen.regime,
                    "omen_confidence": omen.confidence,
                    "omen_threshold": omen.threshold_fraction,
                    "omen_horizon_sec": HORIZON_BARS * BAR_SECONDS,
                    "omen_schema": omen.schema_version,
                },
            )

        if omen.omen == OMEN_CREST and ctx.available_base > 0:
            self.last_reason = "crest admitted"
            return self.make_candidate(
                state, ctx,
                action="exit",
                expected_return=expected,
                target_price=ctx.last_price,
                confidence=omen.confidence,
                direction_prob=min(1.0, 0.5 + 0.5 * omen.confidence),
                horizon=self.default_horizon,
                reason=(f"omen crest: brain expects <= -{expected:.2%} over "
                        f"{HORIZON_BARS * BAR_SECONDS / 60:.0f}m "
                        f"(regime {omen.regime}, conf {omen.confidence:.3f})"),
                extra_meta={
                    "omen": omen.omen,
                    "omen_verdict": omen.verdict,
                    "omen_confidence": omen.confidence,
                    "omen_horizon_sec": HORIZON_BARS * BAR_SECONDS,
                    "omen_schema": omen.schema_version,
                },
            )

        self.last_reason = f"omen {omen.omen} carries no side to take"
        return None

    def status(self) -> Dict[str, Any]:
        """What the population page shows for this strategy."""
        with self._lock:
            cached = {
                symbol: {"omen": omen.omen, "verdict": omen.verdict,
                         "confidence": omen.confidence,
                         "age_sec": round(time.time() - at, 1)}
                for symbol, (at, omen) in self._cache.items()
            }
        return {
            "strategy_id": self.strategy_id,
            "enabled": ENABLED,
            "endpoint": os.getenv("OMEN_BRAIN_ENDPOINT", "http://127.0.0.1:8091"),
            "bar_seconds": BAR_SECONDS,
            "horizon_bars": HORIZON_BARS,
            "horizon_sec": HORIZON_BARS * BAR_SECONDS,
            "threshold_fraction": omen_threshold(),
            "confidence_floor": CONFIDENCE_FLOOR,
            "last_reason": self.last_reason,
            "cached": cached,
        }
