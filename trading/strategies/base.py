"""Strategy plugin architecture.

Each strategy is an independent, CPU-only signal generator that inspects a
pair's rolling sample window (``RouteState.samples``) and emits candidate
trade directives in the exact ``{"directive", "score", "meta"}`` shape the
CDCL solver already arbitrates. Strategies never import TensorFlow and never
block: everything is arithmetic over the in-memory deque.

Contract with the CDCL solver (trading/cdcl_solver.py clause DB):
  - ``directive.expected_return`` must be the NET-of-nothing raw edge and must
    exceed ``context["fee_rate"]`` to pass ``return_above_fees`` — for exit
    candidates this is the positive expected benefit of exiting now.
  - ``meta["confidence"]`` and ``meta["direction_prob"]`` must be > 0.
  - ``meta["strategy"]`` carries the strategy_id for the per-strategy ledger.
"""
from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from services.token_address_book import is_token_address


def env_float(name: str, default: float, *, lo: float | None = None, hi: float | None = None) -> float:
    try:
        val = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        val = default
    if lo is not None:
        val = max(lo, val)
    if hi is not None:
        val = min(hi, val)
    return val


def env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class StrategyContext:
    """Everything a strategy may consult beyond the RouteState window."""

    chain: str
    last_price: float
    last_volume: float
    fee_rate: float
    available_quote: float
    available_base: float
    risk_budget: float = 1.0
    live_trading: bool = False
    # TF model summary — neutral (0.5/0.5/0.0) whenever TF is unavailable.
    direction_prob: float = 0.5
    confidence: float = 0.5
    net_margin: float = 0.0
    opportunity: Optional[Any] = None  # OpportunitySignal or None
    extras: Dict[str, Any] = field(default_factory=dict)


def sample_arrays(
    state: Any,
    lookback_sec: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(timestamps, prices, volumes) from RouteState.samples, oldest first."""
    samples = list(getattr(state, "samples", []) or [])
    if lookback_sec and samples:
        cutoff = samples[-1][0] - float(lookback_sec)
        samples = [s for s in samples if s[0] >= cutoff]
    if not samples:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty
    arr = np.asarray(samples, dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def ema(values: np.ndarray, span: int) -> np.ndarray:
    if values.size == 0:
        return values
    alpha = 2.0 / (max(int(span), 1) + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def rsi(prices: np.ndarray, period: int = 14) -> float:
    """Wilder RSI of the last `period` moves; 50.0 when undefined."""
    if prices.size < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains = np.clip(deltas, 0.0, None)
    losses = np.clip(-deltas, 0.0, None)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss <= 1e-12:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over `window` bars; NaN until the window is full."""
    w = max(int(window), 1)
    out = np.full(values.size, np.nan, dtype=np.float64)
    if values.size < w:
        return out
    cs = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    out[w - 1:] = (cs[w:] - cs[:-w]) / float(w)
    return out


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing median over `window` bars; NaN until the window is full."""
    w = max(int(window), 1)
    out = np.full(values.size, np.nan, dtype=np.float64)
    if values.size < w:
        return out
    view = np.lib.stride_tricks.sliding_window_view(values, w)
    out[w - 1:] = np.median(view, axis=1)
    return out


def rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing minimum over `window` bars; NaN until the window is full."""
    w = max(int(window), 1)
    out = np.full(values.size, np.nan, dtype=np.float64)
    if values.size < w:
        return out
    view = np.lib.stride_tricks.sliding_window_view(values, w)
    out[w - 1:] = np.min(view, axis=1)
    return out


def rolling_vwap(prices: np.ndarray, volumes: np.ndarray, window: int) -> np.ndarray:
    """Trailing VWAP over `window` bars; NaN where the window carries no volume."""
    w = max(int(window), 1)
    out = np.full(prices.size, np.nan, dtype=np.float64)
    if prices.size < w or volumes.size != prices.size:
        return out
    cs_pv = np.concatenate(([0.0], np.cumsum(prices * volumes, dtype=np.float64)))
    cs_v = np.concatenate(([0.0], np.cumsum(volumes, dtype=np.float64)))
    num = cs_pv[w:] - cs_pv[:-w]
    den = cs_v[w:] - cs_v[:-w]
    with np.errstate(all="ignore"):
        out[w - 1:] = np.where(den > 0, num / den, np.nan)
    return out


def exit_benefit_horizon_sec() -> float:
    """How long "not exiting now" actually lasts.

    A position that is not closed by a signal is closed by the stale clock at
    ``GHOST_STALE_EXIT_SECONDS``, so that is the window over which the benefit
    of exiting now has to be measured.
    """
    return env_float("STRATEGY_EXIT_BENEFIT_HORIZON_SEC",
                     env_float("GHOST_STALE_EXIT_SECONDS", 900.0, lo=60.0, hi=6 * 3600.0),
                     lo=60.0, hi=6 * 3600.0)


def measured_exit_benefit(
    ts: np.ndarray,
    prices: np.ndarray,
    reference: np.ndarray,
    *,
    horizon_sec: Optional[float] = None,
    min_comparable: int = 12,
    comparable_frac: float = 0.8,
) -> Optional[float]:
    """What exiting at this much extension has ACTUALLY been worth on this series.

    Every reversion exit in this package computed its ``expected_return`` as
    the distance between the price and some reference -- the rolling mean, the
    VWAP, the window median, the recent low -- and handed that number to the
    CDCL ``return_above_fees`` clause as the benefit of exiting now. Distance
    from a reference is a measure of EXTENSION. It is not a forecast, and on
    this feed it does not behave like one.

    Measured 2026-09-05 over 14 days of ``market_stream``, 2585 firings of the
    RSI-overbought exit across 18 symbols (feed-contaminated symbols excluded):

        claimed benefit (mean extension)        +5.40%
        realised benefit at  300s               -0.04%   t=-0.40
        realised benefit at  900s               +0.16%   t=+1.50
        realised benefit at 1800s               -0.25%   t=-1.88
        exit leg cost                            0.32%

    The claim overstates the best realised horizon by ~34x, clears the fee at
    no horizon, and price falls after the signal only 41-47% of the time. On
    AERO-USDC -- the only symbol the live strategy is permitted to trade -- the
    realised benefit over 240 firings is -0.09%: exiting is worse than holding
    before paying anything to do it.

    It cost real money. atf_static's live exit at 2026-09-05 16:52 recorded
    reason "RSI 74 overbought, harvesting 8.22%" and realised -1.35% for
    -0.023520, one of the two trades that make its permitted book negative and
    hold it demoted off live trading.

    This is the exit-side twin of the entry defect fixed in e07583e, where the
    gate took its expected return from the strategy's own ``target_price`` and
    so approved all 20 live entries ever taken. Same shape: a number the
    strategy invented, checked against a cost that is real.

    So the claim is replaced by a measurement taken from the same window the
    strategy is already looking at: of the earlier bars that were at least this
    extended, what did the price do over the next ``horizon_sec``? The mean of
    that, sign-flipped, is the benefit of exiting -- floored at zero, because a
    signal whose comparable history went UP has no benefit to offer, not a
    negative one to subtract.

    Returns None when fewer than ``min_comparable`` earlier bars are that
    extended. An unmeasurable benefit is not claimed: this repo already treats
    an unmeasurable COST defaulted to zero as a defect, and the optimistic
    default is the same error with the sign reversed. The position keeps its
    stop, its take-profit and its max-hold either way -- only the reversion
    signal abstains.
    """
    n = int(prices.size)
    if n < 4 or int(ts.size) != n or int(reference.size) != n:
        return None
    horizon = float(horizon_sec) if horizon_sec is not None else exit_benefit_horizon_sec()
    if not math.isfinite(horizon) or horizon <= 0:
        return None
    with np.errstate(all="ignore"):
        ext = np.where(prices > 0, (prices - reference) / prices, np.nan)
    ext_now = float(ext[-1]) if math.isfinite(float(ext[-1])) else float("nan")
    if not math.isfinite(ext_now) or ext_now <= 0.0:
        return None
    # First bar at least `horizon` later than each bar. `ts` is oldest-first.
    fwd_idx = np.searchsorted(ts, ts + horizon, side="left")
    fwd = np.full(n, np.nan, dtype=np.float64)
    have = np.nonzero(fwd_idx < n)[0]
    if have.size:
        base = prices[have]
        with np.errstate(all="ignore"):
            fwd[have] = np.where(base > 0, prices[fwd_idx[have]] / base - 1.0, np.nan)
    comparable = np.isfinite(ext) & np.isfinite(fwd) & (ext >= ext_now * float(comparable_frac))
    comparable[-1] = False  # the bar being judged is not evidence about itself
    if int(np.count_nonzero(comparable)) < max(int(min_comparable), 1):
        return None
    benefit = -float(np.mean(fwd[comparable]))
    if not math.isfinite(benefit):
        return None
    return max(0.0, benefit)


def log_slope_per_min(ts: np.ndarray, prices: np.ndarray) -> float:
    """OLS slope of log-price per minute; 0.0 when degenerate."""
    if prices.size < 4:
        return 0.0
    rel_min = (ts - ts[-1]) / 60.0
    if np.allclose(rel_min, rel_min[0]):
        return 0.0
    safe = np.clip(prices, 1e-12, None)
    try:
        with np.errstate(all="ignore"):
            slope, _ = np.polyfit(rel_min, np.log(safe), 1)
        return float(slope) if math.isfinite(slope) else 0.0
    except Exception:
        return 0.0


class Strategy(ABC):
    """Independent buy-low/sell-high signal generator."""

    strategy_id: str = "base"
    default_horizon: str = "30m"
    #: minimum samples in the window before this strategy will evaluate
    min_samples: int = 16

    def enabled(self) -> bool:
        return env_flag(f"STRATEGY_{self.strategy_id.upper()}_ENABLED", "1")

    @abstractmethod
    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        """Return a candidate dict or None. Must not raise for bad data."""

    # ------------------------------------------------------------------
    # Candidate builders
    # ------------------------------------------------------------------

    def _size_enter(self, ctx: StrategyContext, edge: float) -> float:
        """Quote-denominated size: modest, edge-scaled fraction of quote."""
        min_frac = env_float("STRATEGY_MIN_QUOTE_FRAC", 0.03, lo=0.0, hi=0.5)
        max_frac = env_float("STRATEGY_MAX_QUOTE_FRAC", 0.12, lo=0.01, hi=0.5)
        frac = min(max_frac, max(min_frac, 0.04 + min(0.08, max(edge, 0.0) * 2.0)))
        return ctx.available_quote * frac * max(0.0, min(1.0, ctx.risk_budget))

    def _size_exit(self, ctx: StrategyContext, confidence: float) -> float:
        base_frac = env_float("STRATEGY_EXIT_BASE_FRAC", 0.5, lo=0.1, hi=1.0)
        frac = min(1.0, base_frac + 0.5 * max(0.0, confidence - 0.5))
        return ctx.available_base * frac

    def make_candidate(
        self,
        state: Any,
        ctx: StrategyContext,
        *,
        action: str,
        expected_return: float,
        target_price: float,
        confidence: float,
        reason: str,
        direction_prob: Optional[float] = None,
        horizon: Optional[str] = None,
        quote_size: Optional[float] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build a sized `{"directive", "score", "meta"}` candidate.

        ``expected_return`` is the net edge the strategy expects to capture by
        taking `action` NOW — always positive when the trade is worth taking
        (the CDCL ``return_above_fees`` clause rejects anything <= fee_rate).
        """
        if not (math.isfinite(expected_return) and math.isfinite(target_price)):
            return None
        if target_price <= 0 or expected_return <= 0:
            return None
        confidence = max(0.01, min(1.0, float(confidence)))
        if action == "enter":
            requested_quote = ctx.available_quote if quote_size is None else float(quote_size)
            if requested_quote <= 0 or ctx.last_price <= 0:
                return None
            # Only enter USD-stable-quoted pairs. Base/base pairs (JITOSOL-CBBTC,
            # WBTC-WETH, AERO-WETH ...) carry their 'price' as a token ratio, not
            # USD, so every strategy's PnL/RSI/VWAP math is poisoned — producing
            # dust enters that loop, fantasy profits, and RSI-0 garbage. Exits are
            # still allowed so any existing non-USD position can be closed.
            if env_flag("STRATEGY_STABLE_QUOTE_ONLY", "1"):
                _stable = {"USDC", "USDT", "DAI", "USDBC", "USDC.E", "BUSD", "EURC", "USD", "TUSD", "FDUSD"}
                if str(getattr(state, "quote_token", "")).upper() not in _stable:
                    return None
                # Stable-stable pairs (USDT-USDC, DAI-USDC ...) never move more
                # than fees; entering them just churns near-zero "drawdown"
                # losses that dilute win rates and block graduation.
                if str(getattr(state, "base_token", "")).upper() in _stable:
                    return None
            size_quote = (
                self._size_enter(ctx, expected_return - ctx.fee_rate)
                if quote_size is None else requested_quote
            )
            size = size_quote / max(ctx.last_price, 1e-12)
        elif action == "exit":
            if ctx.available_base <= 0:
                return None
            size = self._size_exit(ctx, confidence)
        else:
            return None
        if size <= 0 or not math.isfinite(size):
            return None

        from trading.scheduler import TradeDirective  # local: avoids import cycle

        # The contract the candidate was actually priced against, if it named
        # one. Strategies pass it in extra_meta; it used to stop there, so the
        # bot re-resolved the ticker instead and refused anything outside its
        # symbol book as token_unresolved -- while for the symbols that DO
        # resolve, a ticker shared by dozens of contracts would have picked
        # one arbitrarily. Validated here rather than trusted: is_token_address
        # rejects the 32-byte Uniswap v4 pool ids that discovery also stores,
        # and the native-coin sentinels, so only a 20-byte ERC-20 address
        # travels on the directive.
        token_address = ""
        if extra_meta:
            candidate_addr = extra_meta.get("token_address")
            if is_token_address(candidate_addr):
                token_address = str(candidate_addr).strip()

        directive = TradeDirective(
            action=action,
            symbol=state.symbol,
            base_token=state.base_token,
            quote_token=state.quote_token,
            size=float(size),
            target_price=float(target_price),
            horizon=horizon or self.default_horizon,
            confidence=confidence,
            expected_return=float(expected_return),
            reason=f"{self.strategy_id}: {reason}",
            strategy_id=self.strategy_id,
            token_address=token_address,
        )
        meta: Dict[str, Any] = {
            "strategy": self.strategy_id,
            "confidence": confidence,
            "direction_prob": float(direction_prob if direction_prob is not None else max(0.5, confidence)),
            "risk_penalty": float(ctx.fee_rate),
            "horizon_weight": 1.0,
            "quality": 1.0,
        }
        if extra_meta:
            meta.update(extra_meta)
        return {
            "directive": directive,
            "score": float(expected_return - ctx.fee_rate),
            "meta": meta,
        }


class StrategyRegistry:
    """Holds strategy instances and fans evaluation out across them.

    One registry per BusScheduler; strategies that keep per-symbol state must
    key it by ``state.symbol`` because a scheduler tracks multiple routes.
    """

    def __init__(self, strategies: Optional[Sequence[Strategy]] = None) -> None:
        self._strategies: Dict[str, Strategy] = {}
        #: strategy_id -> why it produced no candidate on the LAST
        #: ``evaluate_all``. Read by ``BusScheduler._log_arbitration``; see
        #: that method for why a skip and a loss must not look alike.
        self.last_skips: Dict[str, str] = {}
        for strat in strategies or []:
            self.register(strat)

    def register(self, strategy: Strategy) -> None:
        self._strategies[strategy.strategy_id] = strategy

    def get(self, strategy_id: str) -> Optional[Strategy]:
        return self._strategies.get(strategy_id)

    def all(self) -> List[Strategy]:
        return list(self._strategies.values())

    def ids(self) -> List[str]:
        return list(self._strategies.keys())

    def evaluate_all(self, state: Any, ctx: StrategyContext) -> List[Dict[str, Any]]:
        """Fan the tick out across every strategy, and record who never got asked.

        THREE SILENT SKIPS USED TO LIVE HERE, and a strategy caught by any of
        them is indistinguishable from one that competed and lost. That is the
        gap under the operator's question about the decision budget: a
        strategy offered zero cycles needs a scheduler fix, one offered many
        and winning none needs a scoring fix, and the log could not tell them
        apart. ``BusScheduler._log_arbitration`` publishes ``last_skips``
        alongside the candidates that did compete, so both halves land in one
        ``entry-arbitration`` row.

        The skips are not exotic; two of them are structural and one hides
        bugs:

        ``min_samples`` -- and it is NOT uniform. Measured 2026-09-10 across
        the 72 registered strategies: ``atf_static`` needs 4 samples, while
        ``ema_cross``, ``bollinger_squeeze``, ``macd_momentum`` and
        ``donchian_breakout`` need 40, and ``omen_reversion`` needs 60. On a
        symbol whose tick window is short, atf_static is the only base
        strategy that can be evaluated AT ALL -- not because it scored better,
        but because it was the only one eligible to score. Ten times the
        warm-up is ten times the wait for a first candidate.

        ``enabled()`` -- a strategy switched off. All 72 read True on
        2026-09-10, so this is currently empty; it will not stay that way.

        ``except Exception`` -- THE ONE THAT HIDES BUGS. A strategy that
        raises on every tick is skipped forever, silently, and looks exactly
        like a strategy with no signal. Nothing counted these. The exception
        type is now recorded, so a permanently-throwing strategy shows up as
        itself rather than as a quiet zero in the evidence table.

        Behaviour is unchanged: the same strategies are skipped for the same
        reasons and the same candidates come back. Only the bookkeeping is new,
        and it is kept in memory -- a dict of at most one entry per registered
        strategy, overwritten each call -- so it costs no I/O on the tick path.
        """
        candidates: List[Dict[str, Any]] = []
        skips: Dict[str, str] = {}
        n_samples = len(getattr(state, "samples", []) or [])
        for strat in self._strategies.values():
            if n_samples < strat.min_samples:
                skips[strat.strategy_id] = (
                    f"min_samples {n_samples}<{strat.min_samples}"
                )
                continue
            try:
                if not strat.enabled():
                    skips[strat.strategy_id] = "disabled"
                    continue
                cand = strat.evaluate(state, ctx)
            except Exception as exc:  # noqa: BLE001 - one strategy cannot stop the tick
                skips[strat.strategy_id] = f"raised {type(exc).__name__}"
                continue
            if cand:
                candidates.append(cand)
            else:
                skips[strat.strategy_id] = "no_signal"
        self.last_skips = skips
        return candidates
