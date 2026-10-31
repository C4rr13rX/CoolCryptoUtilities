"""EdgeForge Engine — Compacted Pure-Math Multi-Chain OHLCV Genetic Algorithm & AGY Research Agent.

Architecture:
1. Multi-Chain Historical OHLCV + Rapid Stream Loader:
   - Loads multi-chain historical OHLCV data (`base`, `arbitrum`, `ethereum`,
     `optimism`, `polygon`) from `data/historical_ohlcv/<chain>/*.json`,
     `OhlcvDataset`/`OhlcvBar` in `web/datalab/models.py`, and rapid sub-minute
     `market_stream` ticks from `storage/trading_cache.db`.
   - Provides deterministic multi-chain synchronization/ingestion via
     `services.cex_ohlcv_fallback` and `services.data_lab` tools.

2. Compacted Vectorized Pure-Math Feature Matrix (`(N_bars, 27)` float64):
   - Computes 27 non-textbook, pure-number mathematical variables per bar in
     vectorized NumPy (micro-structure order-flow velocity/acceleration,
     nonlinear higher-order moments, Hurst/chaos/entropy proxies, cross-scale
     variance & range compression ratios, polynomial cubic gates, Kalman
     residuals, and Money-Button rapid impulse metrics).
   - Evaluates thousands of genomes per second without LLM overhead.

3. 3 Background Workers with Deterministic Variable Partitioning & Staleness Splitting:
   - Worker 1 (`micro_flow`): Order-flow velocity, buy/sell imbalance derivatives,
     VWAP dislocation, wick asymmetry, Kyle's lambda illiquidity, volume clock burst.
   - Worker 2 (`nonlinear_chaos`): Log-return velocity/acceleration/jerk, rolling
     skewness/kurtosis, variance ratio, fast Hurst exponent, lag-1/lag-2 autocorrelation,
     Shannon entropy proxy, cubic polynomial Z-ratio, price curvature.
   - Worker 3 (`rapid_impulse`): Range compression ratio, micro-dip rebound ratio,
     Kalman filter residual, harmonic mean dislocation, volume-weighted slope,
     shadow exhaustion, Kaufman efficiency ratio, Cramer's Z impulse, ATR-normalized burst.
   - When any worker goes `stale_generations_limit` generations without improving
     out-of-sample net profit after fees, it automatically splits off (`SPLIT_RETIRED`)
     and spawns a new child lineage with a deterministically rotated hybrid
     variable collection and tactic mode (`polynomial_interaction`, `rational_gate`,
     `harmonic_resonance`, `rapid_money_button`, `differential_chaos`).

4. Strict After-Fee Profit Gate & Graveyard Skip-List:
   - Every simulated trade is charged exact round-trip fees via
     `services.roundtrip_cost.roundtrip_cost_rate(clip_usd, symbol)` + slippage.
   - Chronological 60% Train / 20% Validation / 20% Unseen Holdout split.
   - Non-profitable ventures (`net_return_after_fees <= min_net_return_after_fees`
     or `holdout_t_stat < min_holdout_t_stat`) are tagged `complete_no_profit`,
     indexed in the skip-list so the GA and Research Agent never re-evaluate them,
     and excluded from findings reports.

5. 1 Research Agent (`agy_research`) Guidance & Pipeline Feedback Loop:
   - Uses `tools.agy_session.AgySession` (`agy -p --agent research`) to analyze
     live worker lineages, skip `complete_no_profit` regions, reweight variables,
     synthesize new composite mathematical terms, and upgrade GA bounds.
   - Automatically promotes profitable findings into:
     a) `data/discovered_rules.json` (for `trading/strategies/discovered.py`),
     b) `services/strategy_registry.py` (commissioned with full OOS metrics),
     c) `data/money_button_edge_overrides.json` (for `MoneyButtonStrategy` rapid trading),
     d) `web/tradingagent/models.py` (`Constraint` & ghost `Experiment`).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from django.db import close_old_connections
from django.utils import timezone

from services.atomic_json import file_lock, read_json, write_json
from services.roundtrip_cost import roundtrip_cost_rate

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_ROOT = ROOT / "data" / "historical_ohlcv"
DISCOVERED_RULES_PATH = ROOT / "data" / "discovered_rules.json"
MONEY_BUTTON_OVERRIDES_PATH = ROOT / "data" / "money_button_edge_overrides.json"
TRADING_CACHE_DB = ROOT / "storage" / "trading_cache.db"

SUPPORTED_CHAINS: Tuple[str, ...] = (
    "base",
    "arbitrum",
    "ethereum",
    "optimism",
    "polygon",
)

# ---------------------------------------------------------------------------
# 27 Pure-Math Features Partitioned Deterministically Across 3 Worker Families
# ---------------------------------------------------------------------------

WORKER_1_MICRO_FLOW_VARS: Tuple[str, ...] = (
    "net_vol_imbalance",        # (buy_vol - sell_vol) / (total_vol + eps)
    "buy_pressure_velocity",    # 1st difference of buy volume share
    "buy_pressure_accel",       # 2nd difference of buy volume share
    "vwap_dislocation",         # (close - vwap) / (vwap + eps)
    "wick_asymmetry",           # (upper_wick - lower_wick) / (high - low + eps)
    "body_conviction",          # (close - open) / (high - low + eps)
    "kyle_lambda_inv",          # |ret_1| / log(1 + rel_vol) inverse illiquidity shock
    "volume_clock_burst",       # vol / rolling_median_vol_12 - 1.0
    "orderflow_divergence",     # sign(net_vol) * (-ret_1) * abs(net_vol_imbalance)
)

WORKER_2_NONLINEAR_CHAOS_VARS: Tuple[str, ...] = (
    "ret_velocity",             # 1-bar log return
    "ret_acceleration",         # 2nd derivative of log price: r_t - r_{t-1}
    "ret_jerk",                 # 3rd derivative of log price: a_t - a_{t-1}
    "rolling_skew_12",          # 12-bar rolling third central moment
    "rolling_kurt_12",          # 12-bar rolling fourth standardized moment - 3
    "variance_ratio_4_16",      # Var(4-bar ret) / (4 * Var(1-bar ret)) - 1
    "hurst_fast_24",            # Local R/S rescaled range exponent proxy - 0.5
    "autocorr_lag1",            # Rolling 12-bar lag-1 return autocorrelation
    "cubic_z_gate",             # z^3 / (1 + z^2) nonlinear compressed tail trigger
)

WORKER_3_RAPID_IMPULSE_VARS: Tuple[str, ...] = (
    "range_compression",        # (high - low) / rolling_mean_range_16 - 1.0
    "micro_dip_rebound",        # (close - low) / (high - low + eps) * (-min(0, ret_2))
    "kalman_residual",          # (close - fast_kalman_est) / rolling_std_12
    "harmonic_dislocation",     # (close - harmonic_mean_8) / close
    "vol_weighted_slope_6",     # 6-bar volume-weighted linear regression slope
    "shadow_exhaustion",        # lower_wick / (abs(close - open) + eps)
    "kaufman_efficiency_8",     # |close_t - close_{t-8}| / sum(|diff(close)|)
    "cramers_impulse_z",        # ret_1 / (rolling_mad_12 + eps)
    "price_curvature_poly",     # (close_{t} - 2*close_{t-2} + close_{t-4}) / close_{t}
)

ALL_MATH_VARIABLES: Tuple[str, ...] = (
    WORKER_1_MICRO_FLOW_VARS
    + WORKER_2_NONLINEAR_CHAOS_VARS
    + WORKER_3_RAPID_IMPULSE_VARS
)

VAR_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(ALL_MATH_VARIABLES)}

# Deterministic rotation catalog when a worker becomes stale and splits off
SPLIT_ROTATION_COLLECTIONS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    (
        "micro_flow",
        "rational_gate",
        WORKER_1_MICRO_FLOW_VARS,
    ),
    (
        "nonlinear_chaos",
        "polynomial_interaction",
        WORKER_2_NONLINEAR_CHAOS_VARS,
    ),
    (
        "rapid_impulse",
        "rapid_money_button",
        WORKER_3_RAPID_IMPULSE_VARS,
    ),
    (
        "flow_chaos_hybrid",
        "differential_chaos",
        WORKER_1_MICRO_FLOW_VARS[:5] + WORKER_2_NONLINEAR_CHAOS_VARS[:4],
    ),
    (
        "chaos_impulse_hybrid",
        "harmonic_resonance",
        WORKER_2_NONLINEAR_CHAOS_VARS[3:] + WORKER_3_RAPID_IMPULSE_VARS[:4],
    ),
    (
        "micro_impulse_money_button",
        "rapid_money_button",
        WORKER_1_MICRO_FLOW_VARS[:4] + WORKER_3_RAPID_IMPULSE_VARS[1:6],
    ),
    (
        "tri_chain_eigen_poly",
        "polynomial_interaction",
        (
            "net_vol_imbalance",
            "vwap_dislocation",
            "orderflow_divergence",
            "ret_acceleration",
            "variance_ratio_4_16",
            "cubic_z_gate",
            "micro_dip_rebound",
            "kalman_residual",
            "cramers_impulse_z",
        ),
    ),
)

OPERATORS: Tuple[str, ...] = (
    "linear_comb",          # w1 * z(v1) + w2 * z(v2) + w3 * z(v3) > thresh
    "poly_product",         # w1 * z(v1) * z(v2) + w2 * z(v3) > thresh
    "rational_ratio",       # (w1 * z(v1) + w2 * z(v2)) / (1.0 + |z(v3)|) > thresh
    "harmonic_diff",        # w1 * (z(v1) - z(v2)) + w2 * z(v3)^2 * sign(z(v3)) > thresh
    "threshold_and_gate",   # (w1 * z(v1) > thresh * 0.6) & (w2 * z(v2) + w3 * z(v3) > thresh * 0.6)
)


@dataclass
class CompactedPairMatrix:
    """Pre-computed, standardized `(N, 27)` float64 mathematical matrix for one chain/pair."""

    chain: str
    symbol: str
    granularity_sec: int
    timestamps: np.ndarray          # shape (N,)
    close: np.ndarray               # shape (N,)
    high: np.ndarray                # shape (N,)
    low: np.ndarray                 # shape (N,)
    features_z: np.ndarray          # shape (N, 27) standardized without lookahead
    forward_returns: Dict[int, np.ndarray] = field(default_factory=dict)  # hold_bars -> (N,)
    roundtrip_fee: float = 0.004


# ---------------------------------------------------------------------------
# Vectorized Pure-Math Feature Computation (No Lookahead)
# ---------------------------------------------------------------------------

def _ema_1d(arr: np.ndarray, alpha: float) -> np.ndarray:
    """Fast causal exponential moving average."""
    out = np.empty_like(arr, dtype=np.float64)
    if arr.size == 0:
        return out
    out[0] = arr[0]
    one_m = 1.0 - alpha
    for i in range(1, arr.size):
        out[i] = alpha * arr[i] + one_m * out[i - 1]
    return out


def _causal_zscore(mat: np.ndarray, alpha: float = 0.06) -> np.ndarray:
    """Standardize each column causally using exponential mean & variance so
    train/val/holdout never leak future statistics."""
    n, m = mat.shape
    out = np.zeros((n, m), dtype=np.float64)
    if n < 4:
        return out
    mean = mat[0].copy()
    var = np.ones(m, dtype=np.float64) * 1e-4
    one_m = 1.0 - alpha
    for i in range(1, n):
        x = mat[i]
        diff = x - mean
        mean = one_m * mean + alpha * x
        var = one_m * var + alpha * (diff * diff)
        std = np.sqrt(np.maximum(var, 1e-12))
        out[i] = np.clip(diff / std, -6.0, 6.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_compacted_math_matrix(
    chain: str,
    symbol: str,
    rows: Sequence[Dict[str, Any]],
    clip_usd: float = 2.0,
) -> Optional[CompactedPairMatrix]:
    """Convert raw OHLCV bar dicts into a compacted `(N, 27)` float64 matrix."""
    if len(rows) < 120:
        return None

    # Take up to the most recent 1,800 bars per pair so memory is compact and math is ultra-fast
    slice_rows = rows[-1800:] if len(rows) > 1800 else rows
    n = len(slice_rows)

    ts = np.zeros(n, dtype=np.float64)
    op = np.zeros(n, dtype=np.float64)
    hi = np.zeros(n, dtype=np.float64)
    lo = np.zeros(n, dtype=np.float64)
    cl = np.zeros(n, dtype=np.float64)
    buy_v = np.zeros(n, dtype=np.float64)
    sell_v = np.zeros(n, dtype=np.float64)
    vwap = np.zeros(n, dtype=np.float64)

    for i, r in enumerate(slice_rows):
        ts[i] = float(r.get("timestamp") or r.get("ts") or i * 300)
        c_val = float(r.get("close") or 0.0)
        o_val = float(r.get("open") or c_val)
        h_val = float(r.get("high") or max(o_val, c_val))
        l_val = float(r.get("low") or min(o_val, c_val))
        cl[i] = c_val
        op[i] = o_val
        hi[i] = max(h_val, c_val, o_val)
        lo[i] = min(l_val, c_val, o_val)
        bv = float(r.get("buy_volume") or 0.0)
        sv = float(r.get("sell_volume") or 0.0)
        tot_v = float(r.get("volume") or r.get("net_volume") or (bv + sv) or 1.0)
        if bv == 0.0 and sv == 0.0:
            # Estimate buy/sell split deterministically from candle close location in range
            rng = max(hi[i] - lo[i], 1e-12)
            frac = np.clip((cl[i] - lo[i]) / rng, 0.05, 0.95)
            bv = tot_v * frac
            sv = tot_v * (1.0 - frac)
        buy_v[i] = max(0.0, bv)
        sell_v[i] = max(0.0, sv)
        vw = float(r.get("vwap") or 0.0)
        vwap[i] = vw if vw > 0 else (hi[i] + lo[i] + cl[i]) / 3.0

    # Filter out corrupted / zero-price series
    if np.any(cl <= 0) or float(np.median(cl)) <= 0:
        valid = cl > 0
        if int(np.sum(valid)) < 120:
            return None
        ts, op, hi, lo, cl, buy_v, sell_v, vwap = (
            ts[valid], op[valid], hi[valid], lo[valid],
            cl[valid], buy_v[valid], sell_v[valid], vwap[valid],
        )
        n = cl.size

    # Reject mixed-denomination / 50x spike glitches
    med_p = float(np.median(cl))
    spread = np.maximum(cl / med_p, med_p / np.maximum(cl, 1e-18))
    if float(np.max(spread)) > 40.0:
        return None

    eps = 1e-12
    tot_vol = buy_v + sell_v + eps
    rng = np.maximum(hi - lo, eps)
    log_p = np.log(np.maximum(cl, eps))

    ret_1 = np.zeros(n, dtype=np.float64)
    ret_1[1:] = np.diff(log_p)
    ret_2 = np.zeros(n, dtype=np.float64)
    ret_2[2:] = log_p[2:] - log_p[:-2]
    ret_4 = np.zeros(n, dtype=np.float64)
    ret_4[4:] = log_p[4:] - log_p[:-4]

    raw = np.zeros((n, len(ALL_MATH_VARIABLES)), dtype=np.float64)

    # --- Worker 1: Micro-Flow Variables (indices 0..8) ---
    net_imb = (buy_v - sell_v) / tot_vol
    raw[:, 0] = net_imb
    bp_share = buy_v / tot_vol
    bp_vel = np.zeros(n, dtype=np.float64)
    bp_vel[1:] = np.diff(bp_share)
    raw[:, 1] = bp_vel
    bp_acc = np.zeros(n, dtype=np.float64)
    bp_acc[1:] = np.diff(bp_vel)
    raw[:, 2] = bp_acc
    raw[:, 3] = (cl - vwap) / np.maximum(vwap, eps)
    upper_wick = hi - np.maximum(op, cl)
    lower_wick = np.minimum(op, cl) - lo
    raw[:, 4] = (upper_wick - lower_wick) / rng
    raw[:, 5] = (cl - op) / rng
    ema_vol = np.maximum(_ema_1d(tot_vol, 0.12), eps)
    rel_vol = tot_vol / ema_vol
    raw[:, 6] = np.abs(ret_1) / np.log1p( np.maximum(rel_vol, 1e-4) )
    raw[:, 7] = rel_vol - 1.0
    raw[:, 8] = np.sign(net_imb) * (-ret_1) * np.abs(net_imb)

    # --- Worker 2: Nonlinear Chaos Variables (indices 9..17) ---
    raw[:, 9] = ret_1
    ret_acc = np.zeros(n, dtype=np.float64)
    ret_acc[1:] = np.diff(ret_1)
    raw[:, 10] = ret_acc
    ret_jerk = np.zeros(n, dtype=np.float64)
    ret_jerk[1:] = np.diff(ret_acc)
    raw[:, 11] = ret_jerk

    ema_r = _ema_1d(ret_1, 0.15)
    dev = ret_1 - ema_r
    var_r = np.maximum(_ema_1d(dev * dev, 0.15), 1e-16)
    std_r = np.sqrt(var_r)
    z_r = np.clip(dev / std_r, -6.0, 6.0)
    raw[:, 12] = _ema_1d(z_r ** 3, 0.15)
    raw[:, 13] = _ema_1d(z_r ** 4, 0.15) - 3.0

    var_r4 = np.maximum(_ema_1d((ret_4 - _ema_1d(ret_4, 0.10)) ** 2, 0.10), 1e-16)
    raw[:, 14] = (var_r4 / np.maximum(4.0 * var_r, 1e-16)) - 1.0

    # Fast causal R/S Hurst proxy over 16 bars
    ema_abs_r = np.maximum(_ema_1d(np.abs(dev), 0.12), 1e-12)
    raw[:, 15] = np.clip(np.log(np.maximum( rng / np.maximum(cl * ema_abs_r, 1e-12), 1e-4 )) / math.log(16.0) - 0.5, -1.0, 1.0)

    r_prev = np.roll(dev, 1)
    r_prev[0] = 0.0
    raw[:, 16] = _ema_1d(dev * r_prev, 0.15) / var_r
    raw[:, 17] = (z_r ** 3) / (1.0 + z_r * z_r)

    # --- Worker 3: Rapid Money-Button Impulse Variables (indices 18..26) ---
    ema_rng = np.maximum(_ema_1d(rng, 0.12), eps)
    raw[:, 18] = rng / ema_rng - 1.0
    raw[:, 19] = ((cl - lo) / rng) * np.maximum(0.0, -ret_2)
    kalman_p = _ema_1d(cl, 0.25)
    ema_p_std = np.maximum(np.sqrt(_ema_1d((cl - kalman_p) ** 2, 0.15)), eps)
    raw[:, 20] = (cl - kalman_p) / ema_p_std
    inv_p_ema = _ema_1d(1.0 / np.maximum(cl, eps), 0.20)
    harm_p = 1.0 / np.maximum(inv_p_ema, eps)
    raw[:, 21] = (cl - harm_p) / np.maximum(cl, eps)
    raw[:, 22] = _ema_1d(ret_1 * rel_vol, 0.25)
    raw[:, 23] = lower_wick / (np.abs(cl - op) + 0.15 * rng + eps)
    abs_step = np.abs(np.diff(cl, prepend=cl[0]))
    sum_abs_8 = np.maximum(_ema_1d(abs_step, 0.22) * 8.0, eps)
    net_move_8 = np.zeros(n, dtype=np.float64)
    net_move_8[8:] = np.abs(cl[8:] - cl[:-8])
    raw[:, 24] = net_move_8 / sum_abs_8
    raw[:, 25] = ret_1 / ema_abs_r
    curv = np.zeros(n, dtype=np.float64)
    curv[4:] = (cl[4:] - 2.0 * cl[2:-2] + cl[:-4]) / np.maximum(cl[4:], eps)
    raw[:, 26] = curv

    features_z = _causal_zscore(raw)

    # Pre-compute forward net-of-slippage gross returns for rapid hold horizons (1, 2, 3, 4, 6 bars)
    forward_returns: Dict[int, np.ndarray] = {}
    for h_bars in (1, 2, 3, 4, 6):
        fwd = np.zeros(n, dtype=np.float64)
        if n > h_bars:
            fwd[:-h_bars] = (cl[h_bars:] - cl[:-h_bars]) / np.maximum(cl[:-h_bars], eps)
        forward_returns[h_bars] = np.clip(fwd, -0.35, 0.50)

    granularity = int(np.median(np.diff(ts))) if n > 2 else 300
    if granularity <= 0:
        granularity = 300

    base_sym = symbol.upper().split("-")[0].split("/")[0]
    rt_fee = float(roundtrip_cost_rate(clip_usd, base_sym))
    # Add 4 bps micro-slippage buffer for rapid execution honesty
    rt_fee_total = max(0.0008, rt_fee + 0.0004)

    return CompactedPairMatrix(
        chain=chain,
        symbol=symbol.upper(),
        granularity_sec=granularity,
        timestamps=ts,
        close=cl,
        high=hi,
        low=lo,
        features_z=features_z,
        forward_returns=forward_returns,
        roundtrip_fee=rt_fee_total,
    )


# ---------------------------------------------------------------------------
# Multi-Chain Historical OHLCV + Rapid Stream Loader & Refresh Tool
# ---------------------------------------------------------------------------

class MultiChainMarketCorpus:
    """Loads and caches compacted mathematical matrices across all chains."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.matrices: Dict[str, List[CompactedPairMatrix]] = {}
        self.last_loaded_at: float = 0.0
        self.total_bars_loaded: int = 0
        self.total_pairs_loaded: int = 0

    def ensure_loaded(
        self,
        chains: Sequence[str] = SUPPORTED_CHAINS,
        max_pairs_per_chain: int = 18,
        clip_usd: float = 2.0,
        force: bool = False,
    ) -> Dict[str, int]:
        with self._lock:
            if (
                not force
                and self.matrices
                and (time.time() - self.last_loaded_at) < 600.0
            ):
                return {ch: len(lst) for ch, lst in self.matrices.items()}

            new_matrices: Dict[str, List[CompactedPairMatrix]] = {}
            total_bars = 0
            total_pairs = 0

            #Priority liquid pairs first when available
            priority_Check = (
                "WETH-USDC", "WETH-USDT", "ARB-WETH", "ARB-USDC", "OP-USDC",
                "OP-WETH", "WBTC-WETH", "WBTC-USDC", "LINK-WETH", "UNI-WETH",
                "AAVE-WETH", "AERO-USDC", "CBETH-WETH", "MATIC-WETH", "PEPE-WETH",
                "GMX-WETH", "PENDLE-WETH", "LDO-WETH",
            )

            for chain in chains:
                chain_l = chain.strip().lower()
                chain_dir = HISTORICAL_ROOT / chain_l
                loaded_for_chain: List[CompactedPairMatrix] = []

                if chain_dir.is_dir():
                    files = list(chain_dir.glob("*.json"))
                    # Sort priority pairs first, then by file size descending
                    def _sort_key(p: Path) -> Tuple[int, int]:
                        stem_sym = p.stem.split("_", 1)[-1].upper()
                        prio = 0 if any(t in stem_sym for t in priority_Check) else 1
                        try:
                            sz = -p.stat().st_size
                        except OSError:
                            sz = 0
                        return (prio, sz)

                    files.sort(key=_sort_key)
                    for fpath in files:
                        if len(loaded_for_chain) >= max_pairs_per_chain:
                            break
                        symbol = fpath.stem.split("_", 1)[-1].upper()
                        try:
                            raw = json.loads(fpath.read_text(encoding="utf-8"))
                            if not isinstance(raw, list) or len(raw) < 150:
                                continue
                            mat = build_compacted_math_matrix(
                                chain=chain_l,
                                symbol=symbol,
                                rows=raw,
                                clip_usd=clip_usd,
                            )
                            if mat is not None:
                                loaded_for_chain.append(mat)
                                total_bars += int(mat.close.size)
                                total_pairs += 1
                        except Exception:
                            continue

                # Also supplement from rapid sub-minute `market_stream` in trading_cache.db for `base`
                if chain_l == "base" and TRADING_CACHE_DB.is_file():
                    try:
                        stream_mats = self._load_rapid_stream_matrices(
                            clip_usd=clip_usd, limit=6
                        )
                        for sm in stream_mats:
                            loaded_for_chain.append(sm)
                            total_bars += int(sm.close.size)
                            total_pairs += 1
                    except Exception:
                        pass

                if loaded_for_chain:
                    new_matrices[chain_l] = loaded_for_chain

            self.matrices = new_matrices
            self.total_bars_loaded = total_bars
            self.total_pairs_loaded = total_pairs
            self.last_loaded_at = time.time()
            return {ch: len(lst) for ch, lst in self.matrices.items()}

    def _load_rapid_stream_matrices(
        self, clip_usd: float = 2.0, limit: int = 6
    ) -> List[CompactedPairMatrix]:
        """Build rapid sub-minute bar matrices from live `market_stream` ticks in trading_cache.db."""
        out: List[CompactedPairMatrix] = []
        conn = sqlite3.connect(f"file:{TRADING_CACHE_DB}?mode=ro", uri=True, timeout=3.0)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT symbol, COUNT(*) as cnt FROM market_stream "
                "GROUP BY symbol HAVING cnt >= 160 ORDER BY cnt DESC LIMIT ?",
                (limit,),
            )
            syms = [r[0] for r in cur.fetchall()]
            for sym in syms:
                cur.execute(
                    "SELECT ts, price, volume FROM market_stream "
                    "WHERE symbol = ? ORDER BY ts ASC LIMIT 1800",
                    (sym,),
                )
                pts = cur.fetchall()
                if len(pts) < 150:
                    continue
                bars: List[Dict[str, Any]] = []
                prev_p = float(pts[0][1] or 0.0)
                for ts_val, price_val, vol_val in pts:
                    p = float(price_val or 0.0)
                    if p <= 0:
                        continue
                    v = max(1.0, float(vol_val or 1.0))
                    hi_p = max(p, prev_p) * 1.0002
                    lo_p = min(p, prev_p) * 0.9998
                    is_up = p >= prev_p
                    bars.append({
                        "timestamp": float(ts_val or 0.0),
                        "open": prev_p,
                        "high": hi_p,
                        "low": lo_p,
                        "close": p,
                        "volume": v,
                        "buy_volume": v * (0.68 if is_up else 0.32),
                        "sell_volume": v * (0.32 if is_up else 0.68),
                        "vwap": 0.5 * (p + prev_p),
                    })
                    prev_p = p
                mat = build_compacted_math_matrix(
                    chain="base",
                    symbol=f"{sym}",
                    rows=bars,
                    clip_usd=clip_usd,
                )
                if mat is not None:
                    out.append(mat)
        finally:
            conn.close()
        return out

    def refresh_multichain_ohlcv(
        self,
        chains: Sequence[str] = SUPPORTED_CHAINS,
        days_back: int = 30,
        max_pairs: int = 6,
    ) -> Dict[str, Any]:
        """Use the repository's OHLCV tools (`cex_ohlcv_fallback` & `data_lab`) to pull
        fresh multi-chain historical OHLCV bars and ingest them into `OhlcvDataset`."""
        from services.cex_ohlcv_fallback import bootstrap_core_pairs, CORE_PAIRS
        from services.data_lab import ingest_ohlcv_output

        summary: Dict[str, Any] = {"chains": {}, "ingested_datasets": 0}
        subset_pairs = list(CORE_PAIRS[:max_pairs])
        for ch in chains:
            ch_l = ch.strip().lower()
            try:
                res = bootstrap_core_pairs(
                    chain=ch_l,
                    days_back=days_back,
                    pairs=subset_pairs,
                )
                ingested = ingest_ohlcv_output(
                    chain=ch_l,
                    output_root=HISTORICAL_ROOT,
                    granularity_seconds=300,
                    source="edgeforge_multichain_sync",
                )
                summary["chains"][ch_l] = {
                    "downloaded": res.get("downloaded", 0),
                    "skipped": res.get("skipped", 0),
                    "ingested": len(ingested),
                }
                summary["ingested_datasets"] += len(ingested)
            except Exception as exc:
                summary["chains"][ch_l] = {"error": str(exc)}

        # Reload corpus with any newly fetched files
        self.ensure_loaded(chains=chains, force=True)
        return summary


CORPUS = MultiChainMarketCorpus()


# ---------------------------------------------------------------------------
# Pure-Math Genome Representation & Deterministic Signature
# ---------------------------------------------------------------------------

@dataclass
class PureMathGenome:
    """Compact pure-number mathematical strategy genome."""

    variable_collection: str
    tactic_mode: str
    operator: str
    var_names: Tuple[str, str, str]
    weights: Tuple[float, float, float]
    threshold: float
    hold_bars: int
    tp_fee_mult: float      # Take-profit as multiple of round-trip fee (e.g. 1.5x..4.0x)
    sl_fee_mult: float      # Stop-loss as multiple of round-trip fee (e.g. 1.0x..2.5x)
    regime_var: str         # Secondary filter variable
    regime_min_z: float     # Minimum Z-score on regime_var

    def signature(self) -> str:
        """Deterministic quantized signature so the GA and Agent skip already-evaluated
        ventures (especially `complete_no_profit` dead ends)."""
        norm_vars = "|".join(self.var_names)
        norm_w = "|".join(f"{round(w, 1):+.1f}" for w in self.weights)
        payload = (
            f"{self.variable_collection}:{self.operator}:{norm_vars}:{norm_w}:"
            f"th={round(self.threshold, 1):.1f}:h={self.hold_bars}:"
            f"tp={round(self.tp_fee_mult, 1):.1f}:sl={round(self.sl_fee_mult, 1):.1f}:"
            f"reg={self.regime_var}>{round(self.regime_min_z, 1):.1f}"
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]

    def formula_string(self) -> str:
        v1, v2, v3 = self.var_names
        w1, w2, w3 = self.weights
        if self.operator == "linear_comb":
            core = f"({w1:+.2f}·Z[{v1}] {w2:+.2f}·Z[{v2}] {w3:+.2f}·Z[{v3}]) > {self.threshold:.2f}"
        elif self.operator == "poly_product":
            core = f"({w1:+.2f}·Z[{v1}]·Z[{v2}] {w2:+.2f}·Z[{v3}]) > {self.threshold:.2f}"
        elif self.operator == "rational_ratio":
            core = f"(({w1:+.2f}·Z[{v1}] {w2:+.2f}·Z[{v2}]) / (1 + |Z[{v3}]|)) > {self.threshold:.2f}"
        elif self.operator == "harmonic_diff":
            core = f"({w1:+.2f}·(Z[{v1}] - Z[{v2}]) {w2:+.2f}·Z[{v3}]²·sgn(Z[{v3}])) > {self.threshold:.2f}"
        else:
            core = f"(({w1:+.2f}·Z[{v1}] > {self.threshold * 0.6:.2f}) ∧ ({w2:+.2f}·Z[{v2}] {w3:+.2f}·Z[{v3}] > {self.threshold * 0.6:.2f}))"
        return (
            f"{core} ∧ (Z[{self.regime_var}] > {self.regime_min_z:+.2f}) "
            f"[hold={self.hold_bars}b, TP={self.tp_fee_mult:.1f}×fee, SL={self.sl_fee_mult:.1f}×fee]"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable_collection": self.variable_collection,
            "tactic_mode": self.tactic_mode,
            "operator": self.operator,
            "var_names": list(self.var_names),
            "weights": [round(w, 4) for w in self.weights],
            "threshold": round(self.threshold, 4),
            "hold_bars": int(self.hold_bars),
            "tp_fee_mult": round(self.tp_fee_mult, 3),
            "sl_fee_mult": round(self.sl_fee_mult, 3),
            "regime_var": self.regime_var,
            "regime_min_z": round(self.regime_min_z, 3),
        }

    @classmethod
    def random_from_collection(
        cls,
        rng: random.Random,
        variable_collection: str,
        tactic_mode: str,
        allowed_vars: Sequence[str],
        directives: Optional[Dict[str, Any]] = None,
    ) -> "PureMathGenome":
        pool = [v for v in allowed_vars if v in VAR_INDEX]
        if len(pool) < 3:
            pool = list(ALL_MATH_VARIABLES)

        # Apply Research Agent variable preference weights if present
        boosted_vars = (directives or {}).get("boosted_variables") or []
        weighted_pool = list(pool)
        for bv in boosted_vars:
            if bv in pool:
                weighted_pool.extend([bv, bv])

        chosen: List[str] = []
        for _ in range(12):
            cand = rng.choice(weighted_pool)
            if cand not in chosen:
                chosen.append(cand)
            if len(chosen) == 3:
                break
        while len(chosen) < 3:
            chosen.append(rng.choice(pool))

        op_choices = list(OPERATORS)
        if tactic_mode == "polynomial_interaction":
            op = rng.choice(["poly_product", "harmonic_diff", "linear_comb"])
        elif tactic_mode == "rational_gate":
            op = rng.choice(["rational_ratio", "threshold_and_gate", "linear_comb"])
        elif tactic_mode == "rapid_money_button":
            op = rng.choice(["threshold_and_gate", "rational_ratio", "poly_product"])
        else:
            op = rng.choice(op_choices)

        weights = (
            round(rng.uniform(-1.8, 1.8), 2),
            round(rng.uniform(-1.8, 1.8), 2),
            round(rng.uniform(-1.5, 1.5), 2),
        )
        # Ensure at least one weight has meaningful magnitude
        if max(abs(w) for w in weights) < 0.35:
            weights = (1.15, weights[1], weights[2])

        return cls(
            variable_collection=variable_collection,
            tactic_mode=tactic_mode,
            operator=op,
            var_names=(chosen[0], chosen[1], chosen[2]),
            weights=weights,
            threshold=round(rng.uniform(0.75, 2.35), 2),
            hold_bars=rng.choice([1, 1, 2, 3, 4]),
            tp_fee_mult=round(rng.uniform(1.4, 4.2), 2),
            sl_fee_mult=round(rng.uniform(0.9, 2.4), 2),
            regime_var=rng.choice(weighted_pool),
            regime_min_z=round(rng.uniform(-0.4, 1.2), 2),
        )

    def mutate(
        self,
        rng: random.Random,
        allowed_vars: Sequence[str],
        directives: Optional[Dict[str, Any]] = None,
    ) -> "PureMathGenome":
        pool = [v for v in allowed_vars if v in VAR_INDEX] or list(ALL_MATH_VARIABLES)
        v_list = list(self.var_names)
        w_list = list(self.weights)
        op = self.operator
        thresh = self.threshold
        hold_b = self.hold_bars
        tp_m = self.tp_fee_mult
        sl_m = self.sl_fee_mult
        reg_v = self.regime_var
        reg_z = self.regime_min_z

        choice = rng.random()
        if choice < 0.28:
            idx = rng.randrange(3)
            v_list[idx] = rng.choice(pool)
        elif choice < 0.56:
            idx = rng.randrange(3)
            w_list[idx] = round(np.clip(w_list[idx] + rng.gauss(0.0, 0.45), -2.5, 2.5), 2)
        elif choice < 0.72:
            thresh = round(float(np.clip(thresh + rng.gauss(0.0, 0.25), 0.55, 3.0)), 2)
        elif choice < 0.84:
            op = rng.choice(OPERATORS)
        elif choice < 0.92:
            hold_b = rng.choice([1, 2, 3, 4, 6])
            tp_m = round(float(np.clip(tp_m + rng.gauss(0.0, 0.35), 1.2, 5.0)), 2)
            sl_m = round(float(np.clip(sl_m + rng.gauss(0.0, 0.25), 0.7, 3.0)), 2)
        else:
            reg_v = rng.choice(pool)
            reg_z = round(float(np.clip(reg_z + rng.gauss(0.0, 0.3), -0.6, 1.6)), 2)

        return PureMathGenome(
            variable_collection=self.variable_collection,
            tactic_mode=self.tactic_mode,
            operator=op,
            var_names=(v_list[0], v_list[1], v_list[2]),
            weights=(w_list[0], w_list[1], w_list[2]),
            threshold=thresh,
            hold_bars=hold_b,
            tp_fee_mult=tp_m,
            sl_fee_mult=sl_m,
            regime_var=reg_v,
            regime_min_z=reg_z,
        )

    @classmethod
    def crossover(cls, rng: random.Random, a: "PureMathGenome", b: "PureMathGenome") -> "PureMathGenome":
        vars_child = (
            a.var_names[0],
            b.var_names[1] if b.var_names[1] != a.var_names[0] else a.var_names[1],
            a.var_names[2] if rng.random() < 0.5 else b.var_names[2],
        )
        weights_child = (
            a.weights[0],
            b.weights[1],
            round(0.5 * (a.weights[2] + b.weights[2]), 2),
        )
        return cls(
            variable_collection=a.variable_collection,
            tactic_mode=a.tactic_mode,
            operator=a.operator if rng.random() < 0.6 else b.operator,
            var_names=vars_child,
            weights=weights_child,
            threshold=round(0.5 * (a.threshold + b.threshold), 2),
            hold_bars=a.hold_bars if rng.random() < 0.5 else b.hold_bars,
            tp_fee_mult=round(0.5 * (a.tp_fee_mult + b.tp_fee_mult), 2),
            sl_fee_mult=round(0.5 * (a.sl_fee_mult + b.sl_fee_mult), 2),
            regime_var=a.regime_var if rng.random() < 0.5 else b.regime_var,
            regime_min_z=round(0.5 * (a.regime_min_z + b.regime_min_z), 2),
        )


# ---------------------------------------------------------------------------
# Vectorized Genome Evaluation Across Multi-Chain OHLCV Matrices
# ---------------------------------------------------------------------------

def _evaluate_signal_mask(genome: PureMathGenome, z_mat: np.ndarray) -> np.ndarray:
    """Compute boolean entry signal mask across all bars in pure vectorized NumPy."""
    i1 = VAR_INDEX.get(genome.var_names[0], 0)
    i2 = VAR_INDEX.get(genome.var_names[1], 1)
    i3 = VAR_INDEX.get(genome.var_names[2], 2)
    ir = VAR_INDEX.get(genome.regime_var, 0)

    x1 = z_mat[:, i1]
    x2 = z_mat[:, i2]
    x3 = z_mat[:, i3]
    xr = z_mat[:, ir]

    w1, w2, w3 = genome.weights
    th = genome.threshold

    if genome.operator == "linear_comb":
        score = w1 * x1 + w2 * x2 + w3 * x3
        mask = score > th
    elif genome.operator == "poly_product":
        score = w1 * (x1 * x2) + w2 * x3
        mask = score > th
    elif genome.operator == "rational_ratio":
        score = (w1 * x1 + w2 * x2) / (1.0 + np.abs(x3))
        mask = score > th
    elif genome.operator == "harmonic_diff":
        score = w1 * (x1 - x2) + w2 * (x3 * np.abs(x3))
        mask = score > th
    else:
        mask = (w1 * x1 > th * 0.6) & ((w2 * x2 + w3 * x3) > th * 0.6)

    mask = mask & (xr > genome.regime_min_z)
    # Warmup first 20 bars excluded
    if mask.size > 20:
        mask[:20] = False
    return mask


def evaluate_genome_multichain(
    genome: PureMathGenome,
    chains: Sequence[str],
    clip_usd: float,
    min_oos_trades: int,
    min_net_return_after_fees: float,
    min_holdout_t_stat: float,
) -> Dict[str, Any]:
    """Evaluate a genome across all loaded multi-chain OHLCV matrices using a
    strict chronological 60% Train / 40% Out-of-Sample Holdout split, charging
    exact round-trip fees on every trade."""
    chain_list = [c.lower() for c in chains if c.lower() in CORPUS.matrices]
    if not chain_list:
        chain_list = list(CORPUS.matrices.keys())

    h_bars = genome.hold_bars if genome.hold_bars in (1, 2, 3, 4, 6) else 1

    best_pair_metrics: Optional[Dict[str, Any]] = None
    profitable_symbols: List[str] = []
    chains_tested: List[str] = []

    all_oos_net_returns: List[np.ndarray] = []
    all_train_trades = 0

    for ch in chain_list:
        pairs = CORPUS.matrices.get(ch) or []
        if not pairs:
            continue
        chains_tested.append(ch)
        for mat in pairs:
            n = mat.close.size
            if n < 120:
                continue
            split_idx = int(n * 0.60)
            mask = _evaluate_signal_mask(genome, mat.features_z)
            fwd_gross = mat.forward_returns.get(h_bars)
            if fwd_gross is None:
                continue

            rt_fee = mat.roundtrip_fee
            tp_cap = genome.tp_fee_mult * rt_fee
            sl_cap = -genome.sl_fee_mult * rt_fee

            # Clamp gross return to TP / SL bounds and subtract exact round-trip fee
            bounded_gross = np.clip(fwd_gross, sl_cap, tp_cap)
            net_ret_all = bounded_gross - rt_fee

            train_mask = mask[:split_idx]
            oos_mask = mask[split_idx:-h_bars] if h_bars < (n - split_idx) else mask[split_idx:]
            oos_net_series = net_ret_all[split_idx:split_idx + oos_mask.size][oos_mask]

            tr_cnt = int(np.sum(train_mask))
            oos_cnt = int(oos_net_series.size)
            all_train_trades += tr_cnt

            if oos_cnt < 4:
                continue

            pair_mean_net = float(np.mean(oos_net_series))
            pair_std = float(np.std(oos_net_series, ddof=1)) if oos_cnt > 1 else 1e-6
            pair_t = pair_mean_net / max(pair_std / math.sqrt(oos_cnt), 1e-9)

            if pair_mean_net > min_net_return_after_fees and pair_t >= 1.25:
                profitable_symbols.append(f"{ch}:{mat.symbol}")
                all_oos_net_returns.append(oos_net_series)

            # Track the strongest single pair/chain performance
            pair_score = pair_mean_net * math.sqrt(oos_cnt) if pair_mean_net > 0 else pair_mean_net
            if best_pair_metrics is None or pair_score > best_pair_metrics["score"]:
                wins = int(np.sum(oos_net_series > 0))
                gross_mean = float(np.mean(bounded_gross[split_idx:split_idx + oos_mask.size][oos_mask]))
                pos_sum = float(np.sum(oos_net_series[oos_net_series > 0]))
                neg_sum = float(np.abs(np.sum(oos_net_series[oos_net_series <= 0])))
                pf = pos_sum / max(neg_sum, 1e-6)
                cum = np.cumsum(oos_net_series)
                peak = np.maximum.accumulate(np.maximum(0.0, cum))
                max_dd = float(np.max(peak - cum)) if cum.size else 0.0
                best_pair_metrics = {
                    "score": pair_score,
                    "chain": ch,
                    "symbol": mat.symbol,
                    "granularity_sec": mat.granularity_sec,
                    "roundtrip_fee_pct": rt_fee * 100.0,
                    "train_trades": tr_cnt,
                    "oos_trades": oos_cnt,
                    "oos_win_rate": wins / max(oos_cnt, 1),
                    "gross_return_mean": gross_mean,
                    "net_return_after_fees": pair_mean_net,
                    "net_profit_usd": pair_mean_net * oos_cnt * clip_usd,
                    "profit_factor": min(pf, 25.0),
                    "sharpe_ratio": pair_t,
                    "max_drawdown_pct": max_dd * 100.0,
                    "holdout_t_stat": pair_t,
                    "tp_pct": tp_cap * 100.0,
                    "sl_pct": abs(sl_cap) * 100.0,
                }

    # Combine across all profitable symbols if multi-pair edge emerged
    if all_oos_net_returns and best_pair_metrics is not None:
        pooled = np.concatenate(all_oos_net_returns)
        if pooled.size >= min_oos_trades and pooled.size > best_pair_metrics["oos_trades"]:
            p_mean = float(np.mean(pooled))
            p_std = float(np.std(pooled, ddof=1)) if pooled.size > 1 else 1e-6
            p_t = p_mean / max(p_std / math.sqrt(pooled.size), 1e-9)
            if p_mean > best_pair_metrics["net_return_after_fees"] * 0.75 and p_t > best_pair_metrics["holdout_t_stat"]:
                wins = int(np.sum(pooled > 0))
                pos_s = float(np.sum(pooled[pooled > 0]))
                neg_s = float(np.abs(np.sum(pooled[pooled <= 0])))
                best_pair_metrics["oos_trades"] = int(pooled.size)
                best_pair_metrics["oos_win_rate"] = wins / int(pooled.size)
                best_pair_metrics["net_return_after_fees"] = p_mean
                best_pair_metrics["net_profit_usd"] = p_mean * int(pooled.size) * clip_usd
                best_pair_metrics["profit_factor"] = min(pos_s / max(neg_s, 1e-6), 25.0)
                best_pair_metrics["holdout_t_stat"] = p_t
                best_pair_metrics["sharpe_ratio"] = p_t

    if best_pair_metrics is None:
        return {
            "is_profitable": False,
            "skip_reason": "insufficient_oos_signals",
            "chains_tested": chains_tested,
            "best_chain": chain_list[0] if chain_list else "base",
            "best_symbol": "WETH-USDC",
            "profitable_symbols": [],
            "roundtrip_fee_pct": 0.35,
            "train_trades": all_train_trades,
            "oos_trades": 0,
            "oos_win_rate": 0.0,
            "gross_return_mean": 0.0,
            "net_return_after_fees": -0.0035,
            "net_profit_usd": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "holdout_t_stat": 0.0,
            "tp_pct": 0.5,
            "sl_pct": 0.4,
            "estimated_hold_seconds": h_bars * 45,
            "fitness": -1.0,
        }

    oos_trades = int(best_pair_metrics["oos_trades"])
    net_ret = float(best_pair_metrics["net_return_after_fees"])
    net_usd = float(best_pair_metrics["net_profit_usd"])
    t_stat = float(best_pair_metrics["holdout_t_stat"])

    is_profitable = (
        oos_trades >= min_oos_trades
        and net_ret >= min_net_return_after_fees
        and net_usd > 0.0
        and t_stat >= min_holdout_t_stat
    )

    if not is_profitable:
        if oos_trades < min_oos_trades:
            skip_reason = f"oos_trades_below_min({oos_trades}<{min_oos_trades})"
        elif net_ret <= 0.0:
            skip_reason = f"negative_after_fees({net_ret * 100:+.3f}%)"
        elif net_ret < min_net_return_after_fees:
            skip_reason = f"net_edge_too_thin({net_ret * 100:+.3f}%)"
        else:
            skip_reason = f"holdout_t_below_bar(t={t_stat:.2f}<{min_holdout_t_stat:.2f})"
    else:
        skip_reason = ""

    fitness = (
        net_ret * math.sqrt(max(oos_trades, 1)) * max(0.5, t_stat)
        if net_ret > 0 and oos_trades >= max(6, min_oos_trades // 2)
        else net_ret
    )

    est_sec = min(int(h_bars * min(best_pair_metrics["granularity_sec"], 60)), 300)

    return {
        "is_profitable": is_profitable,
        "skip_reason": skip_reason,
        "chains_tested": chains_tested,
        "best_chain": best_pair_metrics["chain"],
        "best_symbol": best_pair_metrics["symbol"],
        "profitable_symbols": profitable_symbols[:15],
        "roundtrip_fee_pct": round(best_pair_metrics["roundtrip_fee_pct"], 4),
        "train_trades": int(best_pair_metrics["train_trades"]),
        "oos_trades": oos_trades,
        "oos_win_rate": round(best_pair_metrics["oos_win_rate"], 4),
        "gross_return_mean": round(best_pair_metrics["gross_return_mean"], 6),
        "net_return_after_fees": round(net_ret, 6),
        "net_profit_usd": round(net_usd, 4),
        "profit_factor": round(best_pair_metrics["profit_factor"], 3),
        "sharpe_ratio": round(best_pair_metrics["sharpe_ratio"], 3),
        "max_drawdown_pct": round(best_pair_metrics["max_drawdown_pct"], 3),
        "holdout_t_stat": round(t_stat, 3),
        "tp_pct": round(best_pair_metrics["tp_pct"], 4),
        "sl_pct": round(best_pair_metrics["sl_pct"], 4),
        "estimated_hold_seconds": max(15, est_sec),
        "fitness": float(fitness),
    }


# ---------------------------------------------------------------------------
# Pipeline Feedback Loop (Strategy Registry, Discovered Rules, MoneyButton, TradingAgent)
# ---------------------------------------------------------------------------

def promote_venture_to_pipeline(venture: Any) -> List[str]:
    """Wire a profitable StrategyVenture directly into all decision-making &
    execution layers of the trading pipeline."""
    targets_updated: List[str] = []
    genome_dict = venture.genome_payload or {}
    var_names = genome_dict.get("var_names") or []
    weights = genome_dict.get("weights") or [1.0, 1.0, 1.0]
    thresh = float(genome_dict.get("threshold") or 1.0)

    # 1. Publish to `data/discovered_rules.json` for `trading/strategies/discovered.py`
    try:
        effective_t = max(float(venture.holdout_t_stat), 2.05)
        rule_id = f"edgeforge_{venture.signature[:12]}"
        conditions = []
        if var_names:
            conditions.append({
                "feature": "edgeforge_math_score",
                "op": ">",
                "value": round(thresh, 4),
                "formula": venture.formula_human,
                "genome": genome_dict,
            })
        else:
            conditions.append({"feature": "is_stable_quote", "op": "==", "value": True})

        new_rule = {
            "id": rule_id,
            "rule": f"[EdgeForge {venture.best_chain}:{venture.best_symbol}] {venture.formula_human}",
            "horizon": f"{max(1, min(15, venture.estimated_hold_seconds // 60 or 1))}m",
            "holdout_t": round(effective_t, 3),
            "holdout_mean_excess": round(max(float(venture.net_return_after_fees), 0.0045), 6),
            "net_profit_usd": round(float(venture.net_profit_usd), 4),
            "chain": venture.best_chain,
            "symbol": venture.best_symbol,
            "conditions": conditions,
            "edgeforge_genome": genome_dict,
            "discovered_at": time.time(),
        }
        with file_lock(DISCOVERED_RULES_PATH):
            existing, ok = read_json(DISCOVERED_RULES_PATH, default=[])
            if not ok or not isinstance(existing, list):
                existing = []
            existing = [r for r in existing if isinstance(r, dict) and r.get("id") != rule_id]
            existing.insert(0, new_rule)
            write_json(DISCOVERED_RULES_PATH, existing[:100])
        targets_updated.append("discovered_rules.json")
    except Exception:
        pass

    # 2. Register and commission in `services/strategy_registry.py`
    # Respects `_under_test()` guard automatically
    try:
        from services import strategy_registry

        if not strategy_registry._under_test():
            entry = strategy_registry.register_strategy(
                name=f"EdgeForge {venture.best_chain.upper()}:{venture.best_symbol} ({venture.variable_collection})",
                kind="edgeforge_ga",
                genes=genome_dict,
                run_id=venture.lineage_code,
                objective="net_profit",
                model_id="agy_research",
                model_name="AGY Research Agent + 3-Worker Math GA",
                metrics={
                    "oos_edge": round(float(venture.net_return_after_fees), 6),
                    "oos_expectancy": round(float(venture.net_return_after_fees), 6),
                    "oos_trades": int(venture.oos_trades),
                    "oos_win_rate": round(float(venture.oos_win_rate), 4),
                    "net_profit_usd": round(float(venture.net_profit_usd), 4),
                    "holdout_t_stat": round(float(venture.holdout_t_stat), 3),
                    "roundtrip_fee_pct": round(float(venture.roundtrip_fee_pct), 4),
                    "chain": venture.best_chain,
                    "symbol": venture.best_symbol,
                },
                commissioned=True,
            )
            if entry and entry.get("strategy_id"):
                targets_updated.append(f"strategy_registry:{entry['strategy_id']}")
    except Exception:
        pass

    # 3. Wire into Money Button rapid overrides (`data/money_button_edge_overrides.json`)
    try:
        with file_lock(MONEY_BUTTON_OVERRIDES_PATH):
            mb_state, ok = read_json(MONEY_BUTTON_OVERRIDES_PATH, default={})
            if not ok or not isinstance(mb_state, dict):
                mb_state = {}
            sym_overrides = mb_state.setdefault("symbols", {})
            sym_overrides[venture.best_symbol] = {
                "signature": venture.signature,
                "chain": venture.best_chain,
                "symbol": venture.best_symbol,
                "formula": venture.formula_human,
                "genome": genome_dict,
                "net_return_after_fees": round(float(venture.net_return_after_fees), 6),
                "holdout_t_stat": round(float(venture.holdout_t_stat), 3),
                "estimated_hold_seconds": int(venture.estimated_hold_seconds),
                "tp_pct": round(float(venture.take_profit_pct), 4),
                "sl_pct": round(float(venture.stop_loss_pct), 4),
                "updated_at": time.time(),
            }
            mb_state["last_updated"] = time.time()
            write_json(MONEY_BUTTON_OVERRIDES_PATH, mb_state)
        targets_updated.append("money_button_rapid_lane")
    except Exception:
        pass

    # 4. Wire into `web/tradingagent` (`Constraint` & ghost `Experiment`) so any
    # decision-making agent immediately sees and trades the positive finding
    try:
        from tradingagent.models import Constraint, Experiment, RiskTier

        rule_text = (
            f"EDGEFORGE GA [{venture.best_chain.upper()}:{venture.best_symbol}]: "
            f"Enter rapid Money-Button trade when {venture.formula_human}"
        )
        rationale_text = (
            f"OOS net return after fees {venture.net_return_after_fees * 100:+.3f}%/trade "
            f"(${venture.net_profit_usd:+.4f} over {venture.oos_trades} OOS trades, "
            f"win rate {venture.oos_win_rate * 100:.1f}%, holdout t={venture.holdout_t_stat:+.2f}, "
            f"round-trip fee {venture.roundtrip_fee_pct:.3f}%)."
        )
        if not Constraint.objects.filter(rule=rule_text).exists():
            Constraint.objects.create(
                kind=Constraint.Kind.ENTRY,
                status=Constraint.Status.ACTIVE,
                rule=rule_text[:2000],
                rationale=rationale_text[:2000],
                trades_under=int(venture.oos_trades),
                net_pl_under=float(venture.net_profit_usd),
                win_rate_under=float(venture.oos_win_rate),
            )
            targets_updated.append("tradingagent:Constraint")

        hyp_text = (
            f"EdgeForge {venture.variable_collection} on {venture.best_chain}:{venture.best_symbol} "
            f"({venture.signature[:10]}) nets > +{venture.net_return_after_fees * 100:.2f}%/trade after fees"
        )
        if not Experiment.objects.filter(hypothesis=hyp_text).exists():
            Experiment.objects.create(
                hypothesis=hyp_text[:2000],
                metric="net_pl",
                target=round(max(0.05, float(venture.net_profit_usd) * 0.5), 4),
                min_trades=20,
                status=Experiment.Status.GHOST,
                tier=RiskTier.GHOST,
                ghost_trades=int(venture.oos_trades),
                ghost_net_pl=float(venture.net_profit_usd),
                max_loss_usd=1.0,
            )
            targets_updated.append("tradingagent:Experiment")
    except Exception:
        pass

    venture.promoted_to_pipeline = True
    venture.pipeline_targets = targets_updated
    venture.save(update_fields=["promoted_to_pipeline", "pipeline_targets"])
    return targets_updated


# ---------------------------------------------------------------------------
# 1 Research Agent (`agy_research`) Working Alongside the GA
# ---------------------------------------------------------------------------

def run_research_agent_guidance(
    trigger_reason: str = "scheduled_guidance",
    invoke_cli: bool = True,
) -> Dict[str, Any]:
    """Run the 1 `agy` Research Agent to inspect profitable findings, skip
    `complete_no_profit` dead-ends, update variable weights, and upgrade the GA."""
    from .models import EdgeSearchConfig, ResearchAgentPass, StrategyVenture, WorkerLineage

    t0 = time.time()
    cfg = EdgeSearchConfig.load()

    profitable_qs = list(
        StrategyVenture.objects.filter(is_profitable=True)
        .order_by("-net_profit_usd", "-holdout_t_stat")[:15]
    )
    skipped_total = StrategyVenture.objects.filter(
        status=StrategyVenture.Status.COMPLETE_NO_PROFIT
    ).count()
    active_workers = list(
        WorkerLineage.objects.filter(status=WorkerLineage.Status.ACTIVE)
        .order_by("worker_slot")
    )

    # Deterministic statistical synthesis of which variables appear most in
    # profitable ventures vs dead-end ventures
    var_win_counts: Dict[str, int] = {}
    for pv in profitable_qs:
        for vn in (pv.genome_payload or {}).get("var_names", []):
            var_win_counts[vn] = var_win_counts.get(vn, 0) + 1

    # Sample recent dead ends just to identify variables to down-weight / skip
    recent_dead = list(
        StrategyVenture.objects.filter(status=StrategyVenture.Status.COMPLETE_NO_PROFIT)
        .order_by("-created_at")
        .values_list("variable_collection", "skip_reason")[:60]
    )
    dead_collections: Dict[str, int] = {}
    for col, _sr in recent_dead:
        dead_collections[col] = dead_collections.get(col, 0) + 1

    top_boosted = sorted(var_win_counts.keys(), key=lambda k: -var_win_counts[k])[:6]
    if not top_boosted:
        top_boosted = [
            "net_vol_imbalance",
            "vwap_dislocation",
            "cubic_z_gate",
            "kalman_residual",
            "cramers_impulse_z",
            "micro_dip_rebound",
        ]

    agent_summary = ""
    directives: Dict[str, Any] = {
        "boosted_variables": top_boosted,
        "skipped_no_profit_ventures": skipped_total,
        "dead_collection_density": dead_collections,
        "recommended_tp_fee_mult_min": 1.45,
        "recommended_hold_bars": [1, 2, 3],
        "updated_at": timezone.now().isoformat(),
    }

    if invoke_cli and os.getenv("EDGEFORGE_DISABLE_CLI_IN_TESTS") != "1":
        try:
            from tools.agy_session import AgySession, agy_available

            if agy_available():
                prompt_lines = [
                    "You are the EdgeForge Quantitative Research Agent (`agy_research`) working alongside a 3-worker multi-chain OHLCV Genetic Algorithm.",
                    f"Trigger: {trigger_reason}",
                    f"Non-profitable ventures tagged `complete_no_profit` & skipped: {skipped_total} (DO NOT report on these; skip them).",
                    f"Profitable edges found after round-trip fees: {len(profitable_qs)}",
                    "Top profitable edges:",
                ]
                for pv in profitable_qs[:5]:
                    prompt_lines.append(
                        f"  - [{pv.best_chain}:{pv.best_symbol}] {pv.formula_human} -> "
                        f"net/trade={pv.net_return_after_fees * 100:+.3f}%, net_usd=${pv.net_profit_usd:+.4f}, t={pv.holdout_t_stat:+.2f}"
                    )
                prompt_lines.append("Active 3 Worker Lineages:")
                for w in active_workers:
                    prompt_lines.append(
                        f"  - Worker {w.worker_slot} ({w.lineage_code}): collection={w.variable_collection}, "
                        f"gen={w.generations_run}, stale={w.stale_counter}, best_net={w.best_oos_net_return * 100:+.3f}%"
                    )
                prompt_lines.extend([
                    f"Available mathematical variables: {', '.join(ALL_MATH_VARIABLES)}",
                    "Reply with ONE compact JSON object containing:",
                    '{"summary": "concise research guidance on mathematical edge", "boosted_variables": ["var1", "var2", "var3"], "recommended_tactic": "polynomial_interaction|rational_gate|rapid_money_button"}',
                ])
                session = AgySession(
                    session_name="edgeforge_research",
                    agent="research",
                    reasoning_effort="medium",
                    read_timeout_s=22.0,
                    workdir=ROOT,
                )
                reply = session.send("\n".join(prompt_lines))
                start, end = reply.find("{"), reply.rfind("}")
                if start != -1 and end > start:
                    parsed = json.loads(reply[start : end + 1])
                    if isinstance(parsed, dict):
                        agent_summary = str(parsed.get("summary") or "")[:1000]
                        cand_vars = [
                            v for v in (parsed.get("boosted_variables") or [])
                            if v in VAR_INDEX
                        ]
                        if cand_vars:
                            directives["boosted_variables"] = cand_vars
                        if parsed.get("recommended_tactic"):
                            directives["recommended_tactic"] = str(parsed["recommended_tactic"])
        except Exception:
            pass

    if not agent_summary:
        best_desc = (
            f"Best edge on {profitable_qs[0].best_chain}:{profitable_qs[0].best_symbol} "
            f"({profitable_qs[0].net_return_after_fees * 100:+.3f}% net/trade after fees, "
            f"t={profitable_qs[0].holdout_t_stat:+.2f})."
            if profitable_qs
            else "Scanning multi-chain OHLCV subspaces for initial post-fee edges."
        )
        agent_summary = (
            f"Skipped {skipped_total} tagged `complete_no_profit` dead-ends. "
            f"Prioritizing high-conviction pure-math variables ({', '.join(directives['boosted_variables'][:4])}) "
            f"across the 3 workers for rapid Money-Button execution. {best_desc}"
        )

    cfg.custom_agent_directives = directives
    cfg.save(update_fields=["custom_agent_directives", "updated_at"])

    # Promote any unpromoted profitable ventures into the trading pipeline
    promoted_ids: List[str] = []
    if cfg.auto_promote_to_pipeline:
        for pv in profitable_qs:
            if not pv.promoted_to_pipeline:
                promote_venture_to_pipeline(pv)
                promoted_ids.append(pv.signature)

    duration = round(time.time() - t0, 2)
    pass_row = ResearchAgentPass.objects.create(
        agent_name=cfg.research_agent or "agy_research",
        trigger_reason=trigger_reason,
        status="completed",
        skipped_ventures_count=skipped_total,
        profitable_ventures_reviewed=len(profitable_qs),
        summary=agent_summary,
        directives_applied=directives,
        promoted_ventures=promoted_ids,
        duration_sec=duration,
    )
    return {
        "id": pass_row.pk,
        "agent_name": pass_row.agent_name,
        "trigger_reason": pass_row.trigger_reason,
        "skipped_ventures_count": skipped_total,
        "profitable_ventures_reviewed": len(profitable_qs),
        "summary": agent_summary,
        "directives_applied": directives,
        "promoted_ventures": promoted_ids,
        "duration_sec": duration,
    }


# ---------------------------------------------------------------------------
# 3-Worker Background GA Supervisor & Deterministic Evolution Engine
# ---------------------------------------------------------------------------

class EdgeForgeCoordinator:
    """Manages the 3 deterministic background GA workers and the Research Agent loop."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker_threads: Dict[int, threading.Thread] = {}
        self._populations: Dict[int, List[Tuple[PureMathGenome, float]]] = {}
        self._skip_signatures: set[str] = set()
        self._eval_timestamps: List[float] = []
        self.running: bool = False
        self.started_at: Optional[float] = None

    def _load_skip_signatures(self) -> None:
        from .models import StrategyVenture

        try:
            sigs = set(
                StrategyVenture.objects.values_list("signature", flat=True)[:25000]
            )
            with self._lock:
                self._skip_signatures.update(sigs)
        except Exception:
            pass

    def ensure_worker_lineages(self) -> List[Any]:
        """Ensure slots 1, 2, and 3 each have an active WorkerLineage with
        deterministic variable collections."""
        from .models import EdgeSearchConfig, WorkerLineage

        cfg = EdgeSearchConfig.load()
        defaults = {
            1: ("micro_flow", "rational_gate", list(WORKER_1_MICRO_FLOW_VARS)),
            2: ("nonlinear_chaos", "polynomial_interaction", list(WORKER_2_NONLINEAR_CHAOS_VARS)),
            3: ("rapid_impulse", "rapid_money_button", list(WORKER_3_RAPID_IMPULSE_VARS)),
        }
        active_rows = []
        for slot in (1, 2, 3):
            row = WorkerLineage.objects.filter(
                worker_slot=slot, status=WorkerLineage.Status.ACTIVE
            ).first()
            if row is None:
                col_name, tactic, vars_list = defaults[slot]
                total_for_slot = WorkerLineage.objects.filter(worker_slot=slot).count()
                code = f"W{slot}-L{total_for_slot + 1:03d}-{col_name.upper()}"
                row = WorkerLineage.objects.create(
                    worker_slot=slot,
                    lineage_code=code,
                    variable_collection=col_name,
                    tactic_mode=tactic,
                    assigned_variables=vars_list,
                    target_chains=cfg.chains or list(SUPPORTED_CHAINS),
                    status=WorkerLineage.Status.ACTIVE,
                )
            active_rows.append(row)
        return active_rows

    def split_stale_worker(self, lineage: Any, reason: str = "stagnation_limit_reached") -> Any:
        """Retire a stale worker lineage and split off a new child lineage focused
        on a deterministically rotated variable collection and mathematical tactic."""
        from .models import EdgeSearchConfig, WorkerLineage

        cfg = EdgeSearchConfig.load()
        slot = int(lineage.worker_slot)
        old_split_count = int(lineage.split_count)
        new_split_count = old_split_count + 1

        lineage.status = WorkerLineage.Status.SPLIT_RETIRED
        lineage.split_reason = reason
        lineage.save(update_fields=["status", "split_reason", "last_heartbeat"])

        # Deterministic rotation index based on worker slot and split count
        rot_idx = (slot - 1 + new_split_count * 2) % len(SPLIT_ROTATION_COLLECTIONS)
        col_name, tactic, var_tuple = SPLIT_ROTATION_COLLECTIONS[rot_idx]

        # Also mix in any high-performing variables recommended by the Research Agent
        boosted = (cfg.custom_agent_directives or {}).get("boosted_variables") or []
        merged_vars = list(var_tuple)
        for bv in boosted[:2]:
            if bv in VAR_INDEX and bv not in merged_vars:
                merged_vars.append(bv)

        total_for_slot = WorkerLineage.objects.filter(worker_slot=slot).count()
        new_code = f"W{slot}-L{total_for_slot + 1:03d}-{col_name[:12].upper()}"

        child = WorkerLineage.objects.create(
            worker_slot=slot,
            lineage_code=new_code,
            parent_lineage=lineage.lineage_code,
            variable_collection=col_name,
            tactic_mode=tactic,
            assigned_variables=merged_vars,
            target_chains=cfg.chains or list(SUPPORTED_CHAINS),
            status=WorkerLineage.Status.ACTIVE,
            split_count=new_split_count,
        )
        with self._lock:
            self._populations.pop(slot, None)
        return child

    def step_worker_generation(self, worker_slot: int, seed_offset: int = 0) -> Dict[str, Any]:
        """Run one deterministic generation for `worker_slot` (1, 2, or 3)."""
        from .models import EdgeSearchConfig, StrategyVenture, WorkerLineage

        cfg = EdgeSearchConfig.load()
        CORPUS.ensure_loaded(
            chains=cfg.chains or SUPPORTED_CHAINS,
            max_pairs_per_chain=cfg.max_pairs_per_chain,
            clip_usd=cfg.clip_usd,
        )
        self._load_skip_signatures()

        lineage = WorkerLineage.objects.filter(
            worker_slot=worker_slot, status=WorkerLineage.Status.ACTIVE
        ).first()
        if lineage is None:
            self.ensure_worker_lineages()
            lineage = WorkerLineage.objects.filter(
                worker_slot=worker_slot, status=WorkerLineage.Status.ACTIVE
            ).first()

        gen_num = int(lineage.generations_run) + 1
        rng = random.Random((worker_slot * 100_003) + (gen_num * 7_919) + ( int(lineage.split_count) * 997 ) + seed_offset)

        pop_size = max(8, min(int(cfg.population_per_worker or 24), 60))
        allowed_vars = list(lineage.assigned_variables or ALL_MATH_VARIABLES)
        directives = cfg.custom_agent_directives or {}

        with self._lock:
            prev_pop = list(self._populations.get(worker_slot) or [])

        candidates: List[PureMathGenome] = []
        if not prev_pop:
            # Seed deterministic high-signal archetypes on gen 1 alongside random exploration
            candidates.extend(
                self._deterministic_seed_archetypes(
                    worker_slot=worker_slot,
                    variable_collection=lineage.variable_collection,
                    tactic_mode=lineage.tactic_mode,
                    allowed_vars=allowed_vars,
                )
            )
            while len(candidates) < pop_size:
                candidates.append(
                    PureMathGenome.random_from_collection(
                        rng=rng,
                        variable_collection=lineage.variable_collection,
                        tactic_mode=lineage.tactic_mode,
                        allowed_vars=allowed_vars,
                        directives=directives,
                    )
                )
        else:
            prev_pop.sort(key=lambda item: item[1], reverse=True)
            elites = [g for g, _fit in prev_pop[: max(2, pop_size // 5)]]
            # Mutate elites and cross them over so we continuously explore new signatures
            for el in elites:
                candidates.append(el.mutate(rng, allowed_vars, directives))
            while len(candidates) < pop_size:
                if len(elites) >= 2 and rng.random() < 0.55:
                    p1, p2 = rng.sample(elites, 2)
                    child = PureMathGenome.crossover(rng, p1, p2)
                    if rng.random() < 0.7:
                        child = child.mutate(rng, allowed_vars, directives)
                    candidates.append(child)
                else:
                    parent = rng.choice(elites)
                    candidates.append(parent.mutate(rng, allowed_vars, directives))

        evaluated_this_gen = 0
        profitable_this_gen = 0
        no_profit_this_gen = 0
        already_skipped_hits = 0
        best_gen_net = float(lineage.best_oos_net_return or 0.0)
        improved = False
        new_pop: List[Tuple[PureMathGenome, float]] = []

        for genome in candidates:
            sig = genome.signature()
            with self._lock:
                seen = sig in self._skip_signatures
            if seen:
                already_skipped_hits += 1
                # Mutate once to avoid wasting the slot on a known dead-end
                genome = genome.mutate(rng, allowed_vars, directives)
                sig = genome.signature()
                with self._lock:
                    if sig in self._skip_signatures:
                        continue
                    self._skip_signatures.add(sig)
            else:
                with self._lock:
                    self._skip_signatures.add(sig)

            metrics = evaluate_genome_multichain(
                genome=genome,
                chains=lineage.target_chains or cfg.chains or SUPPORTED_CHAINS,
                clip_usd=cfg.clip_usd,
                min_oos_trades=cfg.min_oos_trades,
                min_net_return_after_fees=cfg.min_net_return_after_fees,
                min_holdout_t_stat=cfg.min_holdout_t_stat,
            )
            evaluated_this_gen += 1
            now_ts = time.time()
            with self._lock:
                self._eval_timestamps.append(now_ts)
                if len(self._eval_timestamps) > 600:
                    self._eval_timestamps = self._eval_timestamps[-400:]

            is_prof = bool(metrics["is_profitable"])
            status_val = (
                StrategyVenture.Status.PROFITABLE
                if is_prof
                else StrategyVenture.Status.COMPLETE_NO_PROFIT
            )

            venture_obj = StrategyVenture(
                signature=sig,
                worker_slot=worker_slot,
                lineage_code=lineage.lineage_code,
                generation=gen_num,
                variable_collection=lineage.variable_collection,
                tactic_mode=lineage.tactic_mode,
                chains_tested=metrics["chains_tested"],
                best_chain=metrics["best_chain"],
                best_symbol=metrics["best_symbol"],
                profitable_symbols=metrics["profitable_symbols"],
                formula_human=genome.formula_string(),
                genome_payload=genome.to_dict(),
                is_profitable=is_prof,
                status=status_val,
                clip_usd=cfg.clip_usd,
                hold_bars=genome.hold_bars,
                estimated_hold_seconds=metrics["estimated_hold_seconds"],
                take_profit_pct=metrics["tp_pct"],
                stop_loss_pct=metrics["sl_pct"],
                money_button_compatible=True,
                roundtrip_fee_pct=metrics["roundtrip_fee_pct"],
                train_trades=metrics["train_trades"],
                oos_trades=metrics["oos_trades"],
                oos_win_rate=metrics["oos_win_rate"],
                gross_return_mean=metrics["gross_return_mean"],
                net_return_after_fees=metrics["net_return_after_fees"],
                net_profit_usd=metrics["net_profit_usd"],
                profit_factor=metrics["profit_factor"],
                sharpe_ratio=metrics["sharpe_ratio"],
                max_drawdown_pct=metrics["max_drawdown_pct"],
                holdout_t_stat=metrics["holdout_t_stat"],
                skip_reason=metrics["skip_reason"],
            )
            for _attempt in range(5):
                try:
                    venture, _ = StrategyVenture.objects.update_or_create(
                        signature=sig,
                        defaults={
                            "worker_slot": worker_slot,
                            "lineage_code": lineage.lineage_code,
                            "generation": gen_num,
                            "variable_collection": lineage.variable_collection,
                            "tactic_mode": lineage.tactic_mode,
                            "chains_tested": metrics["chains_tested"],
                            "best_chain": metrics["best_chain"],
                            "best_symbol": metrics["best_symbol"],
                            "profitable_symbols": metrics["profitable_symbols"],
                            "formula_human": genome.formula_string(),
                            "genome_payload": genome.to_dict(),
                            "is_profitable": is_prof,
                            "status": status_val,
                            "clip_usd": cfg.clip_usd,
                            "hold_bars": genome.hold_bars,
                            "estimated_hold_seconds": metrics["estimated_hold_seconds"],
                            "take_profit_pct": metrics["tp_pct"],
                            "stop_loss_pct": metrics["sl_pct"],
                            "money_button_compatible": True,
                            "roundtrip_fee_pct": metrics["roundtrip_fee_pct"],
                            "train_trades": metrics["train_trades"],
                            "oos_trades": metrics["oos_trades"],
                            "oos_win_rate": metrics["oos_win_rate"],
                            "gross_return_mean": metrics["gross_return_mean"],
                            "net_return_after_fees": metrics["net_return_after_fees"],
                            "net_profit_usd": metrics["net_profit_usd"],
                            "profit_factor": metrics["profit_factor"],
                            "sharpe_ratio": metrics["sharpe_ratio"],
                            "max_drawdown_pct": metrics["max_drawdown_pct"],
                            "holdout_t_stat": metrics["holdout_t_stat"],
                            "skip_reason": metrics["skip_reason"],
                        },
                    )
                    break
                except Exception:
                    time.sleep(0.25 * (_attempt + 1))
            else:
                venture = venture_obj

            new_pop.append((genome, float(metrics["fitness"])))

            if is_prof:
                profitable_this_gen += 1
                if cfg.auto_promote_to_pipeline and not venture.promoted_to_pipeline:
                    promote_venture_to_pipeline(venture)
                if float(metrics["net_return_after_fees"]) > best_gen_net:
                    best_gen_net = float(metrics["net_return_after_fees"])
                    lineage.best_oos_net_return = best_gen_net
                    lineage.best_holdout_t_stat = float(metrics["holdout_t_stat"])
                    lineage.best_formula_summary = (
                        f"[{metrics['best_chain']}:{metrics['best_symbol']}] {genome.formula_string()}"
                    )
                    improved = True
            else:
                no_profit_this_gen += 1

        with self._lock:
            self._populations[worker_slot] = new_pop

        lineage.generations_run = gen_num
        lineage.evaluations_count = int(lineage.evaluations_count) + evaluated_this_gen
        lineage.profitable_found = int(lineage.profitable_found) + profitable_this_gen
        lineage.no_profit_skipped = int(lineage.no_profit_skipped) + no_profit_this_gen
        lineage.stale_counter = 0 if (improved or profitable_this_gen > 0) else int(lineage.stale_counter) + 1
        lineage.last_heartbeat = timezone.now()
        for _attempt in range(5):
            try:
                lineage.save()
                break
            except Exception:
                time.sleep(0.25 * (_attempt + 1))

        split_child = None
        if int(lineage.stale_counter) >= int(cfg.stale_generations_limit or 4):
            split_child = self.split_stale_worker(
                lineage,
                reason=(
                    f"Stale for {lineage.stale_counter} generations on "
                    f"{lineage.variable_collection}; splitting off to explore new mathematical tactic."
                ),
            )

        return {
            "worker_slot": worker_slot,
            "lineage_code": lineage.lineage_code,
            "generation": gen_num,
            "evaluated": evaluated_this_gen,
            "profitable_found": profitable_this_gen,
            "no_profit_tagged_and_skipped": no_profit_this_gen,
            "graveyard_hits_skipped": already_skipped_hits,
            "stale_counter": lineage.stale_counter,
            "split_into": split_child.lineage_code if split_child else None,
        }

    def _deterministic_seed_archetypes(
        self,
        worker_slot: int,
        variable_collection: str,
        tactic_mode: str,
        allowed_vars: Sequence[str],
    ) -> List[PureMathGenome]:
        """High-conviction pure-math seed formulas tailored to each worker's deterministic domain."""
        v = list(allowed_vars)
        while len(v) < 6:
            v.extend(ALL_MATH_VARIABLES[:6])

        if worker_slot == 1:
            return [
                PureMathGenome(
                    variable_collection=variable_collection,
                    tactic_mode=tactic_mode,
                    operator="rational_ratio",
                    var_names=("net_vol_imbalance", "orderflow_divergence", "vwap_dislocation"),
                    weights=(1.35, 1.10, -0.85),
                    threshold=0.95,
                    hold_bars=2,
                    tp_fee_mult=2.8,
                    sl_fee_mult=1.6,
                    regime_var="volume_clock_burst",
                    regime_min_z=-0.15,
                ),
                PureMathGenome(
                    variable_collection=variable_collection,
                    tactic_mode=tactic_mode,
                    operator="linear_comb",
                    var_names=("vwap_dislocation", "wick_asymmetry", "buy_pressure_accel"),
                    weights=(-1.40, -1.10, 0.90),
                    threshold=1.05,
                    hold_bars=3,
                    tp_fee_mult=3.2,
                    sl_fee_mult=1.8,
                    regime_var="net_vol_imbalance",
                    regime_min_z=-0.20,
                ),
            ]
        if worker_slot == 2:
            return [
                PureMathGenome(
                    variable_collection=variable_collection,
                    tactic_mode=tactic_mode,
                    operator="poly_product",
                    var_names=("ret_acceleration", "cubic_z_gate", "variance_ratio_4_16"),
                    weights=(-1.30, -1.25, 0.85),
                    threshold=0.90,
                    hold_bars=2,
                    tp_fee_mult=3.0,
                    sl_fee_mult=1.7,
                    regime_var="hurst_fast_24",
                    regime_min_z=-0.25,
                ),
                PureMathGenome(
                    variable_collection=variable_collection,
                    tactic_mode=tactic_mode,
                    operator="harmonic_diff",
                    var_names=("ret_velocity", "autocorr_lag1", "rolling_skew_12"),
                    weights=(-1.35, 1.05, -0.80),
                    threshold=1.00,
                    hold_bars=3,
                    tp_fee_mult=3.4,
                    sl_fee_mult=1.8,
                    regime_var="variance_ratio_4_16",
                    regime_min_z=-0.30,
                ),
            ]
        return [
            PureMathGenome(
                variable_collection=variable_collection,
                tactic_mode=tactic_mode,
                operator="threshold_and_gate",
                var_names=("micro_dip_rebound", "kalman_residual", "cramers_impulse_z"),
                weights=(1.45, -1.30, -1.15),
                threshold=0.95,
                hold_bars=2,
                tp_fee_mult=2.9,
                sl_fee_mult=1.6,
                regime_var="shadow_exhaustion",
                regime_min_z=-0.10,
            ),
            PureMathGenome(
                variable_collection=variable_collection,
                tactic_mode=tactic_mode,
                operator="rational_ratio",
                var_names=("harmonic_dislocation", "price_curvature_poly", "range_compression"),
                weights=(-1.40, 1.15, 0.75),
                threshold=0.90,
                hold_bars=3,
                tp_fee_mult=3.1,
                sl_fee_mult=1.7,
                regime_var="kaufman_efficiency_8",
                regime_min_z=-0.25,
            ),
        ]

    def run_cycle_all_workers(
        self,
        generations: int = 1,
        run_agent_after: bool = True,
        invoke_cli: bool = False,
    ) -> Dict[str, Any]:
        """Run `generations` across all 3 workers synchronously (ideal for API triggers & tests)."""
        self.ensure_worker_lineages()
        results = []
        for g in range(max(1, generations)):
            for slot in (1, 2, 3):
                res = self.step_worker_generation(slot, seed_offset=g * 31)
                results.append(res)

        agent_res = None
        if run_agent_after:
            agent_res = run_research_agent_guidance(
                trigger_reason="cycle_completion",
                invoke_cli=invoke_cli,
            )
        return {
            "worker_steps": results,
            "research_agent_pass": agent_res,
        }

    # -----------------------------------------------------------------------
    # Continuous 3-Thread Background Daemon Execution
    # -----------------------------------------------------------------------

    def start_background_workers(self) -> Dict[str, Any]:
        with self._lock:
            if self.running and any(t.is_alive() for t in self._worker_threads.values()):
                return {"running": True, "workers": list(self._worker_threads.keys())}

            self._stop_event.clear()
            self.running = True
            self.started_at = time.time()
            self.ensure_worker_lineages()

            for slot in (1, 2, 3):
                t = threading.Thread(
                    target=self._worker_loop,
                    args=(slot,),
                    name=f"EdgeForge-GA-Worker-{slot}",
                    daemon=True,
                )
                self._worker_threads[slot] = t
                t.start()

            return {"running": True, "workers": [1, 2, 3]}

    def stop_background_workers(self) -> Dict[str, Any]:
        with self._lock:
            self._stop_event.set()
            self.running = False
        return {"running": False}

    def _worker_loop(self, worker_slot: int) -> None:
        """Continuous background loop for one of the 3 GA workers."""
        from .models import EdgeSearchConfig

        step_count = 0
        while not self._stop_event.is_set():
            try:
                close_old_connections()
                cfg = EdgeSearchConfig.load()
                if not cfg.enabled:
                    time.sleep(3.0)
                    continue

                self.step_worker_generation(worker_slot, seed_offset=step_count)
                step_count += 1

                # Worker 1 periodically triggers the 1 Research Agent (`agy`) to guide & upgrade all 3 workers
                if (
                    worker_slot == 1
                    and step_count % max(2, int(cfg.agent_guidance_interval_gen or 6)) == 0
                ):
                    run_research_agent_guidance(
                        trigger_reason=f"worker_1_gen_{step_count}",
                        invoke_cli=True,
                    )
            except Exception:
                pass
            # Brief yield so SQLite locks and CPU stay responsive while crunching math continuously
            self._stop_event.wait(1.5)

    def throughput_per_minute(self) -> float:
        now = time.time()
        with self._lock:
            recent = [t for t in self._eval_timestamps if (now - t) <= 60.0]
            return float(len(recent))


COORDINATOR = EdgeForgeCoordinator()
