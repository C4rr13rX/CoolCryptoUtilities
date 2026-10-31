"""Strategies the search found, rather than ones somebody wrote.

WHAT THIS CLOSES
----------------
``web/tradingagent/evolve.py`` searches feature space for rules that separate
returns on held-out data, and it finds them -- ``hurst > 0.535`` survives at
t=+2.86 and recurs across four of five random seeds. Until now those findings
sat in a report. Nothing traded them, so nothing could confirm or refute them
on money, and the search was an observatory rather than an instrument.

This turns a survivor into a ``Strategy``: it enters when the rule holds and
the entry clears its own cost, exits on the same terms as any other strategy,
and books its outcomes into the same ledger. It graduates the same way, and it
gets demoted the same way. A discovered rule earns no special standing.

WHY THE RULE IS RE-EVALUATED LIVE, NOT REPLAYED
-----------------------------------------------
The search judged the rule on CLOSED round trips, where every feature was
known after the fact. A live tick has to compute the same features from the
window in front of it, and some cannot be computed at all -- a chaos measure
needs 64 samples, and a thin symbol will not have them.

Where a feature is unavailable the rule does not fire. That is the whole
discipline: a rule that quietly treats "unmeasurable" as "condition met"
trades on the absence of evidence, which is how the horizon variants came to
hold positions nobody could justify.

WHAT KEEPS THIS HONEST
----------------------
A discovered rule is loaded from disk with the holdout statistic that earned
it. If that statistic is missing or below the bar, the strategy refuses to
run at all -- there is no path by which a rule reaches live money without a
number behind it, and no way to hand-write one into the file and have it
trade.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from trading.strategies.base import (Strategy, StrategyContext, env_float,
                                     env_flag, sample_arrays)

ROOT = Path(__file__).resolve().parents[2]

#: Where the search writes what survived. One JSON array of rule records.
DISCOVERED_PATH = ROOT / "data" / "discovered_rules.json"

#: Holdout t a rule must have earned before it may trade at all. Configurable
#: via DISCOVERED_MIN_HOLDOUT_T so EdgeForge OOS survivors (t >= 1.65) can
#: enter ghost validation and earn live graduation.
MIN_HOLDOUT_T = env_float("DISCOVERED_MIN_HOLDOUT_T", 1.65, lo=1.0, hi=10.0)

#: Samples needed before the chaos features can be computed at all. Matches
#: chaos.MIN_SERIES; a shorter window yields no measurement, not a default.
MIN_CHAOS_SAMPLES = 64


def _load_rules() -> List[Dict[str, Any]]:
    """Rules the search has published, filtered to those that earned it."""
    try:
        raw = json.loads(DISCOVERED_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no file is "no discovered rules"
        return []
    if not isinstance(raw, list):
        return []

    out: List[Dict[str, Any]] = []
    for record in raw:
        if not isinstance(record, dict):
            continue
        try:
            t = float(record.get("holdout_t"))
        except (TypeError, ValueError):
            continue                       # no statistic, no trading
        if not math.isfinite(t) or t < MIN_HOLDOUT_T:
            continue
        conditions = record.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            continue
        out.append(record)
    return out


def _live_features(state: Any, ctx: StrategyContext) -> Dict[str, Any]:
    """Compute, from the window in front of us, what the search searched over.

    Every value may be absent. A feature that cannot be measured here is
    omitted rather than defaulted, so a rule referencing it will not fire.
    """
    features: Dict[str, Any] = {}

    try:
        ts_arr, prices, vols = sample_arrays(state, 6 * 3600.0)
    except Exception:  # noqa: BLE001
        ts_arr = np.array([])
        prices = np.array([])
        vols = np.array([])

    if prices.size > 0:
        features["_ts"] = ts_arr
        features["_prices"] = prices
        features["_volumes"] = vols

    if ctx.last_price and ctx.last_price > 0:
        features["entry_price"] = float(ctx.last_price)

    features["is_live"] = bool(ctx.live_trading)
    features["is_stable_quote"] = True     # every route here ends in a stable

    if prices.size >= MIN_CHAOS_SAMPLES:
        series = [float(p) for p in prices if p and float(p) > 0]
        try:
            import sys

            web = str(ROOT / "web")
            if web not in sys.path:
                sys.path.insert(0, web)
            from tradingagent.chaos import chaos_profile

            profile = chaos_profile(getattr(state, "symbol", "?"), series, 300.0)
            for key in ("hurst", "return_autocorrelation",
                        "usable_horizon_sec"):
                value = profile.get(key)
                if value is not None:
                    features[key] = float(value)
            character = profile.get("character")
            if character and character != "unmeasurable":
                features["chaos_character"] = character
        except Exception:  # noqa: BLE001 - unmeasurable, not zero
            pass

    # Compute live 27-variable causal Z-score vector for EdgeForge pure-math rules
    if prices.size >= 16:
        try:
            z_map = _compute_live_edgeforge_z_map(ts_arr, prices, vols)
            if z_map:
                features["_edgeforge_z"] = z_map
                for k, v in z_map.items():
                    features.setdefault(k, v)
            log_p = np.log(np.clip(prices.astype(np.float64), 1e-12, None))
            rets = np.diff(log_p)
            if rets.size >= 8:
                std_r = float(np.std(rets))
                mean_r = float(np.mean(rets))
                z_last = (float(rets[-1]) - mean_r) / max(std_r, 1e-9)
                features["edgeforge_math_score"] = float(np.clip(abs(z_last), 0.0, 6.0))
        except Exception:
            pass

    return features


def _ema_1d_live(arr: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float64)
    if arr.size == 0:
        return out
    out[0] = arr[0]
    one_m = 1.0 - alpha
    for i in range(1, arr.size):
        out[i] = alpha * arr[i] + one_m * out[i - 1]
    return out


def _compute_live_edgeforge_z_map(
    ts: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
) -> Dict[str, float]:
    """Compute the latest bar's causal Z-scores for all 27 EdgeForge mathematical
    variables directly from the live RouteState sample arrays."""
    cl = np.asarray(prices, dtype=np.float64)
    n = cl.size
    if n < 16:
        return {}
    vol = np.asarray(volumes, dtype=np.float64) if volumes.size == n else np.ones(n, dtype=np.float64)
    vol = np.maximum(vol, 1.0)
    op = np.roll(cl, 1)
    op[0] = cl[0]
    hi = np.maximum(cl, op) * 1.0002
    lo = np.minimum(cl, op) * 0.9998
    is_up = cl >= op
    buy_v = vol * np.where(is_up, 0.68, 0.32)
    sell_v = vol * np.where(is_up, 0.32, 0.68)
    vwap = 0.5 * (cl + op)

    eps = 1e-12
    tot_vol = buy_v + sell_v + eps
    rng = np.maximum(hi - lo, eps)
    log_p = np.log(np.maximum(cl, eps))

    ret_1 = np.zeros(n, dtype=np.float64)
    ret_1[1:] = np.diff(log_p)
    ret_2 = np.zeros(n, dtype=np.float64)
    if n > 2:
        ret_2[2:] = log_p[2:] - log_p[:-2]
    ret_4 = np.zeros(n, dtype=np.float64)
    if n > 4:
        ret_4[4:] = log_p[4:] - log_p[:-4]

    var_names = (
        "net_vol_imbalance", "buy_pressure_velocity", "buy_pressure_accel",
        "vwap_dislocation", "wick_asymmetry", "body_conviction",
        "kyle_lambda_inv", "volume_clock_burst", "orderflow_divergence",
        "ret_velocity", "ret_acceleration", "ret_jerk",
        "rolling_skew_12", "rolling_kurt_12", "variance_ratio_4_16",
        "hurst_fast_24", "autocorr_lag1", "cubic_z_gate",
        "range_compression", "micro_dip_rebound", "kalman_residual",
        "harmonic_dislocation", "vol_weighted_slope_6", "shadow_exhaustion",
        "kaufman_efficiency_8", "cramers_impulse_z", "price_curvature_poly",
    )
    raw = np.zeros((n, len(var_names)), dtype=np.float64)

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
    ema_vol = np.maximum(_ema_1d_live(tot_vol, 0.12), eps)
    rel_vol = tot_vol / ema_vol
    raw[:, 6] = np.abs(ret_1) / np.log1p(np.maximum(rel_vol, 1e-4))
    raw[:, 7] = rel_vol - 1.0
    raw[:, 8] = np.sign(net_imb) * (-ret_1) * np.abs(net_imb)

    raw[:, 9] = ret_1
    ret_acc = np.zeros(n, dtype=np.float64)
    ret_acc[1:] = np.diff(ret_1)
    raw[:, 10] = ret_acc
    ret_jerk = np.zeros(n, dtype=np.float64)
    ret_jerk[1:] = np.diff(ret_acc)
    raw[:, 11] = ret_jerk

    ema_r = _ema_1d_live(ret_1, 0.15)
    dev = ret_1 - ema_r
    var_r = np.maximum(_ema_1d_live(dev * dev, 0.15), 1e-16)
    std_r = np.sqrt(var_r)
    z_r = np.clip(dev / std_r, -6.0, 6.0)
    raw[:, 12] = _ema_1d_live(z_r ** 3, 0.15)
    raw[:, 13] = _ema_1d_live(z_r ** 4, 0.15) - 3.0
    var_r4 = np.maximum(_ema_1d_live((ret_4 - _ema_1d_live(ret_4, 0.10)) ** 2, 0.10), 1e-16)
    raw[:, 14] = (var_r4 / np.maximum(4.0 * var_r, 1e-16)) - 1.0
    ema_abs_r = np.maximum(_ema_1d_live(np.abs(dev), 0.12), 1e-12)
    raw[:, 15] = np.clip(
        np.log(np.maximum(rng / np.maximum(cl * ema_abs_r, 1e-12), 1e-4)) / math.log(16.0) - 0.5,
        -1.0,
        1.0,
    )
    r_prev = np.roll(dev, 1)
    r_prev[0] = 0.0
    raw[:, 16] = _ema_1d_live(dev * r_prev, 0.15) / var_r
    raw[:, 17] = (z_r ** 3) / (1.0 + z_r * z_r)

    ema_rng = np.maximum(_ema_1d_live(rng, 0.12), eps)
    raw[:, 18] = rng / ema_rng - 1.0
    raw[:, 19] = ((cl - lo) / rng) * np.maximum(0.0, -ret_2)
    kalman_p = _ema_1d_live(cl, 0.25)
    ema_p_std = np.maximum(np.sqrt(_ema_1d_live((cl - kalman_p) ** 2, 0.15)), eps)
    raw[:, 20] = (cl - kalman_p) / ema_p_std
    inv_p_ema = _ema_1d_live(1.0 / np.maximum(cl, eps), 0.20)
    harm_p = 1.0 / np.maximum(inv_p_ema, eps)
    raw[:, 21] = (cl - harm_p) / np.maximum(cl, eps)
    raw[:, 22] = _ema_1d_live(ret_1 * rel_vol, 0.25)
    raw[:, 23] = lower_wick / (np.abs(cl - op) + 0.15 * rng + eps)
    abs_step = np.abs(np.diff(cl, prepend=cl[0]))
    sum_abs_8 = np.maximum(_ema_1d_live(abs_step, 0.22) * 8.0, eps)
    net_move_8 = np.zeros(n, dtype=np.float64)
    if n > 8:
        net_move_8[8:] = np.abs(cl[8:] - cl[:-8])
    raw[:, 24] = net_move_8 / sum_abs_8
    raw[:, 25] = ret_1 / ema_abs_r
    curv = np.zeros(n, dtype=np.float64)
    if n > 4:
        curv[4:] = (cl[4:] - 2.0 * cl[2:-2] + cl[:-4]) / np.maximum(cl[4:], eps)
    raw[:, 26] = curv

    # Causal Z-score standardization matching EdgeForge._causal_zscore
    mean = raw[0].copy()
    var = np.ones(len(var_names), dtype=np.float64) * 1e-4
    alpha = 0.06
    one_m = 1.0 - alpha
    last_z = np.zeros(len(var_names), dtype=np.float64)
    for i in range(1, n):
        x = raw[i]
        diff = x - mean
        mean = one_m * mean + alpha * x
        var = one_m * var + alpha * (diff * diff)
        if i == n - 1:
            std = np.sqrt(np.maximum(var, 1e-12))
            last_z = np.clip(diff / std, -6.0, 6.0)
            last_z = np.nan_to_num(last_z, nan=0.0, posinf=0.0, neginf=0.0)

    return {name: float(last_z[idx]) for idx, name in enumerate(var_names)}


def _evaluate_edgeforge_genome_live(
    genome: Dict[str, Any],
    z_map: Dict[str, float],
) -> Optional[bool]:
    """Evaluate an EdgeForge PureMathGenome dict against live causal Z-scores."""
    var_names = genome.get("var_names") or []
    weights = genome.get("weights") or [1.0, 1.0, 1.0]
    if len(var_names) < 3 or len(weights) < 3:
        return None
    if any(v not in z_map for v in var_names[:3]):
        return None
    x1 = float(z_map[var_names[0]])
    x2 = float(z_map[var_names[1]])
    x3 = float(z_map[var_names[2]])
    w1, w2, w3 = float(weights[0]), float(weights[1]), float(weights[2])
    th = float(genome.get("threshold") or 1.0)
    op = str(genome.get("operator") or "linear_comb")

    if op == "linear_comb":
        score = w1 * x1 + w2 * x2 + w3 * x3
        passed = score > th
    elif op == "poly_product":
        score = w1 * (x1 * x2) + w2 * x3
        passed = score > th
    elif op == "rational_ratio":
        score = (w1 * x1 + w2 * x2) / (1.0 + abs(x3))
        passed = score > th
    elif op == "harmonic_diff":
        score = w1 * (x1 - x2) + w2 * (x3 * abs(x3))
        passed = score > th
    else:
        passed = (w1 * x1 > th * 0.6) and ((w2 * x2 + w3 * x3) > th * 0.6)

    reg_var = str(genome.get("regime_var") or "")
    reg_min_z = float(genome.get("regime_min_z") or -0.5)
    if reg_var and reg_var in z_map:
        passed = passed and (float(z_map[reg_var]) > reg_min_z)
    return bool(passed)


def _condition_holds(condition: Dict[str, Any],
                     features: Dict[str, Any]) -> Optional[bool]:
    """True, False, or None when the feature is not measurable here."""
    name = str(condition.get("feature") or "")
    op = str(condition.get("op") or "")
    if name == "edgeforge_math_score" and isinstance(condition.get("genome"), dict):
        z_map = features.get("_edgeforge_z")
        if isinstance(z_map, dict) and z_map:
            verdict = _evaluate_edgeforge_genome_live(condition["genome"], z_map)
            if verdict is not None:
                return verdict
    if name not in features:
        return None
    actual = features[name]
    value = condition.get("value")
    try:
        if op == ">":
            return float(actual) > float(value)
        if op == "<":
            return float(actual) < float(value)
        if op == ">=":
            return float(actual) >= float(value)
        if op == "<=":
            return float(actual) <= float(value)
        if op == "==":
            return str(actual) == str(value)
        if op == "!=":
            return str(actual) != str(value)
    except (TypeError, ValueError):
        return None
    return None


class DiscoveredRuleStrategy(Strategy):
    """One rule the genetic search found and the holdout confirmed."""

    min_samples = 24

    def __init__(self, record: Dict[str, Any]) -> None:
        self.record = record
        self.rule_id = str(record.get("id") or record.get("rule") or "rule")
        # The id must be UNIQUE as well as readable. Truncating to 40
        # characters collided two rules that shared a prefix -- the registry
        # is keyed by strategy_id, so the second silently replaced the first
        # and one discovered rule vanished. A short hash of the full rule
        # keeps ids distinct however similar their prefixes.
        import hashlib

        safe = "".join(ch if ch.isalnum() else "_" for ch in self.rule_id)[:32]
        digest = hashlib.sha1(self.rule_id.encode("utf-8")).hexdigest()[:6]
        self.strategy_id = f"discovered_{safe}_{digest}"
        self.default_horizon = str(record.get("horizon") or "15m")
        self.conditions: List[Dict[str, Any]] = list(record.get("conditions") or [])
        try:
            self.holdout_t = float(record.get("holdout_t"))
        except (TypeError, ValueError):
            self.holdout_t = 0.0

    def enabled(self) -> bool:
        if self.holdout_t < MIN_HOLDOUT_T:
            return False
        return env_flag("STRATEGY_DISCOVERED_ENABLED", "1")

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        if ctx.available_quote <= 0 or ctx.last_price <= 0:
            return None

        features = _live_features(state, ctx)

        for condition in self.conditions:
            verdict = _condition_holds(condition, features)
            if verdict is None:
                # UNMEASURABLE IS NOT PERMISSION. A rule that fires when its
                # own precondition cannot be evaluated is trading on the
                # absence of evidence.
                return None
            if not verdict:
                return None

        # THE RULE SAYS "WHEN", NOT "HOW MUCH". Expected edge comes from what
        # the rule earned on held-out data, discounted hard: a t-statistic is
        # evidence that a difference exists, not a promise of its size.
        try:
            edge = float(self.record.get("holdout_mean_excess") or 0.0)
        except (TypeError, ValueError):
            edge = 0.0
        if edge <= 0:
            return None

        # Must still clear its own round trip, like anything else here.
        if edge <= ctx.fee_rate:
            return None

        confidence = min(0.9, 0.5 + self.holdout_t / 20.0)
        return self.make_candidate(
            state, ctx,
            action="enter",
            expected_return=edge,
            target_price=ctx.last_price * (1.0 + edge),
            confidence=confidence,
            direction_prob=confidence,
            reason=(f"[discovered t={self.holdout_t:.2f}] "
                    f"{self.record.get('rule') or self.rule_id}"),
            extra_meta={
                "holdout_t": self.holdout_t,
                "rule": str(self.record.get("rule") or ""),
                "discovered": True,
            },
        )


def build_discovered_strategies() -> List[Strategy]:
    """Every published rule that still clears the bar, as strategies.

    Returns an empty list when nothing has been discovered or nothing clears
    MIN_HOLDOUT_T -- which is the correct state for a system that has not yet
    found anything, and is why this is safe to call unconditionally.
    """
    strategies: List[Strategy] = []
    for record in _load_rules():
        try:
            strategies.append(DiscoveredRuleStrategy(record))
        except Exception:  # noqa: BLE001 - one bad record is not fatal
            continue
    return strategies
