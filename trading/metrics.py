from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from db import TradingDatabase, get_db


class MetricStage:
    TRAINING = "training"
    GHOST_TRADING = "ghost_trading"
    LIVE_TRADING = "live_trading"
    DATA_STREAM = "data_stream"
    MODEL_FINE_TUNE = "model_fine_tune"
    NEWS = "news_enrichment"
    PIPELINE = "pipeline"
    SAVINGS = "savings"


class FeedbackSeverity:
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


def status_light(severity: str) -> str:
    level = (severity or "").lower()
    if level == FeedbackSeverity.INFO:
        return "🟢"
    if level == FeedbackSeverity.WARNING:
        return "🟠"
    if level == FeedbackSeverity.CRITICAL:
        return "🔴"
    return "⚪"


@dataclass
class TradePerformance:
    symbol: str
    entry_ts: float
    exit_ts: float
    profit: float
    expected_delta: float
    realized_delta: float
    reason: str
    route: Sequence[str]

    @property
    def duration(self) -> float:
        return max(0.0, self.exit_ts - self.entry_ts)


def _safe_array(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64).flatten()
    if arr.size == 0:
        return np.zeros(1, dtype=np.float64)
    mask = np.isfinite(arr)
    if not mask.all():
        arr = arr[mask]
    if arr.size == 0:
        return np.zeros(1, dtype=np.float64)
    return arr


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    if den == 0:
        return default
    return float(num) / float(den)


def classification_report(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    tp_f = float(tp)
    fp_f = float(fp)
    tn_f = float(tn)
    fn_f = float(fn)

    precision = _safe_ratio(tp_f, tp_f + fp_f)
    recall = _safe_ratio(tp_f, tp_f + fn_f)
    specificity = _safe_ratio(tn_f, tn_f + fp_f)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    balanced_acc = (recall + specificity) / 2.0
    false_positive_rate = 1.0 - specificity
    false_negative_rate = _safe_ratio(fn_f, fn_f + tp_f)

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "balanced_accuracy": balanced_acc,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


def distribution_report(values: Iterable[float]) -> Dict[str, float]:
    arr = _safe_array(values)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "value_at_risk_95": 0.0,
            "expected_shortfall_95": 0.0,
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if arr.size > 2:
        skew = float(((arr - mean) ** 3).mean() / (std ** 3 + 1e-9))
        kurt = float(((arr - mean) ** 4).mean() / (std ** 4 + 1e-9)) - 3.0
    else:
        skew = 0.0
        kurt = 0.0
    p05 = float(np.percentile(arr, 5))
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    var95 = float(-np.percentile(arr, 5))
    es95 = float(-arr[arr <= np.percentile(arr, 5)].mean()) if (arr <= np.percentile(arr, 5)).any() else var95
    return {
        "mean": mean,
        "std": std,
        "skewness": skew,
        "kurtosis": kurt,
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "value_at_risk_95": var95,
        "expected_shortfall_95": es95,
    }


def kelly_fraction(prob_win: float, payoff_ratio: float) -> float:
    b = float(payoff_ratio)
    p = float(prob_win)
    q = 1.0 - p
    if b <= -1.0:
        return 0.0
    numerator = (b + 1.0) * p - 1.0
    denominator = b
    if denominator == 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


class MetricsCollector:
    """
    Thin wrapper around TradingDatabase metrics + feedback tables that also
    computes higher-order statistics for trading outcomes.
    """

    def __init__(self, db: Optional[TradingDatabase] = None) -> None:
        self.db = db or get_db()

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def record(
        self,
        stage: str,
        metrics: Dict[str, Any],
        *,
        category: str = "general",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.db.record_metrics(stage=stage, metrics=metrics, category=category, meta=meta)

    def feedback(
        self,
        source: str,
        *,
        severity: str,
        label: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.db.record_feedback_event(source=source, severity=severity, label=label, details=details)
        light = status_light(severity)
        print(f"[feedback] {light} {source}:{label} -> {details or {}}")

    # ------------------------------------------------------------------
    # Derived analytics
    # ------------------------------------------------------------------

    def ghost_trade_snapshot(
        self,
        *,
        limit: int = 500,
        lookback_sec: Optional[float] = None,
    ) -> List[TradePerformance]:
        since_ts = time.time() - lookback_sec if lookback_sec else None
        rows = self.db.fetch_trades(
            limit=limit,
            statuses=["ghost-entry", "ghost-exit", "ghost"],
            wallets=["ghost"],
            since_ts=since_ts,
        )
        entries: Dict[str, Dict[str, Any]] = {}
        performances: List[TradePerformance] = []
        for row in rows:
            status = row.get("status")
            details = row.get("details") or {}
            symbol = row.get("symbol") or details.get("symbol") or "UNKNOWN"
            ts = float(row.get("ts") or details.get("timestamp") or 0.0)
            key = details.get("trade_id") or f"{symbol}-{int(details.get('entry_ts') or ts)}"
            if status == "ghost-entry":
                entries[key] = {
                    "symbol": symbol,
                    "entry_ts": float(details.get("timestamp") or ts),
                    "expected_delta": float(details.get("expected_delta") or details.get("delta") or 0.0),
                    "route": details.get("route") or [],
                }
            elif status == "ghost-exit":
                entry = entries.get(key)
                entry_ts = float(details.get("entry_ts") or ts)
                if entry is None:
                    # best-effort match by symbol
                    entry = next((v for v in entries.values() if v.get("symbol") == symbol), None)
                if entry:
                    performances.append(
                        TradePerformance(
                            symbol=symbol,
                            entry_ts=float(entry.get("entry_ts", entry_ts)),
                            exit_ts=float(details.get("timestamp") or ts),
                            profit=float(details.get("profit") or 0.0),
                            expected_delta=float(entry.get("expected_delta", 0.0)),
                            realized_delta=float(details.get("exit_price", 0.0)) - float(details.get("entry_price", 0.0)),
                            reason=str(details.get("exit_reason") or "unspecified"),
                            route=entry.get("route") or [],
                        )
                    )
        return performances

    def aggregate_trade_metrics(self, trades: Sequence[TradePerformance]) -> Dict[str, float]:
        if not trades:
            return {
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "median_profit": 0.0,
                "kelly_fraction": 0.0,
                "avg_duration_sec": 0.0,
                "avg_expected_vs_realized_delta": 0.0,
            }
        profits = [float(t.profit) for t in trades]
        wins = [p for p in profits if p > 0]
        expected_delta = [float(t.expected_delta) for t in trades]
        realised_delta = [float(t.realized_delta) for t in trades]
        durations = [float(t.duration) for t in trades if math.isfinite(t.duration)]

        mean_profit = float(statistics.mean(profits))
        median_profit = float(statistics.median(profits))
        win_rate = len(wins) / len(trades)

        positives = [abs(p) for p in profits if p > 0]
        negatives = [abs(p) for p in profits if p < 0]
        if positives:
            pos_mean = statistics.mean(positives)
            neg_mean = statistics.mean(negatives) if negatives else pos_mean
            payoff_ratio = pos_mean / (neg_mean or 1e-9)
        else:
            payoff_ratio = 0.0
        kelly = kelly_fraction(win_rate, payoff_ratio)

        if expected_delta and realised_delta:
            delta_diff = statistics.mean(
                (real - exp) for real, exp in zip(realised_delta, expected_delta)
            )
        else:
            delta_diff = 0.0
        avg_duration = statistics.mean(durations) if durations else 0.0
        return {
            "win_rate": float(win_rate),
            "avg_profit": float(mean_profit),
            "median_profit": float(median_profit),
            "kelly_fraction": float(kelly),
            "avg_duration_sec": float(avg_duration),
            "avg_expected_vs_realized_delta": float(delta_diff),
        }
