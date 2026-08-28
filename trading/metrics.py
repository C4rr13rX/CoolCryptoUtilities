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
        return "[OK]"
    if level == FeedbackSeverity.WARNING:
        return "[WARN]"
    if level == FeedbackSeverity.CRITICAL:
        return "[CRIT]"
    return "[--]"


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


@dataclass
class ConfusionMatrixSummary:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def samples(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def report(self) -> Dict[str, float]:
        return classification_report(self.tp, self.fp, self.tn, self.fn)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "threshold": self.threshold,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "samples": self.samples,
        }
        payload.update(self.report())
        return payload


def confusion_from_scores(scores: Sequence[float], labels: Sequence[float | bool], threshold: float) -> ConfusionMatrixSummary:
    prob = np.asarray(list(scores), dtype=np.float64).reshape(-1)
    truth = np.asarray(list(labels), dtype=np.float64).reshape(-1)
    limit = min(prob.shape[0], truth.shape[0])
    if limit == 0:
        return ConfusionMatrixSummary(threshold=threshold, tp=0, fp=0, tn=0, fn=0)
    prob = prob[:limit]
    truth = truth[:limit]
    truths = truth > 0.5
    preds = prob > float(threshold)
    tp = int(np.sum(preds & truths))
    fp = int(np.sum(preds & ~truths))
    tn = int(np.sum(~preds & ~truths))
    fn = int(np.sum(~preds & truths))
    return ConfusionMatrixSummary(threshold=float(threshold), tp=tp, fp=fp, tn=tn, fn=fn)


def confusion_sweep(
    scores: Sequence[float],
    labels: Sequence[float | bool],
    thresholds: Sequence[float],
) -> Dict[float, ConfusionMatrixSummary]:
    results: Dict[float, ConfusionMatrixSummary] = {}
    for threshold in thresholds:
        summary = confusion_from_scores(scores, labels, threshold)
        results[float(threshold)] = summary
    return results


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
        # fetch_trades returns ORDER BY ts DESC, but pairing is causal: an exit
        # can only be matched against an entry already seen. Iterating
        # newest-first meant every exit arrived BEFORE its own entry, so the
        # keyed lookup always missed and the symbol fallback matched the exit
        # to whatever entry happened to be nearest the top of the list.
        #
        # Measured 2026-08-28 on the live ghost book: all 56 paired trades came
        # back with a NEGATIVE hold time (down to -105743s, an "exit" 29 hours
        # before its "entry") and realized_delta values that belonged to other
        # positions entirely -- -59.25 recorded against a -0.00075 trade.
        #
        # Sorting ascending restores causal order. It also makes the returned
        # sequence chronological, which the order-dependent statistics built on
        # top of it -- loss streaks and max drawdown -- silently require; they
        # were being computed on a time-REVERSED series.
        rows = sorted(rows, key=lambda r: float(r.get("ts") or 0.0))
        # symbol -> list of unmatched entries, oldest first
        open_entries: Dict[str, List[Dict[str, Any]]] = {}
        keyed_entries: Dict[str, Dict[str, Any]] = {}
        performances: List[TradePerformance] = []
        for row in rows:
            status = row.get("status")
            details = row.get("details") or {}
            symbol = row.get("symbol") or details.get("symbol") or "UNKNOWN"
            ts = float(row.get("ts") or details.get("timestamp") or 0.0)
            if status == "ghost-entry":
                entry = {
                    "symbol": symbol,
                    "entry_ts": float(details.get("entry_ts") or details.get("timestamp") or ts),
                    "entry_price": float(details.get("entry_price") or 0.0),
                    "expected_delta": float(details.get("expected_delta") or details.get("delta") or 0.0),
                    "route": details.get("route") or [],
                }
                trade_id = details.get("trade_id")
                if trade_id:
                    keyed_entries[str(trade_id)] = entry
                open_entries.setdefault(symbol, []).append(entry)
            elif status == "ghost-exit":
                entry: Optional[Dict[str, Any]] = None
                trade_id = details.get("trade_id")
                if trade_id and str(trade_id) in keyed_entries:
                    entry = keyed_entries.pop(str(trade_id))
                    pending = open_entries.get(symbol) or []
                    if entry in pending:
                        pending.remove(entry)
                elif open_entries.get(symbol):
                    # Same symbol, oldest still-open position: FIFO, and the
                    # entry is CONSUMED so two exits can never claim it.
                    entry = open_entries[symbol].pop(0)
                # An exit that carries its own entry data does not need a
                # matching entry row at all -- atf_static writes entry_ts and
                # entry_price onto the exit, and dropping those exits threw
                # away real outcomes (60 raw exits collapsed to 56 trades).
                own_entry_ts = details.get("entry_ts")
                if own_entry_ts is None:
                    # Rows written before atf_static published entry_ts at the
                    # top level still carry it on the embedded position, or can
                    # have it reconstructed from the recorded hold duration.
                    position = details.get("position")
                    if isinstance(position, dict) and position.get("entry_ts"):
                        own_entry_ts = position.get("entry_ts")
                    elif details.get("age_sec") is not None:
                        own_entry_ts = ts - float(details.get("age_sec") or 0.0)
                if entry is None and own_entry_ts is None:
                    continue
                entry = entry or {}
                entry_price = float(
                    details.get("entry_price") or entry.get("entry_price") or 0.0
                )
                exit_price = float(details.get("exit_price") or 0.0)
                performances.append(
                    TradePerformance(
                        symbol=symbol,
                        entry_ts=float(own_entry_ts or entry.get("entry_ts") or ts),
                        exit_ts=float(details.get("exit_ts") or details.get("timestamp") or ts),
                        profit=float(details.get("profit") or 0.0),
                        expected_delta=float(entry.get("expected_delta", 0.0)),
                        realized_delta=exit_price - entry_price,
                        # atf_static writes "reason"; trading/bot.py writes
                        # "exit_reason". Reading only the latter reported 58 of
                        # 60 real exits as "unspecified", which blinded every
                        # consumer to whether a loss was a stop-loss or a
                        # timer close.
                        reason=str(
                            details.get("exit_reason")
                            or details.get("reason")
                            or "unspecified"
                        ),
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
        # Profit factor: gross wins / gross losses.
        #
        # This was never returned, so every caller doing
        # ``summary.get("profit_factor", 1.0)`` silently read the constant 1.0
        # regardless of the trades. The ghost gate compares it against
        # MIN_GHOST_PROFIT_FACTOR to decide whether a strategy may trade real
        # money, so a genuinely profitable strategy (measured 2026-08-27:
        # true PF 2.077) was being judged on a placeholder -- and any
        # profit-factor threshold above 1.0 could never be satisfied by
        # anything.
        gross_win = float(sum(positives)) if positives else 0.0
        gross_loss = float(sum(negatives)) if negatives else 0.0
        if gross_loss > 0.0:
            profit_factor = gross_win / gross_loss
        elif gross_win > 0.0:
            # No losses yet. Report a large finite value rather than inf:
            # this number is JSON-serialised into snapshots and compared
            # against guardrails, and inf survives neither cleanly.
            profit_factor = 999.0
        else:
            profit_factor = 0.0

        return {
            "win_rate": float(win_rate),
            "avg_profit": float(mean_profit),
            "median_profit": float(median_profit),
            "kelly_fraction": float(kelly),
            "avg_duration_sec": float(avg_duration),
            "avg_expected_vs_realized_delta": float(delta_diff),
            "profit_factor": float(profit_factor),
            "payoff_ratio": float(payoff_ratio),
            "gross_win": gross_win,
            "gross_loss": gross_loss,
        }
