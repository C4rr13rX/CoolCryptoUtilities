from __future__ import annotations

import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from db import TradingDatabase, get_db
from services.logging_utils import log_message


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
    #: Which strategy produced this round trip. Empty when the trade log did
    #: not attribute one -- an unattributed trade belongs to no strategy's
    #: record and must never be counted toward one.
    strategy_id: str = ""
    #: Fractional return, ``exit_price / entry_price - 1``. ``profit`` is in
    #: USD and therefore scales with the clip; a risk limit expressed as a
    #: percentage of the position -- a stop-loss -- can only be compared
    #: against a percentage. None means the round trip did not record a usable
    #: entry price, which is NOT the same as a zero return and must never be
    #: averaged in as one.
    return_pct: Optional[float] = None

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

    @staticmethod
    def _row_strategy_id(details: Dict[str, Any]) -> str:
        """Which strategy a ghost row belongs to, or "" if it does not say.

        Two writers produce ghost rows in different shapes. atf_static writes
        a self-contained exit carrying ``strategy_id`` at the top level;
        trading/bot.py writes an entry whose strategy lives on the embedded
        ``bus_plan``, and an exit that names no strategy at all. Reading only
        the top-level key attributed 18 of 56 trades and left the rest
        anonymous, so an exit's strategy is recovered from its own entry.
        """
        if not isinstance(details, dict):
            return ""
        sid = details.get("strategy_id")
        if not sid:
            plan = details.get("bus_plan")
            if isinstance(plan, dict):
                sid = plan.get("strategy_id")
        if not sid:
            position = details.get("position")
            if isinstance(position, dict):
                sid = position.get("strategy_id")
        return str(sid or "").strip()

    def ghost_trade_snapshot(
        self,
        *,
        limit: int = 500,
        lookback_sec: Optional[float] = None,
        strategy_id: Optional[str] = None,
    ) -> List[TradePerformance]:
        """Paired ghost round trips, optionally narrowed to one strategy.

        ``strategy_id`` filters the returned book to trades that strategy
        actually produced. Pairing still runs over ALL rows first: an exit
        recovers its strategy from its own entry, so filtering the rows before
        pairing would orphan every bot-written exit and silently drop it.

        ``limit`` counts trading_ops ROWS, not paired trades, and when a
        ``lookback_sec`` window is given the window -- not the row cap -- is
        what defines the book. Letting a row cap bound a time-bounded window
        truncates it at the OLD end (``fetch_trades`` is ORDER BY ts DESC), and
        because pairing is causal that does not shrink the book evenly: it
        keeps the newest entries, whose exits have not happened yet, and cuts
        the older entries that the exits inside the window need to pair
        against. Every one of those exits is then dropped as an orphan.

        Measured 2026-09-03 on the live 48h ghost book -- 1196 rows, 1062
        entries against 134 exits, 302 entries in one hour alone -- the newest
        500 rows covered only ~6 hours:

            limit=500     atf_static  7 paired, net -0.16220 | pooled  28, -0.01032
            limit>=1500   atf_static 36 paired, net +0.53014 | pooled 134, +0.68025

        The 500-row book is the one the live gate was reading. It reported
        ``insufficient_samples`` for atf_static and ``negative_margin`` for the
        pooled book -- i.e. ``ghost_validation_block``, the reason live trading
        was shut -- off a sample that excluded 80% of the completed round trips
        and flipped the sign of the P&L on both.

        So the cap is raised to cover the window, and a book that still hits it
        says so instead of quietly reporting a truncated record as the whole
        one. ``GHOST_SNAPSHOT_MAX_ROWS`` bounds the memory; at the measured
        rate (~1200 rows/48h) the default is ~40x headroom.
        """
        since_ts = time.time() - lookback_sec if lookback_sec else None
        effective_limit = int(limit)
        if since_ts is not None:
            try:
                window_cap = int(os.getenv("GHOST_SNAPSHOT_MAX_ROWS", "50000"))
            except (TypeError, ValueError):
                window_cap = 50000
            effective_limit = max(effective_limit, max(1, window_cap))
        rows = self.db.fetch_trades(
            limit=effective_limit,
            statuses=["ghost-entry", "ghost-exit", "ghost"],
            wallets=["ghost"],
            since_ts=since_ts,
        )
        if since_ts is not None and len(rows) >= effective_limit:
            # Truncated after all. Say it: a silently short book is exactly the
            # failure this cap raise exists to stop.
            log_message(
                "metrics",
                "ghost_trade_snapshot hit its row cap (%d rows, window %ss) -- "
                "the book is TRUNCATED and every statistic built on it is short"
                % (len(rows), lookback_sec),
                severity="warning",
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
        # Ghost-exit rows refused below because a real buy stands behind them.
        real_money_exits: List[str] = []
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
                    "strategy_id": self._row_strategy_id(details),
                }
                trade_id = details.get("trade_id")
                if trade_id:
                    keyed_entries[str(trade_id)] = entry
                open_entries.setdefault(symbol, []).append(entry)
            elif status == "ghost-exit":
                # A round trip whose ENTRY was paid for on chain is not ghost
                # evidence, whatever status the row carries.
                #
                # ``entry_tx_hash`` is only ever set by the two paths that
                # spend real money -- the live entry in ``trading/bot.py`` and
                # ``_adopt_onchain_holding`` -- so its presence means tokens
                # were actually bought. When such a position is then marked out
                # against the feed instead of sold, the row records a profit
                # for a sale that never happened, and the position stays in the
                # wallet.
                #
                # Measured 2026-09-04, four such rows exist, all written in one
                # window on 09-03 before ``live_position_cannot_exit_in_simulation``
                # closed the writer (23593e2, 182bd8a). Each carries
                # ``fill_source="simulated"`` and an EMPTY ``tx_hash``:
                #
                #   09-03 17:40:18Z  CBETH-USDC   -0.000005
                #   09-03 19:12:16Z  BSTONK-USDC  -0.142865
                #   09-03 20:03:37Z  CBBTC-USDC   +0.000267
                #   09-03 20:38:54Z  CBETH-USDC   -0.003011
                #
                # The BSTONK line is the whole of atf_static's tail. Its entry
                # 0xcd6fb05c92af5077f9be707727c1d57e0ac1dfedd54d9f87e860376b96ea560b
                # bought 360.264243225392976659 BSTONK for 0.750000 USDC, and
                # ``balanceOf`` on base still returns exactly
                # 360264243225392976659 -- not one wei was sold. Scored as a
                # ghost trade it puts atf_static's ES95 at 0.10213 against a
                # 0.10 guardrail; without it the same 26 real ghost round trips
                # read 0.01723. The gate was refusing live trading on a loss
                # that never occurred.
                #
                # The same four were annulled in ``trade_outcomes`` by
                # scripts/annul_unsettled_live_exits.py, but that book is not
                # this one: the live gate reads ``trading_ops`` through here, so
                # the correction never reached it. This is the reader-side half.
                #
                # The rule is profit-blind -- it drops one WIN and three losses
                # -- and it keys on a fact about the chain, not on the outcome.
                if len(str(details.get("entry_tx_hash") or "")) == 66:
                    real_money_exits.append(
                        "%s %+.6f" % (symbol, float(details.get("profit") or 0.0))
                    )
                    continue
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
                        strategy_id=(
                            self._row_strategy_id(details)
                            or str(entry.get("strategy_id") or "")
                        ),
                        # Both prices are already in hand here, so the return
                        # costs nothing to carry and is the only form in which
                        # a tail can be compared against a stop-loss.
                        return_pct=(
                            (exit_price / entry_price - 1.0)
                            if entry_price > 0 and exit_price > 0
                            else None
                        ),
                    )
                )
        if real_money_exits:
            # Say it out loud. A silently shorter book is how a correction gets
            # mistaken for a measurement.
            log_message(
                "metrics",
                "ghost_trade_snapshot refused %d exit row(s) whose ENTRY settled "
                "on chain -- a real buy marked out against the feed is not a "
                "ghost trade, and the position may still be held: %s"
                % (len(real_money_exits), ", ".join(real_money_exits)),
                severity="warning",
            )
        if strategy_id is not None:
            wanted = str(strategy_id).strip()
            performances = [t for t in performances if t.strategy_id == wanted]
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
