from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from db import TradingDatabase, get_db
from model_definition import (
    ExponentialDecay,
    StateSaver,
    TimeFeatureLayer,
    build_multimodal_model,
    gaussian_nll_loss,
    zero_loss,
    _compute_net_margin,
    _identity,
    _slice_price_log_var,
    _slice_price_mu,
)
from trading.data_loader import HistoricalDataLoader
from trading.optimizer import BayesianBruteForceOptimizer
from trading.metrics import (
    MetricStage,
    MetricsCollector,
    classification_report,
    distribution_report,
)


CUSTOM_OBJECTS = {
    "ExponentialDecay": ExponentialDecay,
    "TimeFeatureLayer": TimeFeatureLayer,
    "gaussian_nll_loss": gaussian_nll_loss,
    "zero_loss": zero_loss,
    "_slice_price_mu": _slice_price_mu,
    "_slice_price_log_var": _slice_price_log_var,
    "_identity": _identity,
    "_compute_net_margin": _compute_net_margin,
}


class TrainingPipeline:
    """
    Coordinates model training, ghost validation, and promotion of candidate models.
    Keeps computations light by using CPU-friendly architectures and synthetic fallbacks.
    """

    def __init__(
        self,
        *,
        db: Optional[TradingDatabase] = None,
        optimizer: Optional[BayesianBruteForceOptimizer] = None,
        model_dir: Optional[Path] = None,
        promotion_threshold: float = 0.65,
    ) -> None:
        self.db = db or get_db()
        self.optimizer = optimizer or BayesianBruteForceOptimizer(
            {
                "learning_rate": (1e-5, 5e-4),
                "epochs": (1.0, 4.0),
            }
        )
        self.model_dir = model_dir or Path(os.getenv("MODEL_DIR", "models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.promotion_threshold = promotion_threshold

        self.min_ghost_trades = int(os.getenv("MIN_GHOST_TRADES_FOR_PROMOTION", "25"))
        self.max_false_positive_rate = float(os.getenv("MAX_FALSE_POSITIVE_RATE", "0.15"))
        self.min_ghost_win_rate = float(os.getenv("MIN_GHOST_WIN_RATE", "0.55"))
        self.min_realized_margin = float(os.getenv("MIN_REALIZED_MARGIN", "0.0"))

        self.window_size = 60
        self.sent_seq_len = 24
        self.tech_count = 12
        self.data_loader = HistoricalDataLoader()
        self.iteration: int = 0
        self.active_accuracy: float = 0.0
        self._load_state()
        self._active_model: Optional[tf.keras.Model] = None
        self.metrics = MetricsCollector(self.db)
        self.focus_lookback_sec = int(os.getenv("GHOST_FOCUS_LOOKBACK_SEC", "172800"))
        self.focus_max_assets = int(os.getenv("GHOST_FOCUS_MAX_ASSETS", "6"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_active_model(self) -> Optional[tf.keras.Model]:
        if self._active_model is not None:
            return self._active_model
        path = self.model_dir / "active_model.keras"
        if not path.exists():
            return None
        try:
            self._active_model = tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)
            return self._active_model
        except Exception as exc:
            print(f"[training] failed to load active model ({exc}); removing corrupted artifact.")
            path.unlink(missing_ok=True)
            return None

    def ensure_active_model(self) -> tf.keras.Model:
        model = self.load_active_model()
        if model is not None:
            return model
        print("[training] no active model found; building a fresh baseline.")
        model, headline_vec, full_vec, losses, loss_weights = build_multimodal_model(
            window_size=self.window_size,
            tech_count=self.tech_count,
            sent_seq_len=self.sent_seq_len,
            asset_vocab_size=self.data_loader.asset_vocab_size,
        )
        self._adapt_vectorizers(headline_vec, full_vec)
        path = self.model_dir / "active_model.keras"
        model.save(path, include_optimizer=False)
        self._active_model = model
        return model

    def train_candidate(self) -> Optional[Dict[str, Any]]:
        proposal = self.optimizer.propose()
        lr = float(proposal.get("learning_rate", 3e-4))
        epochs = max(1, int(round(proposal.get("epochs", 2))))

        pending_iteration = self.iteration + 1

        focus_assets, focus_stats = self._ghost_focus_assets()

        news_items = getattr(self.data_loader, "news_items", []) or []
        if news_items:
            sentiment_counts: Dict[str, int] = {}
            token_coverage: set[str] = set()
            for item in news_items:
                sentiment = str(item.get("sentiment", "neutral")).lower()
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                for token in item.get("tokens") or []:
                    token_coverage.add(str(token).upper())
            total_news = len(news_items)
            news_metrics = {
                "news_items_total": total_news,
                "news_token_coverage": len(token_coverage),
                "news_positive_ratio": sentiment_counts.get("positive", 0) / total_news,
                "news_negative_ratio": sentiment_counts.get("negative", 0) / total_news,
            }
            self.metrics.record(
                MetricStage.NEWS,
                news_metrics,
                category="training_news",
                meta={
                    "iteration": self.iteration,
                    "unique_tokens": list(sorted(token_coverage))[:32],
                },
            )

        inputs, targets = self._prepare_dataset(batch_size=32, dataset_label="full")
        if inputs is None or targets is None:
            self.iteration = pending_iteration
            self._save_state()
            print("[training] insufficient data for candidate training; skipping this cycle.")
            return {"iteration": self.iteration, "status": "skipped", "score": None}

        self.iteration = pending_iteration
        self._save_state()

        if self.active_accuracy >= 0.99 and self._active_model is not None:
            print("[training] active model already at ≥99% accuracy; pausing candidate search.")
            return {
                "iteration": self.iteration,
                "status": "paused",
                "score": self.active_accuracy,
                "message": "active model at target accuracy",
            }

        model, headline_vec, full_vec, losses, loss_weights = build_multimodal_model(
            window_size=self.window_size,
            tech_count=self.tech_count,
            sent_seq_len=self.sent_seq_len,
            asset_vocab_size=self.data_loader.asset_vocab_size,
        )
        self._adapt_vectorizers(headline_vec, full_vec)

        try:
            model.optimizer.learning_rate.assign(lr)
        except Exception:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=losses,
                loss_weights=loss_weights,
                metrics={"price_dir": ["accuracy"]},
            )

        callbacks = [StateSaver()]
        input_order = [tensor.name.split(":")[0] for tensor in model.inputs]
        output_order = [
            "exit_conf",
            "price_mu",
            "price_log_var",
            "price_dir",
            "net_margin",
            "net_pnl",
            "tech_recon",
            "price_gaussian",
        ]
        input_tensors = tuple(inputs[name] for name in input_order)
        target_tensors = tuple(targets[name] for name in output_order)
        train_ds = (
            tf.data.Dataset.from_tensor_slices((input_tensors, target_tensors))
            .batch(16)
            .prefetch(tf.data.AUTOTUNE)
        )
        history = model.fit(train_ds, epochs=epochs, verbose=0, callbacks=callbacks)

        score = float(history.history.get("price_dir_accuracy", [0.0])[-1])
        self.optimizer.update({"learning_rate": lr, "epochs": epochs}, score)
        eval_ds = (
            tf.data.Dataset.from_tensor_slices((input_tensors, target_tensors))
            .batch(32)
            .prefetch(tf.data.AUTOTUNE)
        )
        evaluation = self._evaluate_candidate(model, eval_ds, targets)
        focus_history = self._apply_focus_adaptation(model, focus_assets)
        if focus_history is not None:
            evaluation = self._evaluate_candidate(model, eval_ds, targets)
            evaluation.update(focus_history)

        version = f"candidate-{int(time.time())}"
        path = self.model_dir / f"{version}.keras"
        model.save(path, include_optimizer=False)
        self.db.register_model_version(version=version, metrics={"score": score}, path=str(path), activate=False)

        result = {
            "iteration": self.iteration,
            "version": version,
            "score": score,
            "path": str(path),
            "params": {"learning_rate": lr, "epochs": epochs},
            "evaluation": evaluation,
            "status": "trained",
        }
        evaluation_meta = {
            "iteration": self.iteration,
            "params": {"learning_rate": lr, "epochs": epochs},
            "version": version,
            "focus_assets": focus_assets,
            "ghost_feedback": focus_stats,
        }
        training_metrics = {
            "candidate_score": score,
            "dir_accuracy": self._safe_float(evaluation.get("dir_accuracy", 0.0)),
            "price_dir_precision": self._safe_float(evaluation.get("precision", 0.0)),
            "price_dir_recall": self._safe_float(evaluation.get("recall", 0.0)),
            "price_dir_f1": self._safe_float(evaluation.get("f1_score", 0.0)),
            "profit_factor": self._safe_float(evaluation.get("profit_factor", 0.0)),
            "kelly_fraction": self._safe_float(evaluation.get("kelly_fraction", 0.0)),
            "ghost_win_rate": self._safe_float(evaluation.get("ghost_win_rate", 0.0)),
            "ghost_pred_margin": self._safe_float(evaluation.get("ghost_pred_margin", 0.0)),
            "ghost_realized_margin": self._safe_float(evaluation.get("ghost_realized_margin", 0.0)),
            "false_positive_rate": self._safe_float(evaluation.get("false_positive_rate", 0.0)),
        }
        self.metrics.record(
            MetricStage.TRAINING,
            training_metrics,
            category="candidate",
            meta=evaluation_meta,
        )

        promote = score >= self.promotion_threshold
        gating_reason: Optional[str] = None
        if promote:
            if not evaluation:
                gating_reason = "no evaluation metrics available"
            else:
                ghost_trades = int(evaluation.get("ghost_trades", 0))
                if ghost_trades < self.min_ghost_trades:
                    gating_reason = (
                        f"ghost trades {ghost_trades} below minimum {self.min_ghost_trades}"
                    )
                else:
                    fp_rate = float(evaluation.get("false_positive_rate", 0.0))
                    win_rate = float(evaluation.get("ghost_win_rate", 0.0))
                    realized_margin = float(evaluation.get("ghost_realized_margin", 0.0))
                    if fp_rate > self.max_false_positive_rate:
                        gating_reason = (
                            f"false positive rate {fp_rate:.3f} exceeds limit {self.max_false_positive_rate:.3f}"
                        )
                    elif win_rate < self.min_ghost_win_rate:
                        gating_reason = (
                            f"ghost win rate {win_rate:.3f} below minimum {self.min_ghost_win_rate:.3f}"
                        )
                    elif realized_margin < self.min_realized_margin:
                        gating_reason = (
                            f"realized margin {realized_margin:.6f} below minimum {self.min_realized_margin:.6f}"
                        )
                    elif self.active_accuracy and evaluation.get("dir_accuracy", 0.0) < self.active_accuracy + 0.01:
                        gating_reason = (
                            "retaining existing live model (%.3f) to gather more data before replacement"
                            % self.active_accuracy
                        )
        if gating_reason:
            promote = False
            print(f"[training] promotion deferred: {gating_reason}. Continuing candidate search.")
        if promote:
            self.promote_candidate(path, score=score, metadata=result, evaluation=evaluation)
        else:
            self._print_ghost_summary(evaluation)
            if score < self.promotion_threshold:
                print(f"[training] candidate score {score:.3f} below promotion threshold {self.promotion_threshold:.3f}.")
            else:
                print(
                    "[training] candidate retained for further evaluation despite score %.3f (promotion criteria not met)."
                    % score
                )

        self._save_state()
        return result

    def promote_candidate(
        self,
        path: Path,
        *,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
        evaluation: Optional[Dict[str, float]] = None,
    ) -> None:
        active_path = self.model_dir / "active_model.keras"
        print(f"[training] promoting candidate {path.name} (score={score:.3f}) to active deployment.")
        tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False).save(active_path, include_optimizer=False)
        self.db.register_model_version(
            version=f"active-{int(time.time())}", metrics={"score": score}, path=str(active_path), activate=True
        )
        self._active_model = tf.keras.models.load_model(active_path, custom_objects=CUSTOM_OBJECTS, compile=False)
        self.active_accuracy = float(evaluation.get("dir_accuracy", score)) if evaluation else score
        if metadata:
            summary = metadata.copy()
            if evaluation:
                summary["evaluation"] = {k: self._safe_float(v) for k, v in evaluation.items()}
            self.db.log_trade(
                wallet="system",
                chain="meta",
                symbol="MODEL",
                action="promote",
                status="success",
                details=summary,
            )
        if evaluation:
            self._print_ghost_summary(evaluation)
        self._save_state()

    # ------------------------------------------------------------------
    # Dataset creation
    # ------------------------------------------------------------------

    def _prepare_dataset(
        self,
        batch_size: int = 32,
        *,
        focus_assets: Optional[Sequence[str]] = None,
        dataset_label: str = "full",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        inputs, targets = self.data_loader.build_dataset(
            window_size=self.window_size,
            sent_seq_len=self.sent_seq_len,
            tech_count=self.tech_count,
            focus_assets=focus_assets,
        )
        if inputs is not None and targets is not None:
            sample_count = int(inputs["price_vol_input"].shape[0])
            asset_ids = inputs["asset_id_input"].reshape(-1)
            asset_diversity = int(len(np.unique(asset_ids)))
            price_slice = inputs["price_vol_input"][..., 0]
            price_volatility = float(np.std(price_slice, axis=1).mean()) if sample_count else 0.0
            headlines_flat = inputs["headline_text"].reshape(-1)
            news_coverage = float(
                np.mean([1.0 if str(text).strip() else 0.0 for text in headlines_flat])
            )
            margin_arr = targets["net_margin"].reshape(-1)
            margin_stats = distribution_report(margin_arr)
            dataset_metrics = {
                "samples": sample_count,
                "asset_diversity": asset_diversity,
                "avg_price_volatility": price_volatility,
                "news_coverage_ratio": news_coverage,
            }
            dataset_metrics.update({f"net_margin_{k}": v for k, v in margin_stats.items()})
            self.metrics.record(
                MetricStage.TRAINING,
                dataset_metrics,
                category=f"dataset_{dataset_label}",
                meta={
                    "iteration": self.iteration,
                    "focus_assets": list(focus_assets or []),
                },
            )
            inputs["headline_text"] = tf.convert_to_tensor(inputs["headline_text"], dtype=tf.string)
            inputs["full_text"] = tf.convert_to_tensor(inputs["full_text"], dtype=tf.string)
            inputs["asset_id_input"] = tf.convert_to_tensor(inputs["asset_id_input"], dtype=tf.int32)
            return inputs, targets
        return None, None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _adapt_vectorizers(self, headline_vec, full_vec) -> None:
        texts = self.data_loader.sample_texts(limit=256)
        dataset = tf.data.Dataset.from_tensor_slices(texts)
        headline_vec.adapt(dataset)
        full_vec.adapt(dataset)

    def _load_state(self) -> None:
        try:
            state = self.db.load_state()
        except Exception:
            state = {}
        if isinstance(state, dict):
            tp_state = state.get("training_pipeline") or {}
            if isinstance(tp_state, dict):
                self.iteration = int(tp_state.get("iteration", 0))
                self.active_accuracy = float(tp_state.get("active_accuracy", self.active_accuracy))
                optimizer_state = tp_state.get("optimizer")
                if optimizer_state:
                    self.optimizer.set_state(optimizer_state)

    def _save_state(self) -> None:
        try:
            state = self.db.load_state()
        except Exception:
            state = {}
        if not isinstance(state, dict):
            state = {}
        state["training_pipeline"] = {
            "iteration": int(self.iteration),
            "optimizer": self.optimizer.get_state(),
            "active_accuracy": float(self.active_accuracy),
        }
        try:
            self.db.save_state(state)
        except Exception:
            pass

    def _safe_float(self, value: Any) -> float:
        return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))

    def _print_ghost_summary(self, evaluation: Optional[Dict[str, float]]) -> None:
        if not evaluation:
            print(f"[training] ghost summary (iteration {self.iteration}): no evaluation available")
            return
        print(
            "[training] ghost summary (iteration %d): dir_acc=%.3f, ghost_trades=%d, pred_margin=%.6f, realized_margin=%.6f, win_rate=%.3f, TP=%d, FP=%d, TN=%d, FN=%d"
            % (
                self.iteration,
                self._safe_float(evaluation.get("dir_accuracy", 0.0)),
                int(evaluation.get("ghost_trades", 0)),
                self._safe_float(evaluation.get("ghost_pred_margin", 0.0)),
                self._safe_float(evaluation.get("ghost_realized_margin", 0.0)),
                self._safe_float(evaluation.get("ghost_win_rate", 0.0)),
                int(evaluation.get("true_positives", 0)),
                int(evaluation.get("false_positives", 0)),
                int(evaluation.get("true_negatives", 0)),
                int(evaluation.get("false_negatives", 0)),
            )
        )

    def _evaluate_candidate(
        self, model: tf.keras.Model, dataset: tf.data.Dataset, targets: Dict[str, Any]
    ) -> Dict[str, float]:
        try:
            predictions = model.predict(dataset, verbose=0)
        except Exception:
            return {}
        if not isinstance(predictions, (list, tuple)) or len(predictions) < 8:
            return {}

        exit_conf_pred, price_mu_pred, price_log_var_pred, price_dir_pred, net_margin_pred, net_pnl_pred, _, _ = predictions

        price_dir_true = targets["price_dir"].reshape(-1)
        price_dir_scores = price_dir_pred.reshape(-1)
        price_dir_labels = price_dir_scores > 0.5
        dir_accuracy = self._safe_float(np.mean(price_dir_labels == (price_dir_true > 0.5)))

        net_margin_true = targets["net_margin"].reshape(-1)
        net_margin_pred_flat = net_margin_pred.reshape(-1)

        ghost_trades = int(np.sum(price_dir_labels))
        if ghost_trades > 0:
            ghost_pred_margin = self._safe_float(np.mean(net_margin_pred_flat[price_dir_labels]))
            ghost_real_margin = self._safe_float(np.mean(net_margin_true[price_dir_labels]))
            ghost_win_rate = self._safe_float(np.mean(net_margin_true[price_dir_labels] > 0))
        else:
            ghost_pred_margin = 0.0
            ghost_real_margin = 0.0
            ghost_win_rate = 0.0

        summary = {
            "dir_accuracy": dir_accuracy,
            "avg_exit_conf": self._safe_float(np.mean(exit_conf_pred)),
            "avg_mu_pred": self._safe_float(np.mean(price_mu_pred)),
            "mu_std": self._safe_float(np.std(price_mu_pred)),
            "avg_margin_pred": self._safe_float(np.mean(net_margin_pred_flat)),
            "avg_realized_margin": self._safe_float(np.mean(net_margin_true)),
            "ghost_trades": ghost_trades,
            "ghost_pred_margin": ghost_pred_margin,
            "ghost_realized_margin": ghost_real_margin,
            "ghost_win_rate": ghost_win_rate,
            "total_windows": int(price_dir_scores.shape[0]),
            "true_positives": int(np.sum((price_dir_true > 0.5) & (price_dir_labels == True))),
            "true_negatives": int(np.sum((price_dir_true <= 0.5) & (price_dir_labels == False))),
            "false_positives": int(np.sum((price_dir_true <= 0.5) & (price_dir_labels == True))),
            "false_negatives": int(np.sum((price_dir_true > 0.5) & (price_dir_labels == False))),
            "execution_bias": self._execution_bias(),
        }
        negatives = summary["false_positives"] + summary["true_negatives"]
        summary["false_positive_rate"] = self._safe_float(
            summary["false_positives"] / max(1, negatives)
        )
        class_metrics = classification_report(
            summary["true_positives"],
            summary["false_positives"],
            summary["true_negatives"],
            summary["false_negatives"],
        )
        summary.update(class_metrics)

        real_dist = distribution_report(net_margin_true)
        pred_dist = distribution_report(net_margin_pred_flat)
        summary.update({f"real_margin_{k}": self._safe_float(v) for k, v in real_dist.items()})
        summary.update({f"pred_margin_{k}": self._safe_float(v) for k, v in pred_dist.items()})
        mu_dist = distribution_report(price_mu_pred.reshape(-1))
        summary.update({f"price_mu_{k}": self._safe_float(v) for k, v in mu_dist.items()})

        positive_returns = net_margin_true[net_margin_true > 0]
        negative_returns = net_margin_true[net_margin_true < 0]
        if negative_returns.size > 0:
            profit_factor = self._safe_float(positive_returns.sum() / abs(negative_returns.sum()))
        else:
            profit_factor = self._safe_float(positive_returns.sum())
        summary["profit_factor"] = profit_factor

        if positive_returns.size > 0:
            avg_positive = float(np.mean(positive_returns))
            avg_negative = float(np.mean(np.abs(negative_returns))) if negative_returns.size > 0 else avg_positive
            payoff_ratio = self._safe_float(avg_positive / max(avg_negative, 1e-9))
            kelly = self._safe_float(
                max(
                    0.0,
                    min(
                        1.0,
                        ((payoff_ratio + 1.0) * summary["ghost_win_rate"] - 1.0) / max(payoff_ratio, 1e-9),
                    ),
                )
            )
        else:
            payoff_ratio = 0.0
            kelly = 0.0
        summary["payoff_ratio"] = payoff_ratio
        summary["kelly_fraction"] = kelly
        return summary

    def _execution_bias(self, window: int = 50) -> float:
        fills = self.db.fetch_trade_fills(limit=window)
        ratios: List[float] = []
        for fill in fills:
            expected = float(fill.get("expected_amount") or 0.0)
            executed = float(fill.get("executed_amount") or 0.0)
            if expected > 0:
                ratios.append(executed / expected)
        if not ratios:
            return 1.0
        return float(np.clip(np.mean(ratios), 0.1, 2.0))

    def _ghost_focus_assets(self) -> Tuple[List[str], Dict[str, Any]]:
        trades = self.metrics.ghost_trade_snapshot(limit=500, lookback_sec=self.focus_lookback_sec)
        aggregate = self.metrics.aggregate_trade_metrics(trades)
        symbol_stats: Dict[str, Dict[str, Any]] = {}
        for trade in trades:
            symbol = str(trade.symbol or "UNKNOWN").upper()
            entry = symbol_stats.setdefault(
                symbol,
                {"profits": [], "count": 0, "wins": 0},
            )
            entry["profits"].append(float(trade.profit))
            entry["count"] += 1
            if float(trade.profit) > 0:
                entry["wins"] += 1
        ranked: List[Dict[str, Any]] = []
        for symbol, info in symbol_stats.items():
            avg_profit = float(np.mean(info["profits"])) if info["profits"] else 0.0
            win_rate = float(info["wins"] / max(1, info["count"]))
            penalty = max(0.0, -avg_profit) + max(0.0, 0.55 - win_rate)
            ranked.append(
                {
                    "symbol": symbol,
                    "avg_profit": avg_profit,
                    "win_rate": win_rate,
                    "count": info["count"],
                    "score": penalty,
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        focus_assets = [
            item["symbol"] for item in ranked if item["score"] > 0.0
        ][: self.focus_max_assets]
        self.metrics.record(
            MetricStage.GHOST_TRADING,
            aggregate,
            category="ghost_snapshot",
            meta={
                "iteration": self.iteration,
                "focus_candidates": focus_assets,
                "ranked": ranked[: self.focus_max_assets],
            },
        )
        return focus_assets, {"aggregate": aggregate, "ranked": ranked}

    def _apply_focus_adaptation(
        self,
        model: tf.keras.Model,
        focus_assets: Sequence[str],
    ) -> Optional[Dict[str, float]]:
        if not focus_assets:
            return None
        focus_inputs, focus_targets = self._prepare_dataset(
            batch_size=16,
            focus_assets=focus_assets,
            dataset_label="focus",
        )
        if focus_inputs is None or focus_targets is None:
            return None
        input_order = [tensor.name.split(":")[0] for tensor in model.inputs]
        output_order = [
            "exit_conf",
            "price_mu",
            "price_log_var",
            "price_dir",
            "net_margin",
            "net_pnl",
            "tech_recon",
            "price_gaussian",
        ]
        input_tensors = tuple(focus_inputs[name] for name in input_order)
        target_tensors = tuple(focus_targets[name] for name in output_order)
        focus_ds = (
            tf.data.Dataset.from_tensor_slices((input_tensors, target_tensors))
            .batch(8)
            .prefetch(tf.data.AUTOTUNE)
        )
        history = model.fit(focus_ds, epochs=1, verbose=0)
        metrics = {
            f"focus_{name}": float(values[-1])
            for name, values in history.history.items()
            if values
        }
        self.metrics.record(
            MetricStage.MODEL_FINE_TUNE,
            metrics,
            category="focus_training",
            meta={
                "iteration": self.iteration,
                "assets": list(focus_assets),
            },
        )
        return metrics

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "model_dir": str(self.model_dir),
            "best_score": self.optimizer.best_score,
            "best_params": self.optimizer.best_params,
        }
