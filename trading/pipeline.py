from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hashlib
import math

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
    FeedbackSeverity,
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
        self.target_positive_floor = float(os.getenv("TRAIN_POSITIVE_FLOOR", "0.15"))
        self.decision_threshold = float(os.getenv("PRICE_DIR_THRESHOLD", "0.58"))
        self._load_state()
        self._active_model: Optional[tf.keras.Model] = None
        self.metrics = MetricsCollector(self.db)
        self.focus_lookback_sec = int(os.getenv("GHOST_FOCUS_LOOKBACK_SEC", "172800"))
        self.focus_max_assets = int(os.getenv("GHOST_FOCUS_MAX_ASSETS", "6"))
        self._vectorizer_signature: Optional[str] = None
        self._vectorizer_cache: set[str] = set()
        self._last_dataset_meta: Dict[str, Any] = {}

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

        prep_start = time.perf_counter()
        result = self._prepare_dataset(batch_size=32, dataset_label="full")
        inputs, targets, sample_weights = result
        if inputs is None or targets is None or sample_weights is None:
            self.iteration = pending_iteration
            self._save_state()
            print("[training] insufficient data for candidate training; skipping this cycle.")
            return {"iteration": self.iteration, "status": "skipped", "score": None}
        prep_duration = time.perf_counter() - prep_start
        self.metrics.record(
            MetricStage.TRAINING,
            {
                "dataset_seconds": prep_duration,
                "positive_ratio": float(self._last_dataset_meta.get("positive_ratio", 0.0)),
                "samples": float(self._last_dataset_meta.get("samples", 0)),
            },
            category="runtime_prep",
            meta={"iteration": self.iteration},
        )
        self.metrics.feedback(
            "preflight",
            severity=FeedbackSeverity.INFO,
            label="dataset_ready",
            details={
                "samples": self._last_dataset_meta.get("samples", 0),
                "positive_ratio": self._last_dataset_meta.get("positive_ratio", 0.0),
            },
        )
        self._preflight_checks(inputs, targets)

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
        weight_tensors = tuple(
            tf.convert_to_tensor(sample_weights.get(name, np.ones(targets[name].shape[0], dtype=np.float32)), dtype=tf.float32)
            for name in output_order
        )
        train_ds = (
            tf.data.Dataset.from_tensor_slices((input_tensors, target_tensors, weight_tensors))
            .batch(16)
            .prefetch(tf.data.AUTOTUNE)
        )
        train_start = time.perf_counter()
        history = model.fit(train_ds, epochs=epochs, verbose=0, callbacks=callbacks)
        train_duration = time.perf_counter() - train_start
        self.metrics.record(
            MetricStage.TRAINING,
            {"train_seconds": train_duration, "epochs": float(epochs)},
            category="runtime_train",
            meta={"iteration": self.iteration},
        )

        score = float(history.history.get("price_dir_accuracy", [0.0])[-1])
        self.optimizer.update({"learning_rate": lr, "epochs": epochs}, score)
        eval_ds = (
            tf.data.Dataset.from_tensor_slices((input_tensors, target_tensors))
            .batch(32)
            .prefetch(tf.data.AUTOTUNE)
        )
        eval_start = time.perf_counter()
        evaluation = self._evaluate_candidate(model, eval_ds, targets)
        focus_history = self._apply_focus_adaptation(model, focus_assets)
        if focus_history is not None:
            evaluation = self._evaluate_candidate(model, eval_ds, targets)
            evaluation.update(focus_history)
        if evaluation.get("best_f1", 1.0) < 0.6:
            burst_rerun = self._burst_replay(model, inputs, targets, sample_weights, evaluation)
            if burst_rerun:
                evaluation = self._evaluate_candidate(model, eval_ds, targets)
                evaluation.update(burst_rerun)
        if evaluation.get("best_threshold") is not None:
            self.decision_threshold = float(evaluation["best_threshold"])
        eval_duration = time.perf_counter() - eval_start
        self.metrics.record(
            MetricStage.TRAINING,
            {"eval_seconds": eval_duration},
            category="runtime_eval",
            meta={"iteration": self.iteration},
        )

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
            "best_threshold": self._safe_float(evaluation.get("best_threshold", self.decision_threshold)),
            "ghost_trades_best": self._safe_float(evaluation.get("ghost_trades_best", 0.0)),
            "best_profit_factor": self._safe_float(evaluation.get("best_profit_factor", 0.0)),
            "best_win_rate": self._safe_float(evaluation.get("ghost_win_rate_best", 0.0)),
            "temperature": self._safe_float(evaluation.get("temperature", 1.0)),
            "drift_alert": self._safe_float(evaluation.get("drift_alert", 0.0)),
            "drift_stat": self._safe_float(evaluation.get("drift_stat", 0.0)),
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
                ghost_trades = int(evaluation.get("ghost_trades_best", evaluation.get("ghost_trades", 0)))
                positive_ratio = float(self._last_dataset_meta.get("positive_ratio", 0.0))
                effective_min_trades = self.min_ghost_trades
                if ghost_trades > 0 and positive_ratio < self.target_positive_floor:
                    effective_min_trades = max(5, min(self.min_ghost_trades, ghost_trades))
                if ghost_trades < effective_min_trades:
                    gating_reason = (
                        f"ghost trades {ghost_trades} below minimum {effective_min_trades}"
                    )
                else:
                    fp_rate = float(evaluation.get("false_positive_rate_best", evaluation.get("false_positive_rate", 0.0)))
                    win_rate = float(evaluation.get("ghost_win_rate_best", evaluation.get("ghost_win_rate", 0.0)))
                    realized_margin = float(evaluation.get("ghost_realized_margin_best", evaluation.get("ghost_realized_margin", 0.0)))
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
            self.metrics.feedback(
                "promotion",
                severity=FeedbackSeverity.WARNING,
                label="deferred",
                details={
                    "iteration": self.iteration,
                    "reason": gating_reason,
                    "ghost_trades_best": evaluation.get("ghost_trades_best"),
                    "positive_ratio": self._last_dataset_meta.get("positive_ratio", 0.0),
                },
            )
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
            self.metrics.record(
                MetricStage.TRAINING,
                {
                    "ghost_trades_best": float(evaluation.get("ghost_trades_best", evaluation.get("ghost_trades", 0))),
                    "best_threshold": float(evaluation.get("best_threshold", self.decision_threshold)),
                },
                category="candidate_eval",
                meta={"iteration": self.iteration},
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
        if evaluation and evaluation.get("best_threshold") is not None:
            new_threshold = float(evaluation["best_threshold"])
            self.decision_threshold = float(
                max(
                    0.05,
                    min(0.95, 0.7 * self.decision_threshold + 0.3 * new_threshold),
                )
            )
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
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, np.ndarray]]]:
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
            dir_arr = targets["price_dir"].reshape(-1)
            positive_ratio = float(np.mean(dir_arr > 0.5)) if dir_arr.size else 0.0
            dataset_metrics = {
                "samples": sample_count,
                "asset_diversity": asset_diversity,
                "avg_price_volatility": price_volatility,
                "news_coverage_ratio": news_coverage,
                "positive_ratio": positive_ratio,
            }
            dataset_metrics.update({f"net_margin_{k}": v for k, v in margin_stats.items()})
            dataset_meta = {
                "iteration": self.iteration,
                "focus_assets": list(focus_assets or []),
                "signature": self.data_loader.dataset_signature(),
            }
            self._last_dataset_meta = {
                **dataset_metrics,
                **dataset_meta,
            }
            self.metrics.record(
                MetricStage.TRAINING,
                dataset_metrics,
                category=f"dataset_{dataset_label}",
                meta=dataset_meta,
            )
            inputs["headline_text"] = tf.convert_to_tensor(inputs["headline_text"], dtype=tf.string)
            inputs["full_text"] = tf.convert_to_tensor(inputs["full_text"], dtype=tf.string)
            inputs["asset_id_input"] = tf.convert_to_tensor(inputs["asset_id_input"], dtype=tf.int32)
            price_dir_true = targets["price_dir"].reshape(-1)
            positive_mask = price_dir_true > 0.5
            pos_ratio = float(np.mean(positive_mask)) if price_dir_true.size else 0.0
            sample_weights: Dict[str, np.ndarray] = {}
            for name, value in targets.items():
                sample_weights[name] = np.ones(value.shape[0], dtype=np.float32)
            if 0.0 < pos_ratio < 1.0:
                weight_pos = max(1.0, (1.0 - pos_ratio) / max(pos_ratio, 1e-4))
                sample_weights["price_dir"][positive_mask] = weight_pos
                sample_weights["net_margin"][positive_mask] = weight_pos
            margin_intensity = np.clip(np.abs(margin_arr), 0.1, 5.0)
            sample_weights["net_margin"] *= margin_intensity
            sample_weights["net_pnl"] *= margin_intensity
            for key in sample_weights:
                sample_weights[key] = sample_weights[key].astype(np.float32)
            return inputs, targets, sample_weights
        return None, None, None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _preflight_checks(self, inputs: Dict[str, Any], targets: Dict[str, Any]) -> bool:
        meta = self._last_dataset_meta or {}
        sample_count = int(meta.get("samples", inputs["price_vol_input"].shape[0] if inputs else 0))
        positive_ratio = float(meta.get("positive_ratio", 0.0))
        if sample_count < max(32, self.window_size):
            self.metrics.feedback(
                "preflight",
                severity=FeedbackSeverity.CRITICAL,
                label="insufficient_samples",
                details={"samples": sample_count},
            )
            return False
        if positive_ratio < self.target_positive_floor * 0.5:
            self.metrics.feedback(
                "preflight",
                severity=FeedbackSeverity.WARNING,
                label="low_positive_ratio",
                details={"positive_ratio": positive_ratio},
            )
        return True

    def _adapt_vectorizers(self, headline_vec, full_vec) -> None:
        texts = [text for text in self.data_loader.sample_texts(limit=512) if text]
        if not texts:
            return
        digest = hashlib.sha1("|".join(sorted(texts)).encode("utf-8")).hexdigest()
        needs_init = False
        try:
            vocab = headline_vec.get_vocabulary()
            needs_init = len(vocab) <= 2  # TextVectorization default vocab: ['', '[UNK]']
        except Exception:
            needs_init = True

        if not needs_init and digest == self._vectorizer_signature:
            return

        candidate_texts = [text for text in texts if text not in self._vectorizer_cache]
        if needs_init or not candidate_texts:
            candidate_texts = texts

        if not candidate_texts:
            return

        dataset = tf.data.Dataset.from_tensor_slices(candidate_texts)
        headline_vec.adapt(dataset)
        full_vec.adapt(dataset)
        self._vectorizer_cache.update(candidate_texts)
        self._vectorizer_signature = digest

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
                self.decision_threshold = float(tp_state.get("decision_threshold", self.decision_threshold))
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
            "decision_threshold": float(self.decision_threshold),
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
        summary["positive_ratio"] = self._safe_float(np.mean(price_dir_true > 0.5))
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

        temperature = self._calibrate_temperature(price_dir_scores, price_dir_true)
        summary["temperature"] = self._safe_float(temperature)

        drift_alert, drift_stat = self._page_hinkley(net_margin_true, net_margin_pred_flat)
        summary["drift_alert"] = float(drift_alert)
        summary["drift_stat"] = self._safe_float(drift_stat)
        if drift_alert:
            self.metrics.feedback(
                "drift",
                severity=FeedbackSeverity.WARNING,
                label="margin_drift",
                details={
                    "iteration": self.iteration,
                    "stat": drift_stat,
                },
            )

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

        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        best_score = -np.inf
        best_info: Optional[Dict[str, Any]] = None
        for thr in thresholds:
            thr_labels = price_dir_scores > thr
            tp_thr = int(np.sum((price_dir_true > 0.5) & (thr_labels == True)))
            tn_thr = int(np.sum((price_dir_true <= 0.5) & (thr_labels == False)))
            fp_thr = int(np.sum((price_dir_true <= 0.5) & (thr_labels == True)))
            fn_thr = int(np.sum((price_dir_true > 0.5) & (thr_labels == False)))
            metrics_thr = classification_report(tp_thr, fp_thr, tn_thr, fn_thr)
            realized_thr = 0.0
            win_thr = 0.0
            profit_factor_thr = 0.0
            if thr_labels.any():
                realized_thr = self._safe_float(np.mean(net_margin_true[thr_labels]))
                win_thr = self._safe_float(np.mean(net_margin_true[thr_labels] > 0))
                pos_vals = net_margin_true[thr_labels]
                gains = self._safe_float(np.sum(pos_vals[pos_vals > 0]))
                losses = self._safe_float(np.sum(np.abs(pos_vals[pos_vals < 0])))
                if losses > 0:
                    profit_factor_thr = self._safe_float(gains / losses)
                else:
                    profit_factor_thr = gains
            summary[f"thr_{thr:.2f}_f1"] = self._safe_float(metrics_thr.get("f1_score", 0.0))
            summary[f"thr_{thr:.2f}_precision"] = self._safe_float(metrics_thr.get("precision", 0.0))
            summary[f"thr_{thr:.2f}_recall"] = self._safe_float(metrics_thr.get("recall", 0.0))
            summary[f"thr_{thr:.2f}_profit_factor"] = profit_factor_thr
            summary[f"thr_{thr:.2f}_win_rate"] = win_thr
            objective_thr = self._safe_float(metrics_thr.get("f1_score", 0.0)) + profit_factor_thr
            if objective_thr > best_score:
                best_score = objective_thr
                best_info = {
                    "threshold": thr,
                    "tp": tp_thr,
                    "fp": fp_thr,
                    "tn": tn_thr,
                    "fn": fn_thr,
                    "metrics": metrics_thr,
                    "profit_factor": profit_factor_thr,
                    "win_rate": win_thr,
                    "realized_margin": realized_thr,
                    "predicted_margin": self._safe_float(np.mean(net_margin_pred_flat[thr_labels])) if thr_labels.any() else 0.0,
                    "positives": int(np.sum(thr_labels)),
                }

        if best_info:
            summary["best_threshold"] = self._safe_float(best_info["threshold"])
            summary["ghost_trades_best"] = int(best_info["positives"])
            summary["false_positive_rate_best"] = self._safe_float(best_info["metrics"].get("false_positive_rate", 0.0))
            summary["best_profit_factor"] = self._safe_float(best_info["profit_factor"])
            summary["ghost_realized_margin_best"] = self._safe_float(best_info["realized_margin"])
            summary["ghost_pred_margin_best"] = self._safe_float(best_info["predicted_margin"])
            summary["ghost_win_rate_best"] = self._safe_float(best_info["win_rate"])
            summary["best_precision"] = self._safe_float(best_info["metrics"].get("precision", 0.0))
            summary["best_recall"] = self._safe_float(best_info["metrics"].get("recall", 0.0))
            summary["best_f1"] = self._safe_float(best_info["metrics"].get("f1_score", 0.0))
            summary["best_tp"] = int(best_info["tp"])
            summary["best_fp"] = int(best_info["fp"])
            summary["best_tn"] = int(best_info["tn"])
            summary["best_fn"] = int(best_info["fn"])
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
        focus_inputs, focus_targets, focus_weights = self._prepare_dataset(
            batch_size=16,
            focus_assets=focus_assets,
            dataset_label="focus",
        )
        if focus_inputs is None or focus_targets is None or focus_weights is None:
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
        weight_tensors = tuple(
            tf.convert_to_tensor(focus_weights.get(name, np.ones(focus_targets[name].shape[0], dtype=np.float32)), dtype=tf.float32)
            for name in output_order
        )
        focus_ds = (
            tf.data.Dataset.from_tensor_slices((input_tensors, target_tensors, weight_tensors))
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

    def _burst_replay(
        self,
        model: tf.keras.Model,
        inputs: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        sample_weights: Dict[str, np.ndarray],
        evaluation: Dict[str, Any],
        burst_window: int = 15,
        top_k: int = 64,
    ) -> Optional[Dict[str, float]]:
        try:
            input_order = [tensor.name.split(":")[0] for tensor in model.inputs]
            pred_ds = tf.data.Dataset.from_tensor_slices(tuple(inputs[name] for name in input_order)).batch(64)
            predictions = model.predict(pred_ds, verbose=0)
        except Exception:
            return None
        if not isinstance(predictions, (list, tuple)) or len(predictions) < 4:
            return None
        price_dir_scores = predictions[3].reshape(-1)
        price_dir_true = targets["price_dir"].reshape(-1)
        if price_dir_scores.shape[0] != price_dir_true.shape[0]:
            return None
        residual = np.abs(price_dir_scores - price_dir_true)
        if residual.size == 0:
            return None
        burst_k = min(top_k, residual.size)
        if burst_k == 0:
            return None
        top_idx = np.argsort(residual)[-burst_k:]
        truncated_inputs = self._truncate_inputs(inputs, top_idx, burst_window)
        truncated_targets = {name: value[top_idx] for name, value in targets.items()}
        truncated_weights = {name: sample_weights.get(name, np.ones_like(truncated_targets[name], dtype=np.float32))[top_idx] for name in truncated_targets}
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
        train_ds = (
            tf.data.Dataset.from_tensor_slices(
                (
                    tuple(truncated_inputs[name] for name in input_order),
                    tuple(truncated_targets[name] for name in output_order),
                    tuple(truncated_weights[name] for name in output_order),
                )
            )
            .batch(8)
            .prefetch(tf.data.AUTOTUNE)
        )
        model.fit(train_ds, epochs=1, verbose=0)
        return {
            "burst_samples": float(burst_k),
            "burst_window": float(burst_window),
        }

    def _truncate_inputs(
        self,
        inputs: Dict[str, np.ndarray],
        indices: np.ndarray,
        burst_window: int,
    ) -> Dict[str, np.ndarray]:
        truncated: Dict[str, np.ndarray] = {}
        for name, value in inputs.items():
            subset = value[indices]
            if subset.ndim == 3 and subset.shape[1] >= burst_window:
                truncated[name] = subset[:, -burst_window:, :]
            elif subset.ndim == 2 and subset.shape[1] >= burst_window:
                truncated[name] = subset[:, -burst_window:]
            else:
                truncated[name] = subset
        return truncated

    def _calibrate_temperature(self, probs: np.ndarray, labels: np.ndarray) -> float:
        probs = np.clip(probs.astype(np.float64), 1e-6, 1 - 1e-6)
        logits = np.log(probs) - np.log1p(-probs)
        labels = labels.astype(np.float64)

        def _nll(temp: float) -> float:
            temp = max(temp, 1e-6)
            scaled = 1.0 / (1.0 + np.exp(-logits / temp))
            return float(-np.mean(labels * np.log(scaled) + (1 - labels) * np.log1p(-scaled)))

        temps = np.linspace(0.5, 3.0, num=11)
        losses = [_nll(t) for t in temps]
        return float(temps[int(np.argmin(losses))])

    def _page_hinkley(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        delta: float = 0.005,
        lambd: float = 0.05,
    ) -> Tuple[bool, float]:
        residual = (actual - predicted).astype(np.float64)
        cumulative = 0.0
        min_cumulative = 0.0
        for value in residual:
            cumulative = cumulative + value - delta
            if cumulative < min_cumulative:
                min_cumulative = cumulative
            if cumulative - min_cumulative > lambd:
                return True, cumulative - min_cumulative
        return False, cumulative - min_cumulative

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "model_dir": str(self.model_dir),
            "best_score": self.optimizer.best_score,
            "best_params": self.optimizer.best_params,
            "decision_threshold": float(self.decision_threshold),
        }
