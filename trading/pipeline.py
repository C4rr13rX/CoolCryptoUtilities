from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime, timezone, timedelta
import threading

import hashlib
import math

import numpy as np

from services.tf_runtime import configure_tensorflow
from services.system_profile import SystemProfile, detect_system_profile

configure_tensorflow()
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

from db import TradingDatabase, get_db
from trading.constants import PRIMARY_SYMBOL, top_pairs
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
    ConfusionMatrixSummary,
    confusion_sweep,
    classification_report,
    distribution_report,
)
from services.logging_utils import log_message


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

MODEL_OUTPUT_ORDER: Tuple[str, ...] = (
    "exit_conf",
    "price_mu",
    "price_log_var",
    "price_dir",
    "net_margin",
    "net_pnl",
    "tech_recon",
    "price_gaussian",
)

CONFUSION_WINDOW_BUCKETS: Tuple[Tuple[str, int], ...] = (
    ("5m", 5 * 60),
    ("15m", 15 * 60),
    ("1h", 60 * 60),
    ("6h", 6 * 60 * 60),
    ("1d", 24 * 60 * 60),
    ("7d", 7 * 24 * 60 * 60),
    ("30d", 30 * 24 * 60 * 60),
    ("180d", 180 * 24 * 60 * 60),
)


def _format_horizon_label(seconds: int) -> str:
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    if seconds < 30 * 86400:
        return f"{int(seconds // 86400)}d"
    months = max(1, int(seconds // (30 * 86400)))
    return f"{months}mth"


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
        self.system_profile: SystemProfile = detect_system_profile()
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
        self._train_lock = threading.Lock()

        self.min_ghost_trades = int(os.getenv("MIN_GHOST_TRADES_FOR_PROMOTION", "25"))
        self.max_false_positive_rate = float(os.getenv("MAX_FALSE_POSITIVE_RATE", "0.15"))
        self.min_ghost_win_rate = float(os.getenv("MIN_GHOST_WIN_RATE", "0.55"))
        self.min_realized_margin = float(os.getenv("MIN_REALIZED_MARGIN", "0.0"))

        self.window_size = 60
        self.sent_seq_len = 24
        self.tech_count = 35
        self.data_loader = HistoricalDataLoader()
        self.data_loader.apply_system_profile(self.system_profile)
        self.iteration: int = 0
        self.active_accuracy: float = 0.0
        self.target_positive_floor = float(os.getenv("TRAIN_POSITIVE_FLOOR", "0.15"))
        self.decision_threshold = float(os.getenv("PRICE_DIR_THRESHOLD", "0.58"))
        self.temperature_scale: float = 1.0
        self._load_state()
        self._active_model: Optional[tf.keras.Model] = None
        self.metrics = MetricsCollector(self.db)
        self.focus_lookback_sec = int(os.getenv("GHOST_FOCUS_LOOKBACK_SEC", "172800"))
        self.focus_max_assets = int(os.getenv("GHOST_FOCUS_MAX_ASSETS", "6"))
        self._horizon_targets = {
            "short": float(os.getenv("HORIZON_SHORT_MIN_SAMPLES", "32")),
            "mid": float(os.getenv("HORIZON_MID_MIN_SAMPLES", "24")),
            "long": float(os.getenv("HORIZON_LONG_MIN_SAMPLES", "12")),
        }
        self._horizon_bias: Dict[str, float] = {"short": 1.0, "mid": 1.0, "long": 1.0}
        self._vectorizer_signature: Optional[str] = None
        self._vectorizer_cache: set[str] = set()
        self._last_asset_vocab_requirement: int = 1
        self._last_dataset_meta: Dict[str, Any] = {}
        self._last_sample_meta: Dict[str, Any] = {}
        self.primary_symbol = PRIMARY_SYMBOL
        self._pause_flag_key = "pipeline_pause"
        self._last_news_top_up: float = 0.0
        self._confusion_windows: Dict[str, int] = {label: seconds for label, seconds in CONFUSION_WINDOW_BUCKETS}
        self._last_confusion_report: Dict[str, Dict[str, Any]] = {}

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
            return self._ensure_vectorizers_ready(model)
        print("[training] no active model found; building a fresh baseline.")
        loader_vocab = int(self.data_loader.asset_vocab_size)
        required_vocab = max(loader_vocab, int(getattr(self, "_last_asset_vocab_requirement", loader_vocab)))
        if required_vocab > loader_vocab:
            log_message(
                "training",
                "expanding asset vocabulary for candidate model",
                severity="info",
                details={"required": required_vocab, "loader_vocab": loader_vocab},
            )
        model, headline_vec, full_vec, losses, loss_weights = build_multimodal_model(
            window_size=self.window_size,
            tech_count=self.tech_count,
            sent_seq_len=self.sent_seq_len,
            asset_vocab_size=required_vocab,
        )
        self._adapt_vectorizers(headline_vec, full_vec)
        path = self.model_dir / "active_model.keras"
        model.save(path, include_optimizer=False)
        self._active_model = self._ensure_vectorizers_ready(model)
        return self._active_model

    def warm_dataset_cache(self, *, focus_assets: Optional[Sequence[str]] = None, oversample: bool = False) -> bool:
        """
        Pre-builds a dataset batch so subsequent training runs can reuse cached tensors.
        Returns True when at least one sample was generated.
        """
        try:
            inputs, targets, _ = self._prepare_dataset(
                batch_size=32,
                dataset_label="warmup",
                focus_assets=focus_assets,
                oversample=oversample,
            )
            return inputs is not None and targets is not None
        except Exception as exc:
            print(f"[training] dataset warmup failed: {exc}")
            return False

    def reinforce_news_cache(self, focus_assets: Optional[Sequence[str]] = None) -> bool:
        """
        Ensure recent news is available for the focus assets used in ghost trading.
        """
        try:
            return self._auto_backfill_news(focus_assets or [])
        except Exception as exc:
            print(f"[training] news enrichment failed: {exc}")
            return False

    def ghost_focus_assets(self) -> Tuple[List[str], Dict[str, Any]]:
        return self._ghost_focus_assets()

    def horizon_bias(self) -> Dict[str, float]:
        return dict(self._horizon_bias)

    def _load_active_clone(self) -> tf.keras.Model:
        path = self.model_dir / "active_model.keras"
        if path.exists():
            model = tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)
        else:
            base = self.ensure_active_model()
            model = tf.keras.models.clone_model(base)
            model.build(base.input_shape)
            model.set_weights(base.get_weights())
        model = self._ensure_asset_embedding_capacity(model)
        return self._ensure_vectorizers_ready(model)

    def _ensure_asset_embedding_capacity(self, model: tf.keras.Model) -> tf.keras.Model:
        try:
            layer = model.get_layer("asset_embedding")
        except ValueError:
            return model
        required = int(max(self.data_loader.asset_vocab_size, getattr(self, "_last_asset_vocab_requirement", 1)))
        current = int(getattr(layer, "input_dim", required))
        if required <= current:
            return model
        print(f"[training] expanding asset vocabulary from {current} to {required}.")
        upgraded = self._rebuild_model_with_asset_vocab(required, model)
        path = self.model_dir / "active_model.keras"
        upgraded.save(path, include_optimizer=False)
        self._active_model = upgraded
        reloaded = tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)
        return reloaded

    def _rebuild_model_with_asset_vocab(self, asset_vocab_size: int, source_model: tf.keras.Model) -> tf.keras.Model:
        new_model, _, _, _, _ = build_multimodal_model(
            window_size=self.window_size,
            tech_count=self.tech_count,
            sent_seq_len=self.sent_seq_len,
            asset_vocab_size=asset_vocab_size,
        )
        self._transfer_weights(source_model, new_model)
        return self._ensure_vectorizers_ready(new_model)

    def _transfer_weights(self, source_model: tf.keras.Model, target_model: tf.keras.Model) -> None:
        source_layers = {layer.name: layer for layer in source_model.layers}
        text_layer_cls = tf.keras.layers.TextVectorization
        for layer in target_model.layers:
            source_layer = source_layers.get(layer.name)
            if source_layer is None:
                continue
            if isinstance(layer, text_layer_cls):
                try:
                    vocab = source_layer.get_vocabulary()
                except Exception:
                    vocab = None
                if vocab:
                    layer.set_vocabulary(vocab)
                continue
            source_weights = source_layer.get_weights()
            target_weights = layer.get_weights()
            if not source_weights or not target_weights:
                continue
            if layer.name == "asset_embedding":
                kernel = target_weights[0]
                old_kernel = source_weights[0]
                copy_count = min(kernel.shape[0], old_kernel.shape[0])
                copy_cols = min(kernel.shape[1], old_kernel.shape[1])
                if copy_count and copy_cols:
                    kernel[:copy_count, :copy_cols] = old_kernel[:copy_count, :copy_cols]
                layer.set_weights([kernel])
                continue
            if len(source_weights) != len(target_weights):
                continue
            if not all(sw.shape == tw.shape for sw, tw in zip(source_weights, target_weights)):
                continue
            layer.set_weights(source_weights)

    def _ensure_vectorizers_ready(self, model: tf.keras.Model) -> tf.keras.Model:
        try:
            headline_vec = model.get_layer("headline_vectorizer")
            full_vec = model.get_layer("full_vectorizer")
        except ValueError:
            return model
        self._adapt_vectorizers(headline_vec, full_vec)
        return model

    # ------------------------------------------------------------------
    # Coordination helpers
    # ------------------------------------------------------------------

    def is_paused(self) -> bool:
        try:
            flag = self.db.get_control_flag(self._pause_flag_key)
        except Exception:
            flag = None
        if flag is None:
            return False
        if isinstance(flag, (int, float)):
            return bool(flag)
        return str(flag).strip().lower() in {"1", "true", "yes", "paused"}

    def request_pause(self) -> None:
        try:
            self.db.set_control_flag(self._pause_flag_key, "1")
        except Exception:
            pass

    def clear_pause(self) -> None:
        try:
            self.db.clear_control_flag(self._pause_flag_key)
        except Exception:
            pass

    def train_candidate(self) -> Optional[Dict[str, Any]]:
        if not self._train_lock.acquire(blocking=False):
            log_message("training", "train_candidate already running; skipping overlap", severity="warning")
            return {
                "iteration": self.iteration,
                "status": "busy",
                "score": self.active_accuracy,
            }
        try:
            return self._train_candidate_impl()
        finally:
            self._train_lock.release()

    def _train_candidate_impl(self) -> Optional[Dict[str, Any]]:
        proposal = self.optimizer.propose()
        lr = float(proposal.get("learning_rate", 3e-4))
        epochs = max(1, int(round(proposal.get("epochs", 2))))

        pending_iteration = self.iteration + 1

        if self.is_paused():
            self.iteration = pending_iteration
            self._save_state()
            return {
                "iteration": self.iteration,
                "status": "paused",
                "score": self.active_accuracy,
                "message": "training pause flag set",
            }

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
        self._record_horizon_metrics()
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
        output_order = list(MODEL_OUTPUT_ORDER)
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

        raw_score = float(history.history.get("price_dir_accuracy", [0.0])[-1])
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
        new_temperature = self._safe_float(evaluation.get("temperature", self.temperature_scale))
        if new_temperature > 0:
            new_temperature = float(np.clip(new_temperature, 0.25, 4.0))
            self.temperature_scale = float(0.8 * self.temperature_scale + 0.2 * new_temperature)

        signal_bundle = self._build_candidate_signals(evaluation, focus_stats)
        composite_score = self.optimizer.update(
            {"learning_rate": lr, "epochs": epochs},
            raw_score,
            signals=signal_bundle,
        )

        version = f"candidate-{int(time.time())}"
        path = self.model_dir / f"{version}.keras"
        model.save(path, include_optimizer=False)
        self.db.register_model_version(
            version=version,
            metrics={"score": composite_score, "raw_score": raw_score},
            path=str(path),
            activate=False,
        )

        result = {
            "iteration": self.iteration,
            "version": version,
            "score": composite_score,
            "raw_score": raw_score,
            "path": str(path),
            "params": {"learning_rate": lr, "epochs": epochs},
            "signals": signal_bundle,
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
            "candidate_score": composite_score,
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
            "brier_score": self._safe_float(evaluation.get("brier_score", 0.0)),
            "best_threshold": self._safe_float(evaluation.get("best_threshold", self.decision_threshold)),
            "ghost_trades_best": self._safe_float(evaluation.get("ghost_trades_best", 0.0)),
            "best_profit_factor": self._safe_float(evaluation.get("best_profit_factor", 0.0)),
            "best_win_rate": self._safe_float(evaluation.get("ghost_win_rate_best", 0.0)),
            "temperature": self._safe_float(evaluation.get("temperature", 1.0)),
            "temperature_scale": float(self.temperature_scale),
            "drift_alert": self._safe_float(evaluation.get("drift_alert", 0.0)),
            "drift_stat": self._safe_float(evaluation.get("drift_stat", 0.0)),
        }
        self.metrics.record(
            MetricStage.TRAINING,
            training_metrics,
            category="candidate",
            meta=evaluation_meta,
        )

        promote = composite_score >= self.promotion_threshold
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
            log_message("training", f"promotion deferred: {gating_reason}. Continuing candidate search.", severity="warning")
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
            self.promote_candidate(path, score=composite_score, metadata=result, evaluation=evaluation)
        else:
            self._print_ghost_summary(evaluation)
            if composite_score < self.promotion_threshold:
                log_message(
                    "training",
                    f"candidate score {composite_score:.3f} below promotion threshold {self.promotion_threshold:.3f}.",
                )
            else:
                log_message(
                    "training",
                    f"candidate retained for further evaluation despite score {composite_score:.3f} (promotion criteria not met).",
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
        log_message("training", f"promoting candidate {path.name} (score={score:.3f}) to active deployment.")
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
        selected_files: Optional[Sequence[str]] = None,
        oversample: bool = True,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, np.ndarray]]]:
        attempts = 0
        max_attempts = 4
        inputs: Optional[Dict[str, Any]] = None
        targets: Optional[Dict[str, Any]] = None
        self._last_sample_meta = {}
        while attempts < max_attempts:
            inputs, targets = self.data_loader.build_dataset(
                window_size=self.window_size,
                sent_seq_len=self.sent_seq_len,
                tech_count=self.tech_count,
                focus_assets=focus_assets,
                selected_files=selected_files,
                oversample=oversample,
            )
            if inputs is not None and targets is not None:
                break
            attempts += 1
            self.data_loader.expand_limits()
            time.sleep(0.1)
        if inputs is None or targets is None:
            if focus_assets:
                log_message(
                    "training",
                    "focus dataset unavailable; falling back to global dataset.",
                    severity="warning",
                    details={"focus_assets": list(focus_assets)},
                )
                return self._prepare_dataset(
                    batch_size=batch_size,
                    dataset_label=dataset_label,
                    focus_assets=None,
                    selected_files=selected_files,
                    oversample=oversample,
                )
            try:
                from services.background_workers import TokenDownloadSupervisor

                supervisor = TokenDownloadSupervisor(db=self.db)
                supervisor.base_worker.run_once()
            except Exception as exc:
                log_message("training", f"dataset fallback unable to fetch new data: {exc}", severity="warning")
            self.data_loader.expand_limits(factor=2.0, file_cap=128, sample_cap=8192)
            self.data_loader.invalidate_dataset_cache()
            inputs, targets = self.data_loader.build_dataset(
                window_size=self.window_size,
                sent_seq_len=self.sent_seq_len,
                tech_count=self.tech_count,
                focus_assets=focus_assets,
                selected_files=selected_files,
                oversample=oversample,
            )
        if inputs is None or targets is None:
            if self._auto_backfill_news(focus_assets):
                self.data_loader.invalidate_dataset_cache()
                inputs, targets = self.data_loader.build_dataset(
                    window_size=self.window_size,
                    sent_seq_len=self.sent_seq_len,
                    tech_count=self.tech_count,
                    focus_assets=focus_assets,
                    selected_files=selected_files,
                    oversample=oversample,
                )
        if inputs is not None and targets is not None:
            try:
                self._validate_dataset_shapes(inputs, targets)
                self._validate_dataset_values(inputs, targets)
            except ValueError as exc:
                self.metrics.feedback(
                    "preflight",
                    severity=FeedbackSeverity.CRITICAL,
                    label="dataset_validation_error",
                    details={"error": str(exc)},
                )
                raise
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
            per_asset_ratio: Dict[str, float] = {}
            asset_lexicon = getattr(self.data_loader, "asset_lexicon", {})
            if asset_lexicon and asset_ids.size:
                for idx, symbol in asset_lexicon.items():
                    mask = asset_ids == idx
                    if np.any(mask):
                        per_asset_ratio[symbol] = float(np.mean(dir_arr[mask] > 0.5))
            dataset_metrics = {
                "samples": sample_count,
                "asset_diversity": asset_diversity,
                "avg_price_volatility": price_volatility,
                "news_coverage_ratio": news_coverage,
                "positive_ratio": positive_ratio,
            }
            dataset_meta = {
                "iteration": self.iteration,
                "focus_assets": list(focus_assets or []),
                "signature": self.data_loader.dataset_signature(),
            }
            horizon_profile = self.data_loader.horizon_profile()
            if horizon_profile:
                dataset_metrics["horizon_window_count"] = float(len(horizon_profile))
                avg_horizon_samples = float(
                    np.mean([profile.get("samples", 0.0) for profile in horizon_profile.values()])
                )
                dataset_metrics["horizon_samples_avg"] = avg_horizon_samples
                dataset_meta["horizon_profile_keys"] = list(sorted(horizon_profile.keys()))
                coverage = self.data_loader.horizon_category_summary()
                if coverage:
                    for bucket in ("short", "mid", "long"):
                        dataset_metrics[f"horizon_{bucket}_samples"] = float(coverage.get(bucket, 0.0))
                    deficits = {
                        bucket: max(0.0, self._horizon_targets.get(bucket, 0.0) - coverage.get(bucket, 0.0))
                        for bucket in self._horizon_targets
                    }
                    if any(value > 0 for value in deficits.values()):
                        dataset_meta["horizon_deficit"] = deficits
                        self._handle_horizon_deficit(deficits, focus_assets)
            if per_asset_ratio:
                top_symbol, top_ratio = max(per_asset_ratio.items(), key=lambda kv: kv[1])
                low_symbol, low_ratio = min(per_asset_ratio.items(), key=lambda kv: kv[1])
                dataset_metrics["asset_positive_top"] = top_ratio
                dataset_metrics["asset_positive_bottom"] = low_ratio
                dataset_metrics["asset_positive_span"] = top_ratio - low_ratio
            dataset_metrics.update({f"net_margin_{k}": v for k, v in margin_stats.items()})
            if per_asset_ratio:
                dataset_meta["asset_positive_top_symbol"] = top_symbol
                dataset_meta["asset_positive_bottom_symbol"] = low_symbol
            self._last_dataset_meta = {
                **dataset_metrics,
                **dataset_meta,
            }
            if horizon_profile:
                self._last_dataset_meta["horizon_profile"] = horizon_profile
            self.metrics.record(
                MetricStage.TRAINING,
                dataset_metrics,
                category=f"dataset_{dataset_label}",
                meta=dataset_meta,
            )
            self._last_sample_meta = self.data_loader.last_sample_meta()
            self._maybe_trigger_news_top_up(news_coverage, focus_assets)
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
        self.metrics.feedback(
            "preflight",
            severity=FeedbackSeverity.CRITICAL,
            label="dataset_generation_failed",
            details={"attempts": attempts, "focus_assets": list(focus_assets or [])},
        )
        self._last_sample_meta = {}
        return None, None, None

    def _validate_dataset_shapes(self, inputs: Dict[str, Any], targets: Dict[str, Any]) -> None:
        """
        Guardrail to ensure dataset tensors always match the model signature.
        Raises ValueError with clear messaging when something drifts.
        """
        if not inputs or not targets:
            raise ValueError("Dataset validation requires both inputs and targets.")

        def _shape(arr: Any) -> Tuple[int, ...]:
            return tuple(np.asarray(arr).shape)

        samples = inputs["price_vol_input"].shape[0]
        if samples == 0:
            raise ValueError("Dataset is empty (0 samples).")

        expected_inputs: Dict[str, Tuple[int, ...]] = {
            "price_vol_input": (samples, self.window_size, 2),
            "sentiment_seq": (samples, self.sent_seq_len, 1),
            "tech_input": (samples, self.tech_count),
            "hour_input": (samples, 1),
            "dow_input": (samples, 1),
            "gas_fee_input": (samples, 1),
            "tax_rate_input": (samples, 1),
            "asset_id_input": (samples, 1),
            "headline_text": (samples, 1),
            "full_text": (samples, 1),
        }
        for name, shape in expected_inputs.items():
            if name not in inputs:
                raise ValueError(f"Dataset missing input '{name}'.")
            actual = _shape(inputs[name])
            if actual != shape:
                raise ValueError(f"Input '{name}' shape mismatch: expected {shape}, got {actual}.")

        expected_targets: Dict[str, Tuple[int, ...]] = {
            "exit_conf": (samples, 1),
            "price_mu": (samples, 1),
            "price_log_var": (samples, 1),
            "price_dir": (samples, 1),
            "net_margin": (samples, 1),
            "net_pnl": (samples, 1),
            "tech_recon": (samples, self.tech_count),
            "price_gaussian": (samples, 2),
        }
        for name, shape in expected_targets.items():
            if name not in targets:
                raise ValueError(f"Dataset missing target '{name}'.")
            actual = _shape(targets[name])
            if actual != shape:
                raise ValueError(f"Target '{name}' shape mismatch: expected {shape}, got {actual}.")

    def _validate_dataset_values(self, inputs: Dict[str, Any], targets: Dict[str, Any]) -> None:
        """
        Additional semantic validations to ensure tensors are finite and respect domain constraints.
        """

        def _finite(name: str, arr: Any) -> np.ndarray:
            arr_np = np.asarray(arr)
            if not np.all(np.isfinite(arr_np)):
                raise ValueError(f"Tensor '{name}' contains NaN or infinite values.")
            return arr_np

        price_vol = _finite("price_vol_input", inputs["price_vol_input"]).copy()
        if np.any(price_vol[..., 0] < 0):
            raise ValueError("Price channel contains negative values.")
        vol_thresh = float(os.getenv("VOLUME_ABSOLUTE_LIMIT", "1e12"))
        if np.any(np.abs(price_vol[..., 1]) > vol_thresh):
            log_message(
                "training",
                "clipping anomalous volumes in dataset",
                severity="warning",
                details={"threshold": vol_thresh},
            )
            price_vol[..., 1] = np.clip(price_vol[..., 1], -vol_thresh, vol_thresh)
            inputs["price_vol_input"][..., 1] = price_vol[..., 1]

        _finite("sentiment_seq", inputs["sentiment_seq"])

        _finite("tech_input", inputs["tech_input"])

        hour_input = _finite("hour_input", inputs["hour_input"])
        if np.any((hour_input < 0) | (hour_input > 23)):
            raise ValueError("Hour input outside [0, 23].")

        dow_input = _finite("dow_input", inputs["dow_input"])
        if np.any((dow_input < 0) | (dow_input > 6)):
            raise ValueError("Day-of-week input outside [0, 6].")

        asset_ids = _finite("asset_id_input", inputs["asset_id_input"])
        if np.any(asset_ids < 0):
            raise ValueError("Asset IDs must be non-negative.")
        if asset_ids.size:
            self._last_asset_vocab_requirement = max(1, int(np.max(asset_ids)) + 1)
        else:
            self._last_asset_vocab_requirement = 1

        gas = _finite("gas_fee_input", inputs["gas_fee_input"])
        tax = _finite("tax_rate_input", inputs["tax_rate_input"])
        if np.any(gas < 0) or np.any(tax < 0):
            raise ValueError("Gas or tax inputs contain negative values.")

        price_dir = _finite("price_dir", targets["price_dir"])
        if np.any((price_dir < 0) | (price_dir > 1)):
            raise ValueError("price_dir target must be within [0, 1].")

        exit_conf = _finite("exit_conf", targets["exit_conf"])
        if np.any((exit_conf < 0) | (exit_conf > 1)):
            raise ValueError("exit_conf target must be within [0, 1].")

        net_margin = _finite("net_margin", targets["net_margin"])
        net_pnl = _finite("net_pnl", targets["net_pnl"])
        if not np.allclose(net_margin, net_pnl, atol=1e-6):
            raise ValueError("net_margin and net_pnl diverge; expected mirror values.")

        price_gaussian = _finite("price_gaussian", targets["price_gaussian"])
        if price_gaussian.shape[-1] != 2:
            raise ValueError("price_gaussian last dimension must be 2.")

    def _build_training_dataset(
        self,
        model: tf.keras.Model,
        inputs: Dict[str, Any],
        targets: Dict[str, Any],
        sample_weights: Dict[str, np.ndarray],
        batch_size: int,
    ) -> tf.data.Dataset:
        input_names = getattr(model, "input_names", None) or [tensor.name.split(":")[0] for tensor in model.inputs]
        output_names = getattr(model, "output_names", None) or [tensor.name.split(":")[0] for tensor in model.outputs]
        input_tensors = tuple(tf.convert_to_tensor(inputs[name]) for name in input_names)
        target_tensors = tuple(tf.convert_to_tensor(targets[name]) for name in output_names)
        weight_tensors = []
        for idx, name in enumerate(output_names):
            weights = sample_weights.get(name)
            if weights is None:
                weights = np.ones(target_tensors[idx].shape[0], dtype=np.float32)
            weight_tensors.append(tf.convert_to_tensor(weights, dtype=tf.float32))
        weight_tensors = tuple(weight_tensors)
        dataset = tf.data.Dataset.from_tensor_slices((input_tensors, target_tensors, weight_tensors))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def lab_train_on_files(
        self,
        file_paths: Sequence[str],
        *,
        epochs: int = 1,
        batch_size: int = 16,
    ) -> Tuple[tf.keras.Model, Dict[str, float], Dict[str, Any]]:
        if not file_paths:
            raise ValueError("No files selected for lab training.")
        inputs, targets, sample_weights = self._prepare_dataset(
            batch_size=batch_size,
            dataset_label="lab_train",
            selected_files=file_paths,
        )
        if inputs is None or targets is None or sample_weights is None:
            raise RuntimeError("Unable to build dataset from selected files.")
        lab_model = self._load_active_clone()
        brier_metric = BinaryCrossentropy(name="brier_like", from_logits=False)
        losses = {
            "exit_conf": "binary_crossentropy",
            "price_mu": zero_loss,
            "price_log_var": zero_loss,
            "price_dir": "binary_crossentropy",
            "net_margin": "mse",
            "net_pnl": "mse",
            "tech_recon": "mse",
            "price_gaussian": gaussian_nll_loss,
        }
        loss_weights = {
            "exit_conf": 0.5,
            "price_mu": 0.0,
            "price_log_var": 0.0,
            "price_dir": 0.5,
            "net_margin": 1.0,
            "net_pnl": 0.0,
            "tech_recon": 0.25,
            "price_gaussian": 1.0,
        }
        lab_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss=losses,
            loss_weights=loss_weights,
            metrics={"price_dir": [tf.keras.metrics.AUC(name="auroc"), "accuracy", brier_metric]},
        )
        train_ds = self._build_training_dataset(lab_model, inputs, targets, sample_weights, batch_size)
        history = lab_model.fit(train_ds, epochs=max(1, int(epochs)), verbose=0)
        summary_metrics = {name: float(values[-1]) for name, values in history.history.items() if values}
        info = {
            "samples": int(inputs["price_vol_input"].shape[0]),
            "epochs": int(epochs),
            "files": [str(path) for path in file_paths],
        }
        return lab_model, summary_metrics, info

    def lab_evaluate_on_files(
        self,
        model: tf.keras.Model,
        file_paths: Sequence[str],
        *,
        batch_size: int = 16,
    ) -> Dict[str, float]:
        if not file_paths:
            raise ValueError("No files selected for lab evaluation.")
        inputs, targets, _ = self._prepare_dataset(
            batch_size=batch_size,
            dataset_label="lab_eval",
            selected_files=file_paths,
            oversample=False,
        )
        if inputs is None or targets is None:
            raise RuntimeError("Unable to prepare evaluation dataset.")
        input_names = getattr(model, "input_names", None) or [tensor.name.split(":")[0] for tensor in model.inputs]
        eval_inputs = [inputs[name] for name in input_names]
        predictions = model.predict(eval_inputs, batch_size=batch_size, verbose=0)
        if isinstance(predictions, list):
            pred_map = {name: np.asarray(array) for name, array in zip(model.output_names, predictions)}
        elif isinstance(predictions, dict):
            pred_map = {name: np.asarray(array) for name, array in predictions.items()}
        else:
            raise RuntimeError("Unexpected prediction structure from model.")
        results: Dict[str, float] = {}
        dir_true = np.asarray(targets["price_dir"]).reshape(-1)
        dir_pred = pred_map["price_dir"].reshape(-1)
        results["dir_accuracy"] = float(np.mean((dir_pred >= 0.5) == (dir_true >= 0.5))) if dir_true.size else 0.0
        results["dir_confidence_mean"] = float(np.mean(dir_pred)) if dir_pred.size else 0.0
        results["brier_score"] = float(np.mean((dir_pred - dir_true) ** 2)) if dir_true.size else 0.0
        margin_true = np.asarray(targets["net_margin"]).reshape(-1)
        margin_pred = pred_map["net_margin"].reshape(-1)
        if margin_true.size:
            results["margin_mae"] = float(np.mean(np.abs(margin_true - margin_pred)))
            results["margin_mean_true"] = float(np.mean(margin_true))
            results["margin_mean_pred"] = float(np.mean(margin_pred))
        pnl_pred = pred_map.get("net_pnl")
        if pnl_pred is not None:
            results["pnl_mean_pred"] = float(np.mean(pnl_pred.reshape(-1)))
        results["samples"] = float(dir_true.size)
        results["file_count"] = int(len(file_paths))
        return results

    def lab_collect_news(self, file_paths: Sequence[str]) -> Dict[str, Any]:
        from services.news_lab import collect_news_for_files

        return collect_news_for_files(file_paths, db=self.db)

    def lab_preview_series(
        self,
        file_paths: Sequence[str],
        *,
        batch_size: int = 16,
        include_news: bool = True,
    ) -> Dict[str, Any]:
        if not file_paths:
            raise ValueError("No files selected for preview.")
        inputs, targets, _ = self._prepare_dataset(
            batch_size=batch_size,
            dataset_label="lab_preview",
            selected_files=file_paths,
            oversample=False,
        )
        if inputs is None or targets is None:
            raise RuntimeError("Unable to build preview dataset.")
        model = self.ensure_active_model()
        input_names = getattr(model, "input_names", None) or [tensor.name.split(":")[0] for tensor in model.inputs]
        eval_inputs = [inputs[name] for name in input_names]
        predictions = model.predict(eval_inputs, batch_size=batch_size, verbose=0)
        if isinstance(predictions, list):
            pred_map = {name: np.asarray(array) for name, array in zip(model.output_names, predictions)}
        elif isinstance(predictions, dict):
            pred_map = {name: np.asarray(array) for name, array in predictions.items()}
        else:
            raise RuntimeError("Unexpected prediction structure from model.")

        dir_true = np.asarray(targets["price_dir"]).reshape(-1)
        dir_pred = pred_map["price_dir"].reshape(-1)
        mu_true = np.asarray(targets["price_mu"]).reshape(-1)
        mu_pred = pred_map["price_mu"].reshape(-1)
        margin_true = np.asarray(targets["net_margin"]).reshape(-1)
        margin_pred = pred_map["net_margin"].reshape(-1)

        metrics: Dict[str, float] = {}
        metrics["dir_accuracy"] = float(np.mean((dir_pred >= 0.5) == (dir_true >= 0.5))) if dir_true.size else 0.0
        metrics["dir_confidence_mean"] = float(np.mean(dir_pred)) if dir_pred.size else 0.0
        metrics["brier_score"] = float(np.mean((dir_pred - dir_true) ** 2)) if dir_true.size else 0.0
        if margin_true.size:
            metrics["margin_mae"] = float(np.mean(np.abs(margin_true - margin_pred)))
            metrics["margin_mean_true"] = float(np.mean(margin_true))
            metrics["margin_mean_pred"] = float(np.mean(margin_pred))
        pnl_pred = pred_map.get("net_pnl")
        if pnl_pred is not None:
            metrics["pnl_mean_pred"] = float(np.mean(np.asarray(pnl_pred).reshape(-1)))
        metrics["samples"] = float(dir_true.size)
        metrics["file_count"] = int(len(file_paths))

        meta_payload = self.last_sample_meta()
        records_meta = meta_payload.get("records")
        if not records_meta and meta_payload:
            timestamps = meta_payload.get("timestamps", [])
            symbols = meta_payload.get("symbols", [])
            current_prices = meta_payload.get("current_prices", [])
            future_prices = meta_payload.get("future_prices", [])
            files = meta_payload.get("files", [])
            records_meta = []
            for idx, ts in enumerate(timestamps):
                record = {
                    "timestamp": int(ts),
                    "symbol": symbols[idx] if idx < len(symbols) else "UNKNOWN",
                    "current_price": float(current_prices[idx]) if idx < len(current_prices) else 0.0,
                    "future_price": float(future_prices[idx]) if idx < len(future_prices) else 0.0,
                    "file": files[idx] if idx < len(files) else "",
                }
                records_meta.append(record)

        series: List[Dict[str, Any]] = []
        sample_count = int(dir_true.size)
        for idx in range(sample_count):
            record_meta = records_meta[idx] if records_meta and idx < len(records_meta) else {}
            ts = int(record_meta.get("timestamp", 0))
            current_price = float(record_meta.get("current_price", 0.0))
            future_price = float(record_meta.get("future_price", current_price))
            predicted_return = float(mu_pred[idx]) if idx < mu_pred.size else 0.0
            predicted_price = float(current_price * math.exp(predicted_return)) if current_price > 0 else 0.0
            series.append(
                {
                    "index": idx,
                    "timestamp": ts,
                    "timestamp_iso": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None,
                    "symbol": record_meta.get("symbol"),
                    "file": record_meta.get("file"),
                    "current_price": current_price,
                    "future_price": future_price,
                    "actual_return": float(mu_true[idx]) if idx < mu_true.size else 0.0,
                    "predicted_return": predicted_return,
                    "predicted_price": predicted_price,
                    "dir_true": float(dir_true[idx]) if idx < dir_true.size else 0.0,
                    "dir_probability": float(dir_pred[idx]) if idx < dir_pred.size else 0.0,
                    "net_margin_true": float(margin_true[idx]) if idx < margin_true.size else 0.0,
                    "net_margin_pred": float(margin_pred[idx]) if idx < margin_pred.size else 0.0,
                }
            )

        series.sort(key=lambda entry: entry["timestamp"] or 0)

        preview: Dict[str, Any] = {
            "metrics": metrics,
            "series": series,
            "meta": {
                "samples": len(series),
                "files": [str(path) for path in file_paths],
                "dataset": self._last_dataset_meta,
            },
        }
        if include_news:
            try:
                preview["news"] = self.lab_collect_news(file_paths)
            except Exception as exc:
                preview["news_error"] = str(exc)
        return preview

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

    def _record_horizon_metrics(self) -> None:
        profile = self.data_loader.horizon_profile()
        if not profile:
            return
        summary: Dict[str, float] = {}
        for horizon, stats in profile.items():
            label = _format_horizon_label(int(horizon))
            summary[f"{label}_samples"] = float(stats.get("samples", 0.0))
            summary[f"{label}_mae"] = float(stats.get("mae", 0.0))
        if summary:
            self.metrics.record(
                MetricStage.PIPELINE,
                summary,
                category="horizon_profile",
                meta={"iteration": self.iteration},
            )
            self._last_dataset_meta["horizon_profile"] = profile
        self._horizon_bias = self.data_loader.horizon_bias_snapshot()

    def last_sample_meta(self) -> Dict[str, Any]:
        if not self._last_sample_meta:
            return {}
        meta_copy: Dict[str, Any] = {}
        for key, value in self._last_sample_meta.items():
            if isinstance(value, list):
                meta_copy[key] = [dict(item) for item in value] if key == "records" else list(value)
            else:
                meta_copy[key] = value
        return meta_copy

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

    def _auto_backfill_news(self, focus_assets: Optional[Sequence[str]]) -> bool:
        try:
            symbols = list(focus_assets or top_pairs(limit=min(self.focus_max_assets, 10)))
            if not symbols:
                return False
            lookback_sec = int(os.getenv("CRYPTOPANIC_LOOKBACK_SEC", str(2 * 24 * 3600)))
            backfilled = self.data_loader.request_news_backfill(symbols=symbols, lookback_sec=lookback_sec)
            if not backfilled and os.getenv("CRYPTOPANIC_API_KEY"):
                try:
                    center = datetime.now(timezone.utc)
                    start = center - timedelta(seconds=lookback_sec)
                    end = center + timedelta(seconds=max(3600, lookback_sec // 2))
                    archiver = CryptoNewsArchiver(
                        output_path=Path(os.getenv("CRYPTOPANIC_ARCHIVE_PATH", "data/news/cryptopanic_archive.parquet"))
                    )
                    archive = archiver.backfill(symbols=symbols, start=start, end=end)
                    if not archive.empty:
                        # reload combined news cache so synthetic fallback is avoided
                        self.data_loader.invalidate_dataset_cache()
                        self.data_loader.news_items = self.data_loader._load_news()
                        backfilled = True
                except Exception as arch_exc:
                    log_message("training", f"archive backfill failed: {arch_exc}", severity="warning")
            return backfilled
        except Exception as exc:
            log_message("training", f"news backfill skipped due to error: {exc}", severity="warning")
            return False

    def _maybe_trigger_news_top_up(self, coverage_ratio: float, focus_assets: Optional[Sequence[str]]) -> None:
        min_ratio = float(os.getenv("NEWS_MIN_COVERAGE", "0.4"))
        if coverage_ratio >= min_ratio:
            return
        now = time.time()
        interval = float(os.getenv("NEWS_TOPUP_INTERVAL_SEC", "900"))
        if (now - self._last_news_top_up) < interval:
            return
        candidate_assets: List[str] = list(focus_assets or [])
        if not candidate_assets and self._last_sample_meta.get("symbols"):
            symbols = self._last_sample_meta.get("symbols", [])
            candidate_assets = list(dict.fromkeys(symbols))[: self.focus_max_assets]
        if not candidate_assets:
            return
        if self._auto_backfill_news(candidate_assets):
            self._last_news_top_up = now

    def _handle_horizon_deficit(self, deficits: Dict[str, float], focus_assets: Optional[Sequence[str]]) -> None:
        log_message(
            "training",
            "horizon coverage below target",
            severity="warning",
            details={"deficits": deficits, "focus": list(focus_assets or [])},
        )
        adjustments = self.data_loader.rebalance_horizons(deficits, focus_assets)
        self._horizon_bias = self.data_loader.horizon_bias_snapshot()
        if adjustments:
            self.metrics.feedback(
                "training",
                severity=FeedbackSeverity.INFO,
                label="horizon_rebalance",
                details={"deficits": deficits, "adjustments": adjustments},
            )

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
                self.temperature_scale = float(tp_state.get("temperature_scale", self.temperature_scale))
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
            "temperature_scale": float(self.temperature_scale),
        }
        try:
            self.db.save_state(state)
        except Exception:
            pass

    def _safe_float(self, value: Any) -> float:
        return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))

    def _print_ghost_summary(self, evaluation: Optional[Dict[str, float]]) -> None:
        if not evaluation:
            log_message("training", f"ghost summary (iteration {self.iteration}): no evaluation available", severity="info")
            return
        log_message(
            "training",
            "ghost summary",
            details={
                "iteration": self.iteration,
                "dir_acc": self._safe_float(evaluation.get("dir_accuracy", 0.0)),
                "ghost_trades": int(evaluation.get("ghost_trades", 0)),
                "pred_margin": self._safe_float(evaluation.get("ghost_pred_margin", 0.0)),
                "realized_margin": self._safe_float(evaluation.get("ghost_realized_margin", 0.0)),
                "win_rate": self._safe_float(evaluation.get("ghost_win_rate", 0.0)),
                "TP": int(evaluation.get("true_positives", 0)),
                "FP": int(evaluation.get("false_positives", 0)),
                "TN": int(evaluation.get("true_negatives", 0)),
                "FN": int(evaluation.get("false_negatives", 0)),
            },
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
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

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
        summary["brier_score"] = self._safe_float(np.mean(np.square(price_dir_scores - price_dir_true)))
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

        confusion_report = self._build_confusion_report(price_dir_scores, thresholds)
        if confusion_report:
            summary["confusion_matrices"] = confusion_report
            self._persist_confusion_report(confusion_report)
        else:
            self._last_confusion_report = {}

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

    def _build_confusion_report(
        self,
        price_dir_scores: np.ndarray,
        thresholds: Sequence[float],
    ) -> Dict[str, Dict[str, Any]]:
        meta = self.data_loader.last_sample_meta()
        records = meta.get("records") or []
        if not records:
            return {}
        limit = min(len(records), int(price_dir_scores.shape[0]))
        if limit <= 0:
            return {}
        report: Dict[str, Dict[str, Any]] = {}
        for label, horizon_sec in self._confusion_windows.items():
            key = str(int(horizon_sec))
            score_bucket: List[float] = []
            truth_bucket: List[float] = []
            for idx in range(limit):
                horizons = records[idx].get("horizons") or {}
                if key not in horizons:
                    continue
                score_bucket.append(float(price_dir_scores[idx]))
                truth_bucket.append(1.0 if float(horizons[key]) > 0 else 0.0)
            if len(score_bucket) < 12:
                continue
            sweep = confusion_sweep(score_bucket, truth_bucket, thresholds)
            if not sweep:
                continue
            best_summary = max(
                sweep.values(),
                key=lambda summary: summary.report().get("f1_score", 0.0),
                default=None,
            )
            if best_summary is None:
                continue
            payload = best_summary.to_dict()
            payload["thresholds_tested"] = list(sweep.keys())
            report[label] = payload
        return report

    def _persist_confusion_report(self, report: Dict[str, Dict[str, Any]]) -> None:
        if not report:
            self._last_confusion_report = {}
            return
        self._last_confusion_report = report
        try:
            best_label, best_payload = max(
                report.items(),
                key=lambda item: item[1].get("f1_score", 0.0),
            )
        except ValueError:
            best_label, best_payload = next(iter(report.items()))
        metrics_payload = {
            "horizon": best_label,
            "precision": float(best_payload.get("precision", 0.0)),
            "recall": float(best_payload.get("recall", 0.0)),
            "f1_score": float(best_payload.get("f1_score", 0.0)),
            "samples": float(best_payload.get("samples", 0)),
            "threshold": float(best_payload.get("threshold", self.decision_threshold)),
        }
        try:
            self.metrics.record(
                MetricStage.TRAINING,
                metrics_payload,
                category="confusion_best",
                meta={"horizons": list(report.keys())},
            )
        except Exception:
            pass
        try:
            path = Path("data/reports/confusion_matrices.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "iteration": int(self.iteration),
                "updated_at": int(time.time()),
                "confusion": report,
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def live_readiness_report(self) -> Dict[str, Any]:
        report = self._last_confusion_report or {}
        if not report:
            return {"ready": False, "reason": "no_confusion_data"}
        preferred = ("5m", "15m", "1h", "6h")
        anchor_label = None
        anchor = None
        for label in preferred:
            if label in report:
                anchor_label = label
                anchor = report[label]
                break
        if anchor is None and report:
            anchor_label, anchor = next(iter(report.items()))
        if anchor is None:
            return {"ready": False, "reason": "no_confusion_data"}
        precision = float(anchor.get("precision", 0.0))
        recall = float(anchor.get("recall", 0.0))
        samples = int(anchor.get("samples", 0))
        threshold = float(anchor.get("threshold", self.decision_threshold))
        ready = precision >= 0.58 and recall >= 0.55 and samples >= 64
        reason = "" if ready else "insufficient_accuracy"
        return {
            "ready": ready,
            "reason": reason,
            "horizon": anchor_label,
            "precision": precision,
            "recall": recall,
            "samples": samples,
            "threshold": threshold,
        }

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
        if not focus_assets and self.primary_symbol:
            focus_assets = [self.primary_symbol]
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

    def _build_candidate_signals(
        self,
        evaluation: Dict[str, Any],
        focus_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        signals: Dict[str, float] = {}

        def _add(name: str, value: Any) -> None:
            try:
                val = float(value)
            except (TypeError, ValueError):
                return
            if math.isfinite(val):
                signals[name] = val

        dataset_meta = self._last_dataset_meta or {}
        aggregate = (focus_stats or {}).get("aggregate") or {}
        ranked = (focus_stats or {}).get("ranked") or []

        # Core evaluation metrics
        for key in [
            "dir_accuracy",
            "best_f1",
            "profit_factor",
            "best_profit_factor",
            "margin_mae",
            "drift_stat",
            "kelly_fraction",
            "payoff_ratio",
            "temperature",
        ]:
            _add(key, evaluation.get(key))
        _add("ghost_win_rate", evaluation.get("ghost_win_rate", evaluation.get("ghost_win_rate_best")))
        _add("ghost_win_rate_best", evaluation.get("ghost_win_rate_best"))
        _add("ghost_realized_margin_best", evaluation.get("ghost_realized_margin_best"))
        _add("ghost_pred_margin_best", evaluation.get("ghost_pred_margin_best"))
        _add("ghost_trades_best", evaluation.get("ghost_trades_best"))

        # Dataset statistics
        _add("positive_ratio", dataset_meta.get("positive_ratio"))
        _add("samples", dataset_meta.get("samples"))
        _add("asset_diversity", dataset_meta.get("asset_diversity"))
        _add("avg_price_volatility", dataset_meta.get("avg_price_volatility"))

        # Ghost aggregate metrics
        for key in [
            "win_rate",
            "avg_profit",
            "median_profit",
            "kelly_fraction",
            "avg_duration_sec",
            "avg_expected_vs_realized_delta",
        ]:
            _add(f"ghost_{key}", aggregate.get(key))

        if ranked:
            penalties = [item.get("score") for item in ranked if item.get("score") is not None]
            if penalties:
                _add("focus_penalty_mean", np.mean(penalties))

        confusion = self._last_confusion_report or {}
        for label in ("5m", "15m", "1h", "6h"):
            bucket = confusion.get(label)
            if not bucket:
                continue
            _add(f"confusion_{label}_precision", bucket.get("precision"))
            _add(f"confusion_{label}_recall", bucket.get("recall"))
            _add(f"confusion_{label}_f1", bucket.get("f1_score"))
            _add(f"confusion_{label}_samples", bucket.get("samples"))
            break
        return signals

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
from services.news_archive import CryptoNewsArchiver
