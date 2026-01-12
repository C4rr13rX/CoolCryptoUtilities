from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime, timezone, timedelta
import threading
from collections import Counter

import hashlib
import math

import numpy as np

from services.tf_runtime import configure_tensorflow
from services.system_profile import SystemProfile, detect_system_profile

configure_tensorflow()
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy

from db import TradingDatabase, get_db
from trading.constants import PRIMARY_CHAIN, PRIMARY_SYMBOL, top_pairs
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
from services.watchlists import load_watchlists


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
    ("3m", 90 * 24 * 60 * 60),
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
        if self.system_profile.memory_pressure:
            # Favor smaller, faster candidates on constrained hosts to shorten time-to-first-model.
            os.environ.setdefault("TRAIN_LIGHTWEIGHT", "1")
            os.environ.setdefault("TRAIN_BATCH_SIZE", "12")
        self.db = db or get_db()
        self.model_templates = ["tiny", "base", "robust"]
        if self.system_profile.memory_pressure or self.system_profile.is_low_power:
            # keep template search small on low-power hosts
            self.model_templates = ["tiny", "base"]
        base_search = {
            "learning_rate": (1e-5, 5e-4),
            "epochs": (1.0, 4.0),
        }
        if self.system_profile.memory_pressure or self.system_profile.is_low_power:
            base_search["epochs"] = (1.0, 2.0)
        base_search["template_idx"] = (0, float(len(self.model_templates) - 1))
        self.optimizer = optimizer or BayesianBruteForceOptimizer(base_search)
        self.model_dir = model_dir or Path(os.getenv("MODEL_DIR", "models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._prune_rejected_artifacts()
        self.promotion_threshold = promotion_threshold
        self._train_lock = threading.Lock()

        self.min_ghost_trades = int(os.getenv("MIN_GHOST_TRADES_FOR_PROMOTION", os.getenv("MIN_GHOST_TRADES_OVERRIDE", "25")))
        self.max_false_positive_rate = float(
            os.getenv("MAX_FALSE_POSITIVE_RATE_OVERRIDE", os.getenv("MAX_FALSE_POSITIVE_RATE", "0.15"))
        )
        self.min_ghost_win_rate = float(
            os.getenv("MIN_GHOST_WIN_RATE_OVERRIDE", os.getenv("MIN_GHOST_WIN_RATE", "0.55"))
        )
        self.min_realized_margin = float(os.getenv("MIN_REALIZED_MARGIN", os.getenv("MIN_REALIZED_MARGIN_OVERRIDE", "0.0")))

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
        self.calibration_scale: float = 1.0
        self.calibration_offset: float = 0.0
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
        try:
            default_news_cap = min(self.focus_max_assets, 4)
            news_cap_env = int(os.getenv("NEWS_ENRICH_MAX_SYMBOLS", str(default_news_cap)))
        except Exception:
            news_cap_env = 0
        self._news_enrich_max_symbols = max(1, min(self.focus_max_assets, news_cap_env or default_news_cap))
        try:
            self._news_enrich_page_cap = max(1, int(os.getenv("NEWS_ENRICH_MAX_PAGES", "2")))
        except Exception:
            self._news_enrich_page_cap = 2
        try:
            self._news_enrich_budget = max(0.0, float(os.getenv("NEWS_ENRICH_BUDGET_SEC", "85")))
        except Exception:
            self._news_enrich_budget = 85.0
        try:
            self._news_enrich_min_budget = max(0.0, float(os.getenv("NEWS_ENRICH_MIN_BUDGET_SEC", "6")))
        except Exception:
            self._news_enrich_min_budget = 6.0
        self._news_focus_offset = 0
        self._confusion_windows: Dict[str, int] = {label: seconds for label, seconds in CONFUSION_WINDOW_BUCKETS}
        self._last_confusion_report: Dict[str, Dict[str, Any]] = {}
        self._last_confusion_summary: Dict[str, Any] = {}
        self._last_transition_plan: Dict[str, Any] = {}
        self._last_confusion_refresh: float = 0.0
        self._load_cached_confusion_report()
        self._last_candidate_feedback: Dict[str, Any] = {}
        self._active_approval_margin = float(os.getenv("ACTIVE_APPROVAL_MARGIN", "0.01"))
        self._active_fpr_buffer = float(os.getenv("ACTIVE_APPROVAL_FPR_BUFFER", "0.02"))
        self._thr_precision_weight = float(os.getenv("THR_PRECISION_WEIGHT", "1.0"))
        self._thr_recall_weight = float(os.getenv("THR_RECALL_WEIGHT", "1.0"))
        self._pos_weight_multiplier = float(os.getenv("POS_WEIGHT_MULTIPLIER", "1.0"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _prune_rejected_artifacts(self) -> None:
        """
        Drop candidate artifacts that were not promoted to keep disk and DB clean.
        """
        active_path = (self.model_dir / "active_model.keras").resolve()
        removed: List[str] = []
        for path in self.model_dir.glob("candidate-*.keras"):
            try:
                if path.resolve() == active_path:
                    continue
                path.unlink(missing_ok=True)
                removed.append(path.name)
            except Exception:
                continue
        purged = 0
        try:
            purged = self.db.purge_rejected_models()
        except Exception:
            purged = 0
        if removed or purged:
            log_message(
                "training",
                "pruned rejected candidates",
                details={"files_removed": removed, "db_pruned": purged},
            )

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
        if self.system_profile.memory_pressure or self.system_profile.is_low_power:
            template_idx = 0
        else:
            template_idx = 1
        template_idx = max(0, min(template_idx, len(self.model_templates) - 1))
        template_choice = self._select_model_template(template_idx)
        model, headline_vec, full_vec, losses, loss_weights = build_multimodal_model(
            window_size=self.window_size,
            tech_count=self.tech_count,
            sent_seq_len=self.sent_seq_len,
            asset_vocab_size=required_vocab,
            model_template=template_choice,
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
            deadline = time.time() + self._news_enrich_budget if self._news_enrich_budget > 0 else None
            return self._auto_backfill_news(focus_assets or [], deadline=deadline)
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

    def _select_model_template(self, template_idx: int) -> str:
        choice = self.model_templates[int(max(0, min(template_idx, len(self.model_templates) - 1)))]
        feedback = self._last_candidate_feedback or {}
        fpr = self._safe_float(feedback.get("false_positive_rate_best", feedback.get("false_positive_rate", 0.0)))
        drift = self._safe_float(feedback.get("drift_stat", 0.0))
        if fpr > 0.4:
            return "tiny"
        if drift > 0.1 and "robust" in self.model_templates:
            return "robust"
        return choice

    def _evaluate_active_model(self, eval_ds, targets: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Score the currently active model on the same dataset so we only promote
        candidates that clearly beat the incumbent.
        """
        active = self.load_active_model()
        if active is None:
            return None
        try:
            return self._evaluate_candidate(active, eval_ds, targets)
        except Exception:
            return None

    def _active_approval(self, candidate_eval: Dict[str, float], active_eval: Dict[str, float]) -> bool:
        if os.getenv("DISABLE_ACTIVE_APPROVAL", "0").lower() in {"1", "true", "yes", "on"}:
            return True
        cand_acc = self._safe_float(candidate_eval.get("dir_accuracy", 0.0))
        active_acc = self._safe_float(active_eval.get("dir_accuracy", 0.0))
        cand_fpr = self._safe_float(candidate_eval.get("false_positive_rate_best", candidate_eval.get("false_positive_rate", 0.0)))
        active_fpr = self._safe_float(active_eval.get("false_positive_rate_best", active_eval.get("false_positive_rate", 0.0)))
        if cand_acc < active_acc + self._active_approval_margin:
            return False
        if cand_fpr > max(0.0, active_fpr - self._active_fpr_buffer):
            return False
        return True

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
        template_idx = int(round(proposal.get("template_idx", 0)))
        template_idx = max(0, min(template_idx, len(self.model_templates) - 1))
        template_choice = self._select_model_template(template_idx)

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
            source_counts: Dict[str, int] = {}
            for item in news_items:
                sentiment = str(item.get("sentiment", "neutral")).lower()
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                for token in item.get("tokens") or []:
                    token_coverage.add(str(token).upper())
                source = str(item.get("source") or "").strip()
                if source:
                    source_counts[source] = source_counts.get(source, 0) + 1
            total_news = len(news_items)
            news_metrics = {
                "news_items_total": total_news,
                "news_token_coverage": len(token_coverage),
                "news_positive_ratio": sentiment_counts.get("positive", 0) / total_news,
                "news_negative_ratio": sentiment_counts.get("negative", 0) / total_news,
                "news_sources": len(source_counts),
            }
            self.metrics.record(
                MetricStage.NEWS,
                news_metrics,
                category="training_news",
                meta={
                    "iteration": self.iteration,
                    "unique_tokens": list(sorted(token_coverage))[:32],
                    "sources": [
                        name
                        for name, _ in sorted(source_counts.items(), key=lambda item: item[1], reverse=True)[:12]
                    ],
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

        loader_vocab = int(self.data_loader.asset_vocab_size)
        required_vocab = int(max(loader_vocab, getattr(self, "_last_asset_vocab_requirement", loader_vocab)))
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
        if os.getenv("TRAIN_EARLY_STOP", "1").lower() in {"1", "true", "yes", "on"}:
            try:
                patience = max(0, int(os.getenv("TRAIN_EARLY_STOP_PATIENCE", "1")))
            except Exception:
                patience = 1
            try:
                min_delta = max(0.0, float(os.getenv("TRAIN_EARLY_STOP_MIN_DELTA", "0.0005")))
            except Exception:
                min_delta = 0.0005
            callbacks.append(
                EarlyStopping(
                    monitor="loss",
                    patience=patience,
                    min_delta=min_delta,
                    restore_best_weights=True,
                    verbose=0,
                )
            )
        input_order = [tensor.name.split(":")[0] for tensor in model.inputs]
        output_order = list(MODEL_OUTPUT_ORDER)
        input_tensors = tuple(inputs[name] for name in input_order)
        target_tensors = tuple(targets[name] for name in output_order)
        weight_tensors = tuple(
            tf.convert_to_tensor(sample_weights.get(name, np.ones(targets[name].shape[0], dtype=np.float32)), dtype=tf.float32)
            for name in output_order
        )
        try:
            batch_size = max(8, min(32, int(os.getenv("TRAIN_BATCH_SIZE", "16"))))
        except Exception:
            batch_size = 16
        train_ds = (
            tf.data.Dataset.from_tensor_slices((input_tensors, target_tensors, weight_tensors))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        train_start = time.perf_counter()
        if os.getenv("TRAIN_LIGHTWEIGHT", "0").lower() in {"1", "true", "yes", "on"}:
            epochs = min(epochs, 2)
        max_extra_epochs = int(os.getenv("TRAIN_MAX_EPOCHS_EXTRA", "1"))
        epochs = min(epochs + max_extra_epochs, max(epochs, 3))
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
            .batch(batch_size)
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
        if evaluation.get("calibration_scale") is not None and evaluation.get("calibration_offset") is not None:
            new_scale = self._safe_float(evaluation.get("calibration_scale", self.calibration_scale))
            new_offset = self._safe_float(evaluation.get("calibration_offset", self.calibration_offset))
            new_scale = float(np.clip(new_scale, 0.3, 3.0))
            new_offset = float(np.clip(new_offset, -3.0, 3.0))
            self.calibration_scale = float(0.8 * self.calibration_scale + 0.2 * new_scale)
            self.calibration_offset = float(0.8 * self.calibration_offset + 0.2 * new_offset)

        signal_bundle = self._build_candidate_signals(evaluation, focus_stats)
        composite_score = self.optimizer.update(
            {"learning_rate": lr, "epochs": epochs},
            raw_score,
            signals=signal_bundle,
        )

        result = {
            "iteration": self.iteration,
            "version": None,
            "score": composite_score,
            "raw_score": raw_score,
            "path": None,
            "params": {"learning_rate": lr, "epochs": epochs, "template": template_choice},
            "signals": signal_bundle,
            "evaluation": evaluation,
            "status": "trained",
        }
        evaluation_meta = {
            "iteration": self.iteration,
            "params": {"learning_rate": lr, "epochs": epochs, "template": template_choice},
            "version": None,
            "focus_assets": focus_assets,
            "ghost_feedback": focus_stats,
        }
        ghost_gate = self._ghost_validation()
        ghost_ready = bool(ghost_gate.get("ready", True))
        ghost_tail_guard = float(
            ghost_gate.get(
                "tail_guardrail",
                ghost_gate.get("tail_guard", float(os.getenv("GHOST_TAIL_GUARDRAIL", "0.0"))),
            )
        )
        ghost_tail_risk = float(ghost_gate.get("tail_risk", 0.0))
        ghost_drawdown_guard = float(ghost_gate.get("drawdown_guardrail", 0.0))
        ghost_drawdown = float(ghost_gate.get("max_drawdown", 0.0))
        ghost_tail_block = ghost_tail_guard > 0 and ghost_tail_risk > ghost_tail_guard
        ghost_drawdown_breach = ghost_drawdown_guard > 0 and ghost_drawdown > ghost_drawdown_guard
        ghost_reason = str(ghost_gate.get("reason") or "")
        evaluation_meta["ghost_validation"] = ghost_gate
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
            "ghost_tail_risk": ghost_tail_risk,
            "ghost_tail_guard": ghost_tail_guard,
            "ghost_drawdown": ghost_drawdown,
            "ghost_drawdown_guard": ghost_drawdown_guard,
            "false_positive_rate": self._safe_float(evaluation.get("false_positive_rate", 0.0)),
            "brier_score": self._safe_float(evaluation.get("brier_score", 0.0)),
            "best_threshold": self._safe_float(evaluation.get("best_threshold", self.decision_threshold)),
            "ghost_trades_best": self._safe_float(evaluation.get("ghost_trades_best", 0.0)),
            "best_profit_factor": self._safe_float(evaluation.get("best_profit_factor", 0.0)),
            "best_win_rate": self._safe_float(evaluation.get("ghost_win_rate_best", 0.0)),
            "temperature": self._safe_float(evaluation.get("temperature", 1.0)),
            "temperature_scale": float(self.temperature_scale),
            "calibration_scale": self._safe_float(evaluation.get("calibration_scale", self.calibration_scale)),
            "calibration_offset": self._safe_float(evaluation.get("calibration_offset", self.calibration_offset)),
            "calibration_log_loss": self._safe_float(evaluation.get("calibration_log_loss", 0.0)),
            "drift_alert": self._safe_float(evaluation.get("drift_alert", 0.0)),
            "drift_stat": self._safe_float(evaluation.get("drift_stat", 0.0)),
            "template": template_choice,
        }
        self.metrics.record(
            MetricStage.TRAINING,
            training_metrics,
            category="candidate",
            meta=evaluation_meta,
        )
        self._last_candidate_feedback = dict(evaluation)

        promote = composite_score >= self.promotion_threshold
        gating_reason: Optional[str] = None
        active_eval = self._evaluate_active_model(eval_ds, targets)
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
                    elif active_eval:
                        if not self._active_approval(evaluation, active_eval):
                            gating_reason = "active model approval failed"
                if not gating_reason:
                    if not ghost_ready:
                        gating_reason = f"ghost_validation:{ghost_reason or 'not_ready'}"
                    elif ghost_tail_block:
                        gating_reason = "ghost_tail_risk"
                    elif ghost_drawdown_breach:
                        gating_reason = "ghost_drawdown_guardrail"
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
                    "ghost_ready": ghost_ready,
                    "ghost_tail_risk": ghost_tail_risk,
                    "ghost_tail_guard": ghost_tail_guard,
                    "ghost_drawdown": ghost_drawdown,
                    "ghost_drawdown_guard": ghost_drawdown_guard,
                },
            )
        if promote:
            version = f"candidate-{int(time.time())}"
            path = self.model_dir / f"{version}.keras"
            model.save(path, include_optimizer=False)
            self.db.register_model_version(
                version=version,
                metrics={"score": composite_score, "raw_score": raw_score},
                path=str(path),
                activate=False,
            )
            result["version"] = version
            result["path"] = str(path)
            evaluation_meta["version"] = version
            self.promote_candidate(path, score=composite_score, metadata=result, evaluation=evaluation)
        else:
            self._print_ghost_summary(evaluation)
            evaluation_data = evaluation or {}
            if gating_reason:
                log_message(
                    "training",
                    "candidate gated despite passing score threshold",
                    details={
                        "score": float(composite_score),
                        "threshold": float(self.promotion_threshold),
                        "gating_reason": gating_reason,
                        "ghost_trades": int(evaluation_data.get("ghost_trades_best", evaluation_data.get("ghost_trades", 0))),
                    },
                )
                self.metrics.record(
                    MetricStage.TRAINING,
                    {
                        "ghost_trades_best": float(evaluation_data.get("ghost_trades_best", evaluation_data.get("ghost_trades", 0))),
                        "best_threshold": float(evaluation_data.get("best_threshold", self.decision_threshold)),
                    },
                    category="candidate_eval",
                    meta={"iteration": self.iteration, "gating_reason": gating_reason},
                )
            elif composite_score < self.promotion_threshold:
                log_message(
                    "training",
                    f"candidate score {composite_score:.3f} below promotion threshold {self.promotion_threshold:.3f}.",
                )
            else:
                log_message(
                    "training",
                    "candidate retained despite meeting score threshold",
                    details={
                        "score": float(composite_score),
                        "threshold": float(self.promotion_threshold),
                        "gating_reason": gating_reason or "criteria_not_met",
                    },
                )
                self.metrics.record(
                    MetricStage.TRAINING,
                    {
                        "ghost_trades_best": float(evaluation_data.get("ghost_trades_best", evaluation_data.get("ghost_trades", 0))),
                        "best_threshold": float(evaluation_data.get("best_threshold", self.decision_threshold)),
                    },
                    category="candidate_eval",
                    meta={"iteration": self.iteration},
                )

        if not promote:
            summary_details = {
                "promote": False,
                "score": float(composite_score),
                "threshold": float(self.promotion_threshold),
                "gating_reason": gating_reason or "criteria_not_met",
                "ghost_trades": int(evaluation.get("ghost_trades_best", evaluation.get("ghost_trades", 0)) if evaluation else 0),
                "false_positive_rate": self._safe_float(
                    evaluation.get("false_positive_rate_best", evaluation.get("false_positive_rate", 0.0)) if evaluation else 0.0
                ),
                "win_rate": self._safe_float(
                    evaluation.get("ghost_win_rate_best", evaluation.get("ghost_win_rate", 0.0)) if evaluation else 0.0
                ),
                "realized_margin": self._safe_float(
                    evaluation.get("ghost_realized_margin_best", evaluation.get("ghost_realized_margin", 0.0))
                    if evaluation
                    else 0.0
                ),
                "active_accuracy": float(self.active_accuracy or 0.0),
                "ghost_ready": ghost_ready,
                "ghost_tail_risk": ghost_tail_risk,
                "ghost_tail_guard": ghost_tail_guard,
                "ghost_drawdown": ghost_drawdown,
                "ghost_drawdown_guard": ghost_drawdown_guard,
            }
            log_message("training", "promotion decision", details=summary_details)
            self._maybe_update_threshold_from_evaluation(evaluation, gating_reason)
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
                    horizon_targets = self._effective_horizon_targets()
                    dataset_meta["horizon_targets"] = horizon_targets
                    deficits = {
                        bucket: max(0.0, horizon_targets.get(bucket, 0.0) - coverage.get(bucket, 0.0))
                        for bucket in horizon_targets
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
            self._last_sample_meta = self.data_loader.last_sample_meta()
            lookahead_median = self._compute_lookahead_median(self._last_sample_meta)
            if lookahead_median is not None:
                dataset_meta["lookahead_median_sec"] = int(lookahead_median)
                self._last_dataset_meta["lookahead_median_sec"] = int(lookahead_median)
            self.metrics.record(
                MetricStage.TRAINING,
                dataset_metrics,
                category=f"dataset_{dataset_label}",
                meta=dataset_meta,
            )
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
                weight_pos *= max(0.5, self._pos_weight_multiplier)
                sample_weights["price_dir"][positive_mask] = weight_pos
                sample_weights["net_margin"][positive_mask] = weight_pos
            margin_intensity = np.clip(np.abs(margin_arr), 0.1, 5.0)
            sample_weights["net_margin"] *= margin_intensity
            sample_weights["net_pnl"] *= margin_intensity
            horizon_weights = self._horizon_sample_weights(
                self._last_sample_meta,
                sample_count=sample_weights["price_dir"].shape[0],
            )
            if horizon_weights is not None:
                sample_weights["price_dir"] *= horizon_weights
                sample_weights["net_margin"] *= horizon_weights
                sample_weights["net_pnl"] *= horizon_weights
                weight_stats = {
                    "horizon_weight_mean": float(np.mean(horizon_weights)),
                    "horizon_weight_min": float(np.min(horizon_weights)),
                    "horizon_weight_max": float(np.max(horizon_weights)),
                }
                self._last_dataset_meta.update(weight_stats)
                self.metrics.record(
                    MetricStage.TRAINING,
                    weight_stats,
                    category=f"dataset_{dataset_label}_weights",
                    meta={"iteration": self.iteration},
                )
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
            try:
                path = Path("data/reports/horizon_profile.json")
                path.parent.mkdir(parents=True, exist_ok=True)
                snapshot = {
                    "iteration": int(self.iteration),
                    "updated_at": int(time.time()),
                    "summary": summary,
                    "profile": profile,
                }
                path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            except Exception:
                pass
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

    def _compute_lookahead_median(self, meta: Optional[Dict[str, Any]] = None) -> Optional[int]:
        meta = meta or self._last_sample_meta or {}
        records = meta.get("records") if isinstance(meta, dict) else None
        if not isinstance(records, list):
            return None
        values: List[float] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            try:
                value = float(record.get("lookahead_sec", 0.0))
            except (TypeError, ValueError):
                continue
            if value > 0:
                values.append(value)
        if not values:
            return None
        return int(np.median(np.asarray(values, dtype=np.float64)))

    def horizon_forecast(
        self,
        predicted_return: float,
        *,
        current_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        dataset_meta = self._last_dataset_meta if isinstance(self._last_dataset_meta, dict) else {}
        profile = dataset_meta.get("horizon_profile") if isinstance(dataset_meta, dict) else None
        if not isinstance(profile, dict) or not profile:
            try:
                profile = self.data_loader.horizon_profile()
            except Exception:
                profile = {}
        if not isinstance(profile, dict) or not profile:
            return {}
        base_raw = None
        if isinstance(dataset_meta, dict):
            base_raw = dataset_meta.get("lookahead_median_sec")
        if base_raw is None:
            base_raw = self._compute_lookahead_median()
        try:
            base_sec = int(float(base_raw)) if base_raw is not None else 0
        except (TypeError, ValueError):
            base_sec = 0
        if base_sec <= 0:
            try:
                base_sec = int(getattr(self.data_loader, "_min_horizon_sec", 0) or 0)
            except Exception:
                base_sec = 0
        if base_sec <= 0:
            base_sec = 300
        base_stats = profile.get(str(int(base_sec)), {}) if isinstance(profile, dict) else {}
        base_mean = 0.0
        if isinstance(base_stats, dict):
            base_mean = float(base_stats.get("mean_return", 0.0))
        base_scale = abs(base_mean) if abs(base_mean) > 1e-6 else 0.0
        horizon_entries: List[Tuple[int, Dict[str, Any]]] = []
        for key, stats in profile.items():
            try:
                horizon_sec = int(float(key))
            except (TypeError, ValueError):
                continue
            if not isinstance(stats, dict):
                stats = {}
            horizon_entries.append((horizon_sec, stats))
        try:
            bias_map = dict(getattr(self, "_horizon_bias", {}) or {})
        except Exception:
            bias_map = {}
        short_cutoff = 30 * 60
        mid_cutoff = 24 * 3600
        forecasts: Dict[str, Any] = {}
        for horizon_sec, stats in sorted(horizon_entries, key=lambda entry: entry[0]):
            mean_return = float(stats.get("mean_return", 0.0))
            if base_scale > 0:
                ratio = abs(mean_return) / base_scale
            else:
                ratio = math.sqrt(max(horizon_sec, 1) / max(base_sec, 1))
            ratio = max(0.25, min(6.0, ratio))
            if horizon_sec <= short_cutoff:
                bucket = "short"
            elif horizon_sec <= mid_cutoff:
                bucket = "mid"
            else:
                bucket = "long"
            try:
                bias = float(bias_map.get(bucket, 1.0))
            except (TypeError, ValueError):
                bias = 1.0
            if not math.isfinite(bias) or bias <= 0:
                bias = 1.0
            ratio *= max(0.5, min(2.0, bias))
            horizon_return = float(predicted_return) * ratio
            entry: Dict[str, Any] = {
                "return": horizon_return,
                "mean_return": mean_return,
                "positive_ratio": float(stats.get("positive_ratio", 0.0)),
                "samples": float(stats.get("samples", 0.0)),
            }
            if current_price is not None and current_price > 0:
                entry["price"] = float(current_price * math.exp(horizon_return))
            forecasts[_format_horizon_label(horizon_sec)] = entry
        return {"base_lookahead_sec": int(base_sec), "forecast": forecasts}

    def _adapt_vectorizers(self, headline_vec, full_vec) -> None:
        try:
            sample_limit = int(os.getenv("VECTORIZE_SAMPLE_LIMIT", "512"))
        except Exception:
            sample_limit = 512
        if self.system_profile.memory_pressure or self.system_profile.is_low_power:
            sample_limit = min(sample_limit, 256)
        sample_limit = max(32, sample_limit)
        texts = [text for text in self.data_loader.sample_texts(limit=sample_limit) if text]
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

    def _auto_backfill_news(self, focus_assets: Optional[Sequence[str]], *, deadline: Optional[float] = None) -> bool:
        try:
            symbols_all = list(focus_assets or top_pairs(limit=min(self.focus_max_assets, 10)))
            if not symbols_all:
                return False
            lookback_sec = int(os.getenv("CRYPTOPANIC_LOOKBACK_SEC", str(2 * 24 * 3600)))
            max_symbols = max(1, min(self._news_enrich_max_symbols, len(symbols_all)))
            if len(symbols_all) > max_symbols:
                offset = self._news_focus_offset % len(symbols_all)
                rotated = symbols_all[offset:] + symbols_all[:offset]
                symbols = rotated[:max_symbols]
                self._news_focus_offset = (offset + max_symbols) % len(symbols_all)
            else:
                symbols = symbols_all
                self._news_focus_offset = (self._news_focus_offset + len(symbols_all)) % len(symbols_all)
            if deadline and (deadline - time.time()) < self._news_enrich_min_budget:
                return False
            page_cap = max(1, min(self._news_enrich_page_cap, getattr(self.data_loader, "_cryptopanic_max_pages", self._news_enrich_page_cap)))
            backfilled = self.data_loader.request_news_backfill(
                symbols=symbols,
                lookback_sec=lookback_sec,
                deadline=deadline,
                max_pages=page_cap,
            )
            if not backfilled and os.getenv("CRYPTOPANIC_API_KEY"):
                if deadline and (deadline - time.time()) < self._news_enrich_min_budget:
                    return False
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

    def _maybe_update_threshold_from_evaluation(
        self,
        evaluation: Optional[Dict[str, Any]],
        gating_reason: Optional[str] = None,
    ) -> None:
        if not evaluation:
            return
        best_threshold = evaluation.get("best_threshold")
        if best_threshold is None:
            return
        try:
            candidate = float(best_threshold)
        except (TypeError, ValueError):
            return
        if not 0.0 < candidate < 1.0:
            return
        delta = abs(candidate - self.decision_threshold)
        if delta < 0.02:
            return
        blend = float(os.getenv("TRAINING_THRESHOLD_BLEND", "0.2") or "0.2")
        blend = max(0.05, min(blend, 0.5))
        updated = max(0.05, min(0.95, (1 - blend) * self.decision_threshold + blend * candidate))
        self.decision_threshold = updated
        self.metrics.feedback(
            "training",
            severity=FeedbackSeverity.INFO,
            label="decision_threshold_tuned",
            details={
                "best_threshold": candidate,
                "updated_threshold": updated,
                "reason": gating_reason or "candidate_eval",
            },
        )

    def _effective_horizon_targets(self) -> Dict[str, float]:
        targets = dict(self._horizon_targets)
        try:
            windows = tuple(getattr(self.data_loader, "_horizon_windows", ()) or ())
        except Exception:
            windows = ()
        if not windows:
            return {bucket: 0.0 for bucket in targets}
        short_cutoff = 30 * 60
        mid_cutoff = 24 * 3600
        has_short = any(float(horizon) <= short_cutoff for horizon in windows)
        has_mid = any(short_cutoff < float(horizon) <= mid_cutoff for horizon in windows)
        has_long = any(float(horizon) > mid_cutoff for horizon in windows)
        if not has_short:
            targets["short"] = 0.0
        if not has_mid:
            targets["mid"] = 0.0
        if not has_long:
            targets["long"] = 0.0
        return targets

    def _handle_horizon_deficit(self, deficits: Dict[str, float], focus_assets: Optional[Sequence[str]]) -> None:
        log_message(
            "training",
            "horizon coverage below target",
            severity="warning",
            details={"deficits": deficits, "focus": list(focus_assets or [])},
        )
        try:
            deficit_payload = {}
            total_deficit = 0.0
            for bucket in ("short", "mid", "long"):
                value = float(deficits.get(bucket, 0.0))
                value = max(0.0, value)
                deficit_payload[f"{bucket}_deficit"] = value
                total_deficit += value
            if deficit_payload:
                deficit_payload["total_deficit"] = total_deficit
                self.metrics.record(
                    MetricStage.TRAINING,
                    deficit_payload,
                    category="horizon_deficit",
                    meta={"focus_assets": list(focus_assets or []), "iteration": self.iteration},
                )
        except Exception:
            pass
        adjustments = self.data_loader.rebalance_horizons(deficits, focus_assets)
        synthetic = self.data_loader.backfill_shortfall(deficits, focus_assets)
        if synthetic:
            adjustments["synthetic"] = synthetic
        self._horizon_bias = self.data_loader.horizon_bias_snapshot()
        if adjustments:
            self.metrics.feedback(
                "training",
                severity=FeedbackSeverity.INFO,
                label="horizon_rebalance",
                details={"deficits": deficits, "adjustments": adjustments},
            )

    def _adjust_horizon_bias_from_confusion(self) -> None:
        summary = self._last_confusion_summary or {}
        horizons = summary.get("horizons") if isinstance(summary, dict) else None
        if not isinstance(horizons, dict) or not horizons:
            return
        bucket_scores: Dict[str, float] = {"short": 0.0, "mid": 0.0, "long": 0.0}
        bucket_samples: Dict[str, int] = {"short": 0, "mid": 0, "long": 0}
        for label, metrics in horizons.items():
            if not isinstance(metrics, dict):
                continue
            horizon_sec = metrics.get("horizon_seconds") or self._confusion_windows.get(label)
            if horizon_sec is None:
                continue
            samples = int(metrics.get("samples", 0))
            if samples <= 0:
                continue
            try:
                horizon_val = int(float(horizon_sec))
            except (TypeError, ValueError):
                continue
            if horizon_val <= 30 * 60:
                bucket = "short"
            elif horizon_val <= 24 * 3600:
                bucket = "mid"
            else:
                bucket = "long"
            f1_score = self._safe_float(metrics.get("f1_score", metrics.get("precision", 0.0)))
            bucket_scores[bucket] += f1_score * samples
            bucket_samples[bucket] += samples
        total_samples = sum(bucket_samples.values())
        if total_samples <= 0:
            return
        overall_f1 = sum(bucket_scores.values()) / max(total_samples, 1)
        if overall_f1 <= 0:
            return
        min_samples = int(os.getenv("HORIZON_BIAS_MIN_SAMPLES", "24"))
        current_bias = self.data_loader.horizon_bias_snapshot()
        updates: Dict[str, float] = {}
        bucket_snapshot: Dict[str, float] = {}
        for bucket, samples in bucket_samples.items():
            if samples < min_samples:
                continue
            bucket_f1 = bucket_scores[bucket] / max(samples, 1)
            bucket_snapshot[bucket] = bucket_f1
            delta = overall_f1 - bucket_f1
            if abs(delta) < 0.05:
                continue
            boost = 1.0 + max(-0.2, min(0.2, delta))
            updates[bucket] = current_bias.get(bucket, 1.0) * boost
        if not updates:
            return
        tuned = self.data_loader.tune_horizon_bias(updates)
        if tuned:
            self._horizon_bias = self.data_loader.horizon_bias_snapshot()
            self.metrics.feedback(
                "training",
                severity=FeedbackSeverity.INFO,
                label="horizon_bias_tuned",
                details={
                    "overall_f1": overall_f1,
                    "bucket_f1": bucket_snapshot,
                    "updated_bias": tuned,
                },
            )

    def _horizon_sample_weights(
        self,
        sample_meta: Dict[str, Any],
        *,
        sample_count: int,
    ) -> Optional[np.ndarray]:
        records = sample_meta.get("records") if isinstance(sample_meta, dict) else None
        if not isinstance(records, list) or not records:
            return None
        weights = np.ones(sample_count, dtype=np.float32)
        short_cutoff = 30 * 60
        mid_cutoff = 24 * 3600
        for idx, record in enumerate(records[:sample_count]):
            if not isinstance(record, dict):
                continue
            horizons = record.get("horizons") or {}
            if not isinstance(horizons, dict) or not horizons:
                continue
            bucket_weights: List[float] = []
            for key in horizons.keys():
                try:
                    horizon_sec = int(float(key))
                except (TypeError, ValueError):
                    continue
                if horizon_sec <= short_cutoff:
                    bucket = "short"
                elif horizon_sec <= mid_cutoff:
                    bucket = "mid"
                else:
                    bucket = "long"
                bucket_weights.append(float(self._horizon_bias.get(bucket, 1.0)))
            if bucket_weights:
                weights[idx] = max(0.5, float(np.mean(bucket_weights)))
        mean_weight = float(np.mean(weights)) if weights.size else 1.0
        if mean_weight > 0:
            weights = weights / mean_weight
        return weights.astype(np.float32)

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
                self.calibration_scale = float(tp_state.get("calibration_scale", self.calibration_scale))
                self.calibration_offset = float(tp_state.get("calibration_offset", self.calibration_offset))
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
            "calibration_scale": float(self.calibration_scale),
            "calibration_offset": float(self.calibration_offset),
        }
        try:
            self.db.save_state(state)
        except Exception:
            pass

    def _safe_float(self, value: Any) -> float:
        if isinstance(value, dict):
            for key in ("value", "score", "mean", "avg"):
                if key in value:
                    return self._safe_float(value[key])
            if len(value) == 1:
                try:
                    return self._safe_float(next(iter(value.values())))
                except Exception:
                    pass
            return 0.0
        if isinstance(value, (list, tuple, set)):
            iterator = iter(value)
            try:
                first = next(iterator)
            except StopIteration:
                return 0.0
            return self._safe_float(first)
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:
                return 0.0
        try:
            return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
        except Exception:
            return 0.0

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
            "spread_floor": float(os.getenv("SCHEDULER_SPREAD_FLOOR", "0.002")),
            "slippage_bps": float(os.getenv("SCHEDULER_SLIPPAGE_BPS", "50")),
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
        calibration = self._calibrate_platt(price_dir_scores, price_dir_true)
        if calibration:
            summary["calibration_scale"] = self._safe_float(calibration.get("scale", 1.0))
            summary["calibration_offset"] = self._safe_float(calibration.get("offset", 0.0))
            summary["calibration_log_loss"] = self._safe_float(calibration.get("log_loss", 0.0))

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
            anchor_label, anchor_payload = self._select_confusion_anchor(confusion_report)
            if anchor_label and anchor_payload:
                anchor_f1 = self._safe_float(anchor_payload.get("f1_score", 0.0))
                margin_boost = max(0.0, summary.get("ghost_realized_margin", 0.0))
                summary["margin_confidence"] = self._safe_float(anchor_f1 * (1.0 + margin_boost))
                summary["confusion_anchor"] = anchor_label
                summary["anchor_threshold"] = self._safe_float(
                    anchor_payload.get("threshold", self.decision_threshold)
                )
                self._auto_tune_threshold(anchor_payload)
            self._persist_confusion_report(confusion_report)
            self._persist_confusion_eval(summary)
            capped = self._cap_false_positive_rate(confusion_report)
            if capped is not None:
                summary["best_threshold"] = self._safe_float(capped)
                current_fp = self._safe_float(
                    summary.get("false_positive_rate_best", summary.get("false_positive_rate", 1.0))
                )
                summary["false_positive_rate_best"] = min(current_fp, self.max_false_positive_rate)
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
            objective_thr = (
                self._thr_precision_weight * self._safe_float(metrics_thr.get("precision", 0.0))
                + self._thr_recall_weight * self._safe_float(metrics_thr.get("recall", 0.0))
                + profit_factor_thr
            )
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
            payload["thresholds_tested"] = [float(value) for value in sweep.keys()]
            payload["curve"] = {f"{thr:.4f}": summary.to_dict() for thr, summary in sweep.items()}
            report[label] = payload
        return report

    def _summarize_confusion_report(self, report: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if not report:
            return {}
        summary: Dict[str, Any] = {"horizons": {}}
        dataset_positive = self._safe_float(self._last_sample_meta.get("positive_ratio", 0.0))
        positive_rate = dataset_positive if dataset_positive > 0 else 0.5
        total_samples = 0
        best_label: Optional[str] = None
        best_score = float("-inf")
        for label, payload in report.items():
            metrics = {
                "precision": self._safe_float(payload.get("precision", 0.0)),
                "recall": self._safe_float(payload.get("recall", 0.0)),
                "f1_score": self._safe_float(payload.get("f1_score", 0.0)),
                "samples": int(payload.get("samples", 0)),
                "false_positive_rate": self._safe_float(payload.get("false_positive_rate", 1.0)),
                "threshold": self._safe_float(payload.get("threshold", self.decision_threshold)),
            }
            metrics["lift"] = round(metrics["precision"] / max(positive_rate, 1e-3), 3) if metrics["precision"] else 0.0
            metrics["horizon_seconds"] = self._confusion_windows.get(label)
            total_samples += metrics["samples"]
            summary["horizons"][label] = metrics
            score = metrics["f1_score"] + metrics["precision"]
            if score > best_score:
                best_score = score
                best_label = label
        summary["dominant"] = best_label
        summary["total_samples"] = total_samples
        return summary

    def _select_confusion_anchor(
        self, report: Dict[str, Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        if not report:
            return None, None
        preferred = ("5m", "15m", "1h", "6h")
        for label in preferred:
            payload = report.get(label)
            if payload:
                return label, payload
        best_label: Optional[str] = None
        best_payload: Optional[Dict[str, Any]] = None
        best_score = float("-inf")
        for label, payload in report.items():
            f1 = self._safe_float(payload.get("f1_score", 0.0))
            if f1 > best_score:
                best_score = f1
                best_label = label
                best_payload = payload
        return best_label, best_payload

    def _persist_confusion_report(self, report: Dict[str, Dict[str, Any]]) -> None:
        if not report:
            self._last_confusion_report = {}
            return
        self._last_confusion_report = report
        self._last_confusion_refresh = time.time()
        self._last_confusion_summary = self._summarize_confusion_report(report)
        self._adjust_horizon_bias_from_confusion()
        self._last_transition_plan = self._build_transition_plan()
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

    def _persist_confusion_eval(self, summary: Dict[str, Any]) -> None:
        report_payload: Dict[str, Any] = (
            dict(self._last_confusion_report) if isinstance(self._last_confusion_report, dict) else {}
        )
        payload = {
            "updated_at": int(time.time()),
            "iteration": int(self.iteration),
            "dir_accuracy": self._safe_float(summary.get("dir_accuracy", 0.0)),
            "ghost_trades": int(summary.get("ghost_trades", 0)),
            "ghost_pred_margin": self._safe_float(summary.get("ghost_pred_margin", 0.0)),
            "ghost_realized_margin": self._safe_float(summary.get("ghost_realized_margin", 0.0)),
            "ghost_win_rate": self._safe_float(summary.get("ghost_win_rate", 0.0)),
            "confusion_anchor": summary.get("confusion_anchor"),
            "anchor_threshold": self._safe_float(summary.get("anchor_threshold", self.decision_threshold)),
            "best_threshold": self._safe_float(summary.get("best_threshold", self.decision_threshold)),
            "false_positive_rate": self._safe_float(summary.get("false_positive_rate_best", 1.0)),
            "margin_confidence": self._safe_float(summary.get("margin_confidence", 0.0)),
            "horizon_summary": self._last_confusion_summary,
        }
        path = Path("data/reports/confusion_eval.json")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass
        try:
            path = Path("data/reports/confusion_matrices.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "iteration": int(self.iteration),
                "updated_at": int(time.time()),
                "confusion": report_payload,
                "decision_threshold": float(self.decision_threshold),
                "summary": self._last_confusion_summary,
                "transition_plan": self._last_transition_plan,
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass
        self._persist_live_readiness_snapshot()

    def _load_cached_confusion_report(self) -> None:
        path = Path("data/reports/confusion_matrices.json")
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        report = payload.get("confusion")
        if isinstance(report, dict) and report:
            self._last_confusion_report = report
            self._last_confusion_summary = self._summarize_confusion_report(report)
            self._adjust_horizon_bias_from_confusion()
            self._last_transition_plan = self._build_transition_plan()
            threshold = payload.get("decision_threshold")
            if threshold is not None:
                try:
                    self.decision_threshold = float(threshold)
                except Exception:
                    pass
            updated_at = payload.get("updated_at")
            if updated_at is not None:
                try:
                    self._last_confusion_refresh = float(updated_at)
                except Exception:
                    self._last_confusion_refresh = time.time()
            else:
                self._last_confusion_refresh = time.time()

    def ensure_confusion_fresh(self, *, max_age: Optional[float] = None, force: bool = False) -> bool:
        if max_age is None:
            try:
                max_age = float(os.getenv("CONFUSION_REFRESH_MAX_AGE", "900"))
            except ValueError:
                max_age = 900.0
        now = time.time()
        needs_refresh = force
        if not needs_refresh:
            if not self._last_confusion_report:
                needs_refresh = True
            elif max_age > 0 and (now - self._last_confusion_refresh) > max_age:
                needs_refresh = True
        if not needs_refresh:
            return True
        refreshed = self.prime_confusion_windows(force=True)
        if refreshed:
            self._last_confusion_refresh = time.time()
        return refreshed

    def _cap_false_positive_rate(self, report: Dict[str, Dict[str, Any]]) -> Optional[float]:
        if not report:
            return None
        _, anchor = self._select_confusion_anchor(report)
        if not anchor:
            return None
        curve = anchor.get("curve") or {}
        if not isinstance(curve, dict):
            return None
        ordered = sorted(
            (entry for entry in curve.values() if isinstance(entry, dict)),
            key=lambda entry: float(entry.get("threshold", 0.5)),
        )
        for entry in ordered:
            try:
                fp_rate = float(entry.get("false_positive_rate", 1.0))
            except Exception:
                continue
            if fp_rate > self.max_false_positive_rate:
                continue
            try:
                threshold = float(entry.get("threshold"))
            except Exception:
                threshold = self.decision_threshold
            self.decision_threshold = float(max(0.05, min(0.95, threshold)))
            return self.decision_threshold
        return None

    def _auto_tune_threshold(self, anchor: Dict[str, Any]) -> None:
        if not anchor:
            return
        target_f1 = float(os.getenv("AUTO_THRESHOLD_MIN_F1", "0.62"))
        f1_score = self._safe_float(anchor.get("f1_score", 0.0))
        if f1_score < target_f1:
            return
        try:
            candidate = float(anchor.get("threshold", self.decision_threshold))
        except Exception:
            return
        if candidate <= 0.0:
            return
        blended = round((self.decision_threshold * 0.7) + (candidate * 0.3), 4)
        if abs(blended - self.decision_threshold) < 1e-4:
            return
        previous = self.decision_threshold
        self.decision_threshold = blended
        self._save_state()
        try:
            self.metrics.feedback(
                "training",
                severity=FeedbackSeverity.INFO,
                label="threshold_auto_tune",
                details={
                    "previous": previous,
                    "updated": blended,
                    "anchor_threshold": candidate,
                    "anchor_f1": f1_score,
                },
            )
        except Exception:
            pass

    def _persist_live_readiness_snapshot(self) -> None:
        snapshot = self.live_readiness_report()
        try:
            path = Path("data/reports/live_readiness.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                **snapshot,
                "iteration": int(self.iteration),
                "updated_at": int(time.time()),
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _ghost_validation(self) -> Dict[str, Any]:
        metrics = getattr(self, "metrics", None)
        if metrics is None:
            db = getattr(self, "db", None)
            if db is None:
                return {
                    "ready": False,
                    "reason": "no_metrics",
                    "samples": 0,
                    "win_rate": 0.0,
                    "avg_profit": 0.0,
                    "tail_risk": 0.0,
                    "tail_guardrail": float(os.getenv("GHOST_TAIL_GUARDRAIL", "0.08")),
                }
            metrics = MetricsCollector(db)
        self.metrics = metrics
        lookback = int(os.getenv("GHOST_VALIDATION_LOOKBACK_SEC", str(getattr(self, "focus_lookback_sec", 172800))))
        try:
            trades = metrics.ghost_trade_snapshot(limit=500, lookback_sec=lookback)
        except Exception:
            trades = []
        summary = metrics.aggregate_trade_metrics(trades)
        profit_dist = distribution_report([t.profit for t in trades])
        tail_guard = float(os.getenv("GHOST_TAIL_GUARDRAIL", "0.08"))
        drawdown_guard = float(os.getenv("GHOST_MAX_DRAWDOWN", "0"))
        loss_rate_guard = float(os.getenv("GHOST_MAX_LOSS_RATE", "0.6"))
        loss_streak_guard = int(os.getenv("GHOST_MAX_LOSS_STREAK", "5"))
        dominance_guard = float(os.getenv("GHOST_MAX_SYMBOL_DOMINANCE", "0.82"))
        stale_guard = float(os.getenv("GHOST_MAX_STALE_SEC", str(min(lookback, 86400))))
        min_trades = max(5, int(getattr(self, "min_ghost_trades", 0)))
        min_win_rate = float(getattr(self, "min_ghost_win_rate", 0.0))
        min_margin = float(getattr(self, "min_realized_margin", 0.0))
        min_profit_factor = float(os.getenv("MIN_GHOST_PROFIT_FACTOR", "0.95"))
        win_rate = float(summary.get("win_rate", 0.0))
        avg_profit = float(summary.get("avg_profit", 0.0))
        profit_factor = float(summary.get("profit_factor", 1.0))
        tail_risk = abs(profit_dist.get("expected_shortfall_95", 0.0))
        symbol_counts = Counter([str(t.symbol or "UNKNOWN").upper() for t in trades])
        dominant_share = float(max(symbol_counts.values()) / max(1, len(trades))) if symbol_counts else 0.0
        now_ts = time.time()
        last_trade_ts = max((float(getattr(t, "exit_ts", 0.0)) for t in trades), default=0.0)
        last_trade_age = max(0.0, now_ts - last_trade_ts) if last_trade_ts > 0 else float("inf")
        stale_samples = stale_guard > 0 and (not trades or last_trade_age > stale_guard)
        losses = 0
        max_loss_streak = 0
        current_loss_streak = 0
        for trade in trades:
            profit_val = float(getattr(trade, "profit", 0.0))
            if profit_val <= 0:
                losses += 1
                current_loss_streak += 1
                max_loss_streak = max(max_loss_streak, current_loss_streak)
            else:
                current_loss_streak = 0
        wins = len(trades) - losses
        loss_rate = float(losses / max(1, len(trades)))
        cumulative = 0.0
        trough = 0.0
        for profit in (t.profit for t in trades):
            cumulative += float(profit)
            trough = min(trough, cumulative)
        max_drawdown = abs(trough)
        concentration_block = len(trades) >= max(5, min_trades) and dominant_share > dominance_guard
        win_headroom = 1.0 if min_win_rate <= 0 else max(0.0, min(1.0, win_rate / max(min_win_rate, 1e-9)))
        profit_headroom = 1.0 if min_profit_factor <= 0 else max(0.0, min(1.0, profit_factor / max(min_profit_factor, 1e-9)))
        tail_headroom = 1.0 if tail_guard <= 0 else max(0.0, 1.0 - min(1.0, tail_risk / max(tail_guard, 1e-9)))
        loss_headroom = 1.0 if loss_rate_guard <= 0 else max(0.0, 1.0 - min(1.0, loss_rate / max(loss_rate_guard, 1e-9)))
        drawdown_headroom = 1.0 if drawdown_guard <= 0 else max(0.0, 1.0 - min(1.0, max_drawdown / max(drawdown_guard, 1e-9)))
        health_components = [win_headroom, profit_headroom, tail_headroom, loss_headroom, drawdown_headroom]
        health_score = max(0.0, min(1.0, sum(health_components) / max(1, len(health_components))))
        if stale_samples:
            health_score = 0.0
        if concentration_block:
            health_score = min(health_score, 0.35)
        ready = (
            len(trades) >= min_trades
            and win_rate >= min_win_rate
            and avg_profit >= min_margin
            and profit_factor >= min_profit_factor
            and tail_risk <= tail_guard
            and (drawdown_guard <= 0 or max_drawdown <= drawdown_guard)
            and (loss_rate_guard <= 0 or loss_rate <= loss_rate_guard)
            and (loss_streak_guard <= 0 or max_loss_streak <= loss_streak_guard)
            and not stale_samples
            and not concentration_block
        )
        reason = ""
        if not ready:
            if stale_samples:
                reason = "stale_ghost_book"
            elif len(trades) < min_trades:
                reason = "insufficient_samples"
            elif wins <= 0:
                reason = "no_wins"
            elif win_rate < min_win_rate:
                reason = "low_win_rate"
            elif avg_profit < min_margin:
                reason = "negative_margin"
            elif profit_factor < min_profit_factor:
                reason = "weak_profit_factor"
            elif tail_risk > tail_guard:
                reason = "tail_risk"
            elif drawdown_guard > 0 and max_drawdown > drawdown_guard:
                reason = "drawdown"
            elif loss_rate_guard > 0 and loss_rate > loss_rate_guard:
                reason = "loss_rate"
            elif loss_streak_guard > 0 and max_loss_streak > loss_streak_guard:
                reason = "loss_streak"
            elif concentration_block:
                reason = "symbol_concentration"
        return {
            "ready": ready,
            "reason": reason,
            "samples": len(trades),
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "tail_risk": tail_risk,
            "tail_guardrail": tail_guard,
            "max_drawdown": max_drawdown,
            "drawdown_guardrail": drawdown_guard,
            "min_trades": min_trades,
            "min_win_rate": min_win_rate,
            "min_margin": min_margin,
            "profit_factor": profit_factor,
            "min_profit_factor": min_profit_factor,
            "loss_rate": loss_rate,
            "loss_rate_guardrail": loss_rate_guard,
            "max_loss_streak": max_loss_streak,
            "loss_streak_guardrail": loss_streak_guard,
            "last_trade_ts": last_trade_ts,
            "last_trade_age_sec": last_trade_age if math.isfinite(last_trade_age) else None,
            "stale_guardrail_sec": stale_guard,
            "stale": stale_samples,
            "symbol_dominance": dominant_share,
            "max_symbol_dominance": dominance_guard,
            "unique_symbols": len(symbol_counts),
            "health_score": health_score,
            "wins": wins,
            "losses": losses,
        }

    def _wallet_state(self) -> Dict[str, Any]:
        wallet = (os.getenv("TRADING_WALLET") or os.getenv("WALLET_NAME") or "guardian").strip().lower()
        min_capital_usd = float(os.getenv("LIVE_MIN_CAPITAL_USD", "50"))
        dust_threshold = float(os.getenv("WALLET_DUST_USD", "5"))
        stable_tokens = {"USDC", "USDT", "DAI", "BUSD", "USDBC", "USDC.E", "TUSD", "USDP"}
        native_tokens = {"ETH", "WETH", "MATIC", "BNB", "AVAX", "SOL"}
        stable_usd = 0.0
        native_usd = 0.0
        native_buffer_target = float(os.getenv("LIVE_NATIVE_BUFFER_USD", "5"))
        holdings = 0
        dust_tokens: List[str] = []
        try:
            db = getattr(self, "db", None)
            balances = db.fetch_balances_flat(wallet=wallet, chains=[PRIMARY_CHAIN], include_zero=False) if db else []
        except Exception as exc:
            return {
                "wallet": wallet,
                "stable_usd": 0.0,
                "native_usd": 0.0,
                "sparse": True,
                "fragmented": True,
                "min_capital_usd": min_capital_usd,
                "dust_threshold_usd": dust_threshold,
                "error": str(exc),
            }
        for row in balances:
            symbol = str(row["symbol"] or row["token"] or "").upper()
            usd_val = float(row["usd_amount"] or 0.0)
            holdings += 1
            if symbol in stable_tokens:
                stable_usd += usd_val
            if symbol in native_tokens:
                native_usd += usd_val
            if usd_val < dust_threshold and symbol:
                dust_tokens.append(symbol)
        fragment_ratio = (len(dust_tokens) / holdings) if holdings else 0.0
        fragmented = len(dust_tokens) >= 3 or fragment_ratio >= 0.5
        native_buffer_gap = max(0.0, native_buffer_target - native_usd)
        native_starved = native_buffer_gap > 0.0
        stable_deficit = max(0.0, min_capital_usd - stable_usd)
        sparse_reasons: List[str] = []
        if holdings == 0:
            sparse_reasons.append("empty_wallet")
        if stable_deficit > 0:
            sparse_reasons.append("stable_below_min")
        if native_starved:
            sparse_reasons.append("native_gas_low")
        if fragmented:
            sparse_reasons.append("fragmented")
        sparse = bool(stable_deficit > 0 or fragmented or native_starved or holdings == 0)
        return {
            "wallet": wallet,
            "stable_usd": stable_usd,
            "native_usd": native_usd,
            "sparse": sparse,
            "fragmented": fragmented,
            "fragment_ratio": fragment_ratio,
            "dust_tokens": dust_tokens[:8],
            "dust_threshold_usd": dust_threshold,
            "min_capital_usd": min_capital_usd,
            "stable_deficit_usd": stable_deficit,
            "native_buffer_gap_usd": native_buffer_gap,
            "native_buffer_target_usd": native_buffer_target,
            "native_starved": native_starved,
            "sparse_reasons": sparse_reasons,
        }

    def live_readiness_report(self) -> Dict[str, Any]:
        # Allow environment overrides to tune promotion/readiness thresholds.
        ready_precision_target = float(os.getenv("LIVE_READY_PRECISION", "0.55"))
        ready_recall_target = float(os.getenv("LIVE_READY_RECALL", "0.50"))
        ready_samples_target = int(os.getenv("LIVE_READY_SAMPLES", "48"))
        mini_precision_target = float(os.getenv("LIVE_MINI_PRECISION", "0.45"))
        mini_recall_target = float(os.getenv("LIVE_MINI_RECALL", "0.45"))
        mini_samples_target = int(os.getenv("LIVE_MINI_SAMPLES", "20"))
        allow_mini_as_ready = (os.getenv("LIVE_ALLOW_MINI_READY", "0") or "0").lower() in {"1", "true", "yes", "on"}
        ghost_check = self._ghost_validation()
        wallet_state = self._wallet_state()

        report = self._last_confusion_report or {}
        if report and (not isinstance(self._last_confusion_summary, dict) or not self._last_confusion_summary.get("horizons")):
            self._last_confusion_summary = self._summarize_confusion_report(report)
        if not report:
            # Fallback: use the latest candidate feedback metrics if confusion data is missing.
            fb = self._last_candidate_feedback or {}
            precision_fb = float(fb.get("precision", 0.0))
            recall_fb = float(fb.get("recall", 0.0))
            samples_fb = int(fb.get("ghost_trades", fb.get("samples", 0)))
            fpr_fb = float(fb.get("false_positive_rate", 1.0))
            win_fb = float(fb.get("ghost_win_rate", 0.0))
            has_active = self.load_active_model() is not None
            ready_fb = (
                precision_fb >= ready_precision_target
                and recall_fb >= ready_recall_target
                and samples_fb >= min(ready_samples_target, 32)
                and fpr_fb <= self.max_false_positive_rate
                and win_fb >= self.min_ghost_win_rate
            )
            mini_ready_fb = (
                precision_fb >= mini_precision_target
                and recall_fb >= mini_recall_target
                and samples_fb >= mini_samples_target
                and fpr_fb <= self.max_false_positive_rate
                and win_fb >= max(0.1, self.min_ghost_win_rate * 0.5)
            )
            if not ready_fb and not mini_ready_fb and has_active:
                mini_ready_fb = True
                fpr_fb = min(fpr_fb, self.max_false_positive_rate)
                win_fb = max(win_fb, self.min_ghost_win_rate * 0.5)
            if not ready_fb and mini_ready_fb and allow_mini_as_ready:
                ready_fb = True
                fpr_fb = min(fpr_fb, self.max_false_positive_rate)
                win_fb = max(win_fb, self.min_ghost_win_rate * 0.5)
            report_fb = {
                "ready": ready_fb,
                "reason": "no_confusion_data" if not ready_fb else "",
                "horizon": "fb",
                "precision": precision_fb,
                "recall": recall_fb,
                "samples": samples_fb,
                "threshold": self.decision_threshold,
                "false_positive_rate": fpr_fb,
                "lift": 0.0,
                "dominant": None,
                "mini_ready": mini_ready_fb,
                "mini_reason": "" if mini_ready_fb else "insufficient_accuracy",
                "mini_precision": precision_fb,
                "mini_recall": recall_fb,
                "mini_samples": samples_fb,
                "mini_false_positive_rate": fpr_fb,
            }
            ghost_ready = bool(ghost_check.get("ready"))
            if report_fb["ready"] and not ghost_ready:
                report_fb["ready"] = False
                report_fb["reason"] = f"ghost_{ghost_check.get('reason') or 'not_ready'}"
            if report_fb["ready"] and wallet_state.get("sparse"):
                report_fb["ready"] = False
                sparse_reasons = wallet_state.get("sparse_reasons") or []
                detail = f":{','.join(sparse_reasons)}" if sparse_reasons else ""
                report_fb["reason"] = f"sparse_wallet{detail}"
            report_fb.update(
                {
                    "ghost_ready": ghost_ready,
                    "ghost_reason": ghost_check.get("reason", ""),
                    "ghost_samples": ghost_check.get("samples", 0),
                    "ghost_win_rate": ghost_check.get("win_rate", 0.0),
                    "ghost_tail_risk": ghost_check.get("tail_risk", 0.0),
                    "wallet_state": wallet_state,
                }
            )
            return report_fb
        anchor_label, anchor = self._select_confusion_anchor(report)
        if anchor is None:
            return {"ready": False, "reason": "no_confusion_data"}
        precision = float(anchor.get("precision", 0.0))
        recall = float(anchor.get("recall", 0.0))
        samples = int(anchor.get("samples", 0))
        threshold = float(anchor.get("threshold", self.decision_threshold))
        ready = precision >= ready_precision_target and recall >= ready_recall_target and samples >= ready_samples_target
        reason = "" if ready else "insufficient_accuracy"
        anchor_summary = self._last_confusion_summary.get("horizons", {}).get(anchor_label, {})
        false_positive_rate = float(anchor.get("false_positive_rate", anchor_summary.get("false_positive_rate", 1.0)))
        lift = self._safe_float(anchor_summary.get("lift", 0.0))
        horizons = self._last_confusion_summary.get("horizons", {})
        short_metrics = horizons.get("5m", horizons.get("15m", {}))
        mini_precision = float(short_metrics.get("precision", 0.0))
        mini_recall = float(short_metrics.get("recall", 0.0))
        mini_samples = int(short_metrics.get("samples", 0))
        mini_fpr = float(short_metrics.get("false_positive_rate", 1.0))
        mini_ready = mini_precision >= mini_precision_target and mini_recall >= mini_recall_target and mini_samples >= mini_samples_target and mini_fpr <= 0.9
        mini_reason = "" if mini_ready else "insufficient_accuracy"
        if not ready and not mini_ready and self.active_accuracy >= 0.6:
            mini_ready = True
            mini_reason = "active_model_bootstrap"
        if not mini_ready and precision >= mini_precision_target and recall >= max(0.3, mini_recall_target * 0.7) and samples >= min(mini_samples_target, 32):
            mini_ready = True
            mini_reason = "anchor_bootstrap"
        if not ready and mini_ready and allow_mini_as_ready:
            ready = True
            reason = "mini_ready"
        report_ready = {
            "ready": ready,
            "reason": reason,
            "horizon": anchor_label,
            "precision": precision,
            "recall": recall,
            "samples": samples,
            "threshold": threshold,
            "false_positive_rate": false_positive_rate,
            "lift": lift,
            "dominant": self._last_confusion_summary.get("dominant"),
            "mini_ready": mini_ready,
            "mini_reason": mini_reason,
            "mini_precision": mini_precision,
            "mini_recall": mini_recall,
            "mini_samples": mini_samples,
            "mini_false_positive_rate": mini_fpr,
        }
        ghost_ready = bool(ghost_check.get("ready"))
        if report_ready["ready"] and not ghost_ready:
            report_ready["ready"] = False
            report_ready["reason"] = f"ghost_{ghost_check.get('reason') or 'not_ready'}"
        if report_ready["ready"] and wallet_state.get("sparse"):
            report_ready["ready"] = False
            sparse_reasons = wallet_state.get("sparse_reasons") or []
            detail = f":{','.join(sparse_reasons)}" if sparse_reasons else ""
            report_ready["reason"] = f"sparse_wallet{detail}"
        report_ready.update(
            {
                "ghost_ready": ghost_ready,
                "ghost_reason": ghost_check.get("reason", ""),
                "ghost_samples": ghost_check.get("samples", 0),
                "ghost_win_rate": ghost_check.get("win_rate", 0.0),
                "ghost_tail_risk": ghost_check.get("tail_risk", 0.0),
                "wallet_state": wallet_state,
                "sparse_reasons": wallet_state.get("sparse_reasons"),
            }
        )
        return report_ready

    def confusion_summary(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._last_confusion_summary)) if self._last_confusion_summary else {}

    def transition_plan(self) -> Dict[str, Any]:
        if not self._last_transition_plan:
            self._last_transition_plan = self._build_transition_plan()
        return json.loads(json.dumps(self._last_transition_plan))

    def ghost_live_transition_plan(self) -> Dict[str, Any]:
        self._last_transition_plan = self._build_transition_plan()
        return json.loads(json.dumps(self._last_transition_plan))

    def _build_transition_plan(self) -> Dict[str, Any]:
        readiness = self.live_readiness_report()
        summary = self._last_confusion_summary or self._summarize_confusion_report(self._last_confusion_report or {})
        ready_ratio = float(os.getenv("SAVINGS_READY_RATIO", os.getenv("STABLE_CHECKPOINT_RATIO", "0.15")))
        bootstrap_ratio = float(os.getenv("SAVINGS_BOOTSTRAP_RATIO", os.getenv("PRE_EQUILIBRIUM_CHECKPOINT_RATIO", "0.05")))
        ghost_check = self._ghost_validation()
        wallet_state = self._wallet_state()
        min_live_capital = float(os.getenv("LIVE_MIN_CAPITAL_USD", "50"))
        min_clip_usd = float(os.getenv("LIVE_MIN_CLIP_USD", "10"))
        stable_usd = float(wallet_state.get("stable_usd", 0.0))
        native_usd = float(wallet_state.get("native_usd", 0.0))
        native_buffer_target = float(wallet_state.get("native_buffer_target_usd", os.getenv("LIVE_NATIVE_BUFFER_USD", "5")))
        native_buffer_gap = float(wallet_state.get("native_buffer_gap_usd", max(0.0, native_buffer_target - native_usd)))
        stable_deficit = float(wallet_state.get("stable_deficit_usd", max(0.0, min_live_capital - stable_usd)))
        wallet_sparse_reasons = wallet_state.get("sparse_reasons") or []
        ghost_samples = int(ghost_check.get("samples", 0))
        ghost_min_trades = int(ghost_check.get("min_trades", getattr(self, "min_ghost_trades", 0)))
        ghost_win_rate = float(ghost_check.get("win_rate", 0.0))
        min_ghost_win_rate = float(ghost_check.get("min_win_rate", getattr(self, "min_ghost_win_rate", 0.0)))
        profit_factor = float(ghost_check.get("profit_factor", 1.0))
        min_profit_factor = float(ghost_check.get("min_profit_factor", 1.0))
        loss_rate = float(ghost_check.get("loss_rate", 0.0))
        loss_rate_guard = float(ghost_check.get("loss_rate_guardrail", os.getenv("GHOST_MAX_LOSS_RATE", "0.6")))
        loss_streak = int(ghost_check.get("max_loss_streak", 0))
        loss_streak_guard = int(ghost_check.get("loss_streak_guardrail", os.getenv("GHOST_MAX_LOSS_STREAK", "5")))
        capital_deficit = max(
            0.0,
            float(wallet_state.get("min_capital_usd", min_live_capital)) - float(wallet_state.get("stable_usd", 0.0)),
            stable_deficit,
        )
        tail_risk = float(ghost_check.get("tail_risk", 0.0))
        tail_guard = float(ghost_check.get("tail_guardrail", 0.0))
        drawdown = float(ghost_check.get("max_drawdown", 0.0))
        drawdown_guard = float(ghost_check.get("drawdown_guardrail", 0.0))
        drawdown_breach = drawdown_guard > 0 and drawdown > drawdown_guard
        tail_block = tail_guard > 0 and tail_risk > tail_guard
        loss_rate_block = loss_rate_guard > 0 and loss_rate > loss_rate_guard
        loss_streak_block = loss_streak_guard > 0 and loss_streak > loss_streak_guard
        ghost_ready = bool(ghost_check.get("ready"))
        fragmented_wallet = bool(wallet_state.get("fragmented"))
        wallet_sparse = bool(wallet_state.get("sparse") or fragmented_wallet)
        native_starved = bool(wallet_state.get("native_starved", native_buffer_gap > 0))
        horizons = summary.get("horizons", {})
        allowed = 0
        horizon_plan: Dict[str, Any] = {}
        for label, metrics in horizons.items():
            allowed_flag = metrics.get("precision", 0.0) >= 0.6 and metrics.get("samples", 0) >= 48
            if allowed_flag:
                allowed += 1
            horizon_plan[label] = {
                **metrics,
                "allowed": allowed_flag,
            }
        coverage = allowed / max(1, len(horizon_plan)) if horizon_plan else 0.0
        plan = {
            "live_ready": bool(readiness.get("ready")) if isinstance(readiness, dict) else False,
            "anchor": readiness.get("horizon") if isinstance(readiness, dict) else None,
            "decision_threshold": readiness.get("threshold", self.decision_threshold)
            if isinstance(readiness, dict)
            else self.decision_threshold,
            "savings_ratio_ready": ready_ratio,
            "savings_ratio_bootstrap": bootstrap_ratio,
            "ready_reason": readiness.get("reason", "") if isinstance(readiness, dict) else "",
            "horizons": horizon_plan,
            "coverage": coverage,
            "dominant": summary.get("dominant"),
        }
        safe_to_live = (
            plan["live_ready"]
            and ghost_ready
            and not wallet_sparse
            and not tail_block
            and capital_deficit <= 0
            and not loss_rate_block
            and not loss_streak_block
        )
        recommended_ratio = 0.0
        ghost_sample_buffer = max(0.25, min(1.0, ghost_samples / max(1, ghost_min_trades)))
        tail_headroom = 1.0
        if tail_guard > 0:
            tail_headroom = max(0.2, min(1.0, (tail_guard - tail_risk) / max(tail_guard, 1e-9)))
        drawdown_headroom = 1.0
        if drawdown_guard > 0:
            drawdown_headroom = max(0.2, min(1.0, (drawdown_guard - drawdown) / max(drawdown_guard, 1e-9)))
        loss_health = max(0.25, min(1.0, 1.0 - loss_rate))
        health_score = float(ghost_check.get("health_score", 1.0))
        ghost_risk_multiplier = ghost_sample_buffer * tail_headroom * loss_health * drawdown_headroom
        ghost_risk_multiplier *= max(0.2, min(1.0, health_score))
        if not ghost_ready or tail_block or drawdown_breach or loss_rate_block or loss_streak_block:
            ghost_risk_multiplier = 0.0
        if wallet_sparse:
            ghost_risk_multiplier = min(ghost_risk_multiplier, 0.25)
        if capital_deficit > 0:
            ghost_risk_multiplier = 0.0
        ghost_risk_multiplier = float(max(0.0, min(1.0, ghost_risk_multiplier)))
        validation_margin = min(
            ghost_win_rate - min_ghost_win_rate if min_ghost_win_rate else ghost_win_rate,
            profit_factor - min_profit_factor if min_profit_factor else profit_factor,
        )
        if safe_to_live:
            recommended_ratio = ready_ratio * ghost_sample_buffer * tail_headroom
        elif ghost_ready and not wallet_sparse and not tail_block and capital_deficit <= 0:
            recommended_ratio = bootstrap_ratio * ghost_sample_buffer * tail_headroom
        if recommended_ratio > 0:
            win_rate_headroom = 1.0
            if min_ghost_win_rate > 0:
                win_rate_headroom = max(0.25, min(1.0, ghost_win_rate / max(min_ghost_win_rate, 1e-6)))
            profit_factor_headroom = max(0.25, min(1.0, profit_factor / max(min_profit_factor, 1e-6)))
            recommended_ratio *= win_rate_headroom * profit_factor_headroom * loss_health * drawdown_headroom
        if recommended_ratio > 0 and (ghost_samples < max(1, ghost_min_trades) * 1.5 or validation_margin < 0):
            recommended_ratio *= 0.5
            ghost_risk_multiplier = min(ghost_risk_multiplier, 0.5)
        if drawdown_breach:
            recommended_ratio = 0.0
            safe_to_live = False
        if loss_rate_block or loss_streak_block:
            recommended_ratio = 0.0
            safe_to_live = False
        if recommended_ratio > 0:
            recommended_ratio *= ghost_risk_multiplier
        deployable_stable = max(0.0, stable_usd - capital_deficit - native_buffer_gap)
        capital_ratio_cap = 0.0
        if stable_usd > 0:
            capital_ratio_cap = max(0.0, min(1.0, deployable_stable / stable_usd))
        max_live_usd = float(os.getenv("LIVE_MAX_BOOTSTRAP_USD", "150"))
        first_tranche_cap = float(os.getenv("LIVE_FIRST_TRANCHE_USD", "50"))
        live_ratio_cap = 0.0
        if deployable_stable > 0:
            live_ratio_cap = min(1.0, max_live_usd / max(deployable_stable, 1e-9))
        # Keep the live allocation proportional to available capital and bounded by a minimal ramp cap.
        recommended_ratio = min(recommended_ratio, capital_ratio_cap or recommended_ratio, live_ratio_cap or 1.0)
        recommended_live_usd = recommended_ratio * deployable_stable
        min_clip_block = False
        if recommended_live_usd > 0 and recommended_live_usd < min_clip_usd:
            min_clip_block = True
            recommended_ratio = 0.0
            recommended_live_usd = 0.0
            ghost_risk_multiplier = min(ghost_risk_multiplier, 0.25)
        live_mode = "blocked"
        if recommended_ratio > 0 and safe_to_live:
            live_mode = "ready"
        elif recommended_ratio > 0:
            live_mode = "bootstrap"
        block_reason = ""
        if recommended_ratio == 0.0:
            if not ghost_ready:
                block_reason = "ghost_validation_block"
            elif drawdown_breach:
                block_reason = "ghost_drawdown_block"
            elif loss_rate_block:
                block_reason = "ghost_loss_rate"
            elif loss_streak_block:
                block_reason = "ghost_loss_streak"
            elif wallet_sparse or capital_deficit > 0:
                if capital_deficit > 0:
                    block_reason = "capital_deficit"
                elif native_starved:
                    block_reason = "native_gas_starved"
                else:
                    block_reason = "wallet_sparse"
            elif tail_block:
                block_reason = "tail_risk_block"
            elif min_clip_block:
                block_reason = "min_clip"
            elif not plan["live_ready"]:
                block_reason = "readiness_gate_block"
        plan["ghost_validation"] = ghost_check
        plan["wallet_state"] = wallet_state
        plan["risk_flags"] = {
            "ghost_ready": ghost_ready,
            "ghost_reason": ghost_check.get("reason", ""),
            "ghost_win_rate": ghost_win_rate,
            "tail_risk": tail_risk,
            "tail_guardrail": tail_guard,
            "tail_block": tail_block,
            "loss_rate": loss_rate,
            "loss_rate_guardrail": loss_rate_guard,
            "loss_rate_block": loss_rate_block,
            "max_loss_streak": loss_streak,
            "loss_streak_guardrail": loss_streak_guard,
            "loss_streak_block": loss_streak_block,
            "wallet_sparse": wallet_sparse,
            "capital_deficit": capital_deficit,
            "native_starved": native_starved,
            "live_safe": safe_to_live,
            "live_blocked_reason": block_reason,
            "ghost_samples": ghost_samples,
            "ghost_min_trades": ghost_min_trades,
            "live_mode": live_mode,
            "native_buffer_gap": native_buffer_gap,
            "native_buffer_target": native_buffer_target,
            "fragmented_wallet": fragmented_wallet,
            "fragment_ratio": wallet_state.get("fragment_ratio", 0.0),
            "max_live_start_usd": max_live_usd,
            "deployable_stable_usd": deployable_stable,
            "health_score": health_score,
            "sparse_reasons": wallet_sparse_reasons,
            "min_clip_usd": min_clip_usd,
            "min_clip_block": min_clip_block,
            "recommended_live_usd": recommended_live_usd,
            "stable_deficit_usd": stable_deficit,
        }
        bus_actions: List[Dict[str, Any]] = []

        def add_bus_action(action: str, reason: str, priority: int = 2, window_sec: Optional[int] = None, **kwargs: Any) -> None:
            for existing in bus_actions:
                if existing.get("action") == action and existing.get("reason") == reason:
                    return
            payload: Dict[str, Any] = {"action": action, "reason": reason, "priority": priority}
            if window_sec is not None:
                try:
                    payload["window_sec"] = int(window_sec)
                except Exception:
                    payload["window_sec"] = window_sec
            payload.update({k: v for k, v in kwargs.items() if v is not None})
            bus_actions.append(payload)

        swap_window_sec = int(os.getenv("BUS_SWAP_WINDOW_SEC", "900"))

        if tail_block or drawdown_breach:
            add_bus_action(
                "pause_live",
                "ghost_tail_risk" if not drawdown_breach else "ghost_drawdown",
                priority=0,
                tail_risk=tail_risk,
                guardrail=tail_guard,
                drawdown=drawdown,
                drawdown_guardrail=drawdown_guard,
            )
        if loss_rate_block or loss_streak_block:
            add_bus_action(
                "freeze_live",
                "ghost_loss_rate" if loss_rate_block else "ghost_loss_streak",
                priority=0,
                loss_rate=loss_rate,
                loss_rate_guardrail=loss_rate_guard,
                loss_streak=loss_streak,
                loss_streak_guardrail=loss_streak_guard,
            )
        if not ghost_ready:
            add_bus_action(
                "freeze_live",
                ghost_check.get("reason", "ghost_not_ready"),
                priority=0,
                samples=ghost_check.get("samples", 0),
            )
        if wallet_sparse:
            add_bus_action(
                "freeze_live",
                "sparse_wallet",
                priority=0,
                min_capital_usd=float(wallet_state.get("min_capital_usd", min_live_capital)),
                wallet=wallet_state.get("wallet"),
                reasons=wallet_sparse_reasons,
            )
        if capital_deficit > 0:
            add_bus_action(
                "swap_to_stable",
                "min_live_capital",
                priority=1,
                target_usd=capital_deficit,
                wallet=wallet_state.get("wallet"),
                window_sec=swap_window_sec,
            )
            if native_usd > 0:
                add_bus_action(
                    "swap_native_to_stable",
                    "cover_min_capital",
                    priority=1,
                    native_usd=native_usd,
                    target_usd=min(native_usd, capital_deficit),
                    wallet=wallet_state.get("wallet"),
                    window_sec=swap_window_sec,
                )
        if (
            native_buffer_gap > 0
            and capital_deficit <= 0
            and deployable_stable > 0
            and deployable_stable >= native_buffer_gap * 0.25
        ):
            add_bus_action(
                "swap_stable_to_native",
                "gas_buffer" if recommended_ratio > 0 else "preload_gas_buffer",
                priority=2,
                target_usd=min(native_buffer_gap, deployable_stable),
                wallet=wallet_state.get("wallet"),
                window_sec=swap_window_sec,
            )
        if native_starved and stable_usd <= 0 and not tail_block and not loss_rate_block and not loss_streak_block:
            add_bus_action(
                "freeze_live",
                "native_gas_starved",
                priority=0,
                native_buffer_gap_usd=native_buffer_gap,
            )
        if fragmented_wallet:
            add_bus_action(
                "consolidate_fragments",
                "wallet_fragmentation",
                priority=3,
                dust_tokens=(wallet_state.get("dust_tokens") or [])[:8],
                wallet=wallet_state.get("wallet"),
                dust_threshold_usd=wallet_state.get("dust_threshold_usd"),
            )
        bus_actions.sort(key=lambda act: (act.get("priority", 99), act.get("action") or ""))
        bus_actions_pending = bool(bus_actions)
        bus_freeze_actions = {act.get("action") for act in bus_actions}
        if bus_actions_pending and (
            wallet_sparse
            or capital_deficit > 0
            or native_starved
            or "freeze_live" in bus_freeze_actions
            or "pause_live" in bus_freeze_actions
        ):
            ghost_risk_multiplier = 0.0
            recommended_ratio = 0.0
            if not block_reason:
                block_reason = "bus_actions_pending"
        if bus_actions:
            ghost_risk_multiplier = min(ghost_risk_multiplier, 0.35)
            if recommended_ratio > 0:
                recommended_ratio = min(recommended_ratio, recommended_ratio * ghost_risk_multiplier)
        if recommended_ratio <= 0:
            live_mode = "blocked"
            recommended_live_usd = 0.0
        else:
            recommended_live_usd = recommended_ratio * deployable_stable
            if recommended_live_usd < min_clip_usd:
                min_clip_block = True
                recommended_ratio = 0.0
                recommended_live_usd = 0.0
                live_mode = "blocked"
                ghost_risk_multiplier = min(ghost_risk_multiplier, 0.25)
                if not block_reason:
                    block_reason = "min_clip"
        plan["bus_swap_actions"] = bus_actions
        plan["capital_plan"] = {
            "bootstrap_ratio": bootstrap_ratio,
            "ready_ratio": ready_ratio,
            "recommended_live_ratio": recommended_ratio,
            "min_live_capital_usd": min_live_capital,
            "sparse_wallet": bool(wallet_state.get("sparse")),
            "wallet_stable_usd": stable_usd,
            "wallet_native_usd": native_usd,
            "capital_deficit_usd": capital_deficit,
            "safe_live_bankroll_usd": max(0.0, deployable_stable),
            "ghost_win_rate": ghost_win_rate,
            "ghost_loss_rate": loss_rate,
            "ghost_loss_streak": loss_streak,
            "profit_factor": profit_factor,
            "native_buffer_target_usd": native_buffer_target,
            "native_buffer_gap_usd": native_buffer_gap,
            "native_starved": native_starved,
            "fragmented_wallet": fragmented_wallet,
            "fragment_ratio": wallet_state.get("fragment_ratio", 0.0),
            "max_live_start_usd": max_live_usd,
            "live_capital_cap_usd": min(max_live_usd, deployable_stable) if deployable_stable else 0.0,
            "bus_actions_pending": bool(bus_actions),
            "ghost_risk_multiplier": ghost_risk_multiplier,
            "deployable_stable_usd": deployable_stable,
            "stable_deficit_usd": stable_deficit,
            "min_clip_usd": min_clip_usd,
            "min_clip_block": min_clip_block,
        }
        plan["guardrails"] = {
            "min_ghost_trades": ghost_min_trades,
            "min_live_capital_usd": min_live_capital,
            "tail_guardrail": tail_guard,
            "live_mode": live_mode,
            "ghost_drawdown_guardrail": drawdown_guard,
            "max_live_start_usd": max_live_usd,
            "loss_rate_guardrail": loss_rate_guard,
            "loss_streak_guardrail": loss_streak_guard,
            "min_clip_usd": min_clip_usd,
        }
        plan["capital_plan"]["recommended_live_usd"] = recommended_live_usd
        plan["capital_plan"]["live_ramp_schedule"] = {
            "first_tranche_usd": round(min(first_tranche_cap, recommended_live_usd), 2) if recommended_live_usd > 0 else 0.0,
            "max_live_usd": max_live_usd,
            "deployable_stable_usd": deployable_stable,
            "first_tranche_cap_usd": first_tranche_cap,
        }
        plan["capital_plan"]["ghost_risk_multiplier"] = ghost_risk_multiplier
        plan["recommended_savings_ratio"] = max(0.0, recommended_ratio)
        ghost_halt_reason = ""
        if ghost_risk_multiplier <= 0.0:
            ghost_halt_reason = block_reason or ghost_check.get("reason", "") or ("capital_deficit" if capital_deficit > 0 else "")
        if bus_actions_pending and not ghost_halt_reason:
            ghost_halt_reason = "bus_actions_pending"
        plan["risk_flags"].update(
            {
                "ghost_drawdown": drawdown,
                "ghost_drawdown_guardrail": drawdown_guard,
                "ghost_drawdown_breach": drawdown_breach,
                "ghost_sample_buffer": ghost_sample_buffer,
                "ghost_validation_margin": validation_margin,
                "halt_live": tail_block
                or drawdown_breach
                or not ghost_ready
                or loss_rate_block
                or loss_streak_block
                or wallet_sparse
                or capital_deficit > 0
                or native_starved
                or bus_actions_pending
                or recommended_ratio <= 0,
                "halt_reason": block_reason
                or (ghost_check.get("reason", "") if not ghost_ready else "")
                or ("wallet_sparse" if wallet_sparse else "")
                or ("capital_deficit" if capital_deficit > 0 else "")
                or ("native_starved" if native_starved else "")
                or ("bus_actions_pending" if bus_actions_pending else ""),
                "bus_actions_pending": bus_actions_pending,
                "risk_budget_cap": recommended_ratio,
                "ghost_risk_multiplier": ghost_risk_multiplier,
                "recommended_live_usd": recommended_live_usd,
                "min_clip_block": min_clip_block,
                "min_clip_usd": min_clip_usd,
                "halt_ghost": ghost_risk_multiplier <= 0.0 or bus_actions_pending,
                "ghost_halt_reason": ghost_halt_reason,
            }
        )
        return plan

    def prime_confusion_windows(self, *, min_samples: int = 128, force: bool = False) -> bool:
        if self._last_confusion_report and not force:
            return True
        if not self._train_lock.acquire(blocking=False):
            return False
        try:
            model = self.ensure_active_model()
            dataset = self._prepare_dataset(
                batch_size=32,
                dataset_label="primer",
                focus_assets=None,
                selected_files=None,
                oversample=False,
            )
            inputs, targets, _ = dataset
            if inputs is None or targets is None:
                return False
            sample_count = inputs["price_vol_input"].shape[0] if "price_vol_input" in inputs else 0
            if sample_count < min_samples and not force:
                return False
            input_order = [tensor.name.split(":")[0] for tensor in model.inputs]
            target_order = list(MODEL_OUTPUT_ORDER)
            input_tensors = tuple(inputs[name] for name in input_order)
            target_tensors = tuple(targets[name] for name in target_order)
            eval_ds = (
                tf.data.Dataset.from_tensor_slices((input_tensors, target_tensors))
                .batch(32)
                .prefetch(tf.data.AUTOTUNE)
            )
            evaluation = self._evaluate_candidate(model, eval_ds, targets)
            confusion = evaluation.get("confusion_matrices")
            if confusion:
                self._persist_confusion_report(confusion)
                capped = self._cap_false_positive_rate(confusion)
                if capped is not None:
                    self._save_state()
                log_message(
                    "training",
                    "confusion matrices primed",
                    severity="info",
                    details={"samples": sample_count, "threshold": self.decision_threshold},
                )
                return True
            return False
        finally:
            self._train_lock.release()

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
        sample_meta = self.last_sample_meta()
        dataset_symbols = [str(symbol).upper() for symbol in sample_meta.get("symbols", []) if symbol]
        symbol_counts = Counter(dataset_symbols)
        total_symbol_samples = sum(symbol_counts.values())
        news_ratio = float(self._last_dataset_meta.get("news_coverage_ratio", 0.0))
        short_bias = float(self._horizon_bias.get("short", 1.0))
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
            coverage = 0.0
            if total_symbol_samples:
                coverage = symbol_counts.get(symbol, 0) / total_symbol_samples
                penalty = max(0.0, penalty - coverage * (0.5 + news_ratio))
            if short_bias > 1.0:
                penalty = penalty / short_bias
            ranked.append(
                {
                    "symbol": symbol,
                    "avg_profit": avg_profit,
                    "win_rate": win_rate,
                    "count": info["count"],
                    "coverage": coverage,
                    "news_ratio": news_ratio,
                    "score": penalty,
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        focus_assets = [
            item["symbol"] for item in ranked if item["score"] > 0.0
        ][: self.focus_max_assets]
        if not focus_assets and self.primary_symbol:
            focus_assets = [self.primary_symbol]

        manual_overrides: Dict[str, List[str]] = {}
        try:
            watchlists = load_watchlists(self.db)
        except Exception:
            watchlists = {}
        manual_ghost = [sym for sym in (watchlists or {}).get("ghost", []) if sym]
        if manual_ghost:
            merged: List[str] = []
            for sym in manual_ghost + focus_assets:
                if sym not in merged:
                    merged.append(sym)
            focus_assets = merged[: self.focus_max_assets] or manual_ghost[: self.focus_max_assets]
            manual_overrides["ghost"] = manual_ghost

        self.metrics.record(
            MetricStage.GHOST_TRADING,
            aggregate,
            category="ghost_snapshot",
            meta={
                "iteration": self.iteration,
                "focus_candidates": focus_assets,
                "ranked": ranked[: self.focus_max_assets],
                "manual_overrides": manual_overrides or None,
            },
        )
        focus_stats = {"aggregate": aggregate, "ranked": ranked}
        if manual_overrides:
            focus_stats["manual_overrides"] = manual_overrides
        return focus_assets, focus_stats

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

    def _calibrate_platt(self, probs: np.ndarray, labels: np.ndarray) -> Optional[Dict[str, float]]:
        probs = np.clip(probs.astype(np.float64), 1e-6, 1 - 1e-6)
        labels = labels.astype(np.float64).reshape(-1)
        if labels.size < 32:
            return None
        positives = float(np.sum(labels > 0.5))
        negatives = float(labels.size - positives)
        if positives == 0 or negatives == 0:
            return None
        logits = np.log(probs) - np.log1p(-probs)

        def _sigmoid(values: np.ndarray) -> np.ndarray:
            values = np.clip(values, -30.0, 30.0)
            return 1.0 / (1.0 + np.exp(-values))

        def _log_loss(pred: np.ndarray) -> float:
            pred = np.clip(pred, 1e-6, 1 - 1e-6)
            return float(-np.mean(labels * np.log(pred) + (1.0 - labels) * np.log1p(-pred)))

        base_loss = _log_loss(probs)
        best_loss = base_loss
        best_scale = 1.0
        best_offset = 0.0
        scales = np.linspace(0.5, 2.5, num=9)
        offsets = np.linspace(-1.5, 1.5, num=9)
        for scale in scales:
            scaled = logits * scale
            for offset in offsets:
                pred = _sigmoid(scaled + offset)
                loss = _log_loss(pred)
                if loss < best_loss:
                    best_loss = loss
                    best_scale = float(scale)
                    best_offset = float(offset)
        refine_scales = np.linspace(max(0.3, best_scale - 0.4), best_scale + 0.4, num=7)
        refine_offsets = np.linspace(best_offset - 0.6, best_offset + 0.6, num=7)
        for scale in refine_scales:
            scaled = logits * scale
            for offset in refine_offsets:
                pred = _sigmoid(scaled + offset)
                loss = _log_loss(pred)
                if loss < best_loss:
                    best_loss = loss
                    best_scale = float(scale)
                    best_offset = float(offset)
        if best_loss >= base_loss - 1e-4:
            return None
        return {
            "scale": best_scale,
            "offset": best_offset,
            "log_loss": best_loss,
        }

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
            "temperature_scale": float(self.temperature_scale),
            "calibration_scale": float(self.calibration_scale),
            "calibration_offset": float(self.calibration_offset),
        }
from services.news_archive import CryptoNewsArchiver
