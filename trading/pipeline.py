from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

        self.window_size = 60
        self.sent_seq_len = 24
        self.tech_count = 12
        self.data_loader = HistoricalDataLoader()
        self.iteration: int = 0
        self.active_accuracy: float = 0.0
        self._load_state()
        self._active_model: Optional[tf.keras.Model] = None

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

        inputs, targets = self._prepare_dataset(batch_size=32)
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

        promote = score >= self.promotion_threshold
        if promote and evaluation and self.active_accuracy:
            if evaluation.get("dir_accuracy", 0.0) < self.active_accuracy + 0.01:
                promote = False
                print(
                    "[training] retaining existing live model (%.3f) to gather more data before replacement."
                    % self.active_accuracy
                )
        if promote:
            self.promote_candidate(path, score=score, metadata=result, evaluation=evaluation)
        else:
            self._print_ghost_summary(evaluation)
            print(f"[training] candidate score {score:.3f} below promotion threshold {self.promotion_threshold:.3f}.")

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

    def _prepare_dataset(self, batch_size: int = 32) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        inputs, targets = self.data_loader.build_dataset(
            window_size=self.window_size,
            sent_seq_len=self.sent_seq_len,
            tech_count=self.tech_count,
        )
        if inputs is not None and targets is not None:
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
        }
        return summary

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "model_dir": str(self.model_dir),
            "best_score": self.optimizer.best_score,
            "best_params": self.optimizer.best_params,
        }
