from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from db import TradingDatabase, get_db
from model_definition import (
    ExponentialDecay,
    StateSaver,
    TimeFeatureLayer,
    build_multimodal_model,
    gaussian_nll_loss,
    zero_loss,
)
from trading.data_loader import HistoricalDataLoader
from trading.optimizer import BayesianBruteForceOptimizer


CUSTOM_OBJECTS = {
    "ExponentialDecay": ExponentialDecay,
    "TimeFeatureLayer": TimeFeatureLayer,
    "gaussian_nll_loss": gaussian_nll_loss,
    "zero_loss": zero_loss,
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

        inputs, targets = self._prepare_dataset(batch_size=32)
        if inputs is None or targets is None:
            print("[training] insufficient data for candidate training; skipping this cycle.")
            return None

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

        version = f"candidate-{int(time.time())}"
        path = self.model_dir / f"{version}.keras"
        model.save(path, include_optimizer=False)
        self.db.register_model_version(version=version, metrics={"score": score}, path=str(path), activate=False)

        result = {
            "version": version,
            "score": score,
            "path": path,
            "params": {"learning_rate": lr, "epochs": epochs},
        }

        if score >= self.promotion_threshold:
            self.promote_candidate(path, score=score, metadata=result)
        else:
            print(f"[training] candidate score {score:.3f} below promotion threshold {self.promotion_threshold:.3f}.")

        return result

    def promote_candidate(self, path: Path, *, score: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        active_path = self.model_dir / "active_model.keras"
        print(f"[training] promoting candidate {path.name} (score={score:.3f}) to active deployment.")
        tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False).save(active_path, include_optimizer=False)
        self.db.register_model_version(
            version=f"active-{int(time.time())}", metrics={"score": score}, path=str(active_path), activate=True
        )
        self._active_model = tf.keras.models.load_model(active_path, custom_objects=CUSTOM_OBJECTS, compile=False)
        if metadata:
            self.db.log_trade(
                wallet="system",
                chain="meta",
                symbol="MODEL",
                action="promote",
                status="success",
                details=metadata,
            )

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

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "model_dir": str(self.model_dir),
            "best_score": self.optimizer.best_score,
            "best_params": self.optimizer.best_params,
        }
