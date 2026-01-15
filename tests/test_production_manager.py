from __future__ import annotations

import sys
import threading
import time
import types


def _install_tf_stub() -> None:
    """
    TensorFlow import can hang or be unavailable in lightweight CI.
    Provide a minimal stub so production module imports remain fast.
    """
    if "tensorflow" in sys.modules:
        return

    tf_mod = types.ModuleType("tensorflow")
    keras_mod = types.ModuleType("tensorflow.keras")
    callbacks_mod = types.ModuleType("tensorflow.keras.callbacks")
    losses_mod = types.ModuleType("tensorflow.keras.losses")
    models_mod = types.ModuleType("tensorflow.keras.models")
    layers_mod = types.ModuleType("tensorflow.keras.layers")
    optimizers_mod = types.ModuleType("tensorflow.keras.optimizers")
    metrics_mod = types.ModuleType("tensorflow.keras.metrics")
    data_mod = types.ModuleType("tensorflow.data")

    class _NoOp:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, *args, **kwargs):
            return None

    callbacks_mod.EarlyStopping = _NoOp
    losses_mod.BinaryCrossentropy = _NoOp

    def _dummy_model(*args, **kwargs):
        class _Model:
            def save(self, *args, **kwargs) -> None:
                return None

        return _Model()

    models_mod.load_model = lambda *args, **kwargs: _dummy_model()
    models_mod.clone_model = lambda base, *args, **kwargs: base
    layers_mod.TextVectorization = _NoOp
    optimizers_mod.Adam = _NoOp
    metrics_mod.AUC = _NoOp
    metrics_mod.MeanSquaredError = _NoOp

    class _Dataset:
        @staticmethod
        def from_tensor_slices(*args, **kwargs):
            class _DS:
                def batch(self, *args, **kwargs):
                    return self

                def prefetch(self, *args, **kwargs):
                    return self

            return _DS()

    data_mod.Dataset = _Dataset
    data_mod.AUTOTUNE = 1

    tf_mod.keras = keras_mod
    tf_mod.data = data_mod
    tf_mod.convert_to_tensor = lambda *args, **kwargs: None

    keras_mod.callbacks = callbacks_mod
    keras_mod.losses = losses_mod
    keras_mod.models = models_mod
    keras_mod.layers = layers_mod
    keras_mod.optimizers = optimizers_mod
    keras_mod.metrics = metrics_mod

    sys.modules.update(
        {
            "tensorflow": tf_mod,
            "tensorflow.keras": keras_mod,
            "tensorflow.keras.callbacks": callbacks_mod,
            "tensorflow.keras.losses": losses_mod,
            "tensorflow.keras.models": models_mod,
            "tensorflow.keras.layers": layers_mod,
            "tensorflow.keras.optimizers": optimizers_mod,
            "tensorflow.keras.metrics": metrics_mod,
            "tensorflow.data": data_mod,
        }
    )


_install_tf_stub()

from production import ProductionManager


def test_stale_news_task_is_replaced() -> None:
    # Build a minimal manager without running heavy initialisers.
    manager = ProductionManager.__new__(ProductionManager)
    manager._task_threads = {}
    manager._task_thread_start = {}
    manager._task_backoff_until = {}

    # Seed a stuck thread and mark it old enough to be considered stale.
    stuck = threading.Thread(target=lambda: time.sleep(2), daemon=True)
    stuck.start()
    manager._task_threads["news_enrichment"] = stuck
    manager._task_thread_start["news_enrichment"] = time.time() - 1.0

    executed = threading.Event()

    def fast_task() -> bool:
        executed.set()
        return True

    ok = ProductionManager._run_with_timeout(
        manager,
        fast_task,
        timeout=0.05,
        label="news_enrichment",
        backoff_on_timeout=0.1,
    )
    executed.wait(0.3)

    assert ok is True
    assert executed.is_set()


def test_running_task_enters_backoff_on_skip() -> None:
    manager = ProductionManager.__new__(ProductionManager)
    manager._task_threads = {}
    manager._task_thread_start = {}
    manager._task_backoff_until = {}

    # Seed a currently running thread that has not yet become stale.
    running = threading.Thread(target=lambda: time.sleep(0.5), daemon=True)
    running.start()
    manager._task_threads["news_enrichment"] = running
    manager._task_thread_start["news_enrichment"] = time.time()

    ok = ProductionManager._run_with_timeout(
        manager,
        lambda: True,
        timeout=0.05,
        label="news_enrichment",
        backoff_on_timeout=0.2,
    )

    assert ok is False
    assert manager._task_backoff_until.get("news_enrichment", 0.0) > time.time()
    assert manager._task_threads["news_enrichment"] is running
    running.join(1.0)
