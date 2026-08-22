"""
Absorb TensorFlow imports on Android, where no wheel exists.

Several modules under ``services/`` and ``trading/`` import TensorFlow at
module scope.  On Android that import raises ``ModuleNotFoundError``, and
because those modules sit on the URLconf import path a single missing wheel
takes down the whole site -- the same failure mode the Lambda build hit with
``bs4``.

Rather than editing every call site, this registers a stub in
``sys.modules`` so ``import tensorflow`` succeeds and *using* it fails loudly
and specifically.  The distinction matters:

* importing must succeed, or unrelated views 500;
* calling must fail, and must say why, or a caller silently believes it
  trained a model that never existed.

``tf_available()`` lets callers branch deliberately, and the status endpoints
report ``tf_unavailable`` so the UI can say so instead of showing a blank
chart.

Importing this module is a no-op on any platform where the real TensorFlow is
installed, so it is safe on the desktop too.
"""

from __future__ import annotations

import importlib.util
import sys
import types

_ANDROID = hasattr(sys, "getandroidapilevel")


def tf_available() -> bool:
    """True when a real TensorFlow can be imported."""
    if "tensorflow" in sys.modules:
        return not getattr(sys.modules["tensorflow"], "__is_stub__", False)
    return importlib.util.find_spec("tensorflow") is not None


class TensorFlowUnavailable(RuntimeError):
    """Raised when on-device code actually tries to use TensorFlow."""


def _unavailable(*_args, **_kwargs):
    raise TensorFlowUnavailable(
        "TensorFlow is not available on Android. Model training and inference "
        "run off-device; see serverless/ for the remote path, or export the "
        "model to TFLite for on-device inference."
    )


class _StubModule(types.ModuleType):
    """Any attribute access yields a callable that raises on use."""

    __is_stub__ = True

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        # Submodules (tf.keras.layers, ...) must themselves be stubs so that
        # `from tensorflow.keras import Model` resolves at import time.
        child = _StubModule(f"{self.__name__}.{name}")
        sys.modules[child.__name__] = child
        setattr(self, name, child)
        return child

    def __call__(self, *args, **kwargs):
        return _unavailable(*args, **kwargs)


def install() -> bool:
    """Register the stub when TensorFlow is genuinely missing. Idempotent."""
    if importlib.util.find_spec("tensorflow") is not None:
        return False           # real TF present -- never shadow it
    if "tensorflow" in sys.modules:
        return getattr(sys.modules["tensorflow"], "__is_stub__", False)

    stub = _StubModule("tensorflow")
    stub.__version__ = "0.0.0+android-stub"
    sys.modules["tensorflow"] = stub
    # keras is imported directly in places, not only as tf.keras.
    for name in ("tensorflow.keras", "keras"):
        sys.modules[name] = _StubModule(name)
    return True


# Installed on import: modules that need it import this first, and by then it
# is too late to be explicit about ordering.
INSTALLED = install()
