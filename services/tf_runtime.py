from __future__ import annotations

import os
from typing import Optional

from services.system_profile import SystemProfile, detect_system_profile

_TF_CONFIGURED = False


def configure_tensorflow(profile: Optional[SystemProfile] = None) -> None:
    global _TF_CONFIGURED
    if _TF_CONFIGURED:
        return
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    profile = profile or detect_system_profile()
    try:
        import tensorflow as tf  # noqa: F401
    except Exception:
        return
    try:
        tf.config.set_visible_devices([], "GPU")  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(profile.max_threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, profile.max_threads // 2))
    except Exception:
        pass
    try:
        tf.config.experimental.enable_tensor_float_32_execution(False)  # type: ignore[attr-defined]
    except Exception:
        pass
    _TF_CONFIGURED = True
