from __future__ import annotations

import types

from services import tf_runtime
from services.system_profile import SystemProfile


class _ThreadingStub:
    def __init__(self) -> None:
        self.intra = None
        self.inter = None

    def set_intra_op_parallelism_threads(self, value: int) -> None:
        self.intra = value

    def set_inter_op_parallelism_threads(self, value: int) -> None:
        self.inter = value


def test_configure_tensorflow_uses_profile(monkeypatch):
    stub = types.SimpleNamespace()
    threading_stub = _ThreadingStub()
    stub.config = types.SimpleNamespace(
        set_visible_devices=lambda *_args, **_kwargs: None,
        threading=threading_stub,
        experimental=types.SimpleNamespace(enable_tensor_float_32_execution=lambda *_a, **_k: None),
    )
    monkeypatch.setattr(tf_runtime, "_TF_CONFIGURED", False, raising=False)
    monkeypatch.setattr(tf_runtime, "detect_system_profile", lambda: SystemProfile(4, 32.0, 4, False, False))
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", stub)
    tf_runtime.configure_tensorflow()
    assert threading_stub.intra == 4
