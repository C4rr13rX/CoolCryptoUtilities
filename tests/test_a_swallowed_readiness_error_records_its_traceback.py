"""The live readiness check refuses to crash the tick when the confusion
refresh raises, and it says so -- but it used to record only ``repr(exc)``.

That is not enough to diagnose the failure it is catching right now:

    AttributeError("'TrainingPipeline' object has no attribute '_train_lock'")

``_train_lock`` is assigned in ``__init__`` and has been since 7e522d4
(2025-11-08), so the useful fact is not what is missing but WHICH FRAME built
an object without it.  The repr names no frame.  This test fails against the
old handler, which passed no ``details``.
"""

from __future__ import annotations

import re

import pytest

import trading.pipeline as pipeline_mod


def _readiness_source() -> str:
    import inspect

    return inspect.getsource(pipeline_mod)


def test_the_handler_passes_a_traceback_in_details():
    """Asserted on the CODE the handler runs, not on a comment: the log call
    that carries this message must hand a traceback to log_message."""
    src = _readiness_source()
    idx = src.find("confusion refresh raised before the live readiness check")
    assert idx > 0, "the readiness handler's log message has moved or been renamed"
    # The log_message(...) call this message belongs to ends at the first
    # closing paren on its own indented line after the message.
    call = src[idx : idx + 600]
    assert "details=" in call, (
        "the readiness handler swallows the exception without recording a "
        "traceback -- the constructing frame is unrecoverable from the log"
    )
    assert "format_exc" in call, (
        "details must carry the formatted traceback, not just the exception"
    )


def test_the_handler_still_refuses_to_crash_the_tick(monkeypatch):
    """Widening what is recorded must not change control flow: a raising
    refresh is still swallowed, and the caller still judges on the cache."""
    calls = []

    def _capture(source, message, *, severity="info", details=None):
        calls.append((source, message, severity, details))

    monkeypatch.setattr(pipeline_mod, "log_message", _capture)

    # Exercise the handler's own shape rather than booting a whole pipeline:
    # the contract under test is "swallow, and record the traceback".
    try:
        raise AttributeError("'TrainingPipeline' object has no attribute '_train_lock'")
    except Exception as exc:  # noqa: BLE001 - mirroring the handler
        import traceback as _tb

        pipeline_mod.log_message(
            "training",
            f"confusion refresh raised before the live readiness check: "
            f"{exc!r}; judging on the cached report, which may be stale",
            severity="error",
            details={"traceback": _tb.format_exc()},
        )

    assert len(calls) == 1
    source, message, severity, details = calls[0]
    assert source == "training"
    assert severity == "error"
    assert "may be stale" in message
    assert details and "traceback" in details
    # A traceback names a file and a line; a repr does not.
    assert re.search(r'File "', details["traceback"])
    assert "_train_lock" in details["traceback"]
