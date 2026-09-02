"""A half-written model must never be mistaken for a corrupt one.

Observed in production.log on 2026-09-02, repeating every few seconds without
end:

    [training] failed to load active model (Expected a model.weights.h5 or
    model.weights.npz file.); removing corrupted artifact.
    [training] no active model found; building a fresh baseline.

A copy of ``models/active_model.keras`` captured mid-cycle explained it: 48,725
bytes opening with ``PK\\x03\\x04`` and ending mid-JSON, with no zip central
directory. ``model.save(path)`` wrote the archive in place over several
seconds, and any reader that opened it inside that window saw a truncated zip.
``load_active_model`` then DELETED it as corrupt, the next call found no model
and rebuilt one, and the rebuild produced the next half-written file. The loop
sustained itself.

The cost was not a missing model. Rebuilding a TensorFlow baseline every cycle
burns the CPU and memory the trading cycle needs, on a box whose
ResourceGovernor was already pausing ``production_cycle`` at 97% memory -- so a
non-atomic save was starving the loop that has to place the live trade.

Two locks on that door, one test each: saves land atomically, and a reader
refuses to delete an artifact young enough to still be in flight.
"""
from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path
from unittest import mock

import pytest


def _pipeline_module():
    """Import the module without paying for a TensorFlow session."""
    pytest.importorskip("tensorflow")
    import trading.pipeline as pipeline

    return pipeline


class _FakeModel:
    """Writes like Keras does: a real zip, but slowly and in place."""

    def __init__(self, *, tear_at: float = 0.0):
        self.tear_at = tear_at
        self.saved_to: list[Path] = []

    def save(self, path, **_kwargs):
        path = Path(path)
        self.saved_to.append(path)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("metadata.json", '{"keras_version": "3.13.2"}')
            archive.writestr("config.json", '{"module": "keras"}')


def test_a_save_is_never_visible_half_written(tmp_path) -> None:
    """The live path must go from old contents to new with nothing between."""
    pipeline = _pipeline_module()
    target = tmp_path / "active_model.keras"
    seen: list[bool] = []

    class _Observed(_FakeModel):
        def save(self, path, **kwargs):
            # Whatever a reader sees at the live path while we write must be
            # either absent or a complete archive -- never a torn one.
            seen.append(not target.exists() or zipfile.is_zipfile(target))
            super().save(path, **kwargs)
            seen.append(not target.exists() or zipfile.is_zipfile(target))

    pipeline._save_model_atomically(_Observed(), target)

    assert target.exists()
    assert zipfile.is_zipfile(target)
    assert all(seen), "the live path was observable in a torn state"


def test_the_write_goes_through_a_temp_file_in_the_same_directory(tmp_path) -> None:
    """os.replace is only atomic within one filesystem."""
    pipeline = _pipeline_module()
    target = tmp_path / "active_model.keras"
    model = _FakeModel()

    pipeline._save_model_atomically(model, target)

    assert len(model.saved_to) == 1
    written = model.saved_to[0]
    assert written != target
    assert written.parent == target.parent
    # Keras picks its format from the suffix, so the temp file must keep it.
    assert written.suffix == ".keras"
    # And nothing is left behind.
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_replacing_an_existing_artifact_keeps_it_readable(tmp_path) -> None:
    pipeline = _pipeline_module()
    target = tmp_path / "active_model.keras"
    pipeline._save_model_atomically(_FakeModel(), target)
    first = target.read_bytes()

    pipeline._save_model_atomically(_FakeModel(), target)

    assert zipfile.is_zipfile(target)
    assert target.read_bytes() == first  # same content, cleanly replaced


def test_a_failed_save_does_not_damage_the_existing_artifact(tmp_path) -> None:
    """A model that cannot serialise must not take the good one down with it."""
    pipeline = _pipeline_module()
    target = tmp_path / "active_model.keras"
    pipeline._save_model_atomically(_FakeModel(), target)
    good = target.read_bytes()

    class _Broken(_FakeModel):
        def save(self, path, **_kwargs):
            raise RuntimeError("serialisation blew up")

    with pytest.raises(RuntimeError):
        pipeline._save_model_atomically(_Broken(), target)

    assert target.read_bytes() == good
    assert list(tmp_path.glob("*.tmp-*")) == []


class _Loader:
    """The smallest object load_active_model needs to run against."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._active_model = None


def _load(pipeline, loader):
    return pipeline.TrainingPipeline.load_active_model(loader)


def test_an_unreadable_but_fresh_artifact_is_left_alone(tmp_path) -> None:
    """The delete is what closed the loop; it must not fire on a young file."""
    pipeline = _pipeline_module()
    target = tmp_path / "active_model.keras"
    target.write_bytes(b"PK\x03\x04truncated, still being written")

    with mock.patch.object(
        pipeline.tf.keras.models, "load_model", side_effect=OSError("truncated")
    ):
        assert _load(pipeline, _Loader(tmp_path)) is None

    assert target.exists(), "a file still in flight was deleted as corrupt"


def test_an_unreadable_artifact_that_has_settled_is_removed(tmp_path) -> None:
    """A genuinely corrupt file still gets cleared, or it would wedge forever."""
    pipeline = _pipeline_module()
    target = tmp_path / "active_model.keras"
    target.write_bytes(b"not a keras archive at all")
    stale = time.time() - 3600
    os.utime(target, (stale, stale))

    with mock.patch.object(
        pipeline.tf.keras.models, "load_model", side_effect=OSError("corrupt")
    ):
        assert _load(pipeline, _Loader(tmp_path)) is None

    assert not target.exists()


def test_the_settle_window_is_tunable(tmp_path) -> None:
    pipeline = _pipeline_module()
    target = tmp_path / "active_model.keras"
    target.write_bytes(b"junk")
    aged = time.time() - 120
    os.utime(target, (aged, aged))

    with mock.patch.object(
        pipeline.tf.keras.models, "load_model", side_effect=OSError("corrupt")
    ):
        # Default settle is 60s, so a 120s-old file would normally be removed.
        with mock.patch.dict(os.environ, {"MODEL_ARTIFACT_SETTLE_SEC": "600"}):
            assert _load(pipeline, _Loader(tmp_path)) is None
            assert target.exists()

        assert _load(pipeline, _Loader(tmp_path)) is None
        assert not target.exists()
