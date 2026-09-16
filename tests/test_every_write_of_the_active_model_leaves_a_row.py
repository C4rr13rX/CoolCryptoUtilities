"""Every write of models/active_model.keras must leave a row that says WHO wrote it.

models/active_model.keras had THREE writers and one registrar. promote_candidate
registered a model_versions row and ran the price_mu calibration guard;
TrainingPipeline.ensure_active_model and _ensure_asset_embedding_capacity wrote
the same served path with neither. That is why model_versions held 0 rows on
2026-09-11 while the served artifact had been rebuilt that morning, and why a
1800x price_mu regression could not be attributed to any artifact: nothing
recorded that an artifact had been written at all.

The compounding half is the cadence. bot.py's background refinement loop read
os.path.exists(active_model.keras) to decide whether it had a model, so a
bootstrap write of a NEVER-TRAINED artifact moved it from 300s to 900s -- a 3x
slowdown of the only loop that would train one, entered precisely because there
was nothing trained. These tests pin both halves.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db import TradingDatabase  # noqa: E402
from trading import pipeline as pipeline_mod  # noqa: E402
from trading.pipeline import (  # noqa: E402
    TrainingPipeline,
    is_artifact_trained,
    probe_artifact_training,
    refinement_cadence_for_artifact,
)


class _StubPipeline:
    """Carries only what _register_active_artifact_write actually touches."""

    def __init__(self, db: TradingDatabase) -> None:
        self.db = db

    register = TrainingPipeline._register_active_artifact_write


def _rows(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT version, metrics, path, is_active FROM model_versions ORDER BY id")
        return [
            {"version": r[0], "metrics": json.loads(r[1] or "{}"), "path": r[2], "is_active": r[3]}
            for r in cur.fetchall()
        ]
    finally:
        conn.close()


def _fresh_db(tmp_path: Path):
    db_path = tmp_path / "registry.db"
    return TradingDatabase(str(db_path)), db_path


def _tiny_model(keras):
    """A model whose only 1-D parameters are biases: pristine until it is fit."""
    inp = keras.layers.Input(shape=(3,), name="x")
    hidden = keras.layers.Dense(4, activation="tanh", name="h")(inp)
    out = keras.layers.Dense(1, name="y")(hidden)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.5), loss="mse")
    return model


# ---------------------------------------------------------------------------
# The registry row, and that there is exactly ONE of it per write.
# ---------------------------------------------------------------------------


def test_a_bootstrap_write_leaves_exactly_one_row_naming_it_a_bootstrap(tmp_path):
    db, db_path = _fresh_db(tmp_path)
    artifact = tmp_path / "active_model.keras"
    artifact.write_bytes(b"not a real archive")

    assert _rows(db_path) == [], "the registry must start empty for this to mean anything"

    stub = _StubPipeline(db)
    row_id = stub.register(artifact, "bootstrap_build", {"asset_vocab_size": 7})

    rows = _rows(db_path)
    assert len(rows) == 1, f"one write must leave exactly one row, got {len(rows)}"
    assert row_id is not None
    row = rows[0]
    assert row["metrics"]["provenance"] == "bootstrap_build"
    assert row["metrics"]["calibration_guarded"] is False, (
        "the bootstrap write is not refused on calibration grounds, but the row "
        "must not claim it was guarded"
    )
    assert row["metrics"]["trained"] is not True, (
        "a freshly BUILT artifact was never fit; the row must never say trained"
    )
    assert row["metrics"]["asset_vocab_size"] == 7
    assert row["path"] == str(artifact)
    assert row["is_active"]


def test_the_asset_vocab_rebuild_is_a_distinct_provenance_not_a_promotion(tmp_path):
    db, db_path = _fresh_db(tmp_path)
    artifact = tmp_path / "active_model.keras"
    artifact.write_bytes(b"not a real archive")

    stub = _StubPipeline(db)
    stub.register(artifact, "bootstrap_build", None)
    stub.register(artifact, "asset_vocab_expansion", {"asset_vocab_from": 1, "asset_vocab_to": 6})

    rows = _rows(db_path)
    assert [r["metrics"]["provenance"] for r in rows] == ["bootstrap_build", "asset_vocab_expansion"], (
        "two writes, two rows, each naming its own writer"
    )
    assert [bool(r["is_active"]) for r in rows] == [False, True], (
        "the latest write is the served one"
    )


def test_the_bootstrap_build_is_not_refused_but_it_says_what_it_is(tmp_path, monkeypatch):
    """Criterion: a system with no model is worse than one with a bad model.

    So ``ensure_active_model`` still produces a usable active_model.keras when
    none exists -- it is NOT refused on calibration grounds the way
    ``promote_candidate`` is -- but it must leave a warning and a registry row
    that both say the artifact is untrained and unguarded.

    Driven through the real method with the TensorFlow-shaped collaborators
    stubbed: what is under test is this method's control flow, not Keras.
    """
    import types

    db, db_path = _fresh_db(tmp_path)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    built = object()

    def _fake_build(**kwargs):
        assert kwargs["asset_vocab_size"] == 6
        return built, "headline_vec", "full_vec", {}, {}

    monkeypatch.setattr(
        pipeline_mod,
        "_get_model_defs",
        lambda: types.SimpleNamespace(build_multimodal_model=_fake_build),
    )
    saved: list = []

    def _fake_save(model, path, **_kwargs):
        Path(path).write_bytes(b"a usable artifact stands in for the zip keras would write")
        saved.append(Path(path))

    monkeypatch.setattr(pipeline_mod, "_save_model_atomically", _fake_save)
    logged: list = []
    monkeypatch.setattr(
        pipeline_mod,
        "log_message",
        lambda channel, message, **kw: logged.append((channel, message, kw)),
    )

    stub = types.SimpleNamespace(
        db=db,
        model_dir=model_dir,
        window_size=60,
        tech_count=8,
        sent_seq_len=16,
        data_loader=types.SimpleNamespace(asset_vocab_size=6),
        _last_asset_vocab_requirement=6,
        system_profile=types.SimpleNamespace(memory_pressure=False, is_low_power=False),
        model_templates=["small", "large"],
        load_active_model=lambda: None,
        _select_model_template=lambda idx: "large",
        _adapt_vectorizers=lambda *_a: None,
        _ensure_vectorizers_ready=lambda model: model,
    )
    stub._register_active_artifact_write = types.MethodType(
        TrainingPipeline._register_active_artifact_write, stub
    )

    returned = TrainingPipeline.ensure_active_model(stub)

    assert returned is built, "the bootstrap must return a model, never refuse"
    artifact = model_dir / "active_model.keras"
    assert saved == [artifact] and artifact.exists(), (
        "the bootstrap still produces a usable active_model.keras when none exists"
    )

    rows = _rows(db_path)
    assert len(rows) == 1, f"one bootstrap write, one row, got {len(rows)}"
    assert rows[0]["metrics"]["provenance"] == "bootstrap_build"
    assert rows[0]["metrics"]["trained"] is not True
    assert rows[0]["metrics"]["calibration_guarded"] is False
    assert rows[0]["metrics"]["asset_vocab_size"] == 6

    warnings = [m for m in logged if m[2].get("severity") == "warning"]
    assert warnings, "an untrained artifact reaching the served path is a warning"
    text = " ".join(m[1] for m in warnings).lower()
    assert "untrained" in text and "calibration" in text, text
    details = [m[2].get("details", {}) for m in warnings]
    assert any(d.get("trained") is False and d.get("calibration_guarded") is False for d in details)


def test_a_registry_failure_never_costs_the_caller_its_model(tmp_path):
    """Bookkeeping must not be able to throw out of a writer."""

    class _ExplodingDB:
        def register_model_version(self, **_kwargs):
            raise sqlite3.OperationalError("database is locked")

    stub = _StubPipeline(_ExplodingDB())
    assert stub.register(tmp_path / "active_model.keras", "bootstrap_build", None) is None


def test_every_writer_of_the_active_artifact_registers_it(tmp_path):
    """No FOURTH unregistered writer.

    Source-level on purpose: the failure this prevents is someone adding a
    write of the served path without a registry call, and that is a property of
    the file, not of one run.
    """
    source = (PROJECT_ROOT / "trading" / "pipeline.py").read_text(encoding="utf-8")
    writers = {
        "ensure_active_model": "bootstrap_build",
        "_ensure_asset_embedding_capacity": "asset_vocab_expansion",
        "promote_candidate": "promotion",
    }
    for func, provenance in writers.items():
        marker = f"    def {func}("
        start = source.index(marker)
        nxt = source.find("\n    def ", start + 1)
        body = source[start : nxt if nxt != -1 else len(source)]
        assert "active_model.keras" in body, f"{func} no longer writes the served path"
        assert provenance in body, f"{func} writes the served artifact without registering {provenance}"
        assert ("_register_active_artifact_write" in body) or ("register_model_version" in body), (
            f"{func} writes models/active_model.keras and registers nothing"
        )


# ---------------------------------------------------------------------------
# Is this artifact trained -- from the WEIGHTS alone.
# ---------------------------------------------------------------------------


def test_the_probe_answers_unknown_rather_than_trained_when_it_cannot_read(tmp_path):
    missing = tmp_path / "nope.keras"
    assert probe_artifact_training(missing)["trained"] is None
    assert is_artifact_trained(missing) is False, "unknown is never True"

    garbage = tmp_path / "active_model.keras"
    garbage.write_bytes(b"PK\x03\x04 truncated")
    assert is_artifact_trained(garbage) is False


def _synthetic_artifact(tmp_path: Path, name: str, *, pristine: bool) -> Path:
    """A real .keras archive -- zip + model.weights.h5 -- without TensorFlow.

    Keras writes every 1-D parameter to an EXACT constant at build time (bias
    0.0, LayerNormalization gamma 1.0) and one optimiser step moves it off. That
    is the whole signal the probe reads, so it can be reproduced honestly with
    h5py alone, which matters because this box's TensorFlow does not always
    load.
    """
    h5py = pytest.importorskip("h5py", reason="the probe reads model.weights.h5")
    import numpy as np
    import zipfile

    weights = tmp_path / f"{name}.weights.h5"
    with h5py.File(weights, "w") as handle:
        dense = handle.create_group("layers/dense/vars")
        # 2-D kernels are ignored by the probe; include one so the archive is
        # shaped like a real model rather than a bag of vectors.
        dense.create_dataset("0", data=np.full((3, 4), 0.3, dtype="float32"))
        dense.create_dataset(
            "1",
            data=(
                np.zeros(4, dtype="float32")
                if pristine
                else np.array([0.011, -0.024, 0.006, -0.0013], dtype="float32")
            ),
        )
        norm = handle.create_group("layers/layer_normalization/vars")
        norm.create_dataset(
            "0",
            data=(
                np.ones(4, dtype="float32")
                if pristine
                else np.array([1.004, 0.997, 1.012, 0.989], dtype="float32")
            ),
        )
        norm.create_dataset(
            "1",
            data=(
                np.zeros(4, dtype="float32")
                if pristine
                else np.array([-0.002, 0.005, 0.001, -0.004], dtype="float32")
            ),
        )
    archive = tmp_path / f"{name}.keras"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(weights, "model.weights.h5")
    return archive


def test_initializer_weights_read_as_untrained_and_moved_weights_read_as_trained(tmp_path):
    fresh = _synthetic_artifact(tmp_path, "fresh", pristine=True)
    stepped = _synthetic_artifact(tmp_path, "stepped", pristine=False)

    fresh_probe = probe_artifact_training(fresh)
    assert fresh_probe["vectors"] == 3, fresh_probe
    assert fresh_probe["pristine"] == 3, fresh_probe
    assert fresh_probe["trained"] is False, fresh_probe
    assert is_artifact_trained(fresh) is False

    stepped_probe = probe_artifact_training(stepped)
    assert stepped_probe["pristine"] == 0, stepped_probe
    assert stepped_probe["trained"] is True, stepped_probe
    assert is_artifact_trained(stepped) is True


@pytest.mark.skipif(
    os.getenv("MODEL_ARTIFACT_TF_TESTS", "0").lower() not in {"1", "true", "yes", "on"},
    reason=(
        "opt-in: importing keras on this box emits the oneDNN banner and then "
        "kills the interpreter with no traceback, which silently truncates the "
        "whole pytest session. Set MODEL_ARTIFACT_TF_TESTS=1 where TF loads. "
        "test_initializer_weights_read_as_untrained_and_moved_weights_read_as_trained "
        "proves the same property TF-free."
    ),
)
def test_a_freshly_built_model_is_untrained_and_one_fit_step_makes_it_trained(tmp_path):
    keras = pytest.importorskip("keras", reason="the weights probe is TF-free; BUILDING a model is not")
    pytest.importorskip("h5py", reason="the probe reads model.weights.h5")
    import numpy as np

    model = _tiny_model(keras)
    fresh = tmp_path / "fresh.keras"
    model.save(fresh)
    fresh_probe = probe_artifact_training(fresh)
    assert fresh_probe["vectors"] > 0, fresh_probe["reason"]
    assert fresh_probe["trained"] is False, fresh_probe
    assert is_artifact_trained(fresh) is False

    model.fit(np.ones((8, 3), dtype="float32"), np.ones((8, 1), dtype="float32"), epochs=1, verbose=0)
    fit_once = tmp_path / "fit_once.keras"
    model.save(fit_once)
    fit_probe = probe_artifact_training(fit_once)
    assert fit_probe["trained"] is True, fit_probe
    assert is_artifact_trained(fit_once) is True


# ---------------------------------------------------------------------------
# The compounding half: the placeholder must not slow down the loop that would
# replace it.
# ---------------------------------------------------------------------------


def test_an_untrained_placeholder_does_not_slow_background_refinement(tmp_path):
    cadence, fast = 900.0, 300.0
    path = tmp_path / "active_model.keras"

    assert refinement_cadence_for_artifact(path, cadence=cadence, fast_cadence=fast) == fast, (
        "no artifact at all is the case that most needs the fast loop"
    )

    # The old behaviour: ANY file on that path bought the slow cadence. This is
    # the regression -- a file that exists and was never fit.
    path.write_bytes(b"not a real archive")
    assert path.exists(), "the old check was os.path.exists, and it would pass here"
    assert refinement_cadence_for_artifact(path, cadence=cadence, fast_cadence=fast) == fast, (
        "an artifact that cannot be proven trained must not slow the loop that would train one"
    )


def test_a_trained_artifact_earns_the_slow_cadence(tmp_path, monkeypatch):
    cadence, fast = 900.0, 300.0
    path = tmp_path / "active_model.keras"
    path.write_bytes(b"stand-in for a trained artifact")

    monkeypatch.setattr(
        pipeline_mod,
        "probe_artifact_training",
        lambda _p: {"trained": True, "pristine_fraction": 0.0, "reason": "stub"},
    )
    pipeline_mod._CADENCE_PROBE_CACHE.clear()
    assert refinement_cadence_for_artifact(path, cadence=cadence, fast_cadence=fast) == cadence
    pipeline_mod._CADENCE_PROBE_CACHE.clear()


def test_the_bot_loop_asks_the_weights_not_the_filesystem():
    """bot.py's refinement loop read os.path.exists; that is the bug.

    Read with ``ast`` rather than by importing ``trading.bot``: that module
    pulls in TensorFlow, which takes minutes on this box and has been observed
    to kill the interpreter outright. The failure being pinned is a NAME one --
    the call site shipped calling ``refinement_cadence_for_artifact`` while no
    such name existed anywhere in the tree, a NameError the first time the loop
    ran -- and ast answers that exactly.
    """
    import ast

    source = (PROJECT_ROOT / "trading" / "bot.py").read_text(encoding="utf-8")
    start = source.index("async def _loop():")
    body = source[start : start + 4000]
    assert "refinement_cadence_for_artifact(" in body, (
        "the refinement loop must pick its cadence from the artifact's weights"
    )
    assert "os.path.exists" not in body.split("await asyncio.sleep")[0], (
        "the cadence decision must not be taken from the filesystem again"
    )

    tree = ast.parse(source)
    imported = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "refinement_cadence_for_artifact" in imported, (
        "bot.py calls refinement_cadence_for_artifact without importing it: "
        "NameError inside the background refinement loop"
    )
    assert callable(refinement_cadence_for_artifact), "and the name must exist to import"
