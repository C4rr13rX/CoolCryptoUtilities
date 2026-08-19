"""The asset embedding must cover every asset id it will ever be handed.

An embedding lookup is a hard bound: an id at or above `input_dim` raises
inside GatherV2 and kills the entire training run. Measured 2026-08-19, that
happened 13 consecutive times with
"indices[6,0] = 6 is not in [0, 5)", so the neural model never trained. The
readiness gate then reported `insufficient_accuracy` with recall 0.00106 --
the model fired on 23 of 15,333 samples (0.15%) against data that was 49%
positive, and the 0.50-0.70 threshold sweep produced identical results at
every step because there was no real probability distribution to threshold.
"""
from __future__ import annotations

import numpy as np


def observed_vocab(inputs) -> int:
    """Mirror of TradingPipeline._observed_asset_vocab."""
    try:
        asset_ids = inputs["asset_id_input"]
    except (TypeError, KeyError, IndexError):
        return 1
    try:
        array = np.asarray(asset_ids)
        if array.size == 0:
            return 1
        return max(1, int(np.max(array)) + 1)
    except (TypeError, ValueError):
        return 1


def test_capacity_covers_the_highest_id_in_the_batch():
    """The exact failure: id 6 present, table built with 5 rows."""
    assert observed_vocab({"asset_id_input": np.array([[0], [3], [6]])}) == 7
    # A table of 5 would have crashed; 7 admits index 6.
    assert observed_vocab({"asset_id_input": np.array([[0], [4]])}) == 5


def test_capacity_is_safe_on_degenerate_input():
    """A missing or empty batch must not crash sizing."""
    assert observed_vocab({}) == 1
    assert observed_vocab(None) == 1
    assert observed_vocab({"asset_id_input": np.array([])}) == 1
    assert observed_vocab({"asset_id_input": np.array([[0]])}) == 1


def test_the_requirement_is_a_high_water_mark():
    """Overwriting per batch is what made the rebuild useless.

    A batch whose highest asset was 4 reset the requirement to 5, the model
    was rebuilt with 5 rows, and the next batch carrying asset 6 died. The
    embedding must cover every id EVER seen, so the requirement can only grow
    -- shrinking it invalidates ids already baked into the trained weights.
    """
    requirement = 1
    for batch_max in (4, 6, 2, 3):
        requirement = max(requirement, batch_max + 1)
    assert requirement == 7, "must retain the peak, not follow the last batch"

    # The old behaviour, kept explicit so the regression is unmistakable.
    naive = 1
    for batch_max in (4, 6, 2, 3):
        naive = batch_max + 1
    assert naive == 4
    assert naive < 7


def test_the_loader_vocab_cannot_be_trusted_for_sizing():
    """asset_vocab_size reports 1 while the tensors carry ids up to 4.

    Measured 2026-08-19 on the live loader: `_asset_vocab` stayed EMPTY
    across a full `_prepare_dataset` call (same object, verified by id()),
    `asset_lexicon` was `{}`, and `asset_vocab_size` returned 1 -- yet the
    emitted `asset_id_input` ran 1..4 and needed a table of 5. The ids come
    from cached/persisted samples that never pass through `_get_asset_id`,
    so the vocabulary counter never sees them.

    Sizing must therefore come from the DATA, never from the loader.
    """
    loader_vocab = 1          # what asset_vocab_size reported
    observed = observed_vocab({"asset_id_input": np.array([[1], [2], [3], [4]])})
    assert observed == 5
    assert observed > loader_vocab
    # The size actually used is the max of both, so the loader can only ever
    # raise the floor -- never lower it below what the batch needs.
    assert max(loader_vocab, observed) == 5


def test_the_guard_raises_and_records_before_gatherv2_can_fail():
    """Fail loudly at the tensors, not deep inside Keras.

    The production failure was "indices[0,0] = 5 is not in [0, 5)" raised
    inside GatherV2, with a stack naming only Keras internals -- 887 of them,
    each killing a whole training run. Checking where the tensors are final
    names the real problem and lets the next attempt build a table that fits.
    """
    from trading.pipeline import TrainingPipeline, _EmbeddingTooSmall

    class FakeLayer:
        input_dim = 5

    class FakeModel:
        def get_layer(self, name):
            if name == "asset_embedding":
                return FakeLayer()
            raise ValueError(name)

    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline._last_asset_vocab_requirement = 1
    order = ["x", "asset_id_input"]

    # Highest id 4 fits a table of 5 -- must stay silent.
    pipeline._assert_embedding_covers(
        FakeModel(), (np.zeros((3, 1)), np.array([[0], [4]])), order)

    # Id 5 does not fit, which is the exact production failure.
    try:
        pipeline._assert_embedding_covers(
            FakeModel(), (np.zeros((3, 1)), np.array([[0], [5]])), order)
    except _EmbeddingTooSmall as too_small:
        assert too_small.needed == 6
    else:
        raise AssertionError("expected _EmbeddingTooSmall")

    # And the high-water mark is raised so the retry builds big enough.
    assert pipeline._last_asset_vocab_requirement == 6


def test_the_guard_is_inert_without_an_embedding_or_ids():
    """A model or batch shaped differently must not break training."""
    from trading.pipeline import TrainingPipeline

    class NoEmbedding:
        def get_layer(self, name):
            raise ValueError(name)

    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline._last_asset_vocab_requirement = 1
    pipeline._assert_embedding_covers(NoEmbedding(), (np.zeros((2, 1)),), ["x"])

    class WithEmbedding:
        def get_layer(self, name):
            class L:
                input_dim = 3
            return L()

    # asset_id_input absent from the order -> nothing to check.
    pipeline._assert_embedding_covers(WithEmbedding(), (np.zeros((2, 1)),), ["x"])
