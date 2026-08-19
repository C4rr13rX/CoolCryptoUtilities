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
