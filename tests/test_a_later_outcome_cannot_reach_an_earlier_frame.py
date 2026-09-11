#!/usr/bin/env python3
"""The self-knowledge pools must never learn an outcome before it happened.

Pools 15/16/19 describe the brain's own track record. That is legitimate only
while the record is SETTLED and settled BEFORE the bar being decided. Two ways
to get it wrong, and this file makes both fail loudly:

  * an OPEN prediction reaching a frame -- the prediction_error feedback loop
    that took recall from 100% to 30% on this substrate;
  * a LATER outcome reaching an EARLIER frame -- lookahead, which manufactures
    an edge that evaporates live. This repo has already shipped a fake 78%
    directional and a fake +0.9067%, and both were found later and cost more
    than the truth would have.

Each test here fails against the obvious wrong implementation (filter on
``resolved`` alone, or hand ``self_frames`` the whole history), not merely
against a broken one.
"""

import pytest

from trading.omen_metacognition import self_frames
from trading.omen_resolved_history import ResolvedHistory, walk_forward


def test_an_unsettled_prediction_never_enters_the_history():
    """A prediction with no outcome yet is invisible to every reader."""
    history = ResolvedHistory(horizon_bars=12)
    history.record(100, "trough")
    history.record(101, "crest")

    # Nothing has landed: bar 105 is before either horizon resolves.
    assert history.as_of(105) == ()
    assert history.pending(105) == 2

    # Bar 112 is where the first one lands -- but it has not been settled, so
    # it is DUE and still not a fact. Being due is not the same as being known.
    assert history.as_of(112) == ()

    history.settle(112, "murk")
    seen = history.as_of(112)
    assert len(seen) == 1
    assert seen[0].predicted == "trough"
    assert seen[0].actual == "murk"
    assert seen[0].resolved is True
    # The second prediction resolves at 113 and must still be invisible here.
    assert all(r.predicted != "crest" for r in seen)


def test_a_later_outcome_cannot_leak_backwards_into_an_earlier_frame():
    """Settling bar 200 must not change what bar 150's frame could see.

    This is the test that fails if ``as_of`` filters on the ``resolved`` flag
    instead of on ``resolve_index``: settling a late row would retroactively
    appear in every earlier read.
    """
    history = ResolvedHistory(horizon_bars=10)
    for bar in (100, 140, 190):
        history.record(bar, "trough")

    history.settle(110, "trough")
    before = history.as_of(150)
    assert len(before) == 1, "only the bar-100 prediction has landed by 150"

    # The future happens. Bar 150's view of the past must be unmoved.
    history.settle(150, "slide")
    history.settle(200, "crest")
    after = history.as_of(150)

    assert len(after) == 2, "the row resolving AT 150 is a fact at 150"
    assert all(r.predicted == "trough" for r in after)
    assert history.as_of(149) == before, (
        "settling bar 200 changed what bar 149 was allowed to see -- that is "
        "lookahead, and it is the whole reason this class exists")
    # And the frames built from those two views must differ in nothing but
    # what genuinely landed between them.
    assert self_frames(history.as_of(149)) == self_frames(before)


def test_predictions_must_arrive_in_bar_order():
    """A backwards step means the caller already read a bar it has not reached."""
    history = ResolvedHistory(horizon_bars=6)
    history.record(500, "climb")
    with pytest.raises(ValueError, match="oldest-first"):
        history.record(499, "slide")


def test_an_outcome_cannot_be_overwritten():
    history = ResolvedHistory(horizon_bars=4)
    history.record(10, "trough")
    assert history.settle(14, "trough") == 1
    with pytest.raises(ValueError, match="already settled"):
        history.settle(14, "crest")


def test_the_walk_forward_history_grows_and_never_holds_the_open_call():
    """Every bar's history is a prefix of the next, and never the current bar."""
    bars = list(range(0, 40))
    predictions = ["trough" if b % 2 else "crest" for b in bars]
    actuals = {b: ("trough" if b % 3 else "murk") for b in bars}

    rows = walk_forward(bars, predictions, actuals, horizon_bars=5)
    assert len(rows) == len(bars)

    seen_counts = [len(hist) for _, hist in rows]
    assert seen_counts[:5] == [0, 0, 0, 0, 0], (
        "no prediction can have resolved before one horizon has passed")
    assert seen_counts == sorted(seen_counts), "history must never shrink"

    for position, (bar, hist) in enumerate(rows):
        # The prediction MADE at this bar must not be in this bar's history.
        assert len(hist) <= max(0, position - 5 + 1)


def test_self_frames_move_off_their_na_sentinels_once_the_feeder_runs():
    """The defect this feeder exists to fix, stated as a test.

    With no history the three pools emit ``na`` and carry nothing. With a
    settled history they must say something, and a WRONG-THE-SAME-WAY run must
    be visible -- that is the single fact the brain could not previously know
    about itself.
    """
    empty = self_frames(())
    assert empty["self_outcome"] == "slf hit=na n=0 last=na"
    assert empty["self_error_run"] == "err run=na dir=na"

    # Eight consecutive trough calls into bars that all turned out slide.
    bars = list(range(0, 30))
    rows = walk_forward(bars, ["trough"] * len(bars),
                        {b: "slide" for b in bars}, horizon_bars=3,
                        agreements=[(3, 3)] * len(bars))
    _, history = rows[-1]
    frames = self_frames(history)

    assert frames["self_outcome"] != empty["self_outcome"]
    assert frames["self_error_run"] != empty["self_error_run"]
    assert "dir=trough" in frames["self_error_run"], (
        "a model wrong the same way for many bars must be able to say WHICH "
        "way -- that is the pass-109 defect: trough called 50 times into a "
        "window where 14.2% of bars rose")
    assert "run=m" in frames["self_error_run"]
    assert frames["self_agreement"] != empty["self_agreement"]


def test_the_sample_builder_actually_feeds_the_self_pools():
    """The seam that made a real node report QUERY PATH DEAD.

    Measured pass 110 on a 19-pool node: a query set differing only by pools
    15/16/19 moved 0 of 60 held-out predictions while the arm fired six streams
    per prediction. The pools were sent and read; they moved nothing because
    every sample-building loop called ``build_collections`` without
    ``history=``, so every self frame in the training set was the ``na``
    sentinel and the pools trained as CONSTANTS.

    This test fails against that loop: without the history, all three self
    frames are one value across every sample.
    """
    import importlib
    import json
    import os
    from pathlib import Path

    import trading.omen_brain as omen_brain
    from trading.omen_resolved_history import build_samples_with_history

    # The self_* collections are OFF by default and the gate is read at
    # IMPORT time (omen_brain.META_ENABLED), so without this reload
    # build_collections never emits the keys this test asserts on and it dies
    # with KeyError 'self_outcome' -- a test that cannot see the pools it is
    # named after. `build_samples_with_history` imports build_collections
    # inside its body, so reloading the module is enough to rebind it.
    os.environ["OMEN_META_COLLECTIONS"] = "1"
    try:
        importlib.reload(omen_brain)
        assert omen_brain.META_ENABLED, "the meta gate did not take"

        corpus = Path("data/historical_ohlcv/base/0004_AERO-USDC.json")
        if not corpus.exists():
            import pytest as _pytest
            _pytest.skip("corpus not present")

        bars = [b for b in json.loads(corpus.read_text(encoding="utf-8"))
                if b.get("close")]
        bars.sort(key=lambda b: int(b["timestamp"]))
        stop = len(bars) - 13
        samples = build_samples_with_history(bars, "AERO-USDC", "base", 12,
                                             stop - 400, stop)
        assert len(samples) > 200

        for key in ("self_outcome", "self_agreement", "self_error_run"):
            values = {s["frames"][key] for s in samples}
            assert len(values) > 1, (
                f"{key} is ONE value across {len(samples)} samples -- the history "
                f"is not reaching build_collections, and a constant stream cannot "
                f"move a query however good the pool is")
        # self_outcome and self_error_run are computable from the outcomes alone,
        # so they must be saying something concrete. self_agreement is NOT: it
        # needs how many query sets voted together, which a single-rule driver
        # does not have, so its rate field stays `na` until the node's own
        # predictions drive the walk. That is a real limit of this builder and it
        # is asserted rather than glossed.
        for key in ("self_outcome", "self_error_run"):
            values = {s["frames"][key] for s in samples}
            assert not all(v.endswith("na") for v in values), (
                f"{key} is computable from settled outcomes alone and must carry "
                f"a concrete value, not a sentinel")
        rates = {s["frames"]["self_agreement"] for s in samples}
        assert all("rate=na" in v for v in rates), (
            "self_agreement's rate needs multi-query-set votes; a single-rule "
            "driver cannot supply them, and pretending otherwise would fake the "
            "one signal measured to beat confidence")
    finally:
        os.environ.pop("OMEN_META_COLLECTIONS", None)
        importlib.reload(omen_brain)
