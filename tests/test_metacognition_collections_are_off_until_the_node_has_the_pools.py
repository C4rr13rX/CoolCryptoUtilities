"""The metacognition streams must not reach a node that has no pools for them.

Named after the failure it prevents, which is the same one pools 12/13/14 have
and is worse here because there are two flags. Pools 15-19 exist only in
``brains/market_predictor_v4_meta.identity.toml``. A v2 node declares 11 and a
v3_assoc node 14, and neither degrades gracefully: the node answers
``unknown input pool id 15`` with ``consolidated: False``, and ``OmenBrain``
counts the whole sample as a MISS. Switching the metacognition pools on
against the wrong node therefore does not weaken training, it SILENTLY STOPS
it -- the run looks healthy and teaches nothing. That is why the default is
off and why the default is worth a test.

THE SECOND THING THIS FILE GUARDS is the invariant that made the relation
wiring safe, now that TWO independent flags can move it:

    set(build_collections(...)) == {c.name for c in COLLECTIONS}

It is checked in all FOUR combinations of the relation and metacognition
flags. If the frame keys and the COLLECTIONS entries can disagree, either a
pool receives nothing or a frame is computed for a pool that will never
receive it -- and both failures are invisible at the call site.

THE THIRD is the feedback guard. Pools 15/16/19 describe the brain's own past,
which is legitimate only while that past is SETTLED. An unresolved row
describes an outcome that has not happened yet, and feeding a live prediction
back as its own input is the prediction_error loop that took recall from 100%
to 30% on this substrate. ``build_collections`` must never let one through.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import trading.omen_brain as omen_brain  # noqa: E402
from trading.omen_metacognition import Resolved  # noqa: E402

META_NAMES = ("self_outcome", "self_agreement", "temporal_sequence",
              "temporal_scale", "self_error_run")
META_POOLS = (15, 16, 17, 18, 19)


def _reload(monkeypatch, meta=None, relations=None):
    for name, value in (("OMEN_META_COLLECTIONS", meta),
                        ("OMEN_RELATION_COLLECTIONS", relations)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    return importlib.reload(omen_brain)


@pytest.fixture(autouse=True)
def _restore():
    yield
    os.environ.pop("OMEN_META_COLLECTIONS", None)
    os.environ.pop("OMEN_RELATION_COLLECTIONS", None)
    importlib.reload(omen_brain)


def _bars(n=300, *, vol=0.004, seed=11):
    """Deterministic OHLCV -- a fixed recurrence, so no assertion can flake."""
    bars = []
    price = 100.0
    state = seed
    for i in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        wobble = ((state / 2147483648.0) - 0.5) * 2.0
        price *= 1.0 + vol * wobble
        bars.append({
            "timestamp": 1700000000 + 3600 * i,
            "open": price * (1.0 - vol * wobble * 0.5),
            "high": price * (1.0 + abs(wobble) * vol),
            "low": price * (1.0 - abs(wobble) * vol),
            "close": price,
            "volume": 1000.0 + (state % 500),
        })
    return bars


def test_the_metacognition_pools_are_off_by_default():
    """Off is the safe default because ON against a v2 or v3 node stops
    training outright rather than degrading it."""
    mod = importlib.reload(omen_brain)
    assert mod.META_ENABLED is False
    names = {c.name for c in mod.COLLECTIONS}
    for name in META_NAMES:
        assert name not in names
    frames = mod.build_collections(_bars(), 250, horizon_bars=6, symbol="TEST")
    for name in META_NAMES:
        assert name not in frames, (
            f"{name} was computed for a pool this build will never stream")


def test_enabling_metacognition_streams_pools_fifteen_to_nineteen(monkeypatch):
    mod = _reload(monkeypatch, meta="1")
    assert mod.META_ENABLED is True
    by_name = {c.name: c for c in mod.COLLECTIONS}
    for name, pool in zip(META_NAMES, META_POOLS):
        assert name in by_name, f"{name} missing from COLLECTIONS"
        assert by_name[name].pool_id == pool

    frames = mod.build_collections(_bars(), 250, horizon_bars=6, symbol="TEST")
    brain = mod.OmenBrain.__new__(mod.OmenBrain)
    streamed = {s["pool_id"] for s in brain._streams(frames)}
    for pool in META_POOLS:
        assert pool in streamed, f"pool {pool} is declared but never streamed"


@pytest.mark.parametrize("meta", [None, "0", "1"])
@pytest.mark.parametrize("relations", [None, "0", "1"])
def test_the_frame_keys_and_the_collections_agree_in_every_combination(
        monkeypatch, meta, relations):
    """Two independent flags, one invariant. If they can disagree, a pool gets
    nothing or work is done for a pool that will never receive it."""
    mod = _reload(monkeypatch, meta=meta, relations=relations)
    frames = mod.build_collections(_bars(), 250, horizon_bars=6, symbol="TEST")
    assert set(frames) == {c.name for c in mod.COLLECTIONS}


def test_every_metacognition_prefix_is_byte_disjoint_from_the_others(monkeypatch):
    """Atoms are bytes. A prefix that is a prefix of another lets one
    collection's frame match another's during a query."""
    mod = _reload(monkeypatch, meta="1", relations="1")
    prefixes = [c.prefix for c in mod.COLLECTIONS]
    assert len(set(prefixes)) == len(prefixes), "prefixes must be unique"
    for a in prefixes:
        for b in prefixes:
            if a is not b and a != b:
                assert not a.startswith(b), f"{a!r} starts with {b!r}"
    frames = mod.build_collections(_bars(), 250, horizon_bars=6, symbol="TEST")
    for collection in mod.COLLECTIONS:
        assert frames[collection.name].startswith(collection.prefix + " ")


def test_an_unresolved_prediction_cannot_reach_a_frame(monkeypatch):
    """THE FEEDBACK GUARD. A live prediction fed back as its own input is the
    prediction_error loop that took recall from 100% to 30% here. An
    unresolved row must change nothing about the frames."""
    mod = _reload(monkeypatch, meta="1")
    bars = _bars()

    settled = [Resolved(predicted="rise", actual="rise", agreed=3, asked=3,
                        resolved=True) for _ in range(8)]
    open_row = Resolved(predicted="fall", actual=None, agreed=1, asked=3,
                        resolved=False)

    without = mod.build_collections(bars, 250, horizon_bars=6, symbol="TEST",
                                    history=settled)
    with_open = mod.build_collections(bars, 250, horizon_bars=6, symbol="TEST",
                                      history=settled + [open_row])
    for name in ("self_outcome", "self_agreement", "self_error_run"):
        assert without[name] == with_open[name], (
            f"{name} changed when an UNRESOLVED prediction was appended -- "
            "the open prediction is reaching its own input")


def test_a_settled_history_actually_moves_the_self_frames(monkeypatch):
    """The guard above is only meaningful if a RESOLVED row does get through.
    A test that passes both ways proves nothing."""
    mod = _reload(monkeypatch, meta="1")
    bars = _bars()

    empty = mod.build_collections(bars, 250, horizon_bars=6, symbol="TEST")
    hits = [Resolved(predicted="rise", actual="rise", agreed=3, asked=3,
                     resolved=True) for _ in range(8)]
    misses = [Resolved(predicted="rise", actual="fall", agreed=1, asked=3,
                       resolved=True) for _ in range(8)]

    winning = mod.build_collections(bars, 250, horizon_bars=6, symbol="TEST",
                                    history=hits)
    losing = mod.build_collections(bars, 250, horizon_bars=6, symbol="TEST",
                                   history=misses)

    assert empty["self_outcome"] != winning["self_outcome"], (
        "a settled history did not reach the frame")
    assert winning["self_outcome"] != losing["self_outcome"], (
        "being right and being wrong produce the same frame, so pool 15 "
        "carries no information")
    assert winning["self_error_run"] != losing["self_error_run"], (
        "an error streak is indistinguishable from a hit streak")


def test_the_temporal_frames_read_order_not_just_magnitude(monkeypatch):
    """The whole point of pools 17/18: every other collection is a view of one
    instant. Two series with the SAME moves in a DIFFERENT ORDER must produce
    different sequence frames, or nothing in the topology carries order."""
    mod = _reload(monkeypatch, meta="1")
    index = 250

    def _series(closes):
        bars = _bars()
        for bar, close in zip(bars, closes):
            bar.update(open=close, high=close * 1.001, low=close * 0.999,
                       close=close)
        return bars

    # A run of big ups then big downs, ENDING AT THE DECISION BAR -- the frame
    # reads bars[:index + 1] only, so anything written past it is invisible.
    # Same multiset of steps in both series; only the order differs.
    steps = [1.03, 1.03, 1.03, 0.97, 0.97, 0.97]
    forward, backward = [], []
    price_f = price_b = 100.0
    for step_f, step_b in zip(steps, list(reversed(steps))):
        price_f *= step_f
        price_b *= step_b
        forward.append(price_f)
        backward.append(price_b)
    flat = [100.0] * (index + 1 - len(steps))

    up = mod.build_collections(_series(flat + forward), index,
                               horizon_bars=6, symbol="TEST")
    down = mod.build_collections(_series(flat + backward), index,
                                 horizon_bars=6, symbol="TEST")
    assert up["temporal_sequence"] != down["temporal_sequence"], (
        "reversing the order of the recent moves did not change the sequence "
        "frame -- pool 17 is not carrying order")


def test_the_scale_frame_is_sharp_enough_to_survive_the_dilution_filter(monkeypatch):
    """Named after the failure it prevents, measured in pass 109.

    A direction-token-only scale frame ("s3=u s12=u s48=d agree=split") scored
    0.045 distinct frames per sample on 600 AERO-USDC samples, against the
    dilution law's 0.20 bar -- so ``discriminating_collections`` excluded pool
    18 from the query set and the ONE pool aimed at multi-scale regime never
    fired. Adding a coarse magnitude bucket per scale took it to 0.303.

    Both ends are asserted, because this frame can fail in two opposite
    directions and only one of them is obvious. Too COARSE and the dilution
    law drops it. Too SHARP and it becomes an identifier: SEQUENCE_STEPS was
    cut from 8 to 5 for scoring 0.76 distinct per sample, which maximises
    train recall and is exactly what cannot generalise.
    """
    mod = _reload(monkeypatch, meta="1")
    bars = _bars(700, vol=0.006, seed=29)
    frames = [mod.build_collections(bars, i, horizon_bars=6, symbol="TEST")
              for i in range(260, 700)]

    scale = {f["temporal_scale"] for f in frames}
    ratio = len(scale) / len(frames)
    assert ratio >= mod.MIN_QUERY_DISTINCTNESS, (
        f"temporal_scale is {ratio:.3f} distinct per sample, below the "
        f"{mod.MIN_QUERY_DISTINCTNESS} query bar -- pool 18 will be excluded "
        "from every query and the regime pool will never fire")
    assert ratio <= 0.70, (
        f"temporal_scale is {ratio:.3f} distinct per sample -- approaching an "
        "identifier, which maximises recall and destroys generalisation")


# A "nudge vs dislocation" test was written here and REMOVED, deliberately.
# It asserted that a 0.05%/bar drift and a 5%/bar rise produce different scale
# frames. They do not, and that is correct: every scale is measured in units of
# the move's OWN noise, because a 2% move means nothing without knowing whether
# 2% is a normal hour for this symbol. The frame is scale-invariant by design.
# The evidence that the magnitude bucket carries information is the distinctness
# measurement above -- 0.045 to 0.303 on 600 real AERO-USDC samples -- not a
# constructed pair the normalisation is built to collapse.
