"""A query must fire only the streams that can tell its samples apart.

THE DILUTION LAW, measured 2026-09-07 against a fabric trained on 2725
AERO-USDC pairs (read-only probes, identical samples every row, n=200):

    query streams                       distinct frames / 2725   recall
    temporal + geometry + cross         2725 / 2627 / 709        96.0%
    ... + flow                          282                      92.5%
    ... + volatility                    166                      91.0%
    ... + horizon + instrument          1 / 1                    91.0%
    all seven                                                    91.5%
    ... + regime frame (TRUE token)     4                        93.5%
    ... + regime frame (stage-1 guess)  4                        90.5%

A stream whose frame is shared by many training samples votes for the label
*distribution* over all of them; a stream unique to one sample votes for one
label. Fire enough of the former and they out-vote the latter. The ordering
is monotone in distinctness, and the ranking held on a full experiment run
over the same fabric: 93.6% train recall on the old path, 96.4% on this one.

The regime stream was the worst of all, for two compounding reasons: it has
the lowest distinctness in the design (4 values over 2725 samples), and
``label_regime`` is a DETERMINISTIC causal function of the bars that stage 1
was reproducing at only 73.3% -- with 0.98 confidence when it was wrong. Any
caller that built the frames holds the bars they were built from, so the
regime is arithmetic there and must never be re-guessed.

Every test here fails against the pre-2026-09-07 behaviour, and none needs a
running node.
"""
from __future__ import annotations

import pytest

import trading.omen_brain as ob
from trading.omen_brain import (
    COLLECTIONS_BY_NAME, OMEN_POOL, PREDICT_COLLECTIONS, REGIME_POOL,
    REGIME_TOKENS, OmenBrain, collection_distinctness,
    discriminating_collections, omen_frame,
)


ALL_NAMES = ("geometry", "temporal", "flow", "volatility", "cross",
             "horizon", "instrument")


def _frames(seed: int = 0) -> dict:
    """One well-formed frame per collection, prefixed as the real ones are."""
    return {name: f"{COLLECTIONS_BY_NAME[name].prefix} k={seed}"
            for name in ALL_NAMES}


class _Recorder(OmenBrain):
    """An OmenBrain that answers a fixed omen and records what it was asked.

    Overriding ``_predict`` rather than the transport keeps the assertions on
    the stream set the caller actually built, which is the thing under test.
    """

    def __init__(self, answer: str = "murk") -> None:
        super().__init__(endpoint="http://127.0.0.1:1")
        self.calls: list = []
        self._answer = answer
        self._multi_supported = True  # never touch the network

    def _predict(self, streams, target_pool):
        self.calls.append({"pool": target_pool,
                           "pools": [s["pool_id"] for s in streams]})
        if target_pool == REGIME_POOL:
            return "regime bullrun", 0.9
        return omen_frame(self._answer), 0.9

    def _stage2(self):
        stage2 = [c for c in self.calls if c["pool"] == OMEN_POOL]
        assert stage2, "no stage-2 query was made"
        return stage2[-1]


def _predict(brain: _Recorder, **kwargs):
    return brain.predict(
        _frames(), symbol="AERO-USDC", chain="base", as_of_ts=1_700_000_000,
        price=1.0, horizon_bars=12, bar_seconds=3600, **kwargs)


# --- the query set --------------------------------------------------------

def test_a_query_does_not_fire_the_low_entropy_collections():
    """91.5% -> 96.0%. Firing all seven let flow/volatility out-vote temporal."""
    brain = _Recorder()
    _predict(brain, regime="bullrun")
    fired = set(brain._stage2()["pools"])
    for name in ("flow", "volatility", "horizon", "instrument"):
        assert COLLECTIONS_BY_NAME[name].pool_id not in fired, (
            f"{name} was fired in the query; it is a diluting stream "
            f"(measured distinctness 0.10, 0.06, 0.00, 0.00 of 1.0)")
    for name in ("temporal", "geometry", "cross"):
        assert COLLECTIONS_BY_NAME[name].pool_id in fired, (
            f"{name} discriminates and must be fired")


def test_the_query_set_is_exactly_the_declared_one():
    brain = _Recorder()
    _predict(brain, regime="bullrun")
    assert sorted(brain._stage2()["pools"]) == sorted(
        COLLECTIONS_BY_NAME[n].pool_id for n in PREDICT_COLLECTIONS)


def test_the_regime_frame_is_not_fired_into_the_query_by_default():
    """The lowest-distinctness stream in the design: 4 values over 2725."""
    brain = _Recorder()
    _predict(brain, regime="bullrun")
    assert REGIME_POOL not in brain._stage2()["pools"], (
        "the regime frame cost 2.5 points of recall even when TRUE and 5.5 "
        "as stage 1 actually guessed it")


def test_an_unknown_query_override_does_not_silence_the_brain():
    """A typo in an env override must degrade to all streams, not to none."""
    brain = _Recorder()
    omen = _predict(brain, regime="bullrun",
                    query_collections=["not_a_collection"])
    assert brain._stage2()["pools"], "an empty query cannot answer at all"
    assert omen.verdict == "admitted"


def test_a_caller_can_still_ask_for_every_collection():
    brain = _Recorder()
    _predict(brain, regime="bullrun", query_collections=list(ALL_NAMES))
    assert sorted(brain._stage2()["pools"]) == sorted(
        COLLECTIONS_BY_NAME[n].pool_id for n in ALL_NAMES)


# --- the regime is arithmetic, not a forecast -----------------------------

def test_a_supplied_regime_is_never_re_guessed_by_stage_one():
    """Stage 1 reproduced a deterministic function of the bars at 73.3%."""
    brain = _Recorder()
    omen = _predict(brain, regime="bearrun")
    assert not any(c["pool"] == REGIME_POOL for c in brain.calls), (
        "stage 1 was probed for a regime the caller had already computed")
    assert omen.regime == "bearrun"
    assert omen.regime_confidence == 1.0, "arithmetic is not a 0.98 guess"
    assert omen.support["regime_source"] == "computed"


def test_a_missing_regime_still_falls_back_to_stage_one():
    """A caller with frames but no bars must still get a regime reported."""
    brain = _Recorder()
    omen = _predict(brain)
    assert any(c["pool"] == REGIME_POOL for c in brain.calls)
    assert omen.regime == "bullrun"
    assert omen.support["regime_source"] == "stage1"


def test_a_nonsense_regime_falls_back_rather_than_being_believed():
    brain = _Recorder()
    omen = _predict(brain, regime="moon")
    assert omen.regime in REGIME_TOKENS
    assert omen.support["regime_source"] == "stage1"


def test_the_query_set_is_reported_on_every_omen():
    """A recall number read without knowing what fired is not comparable."""
    brain = _Recorder()
    omen = _predict(brain, regime="chop")
    assert omen.support["query_collections"] == list(PREDICT_COLLECTIONS)


# --- the law itself, as a pure function -----------------------------------

def test_distinctness_is_a_ratio_of_samples_not_a_count():
    """Units. A count would make the threshold corpus-size dependent."""
    sets = [dict(_frames(i), volatility="vol same") for i in range(10)]
    scores = collection_distinctness(sets)
    assert scores["temporal"] == pytest.approx(1.0)
    assert scores["volatility"] == pytest.approx(0.1)
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_a_constant_stream_is_never_chosen_for_the_query():
    sets = [dict(_frames(i), horizon="hzn h=12", instrument="ins aero base")
            for i in range(50)]
    picked = discriminating_collections(sets)
    assert "horizon" not in picked and "instrument" not in picked
    assert "temporal" in picked


def test_a_stream_below_the_threshold_is_dropped_and_above_it_is_kept():
    """cross measured 0.26 and helped; flow measured 0.10 and hurt."""
    sets = []
    for i in range(100):
        frames = _frames(i)
        frames["cross"] = f"crs k={i // 4}"    # 0.25 distinct
        frames["flow"] = f"flw k={i // 10}"    # 0.10 distinct
        sets.append(frames)
    picked = discriminating_collections(sets, minimum=0.20)
    assert "cross" in picked
    assert "flow" not in picked


def test_the_query_never_collapses_to_nothing():
    """Every stream constant is a broken corpus, not a reason to go silent."""
    sets = [_frames(0) for _ in range(20)]
    picked = discriminating_collections(sets, minimum=0.99)
    assert len(picked) >= 1


def test_the_default_query_matches_what_the_law_would_choose():
    """The shipped default must be DERIVED, not a hand-picked list.

    Reproduces the measured AERO-USDC distinctness profile and asserts the
    threshold picks exactly the collections the default names.
    """
    measured = {"temporal": 1.000, "geometry": 0.964, "cross": 0.260,
                "flow": 0.103, "volatility": 0.061,
                "horizon": 0.0, "instrument": 0.0}
    sets = []
    for i in range(1000):
        frames = {}
        for name, ratio in measured.items():
            span = max(1, round(ratio * 1000))
            frames[name] = f"{COLLECTIONS_BY_NAME[name].prefix} k={i % span}"
        sets.append(frames)
    assert set(discriminating_collections(sets)) == set(PREDICT_COLLECTIONS)


def test_the_dilution_threshold_sits_in_the_measured_empty_band():
    """Pinned so it cannot be moved without re-measuring: 0.103 < t <= 0.260."""
    assert 0.103 < ob.MIN_QUERY_DISTINCTNESS <= 0.260
