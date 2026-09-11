"""An L2 arm that scores the REJECTED scheme measures the loser and says L2.

Item [a4ba2028]. ``trading/omen_layers`` ships ``transition_motif`` as L2 --
0.2633 DOWN / 0.2000 UP over a sticky L1, inside the 0.30 identifier ceiling --
and REJECTS the fixed-length ``sequence_motif``, which reads 0.5317 / 0.4083 at
the identical banding. ``scripts/omen_layer_probe`` is the instrument that does
``label_skew`` and ``heldout_edge``, and until this item it computed ONLY
``L2_sequence``: there was no L2_transitions column at all. So its exit code
judged the rejected scheme, and any held-out L2 arm run through it would have
scored the loser under the winner's name.

Three separate defects, each pinned below, and every one of them is node-free:

  1. THE COLUMN. The probe must compute the shipped scheme, from
     trading.omen_layers rather than a local copy, over the window the shipped
     churn cuts were fitted at -- and keep the rejected one beside it, because
     a winner with no control has no margin.
  2. THE BANDING. L2's frames are a function of the L1 alphabet underneath
     them, so a default of margin 0 judges an encoder nobody ships. That exact
     mistake blocked [fa75fa1a] in pass 111.
  3. THE REFUSAL. A layer with no train group at the support floor has nothing
     to fit a motif->trough map on. Scoring it anyway produces an abstention
     that reads as a negative result about the market, when it is a fact about
     the support census.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading.omen_brain import LOOKBACK_BARS  # noqa: E402
from trading.omen_layers import (  # noqa: E402
    IDENTIFIER_CEILING, L1_HYSTERESIS_MARGIN, L2_TRANSITION_STEPS,
    MOTIF_SEQUENCE_STEPS, layer_distinctness, sequence_motif,
    transition_motif,
)
import scripts.omen_layer_probe as probe  # noqa: E402
import scripts.omen_l2_scheme_probe as scheme_probe  # noqa: E402

CORPUS_DIR = ROOT / "data" / "historical_ohlcv" / "base"

#: Enough bars past the lookback to give the 12-bar L2 window real history,
#: ``relative_bands`` real terciles, and the held-out train window enough
#: samples for an L1 group to clear the support floor -- at 120 train samples
#: none does, and every held-out assertion below would be testing the refusal
#: branch instead of the branch it names.
_SAMPLES = 600
_TRAIN = 400
_TEST = 60


@functools.lru_cache(maxsize=1)
def _one_corpus():
    """The first readable base corpus with enough bars, or skip.

    Deliberately a REAL corpus rather than a fixture. The claim under test is
    about a column's values on market data -- a synthetic corpus can pin the
    plumbing (and the refusal test below does exactly that) but cannot show
    that the two schemes actually disagree on the feed we trade.
    """
    if not CORPUS_DIR.is_dir():
        pytest.skip("no base corpus checked out")
    for path in sorted(CORPUS_DIR.glob("*.json")):
        try:
            bars = probe.load_bars(path)
        except Exception:
            continue
        if len(bars) >= LOOKBACK_BARS + _SAMPLES + 20:
            return path.stem.split("_", 1)[-1], bars
    pytest.skip("no base corpus with %d bars" % (LOOKBACK_BARS + _SAMPLES))


@functools.lru_cache(maxsize=1)
def _rows():
    """Built once. Every test here reads the same frames, which is also the
    only way two of them could disagree about what the column contains."""
    symbol, bars = _one_corpus()
    return tuple(probe.build_layer_frames(
        bars, symbol, "base", 12, 0, LOOKBACK_BARS + _SAMPLES,
        margin=L1_HYSTERESIS_MARGIN))


def _window():
    _symbol, bars = _one_corpus()
    return bars[: LOOKBACK_BARS + _SAMPLES]


# --------------------------------------------------------------------------
# 1. The column
# --------------------------------------------------------------------------


def test_every_row_carries_the_shipped_l2_beside_the_rejected_one():
    """Against the old probe this fails on the first row: there was no column.

    Both are required. The winner alone cannot be read -- 0.28 means nothing
    without the 0.53 it beat -- and the loser alone is what this item exists to
    stop.
    """
    rows = _rows()
    assert rows, "no frames built"
    for row in rows:
        assert probe.L2_WINNER in row, (
            "the probe does not compute the shipped L2 scheme, so every "
            "number it prints about 'L2' is about the rejected one")
        assert probe.L2_CONTROL in row
        assert row[probe.L2_WINNER].startswith("co2t "), (
            "the transition frame lost its byte-disjoint prefix; on a "
            "byte-atom substrate that is how one layer's name gets swallowed "
            "by another's")


def test_the_transition_column_is_the_shipped_function_over_the_shipped_window():
    """Recompute it from the L1 column and demand byte equality.

    This is the test that would have caught a probe-local copy of the encoder,
    a step count read from the wrong constant, or a window that quietly
    differed from the one ``L2_CHURN_CUTS`` was fitted at -- the churn band is
    quantised by window length, so a window of 8 re-bands the symbol without
    changing a single name.
    """
    rows = _rows()
    history = [r["L1_cooccurrence"] for r in rows]
    for i, row in enumerate(rows):
        recent = history[max(0, i - probe.L2_TRANSITION_WINDOW + 1):i + 1]
        assert row[probe.L2_WINNER] == transition_motif(
            recent, steps=L2_TRANSITION_STEPS), (
            "row %d's L2 frame is not transition_motif over the last %d bars"
            % (i, probe.L2_TRANSITION_WINDOW))


def test_the_probe_calls_the_layers_encoder_rather_than_a_copy():
    """Two copies of a shipped encoder is how a probe stops measuring the ship.

    The sibling pin in
    tests/test_the_l2_gate_must_judge_the_banding_the_live_path_ships.py makes
    the same demand of ``omen_l2_scheme_probe``; this file makes it of the
    instrument that actually does label_skew and heldout_edge.
    """
    assert probe.transition_motif is transition_motif
    assert probe.sequence_motif is sequence_motif
    assert probe.IDENTIFIER_CEILING == IDENTIFIER_CEILING


def test_the_two_probes_agree_on_how_far_back_a_change_keyed_scheme_looks():
    """A window that drifts between the two probes is a silent re-banding.

    ``omen_l2_scheme_probe`` measured the shipped churn cuts at ITS default
    window. If this probe read a different one, the two would print different
    numbers for the same named scheme and neither would be wrong on its own
    terms -- the worst kind of disagreement to debug.
    """
    args = scheme_probe.build_parser().parse_args(["--corpus", "x.json"])
    assert probe.L2_TRANSITION_WINDOW == args.window


def test_the_rejected_scheme_is_the_more_distinct_of_the_two_on_real_bars():
    """The measured reason transition_motif won, reproduced on this feed.

    If this ever flips, the rejection recorded in [fa75fa1a] no longer
    describes the encoder we ship and the choice needs re-measuring rather
    than inheriting.
    """
    rows = _rows()
    d = layer_distinctness(rows, keys=[probe.L2_WINNER, probe.L2_CONTROL])
    assert d[probe.L2_CONTROL] > d[probe.L2_WINNER], (
        "the fixed-length path is no longer the more near-unique of the two "
        "(control %.4f vs shipped %.4f)"
        % (d[probe.L2_CONTROL], d[probe.L2_WINNER]))


def test_the_two_schemes_are_not_the_same_column_under_two_names():
    """A test suite that never checks this would pass on an aliased column."""
    rows = _rows()
    assert any(r[probe.L2_WINNER] != r[probe.L2_CONTROL] for r in rows)
    assert MOTIF_SEQUENCE_STEPS != probe.L2_TRANSITION_WINDOW


# --------------------------------------------------------------------------
# 2. The banding
# --------------------------------------------------------------------------


def test_the_probes_default_hysteresis_is_the_margin_the_live_path_ships():
    """THE REGRESSION GUARD. Against the old probe this fails outright: the
    default was 0.0, which is plain relative banding -- the configuration that
    produced the pass-111 block by measuring an encoder nobody runs.
    """
    args = probe.build_parser().parse_args(["--corpus", "x.json"])
    assert args.hysteresis == L1_HYSTERESIS_MARGIN
    assert args.hysteresis > 0.0


def test_a_held_out_result_names_the_margin_and_the_layer_it_measured():
    """A number with no encoder on it is how a stale number survives a pass."""
    symbol, _bars = _one_corpus()
    edge = probe.heldout_edge(_window(), symbol, "base", 12, _TRAIN, _TEST,
                              1.3, 20,
                              relative=True, margin=L1_HYSTERESIS_MARGIN)
    assert edge.get("error") is None
    assert edge["margin"] == L1_HYSTERESIS_MARGIN
    assert edge["key"] == "L1_cooccurrence"
    assert edge["banding"] == "relative"


# --------------------------------------------------------------------------
# 3. The refusal
# --------------------------------------------------------------------------


def test_an_l2_arm_with_no_supported_train_group_is_refused_not_scored():
    """The criterion the item states, enforced in code rather than in a habit.

    A support floor nothing clears makes ``buyable`` empty by construction, so
    the rule abstains on every test bar and ``called_net - baseline_net``
    manufactures a loss out of an abstention. Measured on p108_aero at the
    maximum train region the corpus allows: the largest L2 transition group is
    17 samples DOWN against a floor of 20, so there is no arm to spend.
    """
    symbol, _bars = _one_corpus()
    edge = probe.heldout_edge(_window(), symbol, "base", 12, _TRAIN, _TEST,
                              1.3, min_support=10_000, key=probe.L2_WINNER)
    assert edge["unspent"] is True
    assert edge["train_supported_groups"] == 0
    assert "called_net" not in edge, (
        "an unspent arm returned a per-trade net, which is the abstention "
        "this branch exists to stop being quoted as a result")
    assert str(edge["train_largest_group"]) in edge["unspent_reason"]
    assert probe.L2_WINNER in edge["unspent_reason"]


def test_a_spendable_arm_still_returns_its_number():
    """The refusal must not swallow the measurable case as well."""
    symbol, _bars = _one_corpus()
    edge = probe.heldout_edge(_window(), symbol, "base", 12, _TRAIN, _TEST,
                              1.3, 20,
                              relative=True, margin=L1_HYSTERESIS_MARGIN)
    assert edge.get("unspent") is False
    assert edge["train_supported_groups"] >= 1
    assert "called_net" in edge and "baseline_net" in edge


def test_the_scored_layers_own_vocabulary_is_reported_beside_l1s():
    """``l1_vocabulary_train`` must keep meaning L1 whatever layer is scored.

    Existing artifacts and tests read that name. A field whose meaning depends
    on an argument is how an L2 vocabulary gets quoted as an L1 one.
    """
    symbol, _bars = _one_corpus()
    window = _window()
    l1 = probe.heldout_edge(window, symbol, "base", 12, _TRAIN, _TEST,
                            1.3, 20,
                            relative=True, margin=L1_HYSTERESIS_MARGIN)
    l2 = probe.heldout_edge(window, symbol, "base", 12, _TRAIN, _TEST,
                            1.3, 1,
                            relative=True, margin=L1_HYSTERESIS_MARGIN,
                            key=probe.L2_WINNER)
    assert l1["l1_vocabulary_train"] == l2["l1_vocabulary_train"]
    assert l2["vocabulary_train"] > l2["l1_vocabulary_train"], (
        "the L2 transition vocabulary is not wider than the L1 one it is "
        "built from, so either the column is aliased or the two fields are "
        "reporting the same set")
    assert l2["key"] == probe.L2_WINNER
