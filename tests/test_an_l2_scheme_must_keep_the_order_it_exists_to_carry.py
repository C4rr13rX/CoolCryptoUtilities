"""An L2 scheme that collapses repeats must not collapse ORDER with them.

Item [fa75fa1a]. L2 exists to carry sequence -- what happened, and in what
order. Both candidate fixes for the identifier problem work by throwing
information away (transitions drop repeats; run-length replaces a run with a
bucket), and the failure mode of any such scheme is that it throws away the
one thing the layer is for. A scheme that maps A->B->C and C->B->A to the same
frame has become a bag of motifs, and a bag of motifs is L1 with extra steps.

These are cheap, node-free, and they are the guard the acceptance criterion
asks for: prove order survives BEFORE any node arm is spent.
"""
from __future__ import annotations

import pytest

from trading.omen_layers import relative_bands
from scripts.omen_l2_scheme_probe import (
    IDENTIFIER_CEILING, run_length_motif, transition_motif,
)

# Three distinct L1 motifs in the shape cooccurrence_motif emits: a "co1"
# prefix then one stream=band token per stream.
A = "co1 geo=hi tmp=lo flo=mid vol=hi crs=lo"
B = "co1 geo=lo tmp=hi flo=lo vol=mid crs=hi"
C = "co1 geo=mid tmp=mid flo=hi vol=lo crs=mid"


@pytest.mark.parametrize("scheme", [transition_motif, run_length_motif])
def test_the_same_multiset_in_a_different_order_is_a_different_frame(scheme):
    forward = scheme([A, B, C], steps=3)
    backward = scheme([C, B, A], steps=3)
    assert forward != backward, (
        f"{scheme.__name__} maps A->B->C and C->B->A to the same frame; it has "
        "lost the order L2 exists to carry and is a bag of motifs")


def test_transitions_collapse_a_persistent_regime_to_one_symbol():
    """A regime that holds is the case the fixed-length path handled worst.

    Eight bars of A followed by B must not read as eight symbols; that
    collapsing is the whole mechanism by which transitions were expected to
    shrink the vocabulary.
    """
    held = transition_motif([A] * 8 + [B], steps=3)
    brief = transition_motif([A, B], steps=3)
    assert held == brief, (
        "transitions did not drop repeats, so a persistent regime still "
        "contributes one symbol per bar and nothing was gained")


def test_run_length_separates_a_long_hold_from_a_glimpse():
    """Run-length's ONLY justification over transitions is that dwell matters.

    If a motif held for eight bars and a motif seen once produce the same
    frame, run-length is transitions plus a constant, and costs a wider
    alphabet for nothing.
    """
    assert run_length_motif([A] * 8, steps=3) != run_length_motif([A], steps=3)


def test_a_run_length_frame_survives_an_empty_history():
    assert transition_motif([], steps=3) == "co2t path=na"
    assert run_length_motif([], steps=3) == "co2r path=na"


def test_the_ceiling_is_the_one_the_layer_probe_uses():
    """Two instruments disagreeing about what PASS means is worse than one.

    trading/omen_layers states the identifier ceiling; this probe must not
    carry a softer private copy, because a softer copy is how a failing layer
    gets promoted.
    """
    assert IDENTIFIER_CEILING == 0.30


def test_hysteresis_at_zero_margin_reproduces_plain_relative_banding():
    """The comparison arm must be the SAME encoder, not a similar one.

    margin=0 has to be byte-identical to `cooccurrence_motif(frames, bands)`,
    or a hysteresis result is a two-change measurement and says nothing about
    either change.
    """
    from trading.omen_layers import cooccurrence_motif
    from scripts.omen_l2_scheme_probe import sticky_motifs

    frame_sets = [{"geometry": "geo r5", "temporal": "tmp u3",
                   "flow": "flw q9", "volatility": "vol u10",
                   "cross": "crs r7"},
                  {"geometry": "geo r90", "temporal": "tmp u30",
                   "flow": "flw q1", "volatility": "vol u1",
                   "cross": "crs r70"},
                  {"geometry": "geo r50", "temporal": "tmp u15",
                   "flow": "flw q5", "volatility": "vol u5",
                   "cross": "crs r35"}]
    bands = relative_bands(frame_sets)
    assert sticky_motifs(frame_sets, bands, 0.0) == [
        cooccurrence_motif(f, bands=bands) for f in frame_sets]


def test_hysteresis_can_only_reduce_the_change_rate():
    """A sticky slot must never flicker MORE than a free one.

    If a margin ever raised the change rate, the stickiness would be
    implemented backwards and every number measured through it would be
    upside down.
    """
    from scripts.omen_l2_scheme_probe import sticky_motifs

    frame_sets = [{"geometry": f"geo r{v}", "temporal": f"tmp u{v}",
                   "flow": f"flw q{v}", "volatility": f"vol u{v}",
                   "cross": f"crs r{v}"}
                  for v in (1, 90, 2, 88, 3, 91, 4, 87, 5, 89, 45, 46)]
    bands = relative_bands(frame_sets)

    def change_rate(margin):
        motifs = sticky_motifs(frame_sets, bands, margin)
        return sum(1 for a, b in zip(motifs, motifs[1:]) if a != b)

    assert change_rate(1.0) <= change_rate(0.5) <= change_rate(0.0)
