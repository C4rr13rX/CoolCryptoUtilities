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

from trading.omen_layers import L2_TRANSITION_STEPS, churn_band, relative_bands
from scripts.omen_l2_scheme_probe import (
    IDENTIFIER_CEILING, run_length_motif, transition_motif,
)

# Three distinct L1 motifs in the shape cooccurrence_motif emits: a "co1"
# prefix then one stream=band token per stream.
A = "co1 geo=hi tmp=lo flo=mid vol=hi crs=lo"
B = "co1 geo=lo tmp=hi flo=lo vol=mid crs=hi"
C = "co1 geo=mid tmp=mid flo=hi vol=lo crs=mid"


def _path(frame: str) -> str:
    """The kept-symbol path of an L2 frame, without its churn symbol.

    Split out so a test can assert on the path and the dwell SEPARATELY. The
    two carry different things and a test that only ever compares whole frames
    cannot say which of them a change broke -- which is how the persistent-vs-
    alternating collapse survived a green suite.
    """
    return frame.split(" chn=")[0]


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

    THE ASSERTION IS ON THE PATH, NOT ON THE WHOLE FRAME, and the difference
    is the defect Jet found in pass 116. This test used to compare whole
    frames, which pinned "dwell is discarded" as correct and never asked what
    discarding it cost -- so it could not see that the shipped encoder gave a
    persistent regime and an alternating one one identical frame. The path
    collapsing repeats is right and stays asserted; the CHURN symbol beside it
    is what keeps the two apart, and it is asserted separately below.
    """
    held = transition_motif([A] * 8 + [B], steps=3)
    brief = transition_motif([A, B], steps=3)
    assert _path(held) == _path(brief), (
        "transitions did not drop repeats, so a persistent regime still "
        "contributes one symbol per bar and nothing was gained")


def test_a_held_regime_and_an_alternating_one_are_different_frames():
    """The reopen of [fa75fa1a] criterion 3, pinned at the SHIPPED default.

    Four A then four B changes ONCE. A B A B A B A B changes SEVEN times. They
    carry the identical multiset, and at ``L2_TRANSITION_STEPS`` they collapse
    to the identical two kept symbols -- so a scheme whose frame is only the
    kept path cannot tell a regime that held from one that churned, which is
    the property L2 exists to carry. Jet reproduced exactly this in three
    seconds with no node.

    NO ``steps=`` ARGUMENT HERE, DELIBERATELY. Every other test in this file
    passes steps=3 explicitly, and that is why the suite could not see the
    failure: at 3 the pair happens to separate on path length alone, while the
    live path ships 2. A test that never exercises the shipped default cannot
    defend it.
    """
    held = transition_motif([A] * 4 + [B] * 4)
    churning = transition_motif([A, B, A, B, A, B, A, B])
    assert _path(held) == _path(churning), (
        "the premise of this test has changed: the two sequences no longer "
        "share a kept path, so it is no longer testing what the churn symbol "
        "is for")
    assert held != churning, (
        "a persistent regime and an alternating one produce the same L2 frame "
        f"at the shipped L2_TRANSITION_STEPS={L2_TRANSITION_STEPS}; the layer "
        "has lost dwell and only the churn symbol can carry it")


def test_the_churn_symbol_is_one_per_frame_not_one_per_position():
    """WHY churn and not run-length, asserted as a property of the encoder.

    Run-length carries dwell by attaching a bucket to every kept symbol, which
    multiplies the alphabet once per position -- measured 0.7117 DOWN / 0.5983
    UP against a 0.30 ceiling, worse than the fixed path it replaced. The churn
    symbol carries dwell in ONE token for the whole frame, so the alphabet
    grows by a bounded factor however many symbols the path keeps.

    If someone "improves" this by banding per position, this test fails and
    the distinctness regression is caught here rather than in a node arm.
    """
    frame = transition_motif([A] * 4 + [B] * 4 + [C] * 4, steps=3)
    assert frame.count("chn=") == 1, (
        "the churn symbol appears more than once, so it is banding per "
        "position and has become run-length, which fails the guard")
    assert len(_path(frame).split("|")) == 3, (
        "the path stopped keeping its symbols; this test is no longer "
        "measuring a three-symbol frame")


def test_run_length_separates_a_long_hold_from_a_glimpse():
    """Run-length's ONLY justification over transitions is that dwell matters.

    If a motif held for eight bars and a motif seen once produce the same
    frame, run-length is transitions plus a constant, and costs a wider
    alphabet for nothing.
    """
    assert run_length_motif([A] * 8, steps=3) != run_length_motif([A], steps=3)


def test_a_run_length_frame_survives_an_empty_history():
    assert transition_motif([], steps=3) == "co2t path=na chn=na"
    assert run_length_motif([], steps=3) == "co2r path=na"


def test_churn_is_a_rate_so_it_means_the_same_at_any_window_length():
    """The cut points are on changes/adjacencies, not on a raw count.

    A raw count would silently re-band itself the moment someone changed the
    window: ONE change is a churning regime over four bars and an entrenched
    one over sixty, and a count calls them the same thing. The same PROPORTION
    of changes must read the same band at any length, or the constant's
    measured cut point stops meaning what it was measured to mean.
    """
    # Same proportion, wildly different lengths -- must agree.
    assert churn_band([A, B] * 2) == churn_band([A, B] * 30)
    assert churn_band([A] * 4) == churn_band([A] * 60)
    # Same COUNT of changes, different lengths -- must NOT agree. This is the
    # assertion a raw-count implementation fails.
    assert churn_band([A, A, B, B]) != churn_band([A] * 30 + [B] * 30)
    # And the property the layer is for, at full length.
    assert churn_band([A] * 30 + [B] * 30) != churn_band([A, B] * 30)


def test_churn_does_not_count_a_change_across_a_hole():
    """An unbuildable bar is a HOLE, and a hole is not an adjacency.

    build_rows records "" for a bar whose L0 frames could not be built rather
    than dropping it, so the two motifs either side of a gap were never
    observed as neighbours. Counting that as a change would inflate churn by
    however many bars our view of the market dropped out for, and the encoder
    would read a steady regime as churning whenever the feed hiccuped.
    """
    assert churn_band([A, "", A, "", A]) == churn_band([A, A, A])


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
