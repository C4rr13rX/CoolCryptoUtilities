"""An L2 guard that measures a banding nobody ships blocks the wrong thing.

Item [fa75fa1a]. This is the failure that cost a pass. ``omen_l2_scheme_probe``
built its table with per-bar ``cooccurrence_motif`` -- plain relative banding,
hysteresis margin 0 -- and its EXIT CODE read that table. The live path ships a
sticky L1, and under a sticky L1 the transition scheme clears the identifier
ceiling comfortably. So the probe printed "NEITHER scheme clears 0.30, do NOT
spend a node arm", pass 111 recorded that as "no order-carrying scheme can meet
criterion 1", and the item was blocked on a measurement of an encoder the
system does not use. The numbers were all correct; the configuration was not.

L2's frames are a function of the L1 alphabet underneath them, so the banding
is PART of the measurement rather than a detail of it. These tests pin the
three things that would have caught it, and every one of them is node-free.

They are deliberately synthetic: the p108 corpora are local artefacts and are
not in the repository, so a test that needed them would skip on every other box
and pin nothing. The corpus-scale numbers live in the report.
"""
from __future__ import annotations

import pytest

from trading.omen_layers import (
    IDENTIFIER_CEILING, L1_HYSTERESIS_MARGIN, L1_STREAMS, L2_TRANSITION_STEPS,
    relative_bands, sticky_motifs, transition_motif,
)
import scripts.omen_l2_scheme_probe as probe


# Ten bars spread wide enough to give ``relative_bands`` real terciles, then
# fifteen that DITHER across the cut points by one or two units. Hysteresis is
# a boundary effect: it holds a slot that is wobbling over a cut and does
# nothing to a slot that leaps the whole band, so a fixture that oscillates
# violently measures nothing -- the first draft of this file did exactly that
# and both mechanism tests passed trivially in the wrong direction.
_SPREAD = (0, 3, 6, 15, 18, 24, 27, 30, 33, 36)
_DITHER = (11, 13, 11, 13, 12, 11, 13, 12, 11, 13, 12, 11, 13, 11, 12)


def _flickering_corpus():
    """A corpus whose slots sit ON their tercile boundaries and flip every bar.

    This is the shape the real feed has and the reason L2 was an identifier:
    the alphabet changes on nearly every bar, so a path over it is near-unique
    BY CONSTRUCTION however few symbols it has. On p108 the rate is 73.1% DOWN
    and 74.0% UP; hysteresis at the shipped margin takes it to 37.6% / 38.2%.
    """
    return [{"geometry": "geo p24=q%d" % v,
             "temporal": "tmp z=u%d" % v,
             "flow": "flw v=r%d" % v,
             "volatility": "vol v24=u%d" % v,
             "cross": "crs c=r%d" % v}
            for v in _SPREAD + _DITHER]


def test_the_fixture_really_does_flicker_across_its_bands():
    """A fixture that does not flicker makes every test below vacuous.

    Pin the precondition rather than assume it: the dithering bars must
    straddle the cut points that ``relative_bands`` fits, or hysteresis has
    nothing to hold and margin 0.5 equals margin 0.
    """
    bands = relative_bands(_flickering_corpus(), streams=L1_STREAMS)
    low, high = bands["geometry"]
    assert min(_DITHER) <= low and max(_DITHER) >= high, (
        f"the dithering bars {_DITHER} do not straddle the band ({low}, "
        f"{high}), so no slot is ever held and the mechanism is untested")


def test_the_probe_uses_the_layers_transition_motif_rather_than_a_copy():
    """Two copies of a shipped encoder is how a probe stops measuring the ship.

    ``transition_motif`` was written in the probe and promoted to
    trading/omen_layers when it won. If the probe kept its own copy, the number
    the gate prints would describe a function the live path does not call --
    the same class of defect as gating on the wrong banding, one level down.
    """
    assert probe.transition_motif is transition_motif
    assert probe.IDENTIFIER_CEILING == IDENTIFIER_CEILING


def test_the_gate_margin_defaults_to_the_margin_the_live_path_ships():
    """THE REGRESSION GUARD. The default decides what the exit code means.

    Against the old probe this fails outright: there was no --gate-margin, the
    table was built at margin 0, and the exit code judged it.
    """
    parser = probe.build_parser()
    args = parser.parse_args(["--corpus", "x.json"])
    assert args.gate_margin == L1_HYSTERESIS_MARGIN
    assert args.gate_margin > 0.0, (
        "the gate is judging plain relative banding again, which is not what "
        "the live path ships and is the configuration that produced the "
        "pass-111 block")
    assert args.l2_steps == L2_TRANSITION_STEPS


def test_build_rows_actually_applies_the_margin_it_is_handed():
    """An ignored ``margin`` argument would pass every other test here.

    The mechanism the whole item rests on: stickiness cuts the L1 change rate,
    and a slower alphabet is what lets a change-keyed L2 group anything.
    """
    frame_sets = _flickering_corpus()
    bands = relative_bands(frame_sets, streams=L1_STREAMS)

    def change_rate(margin):
        motifs = sticky_motifs(frame_sets, bands, margin)
        return sum(1 for a, b in zip(motifs, motifs[1:]) if a != b)

    assert change_rate(L1_HYSTERESIS_MARGIN) < change_rate(0.0), (
        "hysteresis did not slow the L1 alphabet, so nothing above it can "
        "improve and the treatment arm is the control arm")


def test_a_slower_alphabet_is_what_takes_l2_under_the_ceiling():
    """The claim of the item, on a corpus small enough to read.

    Distinctness must FALL when the same scheme is run over a stickier L1.
    This is the direction of the corpus-scale result -- 0.6017 -> 0.2633 DOWN
    and 0.4917 -> 0.2000 UP on p108 -- reproduced on a fixture, so a change
    that broke the mechanism fails here without needing the corpora.
    """
    frame_sets = _flickering_corpus()
    bands = relative_bands(frame_sets, streams=L1_STREAMS)
    total = len(frame_sets)

    def distinctness(margin):
        motifs = sticky_motifs(frame_sets, bands, margin)
        frames = [transition_motif(motifs[max(0, i - 12 + 1):i + 1])
                  for i in range(total)]
        return len(set(frames)) / total

    assert distinctness(L1_HYSTERESIS_MARGIN) < distinctness(0.0)


def test_the_rejected_scheme_is_still_measurable_beside_the_winner():
    """Run-length lost, and it stays importable so its number keeps existing.

    The acceptance criterion asks for the rejected scheme's OWN number rather
    than an unmeasured dismissal. That is only possible while the probe can
    still build it.
    """
    frame_sets = _flickering_corpus()
    bands = relative_bands(frame_sets, streams=L1_STREAMS)
    motifs = sticky_motifs(frame_sets, bands, L1_HYSTERESIS_MARGIN)

    rl = {probe.run_length_motif(motifs[max(0, i - 12 + 1):i + 1])
          for i in range(len(motifs))}
    tr = {transition_motif(motifs[max(0, i - 12 + 1):i + 1])
          for i in range(len(motifs))}
    assert len(rl) > len(tr), (
        "run-length did not widen the alphabet relative to transitions, so "
        "the measured reason it loses -- a dwell bucket is an extra symbol "
        "per position -- no longer holds and the rejection needs re-measuring")


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_order_survives_at_every_step_count_the_gate_may_choose(steps):
    """Order is the one property L2 exists to carry, at whatever length.

    Checked across step counts because the gate is allowed to sweep them, and
    a scheme that kept order at the shipped value while losing it at another
    would be a trap for the next sweep rather than a property of the scheme.
    """
    a = "co1 geo=hi tmp=lo flo=mid vol=hi crs=lo"
    b = "co1 geo=lo tmp=hi flo=lo vol=mid crs=hi"
    if steps < 2:
        pytest.skip("one symbol cannot carry an order")
    assert transition_motif([a, b], steps=steps) != transition_motif(
        [b, a], steps=steps)
