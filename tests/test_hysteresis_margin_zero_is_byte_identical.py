#!/usr/bin/env python3
"""Hysteresis at margin=0 must BE the plain relative-banded encoder, not resemble it.

[2a53f971] wires ``sticky_motifs`` into trading/omen_layers.py so the live path
can use it. Every number the item asks for is a BACK-TO-BACK comparison --
margin 0.50 against margin 0 on one fabric -- and that comparison is only about
hysteresis if margin=0 reproduces ``cooccurrence_motif(frames, bands=bands)``
BYTE FOR BYTE. If it merely resembles it, the arm changed two things and says
nothing about either.

The part that is easy to get wrong, and already was once: ``relative_bands``
OMITS a stream whose terciles collapse, and ``cooccurrence_motif`` falls back to
absolute SIGN banding for exactly those streams. A hysteresis implementation
that emits "na" for an omitted stream looks correct on a corpus where every
stream bands, and silently becomes a different encoder on one where a stream is
constant -- which is the normal case for volatility on a quiet symbol.

Run: python -X utf8 -m pytest tests/test_hysteresis_margin_zero_is_byte_identical.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.omen_layers import (  # noqa: E402
    L1_STREAMS, cooccurrence_motif, relative_bands, sticky_motifs,
)


def _corpus_with_a_collapsed_stream():
    """Frame sets where volatility is CONSTANT and every other stream varies.

    A constant stream has no terciles, so ``relative_bands`` drops it and the
    fallback branch is the only thing that can band it.
    """
    return [{"geometry": "geo p24=q%d" % v,
             "temporal": "tmp z=u%d" % v,
             "flow": "flw v=r%d" % v,
             "volatility": "vol v24=u10 v168=u10",   # identical on every bar
             "cross": "crs c=r%d" % v}
            for v in (1, 18, 3, 17, 5, 15, 7, 13, 9, 11, 2, 19)]


def test_relative_bands_really_does_omit_the_collapsed_stream():
    """The fixture must exercise the fallback, or the next test proves nothing.

    A test whose interesting branch is never reached passes for the wrong
    reason. Pin the precondition rather than assume it.
    """
    frame_sets = _corpus_with_a_collapsed_stream()
    bands = relative_bands(frame_sets)
    assert "volatility" not in bands, bands
    assert set(bands) == {"geometry", "temporal", "flow", "cross"}


def test_margin_zero_matches_plain_banding_including_the_omitted_stream():
    """The whole comparison arm rests on this equality."""
    frame_sets = _corpus_with_a_collapsed_stream()
    bands = relative_bands(frame_sets)

    assert sticky_motifs(frame_sets, bands, 0.0) == [
        cooccurrence_motif(f, bands=bands) for f in frame_sets]


def test_the_omitted_stream_bands_by_SIGN_and_not_to_na():
    """The failure this file is named for.

    An implementation that emits "na" for a stream ``relative_bands`` omitted
    still passes a byte-identity test run on a corpus where nothing collapses.
    Here volatility is three positive tokens on every bar, so sign banding must
    call it "hi" -- and "vol=na" is the wrong answer that looks like a right
    one.
    """
    frame_sets = _corpus_with_a_collapsed_stream()
    bands = relative_bands(frame_sets)

    for motif in sticky_motifs(frame_sets, bands, 0.0):
        assert "vol=hi" in motif, motif
        assert "vol=na" not in motif, motif


def test_margin_zero_matches_plain_banding_with_no_bands_at_all():
    """A caller with no fitted corpus must get the SAME motifs from either path.

    ``bands=None`` is the documented fallback for a caller that has no corpus
    to fit on. If hysteresis diverged there, a live path that forgot to pass
    cut points would silently be running a third encoder.
    """
    frame_sets = _corpus_with_a_collapsed_stream()

    assert sticky_motifs(frame_sets, None, 0.0) == [
        cooccurrence_motif(f) for f in frame_sets]
    assert sticky_motifs(frame_sets, {}, 0.5) == [
        cooccurrence_motif(f) for f in frame_sets]


def test_a_missing_frame_is_na_under_both_encoders():
    """A hole in the corpus must not become a band either encoder invented."""
    frame_sets = _corpus_with_a_collapsed_stream()
    bands = relative_bands(frame_sets)
    holed = [dict(f) for f in frame_sets]
    holed[3]["geometry"] = ""
    holed[7].pop("flow")

    assert sticky_motifs(holed, bands, 0.0) == [
        cooccurrence_motif(f, bands=bands) for f in holed]
    assert "geo=na" in sticky_motifs(holed, bands, 0.0)[3]
    assert "flo=na" in sticky_motifs(holed, bands, 0.0)[7]


def test_hysteresis_holds_a_band_that_plain_banding_would_flip():
    """margin>0 must actually DO something, or the equality above is trivial.

    A slot sitting just past a tercile boundary flips every bar under plain
    banding and holds under hysteresis. Without this, an implementation that
    ignored ``margin`` entirely would pass every other test in this file.
    """
    frame_sets = _corpus_with_a_collapsed_stream()
    bands = relative_bands(frame_sets)

    def changes(margin):
        motifs = sticky_motifs(frame_sets, bands, margin)
        return sum(1 for a, b in zip(motifs, motifs[1:]) if a != b)

    assert changes(1.0) < changes(0.0)
    assert changes(1.0) <= changes(0.5) <= changes(0.0)


def test_every_l1_stream_appears_in_the_sticky_motif():
    """A dropped slot is the defect 7d2a74e was written to end.

    Five streams, five slots, in L1_STREAMS order -- the same shape
    ``cooccurrence_motif`` emits.
    """
    frame_sets = _corpus_with_a_collapsed_stream()
    motif = sticky_motifs(frame_sets, relative_bands(frame_sets), 0.5)[0]
    assert motif.startswith("co1 ")
    assert [t.split("=")[0] for t in motif.split()[1:]] == [
        name[:3] for name in L1_STREAMS]
