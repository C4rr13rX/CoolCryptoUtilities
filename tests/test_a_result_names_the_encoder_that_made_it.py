"""A held-out result must record WHICH encoder produced it.

THE FAILURE THIS PREVENTS. Pass 110 measured the L1 motif layer's held-out
buy-low edge and reported a negative: -0.5901% UP and -0.2128% DOWN, with
trough precision at or below the base rate in both windows. That result sat in
``data/brain_experiments/LAYER-L1-pass110-cove.md`` reading as settled for a
day, and it was wrong -- not because the arithmetic was wrong but because
``_band_of`` bands on absolute token signs, volatility is a MAGNITUDE whose
frame carries three ``u`` tokens every bar, and flow's quantiles sit near the
neutral centre. Three of five motif slots never varied. Re-measured through
``relative_bands`` the DOWN edge is +1.0378% and trough precision is 32.1%
against a 14.3% base rate -- the opposite sign.

The encoder fix landed in ``trading/omen_layers.py`` (7d2a74e) and the probe
still could not reach it: ``build_layer_frames`` called ``cooccurrence_motif``
with no ``bands`` argument, so every arm ran the blind encoder whatever else
the caller changed. A flag that exists and is unreachable from the instrument
is the same shape as ``omen_experiment.py:437`` reporting one query set and
firing another, and it cost a pass the same way.

So there are two things to hold, and this file holds both:

1. ``--relative-bands`` must actually CHANGE the motif. A flag that is
   accepted and ignored is worse than no flag: it produces two arms that look
   like a controlled comparison and are byte-for-byte the same run.
2. Every held-out result must carry the encoder that produced it, so a number
   lifted out of a JSON file cannot be silently attributed to the wrong one.

Both assertions fail against the pre-fix code: (1) because ``bands`` was never
threaded through, and (2) because the result dict had no ``banding`` key at
all.
"""

from __future__ import annotations

import unittest

from trading.omen_layers import L1_STREAMS, cooccurrence_motif, relative_bands


def _frames(volatility_token: str) -> dict:
    """One L0 frame set. Only ``volatility`` varies between calls.

    The token shapes are the real ones from the corpus -- ``vol v24=u10
    v168=u11 exp=r11 rng=u10`` is what the encoder actually sees, and it is
    all-``u`` by construction because volatility cannot be negative. That is
    the whole defect, so the fixture must not tidy it into something signed.
    """
    return {
        "geometry": "geo p24=q5 body=q17 uw=q0",
        "temporal": "tem z6=u12 z24=d8",
        "flow": "flw v=r7 vt=r10 bs=q9 bs24=q9",
        "volatility": volatility_token,
        "cross": "cro b=u3 e=d2",
    }


class ARelativeBandRunMustDifferFromASignBandRun(unittest.TestCase):
    """The flag must move the motif, not merely be accepted."""

    def test_sign_banding_is_blind_to_a_magnitude_stream(self) -> None:
        """Volatility reads the same band at every magnitude under signs.

        A stream whose tokens never change sign has no sign information, so
        two bars an order of magnitude apart in volatility produce the SAME
        band. This is the measurement that made three of five slots dead, and
        it must be asserted rather than described -- a test that only checked
        the fixed path would pass while the defect was still reachable.
        """
        quiet = cooccurrence_motif(_frames("vol v24=u1 v168=u1 exp=r1 rng=u1"))
        loud = cooccurrence_motif(_frames("vol v24=u19 v168=u19 exp=r19 rng=u19"))
        self.assertEqual(
            quiet,
            loud,
            "sign banding is expected to be blind here; if this ever differs "
            "the defect is fixed at the source and this test should be "
            "rewritten rather than deleted",
        )

    def test_relative_banding_separates_what_sign_banding_merged(self) -> None:
        """The same two bars must land in DIFFERENT motifs once banded.

        This is the assertion that fails against the pre-fix probe path. It
        does not check a flag or a log line -- it checks that the motif string
        the layer emits actually changed, which is the only thing downstream
        consumes.
        """
        corpus = [
            _frames("vol v24=u%d v168=u%d exp=r%d rng=u%d" % (i, i, i, i))
            for i in range(1, 20)
        ]
        bands = relative_bands(corpus, streams=list(L1_STREAMS))
        self.assertIn(
            "volatility",
            bands,
            "relative_bands must find terciles for a stream that genuinely "
            "varies; omitting it would silently restore the blind behaviour",
        )

        quiet = cooccurrence_motif(corpus[0], bands=bands)
        loud = cooccurrence_motif(corpus[-1], bands=bands)
        self.assertNotEqual(
            quiet,
            loud,
            "relative banding produced the same motif for the quietest and "
            "loudest bars in its own corpus -- the cut points are not being "
            "applied, so an arm run with --relative-bands is the blind arm "
            "wearing a different name",
        )


class AHeldOutResultMustNameItsEncoder(unittest.TestCase):
    """The result dict must carry the encoder, not just the invocation."""

    def _edge(self, relative: bool) -> dict:
        from scripts.omen_layer_probe import heldout_edge

        bars = _synthetic_bars()
        return heldout_edge(
            bars,
            symbol="AERO-USDC",
            chain="base",
            horizon=4,
            train=120,
            test=60,
            min_lift=1.3,
            min_support=5,
            relative=relative,
        )

    def test_the_banding_is_reported_on_the_result(self) -> None:
        """Not on stdout, not in the filename -- on the object.

        A number is quoted from the JSON artifact long after the run, and the
        pass-110 reports were mis-cited for exactly one pass because the
        artifact had no field to check. stdout is not an artifact.
        """
        for relative, expected in ((False, "sign"), (True, "relative")):
            with self.subTest(relative=relative):
                out = self._edge(relative)
                if out.get("error"):
                    self.skipTest(
                        "synthetic corpus too short for a held-out split: "
                        + str(out["error"])
                    )
                self.assertEqual(out.get("banding"), expected)
                self.assertIn("band_streams", out)
                self.assertIn("l1_vocabulary_train", out)

    def test_the_two_bandings_do_not_produce_the_same_vocabulary(self) -> None:
        """Two arms that agree exactly are one arm run twice.

        Gale's negative control on the query path (A vs A moved 0/120, A vs
        A+rel moved 79/120) is the pattern being reused: an identical result
        across a change is evidence the change did not fire, and it must fail
        the test rather than be reported as a finding.
        """
        sign = self._edge(False)
        rel = self._edge(True)
        if sign.get("error") or rel.get("error"):
            self.skipTest("synthetic corpus too short for a held-out split")
        self.assertNotEqual(
            sign["l1_vocabulary_train"],
            rel["l1_vocabulary_train"],
            "sign and relative banding produced identically sized motif "
            "vocabularies on the same window -- the bands are not reaching "
            "cooccurrence_motif",
        )


def _synthetic_bars() -> list:
    """A deterministic OHLCV corpus with real variation in every stream.

    Built rather than loaded: this test must not depend on
    ``data/brain_experiments/p108_aero_*.json`` being present, and it must not
    read a file another agent may be rewriting. No randomness -- a flaky guard
    test is a guard nobody keeps.
    """
    bars = []
    price = 100.0
    for i in range(320):
        # Two superimposed cycles so geometry, temporal and volatility all
        # move, and volume swings so flow does too. Amplitude grows with i so
        # the volatility MAGNITUDE genuinely varies -- a constant-amplitude
        # series would make the fixed encoder look correct.
        swing = (1.0 + i / 160.0) * (0.9 if i % 7 < 3 else -0.6)
        price = max(1.0, price + swing)
        high = price * (1.0 + 0.004 * (1 + i % 5))
        low = price * (1.0 - 0.004 * (1 + i % 3))
        bars.append(
            {
                "timestamp": 1_700_000_000 + i * 900,
                "open": price - swing,
                "high": high,
                "low": low,
                "close": price,
                "volume": 1000.0 * (1 + (i % 11)),
            }
        )
    return bars


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
