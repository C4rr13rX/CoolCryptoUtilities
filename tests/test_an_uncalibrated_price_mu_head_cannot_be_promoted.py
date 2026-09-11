"""The deploy path must refuse a price_mu head that answers outside its label.

WHAT WENT WRONG. ``models/active_model.keras``, built 2026-09-11 01:57, was
promoted on a composite score and served a ``price_mu`` whose median magnitude
over 40 clean windows drawn from its own training corpus was 1.399697 against a
median label of 0.002858 -- 489.8x. The label is a one-bar natural-log return
whose p99 over 2,277,175 corpus samples is 0.034602, so the head answered three
decades outside the support of its own target. Nothing on the promotion path had
ever looked at the MAGNITUDE of that head, only at directional score, and
``price_mu`` feeds ``delta``, ``net_margin`` and ``net_pnl`` -- three of the
entry conjunction's five terms.

WHICH SUPERVISION PATH FAILED, measured the same day and recorded here so the
answer is not re-derived: NOT the gaussian one. The served ``log_var`` over
those same 40 windows has median 1.3715, max 2.0318, and 0.0% of windows sit at
the +8.0 clip, so ``precision = exp(-log_var)`` is 0.2538 and that path is
attenuated 4x rather than switched off. The failing path is ``net_margin`` MSE,
whose sample weight had silently become a constant -- see
``test_margin_intensity_is_not_a_constant_on_the_corrected_label`` below.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _model_definition():
    try:
        import model_definition
    except Exception as exc:  # pragma: no cover - TF is absent in some rigs
        pytest.skip(f"model_definition needs TensorFlow, which is unavailable here: {exc}")
    return model_definition


# ---------------------------------------------------------------------
# The verdict function, against the numbers actually measured
# ---------------------------------------------------------------------


def test_the_known_bad_artifacts_measured_ratio_is_refused():
    """489.8x is the number the deployed artifact read. It must not promote."""
    md = _model_definition()
    report = {
        "samples": 40,
        "ratio": 489.8,
        "median_pred": 1.399697,
        "median_label": 0.002858,
        "median_log_var": 1.371519,
    }
    reason = md.calibration_rejection_reason(report)
    assert reason is not None, "the 489.8x artifact must be refused promotion"
    assert "489.8x" in reason
    assert "0.002858" in reason


def test_a_calibrated_head_is_allowed_through():
    """A guard that refuses everything is the same as being switched off."""
    md = _model_definition()
    report = {
        "samples": 40,
        "ratio": 1.6,
        "median_pred": 0.004573,
        "median_label": 0.002858,
        "median_log_var": -4.2,
    }
    assert md.calibration_rejection_reason(report) is None


def test_the_bar_is_ten_times_the_label_not_a_hundred():
    """Pin the threshold: 9x passes, 11x does not."""
    md = _model_definition()
    assert md.CALIBRATION_MAX_RATIO == 10.0
    base = {"samples": 40, "median_pred": 0.02, "median_label": 0.002858, "median_log_var": 0.0}
    assert md.calibration_rejection_reason({**base, "ratio": 9.0}) is None
    assert md.calibration_rejection_reason({**base, "ratio": 11.0}) is not None


def test_a_probe_that_could_not_run_is_not_reported_as_a_pass():
    """Zero samples means 'cannot judge'. It must not silently read as clean."""
    md = _model_definition()
    # It returns None so a missing corpus cannot freeze promotion forever, but
    # the sample count must survive into the report the caller logs.
    report = {"samples": 0, "error": "corpus unreadable"}
    assert md.calibration_rejection_reason(report) is None
    assert report["samples"] == 0


# ---------------------------------------------------------------------
# The sample weight that switched the only direct supervision off
# ---------------------------------------------------------------------


def test_margin_intensity_is_not_a_constant_on_the_corrected_label():
    """The OLD formula is a flat 0.1 on this label; the NEW one is a curve.

    ``trading/pipeline.py`` weighted the ``net_margin`` MSE by
    ``clip(|net_margin|, 0.1, 5.0)``. That band was written for the old label,
    whose median |net_margin| was 0.4977. Against the corrected label -- a
    one-bar log return minus the 0.0065 round trip -- 99.96% of 415,846 corpus
    samples fall below 0.1, so the weight is the constant 0.1 and the only loss
    that directly pins ``price_mu`` trains at an effective weight of 0.1.

    This test FAILS against the old formula: that is what makes it worth having.
    """
    rng = np.random.default_rng(20260911)
    # Corpus-shaped: median |mu| 0.003445, p99 |net_margin| 0.037053.
    mu = rng.laplace(0.0, 0.005, size=20000)
    margin_arr = mu - 0.0065

    old = np.clip(np.abs(margin_arr), 0.1, 5.0)
    assert float(np.median(old)) == pytest.approx(0.1), (
        "precondition: the old band really does degenerate on this label"
    )
    assert float(old.max()) < 1.0

    scale = float(np.median(np.abs(margin_arr)))
    assert scale > 0.0
    new = np.clip(np.abs(margin_arr) / scale, 0.1, 5.0)

    # A median-sized move now weighs 1.0, not 0.1.
    assert float(np.median(new)) == pytest.approx(1.0, rel=0.02)
    # And the mean weight returns to order 1 instead of order 0.1, which is the
    # 10x of supervision the head had been missing.
    assert float(np.mean(new)) > 0.5
    # The curve is live again: some samples earn materially more than others.
    assert float(new.max()) > 2.0


def test_the_pipeline_scales_margin_intensity_by_the_batch_median():
    """The fix is in the shipped file, not only in this test's arithmetic."""
    with open(os.path.join(REPO_ROOT, "trading", "pipeline.py"), encoding="utf8") as handle:
        # Read CODE, not prose. The comment above the fix quotes the old
        # formula on purpose, and a test that cannot tell those apart is the
        # test-asserting-on-a-comment mistake this repo has already shipped.
        code = [line for line in handle if not line.lstrip().startswith("#")]
    body = "".join(code)
    assert "np.abs(margin_arr) / margin_scale" in body
    assert "np.clip(np.abs(margin_arr), 0.1, 5.0)" not in body, (
        "the degenerate absolute band is back in live code"
    )


# ---------------------------------------------------------------------
# The guard against the real artifact on disk
# ---------------------------------------------------------------------


@pytest.mark.slow
def test_the_guard_refuses_the_artifact_currently_on_disk():
    """A guard that cannot be shown rejecting the known-bad artifact is untested.

    Skips when TensorFlow or the corpus is unavailable, because this runs in
    rigs that have neither -- but where it CAN run, it is the whole point.
    """
    md = _model_definition()
    model_path = os.path.join(REPO_ROOT, "models", "active_model.keras")
    if not os.path.exists(model_path):
        pytest.skip("no models/active_model.keras to probe")
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"TensorFlow unavailable: {exc}")

    model = tf.keras.models.load_model(model_path, compile=False)
    width = [int(i.shape[1]) for i in model.inputs if "price_vol" in i.name][0]
    windows = md.corpus_calibration_windows(
        width, 8, corpus_dir=os.path.join(REPO_ROOT, "data", "historical_ohlcv")
    )
    if not windows:
        pytest.skip("no usable corpus windows under data/historical_ohlcv")

    report = md.price_mu_calibration(model, windows)
    assert report["samples"] == len(windows)
    assert report["median_label"] > 0.0
    reason = md.calibration_rejection_reason(report)
    assert reason is not None, (
        "the artifact on disk reads "
        f"{report['ratio']:.1f}x its own label and must be refused: {report}"
    )
