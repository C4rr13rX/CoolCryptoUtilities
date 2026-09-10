"""The model ranked symbols by their price TAG, not by the move.

``price_vol_input`` carried raw quotes into three Conv1D layers with no
normalisation in front of them -- ``ts_norm`` sits AFTER the stack, while
``tech_input`` gets LayerNormalization as its first op. The training corpus
spans 9e-08 to 123,429 on the price channel and the live stream spans 1.672e-23
to 135,744, so one set of convolution weights was covering 27 orders of
magnitude.

Probed against the live models/active_model.keras on 2026-09-07, holding the
SHAPE of the move fixed at +0.2%/step (a +12.7% move over the 60-step window)
and varying only the level:

    level 1e-06   price_dir 0.2019   price_mu -0.1298
    level 1.0     price_dir 0.5276   price_mu +0.5493
    level 21.0    price_dir 0.4822   price_mu -0.1718
    level 1206    price_dir 0.6142   price_mu -0.2791

and then holding the level at 21.0 while varying the actual direction from
-1%/step to +1%/step: price_dir moved 0.4815 -> 0.4808. The price tag swung the
entry signal 585x harder than the direction did, and the entry bar
(direction_prob >= 0.58) was reachable only above a level of ~1000.

The second channel had the matching fault. Live volume is 0.0 on 57,957 of
57,984 market_stream rows (99.95%) because no price source we poll reports a
per-bar traded size, while the corpus median is 125,401 and never zero -- so
every live inference sat off the edge of the training distribution. Moving that
channel from 0 to 125,401 swung price_mu from -0.1718 to +1.1588.
"""

import numpy as np
import pytest

# GUARD ON WHAT THIS FILE ACTUALLY IMPORTS, NOT ON A NAME A STUB SATISFIES.
#
# This file collected fine ALONE and ERRORED in the whole-suite sweep, which is
# how it stayed invisible: it failed silently AND contributed no passing count.
# The cause is cross-test pollution. tests/test_production_manager.py calls
# `_install_tf_stub()` at MODULE SCOPE (line 92), which writes a synthetic
# `tensorflow` + `tensorflow.keras` into sys.modules for the rest of the
# session and never removes it. Once that has run, `importorskip("tensorflow")`
# here SUCCEEDS against the stub instead of skipping -- and then
# `model_definition` does `from keras.callbacks import Callback`, which is the
# TOP-LEVEL keras package the stub does not register. ModuleNotFoundError at
# collection time, in a file that had already declared it can be skipped.
#
# `keras` is the honest guard: it is the module model_definition imports, and
# no stub in this suite fakes it, so this skips when the real package is absent
# (the system interpreter, where tensorflow's Windows cp313 wheel is broken)
# and runs when it is present.
tf = pytest.importorskip("tensorflow")
pytest.importorskip("keras")

import model_definition as md


WINDOW = 12


def _window(level: float, ret_per_step: float, volume: float) -> np.ndarray:
    steps = np.arange(WINDOW, dtype=np.float64)
    prices = level * np.exp(ret_per_step * steps)
    return np.stack([prices, np.full(WINDOW, float(volume))], axis=-1)


def _batch(rows) -> np.ndarray:
    return np.stack(rows).astype(np.float32)


# --------------------------------------------------------------------------
# The layer itself
# --------------------------------------------------------------------------

def test_the_same_move_at_a_different_price_tag_normalises_identically():
    layer = md.PriceVolScaleNorm()
    out = layer(_batch([
        _window(1e-06, 0.002, 0.0),
        _window(1.0, 0.002, 0.0),
        _window(21.0, 0.002, 0.0),
        _window(135744.0, 0.002, 0.0),
    ])).numpy()

    reference = out[0]
    for idx in range(1, out.shape[0]):
        assert np.allclose(out[idx], reference, atol=1e-5), (
            f"row {idx} differs from the 1e-06 window although the move is identical"
        )
    # And the value carried is the move itself: +0.2% per step, cumulative.
    assert np.allclose(reference[:, 0], 0.002 * np.arange(WINDOW), atol=1e-5)


def test_the_direction_of_the_move_survives_normalisation():
    layer = md.PriceVolScaleNorm()
    out = layer(_batch([
        _window(21.0, -0.01, 0.0),
        _window(21.0, 0.0, 0.0),
        _window(21.0, 0.01, 0.0),
    ])).numpy()

    last = out[:, -1, 0]
    assert last[0] < -0.05, f"a -1%/step window must end clearly negative, got {last[0]}"
    assert abs(last[1]) < 1e-6, f"a flat window must end at zero, got {last[1]}"
    assert last[2] > 0.05, f"a +1%/step window must end clearly positive, got {last[2]}"


def test_an_unmeasurable_volume_lands_at_the_centre_not_the_edge():
    """Live volume is 0.0 on 99.95% of rows; the corpus median is 125,401.

    Raw, that put every live inference off the edge of the training
    distribution. Relative, a window with no volume information reads exactly
    like a window of perfectly average volume -- the neutral value.
    """
    layer = md.PriceVolScaleNorm()
    out = layer(_batch([
        _window(21.0, 0.002, 0.0),        # live: no volume reported at all
        _window(21.0, 0.002, 125401.0),   # corpus median, constant across the window
    ])).numpy()

    assert np.allclose(out[0], out[1], atol=1e-5)
    assert np.allclose(out[0][:, 1], 0.0, atol=1e-6)


def test_relative_volume_still_separates_a_spike_from_a_lull():
    layer = md.PriceVolScaleNorm()
    volumes = np.full(WINDOW, 100.0)
    volumes[-1] = 100.0 * WINDOW  # one bar carries a burst
    window = np.stack([np.full(WINDOW, 21.0), volumes], axis=-1)
    out = layer(_batch([window])).numpy()[0][:, 1]

    assert out[-1] > 5.0, f"a {WINDOW}x volume burst must read high, got {out[-1]}"
    assert out[0] < 0.0, f"the quiet bars must read below average, got {out[0]}"


def test_a_degenerate_window_cannot_hand_the_convolutions_an_infinity():
    """A leading zero used to divide the window by ~0.

    ``_prepare_inputs`` reads ``row.get("price", 0.0)``, so a missing quote
    arrives as a literal 0.0 -- and this runs inside the market-stream callback,
    where an exception costs the whole tick including the ghost lane.
    """
    layer = md.PriceVolScaleNorm()
    leading_zero = np.zeros((WINDOW, 2), dtype=np.float64)
    leading_zero[2:, 0] = 2.0 * np.exp(0.005 * np.arange(WINDOW - 2))
    rows = [
        leading_zero,
        np.zeros((WINDOW, 2), dtype=np.float64),                 # nothing at all
        _window(1e-23, 0.002, 0.0),                              # smallest live price seen
        _window(135744.0, 0.002, 1e12),                          # largest live price seen
    ]
    out = layer(_batch(rows)).numpy()

    assert np.all(np.isfinite(out)), "normalisation produced NaN or inf"
    assert np.allclose(out[1], 0.0), "an empty window must read as no information"
    # The anchor is the first STRICTLY POSITIVE price, so the real move survives
    # the two missing bars ahead of it.
    assert out[0][-1, 0] > 0.04


# --------------------------------------------------------------------------
# The graph
# --------------------------------------------------------------------------

_TINY_MODEL = None


def _tiny_model():
    # Built once: three tests need it and construction dominates this file's
    # runtime. None of them mutate it.
    global _TINY_MODEL
    if _TINY_MODEL is not None:
        return _TINY_MODEL
    model, headline_vec, full_vec, _, _ = md.build_multimodal_model(
        window_size=WINDOW,
        tech_count=4,
        sent_seq_len=4,
        headline_vocab=32,
        headline_len=4,
        headline_dim=8,
        full_vocab=32,
        full_len=8,
        full_dim=8,
        hidden_1=16,
        hidden_2=8,
        asset_vocab_size=2,
        model_template="tiny",
    )
    # The pipeline adapts these before it serves or saves anything; an
    # unadapted lookup table raises out of call() and out of save().
    headline_vec.adapt(tf.constant(["a price move", "flat"]))
    full_vec.adapt(tf.constant(["a price move", "flat"]))
    _TINY_MODEL = model
    return model


def test_the_price_window_is_normalised_before_the_convolutions_see_it():
    model = _tiny_model()
    names = {layer.name for layer in model.layers}
    assert md.PRICE_VOL_NORM_LAYER in names

    norm = model.get_layer(md.PRICE_VOL_NORM_LAYER)
    conv = model.get_layer("ts_d1")
    # ts_d1 must consume the normalised tensor, not the raw Input. Before the
    # fix its inbound tensor was price_vol_input itself.
    conv_source = conv.input._keras_history.operation.name
    assert conv_source == md.PRICE_VOL_NORM_LAYER, (
        f"ts_d1 reads '{conv_source}'; the raw quotes are reaching the convolutions"
    )
    assert norm.input._keras_history.operation.name == "price_vol_input"


def test_the_whole_model_gives_the_same_answer_at_every_price_level():
    """The end-to-end assertion the live artifact fails.

    On models/active_model.keras this same sweep moved price_dir 0.2019 ->
    0.6142 and price_mu -0.3654 -> +0.5493.
    """
    model = _tiny_model()
    levels = [1e-06, 1.0, 21.0, 1206.0, 135744.0]
    inputs = {
        "price_vol_input": _batch([_window(lvl, 0.002, 0.0) for lvl in levels]),
        "sentiment_seq": np.zeros((len(levels), 4, 1), np.float32),
        "headline_text": tf.constant([["a price move"]] * len(levels)),
        "full_text": tf.constant([[""]] * len(levels)),
        "tech_input": np.zeros((len(levels), 4), np.float32),
        "hour_input": np.full((len(levels), 1), 3, np.int32),
        "dow_input": np.zeros((len(levels), 1), np.int32),
        "gas_fee_input": np.full((len(levels), 1), 0.0015, np.float32),
        "tax_rate_input": np.full((len(levels), 1), 0.005, np.float32),
        "asset_id_input": np.zeros((len(levels), 1), np.int32),
    }
    order = [t.name.split(":")[0] for t in model.inputs]
    preds = model([tf.constant(inputs[name]) for name in order], training=False)
    out_names = list(getattr(model, "output_names", None) or [])

    for name, tensor in zip(out_names, preds):
        arr = np.asarray(tensor)
        # Per-column range ACROSS the five levels: how much this output moved
        # when only the price tag changed.
        spread = float(np.ptp(arr, axis=0).max())
        assert spread < 1e-4, (
            f"output '{name}' varies by {spread:.6f} across price levels "
            f"{levels} although the move is identical at every one of them"
        )


def test_a_raw_quote_head_is_level_sensitive_so_the_assertion_discriminates():
    """Proof the invariance assertions above are not vacuous.

    Same three dilated causal convolutions, fed straight from the Input the way
    the graph was wired before ``ts_scale_norm``.
    """
    raw_in = tf.keras.layers.Input((WINDOW, 2), name="raw_price_vol")
    x = raw_in
    for rate in (1, 2, 4):
        x = tf.keras.layers.Conv1D(8, 3, padding="causal", dilation_rate=rate)(x)
        x = tf.keras.layers.Activation("swish")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    head = tf.keras.Model(raw_in, tf.keras.layers.Dense(1, activation="sigmoid")(x))

    levels = [1e-06, 1.0, 21.0, 1206.0, 135744.0]
    raw = head(tf.constant(_batch([_window(lvl, 0.002, 0.0) for lvl in levels]))).numpy()
    assert float(raw.max() - raw.min()) > 1e-3, (
        "the raw-quote head was level-invariant, so the tests above prove nothing"
    )

    normed = head(md.PriceVolScaleNorm()(
        tf.constant(_batch([_window(lvl, 0.002, 0.0) for lvl in levels]))
    )).numpy()
    assert float(normed.max() - normed.min()) < 1e-5


def test_the_normalised_model_survives_a_save_and_load_round_trip(tmp_path):
    from trading.pipeline import _custom_objects

    model = _tiny_model()
    path = tmp_path / "roundtrip.keras"
    model.save(path)
    reloaded = tf.keras.models.load_model(path, custom_objects=_custom_objects(), compile=False)
    assert md.PRICE_VOL_NORM_LAYER in {layer.name for layer in reloaded.layers}


# --------------------------------------------------------------------------
# The artifact check that stops a raw-quote model being served or fine-tuned
# --------------------------------------------------------------------------

def test_a_pre_normalisation_artifact_is_recognised_and_not_kept():
    from trading.pipeline import _reads_price_scale

    assert _reads_price_scale(_tiny_model()) is False

    raw_in = tf.keras.layers.Input((WINDOW, 2), name="price_vol_input")
    stale = tf.keras.Model(raw_in, tf.keras.layers.Conv1D(4, 3, padding="causal", name="ts_d1")(raw_in))
    assert _reads_price_scale(stale) is True, (
        "a model with the raw window wired into ts_d1 was accepted as normalised"
    )
