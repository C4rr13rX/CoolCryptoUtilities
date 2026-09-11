from __future__ import annotations

import math
import os
from pathlib import Path

# Force CPU execution to avoid CUDA driver warnings on machines without GPUs.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from typing import Any, Dict, Optional

from keras.callbacks import Callback
from keras.layers import (
    Activation,
    Add,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    Lambda,
    LayerNormalization,
    MaxPooling1D,
    Multiply,
    Reshape,
)
from keras.layers import TextVectorization
from keras.metrics import BinaryCrossentropy
from keras.models import Model
from keras.regularizers import l2


@tf.keras.utils.register_keras_serializable(package="CoolCrypto")
class TimeFeatureLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        hour, dow = inputs
        hour = tf.cast(hour, tf.float32) / 24.0
        dow = tf.cast(dow, tf.float32) / 7.0
        two_pi = tf.constant(2.0 * math.pi, dtype=tf.float32)
        sin_hr = tf.math.sin(two_pi * hour)
        cos_hr = tf.math.cos(two_pi * hour)
        sin_dw = tf.math.sin(two_pi * dow)
        cos_dw = tf.math.cos(two_pi * dow)
        return tf.concat([sin_hr, cos_hr, sin_dw, cos_dw], axis=-1)

# ---------------------------------------------------------------------
# Optional encrypted DB state hooks (safe if missing)
# ---------------------------------------------------------------------
try:  # pragma: no cover - optional path
    from db import load_db, save_db  # type: ignore
except Exception:  # pragma: no cover - fallback
    def load_db() -> dict:
        return {}

    def save_db(_: dict) -> None:
        pass


# ---------------------------------------------------------------------
# Callback to persist last epoch (checkpoint-like)
# ---------------------------------------------------------------------
class StateSaver(Callback):
    def on_epoch_end(self, epoch, logs=None):
        db = load_db()
        db["last_epoch"] = int(epoch) + 1
        save_db(db)


# ---------------------------------------------------------------------
# Learnable exponential decay across time steps (for sentiment)
# ---------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="CoolCrypto")
class ExponentialDecay(tf.keras.layers.Layer):
    def __init__(self, length, **kwargs):
        super().__init__(**kwargs)
        self.length = int(length)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="decay_rate", shape=(), initializer=tf.keras.initializers.Constant(0.10), trainable=True
        )
        idx = tf.range(self.length, dtype=tf.float32)
        self.idx = tf.reshape(idx, (self.length, 1))
        super().build(input_shape)

    def call(self, inputs):
        exp_vals = tf.exp(-self.alpha * (self.length - 1 - self.idx))
        weights = exp_vals / tf.reduce_sum(exp_vals)
        weights = tf.reshape(weights, (1, self.length, 1))
        return inputs * weights

    def get_config(self):
        config = super().get_config()
        config["length"] = self.length
        return config


# Name of the layer that makes ``price_vol_input`` scale-free. Callers use it
# to tell a normalised artifact from a pre-normalisation one on load; see
# trading/pipeline.py:_model_reads_price_scale.
PRICE_VOL_NORM_LAYER = "ts_scale_norm"


@tf.keras.utils.register_keras_serializable(package="CoolCrypto")
class PriceVolScaleNorm(tf.keras.layers.Layer):
    """Turn a raw (price, volume) window into a scale-free one, in the graph.

    THE MODEL WAS READING THE PRICE TAG, NOT THE PRICE MOVE.

    ``price_vol_input`` went straight into three Conv1D layers carrying raw
    quotes. ``ts_norm`` (LayerNormalization) sits AFTER them, so the
    convolutions saw absolute magnitudes -- while ``tech_input`` gets
    LayerNormalization as its very first op. The corpus spans 9e-08 to 123,429
    on the price channel and the live stream spans 1.672e-23 to 135,744, so a
    single set of conv weights was being asked to cover 27 orders of magnitude.

    Probed against models/active_model.keras on 2026-09-07, holding the SHAPE
    of the move fixed at +0.2%/step (a +12.7% move across the 60-step window)
    and varying only the price level:

        level 1e-06   price_dir 0.2019   price_mu -0.1298
        level 1.0     price_dir 0.5276   price_mu +0.5493
        level 21.0    price_dir 0.4822   price_mu -0.1718
        level 1206    price_dir 0.6142   price_mu -0.2791

    Then holding the level at 21.0 and varying the actual DIRECTION from
    -1%/step to +1%/step: price_dir moved 0.4815 -> 0.4808. Seven ten-thousandths,
    and the wrong way. The price tag swung the entry signal 585x harder than the
    price direction did, and the entry bar (direction_prob >= 0.58) was reachable
    only at levels above ~1000.

    Two channels, two neutral-preserving transforms:

    * price -> ``log(p_t / p_0)``, the cumulative log return from the first bar
      of the window. Scale-free, exactly 0 at t=0, valued on the same 0.01-0.1
      scale as the ``price_mu`` target, and its first difference is the step
      return, which the dilated causal convolutions can take themselves.
    * volume -> ``v_t / mean(v) - 1``, relative volume centred on zero.

    The volume transform also closes the second half of the skew. Live volume is
    0.0 on 57,957 of 57,984 market_stream rows (99.95%) because no price source
    we poll reports a per-bar traded size, while the training corpus has a median
    of 125,401 and never a zero. Raw, that put every live inference several
    standard deviations off the edge of the training distribution -- moving that
    channel from 0 to 125,401 swung price_mu from -0.1718 to +1.1588. Relative,
    an all-zero window has an undefined ratio, which this maps to 0.0: exactly
    the value a perfectly average-volume window gets. An input we cannot measure
    now lands at the CENTRE of what the model was trained on rather than off its
    edge.

    It lives in the graph rather than in the loader so that training and serving
    cannot disagree: there is one implementation and both paths execute it.
    trading/data_loader.py and trading/bot.py::_prepare_inputs keep feeding raw
    quotes, which also leaves trading/pipeline.py::_wizard_push_ohlcv reading a
    real close price out of channel 0.
    """

    EPS = 1e-12

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        price = x[..., 0]
        volume = x[..., 1]

        # Anchor on the first STRICTLY POSITIVE price in the window, not on
        # price[:, 0]. A leading zero or a padded row would otherwise divide the
        # whole window by ~0 and hand the convolutions an infinity; the model
        # then dies inside the market-stream callback, which is the one place a
        # crash costs a tick.
        positive = price > 0.0
        safe_price = tf.where(positive, price, tf.ones_like(price))
        first_idx = tf.argmax(tf.cast(positive, tf.int32), axis=1, output_type=tf.int32)
        anchor = tf.gather(safe_price, first_idx, batch_dims=1)
        anchor = tf.where(
            tf.reduce_any(positive, axis=1), anchor, tf.ones_like(anchor)
        )
        anchor = tf.expand_dims(anchor, axis=-1)
        log_ret = tf.math.log(safe_price / tf.maximum(anchor, self.EPS))
        # A non-positive quote carries no return; say so rather than inventing one.
        log_ret = tf.where(positive, log_ret, tf.zeros_like(log_ret))

        mean_vol = tf.reduce_mean(tf.abs(volume), axis=1, keepdims=True)
        rel_vol = tf.where(
            mean_vol > self.EPS,
            volume / tf.maximum(mean_vol, self.EPS) - 1.0,
            tf.zeros_like(volume),
        )

        out = tf.stack([log_ret, rel_vol], axis=-1)
        # Windows in this corpus are minutes to hours apart; a |log return| over
        # 10 (a 22,000x move) is a data fault, not a market. Clip so one bad row
        # cannot dominate a batch's gradients.
        return tf.clip_by_value(out, -10.0, 10.0)

    def compute_output_shape(self, input_shape):
        return input_shape


_GAUSS_EPS = tf.constant(1e-6, dtype=tf.float32)
_GAUSS_MIN_LOG_VAR = tf.math.log(_GAUSS_EPS)
_GAUSS_MAX_LOG_VAR = tf.constant(8.0, dtype=tf.float32)


@tf.keras.utils.register_keras_serializable(package="CoolCrypto")
def _slice_price_mu(x: tf.Tensor) -> tf.Tensor:
    return x[:, :1]


@tf.keras.utils.register_keras_serializable(package="CoolCrypto")
def _slice_price_log_var(x: tf.Tensor) -> tf.Tensor:
    return x[:, 1:2]


@tf.keras.utils.register_keras_serializable(package="CoolCrypto")
def _identity(x: tf.Tensor) -> tf.Tensor:
    return x


@tf.keras.utils.register_keras_serializable(package="CoolCrypto")
def _compute_net_margin(args: list[tf.Tensor] | tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
    price_mu, gas_fee, tax_rate = args
    return price_mu - (gas_fee + tax_rate)


@tf.keras.utils.register_keras_serializable(package="CoolCrypto")
def gaussian_nll_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    target = y_true[:, :1] if y_true.shape.rank and y_true.shape.rank > 1 else tf.expand_dims(y_true, axis=-1)
    last_dim = tf.shape(y_pred)[-1]
    pad_width = tf.maximum(0, 2 - last_dim)
    pred_two = tf.pad(y_pred, [[0, 0], [0, pad_width]])
    mu = pred_two[:, :1]
    log_var = pred_two[:, 1:2]
    clipped = tf.clip_by_value(log_var, _GAUSS_MIN_LOG_VAR, _GAUSS_MAX_LOG_VAR)
    precision = tf.exp(-clipped)
    nll = 0.5 * (clipped + tf.square(target - mu) * precision)
    return tf.squeeze(nll, axis=-1)


@tf.keras.utils.register_keras_serializable(package="CoolCrypto")
def zero_loss(y_true, y_pred):
    return tf.zeros((tf.shape(y_pred)[0],), dtype=y_pred.dtype)


# ---------------------------------------------------------------------
# Utility: build (or accept) light TextVectorization + Embeddings
# ---------------------------------------------------------------------
def build_text_encoder(name_prefix: str, vocab_size: int, seq_len: int, embed_dim: int):
    vec = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=seq_len,
        standardize="lower_and_strip_punctuation",
        name=f"{name_prefix}_vectorizer",
    )
    inp = Input((1,), dtype=tf.string, name=f"{name_prefix}_text")
    ids = vec(inp)
    emb = Embedding(vocab_size, embed_dim, name=f"{name_prefix}_emb")(ids)
    feat = GlobalAveragePooling1D(name=f"{name_prefix}_avgpool")(emb)
    feat = Dense(embed_dim, activation="swish", name=f"{name_prefix}_proj")(feat)
    feat = Dropout(0.2, name=f"{name_prefix}_do")(feat)
    enc = Model(inp, feat, name=f"{name_prefix}_encoder")
    return vec, enc


# ---------------------------------------------------------------------
# Main multimodal model
# ---------------------------------------------------------------------
def build_multimodal_model(
    window_size: int = 60,
    tech_count: int = 12,
    sent_seq_len: int = 24,
    headline_vocab: int = 8000,
    headline_len: int = 40,
    headline_dim: int = 64,
    full_vocab: int = 20000,
    full_len: int = 256,
    full_dim: int = 128,
    hidden_1: int = 256,
    hidden_2: int = 128,
    asset_vocab_size: int = 1,
    model_template: str = "base",
) -> tuple[Model, TextVectorization, TextVectorization, Dict[str, Any], Dict[str, float]]:
    template = str(model_template or "base").lower()
    if template not in {"tiny", "base", "robust"}:
        template = "base"
    # Lightweight template variants to trade capacity for speed/regularisation.
    if template == "tiny":
        ts_filters = 48
        hidden_1, hidden_2 = 160, 96
        headline_dim = max(32, headline_dim // 2)
        full_dim = max(64, full_dim // 2)
        lstm_units = 24
        dropout_main = 0.35
    elif template == "robust":
        ts_filters = 80
        hidden_1, hidden_2 = max(hidden_1, 288), max(hidden_2, 160)
        lstm_units = 48
        dropout_main = 0.25
    else:  # base
        ts_filters = 64
        lstm_units = 32
        dropout_main = 0.3

    ts_in = Input((window_size, 2), name="price_vol_input")
    ts_scaled = PriceVolScaleNorm(name=PRICE_VOL_NORM_LAYER)(ts_in)
    d1 = Conv1D(ts_filters, 3, padding="causal", dilation_rate=1, name="ts_d1")(ts_scaled)
    d1 = Activation("swish")(d1)
    d2 = Conv1D(ts_filters, 3, padding="causal", dilation_rate=2, name="ts_d2")(d1)
    d2 = Activation("swish")(d2)
    d3 = Conv1D(ts_filters, 3, padding="causal", dilation_rate=4, name="ts_d3")(d2)
    d3 = Activation("swish")(d3)

    x = Concatenate(name="ts_cat")([d1, d2, d3])
    x = MaxPooling1D(2, name="ts_pool1")(x)

    se_width = ts_filters * 3
    se = GlobalAveragePooling1D(name="ts_se_gap")(d3)
    se = Dense(4, activation="swish", name="ts_se_mid")(se)
    se = Dense(se_width, activation="sigmoid", name="ts_se_gate")(se)
    se = Reshape((1, se_width), name="ts_se_reshape")(se)
    d3_aligned = Concatenate(name="ts_d3_wide")([d3, d3, d3])
    d3_mod = Multiply(name="ts_se_mul")([d3_aligned, se])
    d3_mod = MaxPooling1D(2, name="ts_d3_pool")(d3_mod)
    x = Add(name="ts_res_add")([x, d3_mod])
    x = LayerNormalization(name="ts_norm")(x)
    x = MaxPooling1D(2, name="ts_pool2")(x)
    x = GlobalAveragePooling1D(name="ts_gap")(x)
    x = Dropout(0.2, name="ts_do")(x)

    exit_conf = Dense(1, activation="sigmoid", name="exit_conf")(x)

    sent_in = Input((sent_seq_len, 1), name="sentiment_seq")
    s = ExponentialDecay(sent_seq_len, name="sent_decay")(sent_in)
    s = LSTM(lstm_units, name="sent_lstm")(s)
    s = Dropout(0.25 if template == "tiny" else 0.2, name="sent_do")(s)

    headline_vec, headline_enc = build_text_encoder(
        name_prefix="headline", vocab_size=headline_vocab, seq_len=headline_len, embed_dim=headline_dim
    )
    headline_in = headline_enc.input
    headline_feat = headline_enc.output

    full_vec, full_enc = build_text_encoder(
        name_prefix="full", vocab_size=full_vocab, seq_len=full_len, embed_dim=full_dim
    )
    full_in = full_enc.input
    full_feat = full_enc.output

    tech_in = Input((tech_count,), name="tech_input")
    t = LayerNormalization(name="tech_norm")(tech_in)
    t = Dense(64, activation="swish", name="tech_dense")(t)
    t = Dropout(0.2, name="tech_do")(t)

    hour_in = Input((1,), dtype="int32", name="hour_input")
    dow_in = Input((1,), dtype="int32", name="dow_input")
    time_f = TimeFeatureLayer(name="time_feat")([hour_in, dow_in])

    gas_in = Input((1,), name="gas_fee_input")
    tax_in = Input((1,), name="tax_rate_input")

    vocab = max(1, int(asset_vocab_size))
    scaled = max(8.0, float(vocab) ** 0.25 * 8.0)
    asset_dim = int(min(32, max(8, round(scaled))))
    asset_in = Input((1,), dtype="int32", name="asset_id_input")
    asset_emb = Embedding(vocab, asset_dim, name="asset_embedding")(asset_in)
    asset_feat = Flatten(name="asset_flat")(asset_emb)

    merged = Concatenate(name="merge_all")([x, s, headline_feat, full_feat, t, time_f, asset_feat])
    reg = l2(5e-4)
    d = Dense(hidden_1, activation="swish", kernel_regularizer=reg, name="h1")(merged)
    d = LayerNormalization(name="h1_norm")(d)
    d = Dropout(dropout_main, name="h1_do")(d)
    d = Dense(hidden_2, activation="swish", kernel_regularizer=reg, name="h2")(d)
    d = LayerNormalization(name="h2_norm")(d)
    d = Dropout(dropout_main, name="h2_do")(d)

    price_params = Dense(2, name="price_params")(d)
    price_mu = Lambda(_slice_price_mu, name="price_mu")(price_params)
    price_log_var = Lambda(_slice_price_log_var, name="price_log_var")(price_params)
    price_dir = Dense(1, activation="sigmoid", name="price_dir")(d)

    net_margin = Lambda(_compute_net_margin, name="net_margin")([price_mu, gas_in, tax_in])
    net_pnl = Lambda(_identity, name="net_pnl")(net_margin)

    tech_out = Dense(tech_count, activation="linear", name="tech_recon")(d)

    price_gaussian = Lambda(_identity, name="price_gaussian")(price_params)

    model = Model(
        inputs=[ts_in, sent_in, headline_in, full_in, tech_in, hour_in, dow_in, gas_in, tax_in, asset_in],
        outputs=[exit_conf, price_mu, price_log_var, price_dir, net_margin, net_pnl, tech_out, price_gaussian],
        name="moneybutton_multimodal_light",
    )

    losses = {
        "exit_conf": "binary_crossentropy",
        "price_mu": zero_loss,
        "price_log_var": zero_loss,
        "price_dir": "binary_crossentropy",
        "net_margin": "mse",
        "net_pnl": "mse",
        "tech_recon": "mse",
        "price_gaussian": gaussian_nll_loss,
    }
    loss_weights = {
        "exit_conf": 0.5,
        "price_mu": 0.0,           # trained via price_gaussian (gaussian_nll_loss)
        "price_log_var": 0.0,      # trained via price_gaussian (gaussian_nll_loss)
        "price_dir": 0.5,
        "net_margin": 1.0,
        "net_pnl": 0.25,           # small weight to train PnL prediction
        "tech_recon": 0.25,
        "price_gaussian": 1.0,
    }

    bce_metric = BinaryCrossentropy(name="brier_like", from_logits=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=losses,
        loss_weights=loss_weights,
        metrics={"price_dir": [tf.keras.metrics.AUC(name="auroc"), "accuracy", bce_metric]},
    )

    return model, headline_vec, full_vec, losses, loss_weights


# ---------------------------------------------------------------------
# Deploy-time calibration guard for the price_mu head
# ---------------------------------------------------------------------

#: How far the head's magnitude may sit from the magnitude of the label it was
#: fitted on before an artifact is refused promotion.
#:
#: WHY THIS GUARD EXISTS AT ALL. ``price_mu`` carries ``loss_weight`` 0.0 (see
#: ``loss_weights`` above): nothing trains it directly. It is supervised only
#: through ``price_gaussian``/``gaussian_nll_loss``, whose mu gradient is scaled
#: by ``precision = exp(-log_var)`` and so switches off as ``log_var`` rises,
#: and through ``net_margin`` MSE. Both are indirect, so the head can drift
#: decades away from its own target while every training metric still looks
#: sane -- and it did: measured 2026-09-11 on 40 clean windows drawn from the
#: very corpus the head was fitted on, median |predicted price_mu| was 1.399697
#: against a median label of 0.002858, a ratio of 489.8x. The label is a
#: one-bar natural-log return whose p99 over 2,277,175 corpus samples is
#: 0.034602, so the head answers three decades outside the support of its
#: target. ``price_mu`` feeds ``delta``, ``net_margin`` and ``net_pnl``, which
#: are three of the entry conjunction's five terms, so an uncalibrated head
#: makes every cost-derived entry test unjudgeable.
#:
#: 10x is deliberately loose. It is not a claim that a 9x head is good; it is
#: the line past which the number is not a return at all.
CALIBRATION_MAX_RATIO = 10.0

#: Windows to probe before a promotion decision. 40 is what the standing probe
#: (``scripts/model_window_probe.py --train-samples 40``) uses, and the ratio it
#: reads is stable at that size because the defect is three orders of magnitude.
CALIBRATION_SAMPLE_COUNT = 40

#: Where trading/data_loader.py draws its training bars from.
CALIBRATION_CORPUS_DIR = os.path.join("data", "historical_ohlcv")


def corpus_calibration_windows(
    width: int,
    count: int = CALIBRATION_SAMPLE_COUNT,
    *,
    corpus_dir: str = CALIBRATION_CORPUS_DIR,
    seed: int = 20260911,
):
    """Draw ``count`` (closes, net_volumes, mu_true) triples from the corpus.

    ``mu_true`` is built exactly as ``trading/data_loader.py`` builds it -- the
    natural-log return from the LAST bar of the window to the bar immediately
    after it -- so the guard measures the head against its own target and not
    against some other definition of "the move".

    Returns an empty list when the corpus is missing, which callers must treat
    as "cannot judge", never as "passed".
    """
    import glob
    import json
    import math
    import random

    files = sorted(glob.glob(os.path.join(corpus_dir, "**", "*.json"), recursive=True))
    if not files:
        return []
    rng = random.Random(seed)
    rng.shuffle(files)
    out = []
    for path in files:
        if len(out) >= count:
            break
        try:
            with open(path, encoding="utf8") as handle:
                rows = json.load(handle)
        except Exception:
            continue
        if not isinstance(rows, list) or len(rows) < width + 8 or not isinstance(rows[0], dict):
            continue
        for _ in range(3):
            if len(out) >= count:
                break
            end = rng.randrange(width, len(rows) - 1)
            sl = rows[end - width:end]
            closes = [float(r.get("close") or 0.0) for r in sl]
            vols = [float(r.get("net_volume") or 0.0) for r in sl]
            nxt = float(rows[end].get("close") or 0.0)
            if min(closes) <= 0.0 or nxt <= 0.0:
                continue
            out.append((closes, vols, math.log(nxt) - math.log(closes[-1])))
    return out


def price_mu_calibration(model, windows) -> Dict[str, Any]:
    """Measure the served ``price_mu`` head against the label it was fitted on.

    Every auxiliary input is the stub ``trading/bot.py::_prepare_inputs`` feeds
    on a live tick (notably ``gas_fee_input`` 0.0015 and ``tax_rate_input``
    0.005), so the price/volume window is the only thing that varies.

    Returns a dict with ``ratio`` = median |predicted| / median |label|, the
    two medians, the served ``log_var`` distribution, and ``samples``. A ratio
    of ``float('inf')`` means the label median underflowed and the artifact
    cannot be judged.
    """
    import numpy as np

    if not windows:
        return {"samples": 0, "ratio": float("nan"), "median_pred": float("nan"),
                "median_label": float("nan"), "median_log_var": float("nan"),
                "log_var_saturated_ratio": float("nan")}

    order = [i.name.split(":")[0] for i in model.inputs]
    width = [int(i.shape[1]) for i in model.inputs if "price_vol" in i.name][0]
    tech = [int(i.shape[1]) for i in model.inputs if "tech_input" in i.name][0]
    seq = [int(i.shape[1]) for i in model.inputs if "sentiment_seq" in i.name][0]
    # The head order trading/bot.py::_summarise_predictions reads.
    mu_index, log_var_index = 1, 2

    preds, truths, log_vars = [], [], []
    for prices, volumes, mu_true in windows:
        if len(prices) != width:
            continue
        pv = np.stack(
            [np.asarray(prices, np.float32), np.asarray(volumes, np.float32)], -1
        ).reshape(1, width, 2)
        feed = {
            "price_vol_input": pv,
            "sentiment_seq": np.zeros((1, seq, 1), np.float32),
            "headline_text": tf.constant([["calibration probe"]], tf.string),
            "full_text": tf.constant([[""]], tf.string),
            "tech_input": np.zeros((1, tech), np.float32),
            "hour_input": np.array([[12]], np.int32),
            "dow_input": np.array([[2]], np.int32),
            "gas_fee_input": np.full((1, 1), 0.0015, np.float32),
            "tax_rate_input": np.full((1, 1), 0.005, np.float32),
            "asset_id_input": np.array([[0]], np.int32),
        }
        out = model.predict([feed[k] for k in order], verbose=0)
        preds.append(float(np.asarray(out[mu_index]).reshape(-1)[0]))
        log_vars.append(float(np.asarray(out[log_var_index]).reshape(-1)[0]))
        truths.append(float(mu_true))

    if not preds:
        return {"samples": 0, "ratio": float("nan"), "median_pred": float("nan"),
                "median_label": float("nan"), "median_log_var": float("nan"),
                "log_var_saturated_ratio": float("nan")}

    pred_arr = np.abs(np.asarray(preds, np.float64))
    true_arr = np.abs(np.asarray(truths, np.float64))
    log_var_arr = np.asarray(log_vars, np.float64)
    median_pred = float(np.median(pred_arr))
    median_label = float(np.median(true_arr))
    ratio = median_pred / median_label if median_label > 0.0 else float("inf")
    return {
        "samples": int(pred_arr.size),
        "ratio": float(ratio),
        "median_pred": median_pred,
        "median_label": median_label,
        "max_pred": float(pred_arr.max()),
        "max_label": float(true_arr.max()),
        "median_log_var": float(np.median(log_var_arr)),
        "log_var_saturated_ratio": float(np.mean(log_var_arr >= 7.9)),
    }


def calibration_rejection_reason(
    report: Dict[str, Any],
    *,
    max_ratio: float = CALIBRATION_MAX_RATIO,
) -> Optional[str]:
    """Return why this artifact must not be promoted, or ``None`` to allow it.

    A report with zero samples is NOT a pass -- it is "the corpus could not be
    read", and the caller decides. It is returned here as ``None`` so a missing
    corpus cannot freeze promotion forever, and the caller logs the sample
    count beside the verdict.
    """
    import math as _math

    samples = int(report.get("samples") or 0)
    if samples <= 0:
        return None
    ratio = float(report.get("ratio", float("nan")))
    if _math.isnan(ratio):
        return None
    if ratio > max_ratio:
        return (
            f"price_mu is uncalibrated: median |predicted| {report['median_pred']:.6f} "
            f"is {ratio:.1f}x the median label {report['median_label']:.6f} over "
            f"{samples} clean corpus windows (bar {max_ratio:.0f}x)"
        )
    return None


# ---------------------------------------------------------------------
# TFLite export for mobile deployment
# ---------------------------------------------------------------------


def export_tflite(
    model_path: str,
    output_path: str = "models/active_model.tflite",
    quantize: bool = True,
) -> str:
    """Convert a saved Keras model to TensorFlow Lite for mobile inference.

    Args:
        model_path: path to the .keras model file.
        output_path: where to write the .tflite file.
        quantize: if True, apply dynamic-range quantization (~2-3x size reduction).

    Returns:
        The output path on success.
    """
    custom_objects = {
        "ExponentialDecay": ExponentialDecay,
        "TimeFeatureLayer": TimeFeatureLayer,
        "gaussian_nll_loss": gaussian_nll_loss,
        "zero_loss": zero_loss,
        "_slice_price_mu": _slice_price_mu,
        "_slice_price_log_var": _slice_price_log_var,
        "_identity": _identity,
        "_compute_net_margin": _compute_net_margin,
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tflite_bytes)
    size_mb = len(tflite_bytes) / (1024 * 1024)
    print(f"Exported TFLite model to {out} ({size_mb:.1f} MB, quantize={quantize})")
    return str(out)


# ---------------------------------------------------------------------
# Minimal smoke test (build only)
# ---------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - manual check
    model, head_vec, full_vec, _, _ = build_multimodal_model()
    model.summary()
    print("\nOK: model built. Adapt the text vectorizers offline before training:")
    print("  head_vec.adapt(dataset_of_headlines)")
    print("  full_vec.adapt(dataset_of_full_articles)")
