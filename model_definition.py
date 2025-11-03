from __future__ import annotations

import math
import os
import tensorflow as tf
from typing import Any, Dict

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
) -> tuple[Model, TextVectorization, TextVectorization, Dict[str, Any], Dict[str, float]]:
    ts_in = Input((window_size, 2), name="price_vol_input")
    d1 = Conv1D(64, 3, padding="causal", dilation_rate=1, name="ts_d1")(ts_in)
    d1 = Activation("swish")(d1)
    d2 = Conv1D(64, 3, padding="causal", dilation_rate=2, name="ts_d2")(d1)
    d2 = Activation("swish")(d2)
    d3 = Conv1D(64, 3, padding="causal", dilation_rate=4, name="ts_d3")(d2)
    d3 = Activation("swish")(d3)

    x = Concatenate(name="ts_cat")([d1, d2, d3])
    x = MaxPooling1D(2, name="ts_pool1")(x)

    se = GlobalAveragePooling1D(name="ts_se_gap")(d3)
    se = Dense(4, activation="swish", name="ts_se_mid")(se)
    se = Dense(192, activation="sigmoid", name="ts_se_gate")(se)
    se = Reshape((1, 192), name="ts_se_reshape")(se)
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
    s = LSTM(32, name="sent_lstm")(s)
    s = Dropout(0.2, name="sent_do")(s)

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
    reg = l2(1e-5)
    d = Dense(hidden_1, activation="swish", kernel_regularizer=reg, name="h1")(merged)
    d = LayerNormalization(name="h1_norm")(d)
    d = Dropout(0.3, name="h1_do")(d)
    d = Dense(hidden_2, activation="swish", kernel_regularizer=reg, name="h2")(d)
    d = LayerNormalization(name="h2_norm")(d)
    d = Dropout(0.3, name="h2_do")(d)

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
        "price_mu": 0.0,
        "price_log_var": 0.0,
        "price_dir": 0.5,
        "net_margin": 1.0,
        "net_pnl": 0.0,
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
# Minimal smoke test (build only)
# ---------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - manual check
    model, head_vec, full_vec, _, _ = build_multimodal_model()
    model.summary()
    print("\nOK: model built. Adapt the text vectorizers offline before training:")
    print("  head_vec.adapt(dataset_of_headlines)")
    print("  full_vec.adapt(dataset_of_full_articles)")
