#!/usr/bin/env python3
import os, math, time, numpy as np, tensorflow as tf
from tensorflow.keras import layers, Model

# Quiet down TF CUDA noise on CPU machines
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------- CONFIG ----------
MODE = os.environ.get("MODE", "price_only")   # "price_only" or "hybrid"
FREEZE_BERT = True
N_SAMPLES = int(os.environ.get("N_SAMPLES", "20000"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
PRICE_STEPS = int(os.environ.get("PRICE_STEPS", "30"))
MAX_TEXT_LEN = int(os.environ.get("MAX_TEXT_LEN", "128"))
# ----------------------------

def build_price_only_model(price_steps=30):
    price_in = layers.Input(shape=(price_steps, 1), name="price_input")
    x = layers.Conv1D(64, 3, activation="relu")(price_in)
    x = layers.MaxPooling1D()(x)
    x = layers.Conv1D(128, 3, activation="relu")(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Flatten()(x)

    time_in = layers.Input(shape=(1,), name="time_input")
    x = layers.Concatenate()([x, time_in])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="linear", name="price_prediction")(x)

    m = Model(inputs=[price_in, time_in], outputs=out)
    m.compile(optimizer="adam", loss="mse")
    return m

def build_hybrid_model(price_steps=30, max_text_len=128):
    from transformers import TFBertModel
    # Preload BERT once
    bert = TFBertModel.from_pretrained("google/mobilebert-uncased")

    # Price branch
    price_in = layers.Input(shape=(price_steps, 1), name="price_input")
    x = layers.Conv1D(64, 3, activation="relu")(price_in)
    x = layers.MaxPooling1D()(x)
    x = layers.Conv1D(128, 3, activation="relu")(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Flatten()(x)

    # Text branch (BERT wants ids + attention mask)
    ids_in = layers.Input(shape=(max_text_len,), dtype="int32", name="input_ids")
    attn_in = layers.Input(shape=(max_text_len,), dtype="int32", name="attention_mask")
    bert_out = bert(input_ids=ids_in, attention_mask=attn_in)
    pooled = bert_out.pooler_output if hasattr(bert_out, "pooler_output") else bert_out[1]
    t = layers.Dense(128, activation="relu")(pooled)
    t = layers.Dropout(0.2)(t)

    # Time branch
    time_in = layers.Input(shape=(1,), name="time_input")

    # Merge
    z = layers.Concatenate()([x, t, time_in])
    z = layers.Dense(128, activation="relu")(z)
    z = layers.Dropout(0.2)(z)
    out = layers.Dense(1, activation="linear", name="price_prediction")(z)

    m = Model(inputs=[price_in, ids_in, attn_in, time_in], outputs=out)

    if FREEZE_BERT:
        for l in m.layers:
            if "tf_bert_model" in getattr(l, "name", ""):
                l.trainable = False
        # (Re)compile after freezing
    m.compile(optimizer="adam", loss="mse")
    return m

# ----- Build the model -----
if MODE == "hybrid":
    model = build_hybrid_model(PRICE_STEPS, MAX_TEXT_LEN)
else:
    model = build_price_only_model(PRICE_STEPS)

# ----- Create a single dummy batch -----
if MODE == "hybrid":
    price_batch = np.random.randn(BATCH_SIZE, PRICE_STEPS, 1).astype("float32")
    ids_batch   = np.random.randint(0, 30522, size=(BATCH_SIZE, MAX_TEXT_LEN), dtype=np.int32)
    attn_batch  = np.ones_like(ids_batch, dtype=np.int32)
    time_batch  = np.random.randn(BATCH_SIZE, 1).astype("float32")
    y_batch     = np.random.randn(BATCH_SIZE, 1).astype("float32")

    @tf.function
    def train_step(p, ids, attn, ti, y):
        with tf.GradientTape() as tape:
            pred = model([p, ids, attn, ti], training=True)
            loss = tf.reduce_mean(tf.keras.losses.mse(y, pred))
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    inputs = (price_batch, ids_batch, attn_batch, time_batch, y_batch)

else:
    price_batch = np.random.randn(BATCH_SIZE, PRICE_STEPS, 1).astype("float32")
    time_batch  = np.random.randn(BATCH_SIZE, 1).astype("float32")
    y_batch     = np.random.randn(BATCH_SIZE, 1).astype("float32")

    @tf.function
    def train_step(p, ti, y):
        with tf.GradientTape() as tape:
            pred = model([p, ti], training=True)
            loss = tf.reduce_mean(tf.keras.losses.mse(y, pred))
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    inputs = (price_batch, time_batch, y_batch)

# ----- Warmup -----
for _ in range(5):
    _ = train_step(*inputs)

# ----- Timed loop -----
ITERS = 30
t0 = time.time()
for _ in range(ITERS):
    _ = train_step(*inputs)
t1 = time.time()

sec_per_step = (t1 - t0) / ITERS
steps_per_epoch = math.ceil(N_SAMPLES / BATCH_SIZE)
sec_per_epoch = sec_per_step * steps_per_epoch

print(f"MODE: {MODE} (FREEZE_BERT={FREEZE_BERT})")
print(f"Seconds per train step: {sec_per_step:.4f}")
print(f"Estimated seconds per epoch (N={N_SAMPLES}, batch={BATCH_SIZE}): {sec_per_epoch:.1f}")
