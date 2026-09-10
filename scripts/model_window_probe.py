"""Probe the DEPLOYED model with windows you control. Read-only.

Answers, repeatably, the question [cdfbf97e] was filed on: is ``price_mu`` a
fractional return, or a saturated function of the price tag?

    python -X utf8 scripts/model_window_probe.py

Must run under the venv that has TensorFlow (the production one):

    D:/Projects/CoolCryptoUtilities/.venv/Scripts/python.exe -X utf8 \
        scripts/model_window_probe.py

Three sections:

  LEVEL   the same shape of move at price levels eight orders of magnitude
          apart. A price-INDEPENDENT output proves the scale-free transform is
          live in the served graph; a price-dependent one proves it is not.
  LIVE    the real last-60 ticks of the busiest symbols, straight out of
          market_stream. This is the number to compare against
          organism_snapshots.prediction.price_mu.
  DIRTY   the same live windows with one row deliberately made foreign. This
          is what reproduces the saturation seen in production.

Measured 2026-09-10 against models/active_model.keras:

  LEVEL   price_mu -0.000620 at level 1e-4 AND at level 1.2e4 -- identical to
          six decimals, so the transform is live and the units are NOT the bug.
  LIVE    price_mu -0.1655 (DRB) to -0.2428 (ALIGN); net_margin tracks it at
          exactly price_mu - 0.0065.
  DIRTY   price_mu -1.5111 (one 100x row), -1.8395 (one foreign-asset row),
          +1.3012 (two assets interleaved) -- which IS the p50 -1.2076 that
          organism_snapshots recorded over 1613 live predictions that day.

The guard for the DIRTY case is trading/data_loader.sanitize_model_price_window
and its test is tests/test_a_foreign_priced_row_cannot_saturate_the_model_window.py.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DEFAULT_DB = os.path.join("storage", "trading_cache.db")
DEFAULT_MODEL = os.path.join("models", "active_model.keras")

#: The order trading/bot.py::_summarise_predictions reads the heads in. The
#: saved graph names its outputs keras_tensor_NNN, which tells a reader
#: nothing, so the names live here beside the code that consumes them.
HEADS = ["exit_conf", "price_mu", "price_log_var", "direction_prob", "net_margin", "net_pnl"]


def _load(model_path: str):
    import tensorflow as tf
    import model_definition  # noqa: F401  registers the custom layers

    model = tf.keras.models.load_model(model_path, compile=False)
    return tf, model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--symbols", type=int, default=6)
    args = ap.parse_args()

    try:
        tf, model = _load(args.model)
    except ModuleNotFoundError as exc:
        print(f"TensorFlow is not importable here ({exc}). Run this under the "
              f"production venv: D:/Projects/CoolCryptoUtilities/.venv/Scripts/python.exe")
        return 2

    order = [i.name.split(":")[0] for i in model.inputs]
    width = [int(i.shape[1]) for i in model.inputs if "price_vol" in i.name][0]
    tech = [int(i.shape[1]) for i in model.inputs if "tech_input" in i.name][0]
    seq = [int(i.shape[1]) for i in model.inputs if "sentiment_seq" in i.name][0]
    has_norm = any(l.__class__.__name__ == "PriceVolScaleNorm" for l in model.layers)

    print(f"model            : {args.model}")
    print(f"window_size      : {width}")
    print(f"PriceVolScaleNorm: {'PRESENT in the served graph' if has_norm else 'ABSENT -- the serving path is raw'}")
    print()

    def run(prices, volumes, label="probe"):
        pv = np.stack(
            [np.asarray(prices, np.float32), np.asarray(volumes, np.float32)], -1
        ).reshape(1, width, 2)
        feed = {
            "price_vol_input": pv,
            "sentiment_seq": np.zeros((1, seq, 1), np.float32),
            "headline_text": tf.constant([[f"{label} price {prices[-1]}"]], tf.string),
            "full_text": tf.constant([[""]], tf.string),
            "tech_input": np.zeros((1, tech), np.float32),
            "hour_input": np.array([[12]], np.int32),
            "dow_input": np.array([[2]], np.int32),
            # The exact two numbers _prepare_inputs feeds on every live tick.
            "gas_fee_input": np.full((1, 1), 0.0015, np.float32),
            "tax_rate_input": np.full((1, 1), 0.005, np.float32),
            "asset_id_input": np.array([[0]], np.int32),
        }
        preds = model.predict([feed[k] for k in order], verbose=0)
        return {h: float(np.asarray(preds[i]).reshape(-1)[0]) for i, h in enumerate(HEADS)}

    def row(label, r):
        print(f"{label:>38} | price_mu {r['price_mu']:>10.6f}  dir_prob {r['direction_prob']:>9.6f}"
              f"  net_margin {r['net_margin']:>11.6f}")

    print("=== LEVEL: +0.2%/step held fixed, only the price tag moves ===")
    for level in (1e-4, 1e-2, 1.0, 21.0, 2469.0, 1.2e4):
        ramp = [level * (1.002 ** i) for i in range(width)]
        row(f"level {level:g}", run(ramp, [0.0] * width))

    conn = sqlite3.connect(args.db)
    now = time.time()
    symbols = [r[0] for r in conn.execute(
        "select symbol,count(*) from market_stream where ts>? group by symbol "
        "having count(*)>=? order by 2 desc limit ?",
        (now - 6 * 3600, width, args.symbols))]

    def window(symbol):
        rows = conn.execute(
            "select price,volume from market_stream where symbol=? order by ts desc limit ?",
            (symbol, width)).fetchall()[::-1]
        if len(rows) < width:
            return None, None
        return [float(r[0] or 0.0) for r in rows], [float(r[1] or 0.0) for r in rows]

    print(f"\n=== LIVE: last {width} ticks per symbol, straight out of market_stream ===")
    live = {}
    for symbol in symbols:
        prices, vols = window(symbol)
        if prices is None:
            continue
        live[symbol] = (prices, vols)
        row(symbol, run(prices, vols, symbol))

    if len(live) >= 2:
        names = list(live)
        a_p, a_v = live[names[0]]
        b_p, _ = live[names[1]]
        print("\n=== DIRTY: the same live windows with one row made foreign ===")
        row(f"{names[0]} clean (control)", run(a_p, a_v, names[0]))
        row(f"{names[0]} one row x100",
            run(a_p[:40] + [a_p[40] * 100.0] + a_p[41:], a_v, names[0]))
        row(f"{names[0]} one {names[1]} row at t=30",
            run(a_p[:30] + [b_p[30]] + a_p[31:], a_v, names[0]))
        row(f"{names[0]}+{names[1]} interleaved",
            run([a_p[i] if i % 2 else b_p[i] for i in range(width)], a_v, "mixed"))

        from trading.data_loader import sanitize_model_price_window
        dirty = a_p[:40] + [a_p[40] * 100.0] + a_p[41:]
        repaired, count = sanitize_model_price_window(dirty)
        print("\n=== REPAIRED: the same dirty window through the guard ===")
        row(f"{names[0]} x100 row, {count} repaired", run(repaired, a_v, names[0]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
