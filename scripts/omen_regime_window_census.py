"""Find 400-bar UP and DOWN held-out windows disjoint from a training window.

Pass 117, Cove, item [f4c0975a]. Node-free: reads the corpus only.
"""
import json
import sys

CORPUS = "data/historical_ohlcv/base/0004_AERO-USDC.json"
HORIZON = 12
TEST = 400
LOOKBACK = 168  # a test anchor reads back this far; the window must not reach into train

bars = json.load(open(CORPUS))
close = [b["close"] for b in bars]
n = len(close)

train_end = int(sys.argv[1]) if len(sys.argv) > 1 else 2600
train_n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
train_start = train_end - train_n

rows = []
# a window is [test_end - TEST, test_end); its earliest anchor reads back LOOKBACK bars,
# so test_end - TEST - LOOKBACK must be >= train_end for full disjointness from training.
first_end = train_end + LOOKBACK + TEST
for test_end in range(first_end, n - HORIZON, 50):
    lo = test_end - TEST
    fwd = [(close[i + HORIZON] - close[i]) / close[i] for i in range(lo, test_end)]
    up = sum(1 for f in fwd if f > 0) / len(fwd)
    mean = sum(fwd) / len(fwd)
    drift = (close[test_end - 1] - close[lo]) / close[lo]
    rows.append((test_end, lo, up, mean, drift))

def show(r):
    test_end, lo, up, mean, drift = r
    print("  test_end %5d  window [%5d,%5d)  up_rate %.4f  mean_fwd %+.5f  drift %+.4f"
          % (test_end, lo, test_end, up, mean, drift))


rows.sort(key=lambda r: r[3])
print("train window [%d, %d)  %d bars; test windows are %d bars, all start >= %d"
      % (train_start, train_end, train_n, TEST, first_end - TEST))
print()
print("MOST DOWN (by mean forward return at h=%d)" % HORIZON)
for r in rows[:5]:
    show(r)
print()
print("MOST UP")
for r in rows[-5:][::-1]:
    show(r)
