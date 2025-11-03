from __future__ import annotations

import sys
import time

from trading.pipeline import TrainingPipeline


def main():
    pipeline = TrainingPipeline()
    iteration = 0
    while True:
        iteration += 1
        print(f"[trainer] starting candidate training iteration {iteration}")
        result = pipeline.train_candidate()
        if result:
            print(f"[trainer] iteration {iteration} completed with score={result['score']:.4f}")
        else:
            print(f"[trainer] iteration {iteration} skipped (insufficient data)")
        sys.stdout.flush()
        time.sleep(30)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[trainer] interrupted; exiting.")
