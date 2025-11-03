from __future__ import annotations

import sys
import time

from trading.pipeline import TrainingPipeline


def main():
    pipeline = TrainingPipeline()
    while True:
        result = pipeline.train_candidate()
        iteration = pipeline.iteration if hasattr(pipeline, "iteration") else None
        if result and result.get("status") == "trained" and result.get("score") is not None:
            print(
                "[trainer] iteration %s completed with score=%.4f"
                % (iteration if iteration is not None else "?", result["score"])
            )
            evaluation = result.get("evaluation") or {}
            if evaluation:
                print(
                    "[trainer] metrics: dir_acc=%.3f ghost_trades=%s pred_margin=%.6f realized_margin=%.6f win_rate=%.3f TP=%d FP=%d TN=%d FN=%d"
                    % (
                        evaluation.get("dir_accuracy", 0.0),
                        evaluation.get("ghost_trades", 0),
                        evaluation.get("ghost_pred_margin", 0.0),
                        evaluation.get("ghost_realized_margin", 0.0),
                        evaluation.get("ghost_win_rate", 0.0),
                        evaluation.get("true_positives", 0),
                        evaluation.get("false_positives", 0),
                        evaluation.get("true_negatives", 0),
                        evaluation.get("false_negatives", 0),
                    )
                )
        elif result and result.get("status") == "skipped":
            print(f"[trainer] iteration {result.get('iteration')} skipped (insufficient data)")
        else:
            print(f"[trainer] iteration {iteration} completed (no score available)")
        sys.stdout.flush()
        time.sleep(30)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[trainer] interrupted; exiting.")
