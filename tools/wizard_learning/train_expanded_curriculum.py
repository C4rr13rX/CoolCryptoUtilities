"""tools/wizard_learning/train_expanded_curriculum.py

Trains all 16 new empirical raw materials and robotics items into the W1z4rD brain.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from tools.wizard_learning.empirical_material import EMPIRICAL_CURRICULUM
from tools.wizard_learning.self_refining_pipeline import SelfRefiningPipeline

def main():
    pipe = SelfRefiningPipeline()
    # The new items start at index 17 (RAW-01 through ROB-08)
    new_items = EMPIRICAL_CURRICULUM[17:]
    print(f"Training {len(new_items)} new empirical items into W1z4rD brain...")

    successful = 0
    for i, item in enumerate(new_items, 1):
        for attempt in range(3):
            res = pipe.train_single_pair(item.question, item.answer, topic_id=item.id, passes=8)
            if res.get("ok"):
                successful += 1
                recalled = res.get("recalled", False)
                dur = res.get("train_duration_ms", 0.0)
                txt = res.get("recalled_text", "")[:65]
                print(f"[{i}/{len(new_items)}] [{item.id}] ok=True recalled={recalled} ({dur}ms) -> {txt}")
                break
            else:
                print(f"[{i}/{len(new_items)}] [{item.id}] attempt {attempt+1} failed: {res.get('error')}. Retrying...")
                time.sleep(0.5)
        time.sleep(0.05)

    settle_res = pipe.controller.settle(reason="post_raw_and_robotics_curriculum")
    print(f"Settled after batch ({successful}/{len(new_items)} trained). Record: {settle_res.get('record', {}).get('reflection', '')}")

if __name__ == "__main__":
    main()
