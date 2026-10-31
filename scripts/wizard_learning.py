"""Manage the foundation-first video queue and RuView bridge.

Metadata intake never learns. 'teach-foundation' is the only bootstrap writer.
Assessments use inference only. A RuView prediction is never its own outcome label.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.wizard_learning.brain import BrainClient, assess_foundation, teach_foundation
from tools.wizard_learning.curriculum import TEACHING_LESSONS, MANIFEST, SUITES, prompt_for
from tools.wizard_learning.ruview import ingest_ruview, poll_ruview
from tools.wizard_learning.seed import seed_videos
from tools.wizard_learning.store import LearningStore, brain_key
from tools.wizard_learning.archive import discover_stokes
from tools.wizard_learning.media import review_video, confirm_video_review, train_video_batch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "seed-videos", "discover-stokes", "review-video", "confirm-review", "train-video", "export-foundation", "assess", "teach-foundation", "ruview"))
    parser.add_argument("--db", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8090")
    parser.add_argument("--chat-path", choices=("/brain/chat", "/chat"), default="/brain/chat")
    parser.add_argument("--brain-id", default="environment-default")
    parser.add_argument("--checkpoint", default=os.getenv("WIZARD_LEARNING_CHECKPOINT_ID", ""))
    parser.add_argument("--timeout", type=float, default=8, help="Per-node-request timeout; use a longer bound for an existing private relay")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--input-pool", type=int, default=1)
    parser.add_argument("--outcome-pool", type=int, default=4)
    parser.add_argument("--ruview-endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--ruview-pool", type=int)
    parser.add_argument("--samples", type=int, default=1, help="Bounded RuView polling, at most 120 samples, 2 fps")
    parser.add_argument("--item", help="Queue item ID for video review")
    parser.add_argument("--review", type=Path, help="Reviewed segment annotation JSON")
    args = parser.parse_args()
    store = LearningStore(args.db)
    profile = {"endpoint": args.endpoint, "chat_path": args.chat_path, "id": args.brain_id}
    if not 1 <= args.timeout <= 120: parser.error("--timeout must be 1..120 seconds")
    client = BrainClient(profile, timeout=args.timeout)
    if args.case_limit is not None and args.case_limit < 1: parser.error("--case-limit must be positive")
    if args.command == "status":
        result = {"gate": store.gate(brain_key(profile), args.checkpoint), "queue": store.items(brain_key(profile), args.checkpoint)}
    elif args.command == "seed-videos": result = seed_videos(store, ROOT / "data/wizard_learning/license_evidence")
    elif args.command == "discover-stokes": result = discover_stokes(store)
    elif args.command == "review-video":
        if not args.item: parser.error("--item is required")
        result = review_video(store, args.item, ROOT / "data/wizard_learning/video_review")
    elif args.command == "confirm-review":
        if not args.item or not args.review: parser.error("--item and --review are required")
        confirm_video_review(store, args.item, json.loads(args.review.read_text(encoding="utf-8")))
        result = {"review_saved": True, "item": args.item, "learning": False}
    elif args.command == "train-video":
        if not args.item: parser.error("--item is required")
        result = train_video_batch(store, args.item, client, args.checkpoint, context_pool=args.input_pool, outcome_pool=args.outcome_pool)
    elif args.command == "export-foundation":
        if not args.output: parser.error("--output is required")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(json.dumps({"input": prompt_for(case), "outcome": case["expected"], "kind": "authored-foundation-lesson"}) for case in TEACHING_LESSONS) + "\n", encoding="utf-8")
        result = {"lessons": len(TEACHING_LESSONS), "held_out_cases": sum(map(len, SUITES.values())), "output": str(args.output), "manifest": MANIFEST}
    elif args.command == "assess":
        result = assess_foundation(client, store, args.checkpoint, case_limit=args.case_limit)
    elif args.command == "teach-foundation":
        # A foundation writer does not unlock anything; all independent tests still follow.
        result = teach_foundation(client, store, input_pool=args.input_pool, outcome_pool=args.outcome_pool)
    else:
        if not 1 <= args.samples <= 120: parser.error("--samples must be 1..120")
        result = []
        for index in range(args.samples):
            frame = poll_ruview(args.ruview_endpoint)
            result.append(ingest_ruview(frame, client, store, args.checkpoint, pool_id=args.ruview_pool, target_pool=args.outcome_pool))
            if index + 1 < args.samples: time.sleep(.5)
    if args.output and args.command != "export-foundation":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (ValueError, PermissionError, OSError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        raise SystemExit(2)
