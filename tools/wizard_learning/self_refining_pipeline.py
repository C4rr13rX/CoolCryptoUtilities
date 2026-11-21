"""tools/wizard_learning/self_refining_pipeline.py

Autonomous Self-Refining Training Pipeline for W1z4rD Brain.
Exercises empirical curriculum topics (physics, circuitry, chemistry, 3D modeling),
trains multi-pool Hebbian associative fabric, verifies recall, and enforces
continuous settling/sleep cycles to maintain lean memory on CPU-only devices.
"""
from __future__ import annotations

import datetime
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from .empirical_material import EMPIRICAL_CURRICULUM, get_training_pairs
from .procedural_3d import generate_mesh
from .settling_controller import get_settling_controller

EVENT_LOG_FILE = Path("runtime/c0d3r/training_events.jsonl")


class SelfRefiningPipeline:
    """Orchestrates autonomous curriculum training, verification, and settling."""

    def __init__(self, endpoint: str = "http://127.0.0.1:8090"):
        self.endpoint = endpoint.rstrip("/")
        self.controller = get_settling_controller()
        EVENT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _log_event(self, event_type: str, data: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": event_type,
            **data,
        }
        try:
            with open(EVENT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def train_single_pair(
        self, question: str, answer: str, topic_id: str = "EMPIRICAL", passes: int = 15
    ) -> dict[str, Any]:
        """Train one Q->A association into the multi-pool fabric and verify recall."""
        # 1. Multi-pool association
        payload = json.dumps({
            "src_pool": "in",
            "src": question,
            "tgt_pool": "out",
            "tgt": answer,
            "passes": passes,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.endpoint}/multi_pool/train_pair",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                train_data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {
                "ok": False,
                "error": f"Train failed: {e}",
                "topic_id": topic_id,
                "recalled": False,
                "recalled_text": f"Error: {e}",
                "train_duration_ms": 0.0,
            }

        train_ms = round((time.time() - t0) * 1000.0, 1)

        # 2. Verify recall via /chat
        verify_req = urllib.request.Request(
            f"{self.endpoint}/chat",
            data=json.dumps({"text": question}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        recalled_text = ""
        decoder = ""
        try:
            with urllib.request.urlopen(verify_req, timeout=5) as resp:
                chat_data = json.loads(resp.read().decode("utf-8"))
                recalled_text = chat_data.get("answer") or chat_data.get("reply") or ""
                decoder = chat_data.get("decoder", "unknown")
        except Exception as e:
            recalled_text = f"Verification timeout: {e}"

        recall_ok = bool(answer[:20].lower() in recalled_text.lower() or answer.lower() in recalled_text.lower())

        # 3. Notify settling controller of completed episode
        settle_status = self.controller.notify_episode_trained()

        result = {
            "ok": True,
            "topic_id": topic_id,
            "train_duration_ms": train_ms,
            "recalled": recall_ok,
            "recalled_text": recalled_text[:120],
            "decoder": decoder,
            "settled": settle_status.get("settled", False),
            "memory_mb": settle_status.get("metrics", {}).get("total_working_set_mb"),
        }

        self._log_event("episode_trained", result)
        return result

    def train_3d_mesh_synthesis(self, category: str, part_type: str) -> dict[str, Any]:
        """Train the brain to map procedural 3D model requests to their exact mathematical specifications."""
        mesh_data = generate_mesh(category, part_type)
        question = f"Generate 3d model {category}:{part_type} with continuous uv coordinates"
        answer = (
            f"Procedural mesh '{mesh_data['name']}': {mesh_data['vertex_count']} vertices, "
            f"{mesh_data['triangle_count']} triangles. BufferGeometry attributes: position, normal, uv."
        )
        return self.train_single_pair(question, answer, topic_id=f"3D-{category.upper()}", passes=12)

    def run_curriculum_batch(self, category: str | None = None, max_items: int = 10) -> dict[str, Any]:
        """Run a batch of empirical curriculum items with automated settling."""
        pairs = get_training_pairs()
        if category and category != "all":
            pairs = [
                (item.question, item.answer, item.id)
                for item in EMPIRICAL_CURRICULUM
                if item.category == category
            ]

        results = []
        for q, a, tid in pairs[:max_items]:
            res = self.train_single_pair(q, a, topic_id=tid)
            results.append(res)
            time.sleep(0.05)

        # Also train one 3D procedural mesh pair
        mesh_res = self.train_3d_mesh_synthesis("machine_part", "motor_housing")
        results.append(mesh_res)

        # Force a settle at the end of the batch if needed
        metrics = self.controller.get_resource_metrics()
        settled_at_end = False
        if metrics["needs_settle"] or metrics["episodes_since_settle"] > 0:
            settle_record = self.controller.settle(reason="batch_completion")
            settled_at_end = True

        self._log_event("batch_completed", {
            "items_trained": len(results),
            "settled_at_end": settled_at_end,
            "final_working_set_mb": self.controller.get_resource_metrics()["total_working_set_mb"],
        })

        return {
            "status": "completed",
            "items_trained": len(results),
            "results": results,
            "settled_at_end": settled_at_end,
            "current_metrics": self.controller.get_resource_metrics(),
            "last_reflection": (self.controller.get_history(1) or [{}])[0].get("reflection", ""),
        }
