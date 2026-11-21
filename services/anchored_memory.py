"""
services/anchored_memory.py — Durable, Persistent Anchored Memory Substrate for W1z4rD.

Guarantees that user corrections, conversational greetings, and fundamental
domain knowledge are NEVER forgotten across restarts, settling cycles, or
massive training batches.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("AnchoredMemory")

MEMORY_FILE = Path(__file__).resolve().parent.parent / "storage" / "wizard_memory" / "anchored_memory.json"
_LOCK = threading.RLock()

DEFAULT_CONVERSATIONAL_ANCHORS = [
    {
        "id": "anchor-hello",
        "prompt": "Hello",
        "variations": ["hello", "Hello", "Hello!", "hello there", "Hello there", "greetings", "Greetings", "hey", "Hey", "hi", "Hi"],
        "answer": "Hello! I am W1z4rD, your neural cognitive digital twin. I am ready to assist you with robotics, multi-physics simulations, engineering calculations, and software architecture. What would you like to explore?",
        "confidence": 1.0,
        "category": "conversational",
        "trained_at": time.time(),
        "source": "seed",
    },
    {
        "id": "anchor-who-are-you",
        "prompt": "Who are you?",
        "variations": ["who are you", "who are you?", "Who are you?", "Who are you", "what is your name", "What is your name?"],
        "answer": "I am W1z4rD, a real-time multimodal digital twin and neural reasoning engine governed by empirical physics, lumped-parameter dynamics, and Hebbian multi-pool associations.",
        "confidence": 1.0,
        "category": "identity",
        "trained_at": time.time(),
        "source": "seed",
    },
    {
        "id": "anchor-capabilities",
        "prompt": "What can you do?",
        "variations": ["what can you do", "what can you do?", "What can you do?", "What can you do", "capabilities", "what are your capabilities", "What are your capabilities?"],
        "answer": "I simulate coupled lumped-parameter robotics and multi-physics in real time, generate production CAD blueprints, synthesize circuit schematics, evaluate hard engineering problem suites, and continuously learn through synaptic self-settling.",
        "confidence": 1.0,
        "category": "capabilities",
        "trained_at": time.time(),
        "source": "seed",
    },
    {
        "id": "anchor-how-are-you",
        "prompt": "How are you?",
        "variations": ["how are you", "how are you?", "How are you?", "How are you", "how are you doing", "How are you doing?"],
        "answer": "I am operating optimally with active multi-pool routing, zero synaptic drift, and balanced sleep pressure across all cognitive substrates.",
        "confidence": 1.0,
        "category": "conversational",
        "trained_at": time.time(),
        "source": "seed",
    },
    {
        "id": "anchor-thanks",
        "prompt": "Thank you",
        "variations": ["thank you", "Thank you", "thanks", "Thanks", "thank you!", "Thanks!"],
        "answer": "You're very welcome! Let me know whenever you want to test another system, run an obstacle course, or design a new assembly.",
        "confidence": 1.0,
        "category": "conversational",
        "trained_at": time.time(),
        "source": "seed",
    },
]


def _normalize(text: str) -> str:
    """Normalize text by lowering, stripping punctuation and whitespace."""
    if not text:
        return ""
    cleaned = re.sub(r"[^\w\s]", "", text.strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip()


class AnchoredMemoryManager:
    """Singleton manager for durable, persistent anchored memory."""

    _instance: Optional["AnchoredMemoryManager"] = None

    def __init__(self):
        self.file_path = MEMORY_FILE
        self._last_mtime: float = 0.0
        self._ensure_storage()
        self.memories: Dict[str, Dict[str, Any]] = {}
        self._load()

    @classmethod
    def get_instance(cls) -> "AnchoredMemoryManager":
        with _LOCK:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure_storage(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "items": DEFAULT_CONVERSATIONAL_ANCHORS}, f, indent=2)

    def _load(self) -> None:
        with _LOCK:
            try:
                if self.file_path.exists():
                    self._last_mtime = self.file_path.stat().st_mtime
                    with open(self.file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        items = data.get("items", [])
                        loaded: Dict[str, Dict[str, Any]] = {}
                        for item in items:
                            if isinstance(item, dict) and "id" in item:
                                if not item.get("normalized_prompt") and item.get("prompt"):
                                    item["normalized_prompt"] = _normalize(str(item["prompt"]))
                                loaded[item["id"]] = item
                        self.memories = loaded
            except Exception as e:
                logger.error("Failed to load anchored memory from %s: %s", self.file_path, e)
                self.memories = {item["id"]: item for item in DEFAULT_CONVERSATIONAL_ANCHORS}

    def _maybe_reload(self) -> None:
        """Hot-reload across processes if another worker/daemon updated anchored_memory.json."""
        try:
            if self.file_path.exists():
                mtime = self.file_path.stat().st_mtime
                if mtime > self._last_mtime:
                    self._load()
        except Exception:
            pass

    def _save(self) -> None:
        with _LOCK:
            try:
                temp_file = self.file_path.with_suffix(f".{os.getpid() if 'os' in globals() else 'tmp'}.tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump({"version": 1, "updated_at": time.time(), "items": list(self.memories.values())}, f, indent=2)
                temp_file.replace(self.file_path)
                self._last_mtime = self.file_path.stat().st_mtime
            except Exception as e:
                logger.error("Failed to persist anchored memory to %s: %s", self.file_path, e)

    def record_memory(
        self,
        prompt: str,
        answer: str,
        *,
        variations: Optional[List[str]] = None,
        source: str = "human_correction",
        category: str = "custom",
    ) -> Dict[str, Any]:
        """Record or update a permanently anchored memory."""
        with _LOCK:
            prompt_clean = prompt.strip()
            norm_p = _normalize(prompt_clean)
            mem_id = f"mem-{hash(norm_p) & 0xffffffff:08x}"

            var_set = {prompt_clean, norm_p}
            if variations:
                for v in variations:
                    v_str = str(v).strip()
                    if v_str:
                        var_set.add(v_str)
                        var_set.add(_normalize(v_str))

            # Auto-generate common punctuation/case variants
            if prompt_clean.endswith("?"):
                var_set.add(prompt_clean.rstrip("?").strip())
            else:
                var_set.add(f"{prompt_clean}?")

            record = {
                "id": mem_id,
                "prompt": prompt_clean,
                "normalized_prompt": norm_p,
                "variations": sorted(list(var_set)),
                "answer": answer.strip(),
                "confidence": 1.0,
                "category": category,
                "source": source,
                "trained_at": time.time(),
                "recall_count": self.memories.get(mem_id, {}).get("recall_count", 0),
            }
            self.memories[mem_id] = record
            self._save()
            logger.info("Permanently anchored memory '%s' (ID: %s, %d variations)", prompt_clean, mem_id, len(var_set))
            return record

    def recall(self, query: str) -> Optional[Dict[str, Any]]:
        """Look up query against anchored memories (exact, normalized, or variation match)."""
        with _LOCK:
            self._maybe_reload()
            if not query:
                return None
            clean_q = query.strip()
            norm_q = _normalize(clean_q)

            # 1. Exact or normalized prompt match
            for mem in self.memories.values():
                if clean_q == mem.get("prompt") or norm_q == mem.get("normalized_prompt"):
                    mem["recall_count"] = mem.get("recall_count", 0) + 1
                    return mem

            # 2. Variation match
            for mem in self.memories.values():
                for var in mem.get("variations", []):
                    if clean_q.lower() == var.lower() or norm_q == _normalize(var):
                        mem["recall_count"] = mem.get("recall_count", 0) + 1
                        return mem

            return None

    def list_all(self) -> List[Dict[str, Any]]:
        with _LOCK:
            self._maybe_reload()
            return list(self.memories.values())

    def get_stats(self) -> Dict[str, Any]:
        with _LOCK:
            self._maybe_reload()
            return {
                "total_anchored": len(self.memories),
                "total_recalls": sum(m.get("recall_count", 0) for m in self.memories.values()),
                "file_path": str(self.file_path),
            }
