from __future__ import annotations

import math
import os
import re
from typing import Iterable


class LocalCorrectionVerifier:
    """CPU-only retrieval over prior correction events.

    Uses the MIT-licensed BAAI/bge-small-en-v1.5 embedding model through
    FastEmbed/ONNX Runtime when available. A deterministic lexical scorer is
    retained as a no-download fallback. The model ranks evidence; it never
    decides whether a factual claim is true.
    """

    MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        self._embedder = None
        self._disabled = os.getenv("ATF_LOCAL_VERIFIER", "1").lower() in {"0", "false", "no", "off"}

    def rank(self, query: str, events: list[dict], *, limit: int = 8) -> list[dict]:
        if not query.strip() or not events:
            return events[:limit]
        documents = [self._event_text(item) for item in events]
        scores = self._embedding_scores(query, documents)
        if scores is None:
            scores = [self._lexical_score(query, document) for document in documents]
        ranked = sorted(zip(scores, events), key=lambda item: item[0], reverse=True)
        return [event for score, event in ranked[:limit] if score > 0.05]

    def _embedding_scores(self, query: str, documents: list[str]) -> list[float] | None:
        if self._disabled:
            return None
        try:
            if self._embedder is None:
                from fastembed import TextEmbedding

                self._embedder = TextEmbedding(model_name=self.MODEL)
            vectors = list(self._embedder.embed([query, *documents]))
            query_vector = vectors[0]
            return [self._cosine(query_vector, vector) for vector in vectors[1:]]
        except Exception:
            self._disabled = True
            return None

    @staticmethod
    def _cosine(left: Iterable[float], right: Iterable[float]) -> float:
        dot = left_norm = right_norm = 0.0
        for a, b in zip(left, right):
            fa, fb = float(a), float(b)
            dot += fa * fb
            left_norm += fa * fa
            right_norm += fb * fb
        denominator = math.sqrt(left_norm) * math.sqrt(right_norm)
        return dot / denominator if denominator else 0.0

    @staticmethod
    def _lexical_score(query: str, document: str) -> float:
        left = set(re.findall(r"[a-z0-9_.-]{3,}", query.lower()))
        right = set(re.findall(r"[a-z0-9_.-]{3,}", document.lower()))
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    @staticmethod
    def _event_text(event: dict) -> str:
        return " ".join(str(event.get(key) or "") for key in (
            "classification", "trigger", "failed_output", "correction", "provider", "model",
        ))
