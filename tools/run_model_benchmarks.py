#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.c0d3r_session import (
    BedrockClient,
    BedrockModelCatalog,
    _build_bedrock_payload,
    _extract_text,
    c0d3r_default_settings,
)


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("questions") or [])


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _parse_number(text: str) -> Optional[float]:
    matches = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", text.lower())
    if not matches:
        return None
    try:
        return float(matches[0])
    except Exception:
        return None


def _match_answer(resp: str, question: Dict[str, Any]) -> bool:
    match_type = question.get("match") or "contains"
    answer = question.get("answer")
    if match_type == "number":
        got = _parse_number(resp)
        if got is None:
            return False
        tol = float(question.get("tolerance", 0.0))
        return abs(got - float(answer)) <= tol
    if match_type == "set":
        expected = {s.strip() for s in str(answer).split(",") if s.strip()}
        found = {s.strip() for s in re.split(r"[,\s]+", resp) if s.strip()}
        return expected.issubset(found)
    return _normalize(str(answer)) in _normalize(resp)


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def _load_candidate_models(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    models = payload.get("models") if isinstance(payload, dict) else payload
    if not models:
        return []
    return [str(m) for m in models if str(m).strip()]


def _score_models(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        results,
        key=lambda r: (-r["accuracy"], r["avg_latency_s"], r["avg_tokens_in"]),
    )


def _recommend_routes(sorted_results: List[Dict[str, Any]]) -> Dict[str, str]:
    if not sorted_results:
        return {}
    top = sorted_results[0]["model_id"]
    second = sorted_results[1]["model_id"] if len(sorted_results) > 1 else top
    return {
        "planner": top,
        "synthesizer": top,
        "executor": top,
        "reviewer": second,
        "refiner": top,
    }


def main() -> int:
    settings = c0d3r_default_settings()
    profile = settings.get("profile")
    region = settings.get("region") or "us-east-1"
    questions_path = Path("config/c0d3r_question_sheet.json")
    models_path = Path("config/c0d3r_benchmark_models.json")
    questions = _load_questions(questions_path)
    if not questions:
        print("No questions found.")
        return 1

    catalog = BedrockModelCatalog(profile=profile, region=region)
    available = {m.model_id for m in catalog.list_models()}
    profiles = catalog.list_inference_profiles()
    profile_map: Dict[str, str] = {}
    for prof in profiles:
        prof_id = str(prof.get("inferenceProfileId") or "")
        models = prof.get("modelIds") or prof.get("models") or []
        if isinstance(models, list):
            for entry in models:
                if isinstance(entry, dict):
                    model_arn = entry.get("modelArn") or ""
                    model_id = entry.get("modelId") or ""
                    if model_id:
                        profile_map[model_id] = prof_id
                    elif model_arn:
                        profile_map[model_arn.split("/")[-1]] = prof_id
                else:
                    profile_map[str(entry)] = prof_id
    candidates = _load_candidate_models(models_path)
    if not candidates:
        candidates = [
            "anthropic.claude-sonnet-4-20250514-v1:0",
            "anthropic.claude-opus-4-20250514-v1:0",
            "deepseek.r1-v1:0",
            "openai.gpt-oss-120b-1:0",
            "mistral.mistral-large-3-675b-instruct",
            "qwen.qwen3-next-80b-a3b",
            "qwen.qwen3-32b-v1:0",
        ]
    candidates = [m for m in candidates if m in available]
    if not candidates:
        print("No candidate models available in this account.")
        return 1

    runtime = BedrockClient(profile=profile, region=region, read_timeout_s=60, connect_timeout_s=10)
    results: List[Dict[str, Any]] = []
    for model_id in candidates:
        correct = 0
        total_latency = 0.0
        total_tokens_in = 0
        total_tokens_out = 0
        invoke_id = profile_map.get(model_id, model_id)
        for q in questions:
            prompt = f"Answer concisely. Final answer only.\nQ: {q['question']}"
            payload = _build_bedrock_payload(
                model_id=invoke_id,
                prompt=prompt,
                max_tokens=256,
                temperature=0.0,
                top_p=0.9,
                images=None,
            )
            start = time.time()
            try:
                response = runtime.invoke(model_id=invoke_id, payload=payload)
            except Exception as exc:
                print(f"{model_id} invocation failed: {exc}")
                response = None
            elapsed = time.time() - start
            text = _extract_text(invoke_id, response or {}) or ""
            total_latency += elapsed
            total_tokens_in += _estimate_tokens(prompt)
            total_tokens_out += _estimate_tokens(text)
            if _match_answer(text, q):
                correct += 1
        accuracy = correct / max(1, len(questions))
        avg_latency = total_latency / max(1, len(questions))
        results.append(
            {
                "model_id": model_id,
                "accuracy": round(accuracy, 4),
                "avg_latency_s": round(avg_latency, 3),
                "avg_tokens_in": round(total_tokens_in / len(questions), 2),
                "avg_tokens_out": round(total_tokens_out / len(questions), 2),
            }
        )
        print(f"{model_id} accuracy={accuracy:.3f} avg_latency={avg_latency:.2f}s")

    ranked = _score_models(results)
    routes = _recommend_routes(ranked)

    out_dir = Path("runtime/c0d3r")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "questions": len(questions),
        "results": ranked,
        "recommended_routes": routes,
    }
    (out_dir / "model_benchmarks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    routes_path = Path("config/c0d3r_rigorous_routes.json")
    if routes:
        routes_payload = {"routes": routes}
        routes_path.write_text(json.dumps(routes_payload, indent=2), encoding="utf-8")

    print("Benchmark complete. Results written to runtime/c0d3r/model_benchmarks.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
