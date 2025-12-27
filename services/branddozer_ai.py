from __future__ import annotations

import json
import os
from typing import Any, List, Optional

from openai import OpenAI

MODEL = "gpt-4o-mini"


def _normalize_api_key(value: Optional[str]) -> str:
    if not value:
        return ""
    cleaned = str(value).strip()
    if cleaned.lower().startswith("bearer "):
        cleaned = cleaned.split(None, 1)[1].strip()
    return cleaned


def _extract_interjections(text: str) -> List[str]:
    if not text:
        return []
    try:
        payload = json.loads(text)
    except Exception:
        return []
    items = payload.get("interjections")
    if not isinstance(items, list):
        return []
    return [str(item).strip() for item in items if str(item).strip()]


def _response_text(resp: Any) -> str:
    if resp is None:
        return ""
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    output = getattr(resp, "output", None)
    if output:
        for item in output:
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content")
            for part in content or []:
                part_type = getattr(part, "type", None)
                if part_type is None and isinstance(part, dict):
                    part_type = part.get("type")
                if part_type in {"output_text", "text"}:
                    value = getattr(part, "text", None)
                    if value is None and isinstance(part, dict):
                        value = part.get("text")
                    if value:
                        return str(value).strip()
    return str(text or "").strip()


def generate_interjections(default_prompt: str, project_name: str | None = None, api_key: Optional[str] = None) -> List[str]:
    """
    Ask OpenAI to expand a default prompt into a set of interjection prompts.
    Retries until valid JSON is returned (up to 3 attempts).
    """
    base_default = (default_prompt or "").strip()
    if not base_default:
        return []
    key = _normalize_api_key(api_key or os.getenv("OPENAI_API_KEY"))
    if not key:
        raise ValueError("OPENAI_API_KEY is not configured")
    client = OpenAI(api_key=key)

    sys_msg = (
        "You produce JSON with key `interjections` containing an array of short, actionable prompts "
        "to run AFTER the default prompt for this project. "
        "Each prompt must be concise, imperative, and safe. "
        "Scope strictly to software delivery tasks a Codex coding agent can perform: "
        "implement features, fix bugs, refactor, add tests, update docs, improve tooling, "
        "review code, or analyze logs. "
        "No operational, HR, or policy tasks. "
        "Return ONLY JSON. Limit to 2-8 items. No prose."
    )
    user_msg = {
        "project": project_name or "Project",
        "default_prompt": base_default,
        "format": {"interjections": ["string"]},
        "constraints": [
            "Be terse (1-2 sentences per item).",
            "Stay within the given project scope and codebase.",
            "Only coding-related tasks (code changes, tests, debugging, docs, tooling).",
            "Avoid destructive actions or secrets.",
        ],
    }
    last_error: Optional[Exception] = None
    for _ in range(3):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": json.dumps(user_msg)},
                ],
                response_format={"type": "json_object"},
            )
            text = _response_text(resp)
        except Exception as exc:
            last_error = exc
            try:
                fallback = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": json.dumps(user_msg)},
                    ],
                    response_format={"type": "json_object"},
                )
                text = (fallback.choices[0].message.content or "").strip()
            except Exception as exc_fallback:
                last_error = exc_fallback
                continue
        cleaned = _extract_interjections(text)
        if cleaned:
            return cleaned
    if last_error:
        raise ValueError(str(last_error))
    return []


__all__ = ["generate_interjections"]
