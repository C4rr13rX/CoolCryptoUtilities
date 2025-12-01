from __future__ import annotations

import json
import os
from typing import List, Optional

from openai import OpenAI

MODEL = "gpt-o4-mini"


def generate_interjections(default_prompt: str, project_name: str | None = None, api_key: Optional[str] = None) -> List[str]:
    """
    Ask OpenAI to expand a default prompt into a set of interjection prompts.
    Retries until valid JSON is returned (up to 3 attempts).
    """
    base_default = (default_prompt or "").strip()
    if not base_default:
        return []
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is not configured")
    client = OpenAI(api_key=key)

    sys_msg = (
        "You produce JSON with key `interjections` containing an array of short, actionable prompts "
        "to run AFTER the default prompt for this project. "
        "Each prompt must be concise, imperative, and safe. "
        "Return ONLY JSON. Limit to 2-8 items. No prose."
    )
    user_msg = {
        "project": project_name or "Project",
        "default_prompt": base_default,
        "format": {"interjections": ["string"]},
        "constraints": [
            "Be terse (1-2 sentences per item).",
            "Stay within the given project scope.",
            "Avoid destructive actions or secrets.",
        ],
    }
    attempts = 0
    while attempts < 3:
        attempts += 1
        resp = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": json.dumps(user_msg)},
            ],
            response_format={"type": "json_object"},
        )
        text = (resp.output_text or "").strip()
        try:
            payload = json.loads(text)
        except Exception:
            continue
        items = payload.get("interjections")
        if not isinstance(items, list):
            continue
        cleaned = [str(item) for item in items if str(item).strip()]
        if cleaned:
            return cleaned
    return []


__all__ = ["generate_interjections"]
