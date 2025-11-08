from __future__ import annotations

from typing import Any, Dict, Optional

from db import TradingDatabase, get_db
from monitoring_guardian.prompt_text import DEFAULT_GUARDIAN_PROMPT

SETTINGS_KEY = "guardian_settings"
ONE_TIME_PROMPT_KEY = "guardian_prompt_once"


def _ensure_db(db: Optional[TradingDatabase] = None) -> TradingDatabase:
    return db or get_db()


def get_guardian_settings(db: Optional[TradingDatabase] = None) -> Dict[str, Any]:
    database = _ensure_db(db)
    stored = database.get_json(SETTINGS_KEY) or {}
    default = {
        "enabled": True,
        "default_prompt": DEFAULT_GUARDIAN_PROMPT,
        "interval_minutes": 120,
    }
    default.update({k: v for k, v in stored.items() if v is not None})
    return default


def update_guardian_settings(updates: Dict[str, Any], db: Optional[TradingDatabase] = None) -> Dict[str, Any]:
    database = _ensure_db(db)
    settings = get_guardian_settings(database)
    settings.update({k: v for k, v in updates.items() if v is not None})
    database.set_json(SETTINGS_KEY, settings)
    return settings


def set_one_time_prompt(prompt: Optional[str], db: Optional[TradingDatabase] = None) -> None:
    database = _ensure_db(db)
    if prompt:
        database.set_json(ONE_TIME_PROMPT_KEY, {"prompt": prompt})
    else:
        database.set_json(ONE_TIME_PROMPT_KEY, {})


def consume_one_time_prompt(db: Optional[TradingDatabase] = None) -> Optional[str]:
    database = _ensure_db(db)
    payload = database.get_json(ONE_TIME_PROMPT_KEY)
    if payload and isinstance(payload, dict):
        prompt = payload.get("prompt")
    else:
        prompt = None
    if prompt:
        database.set_json(ONE_TIME_PROMPT_KEY, {})
    return prompt
