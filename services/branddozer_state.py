from __future__ import annotations

import uuid
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from django.core.management import call_command
from django.db import DatabaseError, OperationalError, connection
from django.utils import timezone

from branddozer.models import BrandProject

STATE_KEY = "branddozer_projects"
_LEGACY_IMPORTED = False
_SCHEMA_READY = False
_SCHEMA_LOCK = threading.Lock()


def _ensure_schema(reason: str = "") -> None:
    """
    Best-effort SQLite schema setup so branddozer tables exist during local runs.
    Skips when using non-SQLite engines to avoid unintended production changes.
    """
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    if (os.getenv("BRANDDOZER_SKIP_AUTOMIGRATE") or "0").lower() in {"1", "true", "yes", "on"}:
        return
    try:
        engine = connection.settings_dict.get("ENGINE", "")
    except Exception:
        return
    if "sqlite" not in engine:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        try:
            tables = set(connection.introspection.table_names())
        except Exception:
            tables = set()
        if "branddozer_brandproject" in tables:
            _SCHEMA_READY = True
            return
        try:
            call_command("migrate", interactive=False, verbosity=0)
        except Exception:
            return
        try:
            tables = set(connection.introspection.table_names())
        except Exception:
            tables = set()
        _SCHEMA_READY = "branddozer_brandproject" in tables


def _normalize_root_path(path_value: str) -> str:
    root = Path(path_value or ".").expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Root path must be an existing directory: {root}")
    return str(root)


def _normalize_interjections(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    try:
        iterable = list(value)
    except Exception:
        return []
    return [str(item).strip() for item in iterable if str(item).strip()]


def _clamp_interval(interval: Any) -> int:
    try:
        numeric = int(interval)
    except Exception:
        numeric = 120
    return max(5, min(numeric, 720))


def _coerce_timestamp(ts_value: Any) -> Optional[Any]:
    if ts_value in (None, "", 0):
        return None
    try:
        # Accept both epoch seconds and ISO strings.
        ts_float = float(ts_value)
        return timezone.datetime.fromtimestamp(ts_float, tz=timezone.utc)
    except Exception:
        try:
            dt = timezone.datetime.fromisoformat(str(ts_value))
            if timezone.is_naive(dt):
                dt = timezone.make_aware(dt, timezone=timezone.utc)
            return dt
        except Exception:
            return None


def _serialize_project(project: BrandProject) -> Dict[str, Any]:
    def _ts(dt_val: Optional[Any]) -> Optional[int]:
        if not dt_val:
            return None
        if timezone.is_naive(dt_val):
            dt_val = timezone.make_aware(dt_val, timezone=timezone.utc)
        return int(dt_val.timestamp())

    return {
        "id": str(project.id),
        "name": project.name,
        "root_path": project.root_path,
        "default_prompt": project.default_prompt,
        "interjections": project.interjections or [],
        "interval_minutes": project.interval_minutes,
        "enabled": project.enabled,
        "last_run": _ts(project.last_run),
        "last_ai_generated": _ts(project.last_ai_generated),
        "log_path": project.log_path,
        "repo_url": project.repo_url,
        "repo_branch": project.repo_branch,
        "created_at": project.created_at.isoformat(),
        "updated_at": project.updated_at.isoformat(),
    }


def _maybe_import_legacy_state() -> None:
    global _LEGACY_IMPORTED
    if _LEGACY_IMPORTED or BrandProject.objects.exists():
        _LEGACY_IMPORTED = True
        return
    try:
        from db import get_db

        legacy = get_db().get_json(STATE_KEY) or []
    except Exception:
        legacy = []
    for entry in legacy:
        try:
            save_project(entry)
        except Exception:
            # Best-effort migration; ignore malformed entries.
            continue
    _LEGACY_IMPORTED = True


def list_projects(db: Optional[Any] = None) -> List[Dict[str, Any]]:  # db retained for backward compatibility
    _ensure_schema("list_projects")
    _maybe_import_legacy_state()
    return [_serialize_project(p) for p in BrandProject.objects.all()]


def get_project(project_id: str, db: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    _ensure_schema("get_project")
    _maybe_import_legacy_state()
    try:
        obj = BrandProject.objects.filter(id=project_id).first()
    except Exception:
        return None
    return _serialize_project(obj) if obj else None


def save_project(payload: Dict[str, Any], db: Optional[Any] = None) -> Dict[str, Any]:
    _ensure_schema("save_project")
    _maybe_import_legacy_state()
    project_id = payload.get("id")
    try:
        project_uuid = uuid.UUID(str(project_id)) if project_id else uuid.uuid4()
    except Exception:
        project_uuid = uuid.uuid4()
    existing = BrandProject.objects.filter(id=project_uuid).first()
    name = payload.get("name") or (existing.name if existing else "Untitled")
    root_path = _normalize_root_path(payload.get("root_path") or (existing.root_path if existing else "."))
    default_prompt = payload.get("default_prompt")
    if default_prompt is None and existing:
        default_prompt = existing.default_prompt
    interjections = _normalize_interjections(
        payload.get("interjections") if "interjections" in payload else (existing.interjections if existing else [])
    )
    interval_minutes = _clamp_interval(payload.get("interval_minutes") or (existing.interval_minutes if existing else 120))
    enabled = bool(payload.get("enabled") if "enabled" in payload else (existing.enabled if existing else False))
    log_path = payload.get("log_path") or (existing.log_path if existing and existing.log_path else f"runtime/branddozer/{project_uuid}.log")
    repo_url = payload.get("repo_url") or (existing.repo_url if existing else "")
    repo_branch = payload.get("repo_branch") or (existing.repo_branch if existing else "")

    defaults = {
        "name": name,
        "root_path": root_path,
        "default_prompt": default_prompt or "",
        "interjections": interjections,
        "interval_minutes": interval_minutes,
        "enabled": enabled,
        "log_path": log_path,
        "repo_url": repo_url,
        "repo_branch": repo_branch,
    }
    if payload.get("last_run") is not None:
        defaults["last_run"] = _coerce_timestamp(payload.get("last_run"))
    if payload.get("last_ai_generated") is not None:
        defaults["last_ai_generated"] = _coerce_timestamp(payload.get("last_ai_generated"))

    obj, _created = BrandProject.objects.update_or_create(id=project_uuid, defaults=defaults)
    return _serialize_project(obj)


def delete_project(project_id: str, db: Optional[Any] = None) -> None:
    _ensure_schema("delete_project")
    _maybe_import_legacy_state()
    BrandProject.objects.filter(id=project_id).delete()


def update_project_fields(project_id: str, updates: Dict[str, Any], db: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    _ensure_schema("update_project")
    _maybe_import_legacy_state()
    obj = BrandProject.objects.filter(id=project_id).first()
    if not obj:
        return None

    if "name" in updates and updates.get("name"):
        obj.name = str(updates["name"])
    if "root_path" in updates and updates.get("root_path"):
        obj.root_path = _normalize_root_path(updates["root_path"])
    if "default_prompt" in updates and updates.get("default_prompt") is not None:
        obj.default_prompt = str(updates.get("default_prompt") or "")
    if "interjections" in updates:
        obj.interjections = _normalize_interjections(updates.get("interjections"))
    if "interval_minutes" in updates and updates.get("interval_minutes") is not None:
        obj.interval_minutes = _clamp_interval(updates["interval_minutes"])
    if "enabled" in updates and updates.get("enabled") is not None:
        obj.enabled = bool(updates["enabled"])
    if "log_path" in updates and updates.get("log_path"):
        obj.log_path = str(updates["log_path"])
    if "repo_url" in updates:
        obj.repo_url = str(updates.get("repo_url") or "")
    if "repo_branch" in updates:
        obj.repo_branch = str(updates.get("repo_branch") or "")
    if "last_run" in updates:
        obj.last_run = _coerce_timestamp(updates.get("last_run"))
    if "last_ai_generated" in updates:
        obj.last_ai_generated = _coerce_timestamp(updates.get("last_ai_generated"))

    obj.save()
    return _serialize_project(obj)
