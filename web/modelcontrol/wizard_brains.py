"""Per-user Wizard brain registry and independent purpose selections."""
from __future__ import annotations

import json
import os
import uuid
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from securevault.models import SecureSetting
from services.secure_settings import decrypt_secret


CATEGORY = "ai"
PROFILES_KEY = "WIZARD_BRAIN_PROFILES"
SELECTION_KEYS = {
    "operations": "C0D3R_WIZARD_BRAIN_ID",
    "chat": "WIZARD_CHAT_BRAIN_ID",
}
DEFAULT_PROFILE_ID = "environment-default"
VALID_CHAT_PATHS = {"/brain/chat", "/chat"}


def _read_setting(user, name: str) -> str:
    if not getattr(user, "is_authenticated", False):
        return os.getenv(name, "")
    setting = SecureSetting.objects.filter(user=user, category=CATEGORY, name=name).first()
    if setting is None:
        return os.getenv(name, "")
    if not setting.is_secret:
        return setting.value_plain or ""
    try:
        return decrypt_secret(setting.encapsulated_key, setting.ciphertext, setting.nonce)
    except Exception:
        return ""


def _write_plain_setting(user, name: str, value: str) -> None:
    setting, _ = SecureSetting.objects.get_or_create(
        user=user, category=CATEGORY, name=name, defaults={"is_secret": False}
    )
    setting.is_secret = False
    setting.value_plain = value
    setting.ciphertext = None
    setting.encapsulated_key = None
    setting.nonce = None
    setting.save()


def normalize_brain_address(endpoint: str, chat_path: str = "") -> tuple[str, str]:
    """Return a validated node base URL and one supported chat path."""
    raw = str(endpoint or "").strip().rstrip("/")
    if not raw:
        raise ValueError("A Wizard brain endpoint is required.")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Wizard brain endpoints must be absolute http:// or https:// URLs.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Wizard brain endpoints cannot contain credentials, a query, or a fragment.")

    embedded_path = parsed.path.rstrip("/")
    selected_path = str(chat_path or "").strip()
    for suffix in ("/brain/chat", "/chat"):
        if embedded_path.endswith(suffix):
            if not selected_path:
                selected_path = suffix
            embedded_path = embedded_path[: -len(suffix)]
            break
    selected_path = selected_path or (
        "/chat"
        if os.getenv("WIZARD_USE_BRAIN_PREFIX", "1").strip().lower() in {"0", "false", "no"}
        else "/brain/chat"
    )
    if not selected_path.startswith("/"):
        selected_path = f"/{selected_path}"
    if selected_path not in VALID_CHAT_PATHS:
        raise ValueError("Wizard chat path must be /brain/chat or /chat.")

    base = urlunsplit((parsed.scheme, parsed.netloc, embedded_path, "", "")).rstrip("/")
    return base, selected_path


def _environment_profile(user) -> dict[str, str]:
    raw = (
        _read_setting(user, "WIZARD_BRAIN_CHAT_URL")
        or os.getenv("WIZARD_BRAIN_URL")
        or os.getenv("WIZARD_NODE_URL")
        or "http://localhost:8090"
    )
    endpoint, chat_path = normalize_brain_address(raw)
    return {
        "id": DEFAULT_PROFILE_ID,
        "name": os.getenv("WIZARD_DEFAULT_BRAIN_NAME", "Default Wizard brain"),
        "endpoint": endpoint,
        "chat_path": chat_path,
    }


def list_wizard_brains(user) -> list[dict[str, str]]:
    profiles = [_environment_profile(user)]
    raw = _read_setting(user, PROFILES_KEY)
    try:
        stored = json.loads(raw) if raw else []
    except (TypeError, ValueError):
        stored = []
    seen = {DEFAULT_PROFILE_ID}
    if isinstance(stored, list):
        for item in stored:
            if not isinstance(item, dict):
                continue
            profile_id = str(item.get("id") or "").strip()
            name = str(item.get("name") or "").strip()
            try:
                endpoint, chat_path = normalize_brain_address(
                    str(item.get("endpoint") or ""), str(item.get("chat_path") or "")
                )
            except ValueError:
                continue
            if not profile_id or profile_id in seen or not name:
                continue
            profiles.append({
                "id": profile_id,
                "name": name[:120],
                "endpoint": endpoint,
                "chat_path": chat_path,
            })
            seen.add(profile_id)
    return profiles


def _store_profiles(user, profiles: list[dict[str, str]]) -> None:
    stored = [item for item in profiles if item["id"] != DEFAULT_PROFILE_ID]
    _write_plain_setting(user, PROFILES_KEY, json.dumps(stored, separators=(",", ":")))


def get_wizard_brain(user, profile_id: str) -> dict[str, str] | None:
    wanted = str(profile_id or "").strip()
    return next((item for item in list_wizard_brains(user) if item["id"] == wanted), None)


def selected_wizard_brain(user, purpose: str) -> dict[str, str]:
    if purpose not in SELECTION_KEYS:
        raise ValueError("Unknown Wizard brain selection purpose.")
    profiles = list_wizard_brains(user)
    selected_id = _read_setting(user, SELECTION_KEYS[purpose]).strip()
    return next((item for item in profiles if item["id"] == selected_id), profiles[0])


def select_wizard_brain(user, purpose: str, profile_id: str) -> dict[str, str]:
    if purpose not in SELECTION_KEYS:
        raise ValueError("Purpose must be chat or operations.")
    profile = get_wizard_brain(user, profile_id)
    if profile is None:
        raise ValueError("The selected Wizard brain does not exist.")
    _write_plain_setting(user, SELECTION_KEYS[purpose], profile["id"])
    return profile


def create_wizard_brain(user, payload: dict[str, Any]) -> dict[str, str]:
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("A brain name is required.")
    endpoint, chat_path = normalize_brain_address(
        str(payload.get("endpoint") or ""), str(payload.get("chat_path") or "")
    )
    profiles = list_wizard_brains(user)
    profile = {
        "id": str(uuid.uuid4()),
        "name": name[:120],
        "endpoint": endpoint,
        "chat_path": chat_path,
    }
    profiles.append(profile)
    _store_profiles(user, profiles)
    return profile


def update_wizard_brain(user, profile_id: str, payload: dict[str, Any]) -> dict[str, str]:
    if profile_id == DEFAULT_PROFILE_ID:
        raise ValueError("The environment-default profile is configured through environment settings.")
    profiles = list_wizard_brains(user)
    index = next((i for i, item in enumerate(profiles) if item["id"] == profile_id), None)
    if index is None:
        raise ValueError("The Wizard brain does not exist.")
    current = profiles[index]
    name = str(payload.get("name", current["name"]) or "").strip()
    if not name:
        raise ValueError("A brain name is required.")
    endpoint, chat_path = normalize_brain_address(
        str(payload.get("endpoint", current["endpoint"]) or ""),
        str(payload.get("chat_path", current["chat_path"]) or ""),
    )
    profiles[index] = {
        "id": profile_id,
        "name": name[:120],
        "endpoint": endpoint,
        "chat_path": chat_path,
    }
    _store_profiles(user, profiles)
    return profiles[index]


def delete_wizard_brain(user, profile_id: str) -> None:
    if profile_id == DEFAULT_PROFILE_ID:
        raise ValueError("The environment-default profile cannot be deleted.")
    profiles = list_wizard_brains(user)
    if not any(item["id"] == profile_id for item in profiles):
        raise ValueError("The Wizard brain does not exist.")
    _store_profiles(user, [item for item in profiles if item["id"] != profile_id])
    for purpose, key in SELECTION_KEYS.items():
        if _read_setting(user, key).strip() == profile_id:
            _write_plain_setting(user, key, DEFAULT_PROFILE_ID)
