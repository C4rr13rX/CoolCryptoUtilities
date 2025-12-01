from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Dict, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from django.conf import settings as django_settings
from django.contrib.auth import get_user_model
from kyber_py import kyber

try:
    import django
    if django_settings.configured and not django.apps.apps.ready:  # type: ignore[attr-defined]
        django.setup()
except Exception:
    pass

try:
    from securevault.models import SecureSetting
except Exception:
    SecureSetting = None  # type: ignore

PLACEHOLDER_PATTERN = re.compile(r"\${([A-Z0-9_]+)}")
_KYBER = kyber.Kyber512
_KEY_LOCK = threading.Lock()
_LEGACY_ENV_CACHE: Optional[Dict[str, str]] = None
_LEGACY_ENV_PATH: Optional[Path] = None
_LEGACY_ENV_MTIME: float = 0.0


def _key_dir() -> Path:
    override = os.getenv("SECURE_VAULT_KEY_DIR")
    if override:
        return Path(override)
    return Path("storage/secure_vault")


def key_directory() -> Path:
    return _key_dir()


def _key_paths() -> tuple[Path, Path, Path]:
    base = _key_dir()
    return base, base / "kyber_public.bin", base / "kyber_private.bin"


def _ensure_keys() -> None:
    key_dir, public_path, private_path = _key_paths()
    key_dir.mkdir(parents=True, exist_ok=True)
    if public_path.exists() and private_path.exists():
        return
    with _KEY_LOCK:
        if public_path.exists() and private_path.exists():
            return
        public_key, private_key = _KYBER.keygen()
        public_path.write_bytes(public_key)
        private_path.write_bytes(private_key)


def _load_public_key() -> bytes:
    _ensure_keys()
    _, public_path, _ = _key_paths()
    return public_path.read_bytes()


def _load_private_key() -> bytes:
    _ensure_keys()
    _, _, private_path = _key_paths()
    return private_path.read_bytes()


def encrypt_secret(value: str) -> Dict[str, bytes]:
    if value is None:
        raise ValueError("value must be provided for secret settings")
    public_key = _load_public_key()
    shared_key, capsule = _KYBER.encaps(public_key)
    aes_key = hashlib.sha256(shared_key).digest()
    aesgcm = AESGCM(aes_key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, value.encode("utf-8"), None)
    return {
        "encapsulated_key": capsule,
        "ciphertext": ciphertext,
        "nonce": nonce,
    }


def decrypt_secret(encapsulated_key: bytes, ciphertext: bytes, nonce: bytes) -> str:
    if not (encapsulated_key and ciphertext and nonce):
        raise ValueError("encrypted payload incomplete")
    private_key = _load_private_key()
    shared_key = _KYBER.decaps(private_key, encapsulated_key)
    aes_key = hashlib.sha256(shared_key).digest()
    aesgcm = AESGCM(aes_key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode("utf-8")


def rotate_keys() -> None:
    key_dir, public_path, private_path = _key_paths()
    with _KEY_LOCK:
        for path in (public_path, private_path):
            if path.exists():
                path.unlink()
        key_dir.mkdir(parents=True, exist_ok=True)
        public_key, private_key = _KYBER.keygen()
        public_path.write_bytes(public_key)
        private_path.write_bytes(private_key)


def mask_value(value: Optional[str]) -> str:
    if not value:
        return ""
    return "•" * min(8, len(value))


def _load_legacy_env() -> Dict[str, str]:
    global _LEGACY_ENV_CACHE, _LEGACY_ENV_PATH, _LEGACY_ENV_MTIME
    env_data: Dict[str, str] = {}
    force_refresh = os.getenv("FORCE_ENV_REFRESH") or os.getenv("FORCE_ENV_RELOAD")
    candidates = [
        Path(".env"),
        Path(".env.postgres"),
        Path(".env.postgres.user"),
        Path.cwd() / ".env",
        Path.cwd() / ".env.postgres",
        Path.cwd() / ".env.postgres.user",
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parents[1] / ".env.postgres",
        Path(__file__).resolve().parents[1] / ".env.postgres.user",
    ]
    chosen: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists():
            chosen = candidate
            break
    if not chosen:
        _LEGACY_ENV_CACHE = {}
        _LEGACY_ENV_PATH = None
        _LEGACY_ENV_MTIME = 0.0
        return {}
    try:
        mtime = chosen.stat().st_mtime
    except Exception:
        mtime = 0.0
    if (
        _LEGACY_ENV_CACHE is not None
        and _LEGACY_ENV_PATH == chosen
        and _LEGACY_ENV_MTIME >= mtime
        and not force_refresh
    ):
        return _LEGACY_ENV_CACHE
    try:
        with chosen.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                name = name.strip()
                if not name or name in env_data:
                    continue
                env_data[name] = value.strip()
    except Exception:
        env_data = {}
    _LEGACY_ENV_CACHE = env_data
    _LEGACY_ENV_PATH = chosen
    _LEGACY_ENV_MTIME = mtime
    return env_data


def get_settings_for_user(user) -> Dict[str, str]:
    if SecureSetting is None:
        return {}
    if user is None:
        return {}
    settings = SecureSetting.objects.filter(user=user)
    results: Dict[str, str] = {}
    for setting in settings:
        if setting.is_secret:
            try:
                value = decrypt_secret(setting.encapsulated_key, setting.ciphertext, setting.nonce)
            except Exception:
                continue
        else:
            value = setting.value_plain or ""
        results[setting.name] = value
    return _resolve_placeholders(results)


def _resolve_placeholders(values: Dict[str, str], max_passes: int = 10) -> Dict[str, str]:
    resolved = dict(values)
    for _ in range(max_passes):
        changed = False
        for key, value in list(resolved.items()):
            if not isinstance(value, str):
                continue
            matches = PLACEHOLDER_PATTERN.findall(value)
            if not matches:
                continue
            new_value = value
            for placeholder in matches:
                replacement = resolved.get(placeholder)
                if replacement is None:
                    continue
                new_value = new_value.replace(f"${{{placeholder}}}", replacement)
            if new_value != value:
                resolved[key] = new_value
                changed = True
        if not changed:
            break
    return resolved


def default_env_user():
    User = get_user_model()
    return User.objects.filter(is_superuser=True).order_by("id").first()


def build_process_env(user=None) -> Dict[str, str]:
    user = user or default_env_user()
    env = dict(os.environ)
    repo_root = Path(__file__).resolve().parents[1]
    repo_bin = repo_root / "bin"
    try:
        repo_bin = repo_bin.resolve()
    except Exception:
        pass
    path = env.get("PATH", "")
    path_entries = [str(repo_bin)]
    if path:
        path_entries.append(path)
    env["PATH"] = os.pathsep.join(path_entries)
    for key, value in _load_legacy_env().items():
        env.setdefault(key, value)
    env.update(get_settings_for_user(user))
    # Prefer Postgres across all services unless explicitly overridden.
    if "DJANGO_DB_VENDOR" not in env and "TRADING_DB_VENDOR" in env:
        env.setdefault("DJANGO_DB_VENDOR", env["TRADING_DB_VENDOR"])
    env.setdefault("TRADING_DB_VENDOR", env.get("DJANGO_DB_VENDOR", "postgres"))
    env.setdefault("ALLOW_SQLITE_FALLBACK", "0")
    return _resolve_placeholders(env)
