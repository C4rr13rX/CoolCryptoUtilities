from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

# CRITICAL ordering: point Django at the real settings module and put repo
# root + web/ on sys.path BEFORE importing anything from django.* . Any later
# django.setup() (ours or triggered incidentally by an import) would otherwise
# run against Django's empty global_settings (DJANGO_SETTINGS_MODULE unset),
# populate 0 apps, and mark the registry "ready" — permanently blocking the
# real INSTALLED_APPS (auth, securevault) from loading. That empty-but-ready
# registry was the true cause of `default_env_user()` returning None and the
# whole vault (MNEMONIC/ALCHEMY/...) failing to hydrate in subprocesses.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "web"):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from django.conf import settings as django_settings
from django.contrib.auth import get_user_model
try:
    from kyber_py import kyber  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    kyber = None  # type: ignore

try:
    import django
    if not django.apps.apps.ready:  # type: ignore[attr-defined]
        django.setup()
except Exception:
    pass

try:
    from securevault.models import SecureSetting
except Exception:
    SecureSetting = None  # type: ignore

PLACEHOLDER_PATTERN = re.compile(r"\${([A-Z0-9_]+)}")
_KYBER = kyber.Kyber512 if kyber is not None else None
_FALLBACK_MARKER = b"fallback-v1"
_KEY_LOCK = threading.Lock()
_LEGACY_ENV_CACHE: Optional[Dict[str, str]] = None
_LEGACY_ENV_PATH: Optional[Path] = None
_LEGACY_ENV_MTIME: float = 0.0


def _key_dir() -> Path:
    override = os.getenv("SECURE_VAULT_KEY_DIR")
    if override:
        return Path(override)
    if django_settings.configured:
        repo_root = getattr(django_settings, "REPO_ROOT", None)
        if repo_root:
            return Path(repo_root) / "storage" / "secure_vault"
        base_dir = getattr(django_settings, "BASE_DIR", None)
        if base_dir:
            return Path(base_dir).parent / "storage" / "secure_vault"
    return Path("storage/secure_vault")


def key_directory() -> Path:
    return _key_dir()


def _key_paths() -> tuple[Path, Path, Path]:
    base = _key_dir()
    return base, base / "kyber_public.bin", base / "kyber_private.bin"

def _fallback_key_path() -> Path:
    base = _key_dir()
    return base / "aes_master.bin"


def _ensure_keys() -> None:
    if _KYBER is None:
        raise RuntimeError("kyber_py not installed; Kyber key material unavailable.")
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

def _ensure_fallback_key() -> None:
    key_dir = _key_dir()
    key_dir.mkdir(parents=True, exist_ok=True)
    path = _fallback_key_path()
    if path.exists():
        return
    with _KEY_LOCK:
        if path.exists():
            return
        path.write_bytes(os.urandom(32))

def _load_fallback_key() -> bytes:
    _ensure_fallback_key()
    return _fallback_key_path().read_bytes()


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
    if _KYBER is not None:
        public_key = _load_public_key()
        shared_key, capsule = _KYBER.encaps(public_key)
        aes_key = hashlib.sha256(shared_key).digest()
        marker = capsule
    else:
        master = _load_fallback_key()
        aes_key = hashlib.sha256(master).digest()
        marker = _FALLBACK_MARKER
    aesgcm = AESGCM(aes_key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, value.encode("utf-8"), None)
    return {
        "encapsulated_key": marker,
        "ciphertext": ciphertext,
        "nonce": nonce,
    }


def _as_bytes(value: Any) -> bytes:
    """Coerce a stored crypto field back to the bytes decryption expects.

    `encrypt_secret` returns raw bytes, but the storage layer round-trips them
    through JSON as ``{"__b64__": "<base64>"}`` strings. Nothing unwrapped
    them on the way back, so `decrypt_secret` received a `str` and died on
    ``str.startswith(b"fallback-v1")`` with a TypeError -- which
    `get_settings_for_user` swallowed via a bare `continue`.

    The effect was total and silent: MNEMONIC and MNEMONIC_0 sit encrypted in
    the admin vault, the wallet could never be unlocked, and every attempt to
    build UltraSwapBridge failed with "Provide MNEMONIC or PRIVATE_KEY" --
    3,277 times in the production log. Live trading was impossible no matter
    what a strategy earned.
    """
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            try:
                payload = json.loads(text)
            except ValueError:
                payload = None
            if isinstance(payload, dict) and "__b64__" in payload:
                return base64.b64decode(payload["__b64__"])
        # A bare base64 string is also accepted; fall back to raw utf-8 bytes
        # so a legacy plain value still reaches the AES layer rather than
        # raising here.
        try:
            return base64.b64decode(text, validate=True)
        except Exception:
            return text.encode("utf-8")
    raise ValueError(f"unsupported secret field type: {type(value).__name__}")


def decrypt_secret(encapsulated_key: Any, ciphertext: Any, nonce: Any) -> str:
    if not (encapsulated_key and ciphertext and nonce):
        raise ValueError("encrypted payload incomplete")
    encapsulated_key = _as_bytes(encapsulated_key)
    ciphertext = _as_bytes(ciphertext)
    nonce = _as_bytes(nonce)
    if encapsulated_key.startswith(_FALLBACK_MARKER):
        master = _load_fallback_key()
        aes_key = hashlib.sha256(master).digest()
    else:
        if _KYBER is None:
            raise RuntimeError("kyber_py not installed; cannot decrypt Kyber-protected secret.")
        private_key = _load_private_key()
        shared_key = _KYBER.decaps(private_key, encapsulated_key)
        aes_key = hashlib.sha256(shared_key).digest()
    aesgcm = AESGCM(aes_key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode("utf-8")


def rotate_keys() -> None:
    with _KEY_LOCK:
        if _KYBER is None:
            path = _fallback_key_path()
            try:
                if path.exists():
                    path.unlink()
            finally:
                _ensure_fallback_key()
            return
        key_dir, public_path, private_path = _key_paths()
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
    model = _load_secure_setting_model()
    if model is None:
        return {}
    if user is None:
        return {}
    try:
        settings = model.objects.filter(user=user)
    except Exception:
        return {}
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


def _ensure_django_ready() -> bool:
    """Configure Django so the encrypted vault (Postgres) is reachable.

    The previous version only called ``django.setup()`` when settings were
    ALREADY configured — so a bare subprocess (production's
    ``main.py --action start_production``, feeders, scripts) that never set
    ``DJANGO_SETTINGS_MODULE`` silently failed: ``SecureSetting`` imported as
    None and ``default_env_user()`` returned None, so MNEMONIC/ALCHEMY/etc.
    never hydrated (log signature: ``secure settings loaded {alchemy: False}``
    and ``unable to initialise UltraSwapBridge``). The web app worked only
    because it runs under manage.py with the settings module already set.

    Now it bootstraps Django itself when unconfigured: puts ``web`` on the
    path, points ``DJANGO_SETTINGS_MODULE`` at the canonical settings, and
    calls ``django.setup()``. Idempotent; a no-op once Django is ready.
    """
    import sys
    from pathlib import Path
    # ALWAYS put repo root + web/ on sys.path first — regardless of whether
    # settings look "configured". Django's settings can be configured (lazy)
    # while the app registry is empty; if django.setup() then runs without
    # web/ on the path it fails importing `securevault`, leaving the registry
    # half-built so even `auth` is missing (LookupError: No installed app with
    # label 'auth'). That aborted default_env_user() and blocked all vault
    # hydration. Fixing the path unconditionally is the load-bearing change.
    root = Path(__file__).resolve().parents[1]
    for p in (root, root / "web"):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")
    try:
        from django.apps import apps as _django_apps
        if not _django_apps.ready:
            django.setup()
        return bool(_django_apps.ready)
    except Exception:
        return False


def _load_secure_setting_model():
    global SecureSetting
    if SecureSetting is not None:
        return SecureSetting
    if not _ensure_django_ready():
        return None
    try:
        from securevault.models import SecureSetting as Model
    except Exception:
        return None
    SecureSetting = Model
    return SecureSetting


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
    try:
        User = get_user_model()
        model = _load_secure_setting_model()
        if model is not None:
            key_names = [
                "MNEMONIC",
                "PRIVATE_KEY",
                "ALCHEMY_API_KEY",
                "INFURA_API_KEY",
                "ANKR_API_KEY",
                "ZEROX_API_KEY",
                "ONEINCH_API_KEY",
                "LIFI_API_KEY",
                "COINGECKO_API_KEY",
                "CRYPTOPANIC_API_KEY",
                "THEGRAPH_API_KEY",
            ]
            setting = (
                model.objects.filter(name__in=key_names)
                .select_related("user")
                .order_by("-updated_at")
                .first()
            )
            if setting and setting.user:
                return setting.user
        superuser = User.objects.filter(is_superuser=True).order_by("id").first()
        if superuser:
            return superuser
        return User.objects.order_by("id").first()
    except Exception:
        return None


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
