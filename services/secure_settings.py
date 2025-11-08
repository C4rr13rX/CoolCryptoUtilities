from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Dict, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from django.contrib.auth import get_user_model


def _ensure_pqcrypto_import():
    try:
        from pqcrypto.kem.ml_kem_512 import decrypt as kyber_decrypt  # type: ignore
        from pqcrypto.kem.ml_kem_512 import encrypt as kyber_encrypt  # type: ignore
        from pqcrypto.kem.ml_kem_512 import generate_keypair as kyber_generate  # type: ignore
        return kyber_decrypt, kyber_encrypt, kyber_generate
    except ModuleNotFoundError:
        import site
        import sys
        candidates = []
        try:
            candidates.append(site.getusersitepackages())
        except Exception:
            pass
        major, minor = sys.version_info[:2]
        candidates.append(Path.home() / f".local/lib/python{major}.{minor}/site-packages")
        for candidate in candidates:
            if not candidate:
                continue
            candidate_path = Path(candidate)
            if candidate_path.exists() and str(candidate_path) not in sys.path:
                sys.path.append(str(candidate_path))
        from pqcrypto.kem.ml_kem_512 import decrypt as kyber_decrypt  # type: ignore
        from pqcrypto.kem.ml_kem_512 import encrypt as kyber_encrypt  # type: ignore
        from pqcrypto.kem.ml_kem_512 import generate_keypair as kyber_generate  # type: ignore
        return kyber_decrypt, kyber_encrypt, kyber_generate


kyber_decrypt, kyber_encrypt, kyber_generate = _ensure_pqcrypto_import()

from securevault.models import SecureSetting

PLACEHOLDER_PATTERN = re.compile(r"\${([A-Z0-9_]+)}")

KEY_DIR = Path("storage/secure_vault")
PUBLIC_KEY_PATH = KEY_DIR / "kyber_public.bin"
PRIVATE_KEY_PATH = KEY_DIR / "kyber_private.bin"


def _ensure_keys() -> None:
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    if PUBLIC_KEY_PATH.exists() and PRIVATE_KEY_PATH.exists():
        return
    public_key, private_key = kyber_generate()
    PUBLIC_KEY_PATH.write_bytes(public_key)
    PRIVATE_KEY_PATH.write_bytes(private_key)


def _load_keys() -> tuple[bytes, bytes]:
    _ensure_keys()
    return PUBLIC_KEY_PATH.read_bytes(), PRIVATE_KEY_PATH.read_bytes()


def encrypt_secret(value: str) -> Dict[str, bytes]:
    if value is None:
        raise ValueError("value must be provided for secret settings")
    public_key, _ = _load_keys()
    capsule, shared_key = kyber_encrypt(public_key)
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
    _, private_key = _load_keys()
    shared_key = kyber_decrypt(encapsulated_key, private_key)
    aes_key = hashlib.sha256(shared_key).digest()
    aesgcm = AESGCM(aes_key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode("utf-8")


def mask_value(value: Optional[str]) -> str:
    if not value:
        return ""
    return "•" * min(8, len(value))


def get_settings_for_user(user) -> Dict[str, str]:
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
    env.update(get_settings_for_user(user))
    return env
