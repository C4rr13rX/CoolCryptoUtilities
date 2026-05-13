"""tools/secret_vault.py — thin façade over the quantum-safe SecureSetting vault.

Lets non-Django code (session adapters, CLI scripts) look up secrets that
live in the project's encrypted vault (`services/secure_settings.py` →
`securevault.SecureSetting`) without each call site needing to bootstrap
Django.

Lookup order:
  1. The vault (DB-backed, Kyber-wrapped AES-GCM ciphertext per setting).
  2. Process env (`os.getenv`).
  3. Empty string.

The vault is the source of truth — env is only a transitional fallback
while we migrate every consumer.  Once everything routes through the
vault we can drop the env path entirely.

Callers should be defensive: vault availability depends on Django being
importable and configured.  `get_secret()` swallows every error and
returns "" if anything fails, so module-load order can't break a session
adapter at import time.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

_INIT_LOCK = threading.Lock()
_DJANGO_READY: Optional[bool] = None  # tri-state: None=unattempted, True/False after first try


def _try_django_setup() -> bool:
    """Best-effort Django bootstrap.  Idempotent and thread-safe."""
    global _DJANGO_READY
    if _DJANGO_READY is not None:
        return _DJANGO_READY
    with _INIT_LOCK:
        if _DJANGO_READY is not None:
            return _DJANGO_READY
        try:
            import django  # type: ignore
            from django.conf import settings as django_settings  # type: ignore
            if not django_settings.configured:
                _DJANGO_READY = False
                return False
            if not django.apps.apps.ready:  # type: ignore[attr-defined]
                django.setup()
            _DJANGO_READY = True
        except Exception:
            _DJANGO_READY = False
        return bool(_DJANGO_READY)


def get_secret(name: str, *, user=None) -> str:
    """Resolve a secret by name.

    name  Secret key (e.g. "OPENAI_API_KEY", "ANTHROPIC_API_KEY").
    user  Optional Django user.  If omitted, uses the project's
          `default_env_user` (typically the superuser).

    Returns the secret value, or "" if it couldn't be resolved.
    """
    if not name:
        return ""

    # 1) Vault lookup (Django-required).
    if _try_django_setup():
        try:
            from services.secure_settings import (
                get_settings_for_user,
                default_env_user,
                decrypt_secret,
            )
            from securevault.models import SecureSetting  # type: ignore

            # Per-user lookup if a user is supplied; otherwise grab the
            # most recent SecureSetting by that name across all users
            # (matches what the rest of the codebase does for shared
            # provider keys like OPENAI_API_KEY).
            if user is not None:
                bundle = get_settings_for_user(user)
                val = bundle.get(name, "")
                if val:
                    return val
            else:
                setting = (
                    SecureSetting.objects.filter(name=name)
                    .order_by("-updated_at")
                    .first()
                )
                if setting is not None:
                    if setting.is_secret:
                        try:
                            return decrypt_secret(
                                setting.encapsulated_key,
                                setting.ciphertext,
                                setting.nonce,
                            )
                        except Exception:
                            pass
                    else:
                        return setting.value_plain or ""
        except Exception:
            # Vault unreachable for this call; fall through to env.
            pass

    # 2) Env fallback (transitional).
    return os.getenv(name, "")


def vault_available() -> bool:
    """For diagnostics: did the vault path succeed in initialising?"""
    return _try_django_setup()
