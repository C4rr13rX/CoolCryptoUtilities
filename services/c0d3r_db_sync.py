from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _setup_django() -> bool:
    try:
        import django
        from django.conf import settings
    except Exception:
        return False
    if not settings.configured:
        project_root = Path(__file__).resolve().parents[1]
        web_root = project_root / "web"
        if web_root.exists() and str(web_root) not in sys.path:
            sys.path.insert(0, str(web_root))
        os.environ.setdefault(
            "DJANGO_SETTINGS_MODULE",
            os.getenv("C0D3R_DJANGO_SETTINGS", "coolcrypto_dashboard.settings"),
        )
        try:
            django.setup()
        except Exception:
            return False
    return True


def _get_user() -> Optional[object]:
    username = str(os.getenv("C0D3R_DB_USER", "")).strip()
    raw_user_id = str(os.getenv("C0D3R_DB_USER_ID", "")).strip()
    if not username and not raw_user_id:
        return None
    if not _setup_django():
        return None
    try:
        from django.contrib.auth import get_user_model
    except Exception:
        return None
    User = get_user_model()
    if raw_user_id:
        try:
            return User.objects.get(id=int(raw_user_id))
        except Exception:
            return None
    if not username:
        return None
    try:
        return User.objects.get(username=username)
    except User.DoesNotExist:
        if not _env_flag("C0D3R_DB_USER_CREATE", True):
            return None
        try:
            user = User.objects.create(username=username, is_active=True)
            try:
                user.set_unusable_password()
                user.save(update_fields=["password"])
            except Exception:
                pass
            return user
        except Exception:
            return None


def _find_or_create_session(user, *, session_name: str, session_id: int | None, workdir: Optional[str]) -> Optional[object]:
    try:
        from core.models import C0d3rWebSession
    except Exception:
        return None
    session = (
        C0d3rWebSession.objects.filter(user=user, metadata__cli_session_name=session_name)
        .order_by("-updated_at")
        .first()
    )
    if session:
        return session
    metadata = {
        "source": "cli",
        "cli_session_name": session_name,
    }
    if session_id:
        metadata["cli_session_id"] = session_id
    if workdir:
        metadata["workdir"] = workdir
    try:
        return C0d3rWebSession.objects.create(
            user=user,
            title=session_name,
            summary="",
            key_points=[],
            metadata=metadata,
        )
    except Exception:
        return None


def sync_cli_exchange(
    *,
    session_name: str,
    session_id: int | None,
    prompt: str,
    response: str,
    model_id: str | None,
    research: Optional[bool],
    workdir: Optional[str],
) -> None:
    if not _env_flag("C0D3R_DB_SYNC", False):
        return
    user = _get_user()
    if not user:
        return
    if not _setup_django():
        return
    session = _find_or_create_session(
        user,
        session_name=session_name,
        session_id=session_id,
        workdir=workdir,
    )
    if not session:
        return
    try:
        from core.models import C0d3rWebMessage
        from django.utils import timezone
    except Exception:
        return
    try:
        C0d3rWebMessage.objects.create(
            session=session,
            role="user",
            content=prompt or "",
            metadata={"source": "cli", "research": bool(research)},
        )
        C0d3rWebMessage.objects.create(
            session=session,
            role="c0d3r",
            content=response or "",
            model_id=model_id or "",
            metadata={"source": "cli"},
        )
        session.model_id = model_id or session.model_id
        session.last_active = timezone.now()
        session.save(update_fields=["model_id", "last_active", "updated_at"])
    except Exception:
        return
