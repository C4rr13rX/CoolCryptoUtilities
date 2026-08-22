"""
Quantum-safe authentication endpoint (API Gateway -> Lambda).

Deliberately standalone: it imports neither Django nor the app registry, so a
login costs a ~0.3s cold start instead of the ~4s the full dashboard needs,
and an attacker hammering the login form cannot pull the whole ORM into memory.
That separation is also what keeps the cost down -- this function is small
enough to stay in the free tier under normal traffic.

Routes (all POST unless noted):

    POST /auth/register        {email, password}         -> {id, email}
    POST /auth/challenge       {}                        -> {challenge_id, server_key}
    POST /auth/login           {challenge_id, email, envelope}
                                                         -> {token, expires_at, user}
    POST /auth/logout          Authorization: Bearer ... -> {ok}
    GET  /auth/session         Authorization: Bearer ... -> {user}

The two-step login exists because the password must be sealed to a *fresh*
post-quantum key: the client cannot encrypt until the server has published an
encapsulation key bound to this specific attempt.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "web")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from serverless.hybrid.pq_auth import (  # noqa: E402
    AuthError,
    AuthService,
)
from serverless.hybrid.smart_storage import get_storage  # noqa: E402

logger = logging.getLogger("serverless.auth")
logging.getLogger().setLevel(os.getenv("DJANGO_LOG_LEVEL", "INFO"))

# Registration is normally closed: this dashboard has a fixed operator, and an
# open endpoint would let anyone create an account and burn Lambda/S3 spend.
REGISTRATION_OPEN = os.getenv("PQ_REGISTRATION_OPEN", "0").lower() in {
    "1", "true", "yes", "on",
}
ALLOWED_EMAILS = {
    e.strip().lower()
    for e in os.getenv("PQ_ALLOWED_EMAILS", "").split(",")
    if e.strip()
}


def _cors_headers() -> dict:
    origin = os.getenv("PQ_ALLOWED_ORIGIN", "")
    headers = {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
        # Credentials must never be cached by an intermediary.
        "X-Content-Type-Options": "nosniff",
    }
    if origin:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
        headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        headers["Access-Control-Allow-Methods"] = "POST,GET,OPTIONS"
    return headers


def _respond(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": _cors_headers(),
        "body": json.dumps(body, default=str),
    }


def _request_path(event: dict) -> str:
    ctx = event.get("requestContext") or {}
    http = ctx.get("http") or {}
    return (
        http.get("path")
        or event.get("rawPath")
        or event.get("path")
        or ctx.get("resourcePath")
        or ""
    )


def _request_method(event: dict) -> str:
    ctx = event.get("requestContext") or {}
    http = ctx.get("http") or {}
    return (http.get("method") or event.get("httpMethod") or "POST").upper()


def _parse_body(event: dict) -> dict:
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        import base64

        raw = base64.b64decode(raw).decode("utf-8")
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _bearer(event: dict) -> str:
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    auth = headers.get("authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else ""


def lambda_handler(event, context):
    storage = get_storage()
    # Warm sandboxes must not serve a row memoised during someone else's
    # request -- that would be a cross-user data leak, not just a stale read.
    storage.reset_memo()
    service = AuthService(storage=storage)

    path = _request_path(event)
    method = _request_method(event)
    route = path.rstrip("/").rsplit("/", 1)[-1].lower()

    if method == "OPTIONS":
        return _respond(204, {})

    started = time.time()
    try:
        if route == "register":
            if not REGISTRATION_OPEN:
                return _respond(403, {"error": "Registration is closed."})
            body = _parse_body(event)
            email = (body.get("email") or "").strip().lower()
            if ALLOWED_EMAILS and email not in ALLOWED_EMAILS:
                # Same wording as the closed-registration case: revealing that
                # an address is *almost* allowed is free reconnaissance.
                return _respond(403, {"error": "Registration is closed."})
            return _respond(201, service.register(email, body.get("password") or ""))

        if route == "challenge":
            return _respond(200, service.begin_login())

        if route == "login":
            body = _parse_body(event)
            envelope = body.get("envelope") or {}
            missing = [k for k in ("kem_ct", "nonce", "sealed") if not envelope.get(k)]
            if missing:
                return _respond(
                    400, {"error": f"envelope missing: {', '.join(missing)}"}
                )
            result = service.complete_login(
                body.get("challenge_id") or "",
                body.get("email") or "",
                envelope,
            )
            return _respond(200, result)

        if route == "logout":
            service.revoke_session(_bearer(event))
            return _respond(200, {"ok": True})

        if route == "session":
            payload = service.check_session(_bearer(event))
            return _respond(
                200,
                {"user": {"id": payload["sub"], "email": payload["email"]},
                 "expires_at": payload["exp"]},
            )

        return _respond(404, {"error": "unknown route"})

    except AuthError as exc:
        # 401 for a rejected credential, 4xx for the caller's mistakes. The
        # message is already vetted as safe to disclose.
        status = {
            "weak_password": 400,
            "bad_email": 400,
            "bad_challenge": 400,
            "exists": 409,
            "locked": 429,
            "disabled": 403,
            "misconfigured": 500,
            "kem_unavailable": 500,
        }.get(exc.code, 401)
        if status >= 500:
            logger.error("auth misconfiguration: %s", exc)
        else:
            logger.info("auth rejected route=%s code=%s", route, exc.code)
        return _respond(status, {"error": str(exc), "code": exc.code})

    except Exception:  # noqa: BLE001
        # Never surface internals from an auth endpoint -- stack traces here
        # describe exactly which check failed.
        logger.exception("auth handler error route=%s", route)
        return _respond(500, {"error": "internal error"})

    finally:
        logger.info(
            "auth route=%s ms=%d s3_gets=%d s3_puts=%d",
            route, int((time.time() - started) * 1000),
            storage.stats.s3_gets, storage.stats.s3_puts,
        )
