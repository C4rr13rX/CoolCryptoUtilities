from __future__ import annotations

import hmac
import os
from pathlib import Path
from django.conf import settings
from django.http.request import split_domain_port, validate_host
from django.urls import resolve
from django.urls.exceptions import Resolver404
from services.logging_utils import log_message


def _local_bridge_token() -> str:
    token = os.getenv("CCU_LOCAL_BRIDGE_TOKEN", "").strip().lstrip("\ufeff")
    if token:
        return token
    try:
        path = Path(getattr(settings, "REPO_ROOT", Path.cwd())) / "runtime" / "local_bridge" / "token.txt"
        return path.read_text(encoding="utf-8-sig").strip().lstrip("\ufeff")
    except Exception:
        return ""


def _is_loopback_request(request) -> bool:
    remote = (request.META.get("REMOTE_ADDR") or "").strip()
    host = (request.META.get("HTTP_HOST") or request.META.get("SERVER_NAME") or "").split(":", 1)[0].strip().lower()
    return remote in {"127.0.0.1", "::1", "localhost"} or host in {"127.0.0.1", "::1", "localhost"}


def _has_valid_local_bridge_token(request) -> bool:
    expected = _local_bridge_token()
    if len(expected) < 32:
        return False
    provided = (request.META.get("HTTP_X_C4_LOCAL_AGENT_TOKEN") or "").strip()
    if len(provided) < 32:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


class LocalBridgeCsrfBypassMiddleware:
    """
    Allow the owner-controlled native bridge to call localhost API endpoints
    without copied browser cookies/CSRF tokens.

    This does not authorize public traffic.  The shared token is only accepted
    when the request is from loopback, because the bridge talks to Django at
    http://127.0.0.1:8000 on the same PC.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path_info or ""
        if path.startswith("/api/") and _is_loopback_request(request) and _has_valid_local_bridge_token(request):
            request.META["CCU_LOCAL_BRIDGE_AUTH"] = "1"
            request._dont_enforce_csrf_checks = True
        return self.get_response(request)


class LocalBridgeUserMiddleware:
    """Attach a local admin user for requests already verified by the bridge token."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.META.get("CCU_LOCAL_BRIDGE_AUTH") == "1":
            try:
                from django.contrib.auth import get_user_model

                User = get_user_model()
                user = (
                    User.objects.filter(is_active=True, is_superuser=True).order_by("id").first()
                    or User.objects.filter(is_active=True, is_staff=True).order_by("id").first()
                    or User.objects.filter(is_active=True).order_by("id").first()
                )
                if user is not None:
                    request.user = user
                    request._cached_user = user
            except Exception:
                pass
        return self.get_response(request)


class DynamicOriginMiddleware:
    """
    Loosens host/CSRF checks by trusting the incoming host automatically.
    Useful when running behind a reverse proxy where hostnames/ports change
    and env vars are not maintained.
    """

    def __init__(self, get_response):
        self.get_response = get_response
        self.enabled = os.getenv("DJANGO_AUTO_TRUST_ORIGINS", "1").lower() in {"1", "true", "yes", "on"}

    def __call__(self, request):
        if self.enabled:
            self._allow_request_host(request)
        return self.get_response(request)

    def _allow_request_host(self, request):
        raw_host = request.META.get("HTTP_HOST") or request.META.get("SERVER_NAME") or ""
        if not raw_host:
            return

        host, port = split_domain_port(raw_host)
        if not host or not validate_host(host, ["*"]):
            return

        # Allow the host dynamically for host header validation.
        if "*" not in settings.ALLOWED_HOSTS and host not in settings.ALLOWED_HOSTS:
            settings.ALLOWED_HOSTS.append(host)

        # Trust the origin (with and without port) for CSRF checks.
        scheme = "https" if request.is_secure() else "http"
        candidates = [f"{scheme}://{host}"]
        if port:
            candidates.append(f"{scheme}://{host}:{port}")

        trusted = list(getattr(settings, "CSRF_TRUSTED_ORIGINS", []))
        updated = False
        for origin in candidates:
            if origin not in trusted:
                trusted.append(origin)
                updated = True
        if updated:
            settings.CSRF_TRUSTED_ORIGINS = trusted


class ApiSlashFallbackMiddleware:
    """
    If an API request 404s without a trailing slash, retry internally with a slash.
    This prevents accidental 404s on POST/PUT/PATCH/DELETE where Django won't
    auto-redirect missing slashes.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if response.status_code != 404:
            return response
        if request.META.get("CCU_SLASH_FALLBACK"):
            return response
        path = request.path_info or ""
        if not path or path.endswith("/"):
            return response
        if not path.startswith("/api/"):
            return response
        candidate = f"{path}/"
        try:
            resolve(candidate)
        except Resolver404:
            return response
        request.META["CCU_SLASH_FALLBACK"] = "1"
        request.path_info = candidate
        request.path = candidate
        request.META["PATH_INFO"] = candidate
        return self.get_response(request)


class ApiEventLogMiddleware:
    """
    Lightweight API logger that writes error responses (or all responses when
    LOG_API_REQUESTS=1) to the shared logging bus.
    """

    # Successful GETs to these endpoints are dashboard polling noise.
    # They still log on 4xx/5xx — only 2xx/3xx GETs are silenced.
    QUIET_GET_PATHS = frozenset({
        "/api/console/status/",
        "/api/console/logs/",
        "/api/guardian/logs/",
        "/api/telemetry/advisories/",
        "/api/telemetry/trades/",
        "/api/telemetry/feedback/",
        "/api/telemetry/metrics/",
        "/api/telemetry/dashboard/",
        "/api/streams/latest/",
    })

    def __init__(self, get_response):
        self.get_response = get_response
        self.log_all = os.getenv("LOG_API_REQUESTS", "0").lower() in {"1", "true", "yes", "on"}
        extra = os.getenv("LOG_API_QUIET_PATHS", "").strip()
        if extra:
            self.quiet_paths = self.QUIET_GET_PATHS | {
                p.strip() for p in extra.split(",") if p.strip()
            }
        else:
            self.quiet_paths = self.QUIET_GET_PATHS

    def __call__(self, request):
        response = self.get_response(request)
        path = request.path_info or ""
        if path.startswith("/api/") or path.startswith("/investigations/"):
            status = getattr(response, "status_code", 200)
            quiet = (
                request.method == "GET"
                and status < 400
                and path in self.quiet_paths
            )
            if quiet:
                return response
            if self.log_all or status >= 400:
                severity = "error" if status >= 500 else "warning" if status >= 400 else "info"
                details = {"query": request.META.get("QUERY_STRING", "")}
                if status == 404 and path.startswith("/api/"):
                    try:
                        details.update(
                            {
                                "root_urlconf": getattr(settings, "ROOT_URLCONF", ""),
                                "settings_module": os.getenv("DJANGO_SETTINGS_MODULE", ""),
                                "method": request.method,
                            }
                        )
                    except Exception:
                        pass
                log_message(
                    "api",
                    f"{request.method} {path} -> {status}",
                    severity=severity,
                    details=details,
                )
        return response
