from __future__ import annotations

import os
from django.conf import settings
from django.http.request import split_domain_port, validate_host
from django.urls import resolve
from django.urls.exceptions import Resolver404


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
