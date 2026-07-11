from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from .models import ModelSpec


@dataclass
class ProviderResponse:
    text: str
    headers: dict[str, str]
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return int(self.input_tokens or 0) + int(self.output_tokens or 0)


class ProviderError(RuntimeError):
    def __init__(self, status: int, message: str, headers: Mapping[str, str] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.headers = {str(key).lower(): str(value) for key, value in (headers or {}).items()}

    @property
    def is_quota(self) -> bool:
        text = str(self).lower()
        return self.status in {402, 429} or any(
            marker in text for marker in ("quota", "rate limit", "too many requests", "credit")
        )

    @property
    def is_auth(self) -> bool:
        return self.status in {401, 403}

    @property
    def retry_after(self) -> float:
        try:
            default = "2592000" if self.status == 402 else "60"
            return max(1.0, float(self.headers.get("retry-after", default)))
        except ValueError:
            return 2592000.0 if self.status == 402 else 60.0


def has_credential(spec: ModelSpec) -> bool:
    if not spec.api_key_env:
        return True
    if spec.provider == "Cloudflare Workers AI" and not os.getenv("CLOUDFLARE_ACCOUNT_ID"):
        return False
    return bool(resolve_secret(spec.api_key_env))


def invoke(
    spec: ModelSpec,
    *,
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> ProviderResponse:
    api_key = resolve_secret(spec.api_key_env)
    if spec.provider == "Pollinations.AI" and not api_key:
        # Anonymous access remains available without a key; a free Seed token
        # raises the documented rate limit when configured.
        api_key = resolve_secret("POLLINATIONS_API_KEY")
    if spec.provider == "Kilo Gateway" and not api_key:
        # Free models support anonymous IP-scoped access; an optional account
        # key raises continuity and makes usage visible in the Kilo dashboard.
        api_key = resolve_secret("KILO_API_KEY")
    if spec.api_key_env and not api_key:
        raise ProviderError(401, f"missing credential {spec.api_key_env}")
    if spec.provider == "Google Gemini API":
        return _invoke_google(spec, api_key, prompt, system, max_tokens, temperature, timeout_s)
    if spec.provider == "Cloudflare Workers AI":
        return _invoke_cloudflare(spec, api_key, prompt, system, max_tokens, temperature, timeout_s)
    if spec.provider == "Cohere":
        return _invoke_cohere(spec, api_key, prompt, system, max_tokens, temperature, timeout_s)
    return _invoke_openai_compatible(spec, api_key, prompt, system, max_tokens, temperature, timeout_s)


def _invoke_openai_compatible(
    spec: ModelSpec,
    api_key: str,
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> ProviderResponse:
    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": spec.model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    if spec.provider == "GitHub Models":
        headers.update({"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2026-03-10"})
    data, response_headers = _post(_url(spec), payload, headers, timeout_s)
    try:
        text = (data["choices"][0]["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(502, f"unexpected {spec.provider} response shape") from exc
    usage = data.get("usage") or {}
    return ProviderResponse(
        text=text,
        headers=response_headers,
        input_tokens=_int_or_none(usage.get("prompt_tokens") or usage.get("input_tokens")),
        output_tokens=_int_or_none(usage.get("completion_tokens") or usage.get("output_tokens")),
    )


def _invoke_google(
    spec: ModelSpec,
    api_key: str,
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> ProviderResponse:
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
    }
    if system.strip():
        payload["systemInstruction"] = {"parts": [{"text": system.strip()}]}
    data, headers = _post(_url(spec), payload, {"x-goog-api-key": api_key}, timeout_s)
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(str(part.get("text") or "") for part in parts).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(502, "unexpected Google response shape") from exc
    usage = data.get("usageMetadata") or {}
    return ProviderResponse(
        text=text,
        headers=headers,
        input_tokens=_int_or_none(usage.get("promptTokenCount")),
        output_tokens=_int_or_none(usage.get("candidatesTokenCount")),
    )


def _invoke_cohere(
    spec: ModelSpec,
    api_key: str,
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> ProviderResponse:
    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": spec.model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data, headers = _post(_url(spec), payload, {"Authorization": f"Bearer {api_key}"}, timeout_s)
    try:
        content = data["message"]["content"]
        text = "".join(str(block.get("text") or "") for block in content).strip()
    except (KeyError, TypeError) as exc:
        raise ProviderError(502, "unexpected Cohere response shape") from exc
    usage = data.get("usage") or {}
    billed = usage.get("billed_units") or {}
    return ProviderResponse(
        text=text,
        headers=headers,
        input_tokens=_int_or_none(billed.get("input_tokens")),
        output_tokens=_int_or_none(billed.get("output_tokens")),
    )


def _invoke_cloudflare(
    spec: ModelSpec,
    api_key: str,
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> ProviderResponse:
    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": prompt})
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    data, headers = _post(_url(spec), payload, {"Authorization": f"Bearer {api_key}"}, timeout_s)
    result = data.get("result") or {}
    text = str(result.get("response") or result.get("text") or "").strip()
    if not text:
        raise ProviderError(502, "unexpected Cloudflare response shape")
    usage = result.get("usage") or {}
    return ProviderResponse(
        text=text,
        headers=headers,
        input_tokens=_int_or_none(usage.get("prompt_tokens")),
        output_tokens=_int_or_none(usage.get("completion_tokens")),
    )


def _url(spec: ModelSpec) -> str:
    base = spec.base_url
    if "{ACCOUNT_ID}" in base:
        base = base.replace("{ACCOUNT_ID}", os.getenv("CLOUDFLARE_ACCOUNT_ID", ""))
    return f"{base}/{spec.endpoint.lstrip('/')}"


def _post(url: str, payload: dict, headers: dict[str, str], timeout_s: float) -> tuple[dict, dict[str, str]]:
    request_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "AgentTheFreeloader/2.0 (+https://github.com/)",
        **headers,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8", errors="replace")
            response_headers = {key.lower(): value for key, value in response.headers.items()}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:2000]
        raise ProviderError(exc.code, body or str(exc), dict(exc.headers.items())) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderError(503, str(exc)) from exc
    try:
        return json.loads(raw), response_headers
    except json.JSONDecodeError as exc:
        raise ProviderError(502, f"invalid JSON response from {url}") from exc


def resolve_secret(name: str) -> str:
    try:
        from tools.secret_vault import get_secret  # type: ignore

        value = get_secret(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, "")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
