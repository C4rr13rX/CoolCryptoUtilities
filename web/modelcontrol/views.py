from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from securevault.models import SecureSetting
from services.secure_settings import decrypt_secret, encrypt_secret
from tools.c0d3rV2.plugins.agent_the_freeloader.adapters import has_credential
from tools.c0d3rV2.plugins.agent_the_freeloader.models import load_catalog
from tools.c0d3rV2.plugins.agent_the_freeloader.feedback import ModelFeedbackStore
from .wizard_brains import (
    create_wizard_brain,
    delete_wizard_brain,
    get_wizard_brain,
    list_wizard_brains,
    select_wizard_brain,
    selected_wizard_brain,
    update_wizard_brain,
)


CATEGORY = "ai"
CONFIG_KEYS = frozenset({
    "C0D3R_BACKEND", "C0D3R_MODEL", "AGENT_FREELOADER_MODELS",
    "C0D3R_WIZARD_BRAIN_ID", "WIZARD_CHAT_BRAIN_ID",
})
BACKENDS = (
    {"id": "auto", "label": "Automatic", "description": "Use the first available configured backend."},
    {"id": "wizard", "label": "Wizard node", "description": "Use the local W1z4rD brain."},
    {"id": "bedrock", "label": "AWS Bedrock", "description": "Use the configured Bedrock model."},
    {"id": "claude", "label": "Anthropic", "description": "Use the Anthropic API directly."},
    {"id": "openai", "label": "OpenAI", "description": "Use the OpenAI API directly."},
    {"id": "freeloader", "label": "AgentTheFreeloader", "description": "Route across free models by quality and quota."},
)
CREDENTIALS = (
    ("WIZARD_BRAIN_CHAT_URL", "Wizard brain URL", False, "Wizard node endpoint, usually http://127.0.0.1:8090/brain/chat"),
    ("AWS_ACCESS_KEY_ID", "AWS access key ID", True, "Required for AWS Bedrock unless an AWS profile is used."),
    ("AWS_SECRET_ACCESS_KEY", "AWS secret access key", True, "AWS secret paired with the access key ID."),
    ("AWS_SESSION_TOKEN", "AWS session token", True, "Optional token for temporary AWS credentials."),
    ("AWS_PROFILE", "AWS profile", False, "Optional local AWS profile name."),
    ("AWS_DEFAULT_REGION", "AWS region", False, "Bedrock region, for example us-east-1."),
    ("ANTHROPIC_API_KEY", "Anthropic API key", True, "Direct Claude API access."),
    ("OPENAI_API_KEY", "OpenAI API key", True, "Direct OpenAI API access."),
    ("GEMINI_API_KEY", "Google Gemini API key", True, "Free-model capacity for ATF."),
    ("GROQ_API_KEY", "Groq API key", True, "Free-model capacity for ATF."),
    ("CEREBRAS_API_KEY", "Cerebras API key", True, "Free-model capacity for ATF."),
    ("SAMBANOVA_API_KEY", "SambaNova API key", True, "Free-model capacity for ATF."),
    ("OPENROUTER_API_KEY", "OpenRouter API key", True, "Free-model capacity for ATF."),
    ("MISTRAL_API_KEY", "Mistral API key", True, "Free-model capacity for ATF."),
    ("COHERE_API_KEY", "Cohere API key", True, "Free-model capacity for ATF."),
    ("HF_TOKEN", "Hugging Face token", True, "Free-model capacity for ATF."),
    ("NVIDIA_API_KEY", "NVIDIA API key", True, "Free-model capacity for ATF."),
    ("FIREWORKS_API_KEY", "Fireworks API key", True, "Free-model capacity for ATF."),
    ("GITHUB_TOKEN", "GitHub Models token", True, "Free-model capacity for ATF."),
    ("CLOUDFLARE_API_TOKEN", "Cloudflare API token", True, "Free-model capacity for ATF."),
    ("CLOUDFLARE_ACCOUNT_ID", "Cloudflare account ID", False, "Account ID paired with the Cloudflare token."),
    ("POLLINATIONS_API_KEY", "Pollinations token", True, "Optional; anonymous access remains available."),
    ("AI_GATEWAY_API_KEY", "Vercel AI Gateway key", True, "Free-credit model capacity for ATF."),
    ("MODELSCOPE_ACCESS_TOKEN", "ModelScope token", True, "Free-model capacity for ATF."),
    ("SPEKA_API_KEY", "Speka API key", True, "Free-model capacity for ATF."),
    ("SCW_SECRET_KEY", "Scaleway API key", True, "One-million-token new-customer capacity for ATF."),
    ("ZHIPU_API_KEY", "Zhipu BigModel API key", True, "Free GLM Flash model capacity for ATF."),
    ("IOINTELLIGENCE_API_KEY", "IO Intelligence API key", True, "Daily free-credit model capacity for ATF."),
    ("DASHSCOPE_API_KEY", "Alibaba Model Studio API key", True, "Per-model 90-day free quotas for ATF."),
    ("SILICONFLOW_API_KEY", "SiliconFlow API key", True, "Fixed free-model capacity for ATF."),
    ("HYPERBOLIC_API_KEY", "Hyperbolic API key", True, "Promotional inference credit for ATF."),
    ("ROUTEWAY_API_KEY", "Routeway API key", True, "Shared 200-request daily free-model capacity for ATF."),
    ("LOGFARE_API_KEY", "Logfare API key", True, "Recurring free inference capacity for ATF; review data-use terms."),
    ("LLM_KIWI_API_KEY", "LLM.kiwi API key", True, "Free auto-routed and hrLLM capacity for ATF."),
    ("LLMRACK_API_KEY", "LLMRack API key", True, "Shared 10,000-token daily free-model capacity for ATF."),
    ("KILO_API_KEY", "Kilo Gateway key", True, "Optional for free models; anonymous access is limited by IP."),
    ("OPENCODE_ZEN_API_KEY", "OpenCode Zen key", True, "Limited-time free coding-model capacity for ATF."),
    ("NAGA_API_KEY", "NagaAI API key", True, "Shared 100-request daily free-model capacity for ATF."),
    ("AIRFORCE_API_KEY", "Api.Airforce key", True, "Shared 1,000-request daily free-model capacity for ATF."),
    ("VOIDAI_API_KEY", "VoidAI API key", True, "Shared 125,000-credit daily free-plan capacity for ATF."),
)
CREDENTIAL_LOOKUP = {item[0]: item for item in CREDENTIALS}

# Curated current premium models per backend, used to populate dropdowns where a
# specific model must be named (Bedrock / Anthropic / OpenAI / Codex). The ATF
# ("freeloader") backend instead draws its models from the live free catalog.
CURATED_MODELS = {
    "claude": [
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-haiku-4-5-20251001",
        "claude-fable-5",
    ],
    "bedrock": [
        "anthropic.claude-opus-4-8-v1:0",
        "anthropic.claude-sonnet-5-v1:0",
        "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-3-5-haiku-20241022-v1:0",
    ],
    "openai": [
        "gpt-5.2",
        "gpt-5.2-mini",
        "o4-mini",
        "o3",
        "gpt-4.1",
    ],
    "codex": [
        "gpt-5.2-codex",
        "gpt-5.1-codex",
        "gpt-5.2",
        "o4-mini",
    ],
    # Claude Code CLI drives its own model set; these are the IDs the CLI
    # accepts for --model.
    "claude_code": [
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-haiku-4-5-20251001",
        "claude-fable-5",
    ],
}

# Agents that can *drive* a delivery run, as opposed to LLM backends that
# merely answer prompts.  C0D3R V2 delegates to a selectable model
# backend; the CLI agents (Codex, Claude Code) bring their own model
# namespace and therefore ignore the backend picker.
AGENTS = (
    {
        "id": "c0d3r",
        "label": "C0D3R V2",
        "description": "In-house recursive agent; you choose its model backend.",
        "owns_model": False,
        "models_key": "",
        "requires_cli": "",
    },
    {
        "id": "codex",
        "label": "Codex CLI",
        "description": "OpenAI Codex CLI agent; uses its own gpt-*-codex models.",
        "owns_model": True,
        "models_key": "codex",
        "requires_cli": "codex",
    },
    {
        "id": "claude_code",
        "label": "Claude Code",
        "description": "Anthropic Claude Code CLI agent; uses its own claude-* models.",
        "owns_model": True,
        "models_key": "claude_code",
        "requires_cli": "claude",
    },
)

# Reasoning/effort vocabularies differ per agent.
AGENT_REASONING = {
    "c0d3r": ["low", "medium", "high", "extra_high"],
    "codex": ["low", "medium", "high", "extra_high"],
    "claude_code": ["low", "medium", "high", "extra_high"],
}


def _agent_catalog() -> list[dict[str, Any]]:
    """Agents plus live CLI availability, so the UI can flag missing tools."""
    import shutil

    out = []
    for agent in AGENTS:
        cli = agent["requires_cli"]
        available = True
        detail = ""
        if cli:
            resolved = shutil.which(cli)
            available = resolved is not None
            detail = resolved or f"`{cli}` not found on PATH"
        out.append({
            **agent,
            "available": available,
            "detail": detail,
            "models": CURATED_MODELS.get(agent["models_key"], []) if agent["owns_model"] else [],
            "reasoning": AGENT_REASONING.get(agent["id"], ["medium"]),
        })
    return out


def _setting_value(user, name: str) -> str:
    setting = SecureSetting.objects.filter(user=user, category=CATEGORY, name=name).first()
    if setting is None:
        return os.getenv(name, "")
    if not setting.is_secret:
        return setting.value_plain or ""
    try:
        return decrypt_secret(setting.encapsulated_key, setting.ciphertext, setting.nonce)
    except Exception:
        return ""


def _save_setting(user, name: str, value: str, *, secret: bool) -> None:
    setting, _ = SecureSetting.objects.get_or_create(
        user=user, category=CATEGORY, name=name, defaults={"is_secret": secret}
    )
    setting.is_secret = secret
    if secret:
        payload = encrypt_secret(value)
        setting.value_plain = None
        setting.ciphertext = payload["ciphertext"]
        setting.encapsulated_key = payload["encapsulated_key"]
        setting.nonce = payload["nonce"]
    else:
        setting.value_plain = value
        setting.ciphertext = None
        setting.encapsulated_key = None
        setting.nonce = None
    setting.save()


def _clear_user_flows(user_id: int) -> None:
    try:
        from tools.c0d3rV2.web_runner import _FLOW_CACHE

        prefix = f"c0d3rv2:user:{user_id}:"
        for key in [key for key in _FLOW_CACHE if key.startswith(prefix)]:
            _FLOW_CACHE.pop(key, None)
    except Exception:
        pass


class ModelControlView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        backend = (_setting_value(request.user, "C0D3R_BACKEND") or "wizard").lower()
        model = _setting_value(request.user, "C0D3R_MODEL")
        selected_atf_models = [
            item.strip() for item in _setting_value(request.user, "AGENT_FREELOADER_MODELS").split(",") if item.strip()
        ]

        stored = {
            row.name: row
            for row in SecureSetting.objects.filter(
                user=request.user, category=CATEGORY, name__in=CREDENTIAL_LOOKUP.keys()
            )
        }
        credentials = []
        for name, label, secret, description in CREDENTIALS:
            configured = bool(_setting_value(request.user, name))
            credentials.append({
                "name": name,
                "label": label,
                "is_secret": secret,
                "description": description,
                "configured": configured,
                "source": "vault" if name in stored else "environment" if os.getenv(name) else "none",
            })

        provider_models: dict[str, list[dict[str, Any]]] = defaultdict(list)
        try:
            specs = load_catalog()
        except Exception:
            specs = []
        for spec in specs:
            provider_models[spec.provider].append({
                "id": spec.model_id,
                "best_at": spec.best_at,
                "configured": has_credential(spec),
                "selected": spec.model_id in selected_atf_models,
            })
        providers = [
            {
                "name": provider,
                "configured": any(item["configured"] for item in models),
                "models": models,
            }
            for provider, models in sorted(provider_models.items())
        ]
        return Response({
            "config": {
                "backend": backend,
                "model": model,
                "atf_models": selected_atf_models,
                "wizard_brain_id": selected_wizard_brain(request.user, "operations")["id"],
            },
            "wizard_brains": list_wizard_brains(request.user),
            "backends": BACKENDS,
            "credentials": credentials,
            "providers": providers,
            "corrections": ModelFeedbackStore().correction_snapshot(limit=100),
        })


class ModelOptionsView(APIView):
    """Lightweight, reusable source of truth for model dropdowns site-wide.

    Every page that lets a user pick an AI model reads this: the site-wide
    default (chosen on the Model Control page), the available backends, the
    curated premium models per backend, and the live Agent-the-Freeloader free
    catalog. Pages default to the site default and can override per use.
    """

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        backend = (_setting_value(request.user, "C0D3R_BACKEND") or "wizard").lower()
        model = _setting_value(request.user, "C0D3R_MODEL")

        catalog = []
        try:
            seen = set()
            for spec in load_catalog():
                if spec.model_id in seen:
                    continue
                seen.add(spec.model_id)
                catalog.append({
                    "id": spec.model_id,
                    "provider": spec.provider,
                    "best_at": spec.best_at,
                    "configured": has_credential(spec),
                })
        except Exception:
            catalog = []
        catalog.sort(key=lambda item: (item["provider"], item["id"]))

        backend_label = {item["id"]: item["label"] for item in BACKENDS}.get(backend, backend)
        default_label = backend_label + (f" · {model}" if model else "")
        return Response({
            "default": {
                "backend": backend,
                "model": model,
                "label": default_label,
                "wizard_brain_id": selected_wizard_brain(request.user, "operations")["id"],
            },
            "wizard_brains": list_wizard_brains(request.user),
            "backends": BACKENDS,
            "agents": _agent_catalog(),
            "curated": CURATED_MODELS,
            "catalog": catalog,
        })


class ModelControlConfigView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        valid_backends = {item["id"] for item in BACKENDS}
        backend = str(request.data.get("backend") or "").strip().lower()
        if backend not in valid_backends:
            return Response({"detail": "Unsupported backend."}, status=status.HTTP_400_BAD_REQUEST)
        model = str(request.data.get("model") or "").strip()
        raw_models = request.data.get("atf_models") or []
        if not isinstance(raw_models, list):
            return Response({"detail": "atf_models must be a list."}, status=status.HTTP_400_BAD_REQUEST)
        catalog_ids = {spec.model_id for spec in load_catalog()}
        atf_models = list(dict.fromkeys(str(item).strip() for item in raw_models if str(item).strip()))
        unknown = [item for item in atf_models if item not in catalog_ids]
        if unknown:
            return Response({"detail": f"Unknown ATF model: {unknown[0]}"}, status=status.HTTP_400_BAD_REQUEST)
        wizard_brain_id = str(request.data.get("wizard_brain_id") or "").strip()
        if not wizard_brain_id:
            wizard_brain_id = selected_wizard_brain(request.user, "operations")["id"]
        if get_wizard_brain(request.user, wizard_brain_id) is None:
            return Response({"detail": "Unknown Wizard brain."}, status=status.HTTP_400_BAD_REQUEST)

        _save_setting(request.user, "C0D3R_BACKEND", backend, secret=False)
        _save_setting(request.user, "C0D3R_MODEL", model, secret=False)
        _save_setting(request.user, "AGENT_FREELOADER_MODELS", ",".join(atf_models), secret=False)
        select_wizard_brain(request.user, "operations", wizard_brain_id)
        _clear_user_flows(request.user.id)

        from tools.ai_backend_mode import set_freeloader_mode
        set_freeloader_mode(backend == "freeloader")
        return Response({
            "saved": True,
            "backend": backend,
            "model": model,
            "atf_models": atf_models,
            "wizard_brain_id": wizard_brain_id,
        })


class WizardBrainListView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        return Response({
            "brains": list_wizard_brains(request.user),
            "selected": {
                purpose: selected_wizard_brain(request.user, purpose)["id"]
                for purpose in ("operations", "chat")
            },
        })

    def post(self, request, *args, **kwargs):
        try:
            profile = create_wizard_brain(request.user, {
                key: request.data.get(key)
                for key in ("name", "endpoint", "chat_path")
            })
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"brain": profile}, status=status.HTTP_201_CREATED)


class WizardBrainDetailView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def patch(self, request, brain_id: str, *args, **kwargs):
        try:
            profile = update_wizard_brain(request.user, brain_id, {
                key: request.data.get(key)
                for key in ("name", "endpoint", "chat_path")
                if key in request.data
            })
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        _clear_user_flows(request.user.id)
        return Response({"brain": profile})

    def delete(self, request, brain_id: str, *args, **kwargs):
        try:
            delete_wizard_brain(request.user, brain_id)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        _clear_user_flows(request.user.id)
        return Response(status=status.HTTP_204_NO_CONTENT)


class WizardBrainSelectionView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        purpose = str(request.data.get("purpose") or "").strip().lower()
        brain_id = str(request.data.get("brain_id") or "").strip()
        try:
            profile = select_wizard_brain(request.user, purpose, brain_id)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        if purpose == "operations":
            _clear_user_flows(request.user.id)
        return Response({"selected": profile, "purpose": purpose})


class ModelCredentialView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, name: str, *args, **kwargs):
        name = name.upper()
        definition = CREDENTIAL_LOOKUP.get(name)
        if definition is None:
            return Response({"detail": "Unsupported credential."}, status=status.HTTP_404_NOT_FOUND)
        value = str(request.data.get("value") or "").strip()
        if not value:
            return Response({"detail": "A value is required."}, status=status.HTTP_400_BAD_REQUEST)
        _save_setting(request.user, name, value, secret=bool(definition[2]))
        _clear_user_flows(request.user.id)
        return Response({"saved": True, "name": name, "configured": True})

    def delete(self, request, name: str, *args, **kwargs):
        name = name.upper()
        if name not in CREDENTIAL_LOOKUP:
            return Response({"detail": "Unsupported credential."}, status=status.HTTP_404_NOT_FOUND)
        deleted, _ = SecureSetting.objects.filter(user=request.user, category=CATEGORY, name=name).delete()
        _clear_user_flows(request.user.id)
        return Response({"deleted": bool(deleted), "name": name})
