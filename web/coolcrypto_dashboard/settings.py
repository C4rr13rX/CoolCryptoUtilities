from __future__ import annotations

import os
import sys
from pathlib import Path

from corsheaders.defaults import default_headers

from services.env_loader import EnvLoader
from dotenv import load_dotenv

EnvLoader.load()

def _split_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _unique(seq: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for item in seq:
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


TESTING = "test" in sys.argv

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent

postgres_env = REPO_ROOT / ".env.postgres"
if postgres_env.exists():
    load_dotenv(postgres_env, override=False)

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "change-me-in-production")
DEBUG = os.getenv("DJANGO_DEBUG", "1").lower() in {"1", "true", "yes"}

allowed_hosts_env = _split_env_list(os.getenv("DJANGO_ALLOWED_HOSTS"))
default_allowed_host = os.getenv("DJANGO_ALLOWED_HOST", "localhost")
ALLOWED_HOSTS = _unique(
    [
        "127.0.0.1",
        "localhost",
        default_allowed_host,
        "testserver",
        *allowed_hosts_env,
    ]
)

csrf_trusted_env = _split_env_list(os.getenv("DJANGO_CSRF_TRUSTED_ORIGINS"))
single_csrf_origin = os.getenv("DJANGO_CSRF_TRUSTED_ORIGIN")
if single_csrf_origin:
    csrf_trusted_env.append(single_csrf_origin.strip())

_base_csrf_trusted = [
    "http://127.0.0.1",
    "http://localhost",
    "https://127.0.0.1",
    "https://localhost",
]
_csrf_from_hosts: list[str] = []
for host in ALLOWED_HOSTS:
    if not host:
        continue
    if host.startswith(("http://", "https://")):
        _csrf_from_hosts.append(host)
        continue
    _csrf_from_hosts.append(f"http://{host}")
    _csrf_from_hosts.append(f"https://{host}")

CSRF_TRUSTED_ORIGINS = _unique(_base_csrf_trusted + csrf_trusted_env + _csrf_from_hosts)

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
    "corsheaders",
    "rest_framework",
    "channels",
    "core.apps.CoreConfig",
    "streams.apps.StreamsConfig",
    "telemetry.apps.TelemetryConfig",
    "opsconsole.apps.OpsConsoleConfig",
    "discovery.apps.DiscoveryConfig",
    "lab.apps.LabConfig",
    "datalab.apps.DatalabConfig",
    "guardianpanel.apps.GuardianPanelConfig",
    "securevault.apps.SecureVaultConfig",
    "walletpanel.apps.WalletPanelConfig",
    "integrations.apps.IntegrationsConfig",
    "branddozer.apps.BranddozerConfig",
]

MIDDLEWARE = [
    "core.middleware.DynamicOriginMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "coolcrypto_dashboard.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "coolcrypto_dashboard.wsgi.application"
ASGI_APPLICATION = "coolcrypto_dashboard.asgi.application"

DB_VENDOR = os.getenv("DJANGO_DB_VENDOR", "postgres").lower()
if DB_VENDOR == "postgres":
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.getenv("POSTGRES_DB", "coolcrypto"),
            "USER": os.getenv("POSTGRES_USER", "postgres"),
            "PASSWORD": os.getenv("POSTGRES_PASSWORD", "postgres"),
            "HOST": os.getenv("POSTGRES_HOST", "127.0.0.1"),
            "PORT": os.getenv("POSTGRES_PORT", "5432"),
            "OPTIONS": {
                "sslmode": os.getenv("POSTGRES_SSLMODE", "prefer"),
            },
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": str(
                Path(os.getenv("DJANGO_SQLITE_PATH", REPO_ROOT / "storage" / "trading_cache.db")).resolve()
            ),
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = os.getenv("DJANGO_TIME_ZONE", "UTC")
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "collected_static"

STATICFILES_DIRS = [
    BASE_DIR / "static",
    BASE_DIR / "frontend" / "dist",
]

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.FormParser",
        "rest_framework.parsers.MultiPartParser",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticatedOrReadOnly",
    ],
}

CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOWED_ORIGINS: list[str] = []
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_HEADERS = list(default_headers) + ["X-CSRFToken", "x-csrftoken", "Authorization"]
CORS_ALLOWED_ORIGIN_REGEXES = [r"^https?://.*$", r"^http://.*$"]

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    }
}

USE_X_FORWARDED_HOST = True
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO" if DEBUG else "WARNING",
    },
}
