"""
Lambda settings overlay for the CoolCrypto dashboard.

This module imports the full production settings and then overrides only the
things that are actually incompatible with a serverless runtime.  The goal is
that ``settings.py`` stays the single source of truth for apps, URLs, DRF,
i18n and middleware, and this file documents -- in one place -- every
assumption that a long-lived Waitress process makes but Lambda cannot.

Activated with ``DJANGO_SETTINGS_MODULE=coolcrypto_dashboard.settings_lambda``.

What changes and why:

1. **No background threads.**  ``core.apps.CoreConfig.ready()`` spawns guardian,
   market-brain, production and cron daemon threads.  Lambda freezes the
   execution environment between invocations, so a thread that sleeps is
   suspended mid-work and resumes at an arbitrary later time -- or never, if
   the sandbox is reaped.  Those workloads move to dedicated scheduled Lambdas
   (see ``serverless/handlers/cron.py``).  We set the same env switches
   ``manage.py`` uses so the threads never start.

2. **No local filesystem writes.**  Only ``/tmp`` is writable on Lambda and it
   does not survive a cold start.  Media goes to S3.

3. **Connection handling.**  Lambda cannot pool connections across concurrent
   sandboxes, so ``CONN_MAX_AGE`` stays 0 and we let RDS Proxy (or PgBouncer
   locally) do the pooling.

4. **Static files.**  WhiteNoise still serves them from the bundle so a single
   function can answer both HTML and asset requests without a CloudFront
   origin, but the manifest storage is relaxed so a missing hashed asset
   degrades to a warning instead of a 500.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Neutralise boot-time background work BEFORE the base settings import Django
# apps.  core/apps.py reads these at AppConfig.ready() time.
# ---------------------------------------------------------------------------
os.environ["GUARDIAN_AUTO_DISABLED"] = "1"
os.environ["PRODUCTION_AUTO_DISABLED"] = "1"
os.environ["CRON_AUTO_DISABLED"] = "1"
os.environ["WIZARD_DISABLE_REFRESHER"] = "1"
os.environ["WIZARD_BRAIN_FEEDER_ENABLED"] = "0"
# core/apps.py only bootstraps when WAITRESS_PORT is set or the command is
# runserver/start_production.  Under Lambda none of those are true, but we make
# the intent explicit so a stray env var cannot re-enable it.
os.environ.pop("WAITRESS_PORT", None)
os.environ["CORE_AUTO_BOOTSTRAP"] = "0"

from .settings import *  # noqa: E402,F401,F403  (re-export the full config)

# ---------------------------------------------------------------------------
# Runtime identity
# ---------------------------------------------------------------------------
IS_LAMBDA = bool(os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
# LocalStack sets a Lambda-shaped environment too, so this stays true locally.

# The `serverless` app holds the WebSocket connection registry. It is added
# here rather than in base settings so the Waitress/Channels deployment keeps
# an unchanged app list and migration set.
INSTALLED_APPS = list(INSTALLED_APPS)  # noqa: F405
if "serverless.apps.ServerlessConfig" not in INSTALLED_APPS:
    INSTALLED_APPS.append("serverless.apps.ServerlessConfig")

# Drop Channels. Its whole job -- holding WebSocket connections in a
# long-lived ASGI process -- is taken over by API Gateway's WebSocket API plus
# serverless/handlers/websocket.py. Keeping it installed would only force the
# dependency into the bundle for an app that never handles a request here:
# the HTTP handler builds its ASGI app from get_asgi_application() rather than
# coolcrypto_dashboard.asgi, so the ProtocolTypeRouter is never constructed.
INSTALLED_APPS = [a for a in INSTALLED_APPS if a != "channels"]
CHANNEL_LAYERS = {}

# ---------------------------------------------------------------------------
# Hosts / CSRF -- API Gateway fronts the function with its own domain.
# ---------------------------------------------------------------------------
_api_domain = os.getenv("API_GATEWAY_DOMAIN", "").strip()
_stage = os.getenv("API_GATEWAY_STAGE", "prod").strip()

ALLOWED_HOSTS = list(ALLOWED_HOSTS)  # noqa: F405
for _host in (_api_domain, ".execute-api.amazonaws.com", ".amazonaws.com",
              "localhost.localstack.cloud", ".localhost.localstack.cloud"):
    if _host and _host not in ALLOWED_HOSTS:
        ALLOWED_HOSTS.append(_host)

# Behind API Gateway everything arrives as HTTPS with the proto in a header.
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
USE_X_FORWARDED_HOST = True

if _api_domain:
    _origin = f"https://{_api_domain}"
    CSRF_TRUSTED_ORIGINS = list(CSRF_TRUSTED_ORIGINS)  # noqa: F405
    if _origin not in CSRF_TRUSTED_ORIGINS:
        CSRF_TRUSTED_ORIGINS.append(_origin)

# API Gateway terminates TLS. Letting Django also redirect produces a loop
# because the function never sees the original scheme without the header above.
SECURE_SSL_REDIRECT = os.getenv("DJANGO_SECURE_SSL_REDIRECT", "0").lower() in {
    "1", "true", "yes", "on"
}

# API Gateway strips the stage from the path when using a $default stage, but
# keeps it for named stages. FORCE_SCRIPT_NAME makes Django's reverse() emit
# stage-prefixed URLs so redirects and static URLs stay correct.
if _stage and _stage not in {"$default", "default"} and os.getenv(
    "API_GATEWAY_STRIP_STAGE", "0"
).lower() not in {"1", "true", "yes", "on"}:
    FORCE_SCRIPT_NAME = f"/{_stage}"
    STATIC_URL = f"/{_stage}/static/"

# ---------------------------------------------------------------------------
# Database.
#
# The hybrid model (serverless/hybrid/) serves the app's own data straight from
# S3, with the browser's AllezORM mirror as the query tier -- no RDS, so no
# per-hour bill and nothing to keep warm.
#
# Django still needs *a* database for the pieces of contrib that assume one
# (sessions, admin, auth tables used by the management commands). With
# HYBRID_DB=1 that is a scratch SQLite file in /tmp: it is created per sandbox,
# used only by those subsystems, and never holds application data worth
# persisting. Set HYBRID_DB=0 to fall back to Postgres.
# ---------------------------------------------------------------------------
HYBRID_DB = os.getenv("HYBRID_DB", "1").lower() in {"1", "true", "yes", "on"}
if HYBRID_DB:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": os.getenv("DJANGO_SQLITE_PATH", "/tmp/django-scratch.db"),
            "OPTIONS": {"timeout": 20},
        }
    }
    # Sessions live in the signed cookie rather than the scratch DB: a DB
    # session would vanish with the sandbox and log the user out at random.
    SESSION_ENGINE = "django.contrib.sessions.backends.signed_cookies"

# ---------------------------------------------------------------------------
# Connection handling -- no cross-invocation pooling.
# ---------------------------------------------------------------------------
DATABASES["default"]["CONN_MAX_AGE"] = 0  # noqa: F405
DATABASES["default"]["CONN_HEALTH_CHECKS"] = False  # noqa: F405
# A cold Lambda behind a busy RDS Proxy should fail fast rather than burn the
# whole 30s API Gateway budget on a connect.
DATABASES["default"].setdefault("OPTIONS", {})  # noqa: F405
if DATABASES["default"]["ENGINE"].endswith("postgresql"):  # noqa: F405
    DATABASES["default"]["OPTIONS"]["connect_timeout"] = int(  # noqa: F405
        os.getenv("DB_CONNECT_TIMEOUT", "5")
    )

# ---------------------------------------------------------------------------
# Media on S3 -- the container filesystem is read-only apart from /tmp.
# ---------------------------------------------------------------------------
AWS_STORAGE_BUCKET_NAME = os.getenv("MEDIA_BUCKET", "")
AWS_S3_ENDPOINT_URL = os.getenv("AWS_S3_ENDPOINT_URL") or None  # LocalStack/MinIO
AWS_S3_REGION_NAME = os.getenv("AWS_REGION", "us-east-1")
AWS_DEFAULT_ACL = None
AWS_S3_FILE_OVERWRITE = False
AWS_QUERYSTRING_AUTH = True
AWS_S3_ADDRESSING_STYLE = os.getenv("AWS_S3_ADDRESSING_STYLE", "path")

STORAGES = dict(STORAGES)  # noqa: F405
if AWS_STORAGE_BUCKET_NAME:
    STORAGES["default"] = {"BACKEND": "storages.backends.s3.S3Storage"}
else:
    # No bucket configured: keep uploads in /tmp so a write does not explode,
    # but they are explicitly ephemeral.
    MEDIA_ROOT = "/tmp/media"

# Static files stay in the deployment bundle. Use the non-manifest storage so a
# stale reference logs instead of raising during a request.
STORAGES["staticfiles"] = {
    "BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"
}
WHITENOISE_USE_FINDERS = False
WHITENOISE_AUTOREFRESH = False

# ---------------------------------------------------------------------------
# Sessions / cache.  InMemory anything is per-sandbox and therefore useless
# across invocations -- a user's session would vanish on the next request if it
# landed on a different container.
# ---------------------------------------------------------------------------
# DB-backed sessions require a durable database. Under the hybrid model the
# only DB is the per-sandbox scratch file, so cookie sessions are set above and
# this branch applies solely to the Postgres fallback.
if not HYBRID_DB:
    SESSION_ENGINE = "django.contrib.sessions.backends.db"

_redis_url = os.getenv("REDIS_URL", "").strip()
if _redis_url:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": _redis_url,
        }
    }
else:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "lambda-local",
        }
    }

# ---------------------------------------------------------------------------
# Logging -- CloudWatch captures stdout. No file handlers (read-only FS).
# ---------------------------------------------------------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "lambda": {"format": "%(levelname)s %(name)s %(message)s"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "lambda"},
    },
    "root": {
        "handlers": ["console"],
        "level": os.getenv("DJANGO_LOG_LEVEL", "INFO"),
    },
    "loggers": {
        # Every cold start otherwise re-emits the full autoreload/static noise.
        "django.utils.autoreload": {"level": "ERROR", "propagate": False},
    },
}
