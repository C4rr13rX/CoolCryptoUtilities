"""
Android settings overlay.

Inherits the full project settings and changes only what an app sandbox makes
impossible.  ``settings.py`` stays the single source of truth for apps, URLs,
DRF and middleware -- the same discipline as ``settings_lambda.py``.

Activated by ``android_bootstrap.py`` via
``DJANGO_SETTINGS_MODULE=coolcrypto_dashboard.settings_android``.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Neutralise boot-time background work BEFORE the base settings import the app
# registry. core/apps.py reads these at AppConfig.ready().
#
# On Android these workloads are Android services (TradingService) with their
# own notifications and lifecycle. As daemon threads inside the HTTP process
# they would be invisible to the OS, unstoppable by the user, and suspended
# mid-work by Doze.
# ---------------------------------------------------------------------------
os.environ["GUARDIAN_AUTO_DISABLED"] = "1"
os.environ["CRON_AUTO_DISABLED"] = "1"
os.environ["PRODUCTION_AUTO_DISABLED"] = "1"
os.environ["WIZARD_DISABLE_REFRESHER"] = "1"
os.environ["WIZARD_BRAIN_FEEDER_ENABLED"] = "0"
os.environ.pop("WAITRESS_PORT", None)

from .settings import *  # noqa: E402,F401,F403

IS_ANDROID = True

# ---------------------------------------------------------------------------
# Filesystem. Only the app sandbox is writable; the APK itself is read-only.
# ---------------------------------------------------------------------------
APP_ROOT = Path(os.getenv("WRITABLE_ROOT", "/data/local/tmp"))

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.getenv("DJANGO_SQLITE_PATH", str(APP_ROOT / "storage" / "app.db")),
        "OPTIONS": {
            "timeout": 30,
            # WAL lets the Django process and the trading service read and
            # write concurrently instead of serialising on a single lock,
            # which on a phone shows up as a visibly stalled UI.
            "init_command": "PRAGMA journal_mode=WAL;",
        },
    }
}

MEDIA_ROOT = APP_ROOT / "storage" / "media"
STATIC_ROOT = APP_ROOT / "static"

# ---------------------------------------------------------------------------
# Serving. The WebView talks to 127.0.0.1, so this is same-origin and the Vue
# GUI's `baseURL: '/api'` works with no frontend change at all.
# ---------------------------------------------------------------------------
DEBUG = False
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "10.0.2.2"]  # 10.0.2.2 = emulator host
CSRF_TRUSTED_ORIGINS = [
    "http://127.0.0.1:8765",
    "http://localhost:8765",
]

# Loopback only -- no TLS to redirect to, and enabling this would make every
# request fail before it reached a view.
SECURE_SSL_REDIRECT = False
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False

# The app is the only client; CORS_ALLOW_ALL_ORIGINS from the base settings
# would be a needless hole on a device that also browses the web.
CORS_ALLOW_ALL_ORIGINS = False
CORS_ALLOWED_ORIGINS = ["http://127.0.0.1:8765", "http://localhost:8765"]
CORS_ALLOWED_ORIGIN_REGEXES = []

# ---------------------------------------------------------------------------
# Static files. WhiteNoise serves the built Vue bundle straight from the APK
# assets, so there is no second web server to run.
# ---------------------------------------------------------------------------
STATICFILES_DIRS = [
    d for d in [
        BASE_DIR / "static",          # noqa: F405
        BASE_DIR / "frontend" / "dist",  # noqa: F405
    ] if Path(d).exists()
]
STORAGES = dict(STORAGES)  # noqa: F405
STORAGES["staticfiles"] = {
    # Non-manifest: a hashed-asset miss should log, not 500 the page.
    "BACKEND": "whitenoise.storage.CompressedStaticFilesStorage",
}
WHITENOISE_USE_FINDERS = True     # serve without a collectstatic pass
WHITENOISE_AUTOREFRESH = False

# ---------------------------------------------------------------------------
# Channels. There is no Redis on a phone, and the in-memory layer is correct
# here precisely because there is exactly one process.
# ---------------------------------------------------------------------------
CHANNEL_LAYERS = {
    "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}
}

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "android-local",
    }
}

# ---------------------------------------------------------------------------
# Logging to stdout, which Chaquopy forwards to logcat.
# ---------------------------------------------------------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"],
             "level": os.getenv("DJANGO_LOG_LEVEL", "INFO")},
    "loggers": {
        "django.utils.autoreload": {"level": "ERROR", "propagate": False},
    },
}

# ---------------------------------------------------------------------------
# Wizard node: the Rust binary running in WizardNodeService on this device.
# ---------------------------------------------------------------------------
WIZARD_NODE_ENDPOINT = os.getenv("WIZARD_NODE_ENDPOINT", "http://127.0.0.1:8090")
