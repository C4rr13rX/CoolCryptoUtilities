"""
API Gateway -> Django HTTP handler.

Mangum adapts the ASGI application to Lambda's event/context calling
convention.  We deliberately wrap the *ASGI* app rather than WSGI so the same
handler can serve the async views and DRF endpoints the project already has,
and so streaming responses degrade gracefully instead of erroring.

The module-level work here (Django setup, URL loading, app registry
population) runs once per cold start and is reused by every subsequent
invocation of that sandbox.  Anything expensive added below this line is paid
on every cold start, so keep it lean.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# The Lambda bundle layout is:  /var/task/{serverless,web,services,...}
# Locally the repo root plays the same role. Both need `web/` on sys.path so
# `coolcrypto_dashboard` and the app packages import as top-level modules.
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "web")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings_lambda"
)

logger = logging.getLogger(__name__)

# Redirect the bundle's write targets to /tmp before ANY services.* module
# is imported -- several create directories at import time and /var/task is
# read-only. Must run before django.setup() pulls in the app registry.
from serverless.bootstrap import prepare_writable_dirs  # noqa: E402

prepare_writable_dirs(ROOT)

import django  # noqa: E402

django.setup()

from django.core.asgi import get_asgi_application  # noqa: E402
from mangum import Mangum  # noqa: E402

# get_asgi_application() rather than importing coolcrypto_dashboard.asgi:
# that module wraps the app in a ProtocolTypeRouter for Channels, whose
# "websocket" branch is meaningless under API Gateway's HTTP API. WebSocket
# traffic is handled by a separate function.
_django_app = get_asgi_application()

# lifespan="off": Lambda has no long-lived process for ASGI startup/shutdown
# events, and Channels' lifespan handler would otherwise hang the first
# invocation waiting for a startup ack that never completes.
handler = Mangum(
    _django_app,
    lifespan="off",
    api_gateway_base_path=os.getenv("API_GATEWAY_BASE_PATH", "/"),
)


def lambda_handler(event, context):
    """Entry point referenced by the function configuration."""
    return handler(event, context)
