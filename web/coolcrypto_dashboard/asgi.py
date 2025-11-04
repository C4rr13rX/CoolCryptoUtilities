from __future__ import annotations

import os

from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from django.urls import path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")

django_asgi_app = get_asgi_application()

try:
    from opsconsole.routing import websocket_urlpatterns as console_ws  # type: ignore
except Exception:
    console_ws = []

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": URLRouter(console_ws),
    }
)
