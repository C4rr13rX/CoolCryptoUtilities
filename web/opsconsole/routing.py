from __future__ import annotations

from django.urls import path

from .consumers import ConsoleLogConsumer

websocket_urlpatterns = [
    path("ws/console/logs/", ConsoleLogConsumer.as_asgi()),
]
