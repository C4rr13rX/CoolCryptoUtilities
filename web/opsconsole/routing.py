from __future__ import annotations

from django.urls import path

from .consumers import AppStateConsumer, ConsoleLogConsumer, WalletStateConsumer

websocket_urlpatterns = [
    path("ws/console/logs/", ConsoleLogConsumer.as_asgi()),
    path("ws/wallet/state/", WalletStateConsumer.as_asgi()),
    path("ws/app/state/", AppStateConsumer.as_asgi()),
]
