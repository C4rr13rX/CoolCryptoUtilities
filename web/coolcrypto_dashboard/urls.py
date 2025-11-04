from __future__ import annotations

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/streams/", include("streams.urls")),
    path("api/telemetry/", include("telemetry.urls")),
    path("api/console/", include("opsconsole.urls")),
    path("", include("core.urls")),
]
