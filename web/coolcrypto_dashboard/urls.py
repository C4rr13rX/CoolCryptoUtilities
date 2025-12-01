from __future__ import annotations

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/streams/", include("streams.urls")),
    path("api/telemetry/", include("telemetry.urls")),
    path("api/console/", include("opsconsole.urls")),
    path("api/datalab/", include("datalab.urls")),
    path("api/lab/", include("lab.urls")),
    path("api/guardian/", include("guardianpanel.urls")),
    path("api/secure/", include("securevault.urls")),
    path("api/wallet/", include("walletpanel.urls")),
    path("api/integrations/", include("integrations.urls")),
    path("api/branddozer/", include("branddozer.urls")),
    path("", include("core.urls")),
]

handler500 = "core.views.guardian_failure_response"
