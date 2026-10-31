from __future__ import annotations

from django.urls import path

from .views import (
    EdgeSearchControlView,
    EdgeSearchExportSpreadsheetView,
    EdgeSearchFindingsView,
    EdgeSearchPromoteView,
    EdgeSearchStatusView,
)

app_name = "edgeforge"

urlpatterns = [
    path("status/", EdgeSearchStatusView.as_view(), name="status"),
    path("findings/", EdgeSearchFindingsView.as_view(), name="findings"),
    path("control/", EdgeSearchControlView.as_view(), name="control"),
    path("promote/<int:venture_id>/", EdgeSearchPromoteView.as_view(), name="promote"),
    path("export/", EdgeSearchExportSpreadsheetView.as_view(), name="export"),
]
