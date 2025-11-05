from __future__ import annotations

from django.urls import path

from . import views

app_name = "core"

urlpatterns = [
    path("", views.LandingView.as_view(), name="index"),
    path("dashboard/", views.DashboardView.as_view(), name="dashboard"),
    path("streams/", views.StreamsView.as_view(), name="streams"),
    path("telemetry/", views.TelemetryView.as_view(), name="telemetry"),
    path("console/", views.ConsoleView.as_view(), name="console"),
]
