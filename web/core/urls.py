from __future__ import annotations

from django.urls import path

from . import views

app_name = "core"

urlpatterns = [
    path("", views.LandingView.as_view(), name="index"),
    path("dashboard/", views.DashboardView.as_view(), name="dashboard"),
    path("organism/", views.OrganismView.as_view(), name="organism"),
    path("streams/", views.StreamsView.as_view(), name="streams"),
    path("telemetry/", views.TelemetryView.as_view(), name="telemetry"),
    path("console/", views.ConsoleView.as_view(), name="console"),
    path("pipeline/", views.PipelinePageView.as_view(), name="pipeline"),
    path("datalab/", views.DataLabPageView.as_view(), name="datalab"),
    path("lab/", views.ModelLabPageView.as_view(), name="lab"),
    path("guardian/", views.GuardianPageView.as_view(), name="guardian"),
    path("settings/", views.SecureSettingsPageView.as_view(), name="settings"),
]
