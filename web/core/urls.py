from __future__ import annotations

from django.urls import path

from . import views

app_name = "core"

urlpatterns = [
    path("", views.LandingView.as_view(), name="index"),
    path("dashboard/", views.DashboardView.as_view(), name="dashboard"),
    path("dashboard", views.DashboardView.as_view()),
    path("organism/", views.OrganismView.as_view(), name="organism"),
    path("organism", views.OrganismView.as_view()),
    path("streams/", views.StreamsView.as_view(), name="streams"),
    path("streams", views.StreamsView.as_view()),
    path("telemetry/", views.TelemetryView.as_view(), name="telemetry"),
    path("telemetry", views.TelemetryView.as_view()),
    path("wallet/", views.WalletPageView.as_view(), name="wallet"),
    path("wallet", views.WalletPageView.as_view()),
    path("pipeline/", views.PipelinePageView.as_view(), name="pipeline"),
    path("pipeline", views.PipelinePageView.as_view()),
    path("datalab/", views.DataLabPageView.as_view(), name="datalab"),
    path("datalab", views.DataLabPageView.as_view()),
    path("lab/", views.ModelLabPageView.as_view(), name="lab"),
    path("lab", views.ModelLabPageView.as_view()),
    path("guardian/", views.GuardianPageView.as_view(), name="guardian"),
    path("guardian", views.GuardianPageView.as_view()),
    path("settings/", views.SecureSettingsPageView.as_view(), name="settings"),
    path("settings", views.SecureSettingsPageView.as_view()),
    path("integrations/", views.IntegrationsPageView.as_view(), name="integrations"),
    path("integrations", views.IntegrationsPageView.as_view()),
    path("advisories/", views.AdvisoriesPageView.as_view(), name="advisories"),
    path("advisories", views.AdvisoriesPageView.as_view()),
    path("<slug:route>/", views.SpaRouteView.as_view(), name="spa-route"),
    path("<slug:route>", views.SpaRouteView.as_view()),
]
