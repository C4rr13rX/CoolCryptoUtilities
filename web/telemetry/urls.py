from __future__ import annotations

from django.urls import path

from . import readiness_views, views

app_name = "telemetry"

urlpatterns = [
    path("metrics/", views.MetricsListView.as_view(), name="metrics"),
    path("feedback/", views.FeedbackListView.as_view(), name="feedback"),
    path("trades/", views.TradeLogView.as_view(), name="trades"),
    path("advisories/", views.AdvisoryListView.as_view(), name="advisories"),
    path("dashboard/", views.DashboardSummaryView.as_view(), name="dashboard"),
    path("pipeline/readiness/", views.PipelineReadinessView.as_view(), name="pipeline-readiness"),
    path("bus/schedule/", views.BusScheduleView.as_view(), name="bus-schedule"),
    path("organism/latest/", views.OrganismLatestView.as_view(), name="organism-latest"),
    path("organism/history/", views.OrganismHistoryView.as_view(), name="organism-history"),
    path("organism/settings/", views.OrganismSettingsView.as_view(), name="organism-settings"),
    # "Can it trade on its own yet?" -- one answer, from the ghost ledger.
    path("readiness/", readiness_views.TradingReadinessView.as_view(),
         name="trading-readiness"),
    path("readiness", readiness_views.TradingReadinessView.as_view()),
]
