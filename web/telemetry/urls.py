from __future__ import annotations

from django.urls import path

from . import population_views, readiness_views, views

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
    # The WHOLE population, not just the ledger's rows. `readiness/` above
    # lists the 37 strategies that have a ledger entry; this lists all 41,
    # including the four that have never closed a ghost round trip and the
    # ones that were tried and permanently barred.
    path("strategies/population/",
         population_views.StrategyPopulationView.as_view(),
         name="strategy-population"),
]
