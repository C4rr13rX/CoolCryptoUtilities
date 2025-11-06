from __future__ import annotations

from django.urls import path

from . import views

app_name = "telemetry"

urlpatterns = [
    path("metrics/", views.MetricsListView.as_view(), name="metrics"),
    path("feedback/", views.FeedbackListView.as_view(), name="feedback"),
    path("trades/", views.TradeLogView.as_view(), name="trades"),
    path("advisories/", views.AdvisoryListView.as_view(), name="advisories"),
    path("dashboard/", views.DashboardSummaryView.as_view(), name="dashboard"),
    path("organism/latest/", views.OrganismLatestView.as_view(), name="organism-latest"),
    path("organism/history/", views.OrganismHistoryView.as_view(), name="organism-history"),
]
