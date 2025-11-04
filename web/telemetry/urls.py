from __future__ import annotations

from django.urls import path

from . import views

app_name = "telemetry"

urlpatterns = [
    path("metrics/", views.MetricsListView.as_view(), name="metrics"),
    path("feedback/", views.FeedbackListView.as_view(), name="feedback"),
    path("trades/", views.TradeLogView.as_view(), name="trades"),
    path("dashboard/", views.DashboardSummaryView.as_view(), name="dashboard"),
]
