from __future__ import annotations

from django.urls import path

from . import views

app_name = "tradingagent"

urlpatterns = [
    path("status/", views.StatusView.as_view(), name="status"),
    path("config/", views.ConfigView.as_view(), name="config"),
    path("promote/", views.PromoteView.as_view(), name="promote"),
    path("runs/", views.RunListView.as_view(), name="runs"),
    path("run-now/", views.RunNowView.as_view(), name="run-now"),
    path("constraints/", views.ConstraintListView.as_view(), name="constraints"),
    path("experiments/", views.ExperimentListView.as_view(), name="experiments"),
    path("recoveries/", views.LossRecoveryListView.as_view(), name="recoveries"),
    path("math/", views.MathAuditView.as_view(), name="math"),
]
