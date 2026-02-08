from __future__ import annotations

from django.urls import path

from . import views

urlpatterns = [
    path("status/", views.CronStatusView.as_view(), name="cron-status"),
    path("status", views.CronStatusView.as_view()),
    path("settings/", views.CronSettingsView.as_view(), name="cron-settings"),
    path("settings", views.CronSettingsView.as_view()),
    path("run/", views.CronRunView.as_view(), name="cron-run"),
    path("run", views.CronRunView.as_view()),
]
