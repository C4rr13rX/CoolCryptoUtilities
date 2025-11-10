from __future__ import annotations

from django.urls import path

from . import views

app_name = "guardianpanel"

urlpatterns = [
    path("settings/", views.GuardianSettingsView.as_view(), name="settings"),
    path("run/", views.GuardianRunView.as_view(), name="run"),
    path("logs/", views.GuardianLogView.as_view(), name="logs"),
]
