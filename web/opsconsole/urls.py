from __future__ import annotations

from django.urls import path

from . import views

app_name = "opsconsole"

urlpatterns = [
    path("start/", views.StartProcessView.as_view(), name="start"),
    path("stop/", views.StopProcessView.as_view(), name="stop"),
    path("status/", views.ProcessStatusView.as_view(), name="status"),
    path("logs/", views.TailLogsView.as_view(), name="logs"),
    path("input/", views.SendInputView.as_view(), name="input"),
]
