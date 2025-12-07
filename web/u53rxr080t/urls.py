from __future__ import annotations

from django.urls import path

from . import views

app_name = "u53rxr080t"

urlpatterns = [
    path("heartbeat/", views.HeartbeatView.as_view(), name="heartbeat"),
    path("agents/", views.AgentsView.as_view(), name="agents"),
    path("tasks/", views.TasksView.as_view(), name="tasks"),
    path("tasks/next/", views.TaskNextView.as_view(), name="task-next"),
    path("tasks/<uuid:task_id>/", views.TaskUpdateView.as_view(), name="task-update"),
    path("findings/", views.FindingView.as_view(), name="findings"),
    path("suggest/", views.SuggestView.as_view(), name="suggest"),
    path("queue/", views.GuardianQueueView.as_view(), name="guardian-queue"),
]
