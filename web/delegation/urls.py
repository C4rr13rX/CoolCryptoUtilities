from __future__ import annotations

from django.urls import path

from . import views

app_name = "delegation"

urlpatterns = [
    path("hosts/", views.DelegationHostListView.as_view(), name="host-list"),
    path("hosts/<int:pk>/", views.DelegationHostDetailView.as_view(), name="host-detail"),
    path("hosts/<int:pk>/regenerate-token/", views.DelegationHostRegenerateTokenView.as_view(), name="host-regen-token"),
    path("hosts/<int:pk>/pair/", views.DelegationHostPairView.as_view(), name="host-pair"),
    path("tasks/", views.DelegatedTaskListView.as_view(), name="task-list"),
    path("logs/", views.DelegationLogListView.as_view(), name="log-list"),
    path("profiles/", views.TaskResourceProfileListView.as_view(), name="profile-list"),
    path("summary/", views.DelegationSummaryView.as_view(), name="summary"),
]
