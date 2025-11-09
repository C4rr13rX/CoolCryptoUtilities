from __future__ import annotations

from django.urls import path

from . import views

app_name = "securevault"

urlpatterns = [
    path("settings/", views.SecureSettingListCreateView.as_view(), name="settings-list"),
    path("settings/<int:pk>/", views.SecureSettingDetailView.as_view(), name="settings-detail"),
    path("settings/import/", views.SecureSettingBulkImportView.as_view(), name="settings-import"),
]
