from __future__ import annotations

from django.urls import path

from . import views

app_name = "integrations"

urlpatterns = [
    path("keys/", views.IntegrationListView.as_view(), name="list"),
    path("keys/<str:name>/", views.IntegrationDetailView.as_view(), name="detail"),
    path("keys/<str:name>/test/", views.IntegrationTestView.as_view(), name="test"),
]
