from __future__ import annotations

from django.urls import path

from . import views

app_name = "lab"

urlpatterns = [
    path("files/", views.LabFilesView.as_view(), name="files"),
    path("status/", views.LabStatusView.as_view(), name="status"),
    path("run/", views.LabStartView.as_view(), name="run"),
    path("news/", views.LabNewsView.as_view(), name="news"),
    path("preview/", views.LabPreviewView.as_view(), name="preview"),
]
