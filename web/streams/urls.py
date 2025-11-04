from __future__ import annotations

from django.urls import path

from . import views

app_name = "streams"

urlpatterns = [
    path("latest/", views.LatestSampleView.as_view(), name="latest"),
    path("recent/", views.RecentSamplesView.as_view(), name="recent"),
]
