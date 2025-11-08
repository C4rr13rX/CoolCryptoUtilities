from __future__ import annotations

from django.urls import path

from . import views

app_name = "datalab"

urlpatterns = [
    path("datasets/", views.DatasetListView.as_view(), name="datasets"),
    path("run/", views.RunJobView.as_view(), name="run"),
    path("status/", views.JobStatusView.as_view(), name="status"),
    path("news/", views.NewsFetchView.as_view(), name="news"),
    path("signals/", views.SignalListView.as_view(), name="signals"),
    path("watchlists/", views.WatchlistView.as_view(), name="watchlists"),
]
