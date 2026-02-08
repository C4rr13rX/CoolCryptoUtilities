from __future__ import annotations

from django.urls import path

from . import views

app_name = "datalab"

urlpatterns = [
    path("datasets/", views.DatasetListView.as_view(), name="datasets"),
    path("run/", views.RunJobView.as_view(), name="run"),
    path("status/", views.JobStatusView.as_view(), name="status"),
    path("news/", views.NewsFetchView.as_view(), name="news"),
    path("news/sources/", views.NewsSourceListView.as_view(), name="news-sources"),
    path("news/sources/<int:source_id>/test/", views.NewsSourceTestView.as_view(), name="news-source-test"),
    path("news/sources/<int:source_id>/run/", views.NewsSourceRunView.as_view(), name="news-source-run"),
    path("signals/", views.SignalListView.as_view(), name="signals"),
    path("watchlists/", views.WatchlistView.as_view(), name="watchlists"),
]
