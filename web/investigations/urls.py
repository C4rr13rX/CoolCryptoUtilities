from __future__ import annotations

from django.urls import path

from . import views

app_name = "investigations"

urlpatterns = [
    path("projects/", views.ProjectListView.as_view(), name="projects"),
    path("projects/<int:project_id>/", views.ProjectDetailView.as_view(), name="project-detail"),
    path("projects/<int:project_id>/targets/", views.TargetListView.as_view(), name="targets"),
    path("targets/<int:target_id>/crawl/", views.TargetCrawlView.as_view(), name="target-crawl"),
    path("projects/<int:project_id>/evidence/", views.EvidenceListView.as_view(), name="evidence"),
    path("projects/<int:project_id>/articles/", views.ArticleListView.as_view(), name="articles"),
    path("articles/<int:article_id>/", views.ArticleDetailView.as_view(), name="article-detail"),
    path("projects/<int:project_id>/entities/", views.EntityListView.as_view(), name="entities"),
    path("projects/<int:project_id>/relations/", views.RelationListView.as_view(), name="relations"),
]
