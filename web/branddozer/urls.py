from __future__ import annotations

from django.urls import path

from . import views

app_name = "branddozer"

urlpatterns = [
    path("projects/", views.ProjectListView.as_view(), name="projects"),
    path("projects/roots/", views.ProjectRootListView.as_view(), name="project-roots"),
    path("projects/github/account/", views.ProjectGitHubAccountView.as_view(), name="project-github-account"),
    path("projects/github/repos/", views.ProjectGitHubRepoListView.as_view(), name="project-github-repos"),
    path("projects/github/branches/", views.ProjectGitHubBranchListView.as_view(), name="project-github-branches"),
    path("projects/import/github/", views.ProjectGitHubImportView.as_view(), name="project-github-import"),
    path("projects/<str:project_id>/", views.ProjectDetailView.as_view(), name="project-detail"),
    path("projects/<str:project_id>/start/", views.ProjectStartView.as_view(), name="project-start"),
    path("projects/<str:project_id>/stop/", views.ProjectStopView.as_view(), name="project-stop"),
    path("projects/<str:project_id>/logs/", views.ProjectLogView.as_view(), name="project-logs"),
    path("projects/<str:project_id>/interjections/", views.ProjectInterjectionSuggestView.as_view(), name="project-interjections"),
]
