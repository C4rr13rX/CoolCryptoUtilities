from __future__ import annotations

from django.urls import path

from . import views
from . import delivery_views
from . import research_views

app_name = "branddozer"

urlpatterns = [
    path("research/papers/", research_views.ResearchPaperListView.as_view(), name="research-papers"),
    path("research/papers/<str:paper_id>/", research_views.ResearchPaperDetailView.as_view(), name="research-paper"),
    path("research/papers/<str:paper_id>/download/", research_views.ResearchPaperDownloadView.as_view(), name="research-paper-download"),
    path("projects/", views.ProjectListView.as_view(), name="projects"),
    path("projects/roots/", views.ProjectRootListView.as_view(), name="project-roots"),
    path("projects/github/account/", views.ProjectGitHubAccountView.as_view(), name="project-github-account"),
    path("projects/github/accounts/", views.ProjectGitHubAccountsView.as_view(), name="project-github-accounts"),
    path("projects/github/accounts/active/", views.ProjectGitHubActiveAccountView.as_view(), name="project-github-active"),
    path("projects/github/repos/", views.ProjectGitHubRepoListView.as_view(), name="project-github-repos"),
    path("projects/github/branches/", views.ProjectGitHubBranchListView.as_view(), name="project-github-branches"),
    path("projects/import/github/", views.ProjectGitHubImportView.as_view(), name="project-github-import"),
    path("projects/import/github/status/<str:job_id>/", views.ProjectGitHubImportStatusView.as_view(), name="project-github-import-status"),
    path("projects/<str:project_id>/publish/", views.ProjectGitHubPublishView.as_view(), name="project-github-publish"),
    path("projects/publish/status/<str:job_id>/", views.ProjectGitHubPublishStatusView.as_view(), name="project-github-publish-status"),
    path("projects/interjections/preview/", views.ProjectInterjectionPreviewView.as_view(), name="project-interjections-preview"),
    path("delivery/runs/", delivery_views.DeliveryRunListView.as_view(), name="delivery-runs"),
    path("delivery/runs/<str:run_id>/", delivery_views.DeliveryRunDetailView.as_view(), name="delivery-run"),
    path("delivery/runs/<str:run_id>/stop/", delivery_views.DeliveryRunStopView.as_view(), name="delivery-run-stop"),
    path("delivery/runs/<str:run_id>/backlog/", delivery_views.DeliveryRunBacklogView.as_view(), name="delivery-backlog"),
    path("delivery/backlog/<str:item_id>/", delivery_views.DeliveryBacklogItemView.as_view(), name="delivery-backlog-item"),
    path("delivery/runs/<str:run_id>/gates/", delivery_views.DeliveryRunGateView.as_view(), name="delivery-gates"),
    path("delivery/runs/<str:run_id>/sessions/", delivery_views.DeliveryRunSessionView.as_view(), name="delivery-sessions"),
    path("delivery/sessions/<str:session_id>/logs/", delivery_views.DeliverySessionLogView.as_view(), name="delivery-session-logs"),
    path("delivery/runs/<str:run_id>/artifacts/", delivery_views.DeliveryRunArtifactView.as_view(), name="delivery-artifacts"),
    path("delivery/artifacts/<str:artifact_id>/file/", delivery_views.DeliveryArtifactFileView.as_view(), name="delivery-artifact-file"),
    path("delivery/runs/<str:run_id>/ui-capture/", delivery_views.DeliveryRunUICaptureView.as_view(), name="delivery-ui-capture"),
    path("delivery/runs/<str:run_id>/governance/", delivery_views.DeliveryRunGovernanceView.as_view(), name="delivery-governance"),
    path("delivery/runs/<str:run_id>/sprints/", delivery_views.DeliveryRunSprintView.as_view(), name="delivery-sprints"),
    path("delivery/runs/<str:run_id>/accept/", delivery_views.DeliveryRunAcceptView.as_view(), name="delivery-accept"),
    path("projects/<str:project_id>/", views.ProjectDetailView.as_view(), name="project-detail"),
    path("projects/<str:project_id>/start/", views.ProjectStartView.as_view(), name="project-start"),
    path("projects/<str:project_id>/lifecycle/", views.ProjectLifecycleView.as_view(), name="project-lifecycle"),
    path("projects/<str:project_id>/preview/", views.ProjectPreviewView.as_view(), name="project-preview"),
    path("projects/<str:project_id>/stop/", views.ProjectStopView.as_view(), name="project-stop"),
    path("projects/<str:project_id>/logs/", views.ProjectLogView.as_view(), name="project-logs"),
    path("projects/<str:project_id>/interjections/", views.ProjectInterjectionSuggestView.as_view(), name="project-interjections"),
    path("product-loop/status/", views.ProductLoopStatusView.as_view(), name="product-loop-status"),
    path("product-loop/cycle/", views.ProductLoopCycleView.as_view(), name="product-loop-cycle"),
    path("atf/class-benchmark/", views.AtfClassBenchmarkView.as_view(), name="atf-class-benchmark"),
]
