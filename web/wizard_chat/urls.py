from django.urls import path
from . import views
from . import agent_views

app_name = "wizard_chat"

urlpatterns = [
    path("message/", views.WizardChatMessageView.as_view(), name="message"),
    path("message", views.WizardChatMessageView.as_view()),
    path("upload/", views.WizardChatUploadView.as_view(), name="upload"),
    path("upload", views.WizardChatUploadView.as_view()),
    path("train/", views.WizardChatTrainView.as_view(), name="train"),
    path("train", views.WizardChatTrainView.as_view()),
    path("status/", views.WizardChatStatusView.as_view(), name="status"),
    path("status", views.WizardChatStatusView.as_view()),
    path("pools/", views.WizardChatPoolsView.as_view(), name="pools"),
    path("pools", views.WizardChatPoolsView.as_view()),
    # Rolling-context snapshot endpoint — UI uses this to show the
    # "active topic" chip and to inspect what the backend sees.
    path("session/", views.WizardChatSessionView.as_view(), name="session"),
    path("session",  views.WizardChatSessionView.as_view()),
    # Live training feed — tails the training_events.jsonl from the
    # W1z4rD node + bundles the current brain snapshot.  Polled by the
    # bottom-of-chat training panel.
    path("training/live/", views.WizardChatTrainingLiveView.as_view(),
         name="training_live"),
    path("training/live",  views.WizardChatTrainingLiveView.as_view()),
    # ── Agent mode (localhost-only; uses C0d3rV2 with shell tools) ─────────
    path("agent/", agent_views.WizardChatAgentView.as_view(), name="agent"),
    path("agent", agent_views.WizardChatAgentView.as_view()),
    path("agent/info/", agent_views.WizardChatAgentInfoView.as_view(),
         name="agent_info"),
    path("agent/info", agent_views.WizardChatAgentInfoView.as_view()),
]
