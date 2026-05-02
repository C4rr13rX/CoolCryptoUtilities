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
    # ── Agent mode (localhost-only; uses C0d3rV2 with shell tools) ─────────
    path("agent/", agent_views.WizardChatAgentView.as_view(), name="agent"),
    path("agent", agent_views.WizardChatAgentView.as_view()),
    path("agent/info/", agent_views.WizardChatAgentInfoView.as_view(),
         name="agent_info"),
    path("agent/info", agent_views.WizardChatAgentInfoView.as_view()),
]
