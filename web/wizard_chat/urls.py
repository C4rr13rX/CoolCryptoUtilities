from django.urls import path
from . import node_update_views, views
from . import agent_views
from . import learning_views
from . import settling_views
from . import vehicle_companion_views
from . import wbrain_views
from .self_design_views import WizardSelfDesignView

from .cluster_views import WizardClusterView

app_name = "wizard_chat"

urlpatterns = [
    path("cluster/", WizardClusterView.as_view(), name="cluster"),
    path("self-design/", WizardSelfDesignView.as_view(), name="self_design"),
    path("learning/", learning_views.WizardLearningView.as_view(), name="learning"),
    path("settling/", settling_views.WizardSettlingView.as_view(), name="settling"),
    path("settling/run/", settling_views.WizardSettlingRunView.as_view(), name="settling_run"),
    path("settling/run", settling_views.WizardSettlingRunView.as_view()),
    path("cognitive-state/", settling_views.WizardCognitiveStateView.as_view(), name="cognitive_state"),
    path("cognitive-state", settling_views.WizardCognitiveStateView.as_view()),
    path("robot-world/", settling_views.WizardRobotWorldBridgeView.as_view(), name="robot_world"),
    path("robot-world", settling_views.WizardRobotWorldBridgeView.as_view()),
    path("empirical/", settling_views.WizardEmpiricalCurriculumView.as_view(), name="empirical"),
    path("empirical/train/", settling_views.WizardEmpiricalTrainView.as_view(), name="empirical_train"),
    path("empirical/train", settling_views.WizardEmpiricalTrainView.as_view()),
    path("mesh/generate/", settling_views.WizardProceduralMeshView.as_view(), name="mesh_generate"),
    path("mesh/generate", settling_views.WizardProceduralMeshView.as_view()),
    path("deduce/", settling_views.WizardDeductiveReasoningView.as_view(), name="deduce"),
    path("deduce", settling_views.WizardDeductiveReasoningView.as_view()),
    path("blueprint/export/", settling_views.WizardBlueprintExportView.as_view(), name="blueprint_export"),
    path("blueprint/export", settling_views.WizardBlueprintExportView.as_view()),
    path("kuzu-workbench/", settling_views.WizardKuzuWorkbenchView.as_view(), name="kuzu_workbench"),
    path("kuzu-workbench", settling_views.WizardKuzuWorkbenchView.as_view()),
    path("message/", views.WizardChatMessageView.as_view(), name="message"),
    path("message", views.WizardChatMessageView.as_view()),
    path("upload/", views.WizardChatUploadView.as_view(), name="upload"),
    path("upload", views.WizardChatUploadView.as_view()),
    path("train/", views.WizardChatTrainView.as_view(), name="train"),
    path("train", views.WizardChatTrainView.as_view()),
    path("anchored-memories/", views.WizardAnchoredMemoriesView.as_view(), name="anchored_memories"),
    path("anchored-memories", views.WizardAnchoredMemoriesView.as_view()),
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
    # Brain visualization: topology sample (nodes + signed edges) and
    # activity stream (fire/train pulses).  Lightweight thin proxies
    # to the node's /multi_pool/topology and /multi_pool/activity so
    # the SPA never talks to the node directly.
    path("topology/", views.WizardChatTopologyView.as_view(), name="topology"),
    path("topology",  views.WizardChatTopologyView.as_view()),
    path("activity/", views.WizardChatActivityView.as_view(), name="activity"),
    path("activity",  views.WizardChatActivityView.as_view()),
    # ── Agent mode (localhost-only; uses C0d3rV2 with shell tools) ─────────
    path("agent/", agent_views.WizardChatAgentView.as_view(), name="agent"),
    path("agent", agent_views.WizardChatAgentView.as_view()),
    path("agent/info/", agent_views.WizardChatAgentInfoView.as_view(),
         name="agent_info"),
    path("agent/info", agent_views.WizardChatAgentInfoView.as_view()),
    # ── Wizard node updates from the C4rr13rX repo ────────────────────────
    # GET reports the installed build; POST checks for and installs a newer
    # one. Same updater the Android WorkManager job runs on a schedule.
    path("node/update/", node_update_views.WizardNodeUpdateView.as_view(),
         name="node_update"),
    path("node/update", node_update_views.WizardNodeUpdateView.as_view()),
    path("node/status/", node_update_views.WizardNodeUpdateStatusView.as_view(),
         name="node_update_status"),
    path("node/status", node_update_views.WizardNodeUpdateStatusView.as_view()),
    # ── Decentralized .wbrain v4 P2P DataMesh & Strict Testnet-Only Ledger ─
    path("wbrain/manifests/", wbrain_views.WizardWbrainManifestsView.as_view(), name="wbrain_manifests"),
    path("wbrain/manifests", wbrain_views.WizardWbrainManifestsView.as_view()),
    path("wbrain/export/", wbrain_views.WizardWbrainExportView.as_view(), name="wbrain_export"),
    path("wbrain/export", wbrain_views.WizardWbrainExportView.as_view()),
    path("wbrain/import/", wbrain_views.WizardWbrainImportView.as_view(), name="wbrain_import"),
    path("wbrain/import", wbrain_views.WizardWbrainImportView.as_view()),
    path("wbrain/testnet/", wbrain_views.WizardWbrainTestnetView.as_view(), name="wbrain_testnet"),
    path("wbrain/testnet", wbrain_views.WizardWbrainTestnetView.as_view()),
    # ── 2025 Chevy Colorado 11.3" Infotainment Companion (Web + Sandboxed Projects + W1z4rD) ─
    path("vehicle-companion/", vehicle_companion_views.WizardVehicleCompanionView.as_view(), name="vehicle_companion"),
    path("vehicle-companion", vehicle_companion_views.WizardVehicleCompanionView.as_view()),
]
