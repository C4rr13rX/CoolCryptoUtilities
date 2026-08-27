from django.urls import path

from . import ga_views, strategy_views, views

app_name = "modelcontrol"

urlpatterns = [
    # Genetic-algorithm searches over trainable brains. The UI binds to these;
    # they are live ahead of it so the screens stay a thin view over logic
    # that already works.
    path("ga/space/", ga_views.GeneSpaceView.as_view(), name="ga-space"),
    path("ga/runs/", ga_views.GARunListView.as_view(), name="ga-runs"),
    path("ga/runs/<str:run_id>/", ga_views.GARunDetailView.as_view(), name="ga-run-detail"),
    path("ga/runs/<str:run_id>/promote/", ga_views.GAPromoteView.as_view(), name="ga-promote"),
    path("ga/models/", ga_views.GAModelListView.as_view(), name="ga-models"),
    # Strategy lifecycle: view, commission/decommission, and re-experiment
    # the same strategy against a different model/brain.
    path("strategies/", strategy_views.StrategyListView.as_view(), name="strategies"),
    path("strategies/objectives/", strategy_views.ObjectiveListView.as_view(), name="strategy-objectives"),
    path("strategies/<str:strategy_id>/", strategy_views.StrategyDetailView.as_view(), name="strategy-detail"),
    path("strategies/<str:strategy_id>/commission/", strategy_views.StrategyCommissionView.as_view(), name="strategy-commission"),
    path("strategies/<str:strategy_id>/compare/", strategy_views.StrategyCompareView.as_view(), name="strategy-compare"),
    path("strategies/<str:strategy_id>/experiment/", strategy_views.StrategyExperimentView.as_view(), name="strategy-experiment"),
    path("", views.ModelControlView.as_view(), name="status"),
    path("options/", views.ModelOptionsView.as_view(), name="options"),
    path("config/", views.ModelControlConfigView.as_view(), name="config"),
    path("wizard-brains/", views.WizardBrainListView.as_view(), name="wizard-brains"),
    path("wizard-brains/selection/", views.WizardBrainSelectionView.as_view(), name="wizard-brain-selection"),
    path("wizard-brains/<str:brain_id>/", views.WizardBrainDetailView.as_view(), name="wizard-brain-detail"),
    path("credentials/<str:name>/", views.ModelCredentialView.as_view(), name="credential"),
]
