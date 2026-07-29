from django.urls import path

from . import views

app_name = "modelcontrol"

urlpatterns = [
    path("", views.ModelControlView.as_view(), name="status"),
    path("options/", views.ModelOptionsView.as_view(), name="options"),
    path("config/", views.ModelControlConfigView.as_view(), name="config"),
    path("wizard-brains/", views.WizardBrainListView.as_view(), name="wizard-brains"),
    path("wizard-brains/selection/", views.WizardBrainSelectionView.as_view(), name="wizard-brain-selection"),
    path("wizard-brains/<str:brain_id>/", views.WizardBrainDetailView.as_view(), name="wizard-brain-detail"),
    path("credentials/<str:name>/", views.ModelCredentialView.as_view(), name="credential"),
]
