from django.urls import path

from . import views

app_name = "modelcontrol"

urlpatterns = [
    path("", views.ModelControlView.as_view(), name="status"),
    path("options/", views.ModelOptionsView.as_view(), name="options"),
    path("config/", views.ModelControlConfigView.as_view(), name="config"),
    path("credentials/<str:name>/", views.ModelCredentialView.as_view(), name="credential"),
]

