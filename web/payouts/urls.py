from django.urls import path

from . import views

app_name = "payouts"

urlpatterns = [
    path("status/", views.status, name="status"),
    path("destinations/save/", views.save_destination, name="save-destination"),
    path("destinations/<int:destination_id>/delete/",
         views.delete_destination, name="delete-destination"),
]
