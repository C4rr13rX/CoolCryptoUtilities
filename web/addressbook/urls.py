from __future__ import annotations

from django.urls import path

from addressbook import views

urlpatterns = [
    path("entries/", views.AddressBookEntryListCreateView.as_view(), name="addressbook-entries"),
    path("entries/<int:pk>/", views.AddressBookEntryDetailView.as_view(), name="addressbook-entry"),
    path("lookup/", views.AddressBookLookupView.as_view(), name="addressbook-lookup"),
]
