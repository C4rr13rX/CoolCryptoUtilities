from __future__ import annotations

from django.urls import path

from . import views

app_name = "walletpanel"

urlpatterns = [
    path("actions/", views.WalletActionsView.as_view(), name="actions"),
    path("run/", views.WalletRunView.as_view(), name="run"),
    path("mnemonic/", views.WalletMnemonicView.as_view(), name="mnemonic"),
    path("state/", views.WalletStateSnapshotView.as_view(), name="state"),
]
