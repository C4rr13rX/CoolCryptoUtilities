from __future__ import annotations

from django.urls import path

from . import views

app_name = "walletpanel"

urlpatterns = [
    path("actions/", views.WalletActionsView.as_view(), name="actions"),
    path("run/", views.WalletRunView.as_view(), name="run"),
    path("mnemonic/", views.WalletMnemonicView.as_view(), name="mnemonic"),
    path("state/", views.WalletStateSnapshotView.as_view(), name="state"),
    path("nfts/preferences/", views.WalletNftPreferenceView.as_view(), name="nft-preferences"),
    path("transfers/", views.WalletTransfersView.as_view(), name="transfers"),
    path("wallets/", views.MultiWalletListView.as_view(), name="wallets"),
    path("wallets/create/", views.MultiWalletCreateView.as_view(), name="wallets-create"),
    path("wallets/reveal/", views.WalletRevealMnemonicView.as_view(), name="wallets-reveal"),
]
