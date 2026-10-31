"""URL routes for Agent Market Research & Arbitrage."""
from __future__ import annotations

from django.urls import path

from . import views

urlpatterns = [
    path("status/", views.AgentResearchStatusView.as_view(), name="agentresearch-status"),
    path("markets/", views.MarketScannerView.as_view(), name="agentresearch-markets"),
    path("arbitrage/", views.ArbitrageExplorerView.as_view(), name="agentresearch-arbitrage"),
    path("ghost-runs/", views.GhostSessionsView.as_view(), name="agentresearch-ghost-runs"),
    path("ghost-runs/<int:pk>/", views.GhostSessionDetailView.as_view(), name="agentresearch-ghost-detail"),
    path("graduated/", views.GraduatedStrategiesView.as_view(), name="agentresearch-graduated"),
    path("graduated/<str:strategy_id>/toggle/", views.GraduatedStrategyToggleView.as_view(), name="agentresearch-graduated-toggle"),
    path("agents/", views.AgentDirectoryView.as_view(), name="agentresearch-agents"),
    path("agents/<str:agent_id>/chat/", views.AgentChatView.as_view(), name="agentresearch-agent-chat"),
    path("control/", views.AgentResearchControlView.as_view(), name="agentresearch-control"),
    path("export/", views.AgentResearchExportView.as_view(), name="agentresearch-export"),
    path("vault/", views.VaultView.as_view(), name="agentresearch-vault"),
    path("vault/reveal/", views.VaultRevealView.as_view(), name="agentresearch-vault-reveal"),
    path("settlement-wallets/", views.SettlementWalletsView.as_view(), name="agentresearch-settlement-wallets"),
    path("settlement-wallets/<str:chain>/", views.SettlementWalletDeleteView.as_view(), name="agentresearch-settlement-wallet-delete"),
    path("lineage/", views.StrategyLineageView.as_view(), name="agentresearch-lineage"),
    path("trades/", views.AgentTradesView.as_view(), name="agentresearch-trades"),
    path("alpha-wallets/", views.AlphaWalletsView.as_view(), name="agentresearch-alpha-wallets"),
    path("alpha-wallets/<int:pk>/confirm/", views.AlphaWalletConfirmView.as_view(), name="agentresearch-alpha-wallet-confirm"),
    path("alpha-wallets/<int:pk>/toggle/", views.AlphaWalletToggleView.as_view(), name="agentresearch-alpha-wallet-toggle"),
    path("profit-markers/", views.ProfitMarkersView.as_view(), name="agentresearch-profit-markers"),
    path("profit-markers/send-all/", views.SendAllProfitsView.as_view(), name="agentresearch-send-all-profits"),
    path("token-limits/", views.TokenLimitsView.as_view(), name="agentresearch-token-limits"),
]
