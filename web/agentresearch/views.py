"""REST API Views for Agent Market Research, Multi-Chain Arbitrage,
Ghost Self-Scoring & Live Graduation, plus the Multi-Agent Directory & Communication Deck.
"""
from __future__ import annotations
from services.token_limiter import TokenLimiter

import csv
import io
import time
from typing import Any, Dict, List
from xml.sax.saxutils import escape as xml_escape

from django.db.models import Avg, Count, Max, Sum
from django.http import HttpRequest, HttpResponse
from django.utils import timezone
from django.views import View
from rest_framework import permissions, status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .engine import (
    RESEARCH_COORDINATOR,
    AgentMessenger,
    AlphaWalletResearcher,
    ArbitrageHunter,
    GhostResearchSimulator,
    GraduationCoordinator,
    MarketDataScanner,
    ProfitAccrualCoordinator,
    VaultManager,
)
from .models import (
    AgentMessage,
    AgentProfile,
    AgentTradeRecord,
    AlphaTraderWallet,
    ArbitrageOpportunity,
    ChainSettlementWallet,
    GhostTrade,
    GhostTradingSession,
    LiveGraduatedStrategy,
    MarketResearchConfig,
    MarketSnapshot,
    ProfitAccrualMarker,
    ProfitVaultConfig,
    StrategyLineageNode,
    WalletFollowTrade,
)


def _agent_dict(a: AgentProfile) -> Dict[str, Any]:
    return {
        "agent_id": a.agent_id,
        "name": a.name,
        "role": a.role,
        "domain": a.domain,
        "system_role_tag": a.system_role_tag,
        "model_tier": a.model_tier,
        "system_prompt": a.system_prompt,
        "is_active": a.is_active,
        "status": a.status,
        "total_ghost_trades": a.total_ghost_trades,
        "total_live_trades": a.total_live_trades,
        "ghost_win_rate": round(a.ghost_win_rate, 4),
        "live_win_rate": round(a.live_win_rate, 4),
        "live_net_pnl_usd": round(a.live_net_pnl_usd, 4),
        "created_at": a.created_at.isoformat(),
        "last_active_at": a.last_active_at.isoformat(),
    }


def _session_dict(s: GhostTradingSession) -> Dict[str, Any]:
    return {
        "id": s.pk,
        "session_id": s.session_id,
        "agent_id": s.agent_id,
        "agent_name": s.agent.name if s.agent else "",
        "strategy_name": s.strategy_name,
        "strategy_type": s.strategy_type,
        "chain": s.chain,
        "symbol": s.symbol,
        "horizon": s.horizon,
        "hypothesis": s.hypothesis,
        "buy_low_rule": s.buy_low_rule,
        "sell_high_rule": s.sell_high_rule,
        "historical_bars_count": s.historical_bars_count,
        "total_trades": s.total_trades,
        "wins": s.wins,
        "losses": s.losses,
        "win_rate": round(s.win_rate, 4),
        "gross_pnl_usd": round(s.gross_pnl_usd, 6),
        "total_fees_usd": round(s.total_fees_usd, 6),
        "net_pnl_usd": round(s.net_pnl_usd, 6),
        "profit_factor": round(s.profit_factor, 4),
        "sharpe_ratio": round(s.sharpe_ratio, 2),
        "max_drawdown_pct": round(s.max_drawdown_pct, 4),
        "holdout_t_stat": round(s.holdout_t_stat, 3),
        "self_score": s.self_score,
        "self_assessment": s.self_assessment,
        "status": s.status,
        "graduated_to_live": s.graduated_to_live,
        "graduation_timestamp": s.graduation_timestamp.isoformat() if s.graduation_timestamp else None,
        "created_at": s.created_at.isoformat(),
    }


def _graduated_dict(g: LiveGraduatedStrategy) -> Dict[str, Any]:
    return {
        "strategy_id": g.strategy_id,
        "agent_id": g.agent_id,
        "agent_name": g.agent.name if g.agent else "",
        "name": g.name,
        "strategy_type": g.strategy_type,
        "chain": g.chain,
        "symbol": g.symbol,
        "buy_low_logic": g.buy_low_logic,
        "sell_high_logic": g.sell_high_logic,
        "ghost_win_rate": round(g.ghost_win_rate, 4),
        "ghost_profit_factor": round(g.ghost_profit_factor, 4),
        "ghost_net_pnl_usd": round(g.ghost_net_pnl_usd, 6),
        "ghost_t_stat": round(g.ghost_t_stat, 3),
        "live_status": g.live_status,
        "live_trades_count": g.live_trades_count,
        "live_win_rate": round(g.live_win_rate, 4),
        "live_net_pnl_usd": round(g.live_net_pnl_usd, 4),
        "clip_usd": g.clip_usd,
        "commissioned_in_registry": g.commissioned_in_registry,
        "wired_to_money_button": g.wired_to_money_button,
        "wired_to_discovered_rules": g.wired_to_discovered_rules,
        "wired_to_strategy_ledger": g.wired_to_strategy_ledger,
        "commissioned_at": g.commissioned_at.isoformat(),
        "last_trade_at": g.last_trade_at.isoformat() if g.last_trade_at else None,
    }


def _wallet_dict(w: ChainSettlementWallet) -> Dict[str, Any]:
    return {
        "id": w.pk,
        "chain": w.chain,
        "stablecoin_symbol": w.stablecoin_symbol,
        "wallet_address": w.wallet_address,
        "is_active": w.is_active,
        "allocated_profit_pct": round(w.allocated_profit_pct, 2),
        "total_settled_usd": round(w.total_settled_usd, 4),
        "last_settled_at": w.last_settled_at.isoformat() if w.last_settled_at else None,
        "updated_at": w.updated_at.isoformat(),
    }


def _lineage_dict(n: StrategyLineageNode) -> Dict[str, Any]:
    return {
        "strategy_id": n.strategy_id,
        "agent_id": n.agent_id,
        "agent_name": n.agent.name if n.agent else "",
        "parent_strategy_id": n.parent_strategy_id,
        "name": n.name,
        "derivation_type": n.derivation_type,
        "execution_mode": n.execution_mode,
        "generation": n.generation,
        "chain": n.chain,
        "symbol": n.symbol,
        "rules_summary": n.rules_summary,
        "is_active": n.is_active,
        "trades_count": n.trades_count,
        "win_rate": round(n.win_rate, 4),
        "profit_factor": round(n.profit_factor, 4),
        "net_pnl_usd": round(n.net_pnl_usd, 4),
        "t_stat": round(n.t_stat, 3),
        "created_at": n.created_at.isoformat(),
        "updated_at": n.updated_at.isoformat(),
    }


def _trade_dict(t: AgentTradeRecord) -> Dict[str, Any]:
    return {
        "id": t.pk,
        "agent_id": t.agent_id,
        "agent_name": t.agent.name if t.agent else "",
        "strategy_id": t.strategy_id if t.strategy else "",
        "trade_mode": t.trade_mode,
        "chain": t.chain,
        "symbol": t.symbol,
        "side": t.side,
        "notional_usd": round(t.notional_usd, 4),
        "entry_price": round(t.entry_price, 4),
        "exit_price": round(t.exit_price, 4),
        "quantity": round(t.quantity, 6),
        "gross_pnl_usd": round(t.gross_pnl_usd, 6),
        "fees_usd": round(t.fees_usd, 6),
        "net_pnl_usd": round(t.net_pnl_usd, 6),
        "return_pct": round(t.return_pct * 100.0, 4),
        "status": t.status,
        "tx_hash": t.tx_hash,
        "entry_time": t.entry_time.isoformat(),
        "exit_time": t.exit_time.isoformat() if t.exit_time else None,
        "exit_reason": t.exit_reason,
        "metadata": t.metadata,
    }


def _alpha_wallet_dict(w: AlphaTraderWallet) -> Dict[str, Any]:
    return {
        "id": w.pk,
        "chain": w.chain,
        "wallet_address": w.wallet_address,
        "alias": w.alias,
        "months_winning": w.months_winning,
        "win_rate": round(w.win_rate * 100.0, 1),
        "total_trades": w.total_trades,
        "estimated_pnl_usd": round(w.estimated_pnl_usd, 2),
        "profit_factor": round(w.profit_factor, 2),
        "avg_holding_time_sec": round(w.avg_holding_time_sec, 1),
        "preferred_tokens": w.preferred_tokens,
        "preferred_dexes": w.preferred_dexes,
        "verification_status": w.verification_status,
        "live_confirm_trades_count": w.live_confirm_trades_count,
        "live_confirm_win_rate": round(w.live_confirm_win_rate * 100.0, 1),
        "live_confirm_net_pnl_usd": round(w.live_confirm_net_pnl_usd, 4),
        "is_mirror_active": w.is_mirror_active,
        "mirror_clip_usd": w.mirror_clip_usd,
        "notes": w.notes,
        "graduated_to_live_at": w.graduated_to_live_at.isoformat() if w.graduated_to_live_at else None,
        "created_at": w.created_at.isoformat(),
    }


def _profit_marker_dict(m: ProfitAccrualMarker) -> Dict[str, Any]:
    return {
        "marker_id": m.marker_id,
        "source": m.source,
        "chain": m.chain,
        "symbol": m.symbol,
        "gross_profit_usd": round(m.gross_profit_usd, 4),
        "allocated_pct": round(m.allocated_pct, 1),
        "accrued_profit_usd": round(m.accrued_profit_usd, 4),
        "status": m.status,
        "under_1000_rule_applied": m.under_1000_rule_applied,
        "dispatched_at": m.dispatched_at.isoformat() if m.dispatched_at else None,
        "dispatch_tx_or_note": m.dispatch_tx_or_note,
        "created_at": m.created_at.isoformat(),
    }


class AgentResearchStatusView(APIView):
    """High-level summary of agents, market coverage, arbitrage, ghost runs, and graduated live strategies."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, *args, **kwargs) -> Response:
        RESEARCH_COORDINATOR.ensure_default_agents()
        VaultManager.ensure_default_settlement_wallets()
        AlphaWalletResearcher.ensure_default_alpha_wallets()
        cfg = MarketResearchConfig.load()
        vault_cfg = ProfitVaultConfig.load()

        agents = AgentProfile.objects.all().order_by("system_role_tag", "name")
        graduated_strategies = LiveGraduatedStrategy.objects.all().order_by("-commissioned_at")
        recent_sessions = GhostTradingSession.objects.all().order_by("-created_at")[:10]
        recent_arbitrage = ArbitrageOpportunity.objects.filter(is_profitable_after_fees=True).order_by("-net_edge_pct")[:8]
        settlement_wallets = ChainSettlementWallet.objects.all().order_by("chain")
        lineage_nodes = StrategyLineageNode.objects.all()
        alpha_wallets = AlphaTraderWallet.objects.all()
        running_totals = ProfitAccrualCoordinator.get_running_totals()

        return Response({
            "token_limits": TokenLimiter.get_all_limits_summary(),
            "config": {
                "enabled": cfg.enabled,
                "chains": cfg.chains,
                "dexes": cfg.dexes,
                "auto_graduate": cfg.auto_graduate,
                "clip_usd": cfg.clip_usd,
                "graduation_min_trades": cfg.graduation_min_trades,
                "graduation_min_winrate": cfg.graduation_min_winrate,
                "graduation_min_profit_factor": cfg.graduation_min_profit_factor,
                "graduation_min_t_stat": cfg.graduation_min_t_stat,
                "graduation_max_drawdown": cfg.graduation_max_drawdown,
                "primary_trader_agent_id": cfg.primary_trader_agent_id,
            },
            "vault": {
                "has_secret": bool(vault_cfg.ciphertext and vault_cfg.encapsulated_key and vault_cfg.nonce),
                "masked_preview": vault_cfg.masked_preview if (vault_cfg.ciphertext and vault_cfg.encapsulated_key) else "No account encrypted",
                "profit_transfer_pct": vault_cfg.profit_transfer_pct,
                "accumulated_pending_profits_usd": round(vault_cfg.accumulated_pending_profits_usd, 4),
                "total_dispatched_profits_usd": round(vault_cfg.total_dispatched_profits_usd, 4),
                "total_profits_transferred_usd": round(vault_cfg.total_profits_transferred_usd, 4),
                "manual_threshold_usd": vault_cfg.manual_threshold_usd,
                "last_manual_dispatch_at": vault_cfg.last_manual_dispatch_at.isoformat() if vault_cfg.last_manual_dispatch_at else None,
                "last_transferred_at": vault_cfg.last_transferred_at.isoformat() if vault_cfg.last_transferred_at else None,
                "label": vault_cfg.label,
            },
            "running_totals": running_totals,
            "counts": {
                "total_agents": agents.count(),
                "active_agents": agents.filter(is_active=True).count(),
                "market_snapshots": MarketSnapshot.objects.count(),
                "arbitrage_opportunities": ArbitrageOpportunity.objects.count(),
                "profitable_arbitrage": ArbitrageOpportunity.objects.filter(is_profitable_after_fees=True).count(),
                "ghost_sessions_count": GhostTradingSession.objects.count(),
                "graduated_live_count": graduated_strategies.count(),
                "live_active_count": graduated_strategies.filter(live_status=LiveGraduatedStrategy.Status.ACTIVE).count(),
                "settlement_wallets_count": settlement_wallets.count(),
                "lineage_nodes_count": lineage_nodes.count(),
                "agent_trades_count": AgentTradeRecord.objects.count(),
                "alpha_wallets_count": alpha_wallets.count(),
                "alpha_mirroring_count": alpha_wallets.filter(is_mirror_active=True).count(),
                "profit_markers_count": ProfitAccrualMarker.objects.count(),
            },
            "agents": [_agent_dict(a) for a in agents],
            "graduated_strategies": [_graduated_dict(g) for g in graduated_strategies[:15]],
            "recent_ghost_sessions": [_session_dict(s) for s in recent_sessions],
            "top_arbitrage_opportunities": [
                {
                    "opp_type": o.opp_type,
                    "asset": o.asset,
                    "buy_chain": o.buy_chain,
                    "buy_dex": o.buy_dex,
                    "buy_price": o.buy_price,
                    "sell_chain": o.sell_chain,
                    "sell_dex": o.sell_dex,
                    "sell_price": o.sell_price,
                    "gross_spread_pct": round(o.gross_spread_pct * 100.0, 3),
                    "net_edge_pct": round(o.net_edge_pct * 100.0, 3),
                    "is_profitable": o.is_profitable_after_fees,
                    "created_at": o.created_at.isoformat(),
                }
                for o in recent_arbitrage
            ],
            "settlement_wallets": [_wallet_dict(w) for w in settlement_wallets],
            "alpha_wallets": [_alpha_wallet_dict(aw) for aw in alpha_wallets[:10]],
        }, status=status.HTTP_200_OK)


class MarketScannerView(APIView):
    """Multi-chain market data scanner with lowest fee calculation and trend indicators."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, *args, **kwargs) -> Response:
        chain_filter = request.query_params.get("chain", "").lower()
        symbol_filter = request.query_params.get("symbol", "").upper()

        qs = MarketSnapshot.objects.all()
        if chain_filter:
            qs = qs.filter(chain=chain_filter)
        if symbol_filter:
            qs = qs.filter(pair_symbol__icontains=symbol_filter)

        snapshots = list(qs.order_by("-created_at")[:60])
        return Response({
            "total_snapshots": qs.count(),
            "results": [
                {
                    "id": m.pk,
                    "chain": m.chain,
                    "dex": m.dex,
                    "pair_symbol": m.pair_symbol,
                    "price": m.price,
                    "bid_price": m.bid_price,
                    "ask_price": m.ask_price,
                    "spread_bps": m.spread_bps,
                    "fee_tier_bps": m.fee_tier_bps,
                    "gas_estimate_usd": m.gas_estimate_usd,
                    "roundtrip_fee_pct": m.roundtrip_fee_pct,
                    "liquidity_depth_usd": m.liquidity_depth_usd,
                    "volume_24h_usd": m.volume_24h_usd,
                    "volatility_atr14": m.volatility_atr14,
                    "trend_direction": m.trend_direction,
                    "trend_strength": m.trend_strength,
                    "kalman_velocity": m.kalman_velocity,
                    "ema_alignment": m.ema_alignment,
                    "rsi_14": m.rsi_14,
                    "created_at": m.created_at.isoformat(),
                }
                for m in snapshots
            ],
        }, status=status.HTTP_200_OK)


class ArbitrageExplorerView(APIView):
    """Cross-chain, cross-DEX, and spatial arbitrage tracking."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, *args, **kwargs) -> Response:
        profitable_only = request.query_params.get("profitable_only", "false").lower() in {"1", "true", "yes"}
        qs = ArbitrageOpportunity.objects.all()
        if profitable_only:
            qs = qs.filter(is_profitable_after_fees=True)

        opps = list(qs.order_by("-net_edge_pct", "-created_at")[:50])
        return Response({
            "count": len(opps),
            "results": [
                {
                    "id": o.pk,
                    "opp_type": o.opp_type,
                    "asset": o.asset,
                    "buy_chain": o.buy_chain,
                    "buy_dex": o.buy_dex,
                    "buy_price": o.buy_price,
                    "sell_chain": o.sell_chain,
                    "sell_dex": o.sell_dex,
                    "sell_price": o.sell_price,
                    "gross_spread_pct": o.gross_spread_pct,
                    "swap_fees_pct": o.swap_fees_pct,
                    "gas_cost_usd": o.gas_cost_usd,
                    "slippage_est_pct": o.slippage_est_pct,
                    "net_edge_pct": o.net_edge_pct,
                    "is_profitable_after_fees": o.is_profitable_after_fees,
                    "status": o.status,
                    "details": o.details,
                    "created_at": o.created_at.isoformat(),
                }
                for o in opps
            ],
        }, status=status.HTTP_200_OK)


class GhostSessionsView(APIView):
    """Historical ghost trading research sessions and trigger endpoint."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, *args, **kwargs) -> Response:
        qs = GhostTradingSession.objects.all().order_by("-created_at")
        return Response({
            "count": qs.count(),
            "results": [_session_dict(s) for s in qs[:40]],
        }, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        """Trigger an immediate historical ghost research pass with agent self-scoring."""
        res = RESEARCH_COORDINATOR.run_agent_ghost_research_pass()
        if not res:
            return Response({"error": "Ghost research pass could not be started."}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"status": "success", "result": res}, status=status.HTTP_201_CREATED)


class GhostSessionDetailView(APIView):
    """Detailed view of an individual ghost session and its simulated trades."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, pk: int, *args, **kwargs) -> Response:
        session = GhostTradingSession.objects.filter(pk=pk).first()
        if not session:
            return Response({"error": "Session not found."}, status=status.HTTP_404_NOT_FOUND)

        trades = session.trades.all().order_by("trade_index")
        return Response({
            "session": _session_dict(session),
            "trades": [
                {
                    "trade_index": t.trade_index,
                    "entry_ts": t.entry_ts.isoformat(),
                    "entry_price": t.entry_price,
                    "exit_ts": t.exit_ts.isoformat(),
                    "exit_price": t.exit_price,
                    "hold_duration_sec": t.hold_duration_sec,
                    "action": t.action,
                    "gross_return_pct": t.gross_return_pct,
                    "fee_deduction_pct": t.fee_deduction_pct,
                    "net_return_pct": t.net_return_pct,
                    "net_pnl_usd": t.net_pnl_usd,
                    "exit_reason": t.exit_reason,
                }
                for t in trades
            ],
        }, status=status.HTTP_200_OK)


class GraduatedStrategiesView(APIView):
    """Strategies graduated from ghost trading to live execution."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, *args, **kwargs) -> Response:
        qs = LiveGraduatedStrategy.objects.all().order_by("-commissioned_at")
        return Response({
            "count": qs.count(),
            "results": [_graduated_dict(g) for g in qs],
        }, status=status.HTTP_200_OK)


class GraduatedStrategyToggleView(APIView):
    """Pause or resume a graduated live strategy."""

    permission_classes = [permissions.AllowAny]

    def post(self, request: Request, strategy_id: str, *args, **kwargs) -> Response:
        strat = LiveGraduatedStrategy.objects.filter(strategy_id=strategy_id).first()
        if not strat:
            return Response({"error": "Strategy not found."}, status=status.HTTP_404_NOT_FOUND)

        action = request.data.get("action", "toggle")
        if action == "pause" or (action == "toggle" and strat.live_status == LiveGraduatedStrategy.Status.ACTIVE):
            strat.live_status = LiveGraduatedStrategy.Status.PAUSED
        else:
            strat.live_status = LiveGraduatedStrategy.Status.ACTIVE
        strat.save(update_fields=["live_status"])

        return Response({
            "strategy_id": strat.strategy_id,
            "live_status": strat.live_status,
        }, status=status.HTTP_200_OK)


class AgentDirectoryView(APIView):
    """Lists all agents across the site and allows creating new agents with custom roles."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, *args, **kwargs) -> Response:
        RESEARCH_COORDINATOR.ensure_default_agents()
        agents = AgentProfile.objects.all().order_by("system_role_tag", "name")
        return Response({
            "count": agents.count(),
            "agents": [_agent_dict(a) for a in agents],
        }, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        """Create a new agent with a custom role, domain, model tier, and directives."""
        name = request.data.get("name", "").strip()
        role = request.data.get("role", "").strip()
        domain = request.data.get("domain", "Market Research & Trading").strip()
        model_tier = request.data.get("model_tier", "gemini-2.5-flash").strip()
        system_prompt = request.data.get("system_prompt", "").strip()

        if not name or not role:
            return Response({"error": "Both 'name' and 'role' are required to create a new agent."}, status=status.HTTP_400_BAD_REQUEST)

        # Generate unique slug
        agent_id = f"custom_{name.lower().replace(' ', '_')[:24]}_{int(time.time()) % 10000}"
        if not system_prompt:
            system_prompt = f"You are {name}, specialized as {role} for R3V3N!R in domain {domain}."

        agent = AgentProfile.objects.create(
            agent_id=agent_id,
            name=name,
            role=role,
            domain=domain,
            system_role_tag=AgentProfile.SystemRole.CUSTOM,
            model_tier=model_tier,
            system_prompt=system_prompt,
            is_active=True,
            status=AgentProfile.Status.IDLE,
        )

        return Response({
            "status": "created",
            "agent": _agent_dict(agent),
        }, status=status.HTTP_201_CREATED)


class AgentChatView(APIView):
    """Two-way communication deck with an individual agent labeled by role."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, agent_id: str, *args, **kwargs) -> Response:
        agent = AgentProfile.objects.filter(agent_id=agent_id).first()
        if not agent:
            return Response({"error": f"Agent '{agent_id}' not found."}, status=status.HTTP_404_NOT_FOUND)

        messages = agent.messages.all().order_by("created_at")[:60]
        return Response({
            "agent": _agent_dict(agent),
            "messages": [
                {
                    "id": m.pk,
                    "sender": m.sender,
                    "content": m.content,
                    "context_data": m.context_data,
                    "created_at": m.created_at.isoformat(),
                }
                for m in messages
            ],
        }, status=status.HTTP_200_OK)

    def post(self, request: Request, agent_id: str, *args, **kwargs) -> Response:
        """Send a message or directive to an individual agent and get its response."""
        agent = AgentProfile.objects.filter(agent_id=agent_id).first()
        if not agent:
            return Response({"error": f"Agent '{agent_id}' not found."}, status=status.HTTP_404_NOT_FOUND)

        user_content = request.data.get("message", "").strip()
        if not user_content:
            return Response({"error": "Message content cannot be empty."}, status=status.HTTP_400_BAD_REQUEST)

        # 1. Record user message
        user_msg = AgentMessage.objects.create(
            agent=agent,
            sender=AgentMessage.Sender.USER,
            content=user_content,
            created_at=timezone.now(),
        )

        # 2. Update agent status to chatting
        orig_status = agent.status
        agent.status = AgentProfile.Status.CHATTING
        agent.last_active_at = timezone.now()
        agent.save(update_fields=["status", "last_active_at"])

        # 3. Dispatch to agent and receive reply
        reply_content = AgentMessenger.send_to_agent(
            agent_profile=agent,
            user_message=user_content,
            context={
                "agent_id": agent.agent_id,
                "role": agent.role,
                "domain": agent.domain,
                "total_ghost_trades": agent.total_ghost_trades,
                "total_live_trades": agent.total_live_trades,
                "live_pnl_usd": agent.live_net_pnl_usd,
            },
        )

        # 4. Record agent message
        agent_msg = AgentMessage.objects.create(
            agent=agent,
            sender=AgentMessage.Sender.AGENT,
            content=reply_content,
            created_at=timezone.now(),
        )

        agent.status = orig_status
        agent.save(update_fields=["status"])

        return Response({
            "user_message": {
                "id": user_msg.pk,
                "sender": user_msg.sender,
                "content": user_msg.content,
                "created_at": user_msg.created_at.isoformat(),
            },
            "agent_message": {
                "id": agent_msg.pk,
                "sender": agent_msg.sender,
                "content": agent_msg.content,
                "created_at": agent_msg.created_at.isoformat(),
            },
        }, status=status.HTTP_201_CREATED)


class AgentResearchControlView(APIView):
    """System control actions (scan markets, run ghost research, toggle auto-graduate)."""

    permission_classes = [permissions.AllowAny]

    def post(self, request: Request, *args, **kwargs) -> Response:
        action = request.data.get("action", "")
        cfg = MarketResearchConfig.load()

        if action == "scan_markets":
            res = RESEARCH_COORDINATOR.run_market_scan_iteration()
            return Response({"status": "success", "action": action, "result": res})
        elif action == "run_ghost_research":
            res = RESEARCH_COORDINATOR.run_agent_ghost_research_pass()
            return Response({"status": "success", "action": action, "result": res})
        elif action == "toggle_auto_graduate":
            cfg.auto_graduate = not cfg.auto_graduate
            cfg.save(update_fields=["auto_graduate"])
            return Response({"status": "success", "auto_graduate": cfg.auto_graduate})
        elif action == "update_config":
            for field in ["enabled", "clip_usd", "graduation_min_trades", "graduation_min_winrate"]:
                if field in request.data:
                    setattr(cfg, field, request.data[field])
            cfg.save()
            return Response({"status": "updated", "auto_graduate": cfg.auto_graduate, "clip_usd": cfg.clip_usd})
        else:
            return Response({"error": f"Unknown action: '{action}'"}, status=status.HTTP_400_BAD_REQUEST)


class AgentResearchExportView(View):
    """Export ghost research runs and graduated strategies as CSV or Excel."""

    def get(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        fmt = request.GET.get("format", "csv").lower()

        sessions = GhostTradingSession.objects.all().order_by("-created_at")
        headers = [
            "ID", "Agent", "Strategy", "Chain", "Symbol", "Total Trades", "Wins", "Losses",
            "Win Rate %", "Gross PnL ($)", "Total Fees ($)", "Net PnL ($)", "Profit Factor",
            "Sharpe", "Max DD %", "t-stat", "Self Score (0-100)", "Status", "Graduated Live", "Created At"
        ]
        rows: List[List[Any]] = []
        for s in sessions:
            rows.append([
                s.session_id,
                s.agent_id,
                s.strategy_name,
                s.chain,
                s.symbol,
                s.total_trades,
                s.wins,
                s.losses,
                round(s.win_rate * 100.0, 2),
                round(s.gross_pnl_usd, 4),
                round(s.total_fees_usd, 4),
                round(s.net_pnl_usd, 4),
                round(s.profit_factor, 2),
                round(s.sharpe_ratio, 2),
                round(s.max_drawdown_pct * 100.0, 2),
                round(s.holdout_t_stat, 2),
                round(s.self_score, 1),
                s.status,
                "YES" if s.graduated_to_live else "NO",
                s.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            ])

        if fmt in {"xlsx", "excel"}:
            # Build clean SpreadsheetML (Excel compatible)
            xml_rows = []
            xml_rows.append("<Row>" + "".join(f'<Cell><Data ss:Type="String">{xml_escape(str(h))}</Data></Cell>' for h in headers) + "</Row>")
            for r in rows:
                xml_cells = []
                for idx, cell in enumerate(r):
                    cell_type = "Number" if isinstance(cell, (int, float)) else "String"
                    xml_cells.append(f'<Cell><Data ss:Type="{cell_type}">{xml_escape(str(cell))}</Data></Cell>')
                xml_rows.append("<Row>" + "".join(xml_cells) + "</Row>")

            content = (
                '<?xml version="1.0"?>\n'
                '<?mso-application progid="Excel.Sheet"?>\n'
                '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"\n'
                ' xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">\n'
                '<Worksheet ss:Name="Agent Research & Ghost Trading">\n'
                '<Table>\n' + "\n".join(xml_rows) + "\n</Table>\n"
                '</Worksheet>\n</Workbook>'
            )
            response = HttpResponse(content, content_type="application/vnd.ms-excel; charset=utf-8")
            response["Content-Disposition"] = 'attachment; filename="agent_market_research_ghost_runs.xlsx"'
            return response
        else:
            # Default CSV export
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(headers)
            writer.writerows(rows)

            response = HttpResponse(output.getvalue(), content_type="text/csv; charset=utf-8")
            response["Content-Disposition"] = 'attachment; filename="agent_market_research_ghost_runs.csv"'
            return response


class VaultView(APIView):
    """Encrypted vault configuration for profit sweep destination account and transfer percentage."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, *args, **kwargs) -> Response:
        cfg = ProfitVaultConfig.load()
        has_secret = bool(cfg.ciphertext and cfg.encapsulated_key and cfg.nonce)
        return Response({
            "has_secret": has_secret,
            "masked_preview": cfg.masked_preview if has_secret else "No account encrypted",
            "profit_transfer_pct": cfg.profit_transfer_pct,
            "total_profits_transferred_usd": round(cfg.total_profits_transferred_usd, 4),
            "last_transferred_at": cfg.last_transferred_at.isoformat() if cfg.last_transferred_at else None,
            "label": cfg.label,
        }, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        secret = request.data.get("secret", "").strip()
        profit_pct = request.data.get("profit_transfer_pct")

        if secret:
            res = VaultManager.store_vault_secret(
                value=secret,
                profit_transfer_pct=float(profit_pct) if profit_pct is not None else None,
            )
            return Response(res, status=status.HTTP_200_OK)
        elif profit_pct is not None:
            updated_pct = VaultManager.update_profit_transfer_pct(float(profit_pct))
            return Response({
                "status": "updated",
                "profit_transfer_pct": updated_pct,
            }, status=status.HTTP_200_OK)
        else:
            return Response({"error": "Either 'secret' or 'profit_transfer_pct' must be provided."}, status=status.HTTP_400_BAD_REQUEST)


class VaultRevealView(APIView):
    """Decrypts and reveals the plaintext account secret on demand."""

    permission_classes = [permissions.AllowAny]

    def post(self, request: Request, *args, **kwargs) -> Response:
        try:
            plaintext = VaultManager.reveal_vault_secret()
            return Response({"status": "decrypted", "secret": plaintext}, status=status.HTTP_200_OK)
        except Exception as exc:
            return Response({"error": f"Failed to decrypt vault secret: {exc}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SettlementWalletsView(APIView):
    """Lists and configures per-chain stablecoin settlement wallet addresses."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, *args, **kwargs) -> Response:
        VaultManager.ensure_default_settlement_wallets()
        wallets = ChainSettlementWallet.objects.all().order_by("chain")
        return Response({
            "count": wallets.count(),
            "wallets": [_wallet_dict(w) for w in wallets],
        }, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        chain = request.data.get("chain", "").strip().lower()
        wallet_address = request.data.get("wallet_address", "").strip()

        if not chain or not wallet_address:
            return Response({"error": "Both 'chain' and 'wallet_address' are required."}, status=status.HTTP_400_BAD_REQUEST)

        symbol = request.data.get("stablecoin_symbol", "USDC").strip().upper()
        allocated_pct = float(request.data.get("allocated_profit_pct", 100.0))
        is_active = bool(request.data.get("is_active", True))

        wallet, _ = ChainSettlementWallet.objects.update_or_create(
            chain=chain,
            defaults={
                "stablecoin_symbol": symbol,
                "wallet_address": wallet_address,
                "allocated_profit_pct": allocated_pct,
                "is_active": is_active,
            },
        )
        return Response({"status": "saved", "wallet": _wallet_dict(wallet)}, status=status.HTTP_200_OK)


class SettlementWalletDeleteView(APIView):
    """Remove a chain settlement wallet."""

    permission_classes = [permissions.AllowAny]

    def delete(self, request: Request, chain: str, *args, **kwargs) -> Response:
        wallet = ChainSettlementWallet.objects.filter(chain=chain.lower()).first()
        if not wallet:
            return Response({"error": f"Wallet for chain '{chain}' not found."}, status=status.HTTP_404_NOT_FOUND)
        wallet.delete()
        return Response({"status": "deleted", "chain": chain}, status=status.HTTP_200_OK)


class StrategyLineageView(APIView):
    """Lineage view: the agent as its own root strategy and child strategies born from it."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, *args, **kwargs) -> Response:
        RESEARCH_COORDINATOR.ensure_default_agents()
        # Ensure root parent agent nodes exist
        agents = AgentProfile.objects.all()
        for a in agents:
            StrategyLineageNode.objects.get_or_create(
                strategy_id=f"agent_{a.agent_id}",
                defaults={
                    "agent": a,
                    "parent_strategy_id": "",
                    "name": f"{a.name} (Root Archetype)",
                    "derivation_type": "parent_agent_core",
                    "execution_mode": StrategyLineageNode.Mode.PARENT_AGENT,
                    "generation": 0,
                    "chain": "multi",
                    "symbol": "ALL",
                    "rules_summary": a.system_prompt[:300] if a.system_prompt else a.role,
                    "is_active": a.is_active,
                    "trades_count": a.total_ghost_trades + a.total_live_trades,
                    "win_rate": a.live_win_rate or a.ghost_win_rate,
                    "net_pnl_usd": a.live_net_pnl_usd,
                },
            )

        nodes = StrategyLineageNode.objects.all().order_by("generation", "created_at")
        return Response({
            "count": nodes.count(),
            "lineage": [_lineage_dict(n) for n in nodes],
        }, status=status.HTTP_200_OK)


class AgentTradesView(APIView):
    """Tracks trades executed by agents in ghost or live mode with dynamic sizing."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, *args, **kwargs) -> Response:
        qs = AgentTradeRecord.objects.all()
        mode = request.query_params.get("mode", "").lower()
        chain = request.query_params.get("chain", "").lower()
        symbol = request.query_params.get("symbol", "").upper()
        agent_id = request.query_params.get("agent_id", "")

        if mode in [AgentTradeRecord.TradeMode.GHOST, AgentTradeRecord.TradeMode.LIVE]:
            qs = qs.filter(trade_mode=mode)
        if chain:
            qs = qs.filter(chain=chain)
        if symbol:
            qs = qs.filter(symbol__icontains=symbol)
        if agent_id:
            qs = qs.filter(agent_id=agent_id)

        trades = list(qs[:100])
        return Response({
            "count": len(trades),
            "trades": [_trade_dict(t) for t in trades],
        }, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        agent_id = request.data.get("agent_id", "agy_market_trader")
        agent = AgentProfile.objects.filter(agent_id=agent_id).first()
        if not agent:
            return Response({"error": f"Agent '{agent_id}' not found."}, status=status.HTTP_404_NOT_FOUND)

        trade_mode = request.data.get("trade_mode", AgentTradeRecord.TradeMode.GHOST)
        chain = request.data.get("chain", "base").lower()
        symbol = request.data.get("symbol", "WETH-USDC").upper()
        side = request.data.get("side", AgentTradeRecord.Side.BUY).lower()
        notional_usd = float(request.data.get("notional_usd", 0.25))
        entry_price = float(request.data.get("entry_price", 2850.0))
        exit_price = float(request.data.get("exit_price", entry_price * 1.004))
        gross_pnl = float(request.data.get("gross_pnl_usd", ((exit_price - entry_price) / entry_price) * notional_usd))
        fees_usd = float(request.data.get("fees_usd", 0.0015))
        net_pnl = gross_pnl - fees_usd
        return_pct = net_pnl / notional_usd if notional_usd > 0 else 0.0

        trade = AgentTradeRecord.objects.create(
            agent=agent,
            trade_mode=trade_mode,
            chain=chain,
            symbol=symbol,
            side=side,
            notional_usd=notional_usd,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=notional_usd / entry_price if entry_price > 0 else 0.0,
            gross_pnl_usd=gross_pnl,
            fees_usd=fees_usd,
            net_pnl_usd=net_pnl,
            return_pct=return_pct,
            status=AgentTradeRecord.Status.CLOSED,
            exit_reason=request.data.get("exit_reason", "take_profit"),
        )

        # Update agent stats
        if trade_mode == AgentTradeRecord.TradeMode.LIVE:
            agent.total_live_trades += 1
            agent.live_net_pnl_usd += net_pnl
            wins = AgentTradeRecord.objects.filter(agent=agent, trade_mode=AgentTradeRecord.TradeMode.LIVE, net_pnl_usd__gt=0).count()
            agent.live_win_rate = wins / agent.total_live_trades if agent.total_live_trades > 0 else 0.0
            agent.save(update_fields=["total_live_trades", "live_net_pnl_usd", "live_win_rate"])
        else:
            agent.total_ghost_trades += 1
            wins = AgentTradeRecord.objects.filter(agent=agent, trade_mode=AgentTradeRecord.TradeMode.GHOST, net_pnl_usd__gt=0).count()
            agent.ghost_win_rate = wins / agent.total_ghost_trades if agent.total_ghost_trades > 0 else 0.0
            agent.save(update_fields=["total_ghost_trades", "ghost_win_rate"])

        return Response({"status": "recorded", "trade": _trade_dict(trade)}, status=status.HTTP_201_CREATED)


class AlphaWalletsView(APIView):
    """Discovers, tracks, and manages top-performing smart money trader wallets on-chain."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, *args, **kwargs) -> Response:
        AlphaWalletResearcher.ensure_default_alpha_wallets()
        chain = request.query_params.get("chain", "").lower()
        verif_status = request.query_params.get("status", "").lower()

        qs = AlphaTraderWallet.objects.all()
        if chain:
            qs = qs.filter(chain=chain)
        if verif_status:
            qs = qs.filter(verification_status=verif_status)

        wallets = list(qs.order_by("-win_rate", "-estimated_pnl_usd"))
        return Response({
            "count": len(wallets),
            "wallets": [_alpha_wallet_dict(w) for w in wallets],
        }, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        action = request.data.get("action", "").lower()
        if action == "scan":
            chain = request.data.get("chain")
            results = AlphaWalletResearcher.scan_onchain_alpha_wallets(chain=chain)
            return Response({"status": "scanned", "discovered_count": len(results), "wallets": results})

        # Manual wallet tracking
        chain = request.data.get("chain", "base").lower()
        wallet_address = request.data.get("wallet_address", "").strip()
        alias = request.data.get("alias", "").strip()
        if not wallet_address:
            return Response({"error": "Wallet address is required."}, status=status.HTTP_400_BAD_REQUEST)

        wallet, created = AlphaTraderWallet.objects.update_or_create(
            wallet_address=wallet_address,
            defaults={
                "chain": chain,
                "alias": alias or f"Alpha Trader ({chain.upper()})",
                "months_winning": int(request.data.get("months_winning", 3)),
                "win_rate": float(request.data.get("win_rate", 0.75)),
                "total_trades": int(request.data.get("total_trades", 30)),
                "estimated_pnl_usd": float(request.data.get("estimated_pnl_usd", 10000.0)),
                "profit_factor": float(request.data.get("profit_factor", 2.2)),
                "mirror_clip_usd": float(request.data.get("mirror_clip_usd", 0.25)),
                "notes": request.data.get("notes", "Manually added alpha wallet"),
            },
        )
        return Response({
            "status": "created" if created else "updated",
            "wallet": _alpha_wallet_dict(wallet),
        }, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)


class AlphaWalletConfirmView(APIView):
    """Processes incoming live data from the wallet, evaluates profitability,
    and immediately graduates straight to live mirror trading once confirmed."""

    permission_classes = [permissions.AllowAny]

    def post(self, request: Request, pk: int, *args, **kwargs) -> Response:
        result = AlphaWalletResearcher.confirm_and_graduate_wallet(wallet_id=pk)
        if "error" in result:
            return Response(result, status=status.HTTP_404_NOT_FOUND)
        return Response(result, status=status.HTTP_200_OK)


class AlphaWalletToggleView(APIView):
    """Enables or disables live mirror execution for an alpha wallet."""

    permission_classes = [permissions.AllowAny]

    def post(self, request: Request, pk: int, *args, **kwargs) -> Response:
        wallet = AlphaTraderWallet.objects.filter(pk=pk).first()
        if not wallet:
            return Response({"error": f"Wallet {pk} not found"}, status=status.HTTP_404_NOT_FOUND)

        wallet.is_mirror_active = not wallet.is_mirror_active
        if wallet.is_mirror_active and wallet.verification_status != AlphaTraderWallet.VerificationStatus.LIVE_MIRRORING:
            wallet.verification_status = AlphaTraderWallet.VerificationStatus.LIVE_MIRRORING
            wallet.graduated_to_live_at = timezone.now()
        wallet.save(update_fields=["is_mirror_active", "verification_status", "graduated_to_live_at"])

        return Response({
            "wallet_id": wallet.pk,
            "is_mirror_active": wallet.is_mirror_active,
            "verification_status": wallet.verification_status,
        }, status=status.HTTP_200_OK)


class ProfitMarkersView(APIView):
    """Lists granular profit accrual markers and running totals (stays put vs dispatched)."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, *args, **kwargs) -> Response:
        status_filter = request.query_params.get("status", "").lower()
        chain = request.query_params.get("chain", "").lower()

        qs = ProfitAccrualMarker.objects.all()
        if status_filter:
            qs = qs.filter(status=status_filter)
        if chain:
            qs = qs.filter(chain=chain)

        markers = list(qs[:100])
        running_totals = ProfitAccrualCoordinator.get_running_totals()
        return Response({
            "running_totals": running_totals,
            "count": len(markers),
            "markers": [_profit_marker_dict(m) for m in markers],
        }, status=status.HTTP_200_OK)


class SendAllProfitsView(APIView):
    """Manual button click to send all total profits.
    Dispatches accumulated profits that were staying put in the pool."""

    permission_classes = [permissions.AllowAny]

    def post(self, request: Request, *args, **kwargs) -> Response:
        result = ProfitAccrualCoordinator.send_all_total_profits()
        return Response(result, status=status.HTTP_200_OK)




class TokenLimitsView(APIView):
    """External Agent Token Limit & Quota Registry.
    Tracks limits for AGY, Codex, and Claude with automatic replenishment countdowns."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, *args, **kwargs) -> Response:
        summary = TokenLimiter.get_all_limits_summary()
        return Response({"token_limits": summary}, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        """Allow resetting or simulating a cooldown for an agent."""
        agent = str(request.data.get("agent") or "").strip().lower()
        action = str(request.data.get("action") or "reset").strip().lower()
        if agent:
            if action in {"clear", "reset", "ready"}:
                TokenLimiter.record_success(agent)
            elif action == "simulate_limit":
                delay = float(request.data.get("delay") or 30.0)
                TokenLimiter.record_rate_limit(agent, "Simulated token limit for testing", delay_seconds=delay)
        return Response({"token_limits": TokenLimiter.get_all_limits_summary()}, status=status.HTTP_200_OK)


class FundTrackerView(APIView):
    """Real-time fund reservation, locked balance, and outbound transfer status for BusScheduler."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, *args, **kwargs) -> Response:
        from services.fund_tracker import get_fund_tracker
        chain_filter = request.query_params.get("chain")
        tracker = get_fund_tracker()
        summary = tracker.get_summary(chain=chain_filter)
        return Response(summary, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        from services.fund_tracker import get_fund_tracker
        action = str(request.data.get("action") or "").strip().lower()
        tracker = get_fund_tracker()

        if action == "release_reservation":
            res_id = str(request.data.get("reservation_id") or "").strip()
            ok = tracker.release_reservation(res_id)
            return Response({"ok": ok, "released": res_id}, status=status.HTTP_200_OK)

        elif action == "record_outbound":
            record = tracker.record_outbound_transfer(
                chain=str(request.data.get("chain") or "base"),
                token=str(request.data.get("token") or "USDC"),
                amount=float(request.data.get("amount") or 0.0),
                destination=str(request.data.get("destination") or ""),
                reason=str(request.data.get("reason") or "manual_record"),
                tx_hash=str(request.data.get("tx_hash") or ""),
                source="operator_api",
            )
            return Response({"ok": True, "transfer": record}, status=status.HTTP_201_CREATED)

        elif action == "clear_all":
            tracker.clear_all()
            return Response({"ok": True, "message": "all fund reservations cleared"}, status=status.HTTP_200_OK)

        return Response({"error": f"Unknown action: {action}"}, status=status.HTTP_400_BAD_REQUEST)
