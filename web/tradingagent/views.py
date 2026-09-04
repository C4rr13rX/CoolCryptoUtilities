"""The agent's API: what it has done, what it believes, and its settings.

Read-mostly by design. The only writes are configuration and a manual run
trigger, because everything the agent learns should come from what it
measured, not from what someone typed into a form.

One write is deliberately restricted: the tier. Promoting the agent to spend
real money is not a UI toggle -- it goes through :func:`promote_tier`, which
refuses unless the evidence at the current tier justifies the next one.
"""

from __future__ import annotations

from typing import Any, Dict

from django.utils import timezone
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import (AgentConfig, AgentRun, Constraint, Experiment,
                     LossRecovery, RiskTier)


def _run_dict(run: AgentRun) -> Dict[str, Any]:
    return {
        "id": run.pk,
        "agent": run.agent,
        "status": run.status,
        "report": run.report,
        "decisions": run.decisions,
        "observations": run.observations,
        "tokens_considered": run.tokens_considered,
        "trades_opened": run.trades_opened,
        "trades_closed": run.trades_closed,
        "net_pl": run.net_pl,
        "score": run.score,
        "score_parts": run.score_parts,
        "duration_sec": run.duration_sec,
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


class StatusView(APIView):
    """Everything the dashboard needs in one call."""

    def get(self, request: Request, *args, **kwargs) -> Response:
        from .engine import market_snapshot, performance_snapshot, wallet_snapshot

        config = AgentConfig.load()
        market = market_snapshot(limit=config.max_tokens_tracked)
        symbols = market.get("symbols", [])

        return Response({
            "config": {
                "enabled": config.enabled,
                "tier": config.tier,
                "clip_usd": config.clip_usd,
                "agent": config.agent,
                "interval_sec": config.interval_sec,
                "max_daily_loss_usd": config.max_daily_loss_usd,
                "max_open_positions": config.max_open_positions,
                "max_tokens_tracked": config.max_tokens_tracked,
                "halted_reason": config.halted_reason,
            },
            "performance": performance_snapshot(),
            "wallet": wallet_snapshot(),
            "market": {
                "ticks_10m": market.get("ticks_10m"),
                "bulls": [s for s in symbols if s["bias"] == "bull"][:10],
                "bears": [s for s in symbols if s["bias"] == "bear"][:10],
                "tracked": len(symbols),
            },
            "counts": {
                "constraints_active": Constraint.objects.filter(
                    status=Constraint.Status.ACTIVE).count(),
                "constraints_proposed": Constraint.objects.filter(
                    status=Constraint.Status.PROPOSED).count(),
                "experiments_ghost": Experiment.objects.filter(
                    status=Experiment.Status.GHOST).count(),
                "experiments_live": Experiment.objects.filter(
                    status=Experiment.Status.LIVE).count(),
                "recoveries_active": LossRecovery.objects.filter(
                    status=LossRecovery.Status.ACTIVE).count(),
            },
        }, status=status.HTTP_200_OK)


class ConfigView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        config = AgentConfig.load()
        return Response({
            "enabled": config.enabled,
            "tier": config.tier,
            "clip_usd": config.clip_usd,
            "agent": config.agent,
            "interval_sec": config.interval_sec,
            "max_daily_loss_usd": config.max_daily_loss_usd,
            "max_open_positions": config.max_open_positions,
            "max_tokens_tracked": config.max_tokens_tracked,
            "halted_reason": config.halted_reason,
            "tiers": [{"value": t.value, "label": t.label,
                       "clip_usd": RiskTier.clip_usd(t.value)}
                      for t in RiskTier],
        }, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        config = AgentConfig.load()

        for field, caster in (("enabled", bool), ("agent", str),
                              ("interval_sec", int),
                              ("max_daily_loss_usd", float),
                              ("max_open_positions", int),
                              ("max_tokens_tracked", int)):
            if field in payload:
                try:
                    setattr(config, field, caster(payload[field]))
                except (TypeError, ValueError):
                    return Response(
                        {"detail": f"{field} must be {caster.__name__}"},
                        status=status.HTTP_400_BAD_REQUEST)

        # Tier is NOT settable here. Spending real money is earned by measured
        # results, so it goes through the promote endpoint, which checks them.
        if "tier" in payload and str(payload["tier"]) != config.tier:
            return Response(
                {"detail": "tier cannot be set directly; use /promote/, which "
                           "checks the evidence for the next tier"},
                status=status.HTTP_400_BAD_REQUEST)

        config.save()
        return Response({"ok": True, "tier": config.tier,
                         "enabled": config.enabled}, status=status.HTTP_200_OK)


class PromoteView(APIView):
    """Move up or down the risk ladder, on evidence rather than on request."""

    #: What the tier below must have produced before the next one is unlocked.
    MIN_TRADES = 20
    MIN_NET_PL = 0.0

    def post(self, request: Request, *args, **kwargs) -> Response:
        config = AgentConfig.load()
        direction = str((request.data or {}).get("direction") or "up").lower()

        if direction == "down":
            # Demotion never needs justifying. Reducing risk is always allowed.
            order = ["ghost", "micro", "small", "normal"]
            index = max(0, order.index(config.tier) - 1) if config.tier in order else 0
            config.tier = order[index]
            config.save()
            return Response({"ok": True, "tier": config.tier,
                             "reason": "demoted on request"})

        from .engine import performance_snapshot

        perf = performance_snapshot()
        closed = int(perf.get("trades_closed") or 0)
        net = float(perf.get("net_pl") or 0.0)

        if closed < self.MIN_TRADES:
            return Response(
                {"ok": False, "tier": config.tier,
                 "detail": f"{closed} closed trades; {self.MIN_TRADES} needed "
                           f"before risking more. A smaller sample is noise."},
                status=status.HTTP_409_CONFLICT)
        if net <= self.MIN_NET_PL:
            return Response(
                {"ok": False, "tier": config.tier,
                 "detail": f"net P/L {net:+.6f} at this tier. Promotion "
                           f"requires the account to have grown."},
                status=status.HTTP_409_CONFLICT)

        config.tier = RiskTier.next_tier(config.tier)
        config.save()
        return Response({"ok": True, "tier": config.tier,
                         "clip_usd": config.clip_usd,
                         "reason": f"{closed} trades, net {net:+.6f}"})


class RunListView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        try:
            limit = min(int(request.query_params.get("limit", 50)), 200)
        except (TypeError, ValueError):
            limit = 50
        runs = AgentRun.objects.all()[:limit]
        return Response({"runs": [_run_dict(r) for r in runs]},
                        status=status.HTTP_200_OK)


class RunNowView(APIView):
    """Trigger one pass by hand, for watching it work."""

    def post(self, request: Request, *args, **kwargs) -> Response:
        from .engine import run_once

        run = run_once()
        return Response(_run_dict(run), status=status.HTTP_200_OK)


class ConstraintListView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        rows = Constraint.objects.all()[:200]
        return Response({"constraints": [{
            "id": c.pk, "rule": c.rule, "kind": c.kind, "status": c.status,
            "rationale": c.rationale, "trades_under": c.trades_under,
            "net_pl_under": c.net_pl_under, "win_rate_under": c.win_rate_under,
            "updated_at": c.updated_at.isoformat(),
        } for c in rows]}, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        """Activate, suspend or retire a constraint the agent proposed."""
        payload = request.data or {}
        try:
            constraint = Constraint.objects.get(pk=payload.get("id"))
        except Constraint.DoesNotExist:
            return Response({"detail": "no such constraint"},
                            status=status.HTTP_404_NOT_FOUND)
        new_status = str(payload.get("status") or "")
        if new_status not in Constraint.Status.values:
            return Response({"detail": f"status must be one of "
                                       f"{Constraint.Status.values}"},
                            status=status.HTTP_400_BAD_REQUEST)
        constraint.status = new_status
        constraint.updated_at = timezone.now()
        constraint.save()
        return Response({"ok": True, "id": constraint.pk,
                         "status": constraint.status})


class ExperimentListView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        rows = Experiment.objects.all()[:200]
        return Response({"experiments": [{
            "id": e.pk, "hypothesis": e.hypothesis, "metric": e.metric,
            "target": e.target, "min_trades": e.min_trades,
            "status": e.status, "tier": e.tier,
            "ghost_trades": e.ghost_trades, "ghost_net_pl": e.ghost_net_pl,
            "live_trades": e.live_trades, "live_net_pl": e.live_net_pl,
            "max_loss_usd": e.max_loss_usd,
            "ready_to_graduate": e.ready_to_graduate,
            "updated_at": e.updated_at.isoformat(),
        } for e in rows]}, status=status.HTTP_200_OK)


class LossRecoveryListView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        rows = LossRecovery.objects.all()[:200]
        return Response({"recoveries": [{
            "id": r.pk, "trigger": r.trigger, "action": r.action,
            "status": r.status, "times_triggered": r.times_triggered,
            "avg_loss_without": r.avg_loss_without,
            "avg_loss_with": r.avg_loss_with,
            "saved_per_trigger": r.saved_per_trigger,
            "updated_at": r.updated_at.isoformat(),
        } for r in rows]}, status=status.HTTP_200_OK)
