"""HTTP surface for the strategies screen.

    GET    /api/model-control/strategies/                  list + live records
    POST   /api/model-control/strategies/                  register a strategy
    GET    /api/model-control/strategies/objectives/       selectable objectives
    GET    /api/model-control/strategies/<id>/             one strategy
    DELETE /api/model-control/strategies/<id>/             remove it
    POST   /api/model-control/strategies/<id>/commission/  commission/decommission
    GET    /api/model-control/strategies/<id>/compare/     results across models
    POST   /api/model-control/strategies/<id>/experiment/  re-run on another model
"""

from __future__ import annotations

import json
from typing import Any, Dict

from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from services import ga_service, strategy_registry


def _body(request) -> Dict[str, Any]:
    try:
        return json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        return {}


def _forbidden():
    return JsonResponse({"error": "authentication required"}, status=403)


def _ledger_record(strategy_id: str) -> Dict[str, Any]:
    """Live ghost/live record, so the screen shows earned performance next to
    the search-time metrics rather than only the latter."""
    try:
        from trading.strategies.ledger import StrategyLedger
        entry = StrategyLedger()._data.get(strategy_id) or {}
    except Exception:
        return {}
    return {
        "ghost": entry.get("ghost", {}),
        "live": entry.get("live", {}),
        "live_approved": entry.get("live_approved", False),
    }


class ObjectiveListView(View):
    def get(self, request):
        if not request.user.is_authenticated:
            return _forbidden()
        return JsonResponse({
            "objectives": [
                {"key": k, "label": v} for k, v in strategy_registry.OBJECTIVE_LABELS.items()
            ],
            "note": "Every objective is scored OUT-OF-SAMPLE. Choosing one "
                    "changes what the search rewards, never whether the reward "
                    "was actually earned on held-out data.",
        })


@method_decorator(csrf_exempt, name="dispatch")
class StrategyListView(View):
    def get(self, request):
        if not request.user.is_authenticated:
            return _forbidden()
        rows = strategy_registry.list_strategies()
        for row in rows:
            sid = row.get("strategy_id", "")
            row["ledger"] = _ledger_record(sid)
            # Lifetime survives ledger resets, so it is the honest answer to
            # "how has this strategy ever actually done".
            row["lifetime"] = strategy_registry.lifetime_metrics(sid)
        return JsonResponse({"strategies": rows})

    def post(self, request):
        if not request.user.is_authenticated:
            return _forbidden()
        p = _body(request)
        entry = strategy_registry.register_strategy(
            name=str(p.get("name") or ""),
            kind=str(p.get("kind") or "genome"),
            genes=p.get("genes") or {},
            run_id=str(p.get("run_id") or ""),
            objective=str(p.get("objective") or "balanced"),
            model_id=str(p.get("model_id") or ""),
            model_name=str(p.get("model_name") or ""),
            metrics=p.get("metrics") or {},
            commissioned=bool(p.get("commissioned", False)),
        )
        return JsonResponse(entry, status=201)


@method_decorator(csrf_exempt, name="dispatch")
class StrategyDetailView(View):
    def get(self, request, strategy_id: str):
        if not request.user.is_authenticated:
            return _forbidden()
        entry = strategy_registry.get_strategy(strategy_id)
        if not entry:
            return JsonResponse({"error": "not found"}, status=404)
        entry["ledger"] = _ledger_record(strategy_id)
        entry["lifetime"] = strategy_registry.lifetime_metrics(strategy_id)
        return JsonResponse(entry)

    def delete(self, request, strategy_id: str):
        if not request.user.is_authenticated:
            return _forbidden()
        if not strategy_registry.delete_strategy(strategy_id):
            return JsonResponse({"error": "not found"}, status=404)
        return JsonResponse({"deleted": strategy_id})


@method_decorator(csrf_exempt, name="dispatch")
class StrategyCommissionView(View):
    def post(self, request, strategy_id: str):
        if not request.user.is_authenticated:
            return _forbidden()
        want = bool(_body(request).get("commissioned", True))
        entry = strategy_registry.set_commissioned(strategy_id, want)
        if entry is None:
            return JsonResponse({"error": "not found"}, status=404)
        if want and not entry.get("commissioned"):
            return JsonResponse(
                {"error": entry.get("commission_error", "refused"), "strategy": entry},
                status=422,
            )
        return JsonResponse(entry)


class StrategyCompareView(View):
    def get(self, request, strategy_id: str):
        if not request.user.is_authenticated:
            return _forbidden()
        rows = strategy_registry.compare_experiments(strategy_id)
        if not rows:
            return JsonResponse({"error": "not found"}, status=404)
        return JsonResponse({"strategy_id": strategy_id, "results": rows})


@method_decorator(csrf_exempt, name="dispatch")
class StrategyExperimentView(View):
    """Re-run this strategy's search against a different model/brain."""

    def post(self, request, strategy_id: str):
        if not request.user.is_authenticated:
            return _forbidden()
        entry = strategy_registry.get_strategy(strategy_id)
        if not entry:
            return JsonResponse({"error": "not found"}, status=404)
        p = _body(request)
        # Pin the search to this strategy's genes unless the caller widens it,
        # so the comparison isolates the model rather than the gene space.
        overrides = p.get("space") or {
            k: v for k, v in (entry.get("genes") or {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        run_id = ga_service.start_run(
            name="%s @ %s" % (entry.get("name"), p.get("model_name") or p.get("model_id") or "no-brain"),
            space_overrides=overrides,
            symbols=p.get("symbols") or None,
            population=int(p.get("population") or 12),
            generations=int(p.get("generations") or 4),
            use_brain=bool(p.get("model_id")),
            use_sentiment=bool(p.get("use_sentiment", True)),
        )
        strategy_registry.add_experiment(
            strategy_id,
            run_id=run_id,
            model_id=str(p.get("model_id") or ""),
            model_name=str(p.get("model_name") or ""),
            objective=str(p.get("objective") or entry.get("objective") or "balanced"),
            metrics={"status": "running"},
        )
        return JsonResponse({"run_id": run_id, "strategy_id": strategy_id}, status=201)
