"""HTTP surface for GA searches over trainable brains.

The UI is not built yet; this is the contract it will bind to, so the screens
end up a thin view over working logic instead of the place the logic gets
invented. Every endpoint here is already exercised by the service layer and
tests.

    GET    /api/model-control/ga/space/           gene space (form schema)
    GET    /api/model-control/ga/runs/            all searches, running first
    POST   /api/model-control/ga/runs/            start a search (any scope)
    GET    /api/model-control/ga/runs/<id>/       one search + generation history
    PATCH  /api/model-control/ga/runs/<id>/       modify a RUNNING search
    DELETE /api/model-control/ga/runs/<id>/       stop a running search
    POST   /api/model-control/ga/runs/<id>/promote/   register champion as a model
    GET    /api/model-control/ga/models/          selectable models (dropdown)
    POST   /api/model-control/ga/models/          set the default model
"""

from __future__ import annotations

import json
from typing import Any, Dict

from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from services import ga_service
from trading.genome.ga import GENE_SPACE


def _body(request) -> Dict[str, Any]:
    try:
        return json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        return {}


def _forbidden():
    return JsonResponse({"error": "authentication required"}, status=403)


class GeneSpaceView(View):
    """The searchable space, shaped for a form builder.

    Each gene reports its kind so the UI can render a slider for a range, a
    multi-select for a choice list, and a pin toggle for either.
    """

    def get(self, request):
        if not request.user.is_authenticated:
            return _forbidden()
        out = {}
        for key, spec in GENE_SPACE.items():
            if isinstance(spec, tuple):
                out[key] = {"kind": "range", "min": spec[0], "max": spec[1]}
            elif isinstance(spec, list):
                out[key] = {"kind": "choice", "choices": spec}
        return JsonResponse({
            "space": out,
            "groups": {
                "shape": ["window", "horizon", "shape_points", "shape_margin",
                          "min_occurrences", "min_consistency", "min_abs_return"],
                "sentiment": ["sentiment_weight", "sentiment_margin", "sentiment_lookback_h"],
                "brain": ["brain_input_pool", "brain_outcome_pool", "brain_weight",
                          "brain_min_confidence"],
                "execution": ["entry_percentile", "target_margin", "stop_margin",
                              "max_hold_bars", "min_volume_ratio"],
            },
            "notes": {
                "fitness": "out-of-sample edge over the majority-class baseline, "
                           "times expectancy, discounted by sample size",
                "why": "shapes selected in-sample scored 62-81% and 50.0% "
                       "out-of-sample against a 53.5% baseline; only held-out "
                       "performance counts",
            },
        })


@method_decorator(csrf_exempt, name="dispatch")
class GARunListView(View):
    def get(self, request):
        if not request.user.is_authenticated:
            return _forbidden()
        return JsonResponse({"runs": ga_service.list_runs()})

    def post(self, request):
        if not request.user.is_authenticated:
            return _forbidden()
        payload = _body(request)
        run_id = ga_service.start_run(
            name=str(payload.get("name") or ""),
            space_overrides=payload.get("space") or {},
            symbols=payload.get("symbols") or None,
            chains=payload.get("chains") or None,
            population=int(payload.get("population") or 16),
            generations=int(payload.get("generations") or 6),
            max_symbols=int(payload.get("max_symbols") or 6),
            use_sentiment=bool(payload.get("use_sentiment", True)),
            use_brain=bool(payload.get("use_brain", False)),
            seed=int(payload.get("seed") or 0),
        )
        return JsonResponse({"run_id": run_id, "status": "started"}, status=201)


@method_decorator(csrf_exempt, name="dispatch")
class GARunDetailView(View):
    def get(self, request, run_id: str):
        if not request.user.is_authenticated:
            return _forbidden()
        run = ga_service.get_run(run_id)
        if not run:
            return JsonResponse({"error": "not found"}, status=404)
        return JsonResponse(run)

    def patch(self, request, run_id: str):
        """Retarget a search that is already running."""
        if not request.user.is_authenticated:
            return _forbidden()
        if not ga_service.update_run(run_id, _body(request)):
            return JsonResponse({"error": "not found"}, status=404)
        return JsonResponse({"run_id": run_id, "status": "config queued"})

    def delete(self, request, run_id: str):
        if not request.user.is_authenticated:
            return _forbidden()
        if not ga_service.stop_run(run_id):
            return JsonResponse({"error": "not running"}, status=404)
        return JsonResponse({"run_id": run_id, "status": "stopping"})


@method_decorator(csrf_exempt, name="dispatch")
class GAPromoteView(View):
    """Register a finished champion as a selectable model.

    Refused when the champion has no out-of-sample edge -- the dropdown must
    never become a menu of overfits.
    """

    def post(self, request, run_id: str):
        if not request.user.is_authenticated:
            return _forbidden()
        payload = _body(request)
        entry = ga_service.register_champion(
            run_id,
            name=str(payload.get("name") or ""),
            make_default=bool(payload.get("make_default", False)),
        )
        if entry is None:
            return JsonResponse(
                {"error": "champion has no out-of-sample edge; not registered"},
                status=422,
            )
        return JsonResponse(entry, status=201)


@method_decorator(csrf_exempt, name="dispatch")
class GAModelListView(View):
    def get(self, request):
        if not request.user.is_authenticated:
            return _forbidden()
        return JsonResponse({"models": ga_service.list_models()})

    def post(self, request):
        if not request.user.is_authenticated:
            return _forbidden()
        model_id = str(_body(request).get("model_id") or "")
        if not ga_service.set_default_model(model_id):
            return JsonResponse({"error": "unknown model"}, status=404)
        return JsonResponse({"default": model_id})
