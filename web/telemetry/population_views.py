"""The whole strategy population as one endpoint.

Serves ``services.strategy_population.collect`` so the pipeline page can show
every strategy the system knows about -- candidate, backtest, ghost, live and
rejected -- rather than only the ones that happen to have a ledger row.

Read-only by design, exactly like ``readiness_views``: graduation is decided
by the ledger as evidence arrives, and nothing reachable from here promotes,
demotes or writes anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class StrategyPopulationView(APIView):
    """
    GET /api/telemetry/strategies/population/

    Every strategy, its stage, how long it has held that stage, its measured
    performance, and what is blocking its next promotion.

    The number to read on each row is ``tradeable``, not ``ghost``. Pooled
    ghost counts are what make the population look busy; the live lane can
    only ever spend on the live-tradeable subset, and that is the population
    both graduation and re-arm consult. Measured 2026-09-07 the two differ by
    56x across the population -- 394 pooled ghost round trips against 7
    live-tradeable ones -- which is why a page that shows only the first
    reports a healthy pipeline and a page that shows both reports the truth.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        try:
            from services.strategy_population import collect

            return Response(collect())
        except Exception as exc:  # noqa: BLE001
            # 503 rather than 500: the population is derived from two JSON
            # files that production rewrites under a lock, so a failure here
            # is far more likely to be "read it again in a moment" than a bug
            # in the caller's request.
            return Response(
                {"error": f"strategy population unavailable: {exc}"}, status=503
            )
