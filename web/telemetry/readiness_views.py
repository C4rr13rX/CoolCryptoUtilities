"""
Autonomous-trading readiness endpoint.

Surfaces the same report as ``scripts/readiness_report.py`` so the question
"can it trade on its own yet?" has one answer, visible in the dashboard,
derived from the ghost ledger rather than from anyone's sense of how long it
has been.

Read-only by design. Graduation is decided by the ledger as evidence arrives;
nothing here can promote a strategy.
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


class TradingReadinessView(APIView):
    """
    GET /api/telemetry/readiness/

    Per-strategy graduation progress, the Wilson lower bound on each win rate,
    the observed evidence rate, and an ETA to the first graduation.

    The lower bound is the number to watch rather than the raw win rate: 7/7
    reads as 100% but is only evidence of ~49% at 95% confidence, which is
    below the 55% bar. That gap is precisely why trade count cannot be waived.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        # Loaded by path, not as `scripts.readiness_report`: adding an
        # __init__.py to scripts/ would make it a package and shadow the
        # W1z4rD repo's own `scripts.market_evolution_service`, which the
        # genome champion gate imports.
        import importlib.util

        module_path = ROOT / "scripts" / "readiness_report.py"
        if not module_path.is_file():
            return Response({"error": "readiness report unavailable"}, status=503)
        try:
            spec = importlib.util.spec_from_file_location(
                "_readiness_report", module_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return Response(module.collect())
        except Exception as exc:  # noqa: BLE001
            return Response(
                {"error": f"readiness report failed: {exc}"}, status=503
            )
