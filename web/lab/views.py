from __future__ import annotations

from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from services.model_lab import LabJobConfig, get_model_lab_runner


class LabFilesView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        runner = get_model_lab_runner()
        return Response({"files": runner.list_files()}, status=status.HTTP_200_OK)


class LabStatusView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        runner = get_model_lab_runner()
        return Response(runner.get_status(), status=status.HTTP_200_OK)


class LabStartView(APIView):
    def post(self, request: Request, *args, **kwargs) -> Response:
        runner = get_model_lab_runner()
        payload = request.data or {}
        train_files = payload.get("train_files") or []
        eval_files = payload.get("eval_files") or []
        try:
            epochs = max(1, int(payload.get("epochs", 1)))
        except (TypeError, ValueError):
            return Response({"detail": "epochs must be an integer"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            batch_size = max(8, int(payload.get("batch_size", 16)))
        except (TypeError, ValueError):
            return Response({"detail": "batch_size must be an integer"}, status=status.HTTP_400_BAD_REQUEST)
        config = LabJobConfig(
            train_files=[str(item) for item in train_files],
            eval_files=[str(item) for item in eval_files],
            epochs=epochs,
            batch_size=batch_size,
        )
        try:
            runner.start_job(config)
        except RuntimeError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_409_CONFLICT)
        except Exception as exc:  # pragma: no cover - defensive
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(runner.get_status(), status=status.HTTP_202_ACCEPTED)
