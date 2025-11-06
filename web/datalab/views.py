from __future__ import annotations

from datetime import datetime, timezone

from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from services.data_lab import fetch_news, get_runner, list_datasets


class DatasetListView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        chain = request.query_params.get("chain")
        category = request.query_params.get("category")
        sort_key = request.query_params.get("sort", "modified")
        order = request.query_params.get("order", "desc")
        entries = list_datasets(chain=chain, category=category, sort_key=sort_key, order=order)
        return Response({"items": entries, "count": len(entries)}, status=status.HTTP_200_OK)


class RunJobView(APIView):
    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        job_type = str(payload.get("job_type") or "").strip()
        if not job_type:
            return Response({"detail": "job_type is required"}, status=status.HTTP_400_BAD_REQUEST)
        options = payload.get("options") or {}
        runner = get_runner()
        try:
            runner.start(job_type, options)
        except RuntimeError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_409_CONFLICT)
        except Exception as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(runner.status(), status=status.HTTP_202_ACCEPTED)


class JobStatusView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        runner = get_runner()
        return Response(runner.status(), status=status.HTTP_200_OK)


class NewsFetchView(APIView):
    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        tokens = payload.get("tokens") or []
        if not isinstance(tokens, list):
            return Response({"detail": "tokens must be a list"}, status=status.HTTP_400_BAD_REQUEST)
        start_raw = payload.get("start")
        end_raw = payload.get("end")
        if not start_raw or not end_raw:
            return Response({"detail": "start and end timestamps are required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            start_dt = datetime.fromisoformat(str(start_raw).replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(str(end_raw).replace("Z", "+00:00"))
        except ValueError:
            return Response({"detail": "start and end must be ISO timestamps"}, status=status.HTTP_400_BAD_REQUEST)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        query = payload.get("query")
        max_pages = payload.get("max_pages")
        try:
            result = fetch_news(tokens=tokens, start=start_dt, end=end_dt, query=query, max_pages=max_pages)
        except Exception as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(result, status=status.HTTP_200_OK)
