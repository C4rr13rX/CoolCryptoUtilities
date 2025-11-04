from __future__ import annotations

from typing import Dict, List

from django.db.models import Max
from rest_framework import generics, status
from rest_framework.request import Request
from rest_framework.response import Response

from .models import MarketStream
from .serializers import MarketStreamSerializer


class LatestSampleView(generics.GenericAPIView):
    serializer_class = MarketStreamSerializer

    def get(self, request: Request, *args, **kwargs) -> Response:
        symbols = request.query_params.getlist("symbol")
        if symbols:
            items = []
            for symbol in symbols:
                sample = (
                    MarketStream.objects.filter(symbol=symbol.upper())
                    .order_by("-ts")
                    .first()
                )
                if sample:
                    items.append(self.get_serializer(sample).data)
            if not items:
                return Response({"detail": "No samples found."}, status=status.HTTP_404_NOT_FOUND)
            return Response(items)

        latest_map: Dict[str, Dict[str, float]] = {}
        aggregates = (
            MarketStream.objects.values("symbol")
            .annotate(latest_ts=Max("ts"))
            .order_by("symbol")
        )
        for entry in aggregates:
            symbol = entry["symbol"]
            ts = entry["latest_ts"]
            sample = (
                MarketStream.objects.filter(symbol=symbol, ts=ts)
                .order_by("-id")
                .first()
            )
            if sample:
                latest_map[symbol] = self.get_serializer(sample).data
        return Response(latest_map)


class RecentSamplesView(generics.ListAPIView):
    serializer_class = MarketStreamSerializer

    def get_queryset(self):
        qs = MarketStream.objects.all()
        symbol = self.request.query_params.get("symbol")
        if symbol:
            qs = qs.filter(symbol=symbol.upper())
        limit = int(self.request.query_params.get("limit", "100"))
        limit = max(1, min(limit, 1000))
        return qs.order_by("-ts")[:limit]
