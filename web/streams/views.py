from __future__ import annotations

import json
import time
from typing import Dict, List

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

        now = time.time()
        # Single-query JOIN for latest record per symbol within active window (48 hours)
        query = """
            SELECT m.*
            FROM market_stream m
            INNER JOIN (
                SELECT symbol, MAX(id) as max_id
                FROM market_stream
                WHERE ts > %s
                GROUP BY symbol
            ) latest ON m.id = latest.max_id
        """
        samples = list(MarketStream.objects.raw(query, [now - 172800.0]))
        if not samples:
            # Fall back to 30 days if trading was paused
            samples = list(MarketStream.objects.raw(query, [now - 2592000.0]))
        if not samples:
            # Absolute fallback: most recent 100 symbols ever seen
            query_all = """
                SELECT m.*
                FROM market_stream m
                INNER JOIN (
                    SELECT symbol, MAX(id) as max_id
                    FROM market_stream
                    GROUP BY symbol
                    ORDER BY max_id DESC
                    LIMIT 100
                ) latest ON m.id = latest.max_id
            """
            samples = list(MarketStream.objects.raw(query_all))

        latest_map: Dict[str, dict] = {}
        for sample in samples:
            raw_val = sample.raw
            if isinstance(raw_val, str):
                try:
                    raw_val = json.loads(raw_val)
                except Exception:
                    pass
            source_val = ""
            if isinstance(raw_val, dict):
                source_val = raw_val.get("source") or raw_val.get("rest", "")
            latest_map[sample.symbol] = {
                "ts": sample.ts,
                "chain": sample.chain,
                "symbol": sample.symbol,
                "price": sample.price,
                "volume": sample.volume,
                "raw": raw_val,
                "source": source_val,
            }
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
