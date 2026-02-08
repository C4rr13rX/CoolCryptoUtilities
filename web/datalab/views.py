from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from discovery.models import DiscoveredToken, HoneypotCheck
from datalab.models import NewsSource, NewsSourceArticle
from django.utils import timezone
from services.data_lab import fetch_news, get_runner, list_datasets
from services.custom_news_sources import fetch_source_articles
from services.signal_scanner import DEFAULT_SCAN_CHAIN, WINDOW_OPTIONS, scan_price_signals
from services.watchlists import load_watchlists, mutate_watchlist


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
            user = request.user if request.user and request.user.is_authenticated else None
            runner.start(job_type, options, user=user)
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


class SignalListView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        window = request.query_params.get("window") or "24h"
        if window not in WINDOW_OPTIONS:
            window = "24h"
        direction = request.query_params.get("direction", "bullish").lower()
        if direction not in {"bullish", "bearish", "all"}:
            direction = "bullish"
        limit_param = request.query_params.get("limit", "60")
        min_volume_param = request.query_params.get("min_volume", "0")
        try:
            limit = max(1, min(int(limit_param), 200))
        except ValueError:
            limit = 60
        try:
            min_volume = max(0.0, float(min_volume_param))
        except ValueError:
            min_volume = 0.0

        signals, scan_meta = scan_price_signals(
            window,
            direction="bullish" if direction == "all" else direction,  # scanner expects "bullish"/"bearish"
            limit=limit,
            min_volume=min_volume,
        )
        if direction == "all":
            # merge both bullish and bearish lists
            bear_signals, bear_meta = scan_price_signals(window, direction="bearish", limit=limit, min_volume=min_volume)
            scan_meta["bearish_hits"] = bear_meta.get("result_count", 0)
            union_map: Dict[str, Dict[str, object]] = {item["symbol"]: item for item in signals}
            for item in bear_signals:
                union_map.setdefault(item["symbol"], item)
            signals = list(union_map.values())
        watchlists = load_watchlists()
        membership: Dict[str, set[str]] = {
            name: set(items) for name, items in watchlists.items()
        }

        symbols = [str(entry.get("symbol") or "").upper() for entry in signals if entry.get("symbol")]
        risk_map: Dict[str, Dict[str, object]] = {}
        token_map: Dict[str, DiscoveredToken] = {}
        if symbols:
            token_map = {
                token.symbol.upper(): token
                for token in DiscoveredToken.objects.filter(symbol__in=symbols)
            }
            latest_checks: Dict[str, HoneypotCheck] = {}
            for check in HoneypotCheck.objects.filter(symbol__in=symbols).order_by("symbol", "-created_at"):
                symbol_key = check.symbol.upper()
                if symbol_key in latest_checks:
                    continue
                latest_checks[symbol_key] = check
            for symbol in symbols:
                token = token_map.get(symbol)
                check = latest_checks.get(symbol)
                risk: Dict[str, object] = {}
                if token:
                    risk["status"] = token.status
                    risk["last_updated"] = token.last_updated.isoformat()
                if check:
                    risk["verdict"] = check.verdict
                    risk["confidence"] = check.confidence
                    risk["checked_at"] = check.created_at.isoformat()
                    if check.details:
                        risk["details"] = check.details
                if risk:
                    risk_map[symbol] = risk

        default_chain = (DEFAULT_SCAN_CHAIN or "base").lower()
        enriched: List[Dict[str, object]] = []
        for item in signals:
            symbol = str(item.get("symbol") or "").upper()
            entry = dict(item)
            chain_value = str(entry.get("chain") or "").strip()
            if not chain_value or chain_value.lower() == "public":
                token = token_map.get(symbol)
                if token and token.chain:
                    entry["chain"] = token.chain
                else:
                    entry["chain"] = default_chain
            entry["watchlists"] = {
                name: symbol in members for name, members in membership.items()
            }
            if symbol in risk_map:
                entry["risk"] = risk_map[symbol]
            enriched.append(entry)
        return Response(
            {
                "items": enriched,
                "count": len(enriched),
                "watchlists": watchlists,
                "window": window,
                "meta": scan_meta,
            },
            status=status.HTTP_200_OK,
        )


class WatchlistView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        data = load_watchlists()
        return Response({"watchlists": data}, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        target = str(payload.get("target") or "").lower()
        action = str(payload.get("action") or "add").lower()
        symbols = payload.get("symbols") or []
        if not isinstance(symbols, list):
            return Response({"detail": "symbols must be a list"}, status=status.HTTP_400_BAD_REQUEST)
        kwargs = {}
        if action == "remove":
            kwargs["remove"] = symbols
        elif action in {"set", "replace"}:
            kwargs["replace"] = symbols
        else:
            kwargs["add"] = symbols
        try:
            updated = mutate_watchlist(target, **kwargs)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"watchlists": updated}, status=status.HTTP_200_OK)


class NewsSourceListView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        sources = NewsSource.objects.order_by("-updated_at")
        payload = [
            {
                "id": src.id,
                "name": src.name,
                "base_url": src.base_url,
                "active": src.active,
                "parser_config": src.parser_config or {},
                "last_error": src.last_error or "",
                "last_run_at": src.last_run_at.isoformat() if src.last_run_at else None,
                "updated_at": src.updated_at.isoformat() if src.updated_at else None,
            }
            for src in sources
        ]
        return Response({"items": payload, "count": len(payload)}, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        name = str(payload.get("name") or "").strip()
        base_url = str(payload.get("base_url") or "").strip()
        if not name or not base_url:
            return Response({"detail": "name and base_url are required"}, status=status.HTTP_400_BAD_REQUEST)
        source = NewsSource.objects.create(
            name=name,
            base_url=base_url,
            active=bool(payload.get("active", True)),
            parser_config=payload.get("parser_config") or {},
        )
        return Response(
            {
                "item": {
                    "id": source.id,
                    "name": source.name,
                    "base_url": source.base_url,
                    "active": source.active,
                    "parser_config": source.parser_config or {},
                }
            },
            status=status.HTTP_201_CREATED,
        )


class NewsSourceTestView(APIView):
    def post(self, request: Request, source_id: int, *args, **kwargs) -> Response:
        try:
            source = NewsSource.objects.get(id=source_id)
        except NewsSource.DoesNotExist:
            return Response({"detail": "source not found"}, status=status.HTTP_404_NOT_FOUND)
        payload = request.data or {}
        max_items = int(payload.get("max_items", 6))
        try:
            items = fetch_source_articles(
                source.base_url,
                config=source.parser_config or {},
                max_items=max_items,
            )
        except Exception as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"items": items, "count": len(items)}, status=status.HTTP_200_OK)


class NewsSourceRunView(APIView):
    def post(self, request: Request, source_id: int, *args, **kwargs) -> Response:
        try:
            source = NewsSource.objects.get(id=source_id)
        except NewsSource.DoesNotExist:
            return Response({"detail": "source not found"}, status=status.HTTP_404_NOT_FOUND)
        payload = request.data or {}
        max_items = int(payload.get("max_items", 12))
        try:
            items = fetch_source_articles(
                source.base_url,
                config=source.parser_config or {},
                max_items=max_items,
            )
        except Exception as exc:
            source.last_error = str(exc)
            source.last_run_at = timezone.now()
            source.save(update_fields=["last_error", "last_run_at", "updated_at"])
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        created = 0
        for item in items:
            NewsSourceArticle.objects.create(
                source=source,
                title=item.get("title") or "",
                url=item.get("url") or "",
                summary=item.get("summary") or "",
                content=item.get("summary") or "",
                metadata=item,
            )
            created += 1
        source.last_error = ""
        source.last_run_at = timezone.now()
        source.save(update_fields=["last_error", "last_run_at", "updated_at"])
        return Response({"saved": created, "count": len(items)}, status=status.HTTP_200_OK)
