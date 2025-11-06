from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from django.db import transaction

from discovery.models import (
    DiscoveredToken,
    DiscoveryEvent,
    HoneypotCheck,
    SwapProbe,
)
from services.discovery.security_checks import goplus_token_security, heuristic_screen
from services.discovery.trending_fetcher import TrendingToken, fetch_trending_tokens


logger = logging.getLogger(__name__)


class DiscoveryCoordinator:
    """Orchestrates token discovery, security screening, and persistence."""

    def __init__(
        self,
        *,
        chains: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.chains = list(chains) if chains else None
        default_limit = int(os.getenv("DISCOVERY_TRENDING_LIMIT", "25"))
        self.limit = limit or default_limit

    def run(self) -> List[DiscoveredToken]:
        tokens = fetch_trending_tokens(limit=self.limit, chains=self.chains)
        logger.info("[discovery] fetched %d trending tokens", len(tokens))
        processed: List[DiscoveredToken] = []
        for trending in tokens:
            try:
                record = self._process_token(trending)
                processed.append(record)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("[discovery] failed to process %s: %s", trending.symbol, exc)
        return processed

    @transaction.atomic
    def _process_token(self, trending: TrendingToken) -> DiscoveredToken:
        symbol = trending.symbol.upper()
        chain = trending.chain.lower() if trending.chain else None
        token, _ = DiscoveredToken.objects.get_or_create(
            symbol=symbol,
            defaults={"chain": chain or "unknown"},
        )

        DiscoveryEvent.objects.create(
            symbol=symbol,
            chain=chain,
            source=trending.dex_id,
            bull_score=trending.price_change_1h,
            bear_score=trending.price_change_24h,
            liquidity_usd=trending.liquidity_usd,
            volume_24h=trending.volume_24h_usd,
            price_change_24h=trending.price_change_24h,
            metadata={
                "pair_address": trending.pair_address,
                "metadata": trending.metadata,
            },
        )

        reports: List[str] = []
        goplus_report = goplus_token_security(trending.pair_address, chain or "unknown")
        if goplus_report:
            HoneypotCheck.objects.create(
                symbol=symbol,
                chain=chain,
                verdict=goplus_report.verdict,
                confidence=goplus_report.confidence,
                details=goplus_report.details,
            )
            reports.append(goplus_report.verdict)

        heuristic_report = heuristic_screen(
            tax_buy=(goplus_report.details.get("buy_tax", 0) if goplus_report else 0.0),
            tax_sell=(goplus_report.details.get("sell_tax", 0) if goplus_report else 0.0),
            liquidity_usd=trending.liquidity_usd,
            price_change_24h=trending.price_change_24h,
        )
        HoneypotCheck.objects.create(
            symbol=symbol,
            chain=chain,
            verdict=heuristic_report.verdict,
            confidence=heuristic_report.confidence,
            details=heuristic_report.details,
        )
        reports.append(heuristic_report.verdict)

        verdict = "honeypot" if "honeypot" in reports else "validated"
        status = verdict if verdict == "honeypot" else "validated"
        token_metadata = dict(token.metadata or {})
        token_metadata["trending"] = {
            "dex_id": trending.dex_id,
            "pair_address": trending.pair_address,
            "price_usd": trending.price_usd,
            "volume_24h_usd": trending.volume_24h_usd,
            "liquidity_usd": trending.liquidity_usd,
            "price_change_24h": trending.price_change_24h,
        }
        if goplus_report:
            token_metadata.setdefault("security_reports", {})["goplus"] = goplus_report.details
        token_metadata.setdefault("security_reports", {})["heuristic"] = heuristic_report.details

        probe = None
        if status == "validated":
            probe = self._run_swap_probe(token, trending)
            if probe:
                token_metadata["last_probe"] = {
                    "success": probe.success,
                    "failure_reason": probe.failure_reason,
                    "metadata": probe.metadata,
                    "created_at": probe.created_at.isoformat(),
                }

        token.status = status
        token.metadata = token_metadata
        token.last_updated = datetime.now(timezone.utc)
        token.save(update_fields=["status", "last_updated", "metadata"])

        return token

    def _run_swap_probe(self, token: DiscoveredToken, trending: TrendingToken) -> Optional[SwapProbe]:
        chain = token.chain
        symbol = token.symbol
        try:
            min_liquidity = float(os.getenv("DISCOVERY_MIN_LIQUIDITY", "25000"))
        except Exception:
            min_liquidity = 25000.0
        try:
            min_volume = float(os.getenv("DISCOVERY_MIN_VOLUME", "50000"))
        except Exception:
            min_volume = 50000.0
        probe_defaults = {
            "success": False,
            "failure_reason": "probe_pending",
            "metadata": {"note": "initialising"},
        }
        probe, _ = SwapProbe.objects.get_or_create(
            symbol=symbol,
            chain=chain,
            defaults=probe_defaults,
        )
        if probe.success and probe.metadata:
            return probe

        metadata = dict(probe.metadata or {})
        metadata.update(
            {
                "pair_address": trending.pair_address,
                "dex_id": trending.dex_id,
                "price_usd": trending.price_usd,
                "volume_24h_usd": trending.volume_24h_usd,
                "liquidity_usd": trending.liquidity_usd,
            }
        )
        liquidity = trending.liquidity_usd or 0.0
        volume = trending.volume_24h_usd or 0.0

        simulated_success = False
        failure_reason = None
        if liquidity >= min_liquidity and volume >= min_volume:
            simulated_success = True
            metadata["mode"] = "virtual"
            metadata["slippage_buffer"] = float(os.getenv("DISCOVERY_PROBE_SLIPPAGE", "0.12"))
        else:
            failure_flags = []
            if liquidity < min_liquidity:
                failure_flags.append("liquidity")
                metadata["liquidity_gap"] = float(min_liquidity - liquidity)
            if volume < min_volume:
                failure_flags.append("volume")
                metadata["volume_gap"] = float(min_volume - volume)
            failure_reason = "insufficient_" + "+".join(failure_flags) if failure_flags else "insufficient_data"

        probe.success = simulated_success
        probe.failure_reason = failure_reason
        probe.metadata = metadata
        probe.save(update_fields=["success", "failure_reason", "metadata"])
        return probe
