from __future__ import annotations

import os
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from db import get_db, TradingDatabase
from services.adaptive_control import AdaptiveLimiter
from services.background_workers import TokenDownloadSupervisor
from services.logging_utils import log_message
from services.news_lab import collect_news_for_terms
from trading.constants import top_pairs


class IdleWorkManager:
    """
    Keeps opportunistic jobs (historical downloads, deep news scrapes, etc.)
    running whenever there is spare CPU/RAM capacity.
    """

    def __init__(
        self,
        *,
        db: Optional[TradingDatabase] = None,
        limiter: Optional[AdaptiveLimiter] = None,
    ) -> None:
        self.db = db or get_db()
        self.supervisor = TokenDownloadSupervisor(db=self.db)
        self.limiter = limiter or AdaptiveLimiter(cpu_soft=45.0, cpu_hard=80.0, mem_soft=0.65, mem_hard=0.9, cool_down=6.0)
        self._jobs: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._job_lock = threading.Lock()
        self._seed_jobs()

    def _seed_jobs(self) -> None:
        with self._job_lock:
            if not self._jobs.empty():
                return
            chains = [chain.strip() for chain in os.getenv("IDLE_DOWNLOAD_CHAINS", "base").split(",") if chain.strip()]
            for chain in chains:
                self._jobs.put({"type": "download_ohlcv", "chain": chain})
            token_batches = self._build_news_batches()
            for batch in token_batches:
                self._jobs.put(
                    {
                        "type": "news_terms",
                        "tokens": batch,
                        "lookback_hours": 72,
                        "label": "recent",
                        "offset_days": 0,
                    }
                )
                self._jobs.put(
                    {
                        "type": "news_terms",
                        "tokens": batch,
                        "lookback_hours": 24 * 7,
                        "label": "weekly",
                        "offset_days": 7,
                    }
                )

    def _build_news_batches(self) -> List[List[str]]:
        tokens: List[str] = []
        for pair in top_pairs(limit=48):
            for token in pair.replace("/", "-").split("-"):
                token = token.strip().upper()
                if token and token not in tokens:
                    tokens.append(token)
        chunk_size = max(2, int(os.getenv("IDLE_NEWS_BATCH", "4")))
        return [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    def run_next_job(self) -> bool:
        try:
            job = self._jobs.get_nowait()
        except queue.Empty:
            self._seed_jobs()
            return False
        try:
            self.limiter.before_task(f"idle:{job['type']}")
            if job["type"] == "download_ohlcv":
                return self._run_download_job(job)
            if job["type"] == "news_terms":
                return self._run_news_job(job)
            return False
        finally:
            self._jobs.task_done()

    def _run_download_job(self, job: Dict[str, Any]) -> bool:
        chain = job.get("chain") or "base"
        os.environ["CHAIN_NAME"] = chain
        try:
            self.supervisor.run_cycle()
            log_message("idle-work", f"download cycle completed for {chain}")
            return True
        except Exception as exc:
            log_message("idle-work", f"download cycle failed for {chain}: {exc}", severity="error")
            time.sleep(2.0)
            return False

    def _run_news_job(self, job: Dict[str, Any]) -> bool:
        tokens: Sequence[str] = job.get("tokens") or []
        if not tokens:
            return True
        lookback = int(job.get("lookback_hours", 72))
        offset = int(job.get("offset_days", 0))
        end = datetime.now(timezone.utc) - timedelta(days=offset)
        start = end - timedelta(hours=lookback)
        label = job.get("label") or "news"
        try:
            result = collect_news_for_terms(
                tokens=tokens,
                start=start,
                end=end,
                max_pages=job.get("max_pages"),
            )
            articles = len(result.get("items", []))
            log_message(
                "idle-work",
                f"news batch ({label})",
                details={
                    "tokens": tokens,
                    "articles": articles,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
            )
            return True
        except Exception as exc:
            log_message(
                "idle-work",
                f"news batch ({label}) failed",
                severity="warning",
                details={"tokens": tokens, "error": str(exc)},
            )
            return False
