from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from db import get_db
from services.cron_profile import load_profile, save_profile
from services.guardian_lock import GuardianLease
from services.logging_utils import log_message
from services.secure_settings import build_process_env, default_env_user

STATE_PATH = Path("runtime/cron/state.json")
LOG_SOURCE = "internal-cron"


def _ensure_state_dir() -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {"tasks": {}}
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"tasks": {}}
    if not isinstance(payload, dict):
        return {"tasks": {}}
    payload.setdefault("tasks", {})
    return payload


def _save_state(state: Dict[str, Any]) -> None:
    _ensure_state_dir()
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _now_ts() -> float:
    return time.time()


def _apply_env(env: Dict[str, str]) -> None:
    for key, value in env.items():
        if value is None:
            continue
        os.environ[key] = str(value)


@dataclass
class TaskOutcome:
    status: str
    message: str


class InternalCronSupervisor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lease: Optional[GuardianLease] = None
        self._poll_interval = float(os.getenv("CRON_POLL_INTERVAL", "30"))
        self._state = _load_state()
        self._status: Dict[str, Any] = {
            "running": False,
            "last_cycle": None,
            "errors": 0,
        }
        self._force_run: set[str] = set()

    # ------------------------------------------------------------------ public API
    def ensure_running(self) -> None:
        if os.getenv("CRON_AUTO_DISABLED") == "1" or os.getenv("INTERNAL_CRON_DISABLED") == "1":
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run_loop, name="internal-cron", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=10.0)
            self._thread = None
            if self._lease:
                self._lease.release()
                self._lease = None
        with self._lock:
            self._status["running"] = False

    def status(self) -> Dict[str, Any]:
        profile = load_profile()
        with self._lock:
            tasks = dict(self._state.get("tasks") or {})
            running = bool(self._thread and self._thread.is_alive())
            status = dict(self._status)
        return {
            "running": running,
            "profile": profile,
            "tasks": tasks,
            "status": status,
            "state_path": str(STATE_PATH),
        }

    def update_profile(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        current = load_profile(force=True)
        merged = dict(current)
        merged.update(updates or {})
        return save_profile(merged)

    def run_once(self, task_id: Optional[str] = None) -> None:
        if task_id:
            self._force_run.add(task_id)
        else:
            for task in load_profile().get("tasks", []):
                tid = str(task.get("id") or "").strip()
                if tid:
                    self._force_run.add(tid)

    # ------------------------------------------------------------------ loop
    def _run_loop(self) -> None:
        lease = GuardianLease("internal-cron", poll_interval=0.5)
        if not lease.acquire(cancel_event=self._stop):
            log_message(LOG_SOURCE, "cron lease busy; skipping start", severity="warning")
            return
        self._lease = lease
        log_message(LOG_SOURCE, "internal cron active", severity="info")
        try:
            while not self._stop.is_set():
                profile = load_profile()
                if not profile.get("enabled", True):
                    with self._lock:
                        self._status["running"] = False
                    if self._stop.wait(self._poll_interval):
                        break
                    continue
                with self._lock:
                    self._status["running"] = True
                now = _now_ts()
                for task in profile.get("tasks", []):
                    if self._stop.is_set():
                        break
                    task_id = str(task.get("id") or "").strip()
                    if not task_id or not task.get("enabled", True):
                        continue
                    state = self._get_task_state(task_id)
                    if state.get("running"):
                        continue
                    if task_id in self._force_run or self._task_due(task, state, now):
                        self._force_run.discard(task_id)
                        self._execute_task(task, profile)
                with self._lock:
                    self._status["last_cycle"] = now
                if self._stop.wait(self._poll_interval):
                    break
        finally:
            with self._lock:
                self._status["running"] = False
            if self._lease:
                self._lease.release()
                self._lease = None

    # ------------------------------------------------------------------ task helpers
    def _get_task_state(self, task_id: str) -> Dict[str, Any]:
        tasks = self._state.setdefault("tasks", {})
        if task_id not in tasks:
            tasks[task_id] = {
                "last_run": None,
                "last_status": "never",
                "last_error": None,
                "last_message": None,
                "last_duration_s": None,
                "running": False,
                "next_run": None,
            }
        return tasks[task_id]

    def _task_due(self, task: Dict[str, Any], state: Dict[str, Any], now: float) -> bool:
        next_run = state.get("next_run")
        if next_run is None:
            interval = int(task.get("interval_minutes") or 60) * 60
            jitter = int(task.get("jitter_seconds") or 0)
            state["next_run"] = now + (random.randint(0, jitter) if jitter > 0 else 0)
            _save_state(self._state)
            return True
        return now >= float(next_run)

    def _execute_task(self, task: Dict[str, Any], profile: Dict[str, Any]) -> None:
        task_id = str(task.get("id") or "unknown")
        state = self._get_task_state(task_id)
        state["running"] = True
        state["last_status"] = "running"
        state["last_message"] = "running"
        _save_state(self._state)
        start = _now_ts()
        outcome = TaskOutcome(status="success", message="completed")
        try:
            outcome = self._run_task(task, profile)
        except Exception as exc:  # pragma: no cover - defensive
            outcome = TaskOutcome(status="error", message=str(exc))
            log_message(LOG_SOURCE, f"task {task_id} failed: {exc}", severity="error")
            with self._lock:
                self._status["errors"] = int(self._status.get("errors", 0)) + 1
        finally:
            end = _now_ts()
            state["running"] = False
            state["last_run"] = end
            state["last_status"] = outcome.status
            state["last_message"] = outcome.message
            state["last_duration_s"] = round(end - start, 2)
            interval = int(task.get("interval_minutes") or 60) * 60
            jitter = int(task.get("jitter_seconds") or 0)
            if outcome.status.startswith("skipped"):
                backoff = min(3600, max(300, interval // 4))
                state["next_run"] = end + backoff
            else:
                state["next_run"] = end + interval + (random.randint(0, jitter) if jitter > 0 else 0)
            _save_state(self._state)

    # ------------------------------------------------------------------ task runner
    def _run_task(self, task: Dict[str, Any], profile: Dict[str, Any]) -> TaskOutcome:
        task_id = str(task.get("id") or "task")
        user = default_env_user()
        env = build_process_env(user)
        _apply_env(env)
        has_mnemonic = bool(env.get("MNEMONIC") or "")
        if task.get("requires_mnemonic") and not has_mnemonic:
            msg = "mnemonic missing; skipping"
            log_message(LOG_SOURCE, f"{task_id}: {msg}", severity="warning")
            return TaskOutcome(status="skipped_mnemonic", message=msg)

        production_active = False
        try:
            production_active = bool(get_db().get_control_flag("production_manager_active"))
        except Exception:
            production_active = False

        steps = [str(step) for step in (task.get("steps") or []) if step]
        context: Dict[str, Any] = {"discovery": None, "candidates": []}
        for step in steps:
            if self._stop.is_set():
                return TaskOutcome(status="skipped", message="stop requested")
            if step == "discovery":
                context["discovery"] = self._run_discovery(profile)
            elif step == "watchlists":
                context["candidates"] = self._update_watchlists(profile)
            elif step == "recommendations":
                self._run_recommendations(profile, context)
            elif step == "downloads":
                if production_active:
                    log_message(LOG_SOURCE, f"{task_id}: downloads skipped (production active)", severity="info")
                else:
                    self._run_downloads(profile)
            elif step == "news":
                self._run_news(profile, context)
            elif step == "training":
                if production_active:
                    log_message(LOG_SOURCE, f"{task_id}: training skipped (production active)", severity="info")
                else:
                    self._run_training(profile)
            elif step == "production":
                self._run_production(profile, production_active)
            elif step == "guardian":
                self._run_guardian()
            elif step == "branddozer":
                self._run_branddozer()
            else:
                log_message(LOG_SOURCE, f"{task_id}: unknown step {step}", severity="warning")
        return TaskOutcome(status="success", message="completed")

    # ------------------------------------------------------------------ step implementations
    def _run_discovery(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        discovery_cfg = profile.get("discovery") or {}
        chains = discovery_cfg.get("chains") or None
        limit = int(discovery_cfg.get("limit") or 40)
        try:
            from services.discovery.coordinator import DiscoveryCoordinator
        except Exception as exc:  # pragma: no cover
            log_message(LOG_SOURCE, f"discovery unavailable: {exc}", severity="warning")
            return {"count": 0, "error": str(exc)}
        coordinator = DiscoveryCoordinator(chains=chains, limit=limit)
        results = coordinator.run()
        log_message(LOG_SOURCE, f"discovery fetched {len(results)} tokens", severity="info")
        return {"count": len(results)}

    def _update_watchlists(self, profile: Dict[str, Any]) -> List[str]:
        discovery_cfg = profile.get("discovery") or {}
        try:
            from datetime import timedelta
            from django.utils import timezone
            from discovery.models import DiscoveryEvent, DiscoveredToken, HoneypotCheck
            from services.watchlists import mutate_watchlist
        except Exception as exc:  # pragma: no cover
            log_message(LOG_SOURCE, f"watchlist update unavailable: {exc}", severity="warning")
            return []

        max_age = float(discovery_cfg.get("max_age_hours") or 36)
        min_liquidity = float(discovery_cfg.get("min_liquidity_usd") or 0)
        min_volume = float(discovery_cfg.get("min_volume_usd") or 0)
        min_change_1h = float(discovery_cfg.get("min_change_1h") or -1000)
        min_change_24h = float(discovery_cfg.get("min_change_24h") or -1000)
        max_tokens = int(discovery_cfg.get("max_tokens") or 25)
        target = str(discovery_cfg.get("watchlist_target") or "stream").lower()
        also_ghost = bool(discovery_cfg.get("also_add_to_ghost", True))

        cutoff = timezone.now() - timedelta(hours=max_age)
        events = (
            DiscoveryEvent.objects.filter(created_at__gte=cutoff)
            .filter(liquidity_usd__gte=min_liquidity)
            .filter(volume_24h__gte=min_volume)
            .filter(bull_score__gte=min_change_1h)
            .filter(price_change_24h__gte=min_change_24h)
            .order_by("-volume_24h", "-liquidity_usd", "-created_at")
        )
        symbols = []
        seen = set()
        for event in events:
            symbol = (event.symbol or "").upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)
            if len(symbols) >= max_tokens:
                break

        if not symbols:
            return []

        # Filter out known honeypots/rejections.
        statuses = {
            tok.symbol.upper(): tok.status
            for tok in DiscoveredToken.objects.filter(symbol__in=symbols)
        }
        latest_checks: Dict[str, HoneypotCheck] = {}
        for check in HoneypotCheck.objects.filter(symbol__in=symbols).order_by("symbol", "-created_at"):
            sym = check.symbol.upper()
            if sym not in latest_checks:
                latest_checks[sym] = check
        filtered = []
        for symbol in symbols:
            status = statuses.get(symbol)
            if status in {"honeypot", "rejected"}:
                continue
            check = latest_checks.get(symbol)
            if check and check.verdict == "honeypot":
                continue
            filtered.append(symbol)

        if not filtered:
            return []

        mutate_watchlist(target, add=filtered)
        if also_ghost:
            mutate_watchlist("ghost", add=filtered)
        log_message(
            LOG_SOURCE,
            f"watchlists updated ({target})",
            details={"count": len(filtered), "symbols": filtered[:8]},
        )
        return filtered

    def _run_downloads(self, profile: Dict[str, Any]) -> None:
        downloads_cfg = profile.get("downloads") or {}
        chains = downloads_cfg.get("chains") or []
        max_pairs = int(downloads_cfg.get("max_pairs") or 0)
        chains = [str(ch).lower().strip() for ch in chains if str(ch).strip()]
        if chains:
            chains = self._prioritize_chains(profile, chains)
            os.environ["DOWNLOAD_WORKER_CHAINS"] = ",".join(chains)
        if max_pairs:
            os.environ["DOWNLOAD_MAX_PAIRS"] = str(max_pairs)
        try:
            from services.background_workers import TokenDownloadSupervisor
        except Exception as exc:  # pragma: no cover
            log_message(LOG_SOURCE, f"download supervisor unavailable: {exc}", severity="warning")
            return
        supervisor = TokenDownloadSupervisor(db=get_db())
        supervisor.run_cycle()
        try:
            get_db().record_feedback_event(
                source="cron",
                severity="info",
                label="downloads",
                details={"chains": chains, "max_pairs": max_pairs},
            )
        except Exception:
            pass
        log_message(LOG_SOURCE, "download supervisor cycle completed", severity="info")

    def _run_news(self, profile: Dict[str, Any], context: Dict[str, Any]) -> None:
        news_cfg = profile.get("news") or {}
        lookback_hours = int(news_cfg.get("lookback_hours") or 72)
        max_pages = news_cfg.get("max_pages")
        max_tokens = int(news_cfg.get("max_tokens") or 8)
        tokens = context.get("candidates") or []
        if tokens:
            tokens = list(tokens)[:max_tokens]
        if not tokens:
            fallback = news_cfg.get("default_tokens") or ["BTC", "ETH", "USDC"]
            tokens = list(fallback)[:max_tokens]
        try:
            from datetime import datetime, timedelta, timezone
            from services.news_lab import collect_news_for_terms
        except Exception as exc:  # pragma: no cover
            log_message(LOG_SOURCE, f"news collector unavailable: {exc}", severity="warning")
            return
        if not tokens:
            return
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=lookback_hours)
        result = collect_news_for_terms(tokens=tokens, start=start, end=end, max_pages=max_pages)
        try:
            get_db().record_feedback_event(
                source="cron",
                severity="info",
                label="news",
                details={"tokens": tokens, "articles": len(result.get("items", []))},
            )
        except Exception:
            pass
        log_message(
            LOG_SOURCE,
            "news batch complete",
            details={"tokens": tokens, "articles": len(result.get("items", []))},
        )

    def _run_recommendations(self, profile: Dict[str, Any], context: Dict[str, Any]) -> None:
        rec_cfg = profile.get("recommendations") or {}
        if not rec_cfg.get("enabled", True):
            return
        try:
            from datetime import timedelta
            from django.utils import timezone
            from discovery.models import DiscoveryEvent, DiscoveredToken, HoneypotCheck, SwapProbe
        except Exception as exc:  # pragma: no cover
            log_message(LOG_SOURCE, f"recommendations unavailable: {exc}", severity="warning")
            return

        lookback_days = float(rec_cfg.get("lookback_days") or 30)
        max_tokens = int(rec_cfg.get("max_tokens") or 8)
        min_liquidity = float(rec_cfg.get("min_liquidity_usd") or 0)
        min_volume = float(rec_cfg.get("min_volume_usd") or 0)
        min_age_hours = float(rec_cfg.get("min_age_hours") or 0)
        max_age_hours = float(rec_cfg.get("max_age_hours") or 0)
        min_score = float(rec_cfg.get("min_score") or 0.0)

        low_fee = rec_cfg.get("low_fee_chains") or os.getenv(
            "SAVINGS_LOW_FEE_CHAINS", "base,arbitrum,optimism,polygon"
        )
        if isinstance(low_fee, str):
            low_fee_chains = [c.strip().lower() for c in low_fee.split(",") if c.strip()]
        else:
            low_fee_chains = [str(c).strip().lower() for c in low_fee if str(c).strip()]

        preferred_chains: set[str] = set()
        if bool(rec_cfg.get("wallet_bias", True)):
            wallet = (
                os.getenv("PRIMARY_WALLET")
                or os.getenv("TRADING_WALLET")
                or os.getenv("WALLET_NAME")
                or ""
            ).strip().lower()
            if wallet:
                try:
                    db = get_db()
                    rows = db.fetch_balances_flat(wallet=wallet, include_zero=False)
                    chain_totals: Dict[str, float] = {}
                    for row in rows:
                        chain = str(row["chain"] or "").lower()
                        usd_val = float(row["usd_amount"] or 0.0)
                        if chain:
                            chain_totals[chain] = chain_totals.get(chain, 0.0) + usd_val
                    for chain, usd_val in sorted(chain_totals.items(), key=lambda kv: kv[1], reverse=True)[:2]:
                        if usd_val > 0:
                            preferred_chains.add(chain)
                except Exception:
                    preferred_chains = set()

        cutoff = timezone.now() - timedelta(days=lookback_days)
        events = DiscoveryEvent.objects.filter(created_at__gte=cutoff).order_by("-created_at")
        if low_fee_chains:
            events = events.filter(chain__in=low_fee_chains)

        latest_events: Dict[tuple[str, str], DiscoveryEvent] = {}
        for event in events:
            symbol = (event.symbol or "").upper()
            chain = (event.chain or "").lower()
            if not symbol or not chain:
                continue
            key = (symbol, chain)
            if key not in latest_events:
                latest_events[key] = event

        if not latest_events:
            return

        symbols = [sym for sym, _ in latest_events.keys()]
        tokens = {
            t.symbol.upper(): t
            for t in DiscoveredToken.objects.filter(symbol__in=symbols)
        }
        checks = HoneypotCheck.objects.filter(symbol__in=symbols).order_by("symbol", "-created_at")
        latest_checks: Dict[str, HoneypotCheck] = {}
        for check in checks:
            sym = check.symbol.upper()
            if sym not in latest_checks:
                latest_checks[sym] = check
        probes = SwapProbe.objects.filter(symbol__in=symbols).order_by("symbol", "-created_at")
        latest_probes: Dict[str, SwapProbe] = {}
        for probe in probes:
            sym = probe.symbol.upper()
            if sym not in latest_probes:
                latest_probes[sym] = probe

        now = timezone.now()
        scored: List[Dict[str, Any]] = []
        for (symbol, chain), event in latest_events.items():
            token = tokens.get(symbol)
            if not token:
                continue
            status = (token.status or "").lower()
            if status in {"honeypot", "rejected"}:
                continue
            check = latest_checks.get(symbol)
            if check and (check.verdict or "").lower() == "honeypot":
                continue
            volume = float(event.volume_24h or 0.0)
            liquidity = float(event.liquidity_usd or 0.0)
            if volume < min_volume or liquidity < min_liquidity:
                continue
            age_hours = (now - token.first_seen).total_seconds() / 3600.0 if token.first_seen else 0.0
            if min_age_hours and age_hours < min_age_hours:
                continue
            if max_age_hours and age_hours > max_age_hours:
                continue
            chain_weight = 1.0
            if chain in low_fee_chains:
                chain_weight += 0.15
            if chain in preferred_chains:
                chain_weight += 0.15
            volatility = abs(float(event.price_change_24h or 0.0))
            score = (
                math.log1p(volume) * 1.1
                + math.log1p(liquidity) * 0.9
                + math.log1p(max(age_hours, 1.0)) * 0.4
                - volatility * 0.02
            )
            score *= chain_weight
            if score < min_score:
                continue
            probe = latest_probes.get(symbol)
            scored.append(
                {
                    "symbol": symbol,
                    "chain": chain,
                    "score": round(score, 4),
                    "volume_24h": volume,
                    "liquidity_usd": liquidity,
                    "age_hours": round(age_hours, 2),
                    "price_change_24h": event.price_change_24h,
                    "probe_ok": bool(probe.success) if probe else None,
                    "probe_reason": probe.failure_reason if probe else None,
                }
            )

        if not scored:
            return
        scored.sort(key=lambda row: row["score"], reverse=True)
        top = scored[:max_tokens]
        signature = "|".join(f"{row['symbol']}:{row['chain']}" for row in top)
        db = get_db()
        existing = db.fetch_advisories(limit=50, include_resolved=False)
        for advisory in existing:
            meta = advisory.get("meta") or {}
            if meta.get("signature") == signature:
                return

        chain_list = ", ".join(sorted({row["chain"] for row in top}))
        message = f"Top candidates on low-fee chains ({chain_list})."
        recommendation = (
            "Shortlist generated from volume/liquidity/age filters. "
            "Review in Data Lab before allocating capital."
        )
        db.record_advisory(
            topic="token_candidates",
            message=message,
            severity="info",
            scope="discovery",
            recommendation=recommendation,
            meta={
                "signature": signature,
                "candidates": top,
                "low_fee_chains": low_fee_chains,
                "preferred_chains": sorted(preferred_chains),
                "filters": {
                    "min_volume_usd": min_volume,
                    "min_liquidity_usd": min_liquidity,
                    "min_age_hours": min_age_hours,
                    "max_age_hours": max_age_hours,
                    "lookback_days": lookback_days,
                },
            },
        )
        log_message(LOG_SOURCE, f"recommendations generated ({len(top)} tokens)", severity="info")

    def _run_training(self, profile: Dict[str, Any]) -> None:
        training_cfg = profile.get("training") or {}
        if not training_cfg.get("enabled", True):
            return
        try:
            from trading.pipeline import TrainingPipeline
        except Exception as exc:  # pragma: no cover
            log_message(LOG_SOURCE, f"training unavailable: {exc}", severity="warning")
            return
        pipeline = TrainingPipeline(db=get_db())
        pipeline.train_candidate()
        try:
            get_db().record_feedback_event(
                source="cron",
                severity="info",
                label="training",
                details={"status": "candidate_cycle_completed"},
            )
        except Exception:
            pass
        log_message(LOG_SOURCE, "training candidate cycle completed", severity="info")

    def _run_production(self, profile: Dict[str, Any], production_active: bool) -> None:
        production_cfg = profile.get("production") or {}
        if not production_cfg.get("enabled", True):
            return
        if production_active:
            log_message(LOG_SOURCE, "production manager already active", severity="info")
            return
        if not self._production_ready(profile, production_cfg):
            log_message(LOG_SOURCE, "production gate not yet satisfied; holding", severity="info")
            try:
                get_db().record_feedback_event(
                    source="cron",
                    severity="warning",
                    label="production_gate",
                    details={"status": "blocked"},
                )
            except Exception:
                pass
            return
        try:
            from services.production_supervisor import production_supervisor
        except Exception as exc:  # pragma: no cover
            log_message(LOG_SOURCE, f"production supervisor unavailable: {exc}", severity="warning")
            return
        production_supervisor.ensure_running()
        try:
            get_db().record_feedback_event(
                source="cron",
                severity="info",
                label="production",
                details={"status": "started"},
            )
        except Exception:
            pass
        log_message(LOG_SOURCE, "production supervisor ensured", severity="info")

    def _run_guardian(self) -> None:
        try:
            from services.guardian_supervisor import guardian_supervisor
        except Exception as exc:  # pragma: no cover
            log_message(LOG_SOURCE, f"guardian unavailable: {exc}", severity="warning")
            return
        guardian_supervisor.ensure_running()
        guardian_supervisor.run_once()

    def _run_branddozer(self) -> None:
        try:
            from services.branddozer_runner import branddozer_manager
            from services.branddozer_state import list_projects
        except Exception as exc:  # pragma: no cover
            log_message(LOG_SOURCE, f"branddozer unavailable: {exc}", severity="warning")
            return
        projects = list_projects()
        started = 0
        for project in projects:
            if not project.get("enabled"):
                continue
            branddozer_manager.start(project.get("id"))
            started += 1
        if started:
            log_message(LOG_SOURCE, f"branddozer ensured {started} project(s)", severity="info")

    def _production_ready(self, profile: Dict[str, Any], production_cfg: Dict[str, Any]) -> bool:
        downloads_cfg = profile.get("downloads") or {}
        chains = production_cfg.get("chains") or downloads_cfg.get("chains") or []
        chains = [str(ch).lower().strip() for ch in chains if str(ch).strip()]
        if not chains:
            return False
        require_index = bool(production_cfg.get("require_pair_index", True))
        min_files = int(production_cfg.get("min_files_per_chain") or 1)
        min_chains_ready = int(production_cfg.get("min_chains_ready") or 1)
        ready_chains = 0
        missing_indexes: list[str] = []
        for chain in chains:
            if require_index:
                index_path = Path("data") / f"pair_index_{chain}.json"
                if not index_path.exists():
                    missing_indexes.append(chain)
                    continue
            files_ready = self._ohlcv_file_count(chain) >= min_files
            db_ready = self._ohlcv_db_count(chain) >= min_files
            if files_ready or db_ready:
                ready_chains += 1
        if missing_indexes:
            log_message(
                LOG_SOURCE,
                "production gate waiting on pair indexes",
                severity="info",
                details={"missing": missing_indexes},
            )
        return ready_chains >= min_chains_ready

    def _prioritize_chains(self, profile: Dict[str, Any], chains: List[str]) -> List[str]:
        rec_cfg = profile.get("recommendations") or {}
        low_fee = rec_cfg.get("low_fee_chains") or os.getenv(
            "SAVINGS_LOW_FEE_CHAINS", "base,arbitrum,optimism,polygon"
        )
        if isinstance(low_fee, str):
            low_fee_chains = [c.strip().lower() for c in low_fee.split(",") if c.strip()]
        else:
            low_fee_chains = [str(c).strip().lower() for c in low_fee if str(c).strip()]

        preferred: List[str] = []
        try:
            wallet = (
                os.getenv("PRIMARY_WALLET")
                or os.getenv("TRADING_WALLET")
                or os.getenv("WALLET_NAME")
                or ""
            ).strip().lower()
            if wallet:
                rows = get_db().fetch_balances_flat(wallet=wallet, include_zero=False)
                chain_totals: Dict[str, float] = {}
                for row in rows:
                    chain = str(row.get("chain") or "").lower()
                    usd_val = float(row.get("usd_amount") or 0.0)
                    if chain:
                        chain_totals[chain] = chain_totals.get(chain, 0.0) + usd_val
                preferred = [c for c, _ in sorted(chain_totals.items(), key=lambda kv: kv[1], reverse=True)]
        except Exception:
            preferred = []

        ordered: List[str] = []
        for chain in preferred:
            if chain in chains and chain not in ordered:
                ordered.append(chain)
        for chain in low_fee_chains:
            if chain in chains and chain not in ordered:
                ordered.append(chain)
        for chain in chains:
            if chain not in ordered:
                ordered.append(chain)
        return ordered

    @staticmethod
    def _ohlcv_file_count(chain: str) -> int:
        root = Path("data") / "historical_ohlcv" / chain
        if not root.exists() or not root.is_dir():
            return 0
        count = 0
        for path in root.glob("*.json"):
            try:
                if path.stat().st_size > 0:
                    count += 1
            except Exception:
                continue
        return count

    @staticmethod
    def _ohlcv_db_count(chain: str) -> int:
        try:
            from datalab.models import OhlcvDataset
        except Exception:
            return 0
        try:
            return int(OhlcvDataset.objects.filter(chain=str(chain).lower()).count())
        except Exception:
            return 0


cron_supervisor = InternalCronSupervisor()
