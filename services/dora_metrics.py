from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional

from django.utils import timezone

from branddozer.models import DeliveryRun


DEFAULT_SUCCESS_STATUSES = {"complete", "awaiting_acceptance"}
DEFAULT_FAILURE_STATUSES = {"blocked", "error"}


@dataclass
class DoraSnapshot:
    window_days: int
    generated_at: str
    deployment_frequency: float
    lead_time_hours: float
    change_failure_rate: float
    mttr_hours: float
    total_releases: int
    failed_releases: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _window_start(window_days: int) -> timezone.datetime:
    return timezone.now() - timedelta(days=max(1, window_days))


def _avg_hours(total_seconds: float, count: int) -> float:
    if not count or total_seconds <= 0:
        return 0.0
    return round((total_seconds / count) / 3600, 3)


def generate_snapshot(
    window_days: int = 7,
    success_statuses: Optional[set[str]] = None,
    failure_statuses: Optional[set[str]] = None,
) -> DoraSnapshot:
    success_statuses = success_statuses or DEFAULT_SUCCESS_STATUSES
    failure_statuses = failure_statuses or DEFAULT_FAILURE_STATUSES
    start_ts = _window_start(window_days)
    runs = DeliveryRun.objects.filter(completed_at__isnull=False, completed_at__gte=start_ts)

    total_releases = runs.filter(status__in=success_statuses).count()
    failed_releases = runs.filter(status__in=failure_statuses).count()
    deployments_per_day = total_releases / float(window_days)

    lead_seconds = 0.0
    lead_count = 0
    for run in runs.filter(status__in=success_statuses):
        if run.started_at and run.completed_at:
            lead_seconds += (run.completed_at - run.started_at).total_seconds()
            lead_count += 1

    mttr_seconds = 0.0
    mttr_count = 0
    for run in runs.filter(status__in=failure_statuses):
        if run.started_at and run.completed_at:
            mttr_seconds += (run.completed_at - run.started_at).total_seconds()
            mttr_count += 1

    release_total = total_releases + failed_releases
    change_failure_rate = (failed_releases / release_total) if release_total else 0.0

    snapshot = DoraSnapshot(
        window_days=window_days,
        generated_at=timezone.now().isoformat(),
        deployment_frequency=round(deployments_per_day, 3),
        lead_time_hours=_avg_hours(lead_seconds, lead_count),
        change_failure_rate=round(change_failure_rate, 3),
        mttr_hours=_avg_hours(mttr_seconds, mttr_count),
        total_releases=total_releases,
        failed_releases=failed_releases,
    )
    return snapshot


def write_snapshot(snapshot: DoraSnapshot, output_dir: str | Path = "runtime/branddozer/reports") -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"dora_{snapshot.generated_at.replace(':', '').replace('-', '')}.json"
    file_path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
    return file_path
