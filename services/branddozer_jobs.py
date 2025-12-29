from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Iterable, Optional

from django.db import connection, transaction
from django.utils import timezone

from branddozer.models import BackgroundJob, BrandProject, DeliveryProject, DeliveryRun


JOB_TERMINAL_STATUSES = {"completed", "error", "canceled"}


def enqueue_job(
    *,
    kind: str,
    project: Optional[BrandProject] = None,
    run: Optional[DeliveryRun] = None,
    user: Optional[Any] = None,
    payload: Optional[Dict[str, Any]] = None,
    message: str = "Queued",
    detail: str = "",
) -> BackgroundJob:
    return BackgroundJob.objects.create(
        kind=kind,
        project=project,
        run=run,
        user=user,
        payload=payload or {},
        status="queued",
        message=message,
        detail=detail,
    )


def get_job(job_id: str, *, user: Optional[Any] = None) -> Optional[BackgroundJob]:
    qs = BackgroundJob.objects.filter(id=job_id)
    if user is not None:
        qs = qs.filter(user=user)
    return qs.first()


def job_payload(job: BackgroundJob) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": str(job.id),
        "kind": job.kind,
        "status": job.status,
        "message": job.message,
        "detail": job.detail,
        "error": job.error,
        "result": job.result,
        "project_id": str(job.project_id) if job.project_id else None,
        "run_id": str(job.run_id) if job.run_id else None,
        "created_at": int(job.created_at.timestamp()) if job.created_at else None,
        "updated_at": int(job.updated_at.timestamp()) if job.updated_at else None,
        "completed_at": int(job.completed_at.timestamp()) if job.completed_at else None,
    }
    if isinstance(job.result, dict) and job.result.get("project"):
        payload["project"] = job.result["project"]
    return payload


def update_job(job_id: str, **updates: Any) -> None:
    allowed = {
        "status",
        "message",
        "detail",
        "error",
        "result",
        "attempts",
        "locked_by",
        "locked_at",
        "started_at",
        "completed_at",
    }
    payload = {key: value for key, value in updates.items() if key in allowed}
    if not payload:
        return
    if payload.get("status") in JOB_TERMINAL_STATUSES and not payload.get("completed_at"):
        payload["completed_at"] = timezone.now()
    payload["updated_at"] = timezone.now()
    BackgroundJob.objects.filter(id=job_id).update(**payload)


def claim_next_job(worker_id: str, *, kinds: Optional[Iterable[str]] = None) -> Optional[BackgroundJob]:
    now = timezone.now()
    with transaction.atomic():
        if connection.vendor == "sqlite":
            qs = BackgroundJob.objects.filter(status="queued")
        else:
            qs = BackgroundJob.objects.select_for_update(skip_locked=True).filter(status="queued")
        if kinds:
            qs = qs.filter(kind__in=list(kinds))
        job = qs.order_by("created_at").first()
        if not job:
            return None
        job.status = "running"
        job.locked_by = worker_id
        job.locked_at = now
        job.started_at = job.started_at or now
        job.attempts += 1
        job.message = job.message or "Running"
        job.save(update_fields=["status", "locked_by", "locked_at", "started_at", "attempts", "message", "updated_at"])
        return job


def complete_job(job: BackgroundJob, *, message: Optional[str] = None, detail: Optional[str] = None, result: Any = None) -> None:
    if message is not None:
        job.message = message
    if detail is not None:
        job.detail = detail
    if result is not None:
        job.result = result
    job.status = "completed"
    job.completed_at = timezone.now()
    job.save(update_fields=["status", "message", "detail", "result", "completed_at", "updated_at"])


def fail_job(job: BackgroundJob, *, error: str, message: Optional[str] = None, detail: Optional[str] = None) -> None:
    job.status = "error"
    job.error = error
    if message is not None:
        job.message = message
    if detail is not None:
        job.detail = detail
    job.completed_at = timezone.now()
    job.save(update_fields=["status", "error", "message", "detail", "completed_at", "updated_at"])


def cancel_stale_jobs(*, stale_seconds: int, kinds: Optional[Iterable[str]] = None) -> int:
    if stale_seconds <= 0:
        return 0
    cutoff = timezone.now() - timedelta(seconds=stale_seconds)
    qs = BackgroundJob.objects.filter(status__in=["queued", "running"], updated_at__lt=cutoff)
    if kinds:
        qs = qs.filter(kind__in=list(kinds))
    count = 0
    for job in qs:
        detail = f"Last update: {job.updated_at.isoformat() if job.updated_at else 'unknown'}"
        update_job(str(job.id), status="canceled", message="Canceled stale job", detail=detail)
        if job.run and job.run.status in {"queued", "running"}:
            run = job.run
            context = dict(run.context or {})
            context["stop_requested"] = True
            context["status_note"] = "Canceled stale job"
            context["status_detail"] = detail
            context["status_ts"] = timezone.now().isoformat()
            run.context = context
            run.status = "blocked"
            run.phase = "stopped"
            run.error = "Canceled stale job"
            run.completed_at = timezone.now()
            run.save(update_fields=["context", "status", "phase", "error", "completed_at"])
            delivery_project = DeliveryProject.objects.filter(project=run.project).first()
            if delivery_project:
                delivery_project.status = "blocked"
                delivery_project.active_run = run
                delivery_project.save(update_fields=["status", "active_run", "updated_at"])
        count += 1
    return count
