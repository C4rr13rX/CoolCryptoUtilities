from __future__ import annotations

import os
import socket
import time
import uuid

from django.core.management import BaseCommand
from django.db import close_old_connections
from django.utils import timezone

from branddozer.models import DeliveryRun
from branddozer import views as branddozer_views
from services.branddozer_delivery import delivery_orchestrator
from services.branddozer_jobs import claim_next_job, complete_job, fail_job, update_job


class Command(BaseCommand):
    help = "Run the BrandDozer background jobs worker."

    def add_arguments(self, parser):
        parser.add_argument(
            "--once",
            action="store_true",
            help="Process one job then exit.",
        )
        parser.add_argument(
            "--idle-sleep",
            type=float,
            default=1.5,
            help="Seconds to sleep when the queue is empty.",
        )
        parser.add_argument(
            "--types",
            nargs="*",
            default=None,
            help="Optional list of job kinds to process (e.g. github_import ui_capture).",
        )

    def handle(self, *args, **options):
        worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        idle_sleep = float(options["idle_sleep"] or 1.5)
        kinds = options.get("types") or None
        run_once = bool(options.get("once"))

        self.stdout.write(self.style.HTTP_INFO(f"BrandDozer worker {worker_id} starting..."))

        while True:
            close_old_connections()
            job = claim_next_job(worker_id, kinds=kinds)
            if not job:
                if run_once:
                    break
                time.sleep(idle_sleep)
                continue

            try:
                self._process_job(job)
            except Exception as exc:
                fail_job(job, error=str(exc), message="Job failed")

            if run_once:
                break

        close_old_connections()

    def _process_job(self, job):
        if job.kind == "github_import":
            if not job.user:
                fail_job(job, error="User context missing for import job", message="Import failed")
                return
            update_job(str(job.id), message="Starting import", detail="")
            branddozer_views._run_import_job(str(job.id), job.user, job.payload)
            return

        if job.kind == "github_publish":
            if not job.user:
                fail_job(job, error="User context missing for publish job", message="Publish failed")
                return
            if not job.project_id:
                fail_job(job, error="Project not found for publish job", message="Publish failed")
                return
            update_job(str(job.id), message="Pushing to GitHub", detail="")
            branddozer_views._run_publish_job(str(job.id), job.user, str(job.project_id), job.payload)
            return

        if job.kind == "delivery_run":
            if not job.run_id:
                fail_job(job, error="Delivery run missing", message="Delivery run failed")
                return
            update_job(str(job.id), message="Running delivery pipeline", detail="")
            delivery_orchestrator.run_existing(job.run_id)
            run = DeliveryRun.objects.filter(id=job.run_id).first()
            if run and run.status == "error":
                fail_job(job, error=run.error or "Delivery run failed", message="Delivery run failed")
                return
            if run and run.status == "blocked":
                complete_job(job, message="Delivery run blocked", result={"run_status": run.status})
                return
            complete_job(
                job,
                message="Delivery run complete",
                result={"run_status": run.status if run else "unknown", "completed_at": timezone.now().isoformat()},
            )
            return

        if job.kind == "ui_capture":
            if not job.run_id:
                fail_job(job, error="Delivery run missing", message="UI capture failed")
                return
            update_job(str(job.id), message="Capturing UI", detail="")
            manual = bool(job.payload.get("manual", True)) if isinstance(job.payload, dict) else True
            delivery_orchestrator.run_ui_review(job.run_id, manual=manual)
            complete_job(job, message="UI capture complete")
            return

        fail_job(job, error=f"Unknown job kind: {job.kind}", message="Job failed")
