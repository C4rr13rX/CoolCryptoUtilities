import os, sys, django
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"web"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE","coolcrypto_dashboard.settings")
os.environ["SECURE_ENV_HYDRATED"]="1"
import django; django.setup()
from services.env_loader import EnvLoader; EnvLoader.load()
from django.utils import timezone
from branddozer.models import DeliveryRun, BrandProject, BackgroundJob
from services.branddozer_jobs import enqueue_job

RUN_ID = "e1cb1af6-00a8-4638-bdda-1e6b3151f0a9"
run = DeliveryRun.objects.get(id=RUN_ID)
proj = run.project

# find original delivery_run job to reuse its user
orig = BackgroundJob.objects.filter(run=run, kind="delivery_run").order_by("-created_at").first()
user = orig.user if orig else None

# enable project (cosmetic; worker is what drives it)
if hasattr(proj, "enabled") and not proj.enabled:
    proj.enabled = True
    proj.save(update_fields=["enabled"])

# reset the errored run to a runnable state so the worker won't short-circuit
ctx = dict(run.context or {})
ctx.pop("stop_requested", None)
run.context = ctx
run.status = "queued"
run.phase = "queued"
run.error = ""
run.completed_at = None
run.save(update_fields=["status","phase","error","completed_at","context"])

job = enqueue_job(kind="delivery_run", project=proj, run=run, user=user,
                  message="Re-queued after catalogue-gate fixes")
print("project:", proj.id, "enabled:", getattr(proj,"enabled",None))
print("run reset -> status:", run.status, "phase:", run.phase)
print("new job:", job.id, "user:", user)
