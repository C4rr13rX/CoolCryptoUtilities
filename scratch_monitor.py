import os, sys, django, time, json
from pathlib import Path
ROOT=Path(".").resolve(); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"web"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE","coolcrypto_dashboard.settings"); os.environ["SECURE_ENV_HYDRATED"]="1"
import django; django.setup()
from branddozer.models import DeliveryRun, BackgroundJob
RID="e1cb1af6-00a8-4638-bdda-1e6b3151f0a9"
last=None
for _ in range(180):  # up to ~90 min
    r=DeliveryRun.objects.get(id=RID); r.refresh_from_db()
    j=BackgroundJob.objects.filter(run_id=RID,kind="delivery_run").order_by("-created_at").first()
    key=(r.status,r.phase,j.status if j else None)
    if key!=last:
        print(f"[{time.strftime('%H:%M:%S')}] run={r.status}/{r.phase} job={j.status if j else '-'} err={(r.error or '')[:140]}", flush=True)
        last=key
    if r.status in ("complete","error","blocked","awaiting_acceptance") and (j is None or j.status in ("completed","error","failed")):
        print("TERMINAL", flush=True); break
    time.sleep(30)
