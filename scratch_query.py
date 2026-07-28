import os, sys, django
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"web"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE","coolcrypto_dashboard.settings")
os.environ["SECURE_ENV_HYDRATED"]="1"
django.setup()
from branddozer.models import DeliveryRun, BrandProject
try:
    from services.branddozer_jobs import BackgroundJob
except Exception:
    from branddozer.models import BackgroundJob
import json
print("=== BrandProjects ===")
for p in BrandProject.objects.all():
    print(p.id, "|", getattr(p,'name',''), "| wf=", getattr(p,'workflow_kind',''), "| enabled=", getattr(p,'enabled',''), "| root=", getattr(p,'root_path',''))
print("\n=== DeliveryRuns (recent) ===")
for r in DeliveryRun.objects.order_by('-created_at')[:8]:
    print(r.id, "| status=",r.status,"| phase=",r.phase,"| created=",r.created_at,"| completed=",r.completed_at)
    print("     prompt:", (r.prompt or "")[:120].replace("\n"," "))
    print("     error:", (r.error or "")[:200])
print("\n=== BackgroundJobs (recent) ===")
for j in BackgroundJob.objects.order_by('-created_at')[:10]:
    print(j.id,"| kind=",j.kind,"| status=",j.status,"| worker=",getattr(j,'worker',''),"| created=",j.created_at,"| updated=",getattr(j,'updated_at',''))
    print("     msg:", (getattr(j,'message','') or '')[:120], "| err:", (getattr(j,'error','') or '')[:160])
