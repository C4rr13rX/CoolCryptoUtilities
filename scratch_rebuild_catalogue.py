import os, sys, json, time
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"web"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE","coolcrypto_dashboard.settings")
os.environ["SECURE_ENV_HYDRATED"]="1"
import django; django.setup()
from services.env_loader import EnvLoader; EnvLoader.load()
from services.branddozer_scientific_catalogue import build_catalogue
root = Path(r"C:\Users\Adam\Desktop\Multiscale Robot World")
t0=time.time()
res = build_catalogue(root)
print("ELAPSED", round(time.time()-t0,1),"s")
print("RESULT", json.dumps(res, indent=2))
