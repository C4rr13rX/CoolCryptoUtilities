from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")

import django
# Prevent settings.py -> EnvLoader -> secure_settings from recursively calling
# django.setup() before INSTALLED_APPS has been defined. Hydrate the vault only
# after the registry is complete.
_secure_env_was_set = "SECURE_ENV_HYDRATED" in os.environ
os.environ["SECURE_ENV_HYDRATED"] = "1"
django.setup()
if not _secure_env_was_set:
    os.environ.pop("SECURE_ENV_HYDRATED", None)
from services.env_loader import EnvLoader
EnvLoader.load()

from services.branddozer_runner import branddozer_manager
from services.branddozer_state import list_projects
from services.guardian_lock import GuardianLease

HEARTBEAT = ROOT / "runtime" / "guardian" / "branddozer-keeper-heartbeat.json"


def main() -> None:
    lease = GuardianLease("branddozer-keeper-process", poll_interval=1.0)
    if not lease.acquire():
        return
    try:
        while True:
            enabled = []
            for project in list_projects():
                if project.get("enabled"):
                    enabled.append(project["id"])
                    # Advanced delivery is owned by the durable BackgroundJob
                    # worker. Starting the in-process refinement loop here
                    # produces an "Unknown workflow" error and leaves the real
                    # DeliveryRun orphaned in a running state.
                    if str(project.get("workflow_kind") or "").strip().lower() == "advanced_delivery":
                        continue
                    branddozer_manager.start(project["id"])
            HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
            HEARTBEAT.write_text(json.dumps({"timestamp": time.time(), "pid": os.getpid(), "enabled_projects": enabled}), encoding="utf-8")
            time.sleep(20)
    finally:
        lease.release()


if __name__ == "__main__":
    main()
