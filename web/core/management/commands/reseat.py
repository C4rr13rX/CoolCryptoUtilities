from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from django.core.management import BaseCommand, call_command



class Command(BaseCommand):
    help = "Build frontend, apply migrations, collect static, then runserver (in that order)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--noinstall",
            action="store_true",
            help="Skip npm install step (default runs npm install before build)",
        )
        parser.add_argument(
            "--runserver-args",
            nargs="*",
            default=None,
            help="Additional arguments to pass through to runserver (e.g. --runserver-args 0.0.0.0:8001)",
        )
        parser.add_argument(
            "--guardian-off",
            action="store_true",
            help="Disable guardian/production auto-start even if enabled in settings.",
        )
        parser.add_argument(
            "--production-off",
            action="store_true",
            help="Disable production manager auto-start even if enabled in settings.",
        )
        parser.add_argument(
            "--branddozer-worker-off",
            action="store_true",
            help="Disable BrandDozer background worker auto-start.",
        )

    def handle(self, *args, **options):
        project_root = Path(__file__).resolve().parents[4]
        frontend_dir = project_root / "web" / "frontend"

        def run(cmd, cwd=None):
            self.stdout.write(self.style.HTTP_INFO(f"$ {' '.join(cmd)}"))
            subprocess.run(cmd, cwd=cwd, check=True)

        # Respect disable flags before Django finishes loading apps that auto-start services.
        if options.get("guardian_off"):
            os.environ["GUARDIAN_AUTO_DISABLED"] = "1"
        if options.get("production_off"):
            os.environ["PRODUCTION_AUTO_DISABLED"] = "1"

        self.stdout.write(self.style.MIGRATE_HEADING("[1/6] Frontend build"))
        if not options["noinstall"]:
            run(["npm", "install"], cwd=str(frontend_dir))
        run(["npm", "run", "build"], cwd=str(frontend_dir))

        self.stdout.write(self.style.MIGRATE_HEADING("[2/6] Make migrations"))
        call_command("makemigrations")

        self.stdout.write(self.style.MIGRATE_HEADING("[3/6] Apply migrations"))
        call_command("migrate")

        self.stdout.write(self.style.MIGRATE_HEADING("[4/6] Collect static"))
        call_command("collectstatic", interactive=False)

        worker_process = None
        if not options.get("branddozer_worker_off") and os.environ.get("BRANDDOZER_WORKER_DISABLED") != "1":
            self.stdout.write(self.style.MIGRATE_HEADING("[5/6] Start BrandDozer worker"))
            worker_env = os.environ.copy()
            worker_env.setdefault("SECURE_VAULT_KEY_DIR", str(project_root / "storage" / "secure_vault"))
            worker_process = subprocess.Popen(
                [sys.executable, "web/manage.py", "branddozer_worker"],
                cwd=str(project_root),
                env=worker_env,
            )

        self.stdout.write(self.style.MIGRATE_HEADING("[6/6] Runserver"))
        runserver_args = options.get("runserver_args") or []
        if all(arg != "--noreload" for arg in runserver_args):
            runserver_args.append("--noreload")
        try:
            call_command("runserver", *runserver_args)
        finally:
            if worker_process and worker_process.poll() is None:
                worker_process.terminate()
