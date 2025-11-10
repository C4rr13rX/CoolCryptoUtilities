from __future__ import annotations

import os
import subprocess
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

    def handle(self, *args, **options):
        project_root = Path(__file__).resolve().parents[4]
        frontend_dir = project_root / "web" / "frontend"

        def run(cmd, cwd=None):
            self.stdout.write(self.style.HTTP_INFO(f"$ {' '.join(cmd)}"))
            subprocess.run(cmd, cwd=cwd, check=True)

        self.stdout.write(self.style.MIGRATE_HEADING("[1/4] Frontend build"))
        if not options["noinstall"]:
            run(["npm", "install"], cwd=str(frontend_dir))
        run(["npm", "run", "build"], cwd=str(frontend_dir))

        self.stdout.write(self.style.MIGRATE_HEADING("[2/4] Database migrations"))
        call_command("migrate")

        self.stdout.write(self.style.MIGRATE_HEADING("[3/4] Collect static"))
        call_command("collectstatic", interactive=False)

        self.stdout.write(self.style.MIGRATE_HEADING("[4/4] Runserver"))
        runserver_args = options.get("runserver_args") or []
        if options.get("guardian_off"):
            os.environ["GUARDIAN_AUTO_DISABLED"] = "1"
        call_command("runserver", *runserver_args)
