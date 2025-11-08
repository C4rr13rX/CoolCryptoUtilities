from __future__ import annotations

import subprocess
from pathlib import Path

from django.core.management import BaseCommand, call_command


class Command(BaseCommand):
    help = "Builds the Vue frontend, runs migrations, collects static, and launches runserver."

    def handle(self, *args, **options):
        root = Path(__file__).resolve().parents[3]
        frontend = root / "web" / "frontend"
        self.stdout.write(self.style.MIGRATE_HEADING("1/4 npm run build"))
        subprocess.run(["npm", "install"], cwd=str(frontend), check=True)
        subprocess.run(["npm", "run", "build"], cwd=str(frontend), check=True)

        self.stdout.write(self.style.MIGRATE_HEADING("2/4 migrate"))
        call_command("migrate")

        self.stdout.write(self.style.MIGRATE_HEADING("3/4 collectstatic"))
        call_command("collectstatic", interactive=False, verbosity=0)

        self.stdout.write(self.style.MIGRATE_HEADING("4/4 runserver"))
        call_command("runserver")
