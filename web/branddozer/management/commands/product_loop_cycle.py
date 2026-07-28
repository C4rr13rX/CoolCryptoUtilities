from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand

from services.branddozer_product_loop import run_product_loop_cycle


class Command(BaseCommand):
    help = "Run BrandDozer's C0D3R+ATF digital product continuous-refinement loop."

    def add_arguments(self, parser):
        parser.add_argument("--root", default="", help="Workspace root. Defaults to ~/Desktop/Apps/BrandDozerDigitalProducts.")
        parser.add_argument("--cycles", type=int, default=1, help="Number of loop cycles to run.")

    def handle(self, *args, **options):
        root = Path(options["root"]).expanduser() if options.get("root") else None
        result = run_product_loop_cycle(root=root, max_cycles=options.get("cycles") or 1)
        self.stdout.write(json.dumps({"project": result.project, "state": result.state}, indent=2))
