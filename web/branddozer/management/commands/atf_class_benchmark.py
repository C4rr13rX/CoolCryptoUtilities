from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from services.atf_class_refinement import run_class_refinement_benchmark


class Command(BaseCommand):
    help = "Benchmark and refine C0D3R+AgentTheFreeloader class generation."

    def add_arguments(self, parser):
        parser.add_argument("--count", type=int, default=4)
        parser.add_argument("--attempts", type=int, default=2)

    def handle(self, *args, **options):
        summary = run_class_refinement_benchmark(
            count=options.get("count") or 4,
            attempts=options.get("attempts") or 2,
        )
        self.stdout.write(json.dumps(summary, indent=2))
