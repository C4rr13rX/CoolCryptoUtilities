from __future__ import annotations

from django.core.management import BaseCommand

from services.dora_metrics import generate_snapshot, write_snapshot


class Command(BaseCommand):
    help = "Generate DORA metrics snapshot and optionally persist it under runtime/branddozer/reports."

    def add_arguments(self, parser):
        parser.add_argument("--window-days", type=int, default=7, help="Time window (days) to evaluate.")
        parser.add_argument(
            "--output-dir",
            default="runtime/branddozer/reports",
            help="Directory where the DORA report JSON should be written.",
        )
        parser.add_argument(
            "--no-write",
            action="store_true",
            help="Only print the snapshot to stdout without persisting it.",
        )

    def handle(self, *args, **options):
        window_days = int(options["window_days"])
        snapshot = generate_snapshot(window_days=window_days)
        self.stdout.write(self.style.SUCCESS(f"DORA snapshot ({window_days}d window):"))
        for key, value in snapshot.to_dict().items():
            self.stdout.write(f" - {key}: {value}")
        if options.get("no_write"):
            return
        path = write_snapshot(snapshot, options["output_dir"])
        self.stdout.write(self.style.HTTP_INFO(f"Wrote snapshot to {path}"))
