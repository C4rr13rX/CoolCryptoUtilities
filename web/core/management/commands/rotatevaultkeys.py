from __future__ import annotations

from django.core.management.base import BaseCommand

from services import secure_settings


class Command(BaseCommand):
    help = "Regenerate the quantum-safe key pair used by the Secure Settings vault."

    def handle(self, *args, **options):
        secure_settings.rotate_keys()
        self.stdout.write(self.style.SUCCESS("Secure vault Kyber key pair rotated."))
        self.stdout.write(f"Keys written to: {secure_settings.key_directory()}")
