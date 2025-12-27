from __future__ import annotations

import os

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Ensure a UI automation admin user exists (creates or updates as needed)."

    def add_arguments(self, parser):
        parser.add_argument("--username", default=os.getenv("BRANDDOZER_UI_ADMIN_USER", "branddozer_qa"))
        parser.add_argument("--password", default=os.getenv("BRANDDOZER_UI_ADMIN_PASS", ""))
        parser.add_argument("--email", default=os.getenv("BRANDDOZER_UI_ADMIN_EMAIL", ""))
        parser.add_argument("--reset", action="store_true", help="Reset password even if user exists.")

    def handle(self, *args, **options):
        username = (options.get("username") or "").strip()
        password = (options.get("password") or "").strip()
        email = (options.get("email") or "").strip() or f"{username}@local.test"
        reset = bool(options.get("reset"))
        if not username:
            self.stderr.write("username is required")
            return
        if not password:
            self.stderr.write("password is required")
            return
        User = get_user_model()
        user, created = User.objects.get_or_create(username=username, defaults={"email": email})
        changed = created
        if created or reset:
            user.set_password(password)
            changed = True
        if not user.is_staff or not user.is_superuser:
            user.is_staff = True
            user.is_superuser = True
            changed = True
        if not user.email:
            user.email = email
            changed = True
        if changed:
            user.save()
        status = "created" if created else "updated" if changed else "unchanged"
        self.stdout.write(f"UI admin {status}: {username}")
