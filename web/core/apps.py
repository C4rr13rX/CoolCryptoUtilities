from __future__ import annotations

import threading
import time

import os
import sys

from django.apps import AppConfig, apps
from django.conf import settings


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"
    verbose_name = "Dashboard Core"
    _guardian_started = False
    _streams_started = False
    _cron_started = False
    _production_started = False

    def ready(self):
        if getattr(settings, "TESTING", False):
            return
        # Avoid duplicate guardian bootstrap when Django's autoreloader spawns the
        # monitor parent process. Only the reloaded "main" process should manage it.
        cmd = sys.argv[1] if len(sys.argv) > 1 else ""
        if cmd == "runserver" and os.environ.get("RUN_MAIN") != "true":
            return
        if os.environ.get("GUARDIAN_AUTO_DISABLED") == "1":
            return
        if CoreConfig._guardian_started:
            return
        CoreConfig._guardian_started = True

        def _bootstrap():
            try:
                while not apps.ready:
                    time.sleep(0.05)
                from services.guardian_supervisor import guardian_supervisor

                guardian_supervisor.start_if_enabled()
            except Exception:
                CoreConfig._guardian_started = False

        threading.Thread(target=_bootstrap, name="guardian-bootstrap", daemon=True).start()

        if not CoreConfig._streams_started:
            try:
                from services.console_stream import start_console_streams

                start_console_streams()
                CoreConfig._streams_started = True
            except Exception:
                pass

        # Auto-start production manager when wallet credentials are available.
        if (
            not CoreConfig._production_started
            and os.environ.get("PRODUCTION_AUTO_DISABLED") != "1"
        ):
            CoreConfig._production_started = True

            def _production_bootstrap():
                try:
                    while not apps.ready:
                        time.sleep(0.05)
                    # Give guardian a moment to settle before spawning production.
                    time.sleep(2.0)
                    from services.guardian_status import snapshot_status
                    state = snapshot_status()
                    if state.get("production", {}).get("running"):
                        return
                    # Only auto-start if wallet credentials exist in SecureVault.
                    from securevault.models import SecureSetting
                    has_wallet = SecureSetting.objects.filter(
                        name__in=["MNEMONIC", "PRIVATE_KEY"]
                    ).exists()
                    if not has_wallet:
                        return
                    from opsconsole.manager import manager as console_manager
                    console_manager.start()
                except Exception:
                    CoreConfig._production_started = False

            threading.Thread(
                target=_production_bootstrap,
                name="production-bootstrap",
                daemon=True,
            ).start()

        if os.environ.get("CRON_AUTO_DISABLED") == "1":
            return
        if CoreConfig._cron_started:
            return
        CoreConfig._cron_started = True

        def _cron_bootstrap():
            try:
                while not apps.ready:
                    time.sleep(0.05)
                from services.internal_cron import cron_supervisor

                cron_supervisor.ensure_running()
            except Exception:
                CoreConfig._cron_started = False

        threading.Thread(target=_cron_bootstrap, name="cron-bootstrap", daemon=True).start()
