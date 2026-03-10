from __future__ import annotations

import atexit
import logging
import signal
import threading
import time

import os
import sys

from django.apps import AppConfig, apps
from django.conf import settings

_shutdown_logger = logging.getLogger("core.shutdown")
_shutdown_lock = threading.Lock()
_shutdown_done = False


def _shutdown_all() -> None:
    """Stop every subprocess / background service that CoreConfig started."""
    global _shutdown_done
    with _shutdown_lock:
        if _shutdown_done:
            return
        _shutdown_done = True

    _shutdown_logger.info("shutdown: stopping all managed services …")

    # 1. Production manager (spawned subprocess)
    try:
        from opsconsole.manager import manager as console_manager
        console_manager.stop(graceful_timeout=8.0)
        _shutdown_logger.info("shutdown: production manager stopped")
    except Exception as exc:
        _shutdown_logger.warning("shutdown: production manager stop failed -> %s", exc)

    # 2. Guardian supervisor (spawned subprocess)
    try:
        from services.guardian_supervisor import guardian_supervisor
        guardian_supervisor.stop()
        _shutdown_logger.info("shutdown: guardian stopped")
    except Exception as exc:
        _shutdown_logger.warning("shutdown: guardian stop failed -> %s", exc)

    # 3. Cron supervisor (background thread)
    try:
        from services.internal_cron import cron_supervisor
        cron_supervisor.stop()
        _shutdown_logger.info("shutdown: cron stopped")
    except Exception as exc:
        _shutdown_logger.warning("shutdown: cron stop failed -> %s", exc)

    # 4. Console stream file tailers
    try:
        from services.console_stream import _STREAMERS
        for streamer in _STREAMERS:
            try:
                streamer.stop()
            except Exception:
                pass
        _shutdown_logger.info("shutdown: console streams stopped")
    except Exception as exc:
        _shutdown_logger.warning("shutdown: console streams stop failed -> %s", exc)

    _shutdown_logger.info("shutdown: all services stopped")


def _signal_handler(signum, frame):
    """Handle SIGINT / SIGTERM by tearing down services, then re-raising."""
    _shutdown_all()
    # Re-raise so Python's default handler can exit cleanly.
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"
    verbose_name = "Dashboard Core"
    _guardian_started = False
    _streams_started = False
    _cron_started = False
    _production_started = False
    _shutdown_registered = False

    def ready(self):
        if getattr(settings, "TESTING", False):
            return
        # Avoid duplicate guardian bootstrap when Django's autoreloader spawns the
        # monitor parent process. Only the reloaded "main" process should manage it.
        cmd = sys.argv[1] if len(sys.argv) > 1 else ""
        if cmd == "runserver" and os.environ.get("RUN_MAIN") != "true":
            return

        # Register shutdown hooks once so Ctrl+C / normal exit cleans everything.
        if not CoreConfig._shutdown_registered:
            CoreConfig._shutdown_registered = True
            atexit.register(_shutdown_all)
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    signal.signal(sig, _signal_handler)
                except (OSError, ValueError):
                    # signal.signal can only be called from the main thread;
                    # atexit covers the fallback.
                    pass
        # Guardian bootstrap — only when explicitly enabled.
        if os.environ.get("GUARDIAN_AUTO_DISABLED") != "1" and not CoreConfig._guardian_started:
            CoreConfig._guardian_started = True

            def _bootstrap():
                import logging
                logger = logging.getLogger("core.apps")
                try:
                    while not apps.ready:
                        time.sleep(0.05)
                    from services.guardian_supervisor import guardian_supervisor

                    guardian_supervisor.start_if_enabled()
                    logger.info("guardian-bootstrap: started")
                except Exception as exc:
                    logger.error("guardian-bootstrap: failed -> %s", exc)
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
                import logging
                logger = logging.getLogger("core.apps")
                try:
                    while not apps.ready:
                        time.sleep(0.05)
                    # Give guardian a moment to settle before spawning production.
                    time.sleep(2.0)
                    from services.guardian_status import snapshot_status
                    state = snapshot_status()
                    if state.get("production", {}).get("running"):
                        logger.info("production-bootstrap: already running, skipping")
                        return
                    # Only auto-start if wallet credentials exist in SecureVault.
                    from securevault.models import SecureSetting
                    has_wallet = SecureSetting.objects.filter(
                        name__in=["MNEMONIC", "PRIVATE_KEY"]
                    ).exists()
                    if not has_wallet:
                        logger.info("production-bootstrap: no wallet credentials found, skipping")
                        return
                    from opsconsole.manager import manager as console_manager
                    result = console_manager.start()
                    logger.info("production-bootstrap: started -> %s", result)
                except Exception as exc:
                    import traceback
                    logger.error("production-bootstrap: failed -> %s\n%s", exc, traceback.format_exc())
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
            import logging
            logger = logging.getLogger("core.apps")
            try:
                while not apps.ready:
                    time.sleep(0.05)
                from services.internal_cron import cron_supervisor

                cron_supervisor.ensure_running()
                logger.info("cron-bootstrap: started")
            except Exception as exc:
                logger.error("cron-bootstrap: failed -> %s", exc)
                CoreConfig._cron_started = False

        threading.Thread(target=_cron_bootstrap, name="cron-bootstrap", daemon=True).start()
