import os
from django.apps import AppConfig


class WizardChatConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "wizard_chat"
    verbose_name = "Wizard Chat"

    def ready(self) -> None:
        # Kick off the node-cache refresher BEFORE any request thread
        # is allocated.  Without this, the first /api/wizard-chat/...
        # request after every Django relaunch eats a cold-cache fetch
        # (10-20 s under training load) — long enough that the
        # supervisor's `/` probe times out in parallel and kills
        # Django, restarting the cycle.  Spawning on AppConfig.ready
        # means the cache is being warmed before waitress even accepts
        # its first connection.
        #
        # Skip in manage.py subcommands (migrate, collectstatic, etc.)
        # so they don't leave a daemon thread behind.
        if os.environ.get("RUN_MAIN") == "true" or \
           os.environ.get("WIZARD_DISABLE_REFRESHER") == "1":
            return
        # Only start under waitress (or any WSGI request loop).
        # `WAITRESS_PORT` is set by run_waitress.py before importing.
        if not os.environ.get("WAITRESS_PORT"):
            return
        try:
            from . import views
            views._ensure_node_refresher()
        except Exception:
            # Never let refresher startup crash Django boot.
            pass
