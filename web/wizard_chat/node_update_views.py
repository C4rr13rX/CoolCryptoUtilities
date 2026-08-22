"""
Wizard-node update endpoints.

Exposes the same updater the Android WorkManager job uses, so the node can be
updated on demand from the GUI as well as on its six-hour schedule.

Deliberately thin: the download, hash verification and atomic install all live
in ``wizard_updater`` (shipped in the Android Python sources) rather than being
duplicated here.  This module only decides *who may ask* and reports the
result.

Off Android the updater module is absent, and these endpoints say so rather
than 500 -- the desktop already has ``launch_revenir.ps1`` and the documented
``cargo build`` procedure for the same job.
"""

from __future__ import annotations

import importlib
import logging
import os

from rest_framework.permissions import IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView

logger = logging.getLogger(__name__)


def _updater():
    """Import the updater, or None when not running on-device."""
    try:
        return importlib.import_module("wizard_updater")
    except ImportError:
        return None


def _files_dir() -> str:
    """The app sandbox root, which the Android bootstrap pins for us."""
    return os.getenv("WRITABLE_ROOT", "")


class WizardNodeUpdateStatusView(APIView):
    """
    GET /api/wizard-chat/node/update/

    Reports the installed node build. Answers the question that has bitten
    this project before -- *which binary is actually running?* -- with the
    recorded version and hash rather than a guess.
    """

    permission_classes = [IsAdminUser]

    def get(self, request):
        updater = _updater()
        if updater is None:
            return Response({
                "supported": False,
                "reason": "on-device updater unavailable (not running on Android)",
            })
        return Response({"supported": True, **updater.status(_files_dir())})


class WizardNodeUpdateView(APIView):
    """
    POST /api/wizard-chat/node/update/

    Checks C4rr13rX for a newer node build and installs it if the hash
    verifies. ``{"force": true}`` reinstalls the current version, which is the
    escape hatch for a corrupted binary whose recorded version still matches.

    Admin-only: this replaces an executable that then runs on the device.
    """

    permission_classes = [IsAdminUser]

    def post(self, request):
        updater = _updater()
        if updater is None:
            return Response({
                "status": "unsupported",
                "reason": "on-device updater unavailable (not running on Android)",
            }, status=501)

        force = bool((request.data or {}).get("force"))
        result = updater.check_and_stage(_files_dir(), force=force)

        if result.get("status") == "installed":
            # The service restart is what makes the new binary take effect;
            # a staged-but-unloaded build is precisely the stale-node failure
            # documented in the deployment notes.
            _request_node_restart()

        code = 200 if result.get("status") in {"installed", "unchanged"} else 502
        return Response(result, status=code)


def _request_node_restart() -> None:
    """
    Ask WizardNodeService to reload the binary.

    Best-effort: on Android the Java layer also restarts the service when the
    WorkManager job installs an update, so a failure here delays the new build
    to the next restart rather than losing it.
    """
    try:
        from java import jclass  # type: ignore

        context = jclass("com.chaquo.python.android.AndroidPlatform").getApplication()
        intent = jclass("android.content.Intent")(
            context, jclass("com.coolcrypto.dashboard.services.WizardNodeService"))
        intent.setAction("com.coolcrypto.dashboard.RESTART_NODE")
        context.startForegroundService(intent)
    except Exception as exc:  # noqa: BLE001
        logger.info("node restart request skipped: %s", exc)
