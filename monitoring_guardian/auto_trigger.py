from __future__ import annotations

import logging
import traceback
from typing import Optional

logger = logging.getLogger(__name__)


def trigger_guardian_on_error(
    source: str,
    error_msg: str,
    exception: Optional[BaseException] = None,
) -> bool:
    """
    Automatically triggers Guardian remediation upon any unhandled error across the system.
    Starts Guardian even if it was disabled, passes diagnostic details, runs remediation,
    and returns to normal operation.
    """
    full_detail = error_msg
    if exception is not None:
        tb = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        full_detail = f"{error_msg}\n\nTraceback:\n{tb}"
    try:
        from services.guardian_supervisor import guardian_supervisor

        return guardian_supervisor.trigger_error_remediation(full_detail, source=source)
    except Exception as exc:
        logger.error("Guardian auto-trigger failed: %s", exc)
        print(f"[guardian-auto-trigger] Error launching auto-remediation: {exc}", flush=True)
        return False
