from __future__ import annotations

from typing import Optional

try:  # pragma: no cover - optional dependency
    from setproctitle import setproctitle as _setproctitle  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade
    _setproctitle = None  # type: ignore


def set_process_name(name: Optional[str]) -> None:
    """
    Attempt to rename the current process for tools like `top`/`ps`.
    Silently ignores environments where setproctitle is unavailable.
    """

    if not name:
        return
    if _setproctitle is None:
        return
    try:
        _setproctitle(str(name))
    except Exception:
        return
