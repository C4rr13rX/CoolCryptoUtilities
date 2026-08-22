"""
Cold-start bootstrap shared by every handler.

Must be imported *before* Django or any ``services.*`` module.

The problem it solves: dozens of modules across ``services/`` and ``trading/``
derive their working directories from ``Path(__file__).parents[N]`` and then
call ``mkdir(parents=True, exist_ok=True)`` at import time --
``services/wallet_state.py`` is the one the telemetry URLconf hits first::

    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    STATE_DIR = _PROJECT_ROOT / "storage" / "wallet_state"
    STATE_DIR.mkdir(parents=True, exist_ok=True)     # PermissionError on Lambda

On Lambda that root is ``/var/task``, which is mounted read-only, so the import
raises ``PermissionError`` and every request 500s before reaching a view.

The fix has two halves:

1. ``services.writable_paths.ensure_dir`` wraps those ``mkdir`` calls so an
   unwritable target falls back to a ``/tmp`` equivalent instead of raising.
2. This module sets ``WRITABLE_ROOT=/tmp`` before anything imports, so the
   fallback is deterministic rather than guessed per call site.

A symlink (``/var/task/storage`` -> ``/tmp/storage``) would have avoided
touching the source, but creating one requires writing inside ``/var/task`` --
precisely the permission Lambda withholds.

The trade-off is that ``/tmp`` is per-sandbox and cleared between cold starts.
That is correct for these paths: they hold caches, scratch state and logs.
Anything that must survive belongs in Postgres or S3 -- notably the cron
schedule, which now lives in EventBridge rather than a state file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_done = False


def prepare_writable_dirs(bundle_root: Path | None = None) -> None:
    """Point every relocatable write target at /tmp. Idempotent per sandbox."""
    global _done
    if _done:
        return
    _done = True

    # Outside Lambda (local dev, tests) the checkout is writable and
    # redirecting would be wrong -- persistence is the point there.
    if not os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        return

    # services.writable_paths.ensure_dir consults this when a mkdir fails, so
    # every import-time directory lands under /tmp instead of raising
    # PermissionError against the read-only bundle.
    #
    # Note a symlink from /var/task/storage -> /tmp/storage cannot work here:
    # creating one requires writing *inside* /var/task, which is exactly the
    # permission Lambda withholds.
    os.environ.setdefault("WRITABLE_ROOT", "/tmp")

    for name in ("storage", "runtime", "logs", "data"):
        try:
            (Path("/tmp") / name).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("bootstrap: could not create /tmp/%s (%s)", name, exc)

    # Modules with their own override honour it directly, which is more
    # predictable than relying on the ensure_dir fallback.
    os.environ.setdefault("CRON_STATE_PATH", "/tmp/runtime/cron/state.json")
    os.environ.setdefault("PUBLIC_API_CACHE_DIR", "/tmp/data/public_api_cache")
    os.environ.setdefault("HOME", "/tmp")
    # Matplotlib/fontconfig and friends write caches into $HOME by default and
    # abort noisily when it is unwritable.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/.mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
