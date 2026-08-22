"""
Resolve directories that the process needs to write to.

Several modules compute a working directory from their own location
(``Path(__file__).parents[N] / "storage" / ...``) and create it at import time.
That is correct for the Waitress deployment, where the checkout is writable,
but fails on AWS Lambda: the deployment bundle is mounted read-only at
``/var/task`` and only ``/tmp`` accepts writes.  A ``PermissionError`` there
happens during ``import``, so it takes down the whole URLconf rather than the
one view that needed the directory.

``ensure_dir`` keeps the existing path when it is usable and transparently
falls back to a ``/tmp`` equivalent when it is not, so callers keep their
current behaviour everywhere except where the old behaviour could not work.

Set ``WRITABLE_ROOT`` to force a specific base (the Lambda handlers set
``/tmp``); otherwise the fallback is derived automatically.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def writable_root() -> Path | None:
    """The base to relocate under, or None when the original path is fine."""
    forced = os.getenv("WRITABLE_ROOT")
    if forced:
        return Path(forced)
    if os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        return Path(tempfile.gettempdir())
    return None


def ensure_dir(path: Path, *, anchor: Path | None = None) -> Path:
    """
    Create *path* and return it; return a writable stand-in if that fails.

    ``anchor`` is the source root *path* was derived from. When given, the
    relocated directory preserves the sub-path below it, so
    ``<bundle>/runtime/branddozer/sessions`` becomes
    ``/tmp/runtime/branddozer/sessions`` rather than collapsing to a basename
    and colliding with a different caller's directory.
    """
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        pass  # read-only filesystem -- fall through to the relocation below

    root = writable_root() or Path(tempfile.gettempdir())
    relative: Path | None = None
    if anchor is not None:
        try:
            relative = path.resolve().relative_to(Path(anchor).resolve())
        except (ValueError, OSError):
            relative = None
    if relative is None:
        # Keep the last two components: enough to stay unique between callers
        # without embedding an absolute path from the build machine.
        parts = path.parts[-2:] if len(path.parts) >= 2 else path.parts[-1:]
        relative = Path(*parts)

    fallback = root / relative
    try:
        fallback.mkdir(parents=True, exist_ok=True)
        logger.debug("relocated unwritable %s -> %s", path, fallback)
        return fallback
    except OSError:
        logger.warning("no writable location for %s; using %s", path, root)
        return root
