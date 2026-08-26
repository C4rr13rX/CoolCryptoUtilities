"""
No blocking call may sit directly inside an async function.

`TradingBot._start_background_refinement` called
`governor.wait_if_pressured(max_wait=120.0)` directly from a coroutine. That
function is synchronous and waits with `time.sleep()`, so it blocked the whole
event loop rather than just its own task. Every market stream shares that loop,
so one bot waiting out CPU pressure froze price acquisition for all ~30 of them
for up to 120 seconds.

The signature that gave it away: writes arriving in synchronised bursts
250-780s apart across completely unrelated symbols. Per-stream faults do not
synchronise; a blocked event loop does.

This was the last of four separate causes of the same symptom -- a frozen
price feed -- and the hardest to see, because every component tested clean in
isolation. A lint-style guard is worth more than another one-off fix here: the
next `time.sleep()` in a coroutine would be just as invisible.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: (module, attribute) pairs that block the calling thread.
BLOCKING_CALLS = {
    ("time", "sleep"),
    ("requests", "get"),
    ("requests", "post"),
    ("requests", "put"),
    ("subprocess", "run"),
    ("subprocess", "call"),
    ("urllib", "urlopen"),
}

#: Bare attribute names that block regardless of what they are called on.
BLOCKING_ATTRS = {"wait_if_pressured"}

SEARCH_DIRS = ("trading", "services")


class _AsyncBlockingVisitor(ast.NodeVisitor):
    """Find blocking calls lexically inside an `async def`, ignoring nested defs."""

    def __init__(self) -> None:
        self.async_depth = 0
        self.hits: list[tuple[int, str]] = []

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.async_depth += 1
        self.generic_visit(node)
        self.async_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # A sync def nested in an async def runs on whatever thread calls it,
        # typically via to_thread, so it is not the loop's problem.
        saved, self.async_depth = self.async_depth, 0
        self.generic_visit(node)
        self.async_depth = saved

    def visit_Call(self, node: ast.Call) -> None:
        if self.async_depth > 0 and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in BLOCKING_ATTRS:
                self.hits.append((node.lineno, f"….{attr}()"))
            elif isinstance(node.func.value, ast.Name):
                pair = (node.func.value.id, attr)
                if pair in BLOCKING_CALLS:
                    self.hits.append((node.lineno, f"{pair[0]}.{pair[1]}()"))
        self.generic_visit(node)


def _scan(path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    visitor = _AsyncBlockingVisitor()
    visitor.visit(tree)
    return visitor.hits


class NoBlockingCallsInAsync(unittest.TestCase):
    def test_no_blocking_calls_inside_coroutines(self):
        """
        Wrap blocking work in `await asyncio.to_thread(...)` instead.

        `asyncio.sleep` is the async equivalent of `time.sleep`; anything that
        genuinely has to block belongs on a worker thread.
        """
        offenders = []
        for folder in SEARCH_DIRS:
            for path in sorted((ROOT / folder).rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                for lineno, what in _scan(path):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno} {what}")
        self.assertFalse(
            offenders,
            "blocking calls inside async functions stall the entire event "
            "loop, freezing every market stream that shares it:\n  "
            + "\n  ".join(offenders),
        )

    def test_the_detector_actually_detects(self):
        """A guard that cannot fail is worthless; prove it catches the pattern."""
        source = (
            "import time\n"
            "async def poll():\n"
            "    time.sleep(120)\n"
        )
        visitor = _AsyncBlockingVisitor()
        visitor.visit(ast.parse(source))
        self.assertEqual(len(visitor.hits), 1)

    def test_a_sync_helper_is_not_flagged(self):
        """Blocking inside a plain def is fine — that is what to_thread runs."""
        source = (
            "import time\n"
            "def blocking_helper():\n"
            "    time.sleep(120)\n"
            "async def caller():\n"
            "    await asyncio.to_thread(blocking_helper)\n"
        )
        visitor = _AsyncBlockingVisitor()
        visitor.visit(ast.parse(source))
        self.assertEqual(visitor.hits, [])


if __name__ == "__main__":
    unittest.main()
