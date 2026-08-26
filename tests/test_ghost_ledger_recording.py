"""
Every closed ghost trade must reach the strategy ledger.

The bug this guards: `services/atf_static_strategy.py` ran its own ghost cycle
and logged closed trades to `trading_ops` without ever calling
`StrategyLedger.record()`. 196 trades over four days were durably recorded and
completely invisible to the graduation gate, so the ledger sat frozen while the
strategy traded continuously.

It was silent because every individual part looked healthy: the bot was
running, trades were closing, rows were being written. Only the *ledger's*
mtime gave it away. A strategy that trades but never accumulates evidence can
never graduate, which makes this a bug that quietly defeats the entire purpose
of the system.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Modules that close ghost positions on their own, outside bot.py's exit path.
#: Each one must report its outcomes to the ledger.
GHOST_CLOSING_MODULES = [
    ROOT / "services" / "atf_static_strategy.py",
]


class GhostLedgerRecording(unittest.TestCase):
    def test_ghost_closing_modules_record_to_the_ledger(self):
        """
        Any module that writes a ghost-exit must also record it.

        Checked structurally rather than by running the loop: these cycles
        need live market data and a populated portfolio, so a behavioural test
        would be skipped in CI exactly when it mattered.
        """
        offenders = []
        for path in GHOST_CLOSING_MODULES:
            source = path.read_text(encoding="utf-8")
            writes_exit = "ghost-exit" in source
            records = (
                "StrategyLedger" in source
                or "strategy_ledger" in source
                or "_record_ghost_outcome" in source
            )
            if writes_exit and not records:
                offenders.append(path.name)
        self.assertFalse(
            offenders,
            "these close ghost trades without recording them to the ledger, "
            "so the graduation gate never sees them: " + ", ".join(offenders),
        )

    def test_no_new_module_closes_ghost_trades_unrecorded(self):
        """
        Catch a *new* module repeating the mistake.

        Scans services/ and trading/ for anything writing a ghost-exit status
        and requires it to reference the ledger. If a future loop is added
        without recording, this fails rather than silently freezing evidence.
        """
        offenders = []
        for folder in ("services", "trading"):
            for path in (ROOT / folder).rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                source = path.read_text(encoding="utf-8", errors="ignore")
                # Writers only: metrics.py reads these statuses for reporting.
                if 'status="ghost-exit"' not in source:
                    continue
                if not any(
                    token in source
                    for token in ("StrategyLedger", "strategy_ledger",
                                  "_record_ghost_outcome")
                ):
                    offenders.append(str(path.relative_to(ROOT)))
        self.assertFalse(
            offenders,
            "ghost trades closed without reaching the graduation gate: "
            + ", ".join(offenders),
        )

    def test_recording_helper_is_failure_tolerant(self):
        """
        A ledger write must never abort a trading cycle.

        Losing one outcome is recoverable; a raised exception that stops the
        loop is not. The helper has to swallow its own failures.
        """
        source = (ROOT / "services" / "atf_static_strategy.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        helper = next(
            (n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "_record_ghost_outcome"),
            None,
        )
        self.assertIsNotNone(helper, "_record_ghost_outcome is missing")
        self.assertTrue(
            any(isinstance(n, ast.Try) for n in ast.walk(helper)),
            "_record_ghost_outcome must not let a ledger failure escape",
        )

    def test_ledger_record_is_reachable_with_the_expected_signature(self):
        """Pin the call contract the helper depends on."""
        import inspect

        from trading.strategies.ledger import StrategyLedger

        sig = inspect.signature(StrategyLedger.record)
        for param in ("strategy_id", "profit", "mode"):
            self.assertIn(param, sig.parameters)


if __name__ == "__main__":
    unittest.main()
