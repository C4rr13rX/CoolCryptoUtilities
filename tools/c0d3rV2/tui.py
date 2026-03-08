"""
Legacy TUI module — kept as a thin compatibility shim.

All TUI functionality now lives in the individual V2 modules:

  terminal_ui.py    TerminalUI      Multi-backend TUI (Textual / prompt_toolkit / Rich)
  header_renderer.py HeaderRenderer Session header display
  usage_tracker.py  UsageTracker    Token / status tracking
  budget_tracker.py BudgetTracker   API spend guard

Initialisation happens in ProcessFlow (process_flow.py) and the CLI
entry point (c0d3rV2_cli.py).
"""
from __future__ import annotations

# Re-export for anything that might still import from here.
from terminal_ui import TerminalUI       # noqa: F401
from header_renderer import HeaderRenderer  # noqa: F401
from usage_tracker import UsageTracker     # noqa: F401
from budget_tracker import BudgetTracker   # noqa: F401
