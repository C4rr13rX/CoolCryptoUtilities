from __future__ import annotations

import os
import sys
from typing import Any

from usage_tracker import UsageTracker


class HeaderRenderer:
    """Builds and renders the session header (model, tokens, cost, budget)."""

    def __init__(self, usage: UsageTracker, ui_manager: Any = None) -> None:
        self.usage = usage
        self.ui_manager = ui_manager
        self.enabled = os.getenv("C0D3R_HEADER", "1").strip().lower() not in {
            "0", "false", "no", "off",
        }
        self.ansi_ok = sys.stdout.isatty() and os.getenv(
            "C0D3R_HEADER_ANSI", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        self._last = ""
        self._frozen = False
        self.budget_usd = float(
            os.getenv("C0D3R_BUDGET_USD", "50.0") or "10.0"
        )
        self.budget_enabled = os.getenv(
            "C0D3R_BUDGET_ENABLED", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        if self.budget_usd < 0:
            self.budget_enabled = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        self._frozen = True

    def resume(self) -> None:
        self._frozen = False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> None:
        if not self.enabled:
            return
        header = self._build_header()
        if self.ui_manager:
            self.ui_manager.set_header(header)
            return
        try:
            if self.ansi_ok:
                sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(header)
            sys.stdout.flush()
            self._last = header
        except OSError:
            self.enabled = False

    def update(self) -> None:
        if not self.enabled or self._frozen:
            return
        header = self._build_header()
        if header == self._last:
            return
        if self.ui_manager:
            self.ui_manager.set_header(header)
            self._last = header
            return
        if not self.ansi_ok:
            return
        try:
            sys.stdout.write("\x1b7\x1b[H")
            sys.stdout.write(header)
            sys.stdout.write("\x1b8")
            sys.stdout.flush()
            self._last = header
        except OSError:
            self.enabled = False

    def render_text(self) -> str:
        """Return the header string without side effects."""
        return self._build_header()

    # ------------------------------------------------------------------
    # Header construction
    # ------------------------------------------------------------------

    def _build_header(self) -> str:
        try:
            from services.bedrock_pricing import estimate_cost, lookup_pricing
        except ImportError:
            return self._build_header_simple()

        model = self.usage.model_id or "unknown"
        in_cost, out_cost = estimate_cost(
            model, self.usage.input_tokens, self.usage.output_tokens
        )
        pricing = lookup_pricing(model)
        if in_cost is None or out_cost is None:
            cost_line = f"Est. cost: N/A (no pricing for {model})"
        else:
            cost_line = (
                f"Est. cost: ${in_cost + out_cost:.6f} "
                f"(in ${in_cost:.6f} / out ${out_cost:.6f})"
            )
        if pricing:
            rate_line = (
                f"Rates: ${pricing.input_per_1k:.4f}/1K in, "
                f"${pricing.output_per_1k:.4f}/1K out"
                f" | as of {pricing.as_of}"
            )
        else:
            rate_line = "Rates: unknown"
        token_line = (
            f"Tokens est: in {self.usage.input_tokens:.0f} "
            f"/ out {self.usage.output_tokens:.0f}"
        )
        status_line = f"Status: {self.usage.status}" + (
            f" | {self.usage.last_action}" if self.usage.last_action else ""
        )
        budget_line = f"Budget: ${self.budget_usd:.2f}"
        return (
            f"c0d3r session | model: {model}\n"
            f"{token_line} | {cost_line} | {budget_line}\n"
            f"{rate_line}\n"
            f"{status_line}\n"
            f"{'-' * 70}\n"
        )

    def _build_header_simple(self) -> str:
        """Fallback when bedrock_pricing is unavailable."""
        model = self.usage.model_id or "unknown"
        token_line = (
            f"Tokens est: in {self.usage.input_tokens:.0f} "
            f"/ out {self.usage.output_tokens:.0f}"
        )
        status_line = f"Status: {self.usage.status}" + (
            f" | {self.usage.last_action}" if self.usage.last_action else ""
        )
        budget_line = f"Budget: ${self.budget_usd:.2f}"
        return (
            f"c0d3r session | model: {model}\n"
            f"{token_line} | {budget_line}\n"
            f"{status_line}\n"
            f"{'-' * 70}\n"
        )
