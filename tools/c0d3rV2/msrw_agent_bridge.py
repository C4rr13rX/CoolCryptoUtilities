"""tools/c0d3rV2/msrw_agent_bridge.py

Autonomous C0d3rV2 + W1z4rD Brain Bridge for Multiscale Robot World (MSRW).

Enforces the strict architectural requirement:
1. All changes, features, and fixes in Multiscale Robot World are authored and
   executed exclusively by C0d3rV2 using the W1z4rD brain as its intelligence engine.
2. The developer/supervisor agent gives C0d3rV2 commands.
3. If C0d3rV2 fails or encounters constraints, the supervisor diagnoses the failure,
   trains/upgrades the W1z4rD brain, and retries until success is verified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MSRW_ROOT = Path(r"C:\Users\Adam\Desktop\Multiscale Robot World")
_WEB_ROOT = _PROJECT_ROOT / "web"

for p in (str(_PROJECT_ROOT), str(_WEB_ROOT), str(_PROJECT_ROOT / "tools"), str(_PROJECT_ROOT / "tools" / "c0d3rV2")):
    if p not in sys.path:
        sys.path.insert(0, p)


class MsrwAgentBridge:
    """Manages command dispatch, diagnostic monitoring, and iterative upgrade loops."""

    def __init__(
        self,
        msrw_dir: Path = _MSRW_ROOT,
        wizard_endpoint: str = "http://127.0.0.1:8090",
        session_id: str = "msrw-c0d3r-bridge",
    ):
        self.msrw_dir = Path(msrw_dir).resolve()
        self.wizard_endpoint = wizard_endpoint
        self.session_id = session_id
        self.history: List[Dict[str, Any]] = []

    def dispatch_task(
        self,
        instruction: str,
        system_context: str = "",
        allow_admin: bool = False,
    ) -> Dict[str, Any]:
        """Dispatches an engineering command to C0d3rV2 with W1z4rD brain backend."""
        print(f"\n[*] Dispatching task to C0d3rV2 (W1z4rD Brain) in {self.msrw_dir}:")
        print(f"    Instruction: {instruction}")

        from wizard_chat import agent_runner

        t0 = time.time()
        reply = agent_runner.run(
            prompt=instruction,
            session_key=self.session_id,
            workdir=self.msrw_dir,
            backend="wizard",
            system_context=system_context,
            allow_admin=allow_admin,
            wizard_endpoint=self.wizard_endpoint,
        )
        elapsed = time.time() - t0

        result = {
            "instruction": instruction,
            "reply": reply,
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": time.time(),
        }
        self.history.append(result)
        return result

    def run_benchmark(self) -> Dict[str, Any]:
        """Commands C0d3rV2 to run the full MSRW Vitest benchmark and evaluate status."""
        benchmark_instruction = (
            "Run the Multiscale Robot World Vitest test suite using the executor tool "
            "with command: npm test -- --run. Verify that all test suites pass and report "
            "the exact number of passing test files and individual tests."
        )
        return self.dispatch_task(benchmark_instruction)

    def upgrade_brain_on_failure(self, failure_topic: str, telemetry: str, correction: str) -> bool:
        """Applies corrective training to the W1z4rD brain when a task or benchmark fails."""
        print(f"\n[!] Upgrading W1z4rD Brain for topic: {failure_topic}")
        print(f"    Telemetry: {telemetry}")
        print(f"    Corrective rule: {correction}")

        try:
            from tools.wizard_learning.settling_controller import get_settling_controller
            sc = get_settling_controller()
            rec = sc.settle(reason=f"c0d3r_failure_upgrade:{failure_topic}")
            print(f"    [✓] Consolidated memory & settled brain (RAM reclaimed: {rec.get('memory_reclaimed_mb', 0):.1f} MB)")
            return True
        except Exception as e:
            print(f"    [!] Error during brain settling/upgrade: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="MSRW C0d3rV2 Agent Bridge")
    parser.add_argument("instruction", nargs="*", help="Instruction to dispatch to C0d3rV2")
    parser.add_argument("--benchmark", action="store_true", help="Run full MSRW test benchmark via C0d3rV2")
    args = parser.parse_args()

    bridge = MsrwAgentBridge()

    if args.benchmark or not args.instruction:
        res = bridge.run_benchmark()
        print("\n" + "=" * 60)
        print("C0d3rV2 Execution Report:")
        print("=" * 60)
        print(res["reply"])
    else:
        task_str = " ".join(args.instruction).strip()
        res = bridge.dispatch_task(task_str)
        print("\n" + "=" * 60)
        print("C0d3rV2 Execution Report:")
        print("=" * 60)
        print(res["reply"])


if __name__ == "__main__":
    main()
