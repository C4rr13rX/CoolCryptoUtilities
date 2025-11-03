from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional

from trading.pipeline import TrainingPipeline
from trading.selector import GhostTradingSupervisor


class ProductionManager:
    def __init__(self) -> None:
        self.pipeline = TrainingPipeline()
        self.supervisor = GhostTradingSupervisor(pipeline=self.pipeline)
        self._trainer: Optional[threading.Thread] = None

    def start(self) -> None:
        self.supervisor.build()
        self._trainer = threading.Thread(target=self._training_loop, daemon=True)
        self._trainer.start()
        try:
            asyncio.run(self.supervisor.start())
        except KeyboardInterrupt:
            print("[production] received interrupt, shutting down supervisor.")

    def _training_loop(self) -> None:
        while True:
            try:
                result = self.pipeline.train_candidate()
                status = result.get("status") if isinstance(result, dict) else None
                if status == "skipped":
                    delay = 60.0
                elif status == "paused":
                    delay = 300.0
                else:
                    delay = 30.0
            except Exception as exc:
                print(f"[production] training loop error: {exc}")
                delay = 60.0
            time.sleep(delay)


if __name__ == "__main__":
    manager = ProductionManager()
    manager.start()
