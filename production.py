from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional

from trading.pipeline import TrainingPipeline
from trading.selector import GhostTradingSupervisor
from services.background_workers import TokenDownloadSupervisor


class ProductionManager:
    def __init__(self) -> None:
        self.pipeline = TrainingPipeline()
        self.supervisor = GhostTradingSupervisor(pipeline=self.pipeline)
        self._trainer: Optional[threading.Thread] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop = threading.Event()
        self._download_supervisor: Optional[TokenDownloadSupervisor] = TokenDownloadSupervisor(db=self.pipeline.db)
        self._active_flag_key = "production_manager_active"

    def start(self) -> None:
        if self.is_running:
            print("[production] manager already running.")
            self._set_active_flag(True)
            return
        self._stop.clear()
        self.supervisor.build()
        try:
            self._trainer = threading.Thread(target=self._training_loop, daemon=True)
            self._trainer.start()
            self._loop_thread = threading.Thread(target=self._run_supervisor_loop, daemon=True)
            self._loop_thread.start()
            if self._download_supervisor:
                self._download_supervisor.start()
            self._set_active_flag(True)
            print("[production] manager started.")
        except Exception:
            self._set_active_flag(False)
            raise

    def stop(self, timeout: float = 15.0) -> None:
        if not self.is_running:
            print("[production] manager is not running.")
            return
        self._stop.set()
        if self._loop and self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self.supervisor.stop(), self._loop)
            try:
                fut.result(timeout=timeout)
            except Exception as exc:
                print(f"[production] supervisor stop error: {exc}")
        if self._loop_thread:
            self._loop_thread.join(timeout=timeout)
        if self._trainer:
            self._trainer.join(timeout=timeout)
        if self._download_supervisor:
            try:
                self._download_supervisor.stop()
            except Exception as exc:
                print(f"[production] download supervisor stop error: {exc}")
        self._loop = None
        self._loop_thread = None
        self._trainer = None
        self._set_active_flag(False)
        print("[production] manager stopped.")

    @property
    def is_running(self) -> bool:
        return self._trainer is not None and self._trainer.is_alive()

    def _run_supervisor_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.supervisor.start())
        except Exception as exc:
            print(f"[production] supervisor loop error: {exc}")
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self._loop = None

    def _training_loop(self) -> None:
        while not self._stop.is_set():
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
            if self._stop.wait(delay):
                break

    def _set_active_flag(self, active: bool) -> None:
        try:
            if active:
                self.pipeline.db.set_control_flag(self._active_flag_key, "1")
            else:
                self.pipeline.db.clear_control_flag(self._active_flag_key)
        except Exception as exc:
            print(f"[production] unable to update active flag: {exc}")


if __name__ == "__main__":
    manager = ProductionManager()
    try:
        manager.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()
