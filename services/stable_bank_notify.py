"""
Stable Bank Threshold Notification Service

Checks whether the pipeline's stable bank has crossed a user-configured
USD threshold and fires a single notification via the AWS API Gateway
endpoint.  Remembers the last notified value so it only alerts once per
threshold crossing (not on every cycle).

Configuration (read from SecureVault / environment):
  STABLE_BANK_NOTIFY_ENDPOINT  – API Gateway URL  (required to enable)
  STABLE_BANK_NOTIFY_API_KEY   – x-api-key header  (required)
  STABLE_BANK_NOTIFY_EMAIL     – recipient address  (required)
  STABLE_BANK_NOTIFY_SENDER    – sender address     (optional, Lambda has default)
  STABLE_BANK_THRESHOLD_USD    – USD threshold       (default: 100)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_STATE_FILE = Path("runtime/notify_state.json")

# Minimum seconds between notifications (prevent spam on rapid cycles).
_COOLDOWN_SEC = int(os.getenv("STABLE_BANK_NOTIFY_COOLDOWN_SEC", "3600"))


class StableBankNotifier:
    """Stateful notifier — call ``check()`` each production cycle."""

    def __init__(self) -> None:
        self._last_notified_usd: float = 0.0
        self._last_notified_ts: float = 0.0
        self._lock = threading.Lock()
        self._load_state()

    # ── public API ───────────────────────────────────────────────────

    def check(self, stable_bank_usd: float) -> bool:
        """Return True if a notification was sent."""
        endpoint = os.environ.get("STABLE_BANK_NOTIFY_ENDPOINT", "").strip()
        api_key = os.environ.get("STABLE_BANK_NOTIFY_API_KEY", "").strip()
        recipient = os.environ.get("STABLE_BANK_NOTIFY_EMAIL", "").strip()
        sender = os.environ.get("STABLE_BANK_NOTIFY_SENDER", "").strip()

        if not endpoint or not api_key or not recipient:
            return False  # not configured — silently skip

        try:
            threshold = float(os.environ.get("STABLE_BANK_THRESHOLD_USD", "100"))
        except (TypeError, ValueError):
            threshold = 100.0

        if threshold <= 0:
            return False

        with self._lock:
            if stable_bank_usd < threshold:
                return False

            # Already notified for this crossing?
            if self._last_notified_usd >= threshold:
                return False

            # Cooldown guard
            now = time.time()
            if now - self._last_notified_ts < _COOLDOWN_SEC:
                return False

            sent = self._send(
                endpoint=endpoint,
                api_key=api_key,
                recipient=recipient,
                sender=sender,
                stable_bank_usd=stable_bank_usd,
                threshold_usd=threshold,
            )

            if sent:
                self._last_notified_usd = stable_bank_usd
                self._last_notified_ts = now
                self._persist_state()

            return sent

    def reset(self) -> None:
        """Reset notification state (e.g. when threshold is changed)."""
        with self._lock:
            self._last_notified_usd = 0.0
            self._last_notified_ts = 0.0
            self._persist_state()

    # ── internals ────────────────────────────────────────────────────

    @staticmethod
    def _send(
        *,
        endpoint: str,
        api_key: str,
        recipient: str,
        sender: str,
        stable_bank_usd: float,
        threshold_usd: float,
    ) -> bool:
        try:
            import urllib.request

            payload = json.dumps({
                "recipient_email": recipient,
                "sender_email": sender,
                "stable_bank_usd": round(stable_bank_usd, 2),
                "threshold_usd": round(threshold_usd, 2),
            }).encode("utf-8")

            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                status = resp.getcode()
                body = resp.read().decode("utf-8", errors="replace")

            if 200 <= status < 300:
                logger.info(
                    "Stable bank notification sent: $%.2f (threshold $%.2f) → %s",
                    stable_bank_usd,
                    threshold_usd,
                    recipient,
                )
                return True

            logger.warning(
                "Stable bank notification failed (HTTP %d): %s",
                status,
                body[:200],
            )
            return False

        except Exception as exc:
            logger.warning("Stable bank notification error: %s", exc)
            return False

    def _load_state(self) -> None:
        try:
            if _STATE_FILE.exists():
                data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
                self._last_notified_usd = float(data.get("last_notified_usd", 0.0))
                self._last_notified_ts = float(data.get("last_notified_ts", 0.0))
        except Exception:
            pass

    def _persist_state(self) -> None:
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _STATE_FILE.write_text(
                json.dumps({
                    "last_notified_usd": self._last_notified_usd,
                    "last_notified_ts": self._last_notified_ts,
                }),
                encoding="utf-8",
            )
        except Exception:
            pass


# Module-level singleton — import and call ``notifier.check(usd)``.
notifier = StableBankNotifier()
