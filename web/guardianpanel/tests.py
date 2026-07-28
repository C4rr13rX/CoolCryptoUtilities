from __future__ import annotations

import json
import tempfile
import urllib.request
from pathlib import Path

from django.test import SimpleTestCase

from monitoring_guardian.fallback_server import GuardianFallbackServer


class GuardianFallbackServerTests(SimpleTestCase):
    def test_serves_recovery_state_without_django(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "state.json"
            log = root / "recovery.log"
            state.write_text(json.dumps({"components": {"django": {"status": "down", "attempts": 2, "detail": "connection refused"}}}), encoding="utf-8")
            log.write_text("attempting repair\nAPI_KEY=must-not-leak", encoding="utf-8")
            server = GuardianFallbackServer(state, log, port=0)
            try:
                self.assertTrue(server.start())
                with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/", timeout=3) as response:
                    body = response.read().decode("utf-8")
                self.assertIn("Guardian is restoring", body)
                self.assertIn("connection refused", body)
                self.assertNotIn("must-not-leak", body)
                self.assertIn("[redacted]", body)
            finally:
                server.stop()
