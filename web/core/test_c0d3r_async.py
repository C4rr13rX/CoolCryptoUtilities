from __future__ import annotations

import json
import time
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.conf import settings
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import C0d3rWebRun, C0d3rWebSession
from .c0d3r_async import _is_conversational_acknowledgement


@override_settings(MIDDLEWARE=[item for item in settings.MIDDLEWARE if not item.startswith("whitenoise.")])
class C0d3rAsyncRunTests(TestCase):
    def test_conversational_acknowledgement_is_not_a_completed_answer(self) -> None:
        self.assertTrue(_is_conversational_acknowledgement(
            "Hello! How can I assist you with your engineering project today?"
        ))
        self.assertFalse(_is_conversational_acknowledgement(
            "The typecheck failed because Vec3.subtract is not implemented."
        ))

    def test_web_runner_has_no_deterministic_canned_social_answer(self) -> None:
        from tools.c0d3rV2.orchestrator import Orchestrator
        from tools.c0d3rV2.web_runner import _should_use_conversationalist_path

        self.assertEqual(Orchestrator._deterministic_direct_answer("hello"), "")
        self.assertTrue(_should_use_conversationalist_path("hello"))
        self.assertTrue(_should_use_conversationalist_path("How are you?"))
        self.assertFalse(_should_use_conversationalist_path("Fix the Django app"))

    def setUp(self) -> None:
        self.user = get_user_model().objects.create_user(username="async-user", password="test-pass")
        self.client.force_login(self.user)

    @patch("core.c0d3r_async.submit_run")
    def test_enqueue_acknowledges_without_waiting_for_model(self, submit_run) -> None:
        started = time.perf_counter()
        response = self.client.post(
            reverse("core:c0d3r-run"),
            data=json.dumps({"prompt": "Build a small class", "backend": "freeloader"}),
            content_type="application/json",
        )
        elapsed = time.perf_counter() - started
        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(payload["status"], "queued")
        self.assertTrue(payload["run_id"])
        self.assertLess(elapsed, 0.5)
        run = C0d3rWebRun.objects.get(id=payload["run_id"])
        self.assertEqual(run.backend, "freeloader")
        self.assertEqual(run.status, "queued")

    def test_status_endpoint_is_user_scoped(self) -> None:
        session = C0d3rWebSession.objects.create(user=self.user, title="test")
        run = C0d3rWebRun.objects.create(session=session, prompt="hello", status="completed", output="done")
        response = self.client.get(reverse("core:c0d3r-run-status", kwargs={"run_id": run.id}))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["output"], "done")

        other = get_user_model().objects.create_user(username="other-user")
        self.client.force_login(other)
        denied = self.client.get(reverse("core:c0d3r-run-status", kwargs={"run_id": run.id}))
        self.assertEqual(denied.status_code, 404)

    def test_running_task_can_be_stopped(self) -> None:
        session = C0d3rWebSession.objects.create(user=self.user, title="stop-test")
        run = C0d3rWebRun.objects.create(session=session, prompt="long task", status="running")
        response = self.client.post(reverse("core:c0d3r-run-status", kwargs={"run_id": run.id}))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["stopped"])
        run.refresh_from_db()
        self.assertEqual(run.status, "cancelled")

    @patch("core.c0d3r_async.submit_run")
    def test_default_backend_is_freeloader(self, submit_run) -> None:
        response = self.client.post(
            reverse("core:c0d3r-run"),
            data=json.dumps({"prompt": "Implement the task"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 202)
        run = C0d3rWebRun.objects.get(id=response.json()["run_id"])
        self.assertEqual(run.backend, "freeloader")
