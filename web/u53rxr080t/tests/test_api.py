from __future__ import annotations

from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from services import guardian_status
from u53rxr080t.models import AgentSession, Finding, Task


class U53ApiTests(APITestCase):
    def setUp(self) -> None:
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="tester", password="test123")
        self.client.force_authenticate(self.user)

    def test_heartbeat_requires_auth(self):
        self.client.logout()
        resp = self.client.post("/api/u53rxr080t/heartbeat/", {"name": "anon"})
        self.assertIn(resp.status_code, {status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN})

    def test_heartbeat_creates_and_updates_agent(self):
        resp = self.client.post(
            "/api/u53rxr080t/heartbeat/",
            {"name": "edge-plugin", "kind": "browser", "platform": "windows", "status": "idle"},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        agent_id = resp.data.get("id")
        self.assertTrue(agent_id)
        agent = AgentSession.objects.get(id=agent_id)
        self.assertEqual(agent.name, "edge-plugin")
        # Update status and ensure same agent is reused
        resp2 = self.client.post(
            "/api/u53rxr080t/heartbeat/",
            {"id": agent_id, "status": "in_progress"},
            format="json",
        )
        self.assertEqual(resp2.status_code, status.HTTP_200_OK)
        agent.refresh_from_db()
        self.assertEqual(agent.status, "in_progress")

    def test_tasks_crud_and_filters(self):
        self.client.post("/api/u53rxr080t/tasks/", {"title": "Task 1", "stage": "overview"}, format="json")
        self.client.post(
            "/api/u53rxr080t/tasks/",
            {"title": "Task 2", "stage": "pipeline", "status": "done"},
            format="json",
        )
        resp_all = self.client.get("/api/u53rxr080t/tasks/")
        self.assertEqual(resp_all.status_code, status.HTTP_200_OK)
        self.assertEqual(len(resp_all.data["tasks"]), 2)
        resp_pending = self.client.get("/api/u53rxr080t/tasks/", {"status": "pending"})
        self.assertEqual(len(resp_pending.data["tasks"]), 1)
        self.assertEqual(resp_pending.data["tasks"][0]["title"], "Task 1")

    def test_task_claim_and_update(self):
        hb = self.client.post("/api/u53rxr080t/heartbeat/", {"name": "agent"}, format="json")
        agent_id = hb.data["id"]
        # two pending tasks
        t1 = self.client.post("/api/u53rxr080t/tasks/", {"title": "T1"}, format="json").data["task"]
        self.client.post("/api/u53rxr080t/tasks/", {"title": "T2"}, format="json")

        claim = self.client.post("/api/u53rxr080t/tasks/next/", {"agent_id": agent_id}, format="json")
        self.assertEqual(claim.status_code, status.HTTP_200_OK)
        claimed = claim.data["task"]
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["title"], "T1")
        task = Task.objects.get(id=t1)
        self.assertEqual(str(task.assigned_to_id), agent_id)
        self.assertEqual(task.status, "in_progress")

        # Update status to done
        upd = self.client.post(f"/api/u53rxr080t/tasks/{t1}/", {"status": "done"}, format="json")
        self.assertEqual(upd.status_code, status.HTTP_200_OK)
        task.refresh_from_db()
        self.assertEqual(task.status, "done")

    def test_findings_create_and_list(self):
        hb = self.client.post("/api/u53rxr080t/heartbeat/", {"name": "agent"}, format="json")
        agent_id = hb.data["id"]
        # Create a couple findings
        for i in range(3):
            resp = self.client.post(
                "/api/u53rxr080t/findings/",
                {
                    "session": agent_id,
                    "title": f"finding {i}",
                    "summary": "something observed",
                    "severity": "warn" if i == 0 else "info",
                    "context": {"step": i},
                },
                format="json",
            )
            self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Finding.objects.count(), 3)
        resp_list = self.client.get("/api/u53rxr080t/findings/", {"limit": 2})
        self.assertEqual(resp_list.status_code, status.HTTP_200_OK)
        self.assertEqual(len(resp_list.data["findings"]), 2)
        self.assertTrue(all("session" in f for f in resp_list.data["findings"]))

    def test_agents_listing(self):
        now = timezone.now()
        AgentSession.objects.create(name="a1", status="idle", last_seen=now)
        AgentSession.objects.create(name="a2", status="in_progress", last_seen=now)
        resp = self.client.get("/api/u53rxr080t/agents/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(len(resp.data["agents"]), 2)
        statuses = {a["status"] for a in resp.data["agents"]}
        self.assertIn("idle", statuses)
        self.assertIn("in_progress", statuses)

    def test_guardian_queue_enqueue_and_list(self):
        # isolate guardian status file to a temp path
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from importlib import reload

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "status.json"
            guardian_status.STATUS_DIR = tmp_path.parent
            guardian_status.STATUS_FILE = tmp_path
            guardian_status.RUN_REQUEST_PATH = guardian_status.STATUS_DIR / "request_run"
            reload(guardian_status)

            # fresh state
            guardian_status._write_state(guardian_status._default_state())  # type: ignore

            resp = self.client.post(
                "/api/u53rxr080t/queue/",
                {"session": "agent-1", "title": "Fix nav", "summary": "nav broken", "severity": "warn"},
                format="json",
            )
            self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
            ticket = resp.data.get("ticket")
            self.assertTrue(ticket)
            listed = self.client.get("/api/u53rxr080t/queue/")
            self.assertEqual(listed.status_code, status.HTTP_200_OK)
            queue = listed.data.get("queue") or []
            self.assertEqual(len(queue), 1)
            self.assertEqual(queue[0]["ticket"], ticket)
