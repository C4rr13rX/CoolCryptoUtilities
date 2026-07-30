"""Tests for the BrandDozer mode catalog.

These lock in the distinction the UI depends on: only the generic loop
actually re-sends its prompt every cycle, and a delivery run's prompt
seeds exactly one phase.
"""
from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.test import APIClient

from branddozer.modes import (
    DELIVERY_MODES,
    PROJECT_MODES,
    catalog,
    delivery_mode,
    project_mode,
)


class ModeCatalogTests(TestCase):
    def test_generic_loop_is_the_only_prompt_loop(self):
        """Workflow projects bypass the prompt loop, so only "" may claim it."""
        for mode in PROJECT_MODES:
            if mode["id"] == "":
                self.assertTrue(mode["uses_prompt_loop"])
                self.assertTrue(mode["supports_interjections"])
            else:
                self.assertFalse(
                    mode["uses_prompt_loop"],
                    f"{mode['id']} bypasses the prompt loop but claims to use it",
                )
                self.assertFalse(mode["supports_interjections"])

    def test_only_prompt_loop_says_every_cycle(self):
        """"every cycle" wording must not leak onto workflow projects."""
        for mode in PROJECT_MODES:
            if not mode["uses_prompt_loop"]:
                self.assertNotIn("every cycle", mode["prompt_label"].lower())

    def test_unknown_workflow_kind_does_not_claim_prompt_loop(self):
        mode = project_mode("some_future_workflow")
        self.assertFalse(mode["uses_prompt_loop"])
        self.assertFalse(mode["supports_interjections"])

    def test_delivery_prompt_seeds_exactly_one_phase(self):
        for mode in DELIVERY_MODES:
            seeded = [p for p in mode["phases"] if p["consumes_run_prompt"]]
            self.assertEqual(
                len(seeded), 1,
                f"{mode['id']} must have exactly one prompt-consuming phase",
            )
            # The seeded phase is the first one to run.
            self.assertTrue(mode["phases"][0]["consumes_run_prompt"])

    def test_research_exposes_its_six_role_subcycles(self):
        mode = delivery_mode("research")
        roles = [p["role"] for p in mode["phases"]]
        self.assertEqual(
            roles,
            [
                "research_planner",
                "literature_reviewer",
                "methods_reviewer",
                "research_writer",
                "citation_auditor",
                "peer_reviewer",
            ],
        )

    def test_research_omits_software_only_fields(self):
        """Smoke tests and team mode are meaningless for a paper."""
        fields = delivery_mode("research")["fields"]
        self.assertNotIn("smoke_test_cmd", fields)
        self.assertNotIn("team_mode", fields)

    def test_unknown_project_type_falls_back_to_software(self):
        self.assertEqual(delivery_mode("nonsense")["id"], "software")


class ModeCatalogApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        user = get_user_model().objects.create_user(
            username="modes-tester", password="pw-modes-123"
        )
        self.client.force_authenticate(user=user)

    def test_endpoint_returns_catalog(self):
        response = self.client.get("/api/branddozer/modes/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            {m["id"] for m in payload["project_modes"]},
            {m["id"] for m in catalog()["project_modes"]},
        )
        self.assertEqual(
            [m["id"] for m in payload["delivery_modes"]], ["software", "research"]
        )

    def test_endpoint_requires_authentication(self):
        response = APIClient().get("/api/branddozer/modes/")
        self.assertIn(response.status_code, (401, 403))
