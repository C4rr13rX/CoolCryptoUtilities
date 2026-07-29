from __future__ import annotations

from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from django.test import TestCase
from rest_framework.test import APIRequestFactory, force_authenticate

from securevault.models import SecureSetting
from services.secure_settings import decrypt_secret
from tools.ai_backend_mode import deactivate_freeloader_mode_for_tests, freeloader_mode_active
from tools.c0d3rV2.plugins.agent_the_freeloader.models import load_catalog
from .views import (
    ModelControlConfigView,
    ModelControlView,
    ModelCredentialView,
    WizardBrainListView,
    WizardBrainSelectionView,
)


class ModelControlApiTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="model-admin", password="test-password"
        )
        self.factory = APIRequestFactory()

    def tearDown(self):
        deactivate_freeloader_mode_for_tests()

    def test_requires_authentication(self):
        request = self.factory.get("/api/model-control/")
        request.user = AnonymousUser()
        response = ModelControlView.as_view()(request)
        self.assertIn(response.status_code, {401, 403})

    def test_lists_backends_credentials_and_free_models(self):
        request = self.factory.get("/api/model-control/")
        force_authenticate(request, user=self.user)
        response = ModelControlView.as_view()(request)
        self.assertEqual(response.status_code, 200)
        payload = response.data
        self.assertTrue({"wizard", "bedrock", "freeloader"}.issubset(
            {item["id"] for item in payload["backends"]}
        ))
        self.assertTrue(any(item["name"] == "GITHUB_TOKEN" for item in payload["credentials"]))
        self.assertTrue(any(provider["models"] for provider in payload["providers"]))

    def test_persists_backend_and_atf_filter(self):
        model_id = load_catalog()[0].model_id
        request = self.factory.post(
            "/api/model-control/config/",
            data={"backend": "freeloader", "model": "", "atf_models": [model_id]},
            format="json",
        )
        force_authenticate(request, user=self.user)
        response = ModelControlConfigView.as_view()(request)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(freeloader_mode_active())
        saved = {
            item.name: item.value_plain
            for item in SecureSetting.objects.filter(user=self.user, category="ai")
        }
        self.assertEqual(saved["C0D3R_BACKEND"], "freeloader")
        self.assertEqual(saved["AGENT_FREELOADER_MODELS"], model_id)

        request = self.factory.post(
            "/api/model-control/config/",
            data={"backend": "wizard", "model": "", "atf_models": []},
            format="json",
        )
        force_authenticate(request, user=self.user)
        response = ModelControlConfigView.as_view()(request)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(freeloader_mode_active())

    def test_encrypts_and_removes_credentials(self):
        request = self.factory.post(
            "/api/model-control/credentials/GITHUB_TOKEN/",
            data={"value": "github-test-token"},
            format="json",
        )
        force_authenticate(request, user=self.user)
        response = ModelCredentialView.as_view()(request, name="GITHUB_TOKEN")
        self.assertEqual(response.status_code, 200)
        setting = SecureSetting.objects.get(user=self.user, category="ai", name="GITHUB_TOKEN")
        self.assertTrue(setting.is_secret)
        self.assertIsNone(setting.value_plain)
        self.assertEqual(
            decrypt_secret(setting.encapsulated_key, setting.ciphertext, setting.nonce),
            "github-test-token",
        )

        request = self.factory.delete("/api/model-control/credentials/GITHUB_TOKEN/")
        force_authenticate(request, user=self.user)
        response = ModelCredentialView.as_view()(request, name="GITHUB_TOKEN")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(
            SecureSetting.objects.filter(user=self.user, category="ai", name="GITHUB_TOKEN").exists()
        )

    def test_wizard_brains_have_independent_chat_and_operations_selections(self):
        create = self.factory.post(
            "/api/model-control/wizard-brains/",
            data={
                "name": "Programming brain",
                "endpoint": "http://127.0.0.1:18095/chat",
                "chat_path": "/chat",
            },
            format="json",
        )
        force_authenticate(create, user=self.user)
        response = WizardBrainListView.as_view()(create)
        self.assertEqual(response.status_code, 201)
        brain = response.data["brain"]
        self.assertEqual(brain["endpoint"], "http://127.0.0.1:18095")

        select = self.factory.post(
            "/api/model-control/wizard-brains/selection/",
            data={"purpose": "operations", "brain_id": brain["id"]},
            format="json",
        )
        force_authenticate(select, user=self.user)
        self.assertEqual(WizardBrainSelectionView.as_view()(select).status_code, 200)

        listing = self.factory.get("/api/model-control/wizard-brains/")
        force_authenticate(listing, user=self.user)
        payload = WizardBrainListView.as_view()(listing).data
        self.assertEqual(payload["selected"]["operations"], brain["id"])
        self.assertEqual(payload["selected"]["chat"], "environment-default")

    def test_rejects_invalid_wizard_brain_address(self):
        request = self.factory.post(
            "/api/model-control/wizard-brains/",
            data={"name": "Invalid", "endpoint": "file:///tmp/brain", "chat_path": "/chat"},
            format="json",
        )
        force_authenticate(request, user=self.user)
        response = WizardBrainListView.as_view()(request)
        self.assertEqual(response.status_code, 400)
