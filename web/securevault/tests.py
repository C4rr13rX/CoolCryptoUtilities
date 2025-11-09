from __future__ import annotations

import importlib
import os
import tempfile

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient

from services import secure_settings as vault


class SecureVaultTestMixin:
    def setUp(self):
        super().setUp()
        self._vault_tmp = tempfile.TemporaryDirectory()
        os.environ["SECURE_VAULT_KEY_DIR"] = self._vault_tmp.name
        importlib.reload(vault)

    def tearDown(self):
        super().tearDown()
        self._vault_tmp.cleanup()
        os.environ.pop("SECURE_VAULT_KEY_DIR", None)
        importlib.reload(vault)


class SecureVaultCryptoTests(SecureVaultTestMixin, TestCase):
    def test_encrypt_decrypt_roundtrip(self):
        payload = vault.encrypt_secret("super-secret-value")
        recovered = vault.decrypt_secret(payload["encapsulated_key"], payload["ciphertext"], payload["nonce"])
        self.assertEqual(recovered, "super-secret-value")


class SecureVaultApiTests(SecureVaultTestMixin, TestCase):
    def setUp(self):
        super().setUp()
        self.client = APIClient()
        User = get_user_model()
        self.user = User.objects.create_user(username="vault-user", password="test-pass")

    def test_requires_authentication(self):
        url = reverse("securevault:settings-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 403)

    def test_placeholder_resolution_for_authenticated_user(self):
        url = reverse("securevault:settings-list")
        self.client.force_authenticate(self.user)
        secret_payload = {
            "name": "API_KEY",
            "category": "default",
            "is_secret": True,
            "value": "abc123",
        }
        plain_payload = {
            "name": "API_URL",
            "category": "default",
            "is_secret": False,
            "value": "https://example.com/${API_KEY}",
        }
        self.client.post(url, secret_payload, format="json")
        self.client.post(url, plain_payload, format="json")

        resolved = vault.get_settings_for_user(self.user)
        self.assertEqual(resolved["API_KEY"], "abc123")
        self.assertEqual(resolved["API_URL"], "https://example.com/abc123")
