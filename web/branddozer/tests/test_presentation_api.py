"""Presentation endpoints: deck first, media on demand."""
from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.test import APIClient

from branddozer.models import BrandProject, ResearchPaper


PAPER_MD = """## Findings

Archival evidence constrains the claim. No successor program was documented.

- The first program ended in January.
"""


class PresentationApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        user = get_user_model().objects.create_user(
            username="deck-tester", password="pw-deck-123"
        )
        self.client.force_authenticate(user=user)
        project = BrandProject.objects.create(name="P", root_path="/tmp/p")
        self.paper = ResearchPaper.objects.create(
            project=project,
            title="A Paper About Something",
            research_question="Q?",
            abstract="An abstract.",
            content_markdown=PAPER_MD,
            status="validated",
        )
        self.url = f"/api/branddozer/research/papers/{self.paper.id}/presentation/"

    def test_deck_is_returned_without_generating_media(self):
        """A reader must be able to open a deck immediately."""
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        deck = response.json()
        self.assertGreater(deck["slide_count"], 0)
        self.assertFalse(deck["narrated"])

    def test_deck_carries_a_playable_timeline(self):
        deck = self.client.get(self.url).json()
        timeline = deck["timeline"]
        self.assertEqual(len(timeline), deck["slide_count"])
        # Timeline marks must be non-decreasing for a player to seek.
        times = [point["at_ms"] for point in timeline]
        self.assertEqual(times, sorted(times))

    def test_deck_offers_the_player_its_options(self):
        options = self.client.get(self.url).json()["options"]
        for key in ("transitions", "word_animations", "color_schemes", "color_ratios"):
            self.assertIn(key, options)
        self.assertIn("crossfade", options["transitions"])

    def test_every_slide_has_a_duration(self):
        deck = self.client.get(self.url).json()
        self.assertTrue(all(s["duration_ms"] > 0 for s in deck["slides"]))

    def test_unknown_paper_is_404(self):
        response = self.client.get(
            "/api/branddozer/research/papers/00000000-0000-0000-0000-000000000000"
            "/presentation/"
        )
        self.assertEqual(response.status_code, 404)

    def test_presentation_requires_authentication(self):
        response = APIClient().get(self.url)
        self.assertIn(response.status_code, (401, 403))

    def test_audio_404s_before_narration_is_generated(self):
        response = self.client.get(
            f"/api/branddozer/research/papers/{self.paper.id}/presentation/audio/0/"
        )
        self.assertEqual(response.status_code, 404)

    def test_score_404s_before_it_is_composed(self):
        response = self.client.get(
            f"/api/branddozer/research/papers/{self.paper.id}/presentation/score/"
        )
        self.assertEqual(response.status_code, 404)

    def test_media_endpoint_requires_authentication(self):
        response = APIClient().post(
            f"/api/branddozer/research/papers/{self.paper.id}/presentation/media/",
            {},
            format="json",
        )
        self.assertIn(response.status_code, (401, 403))
