from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.test import APIClient

from branddozer.models import (
    BacklogItem,
    BrandProject,
    DeliveryRun,
    DeliverySession,
    ResearchClaim,
    ResearchPaper,
    ResearchPaperRevision,
    ResearchSource,
)
from branddozer.research import (
    ResearchPolicy,
    ResearchWorkflow,
    _extract_json,
    _verify_source,
    validate_paper_payload,
)


def _paper_markdown(citation_keys: list[str], words: int = 700) -> str:
    sections = [
        "Abstract",
        "Keywords",
        "Introduction",
        "Methodology",
        "Literature Review",
        "Findings",
        "Discussion",
        "Limitations",
        "Conclusion",
        "References",
    ]
    citations = " ".join(f"Evidence [@{key}]." for key in citation_keys)
    body = " ".join(["qualified archival synthesis"] * words)
    return "\n\n".join(f"## {section}\n\n{citations} {body}" for section in sections)


class ResearchValidationTests(TestCase):
    def setUp(self) -> None:
        self.policy = ResearchPolicy(
            min_words=500,
            min_sources=3,
            min_verified_sources=3,
            min_high_authority_sources=2,
            min_source_domains=2,
            max_revision_rounds=2,
            max_parallel_agents=2,
        )
        self.sources = [
            {
                "citation_key": f"source{i}",
                "url": f"https://authority{i % 2}.edu/paper/{i}",
                "content_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                "authority_tier": 3,
                "verification_status": "verified",
            }
            for i in range(3)
        ]
        self.claims = [
            {
                "claim_text": f"Qualified claim {i}",
                "source_keys": [f"source{i}"],
                "verification_status": "qualified",
            }
            for i in range(3)
        ]

    def test_complete_supported_paper_passes(self) -> None:
        result = validate_paper_payload(
            markdown=_paper_markdown(["source0", "source1", "source2"]),
            sources=self.sources,
            claims=self.claims,
            policy=self.policy,
            peer_review={"recommendation": "accept", "blocking_issues": []},
        )
        self.assertTrue(result["passed"], result)
        self.assertTrue(all(result["checks"].values()))

    def test_unknown_citation_and_unsupported_claim_block_validation(self) -> None:
        claims = [
            *self.claims,
            {
                "claim_text": "Invented result",
                "source_keys": ["missing"],
                "verification_status": "supported",
            },
        ]
        result = validate_paper_payload(
            markdown=_paper_markdown(["source0", "source1", "invented"]),
            sources=self.sources,
            claims=claims,
            policy=self.policy,
            peer_review={"recommendation": "accept", "blocking_issues": []},
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["unknown_citations"], ["invented"])
        self.assertFalse(result["checks"]["claims_supported"])

    def test_peer_review_blocker_forces_rewrite(self) -> None:
        result = validate_paper_payload(
            markdown=_paper_markdown(["source0", "source1", "source2"]),
            sources=self.sources,
            claims=self.claims,
            policy=self.policy,
            peer_review={
                "recommendation": "major_revision",
                "blocking_issues": ["Causal conclusion exceeds archival evidence"],
            },
        )
        self.assertFalse(result["checks"]["peer_review"])
        self.assertIn("Causal conclusion", result["peer_review_blockers"][0])

    def test_c0d3r_multibranch_output_selects_role_complete_payload(self) -> None:
        output = "\n\n".join(
            [
                '{"action":"answer","output":{"title":"partial"}}',
                (
                    '{"action":"answer","output":'
                    '{"title":"complete","research_question":"q","keywords":["k"],'
                    '"scope":"bounded","search_strategy":{"queries":[]},'
                    '"work_packages":[{"title":"chronology"}]}}'
                ),
                '{"action":"complete","output":"done"}',
            ]
        )
        selected = _extract_json(
            output,
            expected_keys={
                "title", "research_question", "keywords", "scope",
                "search_strategy", "work_packages",
            },
        )
        self.assertEqual(selected["title"], "complete")
        self.assertEqual(selected["work_packages"][0]["title"], "chronology")

    def test_source_is_verified_only_when_quoted_passage_was_fetched(self) -> None:
        class Harvester:
            def crawl(self, seeds, *, query, config):
                return {
                    "stored": [
                        {
                            "url": seeds[0],
                            "title": "Authoritative source",
                            "sha256": "c" * 64,
                        }
                    ],
                    "errors": [],
                }

            def document(self, url):
                return {
                    "url": url,
                    "content": (
                        "The bounded controller remained stable throughout the "
                        "declared operating regime under the stated assumptions."
                    ),
                }

        verified = _verify_source(
            Harvester(),
            {
                "url": "https://example.edu/study",
                "verified_passage": (
                    "The bounded controller remained stable throughout the "
                    "declared operating regime"
                ),
            },
            "bounded controller",
        )
        rejected = _verify_source(
            Harvester(),
            {
                "url": "https://example.edu/study",
                "verified_passage": (
                    "This purported result does not occur anywhere in the source."
                ),
            },
            "bounded controller",
        )
        self.assertEqual(verified["verification_status"], "verified")
        self.assertEqual(rejected["verification_status"], "rejected")


class ResearchPaperApiTests(TestCase):
    def setUp(self) -> None:
        self.user = get_user_model().objects.create_user(
            username="researcher", password="secret"
        )
        self.client = APIClient()
        self.client.force_authenticate(self.user)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = BrandProject.objects.create(
            name="Climate Control Research",
            root_path=self.temp.name,
            default_prompt="Research adaptive control",
        )
        self.paper = ResearchPaper.objects.create(
            project=self.project,
            title="Adaptive Control in Nonlinear Thermal Systems",
            research_question="What archival evidence supports adaptive control?",
            abstract="A qualified synthesis of nonlinear thermal control evidence.",
            content_markdown=_paper_markdown(["thermal2024"]),
            keywords=["adaptive control", "thermal systems"],
            status="validated",
            validation_report={"passed": True, "checks": {"peer_review": True}},
            word_count=7200,
            content_sha256="a" * 64,
        )
        ResearchSource.objects.create(
            paper=self.paper,
            citation_key="thermal2024",
            title="Thermal Control Review",
            url="https://example.edu/thermal",
            content_sha256="b" * 64,
            authority_tier=3,
            verification_status="verified",
        )
        ResearchClaim.objects.create(
            paper=self.paper,
            claim_text="Adaptive controllers can be qualified for bounded regimes.",
            source_keys=["thermal2024"],
            verification_status="qualified",
        )
        ResearchPaperRevision.objects.create(
            paper=self.paper,
            version=1,
            content_markdown=self.paper.content_markdown,
            validation_report=self.paper.validation_report,
        )

    def test_search_detail_and_download(self) -> None:
        response = self.client.get(
            "/api/branddozer/research/papers/", {"q": "nonlinear thermal"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["papers"][0]["id"], str(self.paper.id))

        detail = self.client.get(
            f"/api/branddozer/research/papers/{self.paper.id}/"
        )
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(len(detail.data["paper"]["sources"]), 1)
        self.assertEqual(len(detail.data["paper"]["claims"]), 1)
        self.assertEqual(len(detail.data["paper"]["revisions"]), 1)

        download = self.client.get(
            f"/api/branddozer/research/papers/{self.paper.id}/download/",
            {"kind": "markdown"},
        )
        self.assertEqual(download.status_code, 200)
        self.assertIn("attachment;", download["Content-Disposition"])
        self.assertIn(b"## Methodology", download.content)

    def test_delivery_api_records_research_configuration_before_queueing(self) -> None:
        response = self.client.post(
            "/api/branddozer/delivery/runs/",
            {
                "project_id": str(self.project.id),
                "prompt": "Research verifiable adaptive-control evidence.",
                "project_type": "research",
                "research_mode": True,
                "research_config": {
                    "min_words": 6500,
                    "min_verified_sources": 14,
                    "citation_style": "ieee",
                },
                "team_mode": "full",
                "agent_provider": "c0d3r",
                "model_provider": "freeloader",
            },
            format="json",
        )
        self.assertEqual(response.status_code, 201, response.data)
        context = response.data["run"]["context"]
        self.assertTrue(context["research_mode"])
        self.assertEqual(context["research_config"]["min_words"], 6500)
        self.assertEqual(context["agent_provider"], "c0d3r")
        self.assertEqual(context["model_provider"], "freeloader")
        self.assertIn("job_id", context)

    @patch("branddozer.research.run_delivery_turn_detailed")
    def test_c0d3r_agent_uses_separate_freeloader_model_backend(self, routed) -> None:
        routed.return_value = {
            "output": (
                '{"title":"bounded","research_question":"q","keywords":["k"],'
                '"scope":"s","search_strategy":{"queries":[]},'
                '"work_packages":[{"title":"chronology"}]}'
            ),
            "route_history": [[{"outcome": "selected", "provider": "free"}]],
            "models": [{"provider": "free", "model": "test-model"}],
            "turn_model_calls": 1,
            "tool_events": [],
        }
        run = DeliveryRun.objects.create(
            project=self.project,
            prompt="Research a bounded question.",
            context={
                "research_mode": True,
                "agent_provider": "c0d3r",
                "model_provider": "freeloader",
            },
        )
        workflow = ResearchWorkflow(run, Path(self.temp.name))

        result = workflow._call(
            "research_planner",
            "route contract",
            "Return bounded JSON.",
            system="Return JSON only.",
        )

        self.assertEqual(result["title"], "bounded")
        self.assertEqual(routed.call_args.kwargs["backend"], "freeloader")
        session = DeliverySession.objects.get(run=run)
        self.assertEqual(session.meta["agent_provider"], "c0d3r")
        self.assertEqual(session.meta["model_provider"], "freeloader")


class ResearchWorkflowPersistenceTests(TestCase):
    def test_checkpointed_plan_reuses_existing_scrum_backlog(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        project = BrandProject.objects.create(
            name="Checkpoint Research",
            root_path=temp.name,
            default_prompt="Resume evidence",
        )
        plan = {
            "title": "Checkpointed",
            "research_question": "Can the run resume?",
            "keywords": ["checkpoint"],
            "scope": "bounded",
            "search_strategy": {},
            "work_packages": [{"title": "Existing evidence", "query": "q"}],
        }
        run = DeliveryRun.objects.create(
            project=project,
            prompt="Resume evidence",
            context={"research_mode": True, "research_plan": plan},
        )
        existing = BacklogItem.objects.create(
            project=project,
            run=run,
            source="research",
            title="Existing evidence",
            priority=1,
        )
        workflow = ResearchWorkflow(run, Path(temp.name))

        with patch.object(workflow, "_plan") as planner:
            items = workflow._create_scrum(run.context["research_plan"])

        planner.assert_not_called()
        self.assertEqual([item.id for item in items], [existing.id])
        self.assertEqual(BacklogItem.objects.filter(run=run).count(), 1)

    def test_failed_evidence_package_is_quarantined_while_success_continues(
        self
    ) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        project = BrandProject.objects.create(
            name="Quarantine Research",
            root_path=temp.name,
            default_prompt="Research with partial provider failure",
        )
        run = DeliveryRun.objects.create(
            project=project,
            prompt="Research with partial provider failure",
            context={
                "research_mode": True,
                "research_config": {"max_parallel_agents": 1},
            },
        )
        workflow = ResearchWorkflow(run, Path(temp.name))
        good = BacklogItem.objects.create(
            project=project, run=run, title="Good package", meta={}
        )
        failed = BacklogItem.objects.create(
            project=project, run=run, title="Failed package", meta={}
        )

        def review(item, plan):
            if item.id == failed.id:
                raise RuntimeError("provider exhausted")
            return {"findings": ["bounded"], "sources": [], "claims": []}

        with patch.object(workflow, "_review_package", side_effect=review):
            evidence = workflow._collect_evidence([good, failed], {})

        run.refresh_from_db()
        failed.refresh_from_db()
        self.assertEqual(len(evidence), 1)
        self.assertEqual(failed.status, "blocked")
        self.assertEqual(run.context["research_quarantine_count"], 1)
        self.assertEqual(
            run.context["research_quarantine"][0]["title"], "Failed package"
        )

    def test_failed_draft_is_rewritten_and_only_validated_revision_finishes(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        project = BrandProject.objects.create(
            name="Research Run",
            root_path=temp.name,
            default_prompt="Research resilient systems",
        )
        run = DeliveryRun.objects.create(
            project=project,
            prompt="Synthesize archival evidence for resilient engineered systems.",
            acceptance_required=False,
            context={
                "research_mode": True,
                "research_config": {
                    "min_words": 500,
                    "min_sources": 3,
                    "min_verified_sources": 3,
                    "min_high_authority_sources": 2,
                    "min_source_domains": 2,
                    "max_revision_rounds": 2,
                },
            },
        )
        workflow = ResearchWorkflow(run, Path(temp.name))
        plan = {
            "title": "Resilient Engineered Systems",
            "research_question": "Which archival findings support resilience?",
            "keywords": ["resilience", "systems engineering"],
            "target_journal": "Systems Engineering",
            "work_packages": [
                {"title": f"Evidence stream {index}", "query": f"stream {index}"}
                for index in range(3)
            ],
        }
        sources = [
            {
                "citation_key": f"source{i}",
                "title": f"Source {i}",
                "url": f"https://archive{i % 2}.edu/{i}",
                "content_sha256": str(i + 1) * 64,
                "authority_tier": 3,
                "verification_status": "verified",
                "peer_reviewed": True,
            }
            for i in range(3)
        ]
        evidence = [{"sources": sources, "findings": ["bounded finding"]}]
        good_claims = [
            {
                "section": "Findings",
                "claim_text": f"Qualified finding {i}",
                "source_keys": [f"source{i}"],
                "verification_status": "qualified",
                "rationale": "Bounded by the cited archival evidence.",
            }
            for i in range(3)
        ]
        calls = {"write": 0}

        def write_candidate(*args, **kwargs):
            calls["write"] += 1
            keys = ["source0", "invented"] if calls["write"] == 1 else [
                "source0", "source1", "source2"
            ]
            return {
                "title": plan["title"],
                "abstract": "Qualified archival synthesis.",
                "keywords": plan["keywords"],
                "markdown": _paper_markdown(keys),
                "claims": good_claims,
                "change_summary": f"revision {calls['write']}",
            }

        def audit(candidate, supplied_sources, methods):
            if calls["write"] == 1:
                return good_claims, {
                    "recommendation": "major_revision",
                    "blocking_issues": ["Unknown citation"],
                }
            return good_claims, {
                "recommendation": "accept",
                "blocking_issues": [],
            }

        with (
            patch.object(workflow, "_plan", return_value=plan),
            patch.object(workflow, "_collect_evidence", return_value=evidence),
            patch.object(workflow, "_verify_sources", return_value=sources),
            patch.object(workflow, "_methods_review", return_value={"method": "archival"}),
            patch.object(workflow, "_write_candidate", side_effect=write_candidate),
            patch.object(workflow, "_audit", side_effect=audit),
            patch("branddozer.research.RUNTIME_ROOT", Path(temp.name) / "papers"),
        ):
            paper = workflow.execute()

        run.refresh_from_db()
        self.assertEqual(calls["write"], 2)
        self.assertEqual(paper.status, "validated")
        self.assertEqual(paper.revisions.count(), 2)
        self.assertEqual(paper.sources.count(), 3)
        self.assertEqual(paper.claims.count(), 3)
        self.assertTrue(Path(paper.markdown_path).is_file())
        self.assertTrue(Path(paper.pdf_path).is_file())
        self.assertEqual(run.status, "complete")
        self.assertEqual(run.phase, "research_complete")
