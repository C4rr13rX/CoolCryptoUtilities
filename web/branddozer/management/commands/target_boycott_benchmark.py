from __future__ import annotations

from pathlib import Path

from django.core.management import BaseCommand

from branddozer.models import BrandProject
from services.branddozer_delivery import delivery_orchestrator
from services.branddozer_jobs import enqueue_job


BENCHMARK_PROMPT = """
Produce a publication-grade archival research paper about the consumer boycott
of Target Corporation associated with changes to Target's diversity, equity,
inclusion, and minority-focused programs.

Identity and chronology gate:
- Do not assume the boycott, program, organizer, dates, or alleged replacement.
  First identify the event(s) people call the "Target boycott," distinguish
  similarly named or earlier boycotts, and justify the event definition and
  study windows using contemporaneous primary evidence.
- Reconstruct Target's relevant programs before the minority-focused program,
  the minority-focused program itself, any programs that coexisted with it,
  what ended or changed, and every documented successor or replacement.
- Compare written eligibility, rules, funding, governance, implementation, and
  measured reach. Determine whether a successor intrinsically includes the
  populations targeted by the earlier program; do not infer inclusion merely
  from universal wording.

Outcome questions:
1. What economic, social, and psychological impacts on the affected minority
   communities are empirically supportable before, during, and after the
   program/change? Separate Target-specific evidence from broader literature.
2. What impacts are not estimable from available data? Never turn absence of
   evidence into evidence of no effect.
3. If the minority-focused program had continued under current social and
   psychological conditions, what plausible effects might follow for targeted
   individuals' mental health and for relationships between minority groups
   and other Americans? Treat this strictly as counterfactual scenario
   analysis. State assumptions, causal pathways, rival pathways, uncertainty,
   and disconfirming observations. Do not diagnose groups or essentialize race.

Dynamical-systems comparison:
- Operationalize each program's rules as a candidate attractor in a social
  dynamical system. Define state variables, boundary conditions, feedback
  loops, lag structure, observables, similarity measures, and falsification
  criteria before selecting comparisons.
- Search historical and cross-cultural cases for structurally similar rule
  systems for both the minority-focused program and its successor. Cases need
  not concern the same racial groups or the United States.
- "Fractal" must be treated as a testable structural analogy, not as a finding.
  Compare mechanisms at multiple scales, report failed analogies and negative
  cases, and avoid cherry-picking vivid examples.

Validity and motivated-reasoning controls:
- Use a systematic archival-review protocol with explicit search dates,
  databases/search engines, inclusion/exclusion criteria, source hierarchy,
  provenance, and a reproducible evidence table.
- State hypotheses and serious rival hypotheses before synthesis. Include
  evidence favorable and unfavorable to claims made by boycott supporters,
  boycott opponents, Target, and political or advocacy organizations.
- Prefer Target filings and archived corporate materials, government data,
  contemporaneous records, preregistered or peer-reviewed studies, systematic
  reviews, and high-quality longitudinal/quasi-experimental evidence.
- Keep corporate statements, news reporting, advocacy claims, observational
  associations, causal estimates, model-based inference, and counterfactual
  speculation visibly separate.
- Check exact quoted passages against fetched source documents. Cite only
  verified sources. Report conflicts, missing data, measurement error,
  selection effects, confounding, construct validity, external validity, and
  uncertainty. Do not manufacture precision or sources.
- Use neutral terminology and run a hostile citation/claim audit plus an
  independent methods review. Rewrite until all deterministic evidence gates
  pass or clearly report why the evidence cannot support a requested claim.

The final paper must answer each numbered question directly, include a dated
chronology and program-comparison table, distinguish empirical conclusions from
inferences and scenarios, and be suitable for critical review in social
science, applied psychology, economics, and systems-science venues.
""".strip()


class Command(BaseCommand):
    help = "Queue the reproducible Target-boycott archival-research benchmark."

    def add_arguments(self, parser):
        parser.add_argument(
            "--root",
            default="runtime/branddozer/benchmarks/target-boycott",
            help="Workspace used for benchmark session artifacts.",
        )

    def handle(self, *args, **options):
        root = Path(options["root"])
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[4] / root
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)

        project, _ = BrandProject.objects.update_or_create(
            name="Target Boycott Research Benchmark",
            defaults={
                "root_path": str(root),
                "default_prompt": BENCHMARK_PROMPT,
                "enabled": False,
                "workflow_kind": "research_benchmark",
                "workflow_config": {
                    "agent_provider": "c0d3r",
                    "model_provider": "freeloader",
                    "benchmark": "target_boycott_v1",
                },
            },
        )
        run = delivery_orchestrator.create_run(
            str(project.id),
            BENCHMARK_PROMPT,
            mode="existing",
            research_mode=True,
            team_mode="full",
            session_provider="freeloader",
        )
        run.context = {
            **(run.context or {}),
            "agent_provider": "c0d3r",
            "model_provider": "freeloader",
            "benchmark": {
                "name": "target_boycott_v1",
                "hypothesis_lock_required": True,
                "motivated_reasoning_audit": True,
                "dynamical_systems_comparison": True,
            },
            "research_config": {
                "target_journal": (
                    "interdisciplinary social science, applied psychology, "
                    "economics, and systems science"
                ),
                "citation_style": "apa",
                "min_words": 6000,
                "min_sources": 12,
                "min_verified_sources": 10,
                "min_high_authority_sources": 6,
                "min_source_domains": 5,
                "max_revision_rounds": 4,
                "max_parallel_agents": 3,
            },
        }
        run.save(update_fields=["context"])
        job = enqueue_job(
            kind="delivery_run",
            project=project,
            run=run,
            payload={
                "project_type": "research",
                "benchmark": "target_boycott_v1",
                "agent_provider": "c0d3r",
                "model_provider": "freeloader",
            },
            message="Target boycott research benchmark queued",
        )
        run.context = {**run.context, "job_id": str(job.id)}
        run.save(update_fields=["context"])

        self.stdout.write(
            self.style.SUCCESS(
                f"project_id={project.id}\nrun_id={run.id}\njob_id={job.id}"
            )
        )
