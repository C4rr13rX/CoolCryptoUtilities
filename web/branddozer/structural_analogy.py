"""branddozer/structural_analogy.py — compare rule systems across cases.

The method this implements
-------------------------
When a focal case is poorly documented, badly contested, or simply
unretrievable, the usual archival pipeline stalls and produces a report
saying the work could not be done. That is a failure of method, not a
finding: it happened on a live run, where a request for exactly this
analysis produced a paper whose every section explained why something was
"unexecutable", cited no case earlier than 2020, and never once used the
word "attractor".

The way through is to stop treating the subject as the unit of analysis
and treat its *rules* as the unit instead:

1. **Extract the rule set.** Who is eligible, on what criteria, funded
   how, governed by whom, measured against what, enforced by what
   mechanism, and ended how. This is recoverable from far less evidence
   than a full causal history.

2. **Treat the rule set as an attractor.** A rule system pulls behaviour
   toward particular states. Name the state variables, the feedback
   loops, the lags, and what would falsify the reading.

3. **Search across time and place for structurally similar rule sets.**
   The comparison case need not share the industry, country, decade, or
   population — a 19th-century guild rule, a 1970s housing covenant and a
   modern supplier-diversity program can share a structure. This is where
   a model's general knowledge is a legitimate *heuristic for where to
   look*, provided every candidate is then verified against sources.

4. **Read the context around the rules.** How did people respond, what
   did the rule mutate into, and what outcomes followed? Structure plus
   context is what licenses inference about the focal case.

5. **Report negative cases.** Analogies that failed are evidence. A
   comparison set with no failures has been cherry-picked.

Nothing here asserts that the focal case *is* the analogue. It produces
bounded, falsifiable structural inference, with the strength of each
analogy scored and its disanalogies stated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

# The dimensions on which two rule systems are compared. Chosen because
# each is usually recoverable from a primary document and each changes
# behaviour independently of the others.
RULE_DIMENSIONS = (
    "eligibility",       # who qualifies, on what criteria
    "allocation",        # what is distributed, how much, by what formula
    "funding",           # where resources come from, how secured
    "governance",        # who decides, who can change the rules
    "measurement",       # what is counted, reported, audited
    "enforcement",       # what happens on non-compliance
    "duration",          # fixed term, rolling, indefinite, sunset
    "exit",              # how it ends, who can end it, what survives
)

# How strongly a comparison case licenses inference. Deliberately coarse:
# false precision here would be worse than none.
ANALOGY_STRENGTH = {
    "strong": "matches on most dimensions including enforcement and measurement",
    "moderate": "matches on structure but differs on scale, era, or enforcement",
    "weak": "shares surface features only; useful mainly as a negative case",
    "failed": "looked similar but diverges on a dimension that drives outcomes",
}

MIN_COMPARISON_CASES = 4
MIN_NEGATIVE_CASES = 1
MIN_ERAS = 2


@dataclass
class RuleSystem:
    """A programme, policy or institution described by its rules."""

    name: str
    era: str
    context: str
    dimensions: dict[str, str]

    def coverage(self) -> float:
        filled = sum(
            1 for d in RULE_DIMENSIONS if str(self.dimensions.get(d) or "").strip()
        )
        return round(filled / len(RULE_DIMENSIONS), 2)


def extract_rules_prompt(subject: str, evidence: str) -> str:
    """Recover a rule set from whatever documents were reachable."""
    dims = "\n".join(f"  - {d}" for d in RULE_DIMENSIONS)
    return (
        "Extract the RULE SET of the programme or policy below. Return "
        "STRICT JSON only.\n\n"
        f"SUBJECT: {subject}\n\n"
        "Describe it on each dimension, quoting the operative wording where "
        f"you have it:\n{dims}\n\n"
        "Rules are recoverable from far less evidence than a causal history. "
        "Where a dimension is genuinely unknown, write \"unknown\" for it — "
        "do not guess, and do not abandon the whole extraction because some "
        "dimensions are missing.\n\n"
        "Return JSON with: name, era (e.g. '2020-2025'), context (two "
        "sentences on the surrounding conditions), dimensions (an object "
        "keyed by the dimension names above), source_keys (list), and "
        "confidence ('documented', 'partial', or 'inferred').\n\n"
        f"EVIDENCE:\n{evidence[:60000]}"
    )


def find_analogues_prompt(rules: dict[str, Any], *, minimum: int = 6) -> str:
    """Search history and other domains for structurally similar rule sets.

    The model's general knowledge is used as a *pointer* to candidate
    cases; each candidate is verified against sources afterwards. That is
    a legitimate heuristic, and it is the only way to find a 19th-century
    guild rule that matches a modern supplier programme.
    """
    return (
        "Below is a rule system described dimension by dimension. Find "
        "historical and cross-domain cases whose RULES are structurally "
        "similar. Return STRICT JSON only.\n\n"
        f"RULE SYSTEM:\n{json.dumps(rules, indent=1)[:8000]}\n\n"
        "HOW TO SEARCH:\n"
        f"- Propose at least {minimum} candidate cases from ANY era, "
        "country, industry or institution. A medieval guild ordinance, a "
        "1930s lending rule, a 1970s university admissions policy and a "
        "modern procurement target can all be structural matches. Do not "
        "restrict to the same population, sector or decade.\n"
        f"- Span at least {MIN_ERAS} distinct eras. If every case is from "
        "the last ten years, you have not searched.\n"
        "- Match on RULE STRUCTURE — eligibility logic, allocation formula, "
        "governance, measurement, enforcement, exit — not on topic or "
        "sentiment.\n"
        f"- Include at least {MIN_NEGATIVE_CASES} NEGATIVE case: one that "
        "looks similar but diverges on a dimension that drove different "
        "outcomes. Negative cases are evidence, not padding.\n"
        "- For each case state what CONTEXT surrounded the rules, how "
        "people RESPONDED, what the rule MUTATED into, and what OUTCOMES "
        "followed.\n"
        "- These are leads to be verified, not findings. Give each a "
        "`verification_query` a researcher can use to check it.\n\n"
        "Return JSON with `cases`: a list of {name, era, jurisdiction, "
        "domain, context, dimensions (same keys as above), response ("
        "how affected people behaved), mutation (what the rule became), "
        "outcomes, strength ('strong'|'moderate'|'weak'|'failed'), "
        "disanalogies (what does NOT match, required), verification_query}."
    )


def compare_prompt(
    focal: dict[str, Any], cases: list[dict[str, Any]]
) -> str:
    """Score the analogies and state what they license about the focal case."""
    return (
        "Compare the focal rule system against each analogue and state what "
        "the comparison does and does not license. Return STRICT JSON only.\n\n"
        f"FOCAL:\n{json.dumps(focal, indent=1)[:6000]}\n\n"
        f"ANALOGUES:\n{json.dumps(cases, indent=1)[:40000]}\n\n"
        "RULES OF INFERENCE:\n"
        "- Structural similarity licenses a bounded expectation, never a "
        "conclusion about what did happen in the focal case.\n"
        "- Every inference must name the dimensions carrying it and the "
        "dimensions that undercut it.\n"
        "- State a falsification test: what observation would break this "
        "reading?\n"
        "- Report failed analogies explicitly. A comparison set with no "
        "failures has been cherry-picked.\n"
        "- Where the analogues disagree with each other, say so and say "
        "what would distinguish them. Do not average them.\n\n"
        "Return JSON with: attractor (state_variables, feedback_loops, lags, "
        "boundary_conditions), comparisons (list of {case, shared_dimensions, "
        "divergent_dimensions, strength, licensed_expectation, "
        "falsification_test}), convergent_findings (what most analogues "
        "agree on), divergent_findings, and residual_uncertainty."
    )


def validate_case_set(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Check a comparison set is broad and honest enough to reason from."""
    case_list = [c for c in cases if isinstance(c, dict)]
    eras = {str(c.get("era") or "").strip() for c in case_list if c.get("era")}
    domains = {str(c.get("domain") or "").strip() for c in case_list if c.get("domain")}
    negatives = [
        c for c in case_list
        if str(c.get("strength") or "").lower() in {"failed", "weak"}
    ]
    missing_disanalogies = [
        str(c.get("name") or "?") for c in case_list
        if not str(c.get("disanalogies") or "").strip()
    ]
    # A set drawn only from the last decade is not a historical search.
    modern_only = all(
        any(y in str(c.get("era") or "") for y in ("201", "202"))
        for c in case_list
    ) if case_list else True

    checks = {
        "enough_cases": len(case_list) >= MIN_COMPARISON_CASES,
        "spans_multiple_eras": len(eras) >= MIN_ERAS and not modern_only,
        "has_negative_cases": len(negatives) >= MIN_NEGATIVE_CASES,
        "every_case_states_disanalogies": not missing_disanalogies,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "cases": len(case_list),
            "eras": len(eras),
            "domains": len(domains),
            "negative_cases": len(negatives),
        },
        "cases_missing_disanalogies": missing_disanalogies[:10],
    }


def inconclusive_report(subject: str, reason: str, attempted: list[str]) -> dict[str, Any]:
    """The honest last resort, clearly labelled as such.

    If even the comparative route fails, the run must say plainly that the
    result is inconclusive — rather than dressing a failed retrieval up as
    a full-length paper whose findings are all statements of impossibility.
    """
    return {
        "status": "inconclusive",
        "subject": subject,
        "reason": reason,
        "methods_attempted": attempted,
        "headline": (
            f"Inconclusive: {subject} could not be analysed, and no adequate "
            "structural comparison set was assembled."
        ),
        "publishable": False,
    }


__all__ = [
    "RULE_DIMENSIONS",
    "ANALOGY_STRENGTH",
    "MIN_COMPARISON_CASES",
    "MIN_NEGATIVE_CASES",
    "MIN_ERAS",
    "RuleSystem",
    "extract_rules_prompt",
    "find_analogues_prompt",
    "compare_prompt",
    "validate_case_set",
    "inconclusive_report",
]
