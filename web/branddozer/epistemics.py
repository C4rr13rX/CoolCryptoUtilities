"""branddozer/epistemics.py — reasoning-quality gates for research papers.

Why this exists
---------------
The original acceptance gates in ``research.py::validate_paper_payload``
are all *counting* gates: word count, source count, domain count. A paper
could satisfy every one of them and still be weak science — each claim
resting on a single source, causal language asserted from correlational
evidence, no rival explanations considered, and conclusions sitting many
un-audited inferential hops from the evidence.

This module adds the orthogonal axis: how *well reasoned* the paper is.

The four dimensions
-------------------
1. **Corroboration** — a claim carried by two or more *independent*
   sources (different domains, and not all first-party to the subject) is
   epistemically stronger than the same claim carried once. Load-bearing
   causal claims are held to that higher bar.

2. **Causal warrant** — a claim that asserts causation must declare the
   design that licenses it (natural experiment, difference-in-differences,
   dose-response, process tracing, …) and its identification strategy. A
   causal claim backed only by co-occurrence is downgraded to
   *correlational*, which is a finding, not a cause.

3. **Inferential distance** — how many reasoning hops separate a claim
   from primary evidence. Hop 0 is quoted from a document; hop 3 is an
   inference about an inference about an inference. Deep chains are not
   forbidden — complex conclusions need them — but they must be *labelled*,
   and a paper whose headline conclusions are all deep-hop speculation is
   not publication-grade.

4. **Rival hypotheses** — a conclusion that never states what else could
   explain the evidence, and why that alternative is less consistent with
   it, is advocacy rather than analysis.

5. **Modality** — the logical force of a claim. "X did happen", "X must
   happen", "X can happen", "X usually happens" and "X happened in this
   case" are different assertions with different evidentiary burdens.
   Papers routinely slide from *possible* to *actual* to *necessary*
   without new evidence; typing modality makes that slide visible, and
   bars the two illegitimate moves (possible->actual, actual->necessary).

6. **Deniability** — for corporate, legal and political archival work,
   the wording of a primary document is often *engineered* to permit
   later denial: passive voice with no actor, "aspirational" targets with
   no commitment mechanic, "up to" quantities, undefined key terms. A
   paper that reads such text as a firm commitment has been captured by
   its source. Each first-party document claim therefore records how
   deniable the underlying wording is, and highly deniable wording cannot
   silently carry a firm conclusion.

Together these push the writer toward conclusions that reach further
*and* carry their uncertainty honestly, instead of short papers that
merely clear a word count.
"""
from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Iterable


# Claim epistemic types, ordered by the strength of warrant they require.
CLAIM_TYPES = ("descriptive", "correlational", "causal", "counterfactual", "normative")

# Designs that can license a causal claim from archival evidence.
CAUSAL_DESIGNS = (
    "natural_experiment",
    "difference_in_differences",
    "interrupted_time_series",
    "regression_discontinuity",
    "instrumental_variable",
    "dose_response",
    "process_tracing",
    "comparative_case_control",
    "triangulated_documentary",
)

# Language that asserts causation. Used to catch claims typed "descriptive"
# or "correlational" whose prose nonetheless asserts a cause, which is the
# most common way a paper overstates its warrant.
CAUSAL_LANGUAGE = (
    "caused", "causes", "causing", "led to", "resulted in", "drove",
    "triggered", "produced", "brought about", "because of", "due to",
    "as a result of", "consequently", "therefore the", "forced",
    "made them", "responsible for",
)

HEDGE_LANGUAGE = (
    "may", "might", "could", "suggests", "consistent with", "appears",
    "is associated with", "correlates", "plausibly", "we infer",
    "tentatively", "if ", "under the assumption", "cannot be determined",
    "not estimable", "insufficient evidence",
)

# --- Modality -------------------------------------------------------------
# The logical force asserted. Ordered loosely by evidentiary burden.
MODALITIES = (
    "actual",       # it happened; requires documentation of the event
    "possible",     # it could happen; requires a mechanism, not an instance
    "probable",     # it likely happened/holds; requires a base rate or trend
    "necessary",    # it must hold; requires a rule, law, contract or identity
    "counterfactual",  # it would have happened; requires a stated model
)

# Illegitimate modal escalations: concluding more force than the premise
# carries. These are the two classic fallacies in archival argument.
ILLEGITIMATE_MODAL_STEPS = {
    ("possible", "actual"),      # "could have" -> "did"
    ("actual", "necessary"),     # "did once"   -> "must always"
    ("probable", "necessary"),   # "usually"    -> "must"
}

# --- Deniability ----------------------------------------------------------
# Wording patterns that preserve deniability in first-party documents.
DENIABILITY_MARKERS = (
    "aspirational", "aims to", "aim to", "strives", "strive to", "intends to",
    "up to", "as much as", "where feasible", "where appropriate",
    "may include", "among other", "and similar", "such as",
    "committed to exploring", "working toward", "working towards",
    "goal of", "targets of", "we believe", "designed to",
    "is expected to", "anticipates", "from time to time",
    "in our sole discretion", "subject to change", "no assurance",
    "forward-looking",
)

DENIABILITY_LEVELS = ("firm", "qualified", "deniable")


@dataclass(frozen=True)
class EpistemicPolicy:
    """Thresholds for the reasoning-quality gates."""

    # Fraction of causal claims that must be independently corroborated.
    min_causal_corroboration_rate: float = 0.8
    # Independent sources required to call a causal claim corroborated.
    causal_corroboration_sources: int = 2
    # Every causal claim must name a design + identification strategy.
    require_causal_design: bool = True
    # Share of claims allowed to sit at inferential hop >= 3.
    max_deep_inference_rate: float = 0.35
    # Conclusions must consider at least this many rival explanations.
    min_rival_hypotheses: int = 2
    # Claims that must carry an explicit uncertainty/limits statement.
    min_uncertainty_labelled_rate: float = 0.6
    # A paper must reach at least this far: shallow papers that only
    # restate documents are descriptive, not analytic.
    min_analytic_claim_rate: float = 0.25
    # Every claim must declare its modality (actual/possible/...).
    require_modality: bool = True
    # Share of first-party document claims that must record how deniable
    # the underlying wording is.
    min_deniability_assessed_rate: float = 0.8

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "EpistemicPolicy":
        config = config or {}

        def _num(name: str, default: float, lo: float, hi: float) -> float:
            try:
                return max(lo, min(hi, float(config.get(name, default))))
            except (TypeError, ValueError):
                return default

        def _int(name: str, default: int, lo: int, hi: int) -> int:
            try:
                return max(lo, min(hi, int(config.get(name, default))))
            except (TypeError, ValueError):
                return default

        return cls(
            min_causal_corroboration_rate=_num(
                "min_causal_corroboration_rate", 0.8, 0.0, 1.0
            ),
            causal_corroboration_sources=_int(
                "causal_corroboration_sources", 2, 1, 10
            ),
            require_causal_design=bool(
                config.get("require_causal_design", True)
            ),
            max_deep_inference_rate=_num("max_deep_inference_rate", 0.35, 0.0, 1.0),
            min_rival_hypotheses=_int("min_rival_hypotheses", 2, 0, 10),
            min_uncertainty_labelled_rate=_num(
                "min_uncertainty_labelled_rate", 0.6, 0.0, 1.0
            ),
            min_analytic_claim_rate=_num("min_analytic_claim_rate", 0.25, 0.0, 1.0),
            require_modality=bool(config.get("require_modality", True)),
            min_deniability_assessed_rate=_num(
                "min_deniability_assessed_rate", 0.8, 0.0, 1.0
            ),
        )


def _domain(url: str) -> str:
    host = (urllib.parse.urlparse(str(url or "")).hostname or "").lower()
    return host.removeprefix("www.")


def independent_support(
    claim: dict[str, Any], source_by_key: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Measure how independently corroborated a claim is.

    Independence is by *domain*, not by citation count: three quotes from
    one press release are one source, not three. First-party sources are
    counted separately because a company confirming its own program is
    authoritative for what the program says, but not independent
    confirmation of that program's effects.
    """
    keys = [str(k) for k in (claim.get("source_keys") or []) if str(k).strip()]
    verified = [
        source_by_key[k]
        for k in keys
        if k in source_by_key
        and source_by_key[k].get("verification_status") == "verified"
    ]
    domains = {_domain(s.get("url", "")) for s in verified if s.get("url")}
    third_party = {
        _domain(s.get("url", ""))
        for s in verified
        if s.get("url") and not s.get("first_party")
    }
    return {
        "verified_count": len(verified),
        "independent_domains": len(domains),
        "third_party_domains": len(third_party),
        # Genuine corroboration needs >= 2 distinct domains and at least
        # one that is not the subject talking about itself.
        "corroborated": len(domains) >= 2 and len(third_party) >= 1,
    }


def claim_type(claim: dict[str, Any]) -> str:
    declared = str(claim.get("claim_type") or "").strip().lower()
    if declared in CLAIM_TYPES:
        return declared
    return "descriptive"


def asserts_causation(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in CAUSAL_LANGUAGE)


def is_hedged(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in HEDGE_LANGUAGE)


def modality(claim: dict[str, Any]) -> str:
    """Declared logical force of a claim ("" when unset)."""
    declared = str(claim.get("modality") or "").strip().lower()
    return declared if declared in MODALITIES else ""


def modal_escalation(claim: dict[str, Any]) -> dict[str, Any] | None:
    """Detect concluding more modal force than the premise supports.

    A claim may declare `premise_modality` (the force its evidence
    actually carries) alongside `modality` (what the claim asserts).
    Going from "possible" to "actual", or "actual" to "necessary", needs
    new evidence, not rewording.
    """
    concluded = modality(claim)
    premise = str(claim.get("premise_modality") or "").strip().lower()
    if not concluded or premise not in MODALITIES:
        return None
    if (premise, concluded) in ILLEGITIMATE_MODAL_STEPS:
        return {
            "from": premise,
            "to": concluded,
            "claim": str(claim.get("claim_text") or "")[:200],
        }
    return None


def wording_is_deniable(text: str) -> bool:
    """True when quoted wording is hedged enough to permit later denial."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in DENIABILITY_MARKERS)


def deniability_level(claim: dict[str, Any]) -> str:
    """How deniable the underlying wording is ("" when unassessed)."""
    declared = str(claim.get("deniability") or "").strip().lower()
    if declared in DENIABILITY_LEVELS:
        return declared
    return ""


def inference_depth(claim: dict[str, Any]) -> int:
    """Reasoning hops between primary evidence and the claim."""
    try:
        depth = int(claim.get("inference_depth"))
    except (TypeError, ValueError):
        # Unlabelled claims are treated as one hop: asserted from a source
        # but not quoted from it.
        return 1
    return max(0, min(depth, 10))


def evaluate_reasoning(
    *,
    claims: Iterable[dict[str, Any]],
    sources: Iterable[dict[str, Any]],
    policy: EpistemicPolicy,
    rival_hypotheses: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score a paper's reasoning quality.

    Returns checks/metrics in the same shape as the deterministic gates so
    the two can be merged into one validation report.
    """
    claim_list = [c for c in claims if isinstance(c, dict)]
    source_by_key = {
        str(s.get("citation_key") or ""): s for s in sources if isinstance(s, dict)
    }
    rivals = [r for r in (rival_hypotheses or []) if isinstance(r, dict)]

    causal_claims: list[dict[str, Any]] = []
    corroborated_causal = 0
    undesigned_causal: list[dict[str, Any]] = []
    overclaimed: list[dict[str, Any]] = []
    deep_claims: list[dict[str, Any]] = []
    uncertainty_labelled = 0
    analytic_claims = 0
    unmodalised: list[dict[str, Any]] = []
    modal_overreach: list[dict[str, Any]] = []
    first_party_claims = 0
    deniability_assessed = 0
    deniable_but_firm: list[dict[str, Any]] = []

    for index, claim in enumerate(claim_list):
        text = str(claim.get("claim_text") or "")
        ctype = claim_type(claim)
        support = independent_support(claim, source_by_key)
        depth = inference_depth(claim)

        if ctype in {"correlational", "causal", "counterfactual"}:
            analytic_claims += 1
        if depth >= 3:
            deep_claims.append({"index": index, "claim": text[:200], "depth": depth})
        if str(claim.get("uncertainty") or "").strip() or is_hedged(text):
            uncertainty_labelled += 1

        # --- Modality ---
        declared_modality = modality(claim)
        if policy.require_modality and not declared_modality:
            unmodalised.append({"index": index, "claim": text[:200]})
        escalation = modal_escalation(claim)
        if escalation:
            modal_overreach.append({"index": index, **escalation})

        # --- Deniability ---
        # Only meaningful where the support is the subject's own wording.
        cited = [
            source_by_key[str(key)]
            for key in (claim.get("source_keys") or [])
            if str(key) in source_by_key
        ]
        if any(s.get("first_party") for s in cited):
            first_party_claims += 1
            level = deniability_level(claim)
            quoted = str(claim.get("quoted_wording") or "")
            if level:
                deniability_assessed += 1
            elif quoted and wording_is_deniable(quoted):
                # Wording is visibly hedged but the claim never says so.
                deniable_but_firm.append(
                    {"index": index, "claim": text[:200], "reason": "unassessed"}
                )
            # Treating deniable wording as a firm fact is the failure mode.
            if level == "deniable" and declared_modality in {"actual", "necessary"}:
                if not is_hedged(text):
                    deniable_but_firm.append(
                        {
                            "index": index,
                            "claim": text[:200],
                            "reason": f"deniable wording asserted as {declared_modality}",
                        }
                    )

        # Prose that asserts causation while typed weaker is an overclaim,
        # unless the sentence is explicitly hedged.
        if ctype in {"descriptive", "correlational"} and asserts_causation(text):
            if not is_hedged(text):
                overclaimed.append(
                    {"index": index, "claim": text[:200], "declared_type": ctype}
                )

        if ctype in {"causal", "counterfactual"}:
            causal_claims.append(claim)
            if support["corroborated"] and (
                support["verified_count"] >= policy.causal_corroboration_sources
            ):
                corroborated_causal += 1
            design = str(claim.get("causal_design") or "").strip().lower()
            identification = str(claim.get("identification_strategy") or "").strip()
            if policy.require_causal_design and (
                design not in CAUSAL_DESIGNS or not identification
            ):
                undesigned_causal.append(
                    {
                        "index": index,
                        "claim": text[:200],
                        "declared_design": design or "(none)",
                        "has_identification": bool(identification),
                    }
                )

    total = len(claim_list)
    causal_total = len(causal_claims)
    causal_rate = (corroborated_causal / causal_total) if causal_total else 1.0
    deep_rate = (len(deep_claims) / total) if total else 0.0
    uncertainty_rate = (uncertainty_labelled / total) if total else 0.0
    analytic_rate = (analytic_claims / total) if total else 0.0

    deniability_rate = (
        (deniability_assessed / first_party_claims) if first_party_claims else 1.0
    )

    checks = {
        "causal_claims_corroborated": causal_rate >= policy.min_causal_corroboration_rate,
        "causal_claims_have_design": not undesigned_causal,
        "no_causal_overclaiming": not overclaimed,
        "inference_depth_bounded": deep_rate <= policy.max_deep_inference_rate,
        "rival_hypotheses_considered": len(rivals) >= policy.min_rival_hypotheses,
        "uncertainty_labelled": uncertainty_rate >= policy.min_uncertainty_labelled_rate,
        "analytic_reach": analytic_rate >= policy.min_analytic_claim_rate,
        "modality_declared": not unmodalised,
        "no_modal_escalation": not modal_overreach,
        "deniability_assessed": deniability_rate >= policy.min_deniability_assessed_rate,
        "no_deniable_wording_as_fact": not deniable_but_firm,
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "claims": total,
            "causal_claims": causal_total,
            "corroborated_causal_claims": corroborated_causal,
            "causal_corroboration_rate": round(causal_rate, 3),
            "deep_inference_rate": round(deep_rate, 3),
            "uncertainty_labelled_rate": round(uncertainty_rate, 3),
            "analytic_claim_rate": round(analytic_rate, 3),
            "rival_hypotheses": len(rivals),
            "first_party_claims": first_party_claims,
            "deniability_assessed_rate": round(deniability_rate, 3),
            "modal_escalations": len(modal_overreach),
        },
        "causal_without_design": undesigned_causal[:20],
        "causal_overclaims": overclaimed[:20],
        "deep_inference_claims": deep_claims[:20],
        "claims_without_modality": unmodalised[:20],
        "modal_escalations": modal_overreach[:20],
        "deniable_wording_as_fact": deniable_but_firm[:20],
    }


def writer_requirements(policy: EpistemicPolicy) -> str:
    """The reasoning contract, stated to the writer up front.

    Mirrors evaluate_reasoning exactly: the writer is told the same rules
    it will be graded on, so revision rounds correct real defects instead
    of guessing.
    """
    return (
        "\n\nREASONING REQUIREMENTS (scored automatically; these decide "
        "whether the paper is publication-grade, not just long enough):\n"
        "- Type every claim: descriptive, correlational, causal, "
        "counterfactual, or normative. Match the prose to the type — do not "
        "write 'X caused Y' for a claim you typed correlational.\n"
        f"- Every causal or counterfactual claim needs `causal_design` (one of: "
        f"{', '.join(CAUSAL_DESIGNS)}) and a written `identification_strategy` "
        "explaining what rules out confounding.\n"
        f"- At least {int(policy.min_causal_corroboration_rate * 100)}% of causal "
        f"claims must rest on {policy.causal_corroboration_sources}+ verified "
        "sources spanning two or more independent domains, including at least "
        "one that is not the subject describing itself.\n"
        "- Set `inference_depth` per claim: 0 = quoted from a primary "
        "document, 1 = direct reading of one, 2 = synthesis across sources, "
        "3+ = inference built on inference. Reaching far is welcome; "
        f"mislabelling how far is not. Keep hop-3+ claims under "
        f"{int(policy.max_deep_inference_rate * 100)}% of all claims.\n"
        f"- Include a Rival Explanations subsection naming at least "
        f"{policy.min_rival_hypotheses} alternative accounts of the evidence, "
        "each with what would distinguish it and why the record favours or "
        "fails to settle it.\n"
        "- Give each claim an `uncertainty` field, or hedge it in prose, "
        "stating what would overturn it.\n"
        "- A paper that only restates documents is not analysis: at least "
        f"{int(policy.min_analytic_claim_rate * 100)}% of claims should be "
        "correlational, causal, or counterfactual, each properly warranted.\n"
        f"- Set `modality` on every claim (one of: {', '.join(MODALITIES)}) — "
        "the logical force you assert. Also set `premise_modality`: the force "
        "your evidence actually carries. Going possible->actual or "
        "actual->necessary requires new evidence, not rewording; such "
        "escalations are flagged.\n"
        "- For any claim resting on the subject's own documents, quote the "
        "operative wording in `quoted_wording` and set `deniability` to firm, "
        "qualified, or deniable. Wording engineered for deniability "
        "('aspirational', 'up to', 'where feasible', 'aims to', undefined "
        "terms, actorless passive voice) must not be reported as a firm "
        "commitment: say what the text does and does not bind the subject to."
    )


__all__ = [
    "EpistemicPolicy",
    "evaluate_reasoning",
    "writer_requirements",
    "independent_support",
    "inference_depth",
    "claim_type",
    "asserts_causation",
    "is_hedged",
    "modality",
    "modal_escalation",
    "deniability_level",
    "wording_is_deniable",
    "CAUSAL_DESIGNS",
    "CLAIM_TYPES",
    "MODALITIES",
    "DENIABILITY_LEVELS",
    "ILLEGITIMATE_MODAL_STEPS",
]
