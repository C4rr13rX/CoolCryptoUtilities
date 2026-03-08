"""
Environmental Equation Matrix — search, seed, and gap detection.

The matrix is a traversable collection of mathematical equations across
physics, engineering, and mathematics domains.  Each equation carries:
  - Discipline tags (ClassicalMechanics, QuantumMechanics, etc.)
  - Searchable labels (plain-English descriptions)
  - Metadata: variables, units, constraints, assumptions, confidence
  - Links to other equations (bridges, derives, contradicts)

The matrix is stored in Django (relational) and mirrored to Kuzu (graph)
for fast traversal.  Gap detection finds places where disciplines should
bridge but don't — these represent where new physics is needed.
"""
from __future__ import annotations

import re
from typing import Any

import helpers


_MATRIX_SEED_VERSION = "2026-03-08"


def _normalize_equation(text: str) -> str:
    """Normalize equation text for comparison (strip whitespace, lowercase)."""
    return re.sub(r"\s+", "", str(text).lower())


# ------------------------------------------------------------------
# Seed the base matrix with foundational equations
# ------------------------------------------------------------------

def _seed_base_matrix_django() -> None:
    """Seed the Django DB with foundational physics equations if empty."""
    if not helpers._ensure_django_ready():
        return
    try:
        from core.models import Equation, EquationDiscipline, EquationSource
        from django.utils import timezone

        if Equation.objects.filter(
            domains__contains=["ClassicalMechanics"]
        ).exists():
            return

        base = [
            # (discipline, equation, description/label, variables, latex)
            ("ClassicalMechanics", "F = m * a",
             "Newton's second law of motion.",
             ["F", "m", "a"], "F = m \\cdot a"),
            ("ClassicalMechanics", "p = m * v",
             "Linear momentum.",
             ["p", "m", "v"], "p = m \\cdot v"),
            ("ClassicalMechanics", "E_k = (1/2) * m * v**2",
             "Kinetic energy.",
             ["E_k", "m", "v"], "E_k = \\frac{1}{2} m v^2"),
            ("ClassicalMechanics", "W = F * d * cos(theta)",
             "Work done by a force.",
             ["W", "F", "d", "theta"], "W = F d \\cos\\theta"),
            ("Thermodynamics", "dU = T * dS - P * dV",
             "Fundamental thermodynamic relation.",
             ["dU", "T", "dS", "P", "dV"], "dU = T\\,dS - P\\,dV"),
            ("Thermodynamics", "P * V = n * R * T",
             "Ideal gas law.",
             ["P", "V", "n", "R", "T"], "PV = nRT"),
            ("Thermodynamics", "S = -k_B * sum(p_i * ln(p_i))",
             "Gibbs entropy.",
             ["S", "k_B", "p_i"], "S = -k_B \\sum p_i \\ln p_i"),
            ("Electromagnetism", "\u2207 \u00b7 E = \u03c1 / \u03b50",
             "Gauss's law (electric).",
             ["E", "rho", "epsilon_0"], "\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\epsilon_0}"),
            ("Electromagnetism", "\u2207 \u00b7 B = 0",
             "Gauss's law (magnetic — no monopoles).",
             ["B"], "\\nabla \\cdot \\mathbf{B} = 0"),
            ("Electromagnetism", "\u2207 \u00d7 E = -\u2202B/\u2202t",
             "Faraday's law of induction.",
             ["E", "B", "t"], "\\nabla \\times \\mathbf{E} = -\\frac{\\partial \\mathbf{B}}{\\partial t}"),
            ("Electromagnetism", "\u2207 \u00d7 B = \u03bc0 * J + \u03bc0 * \u03b50 * \u2202E/\u2202t",
             "Ampere-Maxwell law.",
             ["B", "mu_0", "J", "epsilon_0", "E", "t"],
             "\\nabla \\times \\mathbf{B} = \\mu_0 \\mathbf{J} + \\mu_0 \\epsilon_0 \\frac{\\partial \\mathbf{E}}{\\partial t}"),
            ("QuantumMechanics", "i * \u0127 * \u2202\u03c8/\u2202t = \u0124 * \u03c8",
             "Time-dependent Schr\u00f6dinger equation.",
             ["hbar", "psi", "H", "t"],
             "i\\hbar \\frac{\\partial \\psi}{\\partial t} = \\hat{H} \\psi"),
            ("QuantumMechanics", "E = \u0127 * \u03c9",
             "Planck-Einstein relation.",
             ["E", "hbar", "omega"], "E = \\hbar \\omega"),
            ("QuantumMechanics", "Delta_x * Delta_p >= hbar / 2",
             "Heisenberg uncertainty principle.",
             ["Delta_x", "Delta_p", "hbar"],
             "\\Delta x \\cdot \\Delta p \\geq \\frac{\\hbar}{2}"),
            ("Relativity", "E^2 = (p*c)^2 + (m*c^2)^2",
             "Energy-momentum relation.",
             ["E", "p", "c", "m"], "E^2 = (pc)^2 + (mc^2)^2"),
            ("Relativity", "E = m * c**2",
             "Mass-energy equivalence.",
             ["E", "m", "c"], "E = mc^2"),
            ("Relativity", "ds^2 = -c^2*dt^2 + dx^2 + dy^2 + dz^2",
             "Minkowski metric (flat spacetime interval).",
             ["ds", "c", "dt", "dx", "dy", "dz"],
             "ds^2 = -c^2 dt^2 + dx^2 + dy^2 + dz^2"),
            ("StatisticalMechanics", "S = k_B * ln(\u03a9)",
             "Boltzmann entropy.",
             ["S", "k_B", "Omega"], "S = k_B \\ln \\Omega"),
            ("FluidDynamics", "rho * (dv/dt + (v . nabla)v) = -nabla(P) + mu*nabla^2(v) + f",
             "Navier-Stokes equation (incompressible).",
             ["rho", "v", "t", "P", "mu", "f"],
             "\\rho\\left(\\frac{\\partial \\mathbf{v}}{\\partial t} + (\\mathbf{v} \\cdot \\nabla)\\mathbf{v}\\right) = -\\nabla P + \\mu \\nabla^2 \\mathbf{v} + \\mathbf{f}"),
            ("InformationTheory", "H = -sum(p_i * log2(p_i))",
             "Shannon entropy.",
             ["H", "p_i"], "H = -\\sum p_i \\log_2 p_i"),
        ]

        source, _ = EquationSource.objects.get_or_create(
            title="Standard physics textbooks",
            defaults={
                "citation": "Standard physics textbooks (collected seed).",
                "tags": ["seed", _MATRIX_SEED_VERSION],
            },
        )

        for domain, eq, desc, variables, latex in base:
            EquationDiscipline.objects.get_or_create(name=domain)
            Equation.objects.get_or_create(
                text=eq,
                defaults={
                    "latex": latex,
                    "domains": [domain],
                    "disciplines": [domain],
                    "variables": variables,
                    "confidence": 0.95,
                    "source": source,
                    "citations": (
                        [source.citation] if source and source.citation else []
                    ),
                    "tool_used": "seed_base_matrix",
                    "captured_at": timezone.now(),
                    "assumptions": [],
                    "constraints": [desc],
                },
            )
    except Exception:
        pass


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------

def _matrix_search(query: str, limit: int = 12) -> dict:
    """
    Search the equation matrix by text, discipline, variables, or metadata.

    Returns {hits: [{equation, domain, label, variables, confidence, eq_id}],
             missing: [str]}.
    """
    if not query:
        return {"hits": [], "missing": []}
    if not helpers._ensure_django_ready():
        return {"hits": [], "missing": []}
    _seed_base_matrix_django()

    try:
        # Prefer graph-accelerated search when available.
        try:
            from services.graph_store import search_graph_equations
            graph_hits = search_graph_equations(query, limit=limit)
            if graph_hits:
                missing = _find_missing_tokens(query, graph_hits)
                return {"hits": graph_hits, "missing": missing}
        except Exception:
            pass

        from core.models import Equation

        q = query.strip()
        tokens = _query_tokens(q)
        hits: list[dict] = []

        # Direct text/latex match.
        if any(ch in q for ch in ("=", "\u2207", "\u2202", "\u0127", "\u03a9", "\u03bb", "\u03bc")):
            qs = Equation.objects.filter(text__icontains=q)
        else:
            qs = (
                Equation.objects.filter(text__icontains=q)
                | Equation.objects.filter(latex__icontains=q)
            )

        for eq in qs[:limit]:
            hits.append(_eq_to_hit(eq))

        # Discipline-based search.
        if tokens:
            discipline_qs = Equation.objects.none()
            for t in tokens:
                discipline_qs = discipline_qs | Equation.objects.filter(
                    disciplines__contains=[t]
                )
            for eq in discipline_qs[:limit]:
                hits.append(_eq_to_hit(eq))

        # Token-based fuzzy search across all equations.
        if tokens:
            for eq in Equation.objects.all()[:200]:
                norm = _normalize_equation(eq.text)
                norm_latex = _normalize_equation(eq.latex or "")
                constraints_text = " ".join(str(c) for c in (eq.constraints or []))
                combined = f"{norm} {norm_latex} {constraints_text.lower()}"
                if any(_normalize_equation(t) in combined for t in tokens):
                    hits.append(_eq_to_hit(eq))

        unique = _dedupe(hits, limit)
        missing = _find_missing_tokens(query, unique)
        return {"hits": unique, "missing": missing}
    except Exception:
        return {"hits": [], "missing": []}


def _matrix_search_by_discipline(discipline: str, limit: int = 20) -> list[dict]:
    """Search equations by discipline tag."""
    if not helpers._ensure_django_ready():
        return []
    _seed_base_matrix_django()
    try:
        from core.models import Equation
        hits: list[dict] = []
        for eq in Equation.objects.filter(disciplines__contains=[discipline])[:limit]:
            hits.append(_eq_to_hit(eq))
        return hits
    except Exception:
        return []


def _matrix_search_by_variables(variables: list[str], limit: int = 20) -> list[dict]:
    """Find equations that reference specific variables."""
    if not variables or not helpers._ensure_django_ready():
        return []
    _seed_base_matrix_django()
    try:
        from core.models import Equation
        hits: list[dict] = []
        for var in variables:
            for eq in Equation.objects.filter(variables__contains=[var])[:limit]:
                hits.append(_eq_to_hit(eq))
        return _dedupe(hits, limit)
    except Exception:
        return []


def _matrix_find_gaps(discipline_a: str, discipline_b: str) -> list[dict]:
    """
    Find equations in discipline_a that share variables with discipline_b
    but have no EquationLink between them — these are potential gaps.
    """
    if not helpers._ensure_django_ready():
        return []
    _seed_base_matrix_django()
    try:
        from core.models import Equation, EquationLink

        eqs_a = list(Equation.objects.filter(disciplines__contains=[discipline_a])[:50])
        eqs_b = list(Equation.objects.filter(disciplines__contains=[discipline_b])[:50])

        gaps: list[dict] = []
        for ea in eqs_a:
            vars_a = set(ea.variables or [])
            for eb in eqs_b:
                vars_b = set(eb.variables or [])
                shared = vars_a & vars_b
                if not shared:
                    continue
                # Check if a link exists.
                linked = EquationLink.objects.filter(
                    from_equation=ea, to_equation=eb,
                ).exists() or EquationLink.objects.filter(
                    from_equation=eb, to_equation=ea,
                ).exists()
                if not linked:
                    gaps.append({
                        "eq_a": ea.text,
                        "eq_b": eb.text,
                        "discipline_a": discipline_a,
                        "discipline_b": discipline_b,
                        "shared_variables": list(shared),
                        "eq_a_id": ea.id,
                        "eq_b_id": eb.id,
                    })
        return gaps
    except Exception:
        return []


def _matrix_get_linked(eq_id: int) -> list[dict]:
    """Get all equations linked to a given equation."""
    if not helpers._ensure_django_ready():
        return []
    try:
        from core.models import Equation, EquationLink

        eq = Equation.objects.filter(id=eq_id).first()
        if not eq:
            return []

        linked: list[dict] = []
        for link in EquationLink.objects.filter(from_equation=eq):
            target = link.to_equation
            linked.append({
                **_eq_to_hit(target),
                "relation": link.relation_type,
                "notes": link.notes or "",
            })
        for link in EquationLink.objects.filter(to_equation=eq):
            source = link.from_equation
            linked.append({
                **_eq_to_hit(source),
                "relation": f"reverse:{link.relation_type}",
                "notes": link.notes or "",
            })
        return linked
    except Exception:
        return []


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

_STOP_WORDS = {
    "that", "this", "with", "from", "into", "then", "what",
    "the", "and", "for", "are", "how", "does", "has", "have",
}


def _eq_to_hit(eq: Any) -> dict:
    """Convert an Equation model instance to a search hit dict."""
    return {
        "eq_id": eq.id,
        "equation": eq.text,
        "latex": eq.latex or "",
        "domain": ",".join(eq.disciplines or eq.domains or []),
        "label": (eq.constraints[0] if eq.constraints else ""),
        "variables": eq.variables or [],
        "confidence": getattr(eq, "confidence", 0.5),
        "summary": "",
    }


def _query_tokens(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-zA-Z_]{3,}", text.lower())
        if t not in _STOP_WORDS
    ]


def _find_missing_tokens(query: str, hits: list[dict]) -> list[str]:
    tokens = _query_tokens(query)
    if not tokens:
        return []
    return [
        t for t in set(tokens[:10])
        if not any(t in str(h.get("equation", "")).lower() for h in hits)
    ]


def _dedupe(hits: list[dict], limit: int) -> list[dict]:
    unique: list[dict] = []
    seen: set[str] = set()
    for item in hits:
        key = item.get("equation", "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique
