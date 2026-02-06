from __future__ import annotations

import re
from dataclasses import dataclass
import os
from typing import Iterable, List, Optional, Tuple

from django.db import transaction

from core.models import Equation, EquationGapFill, EquationSource, EquationVariable


EQUATION_PATTERN = re.compile(r"([A-Za-z0-9_+\-*/^() ]+=[A-Za-z0-9_+\-*/^() ]+)")


@dataclass
class ExtractedEquation:
    text: str
    variables: List[str]


def extract_equations(text: str) -> List[ExtractedEquation]:
    if not text:
        return []
    matches = EQUATION_PATTERN.findall(text)
    results: List[ExtractedEquation] = []
    for match in matches:
        eq = match.strip()
        vars_found = sorted({tok for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", eq)})
        results.append(ExtractedEquation(text=eq, variables=vars_found))
    return results


def infer_variables(equations: Iterable[ExtractedEquation]) -> List[str]:
    vars_all = set()
    for eq in equations:
        vars_all.update(eq.variables)
    return sorted(vars_all)


def ingest_equations(
    text: str,
    *,
    source_title: str = "",
    source_url: str = "",
    discipline_tags: Optional[List[str]] = None,
    citation: str = "",
    raw_excerpt: str = "",
    citations: Optional[List[str]] = None,
    tool_used: str = "",
    captured_at=None,
) -> List[Equation]:
    extracted = extract_equations(text)
    if not extracted:
        return []
    discipline_tags = discipline_tags or []
    with transaction.atomic():
        source = None
        if source_title or source_url:
            source, _ = EquationSource.objects.get_or_create(
                title=source_title or "Unknown Source",
                url=source_url or "",
                defaults={"citation": citation or "", "raw_excerpt": raw_excerpt or ""},
            )
        created: List[Equation] = []
        for eq in extracted:
            eq_model = Equation.objects.create(
                text=eq.text,
                variables=eq.variables,
                disciplines=discipline_tags,
                source=source,
                citations=citations or ([citation] if citation else []),
                tool_used=tool_used or "",
                captured_at=captured_at,
            )
            created.append(eq_model)
            for symbol in eq.variables:
                EquationVariable.objects.get_or_create(symbol=symbol, dimension="")
        try:
            from services.graph_store import sync_graph_from_django
            if os.getenv("C0D3R_GRAPH_SYNC_ON_WRITE", "1").strip().lower() not in {"0", "false", "no", "off"}:
                sync_graph_from_django()
        except Exception:
            pass
        return created


def record_gap_fill(description: str, steps: List[str], equations: Iterable[Equation]) -> EquationGapFill:
    gap = EquationGapFill.objects.create(description=description, steps=steps)
    gap.equations.add(*list(equations))
    return gap
