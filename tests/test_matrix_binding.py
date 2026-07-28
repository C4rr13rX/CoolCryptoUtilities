import pytest

from tools.c0d3rV2.matrix_helpers import (
    _matrix_search, _matrix_search_by_discipline, _matrix_search_by_variables,
    assess_binding, build_equation_chain, classify_domains, validate_dimensions,
)


def test_domain_classification_is_multilabel_and_explainable():
    result = classify_domains("Estimate heat loss, energy cost, price and profit")
    names = {r["domain"] for r in result}
    assert {"thermodynamics", "economics"} <= names
    assert all("evidence" in r for r in result)


def test_equation_chain_reaches_target_through_intermediate():
    result = build_equation_chain(["v = d / t", "E = m * v**2 / 2"], ["d", "t", "m"], ["E"])
    assert not result["unreachable"]
    assert result["paths"]["E"] == ["v = d / t", "E = m * v**2 / 2"]


def test_open_binding_reports_missing_target():
    result = assess_binding(equations=["v = d / t"], knowns=["d", "t"], targets=["E"])
    assert not result["bound"]
    assert result["missing"] == ["E"]


def test_invalid_equation_cannot_close_binding():
    result = assess_binding(equations=["v == d / t"], knowns=["d", "t"], targets=["v"])
    assert not result["bound"]
    assert result["rejected_equations"]


def test_dimensional_mismatch_is_rejected():
    result = validate_dimensions(["d = t"], {"d": "L", "t": "T"})
    assert not result["valid"]
    assert result["invalid"]


@pytest.mark.django_db
def test_seeded_matrix_search_works_on_sqlite_json_fields():
    result = _matrix_search("kinetic energy", limit=5)
    assert any("kinetic" in hit.get("label", "").lower() for hit in result["hits"])
    assert _matrix_search_by_discipline("ClassicalMechanics")
    assert _matrix_search_by_variables(["m", "v"])
