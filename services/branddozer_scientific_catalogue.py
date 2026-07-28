"""Incremental research-to-implementation catalogue for BrandDozer projects."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResearchTarget:
    id: str
    domain: str
    question: str
    validity_domain: str
    implementation_targets: tuple[str, ...]
    test_targets: tuple[str, ...]


TARGETS = (
    ResearchTarget("classical_mechanics", "mechanics", "Ignoring air resistance, how far does an object initially at rest fall near Earth's surface in 3.0 seconds?", "Constant near-surface gravity and negligible drag", ("src/physics/mechanics/",), ("tests/physics/mechanics/",)),
    ResearchTarget("thermodynamics", "thermodynamics", "An insulated vessel mixes 1 kg of water at 20 C with 2 kg at 80 C. Neglect losses. Find equilibrium temperature.", "Lumped, closed, equal-specific-heat water system", ("src/physics/thermal/",), ("tests/physics/thermal/",)),
    ResearchTarget("electromagnetism_circuits", "electrical engineering", "For a first-order RC low-pass with R=3.3 kohm and C=47 nF, calculate the -3 dB cutoff frequency.", "Linear lumped-element first-order RC circuits", ("src/physics/electrical/", "src/robots/components/electrical/"), ("tests/physics/electrical/",)),
    ResearchTarget("quantum_photon", "quantum physics", "What is the energy in electron-volts of a 500 nm photon?", "Single-photon vacuum wavelength relation E=hc/lambda", ("src/physics/quantum/",), ("tests/physics/quantum/",)),
    ResearchTarget("transport_diffusion", "transport physics", "Using one-dimensional diffusion with D=1.0e-9 m^2/s, estimate the characteristic time for 100 micrometers using <x^2>=2Dt.", "Homogeneous one-dimensional Fickian diffusion", ("src/physics/transport/", "src/physics/materials/"), ("tests/physics/transport/",)),
    ResearchTarget("orbital_mechanics", "orbital mechanics", "Assuming a circular orbit, estimate the orbital period in minutes of a satellite 400 km above Earth and state constants.", "Newtonian two-body circular Earth orbit", ("src/physics/astronomy/", "src/world/planet/"), ("tests/physics/astronomy/",)),
    ResearchTarget("radio_propagation", "radio engineering", "At 2.4 GHz over 10 km free space, calculate path loss using FSPL=20log10(d_km)+20log10(f_MHz)+32.44.", "Far-field unobstructed free-space radio link", ("src/physics/electromagnetism/", "src/robots/components/radio/"), ("tests/physics/radio/",)),
    ResearchTarget("rigid_body", "mechanics", "For a closed rigid-body system, which equations govern linear and angular momentum conservation and under what assumptions?", "Classical nonrelativistic rigid bodies", ("src/physics/rigid-body/",), ("tests/physics/conservation/",)),
    ResearchTarget("fluid_wind", "fluid dynamics", "Using the drag equation F = 0.5 * rho * Cd * A * v^2 with air density rho = 1.225 kg/m^3, drag coefficient Cd = 1.2, reference area A = 2.0 m^2, and wind speed v = 20 m/s, calculate the wind loading force in newtons and state the subsonic continuum assumption.", "Continuum, subsonic atmospheric flow with declared Reynolds regime", ("src/physics/fluids/", "src/world/weather/"), ("tests/physics/fluids/",)),
    ResearchTarget("materials_failure", "materials engineering", "For a steel rod with yield strength 250 MPa and cross-sectional area 100 mm^2 under axial load, calculate the tensile force in newtons at which yielding begins using sigma = F / A.", "Engineering-scale components with calibrated material properties", ("src/physics/materials/", "src/robots/failure/"), ("tests/robots/failure/",)),
    ResearchTarget("planetary_model", "planetary science", "Using Newtonian surface gravity g = G * M / R^2 with G = 6.674e-11, planetary mass M = 5.972e24 kg, and radius R = 6.371e6 m, calculate the surface gravitational acceleration in meters per second squared.", "Continuum planetary approximation with explicit spatial resolution", ("src/world/planet/",), ("tests/world/planet/",)),
    ResearchTarget("galactic_model", "astrophysics", "Using the circular-orbit relation v = sqrt(G * M / r) with G = 6.674e-11, enclosed mass M = 1.0e41 kg, and orbital radius r = 2.46e20 m (about 8 kiloparsecs), calculate the orbital speed in meters per second and state the coarse-grained Newtonian assumption.", "Coarse-grained Newtonian/cosmological visualization; not particle-complete", ("src/physics/cosmology/",), ("tests/physics/cosmology/",)),
)


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _markdown(catalogue: dict[str, Any]) -> str:
    lines = ["# Scientific Model Catalogue", "", "Generated incrementally; unsupported records remain inconclusive.", ""]
    for record in catalogue.get("records", []):
        conclusion = record.get("scientific", {}).get("conclusion", {})
        lines.extend([
            f"## {record['id']}", "",
            f"- Domain: {record['domain']}",
            f"- Status: {conclusion.get('status', 'inconclusive')}",
            f"- Validity domain: {record['validity_domain']}",
            f"- Implementation targets: {', '.join(record['implementation_targets'])}",
            f"- Test targets: {', '.join(record['test_targets'])}",
            f"- Equations: {json.dumps(record.get('math_grounding', {}).get('equations', []))}",
            f"- Sources: {', '.join(source.get('url', '') for source in record.get('source_evidence', []))}", "",
        ])
    return "\n".join(lines)


def build_catalogue(root: Path, *, max_targets: int | None = None) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(project_root), str(project_root / "tools" / "c0d3rV2")]
    from tools.c0d3rV2.plugins.agent_the_freeloader import AgentTheFreeloaderSession
    from web_search import WebSearch
    from unbounded_solver import UnboundedSolver
    from tool_registry import ScientificMethodTool

    out_json = root / "research" / "model-catalogue.json"
    out_md = root / "research" / "model-catalogue.md"
    catalogue: dict[str, Any] = {"schema_version": 1, "status": "running", "records": [], "updated_at": ""}
    if out_json.exists():
        try:
            catalogue = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing_records = {record.get("id"): record for record in catalogue.get("records", [])}
    # Preserve proven records; retry and replace inconclusive records whenever
    # source discovery or verification improves.
    catalogue["records"] = [record for record in catalogue.get("records", []) if record.get("translation_status") == "ready"]
    existing = {record.get("id") for record in catalogue["records"]}
    runtime = project_root / "runtime" / "branddozer" / "research"
    session = AgentTheFreeloaderSession(session_name="branddozer-scientific-catalogue", transcript_dir=runtime / "transcripts", workdir=root, timeout_s=25, max_attempts=4, max_tokens=1800)
    web = WebSearch(session, max_results=8)
    scientific = ScientificMethodTool(web, runtime_dir=runtime)
    solver = UnboundedSolver(session, web)
    selected = TARGETS[:max_targets] if max_targets else TARGETS
    for target in selected:
        if target.id in existing:
            continue
        scientific_result = _json_safe(scientific.execute({"question": target.question, "domain": target.domain, "max_sources": 3}))
        grounding = _json_safe(solver.math_grounding(target.question))
        sources = [source for source in scientific_result.get("research", {}).get("results", []) if source.get("evidence_usable")]
        record = {
            "id": target.id, "domain": target.domain, "question": target.question,
            "validity_domain": target.validity_domain,
            "scientific": scientific_result,
            "math_grounding": grounding,
            "source_evidence": sources,
            "implementation_targets": list(target.implementation_targets),
            "test_targets": list(target.test_targets),
            "translation_status": "ready" if grounding.get("equations") and sources else "inconclusive",
        }
        catalogue.setdefault("records", []).append(record)
        catalogue["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _atomic_write(out_json, json.dumps(catalogue, indent=2, default=str))
        _atomic_write(out_md, _markdown(catalogue))
    ready = sum(record.get("translation_status") == "ready" for record in catalogue.get("records", []))
    total = len(selected)
    unresolved = [record.get("id") for record in catalogue.get("records", []) if record.get("translation_status") != "ready"]
    readiness = ready / max(1, total)
    catalogue["coverage"] = {"ready": ready, "total": total, "readiness": round(readiness, 4), "unresolved": unresolved}
    catalogue["status"] = "complete" if readiness >= 0.75 else "needs_remediation"
    catalogue["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_write(out_json, json.dumps(catalogue, indent=2, default=str))
    _atomic_write(out_md, _markdown(catalogue))
    return {"status": catalogue["status"], "records": len(catalogue.get("records", [])),
            "coverage": catalogue["coverage"], "paths": [str(out_json), str(out_md)]}


def build_validation_matrix(root: Path) -> dict[str, Any]:
    catalogue_path = root / "research" / "model-catalogue.json"
    if not catalogue_path.exists():
        raise RuntimeError("Scientific model catalogue must exist before the validation matrix")
    catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
    rows = []
    for record in catalogue.get("records", []):
        grounding = record.get("math_grounding") or {}
        sources = record.get("source_evidence") or []
        rows.append({
            "model_id": record.get("id"), "domain": record.get("domain"),
            "translation_status": record.get("translation_status", "inconclusive"),
            "validity_domain": record.get("validity_domain"),
            "equations": grounding.get("equations") or [],
            "constraints": grounding.get("constraints") or [],
            "reference_solutions": grounding.get("solutions") or [],
            "source_ids": [source.get("provenance_sha256") for source in sources],
            "implementation_targets": record.get("implementation_targets") or [],
            "test_targets": record.get("test_targets") or [],
            "required_checks": ["units", "reference tolerance", "validity bounds", "conservation where applicable", "deterministic replay", "adjacent-scale transition"],
        })
    matrix = {"schema_version": 1, "catalogue": str(catalogue_path), "rows": rows,
              "coverage": {"models": len(rows), "with_sources": sum(bool(row["source_ids"]) for row in rows),
                           "with_equations": sum(bool(row["equations"]) for row in rows),
                           "with_consumers": sum(bool(row["implementation_targets"] and row["test_targets"]) for row in rows)}}
    json_path = root / "research" / "validation-matrix.json"
    md_path = root / "research" / "traceability-matrix.md"
    _atomic_write(json_path, json.dumps(matrix, indent=2, default=str))
    lines = ["# Research Traceability Matrix", "", "| Model | Status | Implementation | Tests | Sources | Equations |", "|---|---|---|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model_id']} | {row['translation_status']} | {', '.join(row['implementation_targets'])} | {', '.join(row['test_targets'])} | {len(row['source_ids'])} | {len(row['equations'])} |")
    _atomic_write(md_path, "\n".join(lines))
    return {"status": "complete", "rows": len(rows), "coverage": matrix["coverage"], "paths": [str(json_path), str(md_path)]}
