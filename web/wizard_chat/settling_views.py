"""web/wizard_chat/settling_views.py

API endpoints for:
- Brain Settling & Sleep-Pressure Feedback Loop (preventing RAM bloat on CPU-only PCs)
- Empirically Validated 3rd-Party Training Pipeline (commercially usable)
- First-Principles Procedural 3D Mesh Generation & UV Mapping
"""
from __future__ import annotations

import json
import pathlib
import sys
import urllib.request
from django.http import HttpResponse, JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

_REPO_ROOT = str(pathlib.Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.wizard_learning.cognitive_state import get_cognitive_state_manager
from tools.wizard_learning.settling_controller import get_settling_controller
from tools.wizard_learning.empirical_material import (
    EMPIRICAL_CURRICULUM,
    get_curriculum_by_category,
)
from tools.wizard_learning.self_refining_pipeline import SelfRefiningPipeline
from tools.wizard_learning.procedural_3d import generate_mesh


class WizardSettlingView(View):
    """GET /api/wizard-chat/settling/ — inspect RAM, sleep pressure, and settling history."""

    def get(self, request):
        controller = get_settling_controller()
        metrics = controller.get_resource_metrics()
        history = controller.get_history(limit=10)
        cog = get_cognitive_state_manager().read()
        return JsonResponse({
            "ok": True,
            "metrics": metrics,
            "history": history,
            "last_settled_at": controller.last_settled_at,
            "total_settle_cycles": controller.total_settle_cycles,
            "cognitive_state": cog,
        })


@method_decorator(csrf_exempt, name="dispatch")
class WizardSettlingRunView(View):
    """POST /api/wizard-chat/settling/run/ — trigger manual sleep / consolidation."""

    def post(self, request):
        reason = "manual_user_request"
        try:
            body = json.loads(request.body)
            if isinstance(body, dict) and body.get("reason"):
                reason = str(body["reason"])
        except Exception:
            pass

        controller = get_settling_controller()
        result = controller.settle(reason=reason)
        return JsonResponse({
            "ok": True,
            "settled": result.get("settled", False),
            "record": result.get("record", {}),
            "metrics": result.get("metrics_after", {}),
        })


class WizardEmpiricalCurriculumView(View):
    """GET /api/wizard-chat/empirical/ — list third-party empirically validated curriculum."""

    def get(self, request):
        category = request.GET.get("category")
        items = get_curriculum_by_category(category)
        return JsonResponse({
            "ok": True,
            "count": len(items),
            "items": [
                {
                    "id": item.id,
                    "category": item.category,
                    "topic": item.topic,
                    "question": item.question,
                    "answer": item.answer,
                    "formula": item.formula,
                    "units": item.units,
                    "source_citation": item.source_citation,
                    "license": item.license,
                    "commercial_use_permitted": item.commercial_use_permitted,
                    "empirically_validated": item.empirically_validated,
                }
                for item in items
            ],
        })


@method_decorator(csrf_exempt, name="dispatch")
class WizardEmpiricalTrainView(View):
    """POST /api/wizard-chat/empirical/train/ — trigger autonomous training batch."""

    def post(self, request):
        category = None
        max_items = 10
        try:
            body = json.loads(request.body)
            if isinstance(body, dict):
                category = body.get("category")
                max_items = int(body.get("max_items", 10))
        except Exception:
            pass

        pipeline = SelfRefiningPipeline()
        result = pipeline.run_curriculum_batch(category=category, max_items=max_items)
        return JsonResponse({
            "ok": True,
            "batch_result": result,
        })


@method_decorator(csrf_exempt, name="dispatch")
class WizardProceduralMeshView(View):
    """POST /api/wizard-chat/mesh/generate/ — generate 3D procedural model with UVs."""

    def post(self, request):
        category = "machine_part"
        part_type = "motor_housing"
        try:
            body = json.loads(request.body)
            if isinstance(body, dict):
                category = str(body.get("category") or category)
                part_type = str(body.get("part_type") or part_type)
        except Exception:
            pass

        mesh_data = generate_mesh(category, part_type)
        return JsonResponse({
            "ok": True,
            "mesh": mesh_data,
        })


@method_decorator(csrf_exempt, name="dispatch")
class WizardDeductiveReasoningView(View):
    """POST /api/wizard-chat/deduce/ — solve novel unseen engineering challenges from first principles."""

    def post(self, request):
        from tools.wizard_learning.deductive_reasoner import (
            get_deductive_reasoner,
            RobotDesignRequirement,
        )

        req = RobotDesignRequirement()
        try:
            body = json.loads(request.body)
            if isinstance(body, dict):
                req = RobotDesignRequirement(
                    payload_kg=float(body.get("payload_kg", 15.0)),
                    reach_meters=float(body.get("reach_meters", 1.2)),
                    wind_speed_ms=float(body.get("wind_speed_ms", 0.0)),
                    ambient_temp_kelvin=float(body.get("ambient_temp_kelvin", 295.15)),
                    bus_voltage_v=float(body.get("bus_voltage_v", 24.0)),
                    operating_hours=float(body.get("operating_hours", 4.0)),
                    safety_factor_target=float(body.get("safety_factor_target", 2.0)),
                    preferred_material=body.get("preferred_material"),
                )
        except Exception:
            pass

        reasoner = get_deductive_reasoner()
        solution = reasoner.deduce_robot_system(req)
        return JsonResponse({
            "ok": True,
            "solution": {
                "task_summary": solution.task_summary,
                "selected_material": solution.selected_material,
                "material_properties": solution.material_properties,
                "structural_mechanics": solution.structural_mechanics,
                "powertrain_electromechanics": solution.powertrain_electromechanics,
                "battery_chemistry": solution.battery_chemistry,
                "fastener_and_assembly": solution.fastener_and_assembly,
                "verification_passed": solution.verification_passed,
                "verification_report": solution.verification_report,
                "cad_mesh_metadata": solution.cad_mesh_metadata,
                "wavefront_obj": solution.wavefront_obj,
                "threejs_geometry": getattr(solution, "threejs_geometry", {}),
                "blueprint_svg": getattr(solution, "blueprint_svg", ""),
                "blueprint_dxf": getattr(solution, "blueprint_dxf", ""),
                "tolerance_stackup": getattr(solution, "tolerance_stackup", {}),
                "bill_of_materials": getattr(solution, "bill_of_materials", []),
                "manufacturing_plan": getattr(solution, "manufacturing_plan", []),
            },
        })


class WizardBlueprintExportView(View):
    """GET /api/wizard-chat/blueprint/export/?type=(robot_arm|skyscraper|skyscraper_mep|gear)&format=(svg|dxf)"""

    def get(self, request):
        bp_type = request.GET.get("type", "robot_arm")
        fmt = request.GET.get("format", "svg").lower()
        as_file = request.GET.get("download", "0") in ("1", "true")

        from tools.wizard_learning.blueprint_engine import EngineeringBlueprintEngine

        content = ""
        filename = f"W1Z4RD_{bp_type.upper()}.{fmt}"

        if bp_type in ("robot_arm", "arm"):
            content = EngineeringBlueprintEngine.generate_robot_arm_blueprint(format=fmt)
            filename = f"W1Z-ROB-001_ROBOT_ARM_BOOM.{fmt}"
        elif bp_type in ("skyscraper", "skyscraper_elevation", "elevation"):
            content = EngineeringBlueprintEngine.generate_skyscraper_structural_blueprint(format=fmt)
            filename = f"W1Z-SKYS-001_SKYSCRAPER_ELEVATION.{fmt}"
        elif bp_type in ("skyscraper_mep", "mep", "floorplan"):
            content = EngineeringBlueprintEngine.generate_skyscraper_mep_floorplan_blueprint(format=fmt)
            filename = f"W1Z-SKYS-002_SKYSCRAPER_MEP_FLOORPLAN.{fmt}"
        elif bp_type in ("gear", "involute_gear", "part"):
            content = EngineeringBlueprintEngine.generate_component_blueprint("involute_spur_gear", format=fmt)
            filename = f"W1Z-PRT-001_INVOLUTE_SPUR_GEAR.{fmt}"
        else:
            return JsonResponse({"ok": False, "error": f"Unknown blueprint type: {bp_type}"}, status=400)

        if as_file:
            from django.http import HttpResponse
            content_type = "application/dxf" if fmt == "dxf" else "image/svg+xml; charset=utf-8"
            response = HttpResponse(content, content_type=content_type)
            response["Content-Disposition"] = f'attachment; filename="{filename}"'
            return response

        return JsonResponse({
            "ok": True,
            "type": bp_type,
            "format": fmt,
            "filename": filename,
            "content": content,
        })


@method_decorator(csrf_exempt, name="dispatch")
class WizardKuzuWorkbenchView(View):
    """GET/POST /api/wizard-chat/kuzu-workbench/ — Mathematical Workbench for Unbounded Problems.

    Binds environmental problems as mathematical graphs, verifies closure,
    and prevents hallucinations via SymPy equation validation.
    """

    def get(self, request):
        from tools.wizard_learning.kuzu_workbench import get_kuzu_workbench
        wb = get_kuzu_workbench()
        stats = wb.get_stats()
        query = request.GET.get("q", "")
        disc = request.GET.get("discipline", "")
        var = request.GET.get("variable", "")
        hits = []
        if query or disc or var:
            hits = wb.search_workbench(query=query, discipline=disc, variable=var, limit=20)

        return JsonResponse({
            "ok": True,
            "workbench_available": wb.is_available(),
            "stats": stats,
            "results": hits,
        })

    def post(self, request):
        try:
            body = json.loads(request.body)
        except Exception:
            return JsonResponse({"ok": False, "error": "Invalid JSON body"}, status=400)

        problem = str(body.get("problem", "")).strip()
        knowns = body.get("knowns") or []
        targets = body.get("targets") or []
        candidates = body.get("candidate_equations") or []
        dimensions = body.get("dimensions") or None

        if not problem and not targets:
            return JsonResponse({"ok": False, "error": "Must provide 'problem' and 'targets'"}, status=400)

        from tools.wizard_learning.kuzu_workbench import get_kuzu_workbench
        wb = get_kuzu_workbench()
        res = wb.bind_environmental_problem(
            problem=problem,
            knowns=knowns,
            targets=targets,
            candidate_equations=candidates,
            dimensions=dimensions,
        )

        return JsonResponse({
            "ok": True,
            "bound": res.bound,
            "closure": res.closure,
            "problem": res.problem,
            "knowns": res.knowns,
            "targets": res.targets,
            "reachable_targets": res.reachable_targets,
            "missing_targets": res.missing_targets,
            "dimensional_valid": res.dimensional_valid,
            "anomalies": res.anomalies,
            "sub_questions": res.sub_questions,
            "summary": res.summary,
            "equations_count": len(res.equations),
        })


_EYE_DESIGN_CACHE: dict[str, tuple[float, dict | None]] = {}


class WizardCognitiveStateView(View):
    """GET /api/wizard-chat/cognitive-state/ — instant (<5ms), lock-free live
    W1z4rD + C0d3rV2 cognitive state, 24/7 self-education/training/testing
    counters, and chat-availability readiness."""

    def get(self, request):
        import time as _time
        cog = get_cognitive_state_manager().read()
        settling = get_settling_controller().get_resource_metrics()
        # Cross-check node process liveness from cached settling metrics without
        # blocking Waitress threads on the node's inner Mutex during training/sleep.
        if settling.get("node_pid") is None and cog.get("mode") not in ("SLEEPING_SETTLING", "OFFLINE"):
            cog = {
                **cog,
                "mode": "OFFLINE",
                "eye_lifecycle_state": "offline",
                "mode_label": "Node Offline / Booting",
                "can_chat": False,
                "chat_unavailable_reason": "W1z4rD node process is offline or restarting.",
            }
        elif settling.get("node_pid") is not None and cog.get("mode") == "OFFLINE":
            cog = {
                **cog,
                "mode": "IDLE_READY",
                "eye_lifecycle_state": "waking",
                "mode_label": "Ready to Chat",
                "can_chat": True,
                "chat_unavailable_reason": "",
                "detail": "W1z4rD + C0d3rV2 online and ready.",
            }

        eye_design = None
        try:
            now_t = _time.time()
            cached_ed = _EYE_DESIGN_CACHE.get("default")
            if cached_ed and (now_t - cached_ed[0]) < 2.5:
                eye_design = cached_ed[1]
            else:
                from .views import _brain_for_request
                from tools.wizard_learning.self_design import current as current_self_design
                from tools.wizard_learning.store import LearningStore, brain_key
                profile = _brain_for_request(request)
                eye_design = current_self_design(LearningStore(), brain_key(profile))
                _EYE_DESIGN_CACHE["default"] = (now_t, eye_design)
        except Exception:
            eye_design = None

        return JsonResponse({
            "ok": True,
            "cognitive_state": cog,
            "settling_metrics": settling,
            "current_eye_design": eye_design,
        })


def _cors_json(payload: dict, status: int = 200) -> JsonResponse:
    resp = JsonResponse(payload, status=status)
    resp["Access-Control-Allow-Origin"] = "*"
    resp["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@method_decorator(csrf_exempt, name="dispatch")
class WizardRobotWorldBridgeView(View):
    """GET/POST /api/wizard-chat/robot-world/ — Live telemetry & controller bridge
    between Multiscale Robot World (BrandDozer) and W1z4rD Brain + C0d3rV2."""

    def options(self, request):
        resp = HttpResponse(status=204)
        resp["Access-Control-Allow-Origin"] = "*"
        resp["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    def get(self, request):
        cog = get_cognitive_state_manager().read()
        telemetry_path = pathlib.Path(_REPO_ROOT) / "runtime" / "c0d3r" / "msrw_swarm_telemetry.json"
        telemetry = {}
        if telemetry_path.exists():
            try:
                telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
            except Exception:
                telemetry = {}
        return _cors_json({
            "ok": True,
            "robot_world": cog.get("robot_world", {}),
            "telemetry": telemetry,
        })

    def post(self, request):
        try:
            body = json.loads(request.body)
        except Exception:
            return _cors_json({"ok": False, "error": "Invalid JSON"}, status=400)

        action = str(body.get("action") or "sync_telemetry")
        cog_mgr = get_cognitive_state_manager()

        if action == "decide":
            # Query W1z4rD brain for closed-loop robot controller guidance
            prompt = str(body.get("prompt") or "").strip()
            robot_id = str(body.get("robot_id") or "wizard-prime")
            answer = ""
            decoder = "physics_policy"
            if prompt:
                try:
                    req = urllib.request.Request(
                        "http://127.0.0.1:8090/chat",
                        data=json.dumps({"text": prompt, "hops": 2, "min_strength": 0.05}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=4.0) as r:
                        d = json.loads(r.read().decode("utf-8"))
                        if d.get("answer") and d.get("decoder") != "char_chain":
                            answer = str(d["answer"])
                            decoder = str(d.get("decoder") or "multi_pool")
                except Exception:
                    pass
            return _cors_json({
                "ok": True,
                "robot_id": robot_id,
                "wizard_guidance": answer,
                "decoder": decoder,
            })

        # Sync swarm telemetry & train W1z4rD on any newly verified Mark N -> Mark N+1 body upgrades
        telemetry = body.get("telemetry") if isinstance(body.get("telemetry"), dict) else body
        telemetry_path = pathlib.Path(_REPO_ROOT) / "runtime" / "c0d3r" / "msrw_swarm_telemetry.json"
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            telemetry_path.write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
        except Exception:
            pass

        rw_patch = {}
        for k in ("active_controllers", "robots_built", "body_generation", "primary_mark", "last_upgrade_summary", "physics_tests_passed"):
            if k in telemetry:
                rw_patch[k] = telemetry[k]
        if rw_patch:
            cog_mgr.update(robot_world=rw_patch)

        lessons_trained = 0
        for lesson in (telemetry.get("learned_lessons") or [])[:4]:
            if isinstance(lesson, dict) and lesson.get("question") and lesson.get("answer"):
                try:
                    req = urllib.request.Request(
                        "http://127.0.0.1:8090/multi_pool/train_pair",
                        data=json.dumps({
                            "src_pool": "in",
                            "src": str(lesson["question"]),
                            "tgt_pool": "out",
                            "tgt": str(lesson["answer"]),
                            "passes": 15,
                        }).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=10.0):
                        lessons_trained += 1
                except Exception:
                    pass

        return _cors_json({
            "ok": True,
            "lessons_trained": lessons_trained,
            "robot_world": cog_mgr.read().get("robot_world", {}),
        })
