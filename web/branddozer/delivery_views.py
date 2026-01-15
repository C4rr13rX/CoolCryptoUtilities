from __future__ import annotations

from typing import Any, Dict, List, Tuple

from pathlib import Path
import mimetypes

from django.utils import timezone
from django.http import FileResponse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from services.branddozer_delivery import delivery_orchestrator
from services.branddozer_jobs import enqueue_job, job_payload, update_job
from branddozer.models import (
    AcceptanceRecord,
    BacklogItem,
    BackgroundJob,
    DeliveryProject,
    DeliveryArtifact,
    DeliveryRun,
    DeliverySession,
    GateRun,
    GovernanceArtifact,
    Sprint,
)

RUNTIME_ROOT = Path("runtime/branddozer").resolve()


def _tail_log(path: Path, limit: int = 200) -> List[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
    except Exception:
        return []
    return [line.rstrip("\n") for line in lines[-limit:]]


def _read_log_from(path: Path, cursor: int, max_bytes: int = 200000) -> Tuple[List[str], int, bool]:
    if not path.exists():
        return [], cursor, False
    try:
        with path.open("rb") as handle:
            handle.seek(max(cursor, 0))
            data = handle.read(max_bytes)
    except Exception:
        return [], cursor, False
    if not data:
        return [], cursor, False
    last_newline = data.rfind(b"\n")
    if last_newline == -1:
        text = data.decode("utf-8", errors="ignore")
        return [text.rstrip("\n")], cursor + len(data), False
    chunk = data[: last_newline + 1]
    new_cursor = cursor + last_newline + 1
    text = chunk.decode("utf-8", errors="ignore")
    lines = [line.rstrip("\n") for line in text.splitlines()]
    has_more = len(data) > last_newline + 1
    return lines, new_cursor, has_more


def _safe_runtime_file(path: str) -> Path | None:
    if not path:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        return None
    try:
        resolved.relative_to(RUNTIME_ROOT)
    except ValueError:
        return None
    return resolved


def _run_payload(run: DeliveryRun) -> Dict[str, Any]:
    return {
        "id": str(run.id),
        "project_id": str(run.project_id),
        "prompt": run.prompt,
        "mode": run.mode,
        "status": run.status,
        "phase": run.phase,
        "iteration": run.iteration,
        "sprint_count": run.sprint_count,
        "acceptance_required": run.acceptance_required,
        "acceptance_recorded": run.acceptance_recorded,
        "definition_of_done": run.definition_of_done,
        "context": run.context,
        "error": run.error,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


class DeliveryRunListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        project_id = request.query_params.get("project_id")
        qs = DeliveryRun.objects.all()
        if project_id:
            qs = qs.filter(project_id=project_id)
        return Response({"runs": [_run_payload(run) for run in qs[:50]]}, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        data = request.data or {}
        project_id = data.get("project_id")
        prompt = (data.get("prompt") or "").strip()
        mode = data.get("mode") or "auto"
        research_mode = bool(data.get("research") or data.get("research_mode"))
        if not project_id:
            return Response({"detail": "project_id is required"}, status=status.HTTP_400_BAD_REQUEST)
        if not prompt:
            return Response({"detail": "prompt is required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            run = delivery_orchestrator.create_run(project_id, prompt, mode=mode, research_mode=research_mode)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        job = enqueue_job(
            kind="delivery_run",
            project=run.project,
            run=run,
            user=request.user,
            payload={"mode": mode},
            message="Queued",
        )
        context = {**(run.context or {}), "job_id": str(job.id)}
        if research_mode:
            context["research_mode"] = True
        run.context = context
        run.save(update_fields=["context"])
        return Response({"run": _run_payload(run), "job_id": str(job.id)}, status=status.HTTP_201_CREATED)


class DeliveryRunDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, run_id: str, *args, **kwargs) -> Response:
        run = DeliveryRun.objects.filter(id=run_id).first()
        if not run:
            return Response({"detail": "Run not found"}, status=status.HTTP_404_NOT_FOUND)
        job = BackgroundJob.objects.filter(run=run).order_by("-created_at").first()
        payload = _run_payload(run)
        payload["job"] = job_payload(job) if job else None
        return Response({"run": payload}, status=status.HTTP_200_OK)


class DeliveryRunStopView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, run_id: str, *args, **kwargs) -> Response:
        run = DeliveryRun.objects.filter(id=run_id).first()
        if not run:
            return Response({"detail": "Run not found"}, status=status.HTTP_404_NOT_FOUND)
        context = dict(run.context or {})
        context["stop_requested"] = True
        context["status_note"] = "Stopped"
        context["status_detail"] = "Stopped by user"
        run.context = context
        run.status = "blocked"
        run.phase = "stopped"
        run.error = "Stopped by user"
        run.completed_at = timezone.now()
        run.save(update_fields=["context", "status", "phase", "error", "completed_at"])
        delivery_project = DeliveryProject.objects.filter(project=run.project).first()
        if delivery_project:
            delivery_project.status = "blocked"
            delivery_project.active_run = run
            delivery_project.save(update_fields=["status", "active_run", "updated_at"])
        job = BackgroundJob.objects.filter(run=run).order_by("-created_at").first()
        if job and job.status in {"queued", "running"}:
            update_job(str(job.id), status="canceled", message="Canceled by user", detail="Delivery run stopped")
        return Response({"run": _run_payload(run)}, status=status.HTTP_200_OK)


class DeliveryRunBacklogView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, run_id: str, *args, **kwargs) -> Response:
        items = BacklogItem.objects.filter(run_id=run_id).order_by("priority", "-created_at")
        payload = []
        for item in items:
            payload.append(
                {
                    "id": str(item.id),
                    "kind": item.kind,
                    "title": item.title,
                    "description": item.description,
                    "acceptance_criteria": item.acceptance_criteria,
                    "priority": item.priority,
                    "estimate_points": item.estimate_points,
                    "status": item.status,
                    "source": item.source,
                }
            )
        return Response({"backlog": payload}, status=status.HTTP_200_OK)


class DeliveryBacklogItemView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request: Request, item_id: str, *args, **kwargs) -> Response:
        item = BacklogItem.objects.filter(id=item_id).first()
        if not item:
            return Response({"detail": "Backlog item not found"}, status=status.HTTP_404_NOT_FOUND)
        updates = request.data or {}
        if "status" in updates:
            item.status = updates.get("status") or item.status
        if "priority" in updates:
            item.priority = int(updates.get("priority") or item.priority)
        item.save(update_fields=["status", "priority", "updated_at"])
        return Response({"item": {"id": str(item.id), "status": item.status, "priority": item.priority}}, status=status.HTTP_200_OK)


class DeliveryRunGateView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, run_id: str, *args, **kwargs) -> Response:
        gates = GateRun.objects.filter(run_id=run_id).order_by("-created_at")
        payload = []
        for gate in gates:
            payload.append(
                {
                    "id": gate.id,
                    "stage": gate.stage,
                    "name": gate.name,
                    "status": gate.status,
                    "command": gate.command,
                    "stdout": gate.stdout,
                    "stderr": gate.stderr,
                    "exit_code": gate.exit_code,
                    "duration_ms": gate.duration_ms,
                    "created_at": gate.created_at.isoformat(),
                }
            )
        return Response({"gates": payload}, status=status.HTTP_200_OK)


class DeliveryRunSessionView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, run_id: str, *args, **kwargs) -> Response:
        sessions = DeliverySession.objects.filter(run_id=run_id).order_by("-created_at")
        payload = []
        for session in sessions:
            payload.append(
                {
                    "id": str(session.id),
                    "role": session.role,
                    "name": session.name,
                    "status": session.status,
                    "workspace_path": session.workspace_path,
                    "log_path": session.log_path,
                    "last_heartbeat": session.last_heartbeat.isoformat() if session.last_heartbeat else None,
                    "created_at": session.created_at.isoformat(),
                    "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                }
            )
        return Response({"sessions": payload}, status=status.HTTP_200_OK)


class DeliverySessionLogView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, session_id: str, *args, **kwargs) -> Response:
        limit = int(request.query_params.get("limit", "200"))
        limit = max(20, min(limit, 2000))
        session = DeliverySession.objects.filter(id=session_id).first()
        if not session:
            return Response({"detail": "Session not found"}, status=status.HTTP_404_NOT_FOUND)
        path = Path(session.log_path or "")
        cursor_param = request.query_params.get("cursor")
        max_bytes = request.query_params.get("max_bytes")
        if cursor_param is not None:
            try:
                cursor = int(cursor_param)
            except (TypeError, ValueError):
                cursor = 0
            try:
                max_bytes_value = int(max_bytes) if max_bytes else 200000
            except (TypeError, ValueError):
                max_bytes_value = 200000
            lines, next_cursor, has_more = _read_log_from(path, cursor, max_bytes=max_bytes_value) if session.log_path else ([], cursor, False)
            return Response(
                {"lines": lines, "status": session.status, "cursor": next_cursor, "has_more": has_more},
                status=status.HTTP_200_OK,
            )
        lines = _tail_log(path, limit=limit) if session.log_path else []
        return Response({"lines": lines, "status": session.status}, status=status.HTTP_200_OK)


class DeliveryRunArtifactView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, run_id: str, *args, **kwargs) -> Response:
        artifacts = DeliveryArtifact.objects.filter(run_id=run_id).order_by("-created_at")
        payload: List[Dict[str, Any]] = []
        for artifact in artifacts:
            payload.append(
                {
                    "id": str(artifact.id),
                    "kind": artifact.kind,
                    "title": artifact.title,
                    "path": artifact.path,
                    "created_at": artifact.created_at.isoformat(),
                }
            )
        return Response({"artifacts": payload}, status=status.HTTP_200_OK)


class DeliveryArtifactFileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, artifact_id: str, *args, **kwargs) -> Response:
        artifact = DeliveryArtifact.objects.filter(id=artifact_id).first()
        if not artifact or not artifact.path:
            return Response({"detail": "Artifact not found"}, status=status.HTTP_404_NOT_FOUND)
        safe_path = _safe_runtime_file(artifact.path)
        if not safe_path:
            return Response({"detail": "Artifact unavailable"}, status=status.HTTP_404_NOT_FOUND)
        content_type, _ = mimetypes.guess_type(str(safe_path))
        response = FileResponse(safe_path.open("rb"), content_type=content_type or "application/octet-stream")
        if request.query_params.get("download") == "1":
            response["Content-Disposition"] = f'attachment; filename="{safe_path.name}"'
        return response


class DeliveryRunUICaptureView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, run_id: str, *args, **kwargs) -> Response:
        run = DeliveryRun.objects.filter(id=run_id).first()
        if not run:
            return Response({"detail": "Run not found"}, status=status.HTTP_404_NOT_FOUND)
        job = enqueue_job(
            kind="ui_capture",
            project=run.project,
            run=run,
            user=request.user,
            payload={"manual": True},
            message="Queued",
        )
        return Response({"status": "queued", "job_id": str(job.id)}, status=status.HTTP_202_ACCEPTED)


class DeliveryRunGovernanceView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, run_id: str, *args, **kwargs) -> Response:
        artifacts = GovernanceArtifact.objects.filter(run_id=run_id).order_by("-created_at")
        payload = []
        for artifact in artifacts:
            payload.append(
                {
                    "id": artifact.id,
                    "kind": artifact.kind,
                    "summary": artifact.summary,
                    "content": artifact.content,
                    "data": artifact.data,
                    "created_at": artifact.created_at.isoformat(),
                }
            )
        return Response({"governance": payload}, status=status.HTTP_200_OK)


class DeliveryRunSprintView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, run_id: str, *args, **kwargs) -> Response:
        sprints = Sprint.objects.filter(run_id=run_id).order_by("-number")
        payload = []
        for sprint in sprints:
            payload.append(
                {
                    "id": sprint.id,
                    "number": sprint.number,
                    "goal": sprint.goal,
                    "status": sprint.status,
                    "started_at": sprint.started_at.isoformat() if sprint.started_at else None,
                    "completed_at": sprint.completed_at.isoformat() if sprint.completed_at else None,
                }
            )
        return Response({"sprints": payload}, status=status.HTTP_200_OK)


class DeliveryRunAcceptView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, run_id: str, *args, **kwargs) -> Response:
        run = DeliveryRun.objects.filter(id=run_id).first()
        if not run:
            return Response({"detail": "Run not found"}, status=status.HTTP_404_NOT_FOUND)
        if run.status != "awaiting_acceptance":
            return Response({"detail": "Run is not awaiting acceptance"}, status=status.HTTP_400_BAD_REQUEST)
        notes = request.data.get("notes") or ""
        checklist = request.data.get("checklist") or []
        AcceptanceRecord.objects.create(
            project=run.project,
            run=run,
            user_name=str(request.user),
            notes=notes,
            checklist=checklist,
        )
        run.status = "complete"
        run.acceptance_recorded = True
        run.completed_at = timezone.now()
        run.save(update_fields=["status", "acceptance_recorded", "completed_at"])
        return Response({"run": _run_payload(run)}, status=status.HTTP_200_OK)
