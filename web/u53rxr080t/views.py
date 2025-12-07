from __future__ import annotations

import uuid
import os
import json
from django.utils import timezone
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from openai import OpenAI

from services.u53_agent import send_codex_update
from services.guardian_status import enqueue_slot, snapshot_status
from .models import AgentSession, Finding, Task


def _serialize_task(task: Task) -> dict:
    return {
        "id": str(task.id),
        "title": task.title,
        "description": task.description,
        "stage": task.stage,
        "status": task.status,
        "target_url": task.target_url,
        "assigned_to": str(task.assigned_to_id) if task.assigned_to_id else None,
        "meta": task.meta,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


def _serialize_agent(agent: AgentSession) -> dict:
    return {
        "id": str(agent.id),
        "name": agent.name,
        "kind": agent.kind,
        "platform": agent.platform,
        "browser": agent.browser,
        "status": agent.status,
        "last_seen": agent.last_seen,
        "meta": agent.meta,
    }


class HeartbeatView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        agent_id = payload.get("id") or str(uuid.uuid4())
        agent, _created = AgentSession.objects.update_or_create(
            id=agent_id,
            defaults={
                "name": payload.get("name") or "",
                "kind": payload.get("kind") or "browser",
                "platform": payload.get("platform") or "",
                "browser": payload.get("browser") or "",
                "status": payload.get("status") or "idle",
                "meta": payload.get("meta") or {},
                "last_seen": timezone.now(),
            },
        )
        return Response({"id": str(agent.id), "status": agent.status}, status=status.HTTP_200_OK)


class TasksView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        tasks = Task.objects.all()
        status_filter = request.query_params.get("status")
        if status_filter:
            tasks = tasks.filter(status=status_filter)
        serialized = [_serialize_task(t) for t in tasks]
        return Response({"tasks": serialized}, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        task = Task.objects.create(
            title=payload.get("title") or "UX sweep",
            description=payload.get("description") or "",
            stage=payload.get("stage") or "overview",
            target_url=payload.get("target_url") or "",
            status=payload.get("status") or "pending",
            meta=payload.get("meta") or {},
        )
        return Response({"task": str(task.id)}, status=status.HTTP_201_CREATED)


class TaskNextView(APIView):
    """
    Assign the next pending task to the requesting agent (by id) and mark it in_progress.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        agent_id = payload.get("agent_id") or payload.get("agent")
        agent = get_object_or_404(AgentSession, id=agent_id)
        task = (
            Task.objects.filter(status="pending", assigned_to__isnull=True)
            .order_by("created_at")
            .first()
        )
        if not task:
            return Response({"task": None}, status=status.HTTP_200_OK)
        task.assigned_to = agent
        task.status = "in_progress"
        task.save(update_fields=["assigned_to", "status", "updated_at"])
        return Response({"task": _serialize_task(task)}, status=status.HTTP_200_OK)


class TaskUpdateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, task_id: str, *args, **kwargs) -> Response:
        task = get_object_or_404(Task, id=task_id)
        payload = request.data or {}
        if "status" in payload:
            task.status = payload.get("status") or task.status
        if "meta" in payload:
            task.meta = payload.get("meta") or task.meta
        if "assigned_to" in payload:
            agent = AgentSession.objects.filter(id=payload.get("assigned_to")).first()
            task.assigned_to = agent
        task.save()
        return Response({"id": str(task.id), "status": task.status}, status=status.HTTP_200_OK)


class FindingView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        qs = Finding.objects.select_related("session").all()
        limit = request.query_params.get("limit")
        if limit:
            try:
                qs = qs[: int(limit)]
            except Exception:
                qs = qs[:100]
        data = [
            {
                "id": str(f.id),
                "session": str(f.session_id),
                "title": f.title,
                "summary": f.summary,
                "severity": f.severity,
                "screenshot_url": f.screenshot_url,
                "context": f.context,
                "created_at": f.created_at,
            }
            for f in qs
        ]
        return Response({"findings": data}, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        session_id = payload.get("session") or payload.get("session_id")
        if not session_id:
            return Response({"detail": "session is required"}, status=status.HTTP_400_BAD_REQUEST)
        session = get_object_or_404(AgentSession, id=session_id)
        finding = Finding.objects.create(
            session=session,
            title=payload.get("title") or "UX Finding",
            summary=payload.get("summary") or "",
            severity=payload.get("severity") or "info",
            screenshot_url=payload.get("screenshot_url") or "",
            context=payload.get("context") or {},
        )
        try:
            send_codex_update(f"[UX robot] {finding.title}", payload.get("summary") or "")
        except Exception:
            pass
        return Response({"id": str(finding.id)}, status=status.HTTP_201_CREATED)


class SuggestView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, *args, **kwargs) -> Response:
        """
        Send a screenshot/context to OpenAI to get suggested next actions.
        """
        payload = request.data or {}
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return Response({"actions": [], "detail": "OPENAI_API_KEY not configured"}, status=status.HTTP_200_OK)

        client = OpenAI(api_key=key)
        shot = payload.get("screenshot")
        task = payload.get("task") or {}
        context = payload.get("context") or {}

        sys_prompt = "You are a QA automation assistant for a crypto trading dashboard. Return JSON with a list of CSS selectors and short instructions for what to click or inspect next. Be safe and non-destructive."
        user_content = [
            {
                "type": "text",
                "text": f"Task: {task.get('title') or ''}\\nStage: {task.get('stage') or ''}\\nContext: {context}",
            }
        ]
        if shot:
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": {"url": shot, "detail": "low"},
                }
            )
        try:
            resp = client.responses.create(
                model="gpt-o4-mini",
                input=[
                    {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
            )
            text = (resp.output_text or "").strip()
            payload_json = json.loads(text) if text else {}
            actions = payload_json.get("actions") or payload_json.get("steps") or []
        except Exception:
            actions = []
        return Response({"actions": actions}, status=status.HTTP_200_OK)


class GuardianQueueView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        snapshot = snapshot_status()
        return Response(
            {
                "queue": snapshot.get("queue", {}).get("guardian", []),
                "history": snapshot.get("history", {}).get("guardian", []),
                "slots": snapshot.get("slots", {}).get("guardian"),
            },
            status=status.HTTP_200_OK,
        )

    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        owner = payload.get("session") or payload.get("agent") or "ux_robot"
        metadata = {
            "title": payload.get("title") or "UX robot update",
            "summary": payload.get("summary") or payload.get("details") or "",
            "severity": payload.get("severity") or "info",
            "meta": payload.get("meta") or {},
        }
        ticket = enqueue_slot("guardian", owner, metadata)
        if not ticket:
            return Response({"detail": "queue unavailable"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        return Response({"ticket": ticket}, status=status.HTTP_201_CREATED)


class AgentsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        agents = AgentSession.objects.all()
        serialized = [_serialize_agent(agent) for agent in agents]
        return Response({"agents": serialized}, status=status.HTTP_200_OK)
