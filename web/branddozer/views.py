from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from services.branddozer_runner import branddozer_manager
from services.branddozer_ai import generate_interjections
from services.branddozer_state import delete_project, get_project, list_projects, save_project, update_project_fields
from services.api_integrations import get_integration_value


class ProjectListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        projects = list_projects()
        runtime = branddozer_manager.snapshot()
        runtime_map = {entry["id"]: entry for entry in runtime}
        payload = []
        for project in projects:
            entry = dict(project)
            entry.update(runtime_map.get(project.get("id"), {}))
            payload.append(entry)
        return Response({"projects": payload}, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        data = request.data or {}
        try:
            project = save_project(data)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(project, status=status.HTTP_201_CREATED)


class ProjectDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request: Request, project_id: str, *args, **kwargs) -> Response:
        updates = request.data or {}
        try:
            project = update_project_fields(project_id, updates)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        if not project:
            return Response({"detail": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(project, status=status.HTTP_200_OK)

    def delete(self, request: Request, project_id: str, *args, **kwargs) -> Response:
        delete_project(project_id)
        branddozer_manager.stop(project_id)
        return Response(status=status.HTTP_204_NO_CONTENT)


class ProjectStartView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, project_id: str, *args, **kwargs) -> Response:
        if not get_project(project_id):
            return Response({"detail": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        result = branddozer_manager.start(project_id)
        return Response(result, status=status.HTTP_200_OK)


class ProjectStopView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, project_id: str, *args, **kwargs) -> Response:
        result = branddozer_manager.stop(project_id)
        return Response(result, status=status.HTTP_200_OK)


class ProjectLogView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, project_id: str, *args, **kwargs) -> Response:
        limit = int(request.query_params.get("limit", "200"))
        limit = max(10, min(limit, 2000))
        lines = branddozer_manager.tail_log(project_id, limit=limit)
        return Response({"lines": lines}, status=status.HTTP_200_OK)


class ProjectInterjectionSuggestView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, project_id: str, *args, **kwargs) -> Response:
        project = get_project(project_id)
        if not project:
            return Response({"detail": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        default_prompt = (request.data or {}).get("default_prompt") or project.get("default_prompt") or ""
        if not default_prompt.strip():
            return Response({"detail": "Default prompt is required"}, status=status.HTTP_400_BAD_REQUEST)
        api_key = get_integration_value(request.user, "OPENAI_API_KEY", reveal=True) or None
        try:
            interjections = generate_interjections(default_prompt, project.get("name"), api_key=api_key)
        except Exception as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        if interjections:
            update_project_fields(project_id, {"interjections": interjections, "last_ai_generated": int(time.time())})
        return Response({"interjections": interjections}, status=status.HTTP_200_OK)


class ProjectRootListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        raw_path = request.query_params.get("path") or str(Path.home())
        base_path = Path(raw_path).expanduser().resolve()
        home = Path.home().resolve()
        if not base_path.exists() or not base_path.is_dir():
            return Response({"detail": "Path must be an existing directory"}, status=status.HTTP_400_BAD_REQUEST)

        directories = []
        try:
            entries = sorted(base_path.iterdir(), key=lambda p: p.name.lower())
        except PermissionError:
            return Response({"detail": "Permission denied"}, status=status.HTTP_403_FORBIDDEN)

        for entry in entries:
            if not entry.is_dir():
                continue
            try:
                directories.append(
                    {
                        "name": entry.name,
                        "path": str(entry.resolve()),
                        "writable": os.access(entry, os.W_OK),
                    }
                )
            except PermissionError:
                continue

        parent = str(base_path.parent) if base_path != base_path.parent else None
        return Response(
            {
                "current_path": str(base_path),
                "parent": parent,
                "home": str(home),
                "directories": directories,
            },
            status=status.HTTP_200_OK,
        )


class ProjectGitHubImportView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, *args, **kwargs) -> Response:
        data = request.data or {}
        token = data.get("token") or data.get("pat")
        repo_url = data.get("repo_url") or data.get("repository") or ""
        branch = data.get("branch") or ""
        project_name = data.get("name")
        destination = data.get("destination")
        default_prompt = data.get("default_prompt") or ""

        if not repo_url:
            return Response({"detail": "Repository URL is required"}, status=status.HTTP_400_BAD_REQUEST)
        if not token:
            return Response({"detail": "GitHub personal access token is required"}, status=status.HTTP_400_BAD_REQUEST)
        if shutil.which("git") is None:
            return Response({"detail": "Git is not available on the server"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        parsed_repo = urlparse(repo_url if "://" in repo_url else f"https://{repo_url}")
        if not parsed_repo.scheme:
            parsed_repo = parsed_repo._replace(scheme="https")
        if not parsed_repo.netloc or "." not in parsed_repo.netloc:
            parsed_repo = urlparse(f"https://github.com/{repo_url.lstrip('/')}")
        if not parsed_repo.netloc:
            return Response({"detail": "Repository URL is invalid"}, status=status.HTTP_400_BAD_REQUEST)

        repo_name = Path(parsed_repo.path).stem or "branddozer-project"
        base_dir = Path(destination or (Path.home() / "BrandDozerProjects" / repo_name)).expanduser().resolve()
        home = Path.home().resolve()

        try:
            is_relative = base_dir.is_relative_to(home)
        except AttributeError:
            try:
                base_dir.relative_to(home)
                is_relative = True
            except Exception:
                is_relative = False

        if not is_relative:
            return Response({"detail": "Destination must be inside the home directory"}, status=status.HTTP_400_BAD_REQUEST)

        base_dir.parent.mkdir(parents=True, exist_ok=True)
        final_dir = base_dir
        suffix = 1
        while final_dir.exists():
            final_dir = base_dir.parent / f"{base_dir.name}-{suffix}"
            suffix += 1

        clone_base = parsed_repo
        clone_url = clone_base.geturl()
        if token:
            safe_netloc = f"{token}@{clone_base.netloc}"
            clone_url = clone_base._replace(netloc=safe_netloc).geturl()

        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd.extend(["-b", branch])
        cmd.extend([clone_url, str(final_dir)])

        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_SSH_COMMAND"] = "ssh -o StrictHostKeyChecking=no"
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            if final_dir.exists():
                shutil.rmtree(final_dir, ignore_errors=True)
            return Response({"detail": result.stderr.strip() or "Failed to clone repository"}, status=status.HTTP_400_BAD_REQUEST)
        clean_origin = clone_base.geturl()
        subprocess.run(
            ["git", "-C", str(final_dir), "remote", "set-url", "origin", clean_origin],
            capture_output=True,
            text=True,
        )

        payload = {
            "name": project_name or final_dir.name,
            "root_path": str(final_dir),
            "default_prompt": default_prompt,
            "interjections": data.get("interjections") or [],
            "interval_minutes": data.get("interval_minutes"),
            "repo_url": repo_url,
            "repo_branch": branch,
        }
        try:
            project = save_project(payload)
        except ValueError as exc:
            shutil.rmtree(final_dir, ignore_errors=True)
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        return Response({"project": project, "destination": str(final_dir)}, status=status.HTTP_201_CREATED)
