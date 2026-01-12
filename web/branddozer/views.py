from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import quote, urlparse

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView
from django.db import close_old_connections
from django.utils.text import slugify

from branddozer.models import BrandProject
from services.branddozer_runner import branddozer_manager
from services.branddozer_ai import generate_interjections
from services.branddozer_github import (
    _git_checkout_branch as _git_checkout_branch_service,
    _git_current_branch as _git_current_branch_service,
    _git_fetch_with_token as _git_fetch_with_token_service,
    _git_is_clean as _git_is_clean_service,
    _git_is_repo as _is_git_repo_service,
    _git_remote_origin as _git_remote_origin_service,
    _normalize_repo_full_name as _normalize_repo_full_name_service,
    _infer_full_name_from_url as _infer_full_name_from_url_service,
    _repo_full_name_from_remote as _repo_full_name_from_remote_service,
    _strip_auth_from_url as _strip_auth_from_url_service,
    _https_remote_from_any as _https_remote_from_any_service,
    _with_token_url as _with_token_url_service,
    publish_project as publish_project_service,
    fetch_github_profile,
    fetch_repo_details,
    get_active_github_account,
    get_github_account,
    list_github_repos,
    list_repo_branches,
    create_github_repo,
    list_github_accounts,
    set_active_github_account,
    update_github_account_meta,
    upsert_github_account,
    _run_git as _run_git_service,
)
from services.branddozer_state import delete_project, get_project, list_projects, save_project, update_project_fields
from services.branddozer_jobs import enqueue_job, get_job, job_payload, update_job
from services.api_integrations import get_integration_value
from services.logging_utils import log_message


def _normalize_repo_full_name(raw_path: str) -> str:
    return _normalize_repo_full_name_service(raw_path)


def _infer_full_name_from_url(repo_url: str) -> str:
    return _infer_full_name_from_url_service(repo_url)


def _update_import_job(job_id: str, **updates: Any) -> None:
    update_job(job_id, **updates)


def _get_import_job(job_id: str, user: Any) -> Optional[Dict[str, Any]]:
    job = get_job(job_id, user=user)
    if not job:
        return None
    return job_payload(job)


def _update_publish_job(job_id: str, **updates: Any) -> None:
    update_job(job_id, **updates)


def _get_publish_job(job_id: str, user: Any) -> Optional[Dict[str, Any]]:
    job = get_job(job_id, user=user)
    if not job:
        return None
    return job_payload(job)


def _run_publish_job(job_id: str, user: Any, project_id: str, data: Dict[str, Any]) -> None:
    close_old_connections()
    try:
        _update_publish_job(job_id, status="running", message="Pushing to GitHub")
        def _progress(message: str, detail: str = "") -> None:
            _update_publish_job(job_id, status="running", message=message, detail=detail)

        result = _publish_github_project(user, project_id, data, progress=_progress)
        _update_publish_job(
            job_id,
            status="completed",
            message="Push complete" if result.get("status") == "pushed" else result.get("detail", "No changes to commit."),
            result=result,
        )
    except Exception as exc:
        _update_publish_job(
            job_id,
            status="error",
            message="Push failed",
            error=str(exc),
        )
    close_old_connections()


def _run_git(args: list[str], cwd: Optional[Path] = None, timeout: int = 900) -> subprocess.CompletedProcess:
    return _run_git_service(args, cwd=cwd, timeout=timeout)


def _is_git_repo(path: Path) -> bool:
    return _is_git_repo_service(path)


def _is_empty_dir(path: Path) -> bool:
    try:
        return not any(path.iterdir())
    except Exception:
        return False


def _strip_auth_from_url(raw_url: str) -> str:
    return _strip_auth_from_url_service(raw_url)


def _repo_full_name_from_remote(raw_url: str) -> str:
    return _repo_full_name_from_remote_service(raw_url)


def _https_remote_from_any(raw_url: str) -> str:
    return _https_remote_from_any_service(raw_url)


def _scrub_token(message: str, token: Optional[str]) -> str:
    if not message:
        return ""
    cleaned = message
    if token:
        cleaned = cleaned.replace(token, "***")
        cleaned = cleaned.replace(quote(token, safe=""), "***")
    return cleaned


def _git_current_branch(path: Path) -> str:
    return _git_current_branch_service(path)


def _git_remote_origin(path: Path) -> Optional[str]:
    return _git_remote_origin_service(path)


def _git_fetch_with_token(path: Path, *, token_url: Optional[str], restore_url: Optional[str]) -> None:
    return _git_fetch_with_token_service(path, token_url=token_url, restore_url=restore_url)


def _git_checkout_branch(path: Path, branch: str) -> None:
    return _git_checkout_branch_service(path, branch)


def _git_is_clean(path: Path) -> bool:
    return _git_is_clean_service(path)


def _with_token_url(remote_url: str, token: str) -> str:
    return _with_token_url_service(remote_url, token)


def _git_has_identity(path: Path) -> bool:
    name = _run_git(["config", "--get", "user.name"], cwd=path, timeout=5)
    email = _run_git(["config", "--get", "user.email"], cwd=path, timeout=5)
    return bool(name.stdout.strip() and email.stdout.strip())


def _ensure_git_identity(path: Path, username: str) -> None:
    if _git_has_identity(path):
        return
    safe_user = username or "branddozer"
    email = f"{safe_user}@users.noreply.github.com"
    _run_git(["config", "user.name", safe_user], cwd=path, timeout=5)
    _run_git(["config", "user.email", email], cwd=path, timeout=5)


def _with_token_url(remote_url: str, token: str) -> str:
    return _with_token_url_service(remote_url, token)


def _publish_github_project(
    user: Any,
    project_id: str,
    data: Dict[str, Any],
    progress: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    return publish_project_service(user, project_id, data, progress=progress)


def _find_existing_project(root_path: str, repo_url: str) -> Optional[Dict[str, Any]]:
    target_full = _repo_full_name_from_remote(repo_url) if repo_url else ""
    for project in list_projects():
        if project.get("root_path") == root_path:
            return project
        if target_full and _repo_full_name_from_remote(project.get("repo_url") or "") == target_full:
            return project
    return None


def _import_github_project(user, data: Dict[str, Any], job_id: Optional[str] = None) -> Dict[str, Any]:
    def _step(message: str, detail: Optional[str] = None) -> None:
        if not job_id:
            return
        updates: Dict[str, Any] = {"status": "running", "message": message}
        if detail is not None:
            updates["detail"] = detail
        _update_import_job(job_id, **updates)

    _step("Validating request")
    token = data.get("token") or data.get("pat")
    repo_url = data.get("repo_url") or data.get("repository") or ""
    repo_full_name = data.get("repo_full_name") or data.get("full_name") or ""
    branch = data.get("branch") or ""
    project_name = data.get("name")
    destination = data.get("destination")
    default_prompt = data.get("default_prompt") or ""
    username = data.get("username") or data.get("github_username")
    account_id = data.get("account_id") or data.get("github_account_id")
    remember_token = data.get("remember_token", True)

    if repo_full_name and not repo_url:
        repo_url = f"https://github.com/{repo_full_name}.git"
    if not repo_url and not repo_full_name:
        raise ValueError("Repository is required")

    account = None
    if account_id:
        try:
            account = get_github_account(user, account_id, reveal_token=True)
        except ValueError:
            raise ValueError("GitHub account not found")
        token = token or account.token
        username = username or account.username
    else:
        account = get_active_github_account(user, reveal_token=True)
        if account:
            token = token or account.token
            username = username or account.username
    if account and account.has_token and not token:
        raise ValueError(
            "GitHub token could not be unlocked. Re-enter the token under Import from GitHub → Accounts to re-save it."
        )
    if not token:
        raise ValueError("GitHub personal access token is required")
    if shutil.which("git") is None:
        raise ValueError("Git is not available on the server")

    _step("Resolving repository")
    parsed_repo = urlparse(repo_url if "://" in repo_url else f"https://{repo_url}")
    if not parsed_repo.scheme:
        parsed_repo = parsed_repo._replace(scheme="https")
    if not parsed_repo.netloc or "." not in parsed_repo.netloc:
        parsed_repo = urlparse(f"https://github.com/{repo_url.lstrip('/')}")
    if not parsed_repo.netloc:
        raise ValueError("Repository URL is invalid")

    repo_full_name = repo_full_name or _normalize_repo_full_name(parsed_repo.path)
    repo_name = Path(parsed_repo.path).stem or "branddozer-project"
    if not branch and repo_full_name:
        try:
            details = fetch_repo_details(token, repo_full_name)
            branch = details.get("default_branch") or branch
        except Exception:
            branch = branch or ""

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
        raise ValueError("Destination must be inside the home directory")

    base_dir.parent.mkdir(parents=True, exist_ok=True)
    final_dir = base_dir
    created_new_dir = False

    _step("Preparing destination")
    if final_dir.exists() and not _is_git_repo(final_dir):
        if _is_empty_dir(final_dir):
            init_result = _run_git(["init"], cwd=final_dir, timeout=20)
            if init_result.returncode != 0:
                raise ValueError(_scrub_token(init_result.stderr.strip() or "Failed to initialize repository", token))
        else:
            suffix = 1
            while final_dir.exists():
                final_dir = base_dir.parent / f"{base_dir.name}-{suffix}"
                suffix += 1
            created_new_dir = True
    elif not final_dir.exists():
        created_new_dir = True

    clone_base = parsed_repo
    clean_origin = _strip_auth_from_url(clone_base.geturl())
    clone_url = clone_base.geturl()
    if token:
        safe_netloc = f"{quote(token, safe='')}@{clone_base.netloc}"
        clone_url = clone_base._replace(netloc=safe_netloc).geturl()

    repo_owner = username or (repo_full_name.split("/")[0] if repo_full_name and "/" in repo_full_name else None)

    if final_dir.exists() and _is_git_repo(final_dir):
        _step("Updating existing repository")
        sync_warning = ""
        if not _git_is_clean(final_dir):
            sync_warning = "Local changes detected; skipping sync."
        origin_url = _git_remote_origin(final_dir)
        requested_full_name = repo_full_name or _infer_full_name_from_url(repo_url)
        remote_full_name = _repo_full_name_from_remote(origin_url or "")
        if requested_full_name and remote_full_name and requested_full_name != remote_full_name:
            raise ValueError("Destination contains a different repository. Choose another folder.")
        if not sync_warning:
            try:
                _git_fetch_with_token(final_dir, token_url=clone_url, restore_url=clean_origin)
                branch = branch or _git_current_branch(final_dir)
                if branch:
                    _git_checkout_branch(final_dir, branch)
                    remote_sha = _run_git(["rev-parse", f"origin/{branch}"], cwd=final_dir, timeout=10)
                    if remote_sha.returncode != 0:
                        raise ValueError(remote_sha.stderr.strip() or f"Branch {branch} not found on origin")
                    local_sha = _run_git(["rev-parse", "HEAD"], cwd=final_dir, timeout=10)
                    if local_sha.returncode != 0:
                        raise ValueError(local_sha.stderr.strip() or "Failed to resolve local HEAD")
                    if local_sha.stdout.strip() != remote_sha.stdout.strip():
                        merge = _run_git(["merge", "--ff-only", f"origin/{branch}"], cwd=final_dir)
                        if merge.returncode != 0:
                            raise ValueError(merge.stderr.strip() or "Cannot fast-forward to origin")
                        _step("Repository updated")
                    else:
                        _step("Repository already up to date", "Everything is already imported.")
                else:
                    sync_warning = "Unable to determine active branch; using existing files."
            except ValueError as exc:
                sync_warning = _scrub_token(str(exc), token)
        if sync_warning:
            _step("Using existing repository", sync_warning)
        branch = branch or _git_current_branch(final_dir)
    elif final_dir.exists() and not created_new_dir:
        _step("Preparing repository")
        remote_add = _run_git(["remote", "add", "origin", clean_origin], cwd=final_dir, timeout=10)
        if remote_add.returncode != 0:
            raise ValueError(_scrub_token(remote_add.stderr.strip() or "Failed to add origin remote", token))
        try:
            _git_fetch_with_token(final_dir, token_url=clone_url, restore_url=clean_origin)
        except ValueError as exc:
            raise ValueError(_scrub_token(str(exc), token)) from exc
        branch = branch or _git_current_branch(final_dir)
        if branch:
            _git_checkout_branch(final_dir, branch)
        else:
            raise ValueError("Unable to determine default branch for the repository")
    else:
        _step("Cloning repository")
        cmd = ["clone", "--depth", "1", "--single-branch"]
        if branch:
            cmd.extend(["-b", branch])
        cmd.extend([clone_url, str(final_dir)])
        result = _run_git(cmd, cwd=None)
        if result.returncode != 0:
            if created_new_dir and final_dir.exists():
                shutil.rmtree(final_dir, ignore_errors=True)
            raise ValueError(_scrub_token(result.stderr.strip() or "Failed to clone repository", token))
        _run_git(["remote", "set-url", "origin", clean_origin], cwd=final_dir, timeout=10)

    account_id_for_update = account.account_id if account else None
    if remember_token and token and data.get("token"):
        if account_id_for_update:
            upsert_github_account(user, username=repo_owner, token=data.get("token"), account_id=account_id_for_update)
        else:
            upsert_github_account(user, username=repo_owner, token=data.get("token"))
    elif repo_owner and account_id_for_update:
        update_github_account_meta(user, account_id_for_update, username=repo_owner)

    _step("Registering project")
    repo_url_clean = clean_origin or f"https://github.com/{repo_full_name}"
    payload = {
        "name": project_name or final_dir.name,
        "root_path": str(final_dir),
        "default_prompt": default_prompt,
        "interjections": data.get("interjections") or [],
        "interval_minutes": data.get("interval_minutes"),
        "repo_url": repo_url_clean,
        "repo_branch": branch,
    }
    existing = _find_existing_project(str(final_dir), repo_url_clean)
    if existing:
        project = update_project_fields(existing["id"], payload)
    else:
        project = save_project(payload)
    return project


def _run_import_job(job_id: str, user, data: Dict[str, Any]) -> None:
    close_old_connections()
    _update_import_job(job_id, status="running", message="Starting import", detail="")
    token_for_scrub = data.get("token") or data.get("pat")
    if not token_for_scrub:
        account_id = data.get("account_id") or data.get("github_account_id")
        try:
            account = (
                get_github_account(user, account_id, reveal_token=True)
                if account_id
                else get_active_github_account(user, reveal_token=True)
            )
            token_for_scrub = account.token if account else None
        except Exception:
            token_for_scrub = None
    try:
        project = _import_github_project(user, data, job_id=job_id)
        _update_import_job(
            job_id,
            status="completed",
            message="Import complete",
            result={"project": project},
        )
    except Exception as exc:
        _update_import_job(
            job_id,
            status="error",
            message="Import failed",
            error=_scrub_token(str(exc), token_for_scrub),
        )
    finally:
        close_old_connections()


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
            log_message(
                "branddozer.interjections",
                "AI interjection generation failed",
                severity="error",
                details={"project_id": project_id, "error": str(exc)},
            )
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        if interjections:
            update_project_fields(project_id, {"interjections": interjections, "last_ai_generated": int(time.time())})
        return Response({"interjections": interjections}, status=status.HTTP_200_OK)


class ProjectInterjectionPreviewView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, *args, **kwargs) -> Response:
        data = request.data or {}
        default_prompt = (data or {}).get("default_prompt") or ""
        if not str(default_prompt).strip():
            return Response({"detail": "Default prompt is required"}, status=status.HTTP_400_BAD_REQUEST)
        project_name = (data or {}).get("project_name") or "Project"
        api_key = get_integration_value(request.user, "OPENAI_API_KEY", reveal=True) or None
        try:
            interjections = generate_interjections(str(default_prompt), str(project_name), api_key=api_key)
        except Exception as exc:
            log_message(
                "branddozer.interjections",
                "AI interjection preview failed",
                severity="error",
                details={"project_name": str(project_name), "error": str(exc)},
            )
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
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


class ProjectGitHubAccountsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        payload = list_github_accounts(request.user)
        return Response(payload, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        data = request.data or {}
        token = data.get("token") or data.get("pat")
        username = data.get("username") or data.get("login")
        account_id = data.get("account_id")
        label = data.get("label")
        if not token:
            return Response({"detail": "GitHub personal access token is required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            profile = fetch_github_profile(token)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        username = profile.get("login") or username
        payload = upsert_github_account(
            request.user,
            token=token,
            username=username,
            label=label,
            account_id=account_id,
        )
        payload["profile"] = profile
        return Response(payload, status=status.HTTP_201_CREATED if not account_id else status.HTTP_200_OK)


class ProjectGitHubActiveAccountView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, *args, **kwargs) -> Response:
        account_id = (request.data or {}).get("account_id")
        if not account_id:
            return Response({"detail": "Account ID is required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            payload = set_active_github_account(request.user, account_id)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(payload, status=status.HTTP_200_OK)


class ProjectGitHubAccountView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        account = get_active_github_account(request.user, reveal_token=True)
        if not account or not account.token:
            return Response(
                {
                    "connected": False,
                    "username": account.username if account else "",
                    "has_token": bool(account.has_token) if account else False,
                    "account_id": account.account_id if account else "",
                },
                status=status.HTTP_200_OK,
            )
        try:
            profile = fetch_github_profile(account.token)
        except ValueError as exc:
            return Response(
                {
                    "connected": False,
                    "username": account.username or "",
                    "has_token": True,
                    "account_id": account.account_id,
                    "detail": str(exc),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        payload = {
            "connected": True,
            "username": account.username or profile.get("login") or "",
            "has_token": True,
            "profile": profile,
            "account_id": account.account_id,
        }
        return Response(payload, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        data = request.data or {}
        token = data.get("token") or data.get("pat")
        username = data.get("username") or data.get("login")
        account_id = data.get("account_id")
        if not token:
            return Response({"detail": "GitHub personal access token is required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            profile = fetch_github_profile(token)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        username = profile.get("login") or username
        payload = upsert_github_account(
            request.user,
            token=token,
            username=username,
            account_id=account_id,
        )
        active = payload.get("active_account") or {}
        response_payload = {
            "connected": True,
            "username": active.get("username") or username or "",
            "has_token": bool(active.get("has_token")),
            "profile": profile,
            "account_id": active.get("id") or payload.get("active_id") or "",
        }
        return Response(response_payload, status=status.HTTP_201_CREATED if not account_id else status.HTTP_200_OK)


class ProjectGitHubRepoListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        username = request.query_params.get("username") or None
        account_id = request.query_params.get("account_id") or request.query_params.get("account")
        try:
            account = (
                get_github_account(request.user, account_id, reveal_token=True)
                if account_id
                else get_active_github_account(request.user, reveal_token=True)
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        if not account or not account.token:
            return Response({"detail": "GitHub token is required. Connect your account first."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            repos = list_github_repos(account.token, username=username)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(
            {
                "repos": repos,
                "count": len(repos),
                "username": username or account.username,
                "account_id": account.account_id,
            },
            status=status.HTTP_200_OK,
        )


class ProjectGitHubBranchListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        repo_full_name = request.query_params.get("repo") or request.query_params.get("full_name")
        if not repo_full_name:
            return Response({"detail": "Repository full name is required"}, status=status.HTTP_400_BAD_REQUEST)
        account_id = request.query_params.get("account_id") or request.query_params.get("account")
        try:
            account = (
                get_github_account(request.user, account_id, reveal_token=True)
                if account_id
                else get_active_github_account(request.user, reveal_token=True)
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        if not account or not account.token:
            return Response({"detail": "GitHub token is required. Connect your account first."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            branches = list_repo_branches(account.token, repo_full_name)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"branches": branches, "account_id": account.account_id}, status=status.HTTP_200_OK)


class ProjectGitHubImportView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, *args, **kwargs) -> Response:
        data = request.data or {}
        async_mode = bool(data.get("async")) or bool(data.get("job"))
        if async_mode:
            payload = dict(data)
            token_value = payload.get("token") or payload.get("pat")
            if token_value and payload.get("remember_token", True):
                username = payload.get("username") or payload.get("github_username")
                account_id = payload.get("account_id") or payload.get("github_account_id")
                accounts_payload = upsert_github_account(request.user, username=username, token=token_value, account_id=account_id)
                payload["account_id"] = accounts_payload.get("active_id") or account_id
                payload.pop("token", None)
                payload.pop("pat", None)
            if payload.get("github_account_id") and not payload.get("account_id"):
                payload["account_id"] = payload.get("github_account_id")
            job = enqueue_job(
                kind="github_import",
                user=request.user,
                payload=payload,
                message="Queued",
            )
            return Response({"job_id": str(job.id), "status": "queued"}, status=status.HTTP_202_ACCEPTED)
        try:
            project = _import_github_project(request.user, data)
        except Exception as exc:
            token_for_scrub = data.get("token") or data.get("pat")
            if not token_for_scrub:
                account_id = data.get("account_id") or data.get("github_account_id")
                try:
                    account = (
                        get_github_account(request.user, account_id, reveal_token=True)
                        if account_id
                        else get_active_github_account(request.user, reveal_token=True)
                    )
                    token_for_scrub = account.token if account else None
                except Exception:
                    token_for_scrub = None
            return Response({"detail": _scrub_token(str(exc), token_for_scrub)}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"project": project}, status=status.HTTP_201_CREATED)


class ProjectGitHubImportStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, job_id: str, *args, **kwargs) -> Response:
        job = _get_import_job(job_id, request.user)
        if not job:
            return Response({"detail": "Import job not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(job, status=status.HTTP_200_OK)


class ProjectGitHubPublishView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, project_id: str, *args, **kwargs) -> Response:
        data = request.data or {}
        use_async = str(data.get("async", "true")).strip().lower() not in {"0", "false", "no", "off"}
        if use_async:
            project = BrandProject.objects.filter(id=project_id).first()
            if not project:
                return Response({"detail": "Project not found"}, status=status.HTTP_404_NOT_FOUND)
            payload = dict(data)
            if payload.get("github_account_id") and not payload.get("account_id"):
                payload["account_id"] = payload.get("github_account_id")
            job = enqueue_job(
                kind="github_publish",
                project=project,
                user=request.user,
                payload=payload,
                message="Queued",
            )
            return Response({"job_id": str(job.id), "status": "queued"}, status=status.HTTP_202_ACCEPTED)
        try:
            result = _publish_github_project(request.user, project_id, data)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(result, status=status.HTTP_200_OK)


class ProjectGitHubPublishStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, job_id: str, *args, **kwargs) -> Response:
        job = _get_publish_job(job_id, request.user)
        if not job:
            return Response({"detail": "Publish job not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(job, status=status.HTTP_200_OK)
