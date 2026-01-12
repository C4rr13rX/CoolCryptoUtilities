from __future__ import annotations

import os
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote, urlparse
from uuid import uuid4

import requests
from django.db import transaction
from django.utils.text import slugify

from services.branddozer_state import get_project, update_project_fields
from securevault.models import SecureSetting
from services.secure_settings import decrypt_secret, encrypt_secret

GITHUB_API = "https://api.github.com"
GITHUB_CATEGORY = "branddozer"
GITHUB_TOKEN_NAME = "GITHUB_PAT"
GITHUB_USER_NAME = "GITHUB_USERNAME"
GITHUB_ACCOUNTS_NAME = "GITHUB_ACCOUNTS"
GITHUB_ACTIVE_NAME = "GITHUB_ACTIVE"
GITHUB_TOKEN_PREFIX = "GITHUB_PAT:"
USER_AGENT = "BrandDozer/1.0"
TIMEOUT = 10


@dataclass
class GitHubAuth:
    username: Optional[str]
    token: Optional[str]
    has_token: bool


@dataclass
class GitHubAccount:
    account_id: str
    username: Optional[str]
    token: Optional[str]
    has_token: bool
    label: Optional[str] = None


def _headers(token: Optional[str]) -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _next_link(link_header: Optional[str]) -> Optional[str]:
    if not link_header:
        return None
    matches = re.findall(r'<([^>]+)>; rel="next"', link_header)
    return matches[0] if matches else None


def _get_setting(user, name: str) -> Optional[SecureSetting]:
    return SecureSetting.objects.filter(user=user, category=GITHUB_CATEGORY, name=name).first()


def _token_setting_name(account_id: str) -> str:
    return f"{GITHUB_TOKEN_PREFIX}{account_id}"


def _load_accounts(user) -> List[Dict[str, Any]]:
    setting = _get_setting(user, GITHUB_ACCOUNTS_NAME)
    if not setting or not setting.value_plain:
        return []
    try:
        payload = json.loads(setting.value_plain)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    accounts = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        account_id = str(entry.get("id") or "").strip()
        if not account_id:
            continue
        accounts.append(
            {
                "id": account_id,
                "username": (entry.get("username") or "").strip() or None,
                "label": (entry.get("label") or "").strip() or None,
            }
        )
    return accounts


def _save_accounts(user, accounts: List[Dict[str, Any]]) -> None:
    normalized = []
    for entry in accounts:
        account_id = str(entry.get("id") or "").strip()
        if not account_id:
            continue
        normalized.append(
            {
                "id": account_id,
                "username": (entry.get("username") or "").strip() or None,
                "label": (entry.get("label") or "").strip() or None,
            }
        )
    with transaction.atomic():
        setting, _ = SecureSetting.objects.get_or_create(
            user=user,
            category=GITHUB_CATEGORY,
            name=GITHUB_ACCOUNTS_NAME,
            defaults={"is_secret": False},
        )
        setting.is_secret = False
        setting.value_plain = json.dumps(normalized)
        setting.ciphertext = None
        setting.encapsulated_key = None
        setting.nonce = None
        setting.save()


def _get_active_account_id(user) -> Optional[str]:
    setting = _get_setting(user, GITHUB_ACTIVE_NAME)
    if not setting or not setting.value_plain:
        return None
    return setting.value_plain.strip() or None


def _set_active_account_id(user, account_id: str) -> None:
    with transaction.atomic():
        setting, _ = SecureSetting.objects.get_or_create(
            user=user,
            category=GITHUB_CATEGORY,
            name=GITHUB_ACTIVE_NAME,
            defaults={"is_secret": False},
        )
        setting.is_secret = False
        setting.value_plain = account_id
        setting.ciphertext = None
        setting.encapsulated_key = None
        setting.nonce = None
        setting.save()


def _store_account_token(user, account_id: str, token: str) -> None:
    payload = encrypt_secret(token)
    with transaction.atomic():
        setting, _ = SecureSetting.objects.get_or_create(
            user=user,
            category=GITHUB_CATEGORY,
            name=_token_setting_name(account_id),
            defaults={"is_secret": True},
        )
        setting.is_secret = True
        setting.value_plain = None
        setting.ciphertext = payload["ciphertext"]
        setting.encapsulated_key = payload["encapsulated_key"]
        setting.nonce = payload["nonce"]
        setting.save()


def _account_has_token(user, account_id: str) -> bool:
    return bool(_get_setting(user, _token_setting_name(account_id)))


def _get_account_token(user, account_id: str, *, reveal_token: bool = False) -> Optional[str]:
    setting = _get_setting(user, _token_setting_name(account_id))
    if not setting:
        return None
    if not reveal_token:
        return None
    if setting.is_secret:
        try:
            return decrypt_secret(setting.encapsulated_key, setting.ciphertext, setting.nonce)
        except Exception:
            return None
    return setting.value_plain or None


def _bootstrap_accounts_from_legacy(user) -> List[Dict[str, Any]]:
    accounts = _load_accounts(user)
    if accounts:
        return accounts
    legacy = get_saved_github_auth(user, reveal_token=True)
    if not legacy.has_token and not legacy.username:
        return []
    account_id = uuid4().hex
    accounts = [
        {
            "id": account_id,
            "username": legacy.username,
            "label": legacy.username or "GitHub",
        }
    ]
    _save_accounts(user, accounts)
    _set_active_account_id(user, account_id)
    if legacy.token:
        _store_account_token(user, account_id, legacy.token)
    return accounts


def list_github_accounts(user) -> Dict[str, Any]:
    accounts = _bootstrap_accounts_from_legacy(user)
    if not accounts:
        accounts = _load_accounts(user)
    active_id = _get_active_account_id(user)
    if accounts and (not active_id or not any(acc["id"] == active_id for acc in accounts)):
        active_id = accounts[0]["id"]
        _set_active_account_id(user, active_id)
    annotated = []
    for acc in accounts:
        has_token = _account_has_token(user, acc["id"])
        token_value = _get_account_token(user, acc["id"], reveal_token=True) if has_token else None
        annotated.append(
            {
                **acc,
                "has_token": has_token,
                "token_locked": bool(has_token and not token_value),
            }
        )
    active = next((acc for acc in annotated if acc["id"] == active_id), None)
    return {"accounts": annotated, "active_id": active_id, "active_account": active}


def get_github_account(user, account_id: str, *, reveal_token: bool = False) -> GitHubAccount:
    accounts = _bootstrap_accounts_from_legacy(user)
    if not accounts:
        accounts = _load_accounts(user)
    account = next((acc for acc in accounts if acc["id"] == account_id), None)
    if not account:
        raise ValueError("GitHub account not found")
    token = _get_account_token(user, account_id, reveal_token=reveal_token)
    return GitHubAccount(
        account_id=account_id,
        username=account.get("username"),
        token=token,
        has_token=_account_has_token(user, account_id),
        label=account.get("label"),
    )


def get_active_github_account(user, *, reveal_token: bool = False) -> Optional[GitHubAccount]:
    payload = list_github_accounts(user)
    active_id = payload.get("active_id")
    if not active_id:
        return None
    return get_github_account(user, active_id, reveal_token=reveal_token)


def upsert_github_account(
    user,
    *,
    token: Optional[str],
    username: Optional[str],
    label: Optional[str] = None,
    account_id: Optional[str] = None,
) -> Dict[str, Any]:
    accounts = _bootstrap_accounts_from_legacy(user)
    if not accounts:
        accounts = _load_accounts(user)
    account = next((acc for acc in accounts if acc["id"] == account_id), None) if account_id else None
    if not account:
        if not token:
            raise ValueError("GitHub token is required")
        account_id = uuid4().hex
        account = {"id": account_id, "username": username, "label": label or username or "GitHub"}
        accounts.append(account)
    else:
        if username:
            account["username"] = username
        if label:
            account["label"] = label
    if token:
        _store_account_token(user, account_id, token)
    _save_accounts(user, accounts)
    _set_active_account_id(user, account_id)
    return list_github_accounts(user)


def update_github_account_meta(user, account_id: str, *, username: Optional[str] = None, label: Optional[str] = None) -> None:
    accounts = _load_accounts(user)
    if not accounts:
        accounts = _bootstrap_accounts_from_legacy(user)
    account = next((acc for acc in accounts if acc["id"] == account_id), None)
    if not account:
        return
    if username:
        account["username"] = username
    if label:
        account["label"] = label
    _save_accounts(user, accounts)


def set_active_github_account(user, account_id: str) -> Dict[str, Any]:
    accounts = list_github_accounts(user).get("accounts") or []
    if not any(acc["id"] == account_id for acc in accounts):
        raise ValueError("GitHub account not found")
    _set_active_account_id(user, account_id)
    return list_github_accounts(user)


def get_saved_github_auth(user, *, reveal_token: bool = False) -> GitHubAuth:
    token_setting = _get_setting(user, GITHUB_TOKEN_NAME)
    username_setting = _get_setting(user, GITHUB_USER_NAME)
    has_token = bool(token_setting)
    token: Optional[str] = None
    if reveal_token and token_setting:
        if token_setting.is_secret:
            try:
                token = decrypt_secret(token_setting.encapsulated_key, token_setting.ciphertext, token_setting.nonce)
            except Exception:
                token = None
        else:
            token = token_setting.value_plain or None
    username = username_setting.value_plain if username_setting else None
    if username:
        username = username.strip() or None
    return GitHubAuth(username=username, token=token, has_token=has_token)


def store_github_auth(user, *, username: Optional[str], token: Optional[str]) -> GitHubAuth:
    normalized_username = (username or "").strip() or None
    normalized_token = (token or "").strip() or None
    with transaction.atomic():
        if normalized_username is not None:
            username_setting, _ = SecureSetting.objects.get_or_create(
                user=user,
                category=GITHUB_CATEGORY,
                name=GITHUB_USER_NAME,
                defaults={"is_secret": False},
            )
            username_setting.is_secret = False
            username_setting.value_plain = normalized_username
            username_setting.ciphertext = None
            username_setting.encapsulated_key = None
            username_setting.nonce = None
            username_setting.save()
        if normalized_token is not None:
            token_setting, _ = SecureSetting.objects.get_or_create(
                user=user,
                category=GITHUB_CATEGORY,
                name=GITHUB_TOKEN_NAME,
                defaults={"is_secret": True},
            )
            token_setting.is_secret = True
            payload = encrypt_secret(normalized_token)
            token_setting.value_plain = None
            token_setting.ciphertext = payload["ciphertext"]
            token_setting.encapsulated_key = payload["encapsulated_key"]
            token_setting.nonce = payload["nonce"]
            token_setting.save()
    return get_saved_github_auth(user, reveal_token=False)


def fetch_github_profile(token: str) -> Dict[str, Any]:
    resp = requests.get(f"{GITHUB_API}/user", headers=_headers(token), timeout=TIMEOUT)
    if resp.status_code == 401:
        raise ValueError("GitHub token is invalid or expired")
    if resp.status_code == 403:
        raise ValueError("GitHub rate limit hit; try again shortly")
    resp.raise_for_status()
    data = resp.json() or {}
    return {
        "login": data.get("login"),
        "name": data.get("name"),
        "avatar_url": data.get("avatar_url"),
        "html_url": data.get("html_url"),
    }


def list_github_repos(token: str, username: Optional[str] = None) -> List[Dict[str, Any]]:
    repos: List[Dict[str, Any]] = []
    url = f"{GITHUB_API}/user/repos" if not username else f"{GITHUB_API}/users/{username}/repos"
    params = {"per_page": 100, "sort": "updated", "affiliation": "owner,collaborator"}
    headers = _headers(token)
    while url:
        resp = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
        params = None  # only for first request
        if resp.status_code == 401:
            raise ValueError("GitHub token rejected while listing repositories")
        resp.raise_for_status()
        for entry in resp.json() or []:
            repos.append(
                {
                    "id": entry.get("id"),
                    "name": entry.get("name"),
                    "full_name": entry.get("full_name"),
                    "private": entry.get("private"),
                    "description": entry.get("description"),
                    "default_branch": entry.get("default_branch"),
                    "updated_at": entry.get("updated_at"),
                    "clone_url": entry.get("clone_url"),
                    "ssh_url": entry.get("ssh_url"),
                    "owner": (entry.get("owner") or {}).get("login"),
                }
            )
        url = _next_link(resp.headers.get("Link"))
    return repos


def list_repo_branches(token: str, repo_full_name: str) -> List[Dict[str, str]]:
    url = f"{GITHUB_API}/repos/{repo_full_name}/branches"
    headers = _headers(token)
    params = {"per_page": 100}
    branches: List[Dict[str, str]] = []
    while url:
        resp = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
        params = None
        if resp.status_code == 404:
            raise ValueError("Repository not found on GitHub")
        if resp.status_code == 401:
            raise ValueError("GitHub token rejected while listing branches")
        resp.raise_for_status()
        for entry in resp.json() or []:
            branches.append(
                {
                    "name": entry.get("name"),
                    "protected": bool(entry.get("protected")),
                }
            )
        url = _next_link(resp.headers.get("Link"))
    return branches


def fetch_repo_details(token: str, repo_full_name: str) -> Dict[str, Any]:
    resp = requests.get(f"{GITHUB_API}/repos/{repo_full_name}", headers=_headers(token), timeout=TIMEOUT)
    if resp.status_code == 404:
        raise ValueError("Repository not found on GitHub")
    if resp.status_code == 401:
        raise ValueError("GitHub token rejected while fetching repository")
    resp.raise_for_status()
    data = resp.json() or {}
    return {
        "full_name": data.get("full_name") or repo_full_name,
        "default_branch": data.get("default_branch"),
        "description": data.get("description"),
        "clone_url": data.get("clone_url"),
        "ssh_url": data.get("ssh_url"),
    }


def create_github_repo(
    token: str,
    *,
    name: str,
    description: Optional[str] = None,
    private: bool = True,
) -> Dict[str, Any]:
    payload = {
        "name": name,
        "description": description or "",
        "private": private,
        "auto_init": False,
    }
    resp = requests.post(f"{GITHUB_API}/user/repos", headers=_headers(token), json=payload, timeout=TIMEOUT)
    if resp.status_code == 401:
        raise ValueError("GitHub token rejected while creating repository")
    if resp.status_code == 403:
        raise ValueError(
            "GitHub denied repository creation. Ensure the selected account has permission to create repos and the token "
            "includes `repo` (or `public_repo`) scope. If this is an org or SSO-protected account, approve the token in "
            "GitHub → Settings → Developer settings → Personal access tokens."
        )
    if resp.status_code == 422:
        raise ValueError("Repository name already exists for this GitHub user")
    resp.raise_for_status()
    data = resp.json() or {}
    return {
        "full_name": data.get("full_name"),
        "default_branch": data.get("default_branch") or "main",
        "description": data.get("description"),
        "clone_url": data.get("clone_url"),
        "html_url": data.get("html_url"),
        "private": data.get("private"),
    }


_GIT_TIMEOUT = 900


def _normalize_repo_full_name(raw_path: str) -> str:
    cleaned = raw_path.strip().rstrip("/")
    if cleaned.lower().endswith(".git"):
        cleaned = cleaned[:-4]
    segments = [segment for segment in cleaned.split("/") if segment]
    if len(segments) >= 2:
        return f"{segments[-2]}/{segments[-1]}"
    return ""


def _infer_full_name_from_url(repo_url: str) -> str:
    parsed_repo = urlparse(repo_url if "://" in repo_url else f"https://{repo_url}")
    return _normalize_repo_full_name(parsed_repo.path)


def _github_permission_help() -> str:
    return (
        "Ensure the selected GitHub account has access to the repository and the token includes `repo` "
        "(or `public_repo`) scope. If the account uses SSO or org permissions, approve the token under "
        "GitHub → Settings → Developer settings → Personal access tokens."
    )


def _git_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_SSH_COMMAND"] = "ssh -o StrictHostKeyChecking=no"
    return env


def _run_git(args: List[str], cwd: Optional[Path] = None, timeout: int = _GIT_TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        env=_git_env(),
        timeout=timeout,
    )


def _is_git_repo(path: Path) -> bool:
    result = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=path, timeout=10)
    return result.returncode == 0


def _git_is_repo(path: Path) -> bool:
    return _is_git_repo(path)


def _strip_auth_from_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if "@" not in parsed.netloc:
        return parsed.geturl()
    netloc = parsed.netloc.split("@", 1)[1]
    return parsed._replace(netloc=netloc).geturl()


def _repo_full_name_from_remote(raw_url: str) -> str:
    if raw_url.startswith("git@") and ":" in raw_url:
        path = raw_url.split(":", 1)[1]
        return _normalize_repo_full_name(path)
    parsed = urlparse(raw_url if "://" in raw_url else f"https://{raw_url}")
    return _normalize_repo_full_name(parsed.path)


def _https_remote_from_any(raw_url: str) -> str:
    if not raw_url:
        return ""
    if raw_url.startswith("git@") and ":" in raw_url:
        path = raw_url.split(":", 1)[1]
        return f"https://github.com/{_normalize_repo_full_name(path)}.git"
    parsed = urlparse(raw_url if "://" in raw_url else f"https://{raw_url}")
    if parsed.scheme and parsed.netloc:
        return parsed.geturl()
    return f"https://github.com/{_normalize_repo_full_name(parsed.path)}.git"


def _git_current_branch(path: Path) -> str:
    result = _run_git(["symbolic-ref", "--short", "HEAD"], cwd=path, timeout=10)
    if result.returncode == 0:
        branch = result.stdout.strip()
        if branch:
            return branch
    result = _run_git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], cwd=path, timeout=10)
    if result.returncode == 0:
        origin_head = result.stdout.strip()
        if origin_head.startswith("origin/"):
            return origin_head.split("/", 1)[1]
    return ""


def _git_remote_origin(path: Path) -> Optional[str]:
    result = _run_git(["remote", "get-url", "origin"], cwd=path, timeout=10)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


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
    parsed = urlparse(remote_url)
    if not parsed.scheme:
        parsed = urlparse(f"https://{remote_url}")
    if parsed.scheme in {"http", "https"}:
        safe_netloc = f"{quote(token, safe='')}@{parsed.netloc}"
        return parsed._replace(netloc=safe_netloc).geturl()
    return remote_url


def _git_commit_all(path: Path, message: str) -> bool:
    status = _run_git(["status", "--porcelain"], cwd=path, timeout=10)
    if status.returncode != 0:
        raise ValueError(status.stderr.strip() or "Failed to check git status")
    if not status.stdout.strip():
        return False
    add = _run_git(["add", "-A"], cwd=path, timeout=60)
    if add.returncode != 0:
        raise ValueError(add.stderr.strip() or "Failed to stage changes")
    commit = _run_git(["commit", "-m", message], cwd=path, timeout=120)
    if commit.returncode != 0:
        raise ValueError(commit.stderr.strip() or "Failed to commit changes")
    return True


def publish_project(
    user: Any,
    project_id: str,
    data: Dict[str, Any],
    progress: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    project = get_project(project_id)
    if not project:
        raise ValueError("Not found")

    def _step(message: str, detail: str = "") -> None:
        if not progress:
            return
        try:
            progress(message, detail)
        except Exception:
            return

    commit_message = (data.get("message") or "Update from BrandDozer").strip()
    account_id = data.get("account_id") or data.get("github_account_id")
    private = bool(data.get("private", True))
    repo_name = (data.get("repo_name") or slugify(project.get("name") or "") or "").strip()
    root = Path(project.get("root_path") or "")

    if not root.exists():
        raise ValueError("Project path does not exist on the server")
    if shutil.which("git") is None:
        raise ValueError("Git is not available on the server")

    _step("Loading GitHub account")
    try:
        account = (
            get_github_account(user, account_id, reveal_token=True)
            if account_id
            else get_active_github_account(user, reveal_token=True)
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if not account:
        accounts_payload = list_github_accounts(user)
        if accounts_payload.get("accounts"):
            raise ValueError(
                "No active GitHub account selected. Choose an account under Import from GitHub → Accounts, then try again."
            )
        raise ValueError(
            "No GitHub account connected. Add a personal access token under Import from GitHub → Accounts, then try again."
        )
    if not account.token:
        if account.has_token:
            raise ValueError(
                "GitHub token could not be unlocked. Re-enter the token under Import from GitHub → Accounts to re-save it."
            )
        raise ValueError(
            "No GitHub account connected. Add a personal access token under Import from GitHub → Accounts, "
            "then select it and try again."
        )

    token = account.token
    username = account.username
    if not username:
        try:
            profile = fetch_github_profile(token)
            username = profile.get("login") or username
        except ValueError:
            username = username or "branddozer"

    _step("Preparing repository")
    if not _is_git_repo(root):
        init = _run_git(["init"], cwd=root, timeout=30)
        if init.returncode != 0:
            raise ValueError(init.stderr.strip() or "Failed to initialize git repository")

    _ensure_git_identity(root, username or "branddozer")

    origin = _git_remote_origin(root)
    repo_url = project.get("repo_url") or ""
    if not origin:
        if repo_url:
            origin = _https_remote_from_any(repo_url)
            remote_add = _run_git(["remote", "add", "origin", origin], cwd=root, timeout=10)
            if remote_add.returncode != 0:
                raise ValueError(remote_add.stderr.strip() or "Failed to add origin remote")
        else:
            _step("Creating repository on GitHub")
            if not repo_name:
                repo_name = slugify(root.name) or "branddozer-project"
            try:
                created = create_github_repo(
                    token,
                    name=repo_name,
                    description=project.get("name") or repo_name,
                    private=private,
                )
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            origin = created.get("clone_url") or f"https://github.com/{username}/{repo_name}.git"
            remote_add = _run_git(["remote", "add", "origin", origin], cwd=root, timeout=10)
            if remote_add.returncode != 0:
                raise ValueError(remote_add.stderr.strip() or "Failed to add origin remote")
            repo_url = created.get("html_url") or origin

    branch = (data.get("branch") or project.get("repo_branch") or _git_current_branch(root) or "main").strip()
    if branch:
        _run_git(["checkout", "-B", branch], cwd=root, timeout=10)

    clean_origin = _strip_auth_from_url(_https_remote_from_any(origin))
    token_url = _with_token_url(clean_origin, token)
    ahead_behind: Optional[Dict[str, int]] = None
    try:
        _git_fetch_with_token(root, token_url=token_url, restore_url=clean_origin)
        ahead_behind = _git_ahead_behind(root, branch)
    except Exception:
        ahead_behind = None

    _step("Committing changes")
    try:
        committed = _git_commit_all(root, commit_message)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    _step("Pushing to GitHub")
    push = _run_git(["push", "-u", token_url, branch], cwd=root, timeout=_GIT_TIMEOUT)
    if push.returncode != 0:
        message = push.stderr.strip() or "Failed to push to GitHub"
        lowered = message.lower()
        if "permission" in lowered or "403" in lowered or "authentication" in lowered:
            message = f"GitHub rejected the push. {_github_permission_help()}"
        raise ValueError(_scrub_token(message, token))

    _run_git(["remote", "set-url", "origin", clean_origin], cwd=root, timeout=10)
    update_project_fields(project_id, {"repo_url": repo_url or clean_origin, "repo_branch": branch})
    output = f"{push.stdout}\n{push.stderr}".lower()
    if "everything up-to-date" in output or "everything up to date" in output:
        detail = "No changes to commit or push."
        if ahead_behind:
            detail = f"{detail} Local ahead: {ahead_behind['ahead']}, behind: {ahead_behind['behind']}."
        return {
            "status": "no_changes",
            "detail": detail,
            "repo_url": repo_url or clean_origin,
            "branch": branch,
            "ahead": ahead_behind["ahead"] if ahead_behind else None,
            "behind": ahead_behind["behind"] if ahead_behind else None,
        }
    if not committed:
        return {
            "status": "pushed",
            "detail": "Pushed existing commits.",
            "repo_url": repo_url or clean_origin,
            "branch": branch,
            "ahead": ahead_behind["ahead"] if ahead_behind else None,
            "behind": ahead_behind["behind"] if ahead_behind else None,
        }
    return {
        "status": "pushed",
        "repo_url": repo_url or clean_origin,
        "branch": branch,
        "ahead": ahead_behind["ahead"] if ahead_behind else None,
        "behind": ahead_behind["behind"] if ahead_behind else None,
    }


def _git_ahead_behind(path: Path, branch: str) -> Optional[Dict[str, int]]:
    if not branch:
        return None
    result = _run_git(["rev-list", "--left-right", "--count", f"origin/{branch}...{branch}"], cwd=path, timeout=10)
    if result.returncode != 0:
        return None
    parts = result.stdout.strip().split()
    if len(parts) < 2:
        return None
    try:
        behind = int(parts[0])
        ahead = int(parts[1])
    except ValueError:
        return None
    return {"ahead": ahead, "behind": behind}


def _git_fetch_with_token(path: Path, *, token_url: Optional[str], restore_url: Optional[str]) -> None:
    result = _run_git(["fetch", "origin"], cwd=path)
    if result.returncode == 0:
        return
    if not token_url:
        raise ValueError(result.stderr.strip() or "Failed to fetch repository")
    original_url = _git_remote_origin(path)
    if not original_url and restore_url:
        _run_git(["remote", "add", "origin", restore_url], cwd=path, timeout=10)
        original_url = restore_url
    try:
        _run_git(["remote", "set-url", "origin", token_url], cwd=path, timeout=10)
        retry = _run_git(["fetch", "origin"], cwd=path)
        if retry.returncode != 0:
            raise ValueError(retry.stderr.strip() or "Failed to fetch repository with token")
    finally:
        if original_url:
            _run_git(["remote", "set-url", "origin", original_url], cwd=path, timeout=10)


def _git_checkout_branch(path: Path, branch: str) -> None:
    if not branch:
        return
    exists = _run_git(["show-ref", "--verify", f"refs/heads/{branch}"], cwd=path, timeout=10)
    if exists.returncode == 0:
        result = _run_git(["checkout", branch], cwd=path)
        if result.returncode != 0:
            raise ValueError(result.stderr.strip() or f"Failed to checkout branch {branch}")
        return
    result = _run_git(["checkout", "-B", branch, f"origin/{branch}"], cwd=path)
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or f"Branch {branch} not found on origin")


def _git_is_clean(path: Path) -> bool:
    result = _run_git(["status", "--porcelain"], cwd=path, timeout=10)
    return result.returncode == 0 and not result.stdout.strip()
