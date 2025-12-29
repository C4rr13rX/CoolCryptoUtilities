from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests
from django.db import transaction

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
