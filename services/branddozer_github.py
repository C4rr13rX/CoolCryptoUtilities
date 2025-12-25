from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from django.db import transaction

from securevault.models import SecureSetting
from services.secure_settings import decrypt_secret, encrypt_secret

GITHUB_API = "https://api.github.com"
GITHUB_CATEGORY = "branddozer"
GITHUB_TOKEN_NAME = "GITHUB_PAT"
GITHUB_USER_NAME = "GITHUB_USERNAME"
USER_AGENT = "BrandDozer/1.0"
TIMEOUT = 10


@dataclass
class GitHubAuth:
    username: Optional[str]
    token: Optional[str]
    has_token: bool


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
