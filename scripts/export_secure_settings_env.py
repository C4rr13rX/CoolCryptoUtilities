#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _setup_django(db_path: str | None, key_dir: str | None) -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")
    os.environ.setdefault("DJANGO_PREFER_SQLITE_FALLBACK", "1")
    os.environ.setdefault("DJANGO_DB_VENDOR", "sqlite")
    os.environ.setdefault("ALLOW_SQLITE_FALLBACK", "1")
    if db_path:
        os.environ["DJANGO_SQLITE_PATH"] = db_path
    if key_dir:
        os.environ["SECURE_VAULT_KEY_DIR"] = key_dir
    from services.env_loader import EnvLoader

    EnvLoader.load()
    import django

    django.setup()


def _format_env_value(value: str) -> str:
    if "\n" in value:
        return '"' + value.replace('"', '\\"').replace("\n", "\\n") + '"'
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Secure Settings to a .env file (best-effort).")
    parser.add_argument("--username", required=True, help="Django username to export.")
    parser.add_argument("--db-path", help="Optional SQLite DB path (defaults to settings).")
    parser.add_argument("--key-dir", help="Optional secure vault key dir.")
    parser.add_argument("--output", help="Write to file instead of stdout.")
    args = parser.parse_args()

    _setup_django(args.db_path, args.key_dir)
    from django.contrib.auth import get_user_model
    from securevault.models import SecureSetting
    from services.secure_settings import decrypt_secret

    User = get_user_model()
    user = User.objects.filter(username=args.username).first()
    if not user:
        print("User not found", file=sys.stderr)
        return 1

    settings = SecureSetting.objects.filter(user=user).order_by("category", "name")
    lines: list[str] = []
    last_category = None
    for setting in settings:
        if setting.is_secret:
            try:
                value = decrypt_secret(setting.encapsulated_key, setting.ciphertext, setting.nonce)
            except Exception:
                value = ""
        else:
            value = setting.value_plain or ""
        if not value:
            continue
        category = (setting.category or "default").lower()
        if category != last_category:
            if lines:
                lines.append("")
            lines.append(f"# [{category}]")
            last_category = category
        lines.append(f"{setting.name}={_format_env_value(value)}")

    output = ("\n".join(lines).strip() + "\n") if lines else ""
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
