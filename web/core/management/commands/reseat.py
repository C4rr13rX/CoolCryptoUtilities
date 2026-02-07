from __future__ import annotations

import os
import subprocess
import shutil
import sys
from pathlib import Path

from django.core.management import BaseCommand, call_command



class Command(BaseCommand):
    help = "Build frontend, apply migrations, collect static, then runserver (in that order)."
    requires_system_checks = []

    def add_arguments(self, parser):
        parser.add_argument(
            "--noinstall",
            action="store_true",
            help="Skip npm install step (default runs npm install before build)",
        )
        parser.add_argument(
            "--runserver-args",
            nargs="*",
            default=None,
            help="Additional arguments to pass through to runserver (e.g. --runserver-args 0.0.0.0:8001)",
        )
        parser.add_argument(
            "--guardian-off",
            action="store_true",
            help="Disable guardian/production auto-start even if enabled in settings.",
        )
        parser.add_argument(
            "--production-off",
            action="store_true",
            help="Disable production manager auto-start even if enabled in settings.",
        )
        parser.add_argument(
            "--branddozer-worker-off",
            action="store_true",
            help="Disable BrandDozer background worker auto-start.",
        )
        parser.add_argument(
            "--no-runserver",
            action="store_true",
            help="Skip launching the Django dev server after setup.",
        )

    def handle(self, *args, **options):
        project_root = Path(__file__).resolve().parents[4]
        frontend_dir = project_root / "web" / "frontend"

        def ensure_venv():
            venv_root = project_root / ".venv"
            if os.name == "nt":
                venv_python = venv_root / "Scripts" / "python.exe"
            else:
                venv_python = venv_root / "bin" / "python"
            try:
                current = Path(sys.executable).resolve()
            except Exception:
                current = None
            if venv_python.exists() and current and current != venv_python.resolve():
                self.stdout.write(
                    self.style.WARNING(f"Switching to venv python: {venv_python}")
                )
                os.execv(
                    str(venv_python),
                    [str(venv_python), *sys.argv],
                )

        def resolve_cmd(cmd):
            if not cmd:
                return cmd
            exe = cmd[0]
            if os.name == "nt":
                found = shutil.which(exe)
                if found:
                    if str(found).lower().endswith(".ps1"):
                        return [
                            "powershell",
                            "-NoProfile",
                            "-ExecutionPolicy",
                            "Bypass",
                            "-File",
                            found,
                            *cmd[1:],
                        ]
                    return [found, *cmd[1:]]
                for candidate in (f"{exe}.cmd", f"{exe}.exe", f"{exe}.bat"):
                    path = shutil.which(candidate)
                    if path:
                        return [path, *cmd[1:]]
                if exe.lower() == "npm":
                    roots = [
                        os.getenv("ProgramFiles"),
                        os.getenv("ProgramFiles(x86)"),
                        os.getenv("LocalAppData"),
                        os.getenv("APPDATA"),
                    ]
                    extra = [
                        Path("C:/Program Files/nodejs/npm.cmd"),
                        Path("C:/Program Files (x86)/nodejs/npm.cmd"),
                    ]
                    for root in filter(None, roots):
                        extra.extend(
                            [
                                Path(root) / "nodejs" / "npm.cmd",
                                Path(root) / "npm" / "npm.cmd",
                                Path(root) / "nvm" / "npm.cmd",
                                Path(root) / "NVM" / "npm.cmd",
                                Path(root) / "nvs" / "nodejs" / "npm.cmd",
                            ]
                        )
                    for candidate in extra:
                        try:
                            if candidate.exists():
                                return [str(candidate), *cmd[1:]]
                        except Exception:
                            continue
            else:
                path = shutil.which(exe)
                if path:
                    return [path, *cmd[1:]]
            return cmd

        def run(cmd, cwd=None):
            resolved = resolve_cmd(cmd)
            self.stdout.write(self.style.HTTP_INFO(f"$ {' '.join(resolved)}"))
            subprocess.run(resolved, cwd=cwd, check=True)

        def ensure_python_deps():
            req_main = project_root / "requirements.txt"
            req_legacy = project_root / "requirements_legacy.txt"
            self.stdout.write(self.style.MIGRATE_HEADING("[1/7] Python dependencies"))
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "--version"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception:
                run([sys.executable, "-m", "ensurepip", "--upgrade"])
            try:
                run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "-r",
                        str(req_main),
                    ],
                    cwd=str(project_root),
                )
            except subprocess.CalledProcessError:
                if req_legacy.exists():
                    self.stdout.write(
                        self.style.WARNING(
                            "Primary requirements failed, retrying with requirements_legacy.txt"
                        )
                    )
                    run(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--disable-pip-version-check",
                            "-r",
                            str(req_legacy),
                        ],
                        cwd=str(project_root),
                    )
                else:
                    raise

        # Respect disable flags before Django finishes loading apps that auto-start services.
        if options.get("guardian_off"):
            os.environ["GUARDIAN_AUTO_DISABLED"] = "1"
        if options.get("production_off"):
            os.environ["PRODUCTION_AUTO_DISABLED"] = "1"

        ensure_venv()
        ensure_python_deps()
        call_command("check")

        self.stdout.write(self.style.MIGRATE_HEADING("[2/7] Frontend build"))
        if not options["noinstall"]:
            run(["npm", "install"], cwd=str(frontend_dir))
        run(["npm", "run", "build"], cwd=str(frontend_dir))

        self.stdout.write(self.style.MIGRATE_HEADING("[3/7] Make migrations"))
        call_command("makemigrations")

        self.stdout.write(self.style.MIGRATE_HEADING("[4/7] Apply migrations"))
        call_command("migrate")

        self.stdout.write(self.style.MIGRATE_HEADING("[5/7] Collect static"))
        call_command("collectstatic", interactive=False, clear=True)

        worker_process = None
        if not options.get("branddozer_worker_off") and os.environ.get("BRANDDOZER_WORKER_DISABLED") != "1":
            self.stdout.write(self.style.MIGRATE_HEADING("[6/7] Start BrandDozer worker"))
            worker_env = os.environ.copy()
            worker_env.setdefault("SECURE_VAULT_KEY_DIR", str(project_root / "storage" / "secure_vault"))
            worker_process = subprocess.Popen(
                [sys.executable, "web/manage.py", "branddozer_worker"],
                cwd=str(project_root),
                env=worker_env,
            )

        if options.get("no_runserver"):
            self.stdout.write(self.style.MIGRATE_HEADING("[7/7] Runserver skipped"))
        else:
            self.stdout.write(self.style.MIGRATE_HEADING("[7/7] Runserver"))
            runserver_args = options.get("runserver_args") or []
            if all(arg != "--noreload" for arg in runserver_args):
                runserver_args.append("--noreload")
            try:
                call_command("runserver", *runserver_args)
            finally:
                if worker_process and worker_process.poll() is None:
                    worker_process.terminate()
