# tools/codex_session.py
from __future__ import annotations

import os
import pty
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def codex_default_settings() -> dict[str, Any]:
    """
    Resolve Codex CLI settings from environment defaults.
    """
    return {
        "model": os.getenv("CODEX_MODEL", "gpt-5.1-codex-max"),
        "reasoning_effort": os.getenv("CODEX_REASONING_EFFORT", "xhigh"),
        "sandbox_mode": os.getenv("CODEX_SANDBOX_MODE", "danger-full-access"),
        "approval_policy": os.getenv("CODEX_APPROVAL_POLICY", "never"),
        "bypass_sandbox_confirm": _env_bool("CODEX_BYPASS_APPROVALS", True),
    }


class CodexSession:
    """
    Headless wrapper for the `codex` CLI, suitable for in-process use:
      - Non-interactive attempts: --input, --prompt, positional arg, stdin
      - Interactive TUI fallback via PTY with progressive keystroke strategy:
          prompt⏎  -> (wait) -> ⏎  -> /run⏎  -> /review⏎  -> /status⏎
      - Replies to DSR ESC[6n with ESC[1;1R (avoid "cursor position" errors)
      - Sets TERM/COLUMNS/LINES; streams live; logs transcripts
    """

    def __init__(
        self,
        session_name: str,
        executable: str = "codex",
        transcript_dir: str | Path = "codex_transcripts",
        read_timeout_s: float | None = None,
        stream_default: bool = True,
        verbose_default: bool = False,
        term_rows: int = 40,
        term_cols: int = 120,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        sandbox_mode: Optional[str] = None,
        approval_policy: Optional[str] = None,
        bypass_sandbox_confirm: Optional[bool] = None,
        workdir: str | Path | None = None,
    ) -> None:
        self.session_name = session_name
        self.executable = executable
        self.read_timeout_s = None if read_timeout_s is None else float(read_timeout_s)
        self.stream_default = stream_default
        self.verbose_default = verbose_default
        self.term_rows = term_rows
        self.term_cols = term_cols

        self.transcript_dir = Path(transcript_dir)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = self.transcript_dir / f"{session_name}.log"

        defaults = codex_default_settings()
        self.model = model if model is not None else defaults["model"]
        self.model_reasoning_effort = reasoning_effort if reasoning_effort is not None else defaults["reasoning_effort"]
        self.sandbox_mode = sandbox_mode if sandbox_mode is not None else defaults["sandbox_mode"]
        self.approval_policy = approval_policy if approval_policy is not None else defaults["approval_policy"]
        self.bypass_sandbox_confirm = (
            bypass_sandbox_confirm if bypass_sandbox_confirm is not None else defaults["bypass_sandbox_confirm"]
        )
        self._cli_prefix = self._build_cli_prefix()
        self.workdir = Path(workdir).resolve() if workdir else None
        self._stream_callback = None

        self._codex_available = shutil.which(self.executable) is not None

    # ===== Public API ==========================================================

    def send(
        self,
        prompt: str,
        *,
        stream: Optional[bool] = None,
        verbose: Optional[bool] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        images: Optional[Sequence[str]] = None,
    ) -> str:
        stream = self.stream_default if stream is None else stream
        verbose = self.verbose_default if verbose is None else verbose
        prev_callback = self._stream_callback
        if stream_callback is not None:
            self._stream_callback = stream_callback
        pty_attempted = False
        cli_prefix = [*self._cli_prefix, *self._image_args(images)]

        if not self._codex_available:
            resp = "[codex missing] Install or place `codex` on PATH."
            self._append_transcript(prompt, resp)
            if stream:
                self._print(resp + "\n")
            self._stream_callback = prev_callback
            return resp

        # 1) Try non-interactive flags first (some builds support these)
        # Prefer the non-interactive exec subcommand when available (streams stderr/stdout live)
        exec_out, exec_rc, exec_err = self._run_exec_stream(prompt, stream=stream, verbose=verbose, cli_prefix=cli_prefix)
        exec_text = exec_out.strip()
        exec_err_text = exec_err.strip()
        if exec_rc != 127 and (exec_text or exec_err_text):
            combined = exec_text or exec_err_text
            if exec_rc == 0 or not self._looks_like_exec_unsupported(combined):
                self._append_transcript(
                    prompt,
                    exec_out if not exec_err_text else f"{exec_out}\n[stderr]\n{exec_err}",
                )
                if not stream:
                    self._print(combined + ("\n" if not combined.endswith("\n") else ""))
                self._stream_callback = prev_callback
                return combined

        if stream:
            final, rc5 = self._run_streaming_pty(cli_prefix, prompt, verbose=verbose)
            pty_attempted = True
            if final.strip():
                self._append_transcript(prompt, final)
                self._stream_callback = prev_callback
                return final

        out, rc, _ = self._run_simple([*cli_prefix, "--input", prompt])
        if rc == 0 and out.strip():
            self._append_transcript(prompt, out)
            if stream:
                self._print(out)
            self._stream_callback = prev_callback
            return out

        out2, rc2, _ = self._run_simple([*cli_prefix, "--prompt", prompt])
        if rc2 == 0 and out2.strip():
            self._append_transcript(prompt, out2)
            if stream:
                self._print(out2)
            self._stream_callback = prev_callback
            return out2

        out3, rc3, _ = self._run_simple([*cli_prefix, prompt])
        if rc3 == 0 and out3.strip():
            self._append_transcript(prompt, out3)
            if stream:
                self._print(out3)
            self._stream_callback = prev_callback
            return out3

        out4, rc4, _ = self._run_simple([*cli_prefix], input_text=prompt)
        if rc4 == 0 and out4.strip():
            self._append_transcript(prompt, out4)
            if stream:
                self._print(out4)
            self._stream_callback = prev_callback
            return out4

        # 2) Interactive PTY fallback with progressive actions
        if not pty_attempted:
            final, rc5 = self._run_streaming_pty(cli_prefix, prompt, verbose=verbose)
            if final.strip():
                self._append_transcript(prompt, final)
                self._stream_callback = prev_callback
                return final

        # Last resort: PTY with positional arg
        final2, rc6 = self._run_streaming_pty([*cli_prefix, prompt], None, verbose=verbose)
        self._append_transcript(prompt, final2 if final2.strip() else f"[codex error] exit {rc6}")
        self._stream_callback = prev_callback
        return final2

    # ===== Simple runners ======================================================

    def _run_simple(self, cmd: Sequence[str], input_text: Optional[str] = None) -> tuple[str, int, str]:
        try:
            proc = subprocess.run(
                list(cmd),
                input=input_text,
                capture_output=True,
                text=True,
                check=False,
                cwd=str(self.workdir) if self.workdir else None,
            )
            return (proc.stdout or ""), proc.returncode, (proc.stderr or "")
        except FileNotFoundError:
            return "", 127, "codex executable not found"
        except Exception as e:
            return "", 1, f"exec error: {e!r}"

    # ===== PTY streaming with tiny emulator & progressive strategy =============

    def _run_streaming_pty(
        self, cmd: Sequence[str], prompt_or_none: Optional[str], *, verbose: bool
    ) -> Tuple[str, int]:
        if verbose:
            self._print(f"[codex] exec: {' '.join(map(self._q, cmd))}\n")

        env = os.environ.copy()
        env.setdefault("TERM", "xterm-256color")
        env.setdefault("COLUMNS", str(self.term_cols))
        env.setdefault("LINES", str(self.term_rows))
        env.setdefault("PAGER", "cat")

        master_fd, slave_fd = pty.openpty()
        try:
            child = subprocess.Popen(
                list(cmd),
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                text=False,
                close_fds=True,
                env=env,
                cwd=str(self.workdir) if self.workdir else None,
            )
        except Exception as e:
            for fd in (master_fd, slave_fd):
                try:
                    os.close(fd)
                except Exception:
                    pass
            return f"[pty spawn error] {e!r}", 1

        captured: list[bytes] = []
        total_bytes = 0
        control_buf = bytearray()

        # State machine for progressive actions
        banner_seen = False
        # Build the progressive keystroke script
        actions: list[tuple[str, list[bytes]]] = []
        if prompt_or_none is not None:
            actions.append(
                ("prompt", [prompt_or_none.encode("utf-8", errors="replace"), b"\r"])
            )
        actions.extend(
            [
                ("newline", [b"\r"]),
                ("/run", [b"/run\r"]),
                ("/review", [b"/review\r"]),
                ("/status", [b"/status\r"]),
                ("/exit", [b"/exit\r"]),
            ]
        )
        step = 0
        last_action_time = 0.0
        action_interval = 1.0  # seconds between retries after no new output
        start_time = time.time()

        def _log_action(label: str) -> None:
            if verbose:
                self._print(f"[codex] sent {label}\n")

        def do_action():
            nonlocal step, last_action_time
            if step >= len(actions):
                return
            label, payloads = actions[step]
            try:
                for payload in payloads:
                    os.write(master_fd, payload)
                _log_action(label)
                step += 1
                last_action_time = time.time()
            except OSError:
                pass

        timed_out = False
        try:
            timeout_disabled = self.read_timeout_s is None or self.read_timeout_s <= 0
            end_time = None if timeout_disabled else time.time() + self.read_timeout_s
            max_deadline = None if timeout_disabled else time.time() + max(self.read_timeout_s, 5.0)
            last_total = 0

            while True:
                now = time.time()
                if not timeout_disabled and max_deadline is not None and now > max_deadline:
                    timed_out = True
                    try:
                        child.terminate()
                    except Exception:
                        pass
                    try:
                        child.wait(timeout=1.0)
                    except Exception:
                        try:
                            child.kill()
                        except Exception:
                            pass
                    break
                if child.poll() is not None and not timeout_disabled and end_time is not None and now > end_time:
                    break

                r, _, _ = select.select([master_fd], [], [], 0.05)
                if r:
                    try:
                        chunk = os.read(master_fd, 4096)
                    except OSError:
                        break
                    if not chunk:
                        if child.poll() is not None:
                            break
                    else:
                        printable = self._terminal_emulator_step(master_fd, chunk, control_buf)
                        if printable:
                            if not banner_seen:
                                if (b"OpenAI Codex" in printable) or (b"To get started" in printable) or printable.strip():
                                    banner_seen = True
                            captured.append(printable)
                            total_bytes += len(printable)
                            self._print(printable.decode("utf-8", errors="replace"))

                if child.poll() is not None and not r and total_bytes == last_total:
                    break

                # Progressive actions once banner is visible
                ready_for_actions = banner_seen or (total_bytes > 0) or ((time.time() - start_time) > 1.5)
                if ready_for_actions and step < len(actions):
                    if step == 0 and (time.time() - last_action_time) > 0.2:
                        do_action()  # prompt⏎ as soon as CLI looks ready
                    elif (time.time() - last_action_time) >= action_interval:
                        # If no new output since last action, escalate
                        if total_bytes == last_total:
                            do_action()
                        else:
                            # extend patience when output is flowing
                            last_action_time = time.time()

                # Update pacing / timeout
                if total_bytes > last_total:
                    last_total = total_bytes
                    if not timeout_disabled:
                        end_time = time.time() + 2.0  # keep waiting while output grows
                        max_deadline = time.time() + max(self.read_timeout_s, 5.0)

            # Try a graceful EOF to flush final output
            try:
                os.write(master_fd, b"\x04")
            except OSError:
                pass

            rc = child.wait(timeout=1.0)
        except Exception:
            try:
                child.terminate()
            except Exception:
                pass
            try:
                rc = child.wait(timeout=0.5)
            except Exception:
                rc = 1
        finally:
            for fd in (master_fd, slave_fd):
                try:
                    os.close(fd)
                except Exception:
                    pass

        stdout = b"".join(captured).decode("utf-8", errors="replace")
        if timed_out and not stdout.strip():
            return "[codex timeout]", 124
        return stdout, rc

    # ===== Non-interactive exec runner with live streaming =====================

    def _run_exec_stream(
        self,
        prompt: str,
        *,
        stream: bool,
        verbose: bool,
        cli_prefix: Optional[Sequence[str]] = None,
    ) -> Tuple[str, int, str]:
        prefix = list(cli_prefix) if cli_prefix is not None else list(self._cli_prefix)
        cmd = [*prefix, "exec", "--color", "never", "-"]
        if verbose:
            self._print(f"[codex] exec-stream: {' '.join(map(self._q, cmd))}\n")
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                close_fds=True,
                cwd=str(self.workdir) if self.workdir else None,
            )
        except FileNotFoundError:
            return "", 127, "codex executable not found"
        except Exception as exc:
            return "", 1, f"exec-stream spawn error: {exc!r}"

        try:
            if proc.stdin is not None:
                data = prompt.encode("utf-8", errors="replace")
                proc.stdin.write(data)
                proc.stdin.close()
        except Exception:
            pass

        stdout_buf = bytearray()
        stderr_buf = bytearray()
        fd_map: dict[int, str] = {}

        def _register(pipe: Optional[int], label: str) -> None:
            if pipe is not None:
                fd_map[pipe] = label

        stdout_fd = proc.stdout.fileno() if proc.stdout is not None else None
        stderr_fd = proc.stderr.fileno() if proc.stderr is not None else None
        _register(stdout_fd, "stdout")
        _register(stderr_fd, "stderr")

        timeout_disabled = self.read_timeout_s is None or self.read_timeout_s <= 0
        deadline = None if timeout_disabled else time.time() + self.read_timeout_s
        timed_out = False
        while fd_map:
            try:
                ready, _, _ = select.select(list(fd_map.keys()), [], [], 0.1)
            except (InterruptedError, OSError):
                ready = []
            if not timeout_disabled and deadline is not None and time.time() > deadline and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
                timed_out = True
                break
            if not ready:
                if proc.poll() is None:
                    continue
                # child exited; drain remaining pipes
                ready = list(fd_map.keys())
            for fd in ready:
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    chunk = b""
                if not chunk:
                    fd_map.pop(fd, None)
                    continue
                target = fd_map.get(fd)
                if target == "stdout":
                    stdout_buf.extend(chunk)
                else:
                    stderr_buf.extend(chunk)
                if stream:
                    self._print(chunk.decode("utf-8", errors="replace"))
                if not timeout_disabled:
                    deadline = time.time() + self.read_timeout_s
        try:
            try:
                rc = proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    rc = proc.wait(timeout=1.0)
                except Exception:
                    rc = 1

            if timed_out:
                rc = 124
                if not stdout_buf and not stderr_buf:
                    stderr_buf.extend(b"[codex timeout]")
        finally:
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()

        return (
            stdout_buf.decode("utf-8", errors="replace"),
            rc,
            stderr_buf.decode("utf-8", errors="replace"),
        )

    # ===== Minimal terminal emulation =========================================

    def _terminal_emulator_step(self, master_fd: int, chunk: bytes, control_buf: bytearray) -> bytes:
        """
        Intercepts & answers a few terminal control queries.
         - DSR (cursor position) request: ESC [ 6 n  ->  reply ESC [ 1 ; 1 R
        Returns bytes to display (filters the raw query).
        """
        control_buf.extend(chunk)
        out = bytearray()
        i = 0
        while i < len(control_buf):
            b = control_buf[i]
            if b == 0x1B:  # ESC
                if i + 3 < len(control_buf) and control_buf[i + 1] == ord('['):
                    # DSR cursor report?
                    if control_buf[i + 2] == ord('6') and control_buf[i + 3] == ord('n'):
                        i += 4
                        try:
                            os.write(master_fd, b"\x1b[1;1R")
                        except OSError:
                            pass
                        continue
                    # Pass other CSI sequences through (up to final byte 0x40..0x7E)
                    j = i + 2
                    while j < len(control_buf) and not (0x40 <= control_buf[j] <= 0x7E):
                        j += 1
                    if j < len(control_buf):
                        out.extend(control_buf[i : j + 1])
                        i = j + 1
                        continue
                    else:
                        break  # incomplete CSI; wait for more
                else:
                    if i + 1 >= len(control_buf):
                        break
                    out.extend(control_buf[i : i + 2])
                    i += 2
                    continue
            else:
                out.append(b)
                i += 1

        if i:
            del control_buf[:i]
        return bytes(out)

    # ===== Utilities ===========================================================

    def _append_transcript(self, prompt: str, response: str) -> None:
        divider = "=" * 80
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.transcript_path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"{divider}\nTIMESTAMP: {ts}\nSESSION: {self.session_name}\n"
                f"PROMPT:\n{prompt}\nRESPONSE:\n{response}\n"
            )

    def _print(self, s: str) -> None:
        if self._stream_callback:
            try:
                self._stream_callback(s)
            except Exception:
                pass
        sys.stdout.write(s)
        sys.stdout.flush()

    def _looks_like_exec_unsupported(self, text: str) -> bool:
        lower = text.lower()
        if "exec" not in lower:
            return False
        needles = (
            "unknown command",
            "unknown subcommand",
            "no such command",
            "unrecognized arguments",
            "invalid choice",
        )
        return any(needle in lower for needle in needles)

    def _build_cli_prefix(self) -> list[str]:
        parts = [self.executable]
        if self.model:
            parts.extend(["--model", self.model])
        if self.model_reasoning_effort:
            parts.extend(["-c", f'model_reasoning_effort="{self.model_reasoning_effort}"'])
        if self.sandbox_mode:
            parts.extend(["--sandbox", self.sandbox_mode])
        if self.bypass_sandbox_confirm:
            parts.append("--dangerously-bypass-approvals-and-sandbox")
        elif self.approval_policy:
            parts.extend(["--ask-for-approval", self.approval_policy])
        return parts

    @staticmethod
    def _image_args(images: Optional[Sequence[str]]) -> list[str]:
        if not images:
            return []
        args: list[str] = []
        for image in images:
            if not image:
                continue
            path = Path(str(image)).expanduser()
            if path.exists():
                args.extend(["--image", str(path)])
        return args

    @staticmethod
    def _q(x: str) -> str:
        return x if all(c.isalnum() or c in "-_./:" for c in x) else repr(x)
