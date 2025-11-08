#!/usr/bin/env python3
"""
Utility helper to interact with the Codex CLI with persistent transcripts.

Usage:
  python tools/codex_session.py --session my-session --prompt "Summarize logs"

In code:
  from tools.codex_session import CodexSession
  session = CodexSession("my-session")
  response = session.send("Fix the latest warning in production.log")
"""

from __future__ import annotations

import argparse
import os
import pty
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence


class CodexSession:
    def __init__(
        self,
        session_name: str,
        executable: str = "codex",
        transcript_dir: str | Path = "codex_transcripts",
        read_timeout_s: float = 10.0,
    ) -> None:
        self.session_name = session_name
        self.executable = executable
        self.read_timeout_s = read_timeout_s

        self.transcript_dir = Path(transcript_dir)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = self.transcript_dir / f"{session_name}.log"

        self._codex_available = shutil.which(self.executable) is not None
        self._help_cache: Optional[str] = None
        if self._codex_available:
            try:
                proc = subprocess.run(
                    [self.executable, "--help"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self._help_cache = (proc.stdout or "") + (proc.stderr or "")
            except Exception:
                self._help_cache = ""

    # ------------------------ Public API ------------------------

    def send(self, prompt: str, *, stream: bool = False) -> str:
        """
        Send a prompt to the `codex` CLI and return its response.
        Strategy:
          1) Try positional-arg mode:  codex "<prompt>"
          2) If that fails or complains "stdout is not a terminal", retry by piping stdin
          3) If it still complains about TTY, retry inside a PTY (pseudo-terminal)
          4) Log everything to transcripts/<session>.log
        """
        if not self._codex_available:
            resp = "[codex missing] Install or place `codex` on PATH."
            self._append_transcript(prompt, resp)
            if stream:
                print(resp)
            return resp

        # Attempt 1: positional argument
        resp, rc, err = self._run_simple([self.executable, prompt])
        if rc == 0 and resp.strip():
            self._append_transcript(prompt, resp)
            if stream:
                print(resp)
            return resp

        # Attempt 2: stdin pipe
        resp2, rc2, err2 = self._run_simple([self.executable], input_text=prompt)
        if rc2 == 0 and resp2.strip():
            self._append_transcript(prompt, resp2)
            if stream:
                print(resp2)
            return resp2

        # Attempt 3: PTY fallback (handles CLIs that require a TTY)
        # Try with stdin-first style
        resp3, rc3 = self._run_with_pty([self.executable], input_text=prompt)
        if rc3 == 0 and resp3.strip():
            self._append_transcript(prompt, resp3)
            if stream:
                print(resp3)
            return resp3

        # Final attempt: PTY with positional arg as well
        resp4, rc4 = self._run_with_pty([self.executable, prompt], input_text=None)
        if rc4 == 0 and resp4.strip():
            self._append_transcript(prompt, resp4)
            if stream:
                print(resp4)
            return resp4

        # If all attempts failed, surface the most informative error we saw
        msg = (err2 or err or "").strip() or "codex returned no output"
        final = f"[codex error] {msg}"
        self._append_transcript(prompt, final)
        if stream:
            print(final)
        return final

    # ------------------------ Internals ------------------------

    def _run_simple(
        self, cmd: Sequence[str], input_text: Optional[str] = None
    ) -> tuple[str, int, str]:
        try:
            proc = subprocess.run(
                list(cmd),
                input=input_text,
                capture_output=True,
                text=True,
                check=False,
            )
            stdout = (proc.stdout or "")
            stderr = (proc.stderr or "")
            return stdout, proc.returncode, stderr
        except FileNotFoundError:
            return "", 127, "codex executable not found"
        except Exception as e:
            return "", 1, f"exec error: {e!r}"

    def _run_with_pty(
        self, cmd: Sequence[str], input_text: Optional[str]
    ) -> tuple[str, int]:
        """
        Run a command under a pseudo-terminal so CLIs that require a TTY
        (complain 'stdout is not a terminal') can work.
        """
        # Allocate PTY
        master_fd, slave_fd = pty.openpty()

        # Spawn the child attached to the PTY
        try:
            child = subprocess.Popen(
                list(cmd),
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                text=False,  # raw bytes; we'll decode after
                close_fds=True,
            )
        except Exception as e:
            os.close(master_fd)
            os.close(slave_fd)
            return f"[pty spawn error] {e!r}", 1

        # We write input (if any) to the PTY, then send Ctrl-D to signal EOF
        output_chunks: list[bytes] = []
        try:
            if input_text is not None:
                os.write(master_fd, input_text.encode("utf-8", errors="replace"))
                os.write(master_fd, b"\n")
            # Send EOF (Ctrl-D) to signal we're done providing input
            try:
                os.write(master_fd, b"\x04")
            except OSError:
                pass

            # Read until timeout and process exit
            end_time = time.time() + self.read_timeout_s
            while True:
                # Break if child exited and no more data
                if child.poll() is not None and time.time() > end_time:
                    break

                rlist, _, _ = select.select([master_fd], [], [], 0.1)
                if rlist:
                    try:
                        chunk = os.read(master_fd, 4096)
                    except OSError:
                        break
                    if not chunk:
                        # EOF
                        if child.poll() is not None:
                            break
                    else:
                        output_chunks.append(chunk)

                # Extend timeout a bit while data is flowing
                if output_chunks:
                    end_time = time.time() + 0.5

            rc = child.wait(timeout=1.0)
        except Exception:
            # If anything goes wrong, try to terminate the child
            try:
                child.terminate()
            except Exception:
                pass
            try:
                rc = child.wait(timeout=0.5)
            except Exception:
                rc = 1
        finally:
            try:
                os.close(master_fd)
            except Exception:
                pass
            try:
                os.close(slave_fd)
            except Exception:
                pass

        stdout = b"".join(output_chunks).decode("utf-8", errors="replace")
        return stdout, rc

    def _append_transcript(self, prompt: str, response: str) -> None:
        divider = "=" * 80
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.transcript_path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"{divider}\nTIMESTAMP: {ts}\nSESSION: {self.session_name}\n"
                f"PROMPT:\n{prompt}\nRESPONSE:\n{response}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a prompt to the Codex CLI and log the transcript."
    )
    parser.add_argument("--session", required=True, help="Local transcript session name/ID to append to.")
    parser.add_argument("--prompt", required=True, help="Prompt text to send.")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print the Codex response directly (in addition to returning it).",
    )
    parser.add_argument(
        "--exe",
        default="codex",
        help="Path or name of the Codex CLI executable (default: codex).",
    )
    parser.add_argument(
        "--outdir",
        default="codex_transcripts",
        help="Directory for transcript logs (default: codex_transcripts).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per-call read timeout for PTY fallback (seconds).",
    )
    args = parser.parse_args()

    session = CodexSession(
        args.session, executable=args.exe, transcript_dir=args.outdir, read_timeout_s=args.timeout
    )
    response = session.send(args.prompt, stream=args.stream)
    if not args.stream:
        print(response)


if __name__ == "__main__":
    main()
