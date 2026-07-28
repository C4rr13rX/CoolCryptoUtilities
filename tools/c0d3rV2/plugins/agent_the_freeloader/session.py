from __future__ import annotations

import os
import json
import time
import uuid
import re
from pathlib import Path
from typing import Any

from .adapters import has_credential
from .models import PoolLimit, load_catalog, merge_pool_limits
from .quota import QuotaLedger
from .router import FreeloaderRouter


class AgentTheFreeloaderSession:
    """C0d3rV2-compatible session backed by quota-aware free models."""

    MODEL_ID = "agent-the-freeloader"

    def __init__(
        self,
        session_name: str = "agent-the-freeloader",
        transcript_dir: str | Path | None = None,
        *,
        workdir: str | Path = "",
        catalog_path: str | Path | None = None,
        state_path: str | Path | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        timeout_s: float = 60.0,
        max_attempts: int = 6,
        allowed_models: list[str] | set[str] | tuple[str, ...] | None = None,
        transcript_enabled: bool = True,
        **_ignored: Any,
    ) -> None:
        from tools.ai_backend_mode import activate_freeloader_mode

        activate_freeloader_mode()
        self.session_name = session_name
        self.transcript_dir = Path(transcript_dir) if transcript_dir else None
        self.transcript_enabled = transcript_enabled
        self.workdir = Path(workdir).resolve() if workdir else Path.cwd()
        configured_max_tokens = os.getenv("AGENT_FREELOADER_MAX_TOKENS", "").strip()
        self.max_tokens = int(configured_max_tokens or max_tokens)
        self.temperature = float(temperature)
        self._session_id = str(uuid.uuid4())
        self.last_error = ""
        self._turn_calls = 0
        self._turn_call_budget = 0
        self.route_history: list[list[dict]] = []
        from .local_verifier import LocalCorrectionVerifier
        self._local_verifier = LocalCorrectionVerifier()
        specs = load_catalog(catalog_path)
        limits = merge_pool_limits(specs)
        # OpenRouter grants only 50 free requests/day until the account has
        # purchased at least $10 in credits.  Default conservatively; users who
        # qualify can opt into the documented 1,000/day pool.
        if "openrouter:free" in limits:
            purchased = os.getenv("AGENT_FREELOADER_OPENROUTER_1K_RPD", "").lower() in {"1", "true", "yes"}
            old = limits["openrouter:free"]
            limits["openrouter:free"] = PoolLimit(
                requests_per_minute=old.requests_per_minute or 20,
                requests_per_day=1000 if purchased else 50,
                requests_per_month=old.requests_per_month,
                tokens_per_minute=old.tokens_per_minute,
                tokens_per_day=old.tokens_per_day,
            )
        configured_state = os.getenv("AGENT_FREELOADER_STATE_PATH", "").strip()
        project_root = Path(__file__).resolve().parents[4]
        resolved_state = (
            Path(state_path)
            if state_path
            else Path(configured_state).expanduser()
            if configured_state
            else project_root / "runtime" / "agent_the_freeloader" / "quota.json"
        )
        self.ledger = QuotaLedger(limits, state_path=resolved_state)
        self.router = FreeloaderRouter(
            specs,
            self.ledger,
            max_attempts=max_attempts,
            timeout_s=float(os.getenv("AGENT_FREELOADER_TIMEOUT_S", "").strip() or timeout_s),
            allowed_models=set(allowed_models or ()),
        )

    def send(
        self,
        prompt: str,
        *,
        stream: bool = False,
        system: str = "",
        stream_callback: Any = None,
        **kwargs: Any,
    ) -> str:
        budget = int(getattr(self, "_turn_call_budget", 0) or 0)
        calls = int(getattr(self, "_turn_calls", 0) or 0)
        if budget and calls >= budget:
            self.last_error = f"ATF turn model-call budget exhausted ({budget})"
            raise RuntimeError(self.last_error)
        self._turn_calls = calls + 1
        phase = self._phase_name(system)
        phase_sticky = getattr(self, "_phase_sticky", {})
        preferred_identity = str(phase_sticky.get(phase) or "")
        phase_counts = getattr(self, "_phase_model_counts", {})
        excluded = set(getattr(self, "_turn_banned", set()))
        atomic_implementation = "atomic implementation policy" in system.lower()
        default_affinity = "2" if atomic_implementation else "4"
        affinity_limit = max(1, int(os.getenv("ATF_PHASE_MODEL_CALLS", default_affinity)))
        if preferred_identity and int(phase_counts.get((phase, preferred_identity), 0)) >= affinity_limit:
            excluded.add(preferred_identity)
        try:
            reply = self.router.send(
                prompt,
                system=system,
                max_tokens=int(kwargs.get("max_tokens") or self.max_tokens),
                temperature=float(kwargs.get("temperature") or self.temperature),
                preferred_identity=preferred_identity,
                excluded_identities=excluded,
            )
        except Exception as exc:
            self.last_error = str(exc)
            self.route_history.append(self._phase_trace(system))
            raise
        self.last_error = ""
        self.route_history.append(self._phase_trace(system))
        selected = [item for item in self.router.last_trace if item.get("outcome") == "selected"]
        if selected:
            item = selected[-1]
            phase_sticky[phase] = f"{item.get('provider', '')}:{item.get('model', '')}"
            self._phase_sticky = phase_sticky
            count_key = (phase, phase_sticky[phase])
            phase_counts[count_key] = int(phase_counts.get(count_key, 0)) + 1
            self._phase_model_counts = phase_counts
        self._append_transcript(prompt, system, reply)
        if stream and stream_callback and reply:
            stream_callback(reply)
        return reply

    def begin_turn(self, max_calls: int = 0) -> None:
        """Reset the all-inclusive call budget for one C0d3rV2 turn."""
        self._turn_calls = 0
        self._turn_call_budget = max(0, int(max_calls))
        self.last_error = ""
        self._phase_sticky: dict[str, str] = {}
        self._phase_model_counts: dict[tuple[str, str], int] = {}
        self._turn_banned: set[str] = set()

    def get_model_id(self) -> str:
        return self.router.last_model_id

    def report_outcome(self, *, success: bool, reason: str = "") -> None:
        """Feed validation results back into persistent model ranking."""
        selected = [
            item
            for trace in self.route_history
            for item in trace
            if item.get("outcome") == "selected"
        ]
        if not selected:
            return
        # The final serving model is most responsible for the final action.
        item = selected[-1]
        self.router.report_outcome(
            str(item.get("provider") or ""),
            str(item.get("model") or ""),
            success=success,
            reason=reason,
        )

    def report_correction(
        self,
        *,
        classification: str,
        trigger: str,
        failed_output: str = "",
        correction: str = "",
        resolved: bool = False,
        is_hallucination: bool = True,
        metadata: dict | None = None,
        origin_attribution: dict | None = None,
    ) -> int | None:
        """Persist a correction event against the model that produced the bad step."""
        selected = [item for item in self.router.last_trace if item.get("outcome") == "selected"]
        item = dict(origin_attribution or (selected[-1] if selected else {}))
        provider = str(item.get("provider") or "")
        model_id = str(item.get("model") or "")
        if not provider or not model_id:
            return None
        event_id = self.router.feedback.record_correction(
            provider,
            model_id,
            session_name=self.session_name,
            classification=classification,
            is_hallucination=is_hallucination,
            trigger=trigger,
            failed_output=failed_output,
            correction=correction,
            resolved=resolved,
            metadata=metadata,
        )
        if is_hallucination:
            self.router.report_outcome(provider, model_id, success=False, reason=f"{classification}: {trigger}")
            identity = f"{provider}:{model_id}"
            self._phase_sticky = {
                phase: selected_identity
                for phase, selected_identity in getattr(self, "_phase_sticky", {}).items()
                if selected_identity != identity
            }
            self._turn_banned = set(getattr(self, "_turn_banned", set())) | {identity}
        return event_id

    def correction_guidance(self, context: str = "", limit: int = 8) -> str:
        """Return compact recent failure memory for prompt-level error prevention."""
        events = self.router.feedback.correction_snapshot(limit=max(1, limit * 8))
        events = self._local_verifier.rank(context or "recent model errors", events, limit=limit)
        lines: list[str] = []
        for event in events:
            if len(lines) >= limit:
                break
            bad_edit = str(event.get("failed_output") or "").strip().replace("\n", " ")[:180]
            lines.append(
                f"- {event['provider']}/{event['model']} [{event['classification']}]: "
                f"{event['trigger'][:240]} -> {event['correction'][:160] or 'unresolved'}"
                + (f"; rejected edit: {bad_edit}" if bad_edit else "")
            )
        if not lines:
            return ""
        return ("[Recent ATF correction memory — do not repeat these errors]\n" + "\n".join(lines))[:2500]

    def _append_transcript(self, prompt: str, system: str, reply: str) -> None:
        if not self.transcript_enabled or not self.transcript_dir:
            return
        try:
            self.transcript_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            safe_session = re.sub(r"[^A-Za-z0-9._-]+", "-", self.session_name).strip("-._") or "atf-session"
            path = self.transcript_dir / f"{safe_session}_{stamp}_{uuid.uuid4().hex[:8]}.json"
            path.write_text(
                json.dumps(
                    {
                        "session": self.session_name,
                        "model": self.get_model_id(),
                        "system": system,
                        "prompt": prompt,
                        "reply": reply,
                        "route": self.last_route,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            # Transcript failures are operational failures, not optional noise. Keep
            # this independent of the normal transcript filename so invalid session
            # identifiers or serialization problems remain diagnosable.
            try:
                self.transcript_dir.mkdir(parents=True, exist_ok=True)
                error_path = self.transcript_dir / "transcript_errors.log"
                with error_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"{time.strftime('%Y-%m-%dT%H:%M:%S')} "
                        f"session={self.session_name!r} error={exc!r}\n"
                    )
            except Exception:
                pass

    def _phase_trace(self, system: str) -> list[dict]:
        phase = self._phase_name(system)
        return [{**item, "phase": phase} for item in self.router.last_trace]

    @staticmethod
    def _phase_name(system: str) -> str:
        lowered = (system or "").lower()
        if "restate the following task" in lowered:
            phase = "reformulation"
        elif "key 'branches'" in lowered:
            phase = "planning"
        elif "tool call failed" in lowered:
            phase = "fix"
        elif "executing one branch" in lowered:
            phase = "agent"
        elif "search" in lowered or "research" in lowered:
            phase = "research"
        else:
            phase = "other"
        return phase

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def last_route(self) -> list[dict]:
        return list(self.router.last_trace)

    @classmethod
    def probe(cls, catalog_path: str | Path | None = None) -> dict:
        try:
            specs = load_catalog(catalog_path)
        except Exception as exc:
            return {"online": False, "model": cls.MODEL_ID, "error": str(exc)}
        available = [spec for spec in specs if has_credential(spec)]
        return {
            "online": bool(available),
            "model": cls.MODEL_ID,
            "configured_models": len(available),
            "configured_providers": sorted({spec.provider for spec in available}),
            "error": "" if available else "no free-provider credentials configured",
        }
