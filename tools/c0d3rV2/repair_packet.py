"""Deterministic repair contracts for small or unreliable model backends.

The packet is deliberately model-agnostic.  It converts validator output into
one bounded unit of work, retains failures that have already been eliminated,
and progressively slices a repair down to component/class/method scope when a
provider repeats itself without changing the externally observed result.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


_VOLATILE = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*(?:ms|seconds?|minutes?)\b|0x[0-9a-f]+|"
    r"[A-Fa-f0-9]{8}-[A-Fa-f0-9-]{27,})",
    re.IGNORECASE,
)


def _stable_text(value: Any) -> str:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value or ""))
    text = _VOLATILE.sub("<volatile>", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _signature(kind: str, text: str) -> str:
    stable = _stable_text(text)[:800]
    digest = hashlib.sha256(f"{kind}:{stable}".encode("utf-8")).hexdigest()[:12]
    return f"{kind}:{digest}"


def failure_signals(result: dict[str, Any]) -> list[dict[str, str]]:
    """Extract portable, stable failures from a smoke/acceptance result."""
    signals: list[dict[str, str]] = []
    for gap in result.get("quality_gaps") or []:
        signals.append({"kind": "quality", "message": str(gap), "id": _signature("quality", str(gap))})
    setup = result.get("dependency_setup") or {}
    command_rows = list(result.get("commands") or [])
    if setup and int(setup.get("exit_code") or 0) != 0:
        command_rows.insert(0, setup)
    if result.get("error"):
        command_rows.append({"stderr": result.get("error"), "exit_code": 1})
    for row in command_rows:
        if int(row.get("exit_code") or 0) == 0:
            continue
        raw = "\n".join(str(row.get(key) or "") for key in ("diagnostics", "stderr", "stdout"))
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        def specificity(line: str) -> int:
            score = 0
            lowered = line.lower()
            if re.search(r"(?:^|\s)(?:src|app|lib)/[^\s:]+:\d+", line, re.IGNORECASE):
                score += 20
            elif re.search(r"(?:^|\s)tests?/[^\s:]+:\d+", line, re.IGNORECASE):
                score += 12
            if "node_modules" in lowered:
                score -= 8
            if re.search(r"\b(?:typeerror|referenceerror|syntaxerror|error):", line, re.IGNORECASE):
                score += 10
            if re.search(r"missing dependency|cannot find (?:module|package|dependency)|eresolve|etarget", line, re.IGNORECASE):
                score += 15
            if re.search(r"expected .+ (?:received|got|found)", line, re.IGNORECASE):
                score += 8
            if re.search(r"\b(?:test files|tests?)\s+\d+\s+failed", line, re.IGNORECASE):
                score -= 5
            return score

        ranked = sorted(lines, key=specificity, reverse=True)
        positive = [line for line in ranked if specificity(line) > 0]
        useful_parts = positive[:1]
        # Preserve the causal call chain (implementation, composition root,
        # consumer/test), not just the two highest-scoring implementation
        # frames. This is the bounded integration surface for a port migration.
        seen_locations = set()
        for line in useful_parts:
            match = re.search(r"(?:src|tests?|app|lib)/[^\s:]+", line, re.I)
            if match:
                seen_locations.add(match.group(0).lower())
        for line in positive[1:]:
            match = re.search(r"(?:src|tests?|app|lib)/[^\s:]+", line, re.I)
            if match and match.group(0).lower() not in seen_locations:
                useful_parts.append(line)
                seen_locations.add(match.group(0).lower())
            if len(useful_parts) >= 4:
                break
        for line in positive[1:]:
            if line not in useful_parts and re.search(r"\b(?:typeerror|referenceerror|syntaxerror|error):", line, re.I):
                useful_parts.append(line)
                break
        useful = " | ".join(useful_parts) or (ranked[0] if ranked else "command failed")
        kind = "dependency" if re.search(
            r"missing dependency|cannot find (?:module|package|dependency)|eresolve|etarget",
            raw, re.IGNORECASE,
        ) else "compile_or_test"
        signals.append({"kind": kind, "message": useful[:1000], "id": _signature(kind, useful)})
    # Preserve ordering but collapse duplicate normalized messages.
    unique: dict[str, dict[str, str]] = {}
    for signal in signals:
        unique.setdefault(signal["id"], signal)
    return list(unique.values())


def _extract_contracts(root: Path, paths: Iterable[str], limit: int = 24) -> list[str]:
    """Extract declarations that communicate integration shapes without full files."""
    declarations: list[str] = []
    patterns = (
        re.compile(r"(?:export\s+)?interface\s+\w+(?:\s+extends\s+[^\{]+)?\s*\{[^}]{0,1800}\}", re.S),
        re.compile(r"(?:export\s+)?type\s+\w+(?:<[^;=]+>)?\s*=\s*[^;]{1,1000};", re.S),
        re.compile(r"(?:export\s+)?(?:abstract\s+)?class\s+\w+[^\{]{0,300}\{", re.S),
        re.compile(r"(?:export\s+)?(?:async\s+)?function\s+\w+\s*\([^)]*\)\s*(?::\s*[^\{;]+)?", re.S),
        re.compile(
            r"^\s*(?:(?:public|private|protected|static|readonly|abstract)\s+)*"
            r"(?:async\s+)?[A-Za-z_$][\w$]*\s*\([^)]*\)\s*(?::\s*[^\{;]+)?\s*\{",
            re.M,
        ),
    )
    for relative in paths:
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except (OSError, ValueError):
            continue
        for pattern in patterns:
            for match in pattern.finditer(text):
                declaration = re.sub(r"\s+", " ", match.group(0)).strip()
                item = f"{relative}: {declaration[:1200]}"
                if item not in declarations:
                    declarations.append(item)
                if len(declarations) >= limit:
                    return declarations
    return declarations


def _focus_paths(root: Path, paths: list[str], signal: dict[str, str]) -> list[str]:
    """Choose the smallest file surface that can resolve one failure."""
    kind = signal.get("kind") or ""
    message = (signal.get("message") or "").lower()
    manifests = {
        "package.json", "pyproject.toml", "requirements.txt", "cargo.toml",
        "composer.json", "pom.xml", "build.gradle", "build.gradle.kts", "go.mod",
    }
    if kind == "dependency":
        selected = [path for path in paths if Path(path).name.lower() in manifests]
        return selected[:1] or paths[:1]
    mentioned = [
        path for path in paths
        if path.lower() in message or Path(path).name.lower() in message
    ]
    if mentioned:
        return mentioned[:4]
    content_patterns: list[re.Pattern[str]] = []
    if "placeholder" in message:
        content_patterns.append(re.compile(r"\bplaceholder\b", re.IGNORECASE))
    if any(term in message for term in ("webglrenderer", "unsafe double assertion", "counterfeit")):
        content_patterns.append(re.compile(r"as\s+unknown\s+as\s+(?:THREE\.)?WebGLRenderer", re.IGNORECASE))
    if any(term in message for term in ("shim", "platform mock", "dummy", "fake")):
        content_patterns.append(re.compile(r"(?:prototype\.|\bmock\b|\bdummy\b|\bfake\b)", re.IGNORECASE))
    if content_patterns:
        selected: list[str] = []
        for path in paths:
            try:
                text = (root / path).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if any(pattern.search(text) for pattern in content_patterns):
                selected.append(path)
        if selected:
            return selected[:2]
    return paths[:2]


@dataclass
class RepairPacket:
    step: str
    paths: list[str]
    validator_command: str
    active_failures: list[dict[str, str]]
    resolved_failures: list[dict[str, str]] = field(default_factory=list)
    repeated_outcome_count: int = 0
    recurrence_count: int = 0
    contracts: list[str] = field(default_factory=list)
    focus_failure: dict[str, str] = field(default_factory=dict)
    focus_paths: list[str] = field(default_factory=list)

    @property
    def scope_level(self) -> str:
        count = self.repeated_outcome_count
        if count <= 0:
            return "component"
        if count == 1:
            return "class"
        return "method"

    @property
    def forbidden(self) -> list[str]:
        return [
            "Do not replace production dependencies with mocks, stubs, dummy objects, unsafe assertions, or catch-all fallbacks.",
            "Do not weaken, delete, skip, or rewrite acceptance tests merely to obtain a pass.",
            "Do not re-plan the project, claim actions in prose, or touch files outside the declared repair surface.",
            "Do not reintroduce any failure listed under resolved_failures.",
        ]

    @property
    def recommended_pattern(self) -> str:
        message = (self.focus_failure.get("message") or "").lower()
        if any(term in message for term in ("webgl", "renderer", "context")) and re.search(
            r"(?:src|app)/main\.[a-z]+", message,
        ) and re.search(r"tests?/", message):
            return (
                "Pure import boundary plus dependency inversion: importing a module must not construct browser, "
                "device, network, database, or process resources. Export a bootstrap/composition-root function; "
                "inject a minimal truthful port into domain/class construction; call the real adapter only from "
                "the runtime entry point, and pass explicit doubles from tests."
            )
        if any(term in message for term in ("webgl", "appendchild", "platform", "renderer", "context")):
            return (
                "Dependency inversion: define a minimal production port with truthful return types; "
                "inject it into the class/function; construct the real library adapter only at the "
                "browser composition root; provide an explicit double from test code."
            )
        if self.focus_failure.get("kind") == "dependency":
            return "Manifest reconciliation: add the exact compatible dependency without replacing or removing established scripts/dependencies."
        if "placeholder" in message:
            return "Replace the placeholder with the smallest observable domain behavior, explicit invariants, and typed state transitions required by its consumers."
        return "Repair the most specific failing call site while preserving public interfaces and already passing behavior."

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "c0d3r.repair-packet/v1",
            "step": self.step,
            "scope_level": self.scope_level,
            "repair_paths": self.paths,
            "active_failures": self.active_failures,
            "focus_failure": self.focus_failure,
            "focus_paths": self.focus_paths,
            "deferred_failures": [
                item for item in self.active_failures
                if item.get("id") != self.focus_failure.get("id")
            ],
            "resolved_failures": self.resolved_failures,
            "repeated_outcome_count": self.repeated_outcome_count,
            "recurrence_count": self.recurrence_count,
            "integration_contracts": self.contracts,
            "required_transition": (
                "Resolve only focus_failure with exactly one cohesive mutation within focus_paths, then validate. "
                "At method scope, implement only one method and preserve its containing class and public interfaces."
            ),
            "recommended_pattern": self.recommended_pattern,
            "migration_sequence": (
                "For a cross-file architecture correction, this packet may introduce one truthful port/injection "
                "point even if existing consumers still fail. Validation will name those consumers and later "
                "packets will migrate them one at a time. Never preserve a bad architecture with a fake fallback."
            ),
            "validator_command": self.validator_command,
            "forbidden": self.forbidden,
        }

    def to_prompt(self) -> str:
        return "Deterministic repair packet (authoritative):\n" + json.dumps(self.to_dict(), indent=2)


def advance_repair_state(
    prior: dict[str, Any] | None,
    result: dict[str, Any],
    *,
    step: str,
    paths: list[str],
    root: Path,
) -> tuple[dict[str, Any], RepairPacket]:
    """Reconcile fresh evidence with history and choose the next repair grain."""
    prior = dict(prior or {})
    previous_active = {item.get("id"): item for item in prior.get("active_failures") or [] if item.get("id")}
    resolved = {item.get("id"): item for item in prior.get("resolved_failures") or [] if item.get("id")}
    active = failure_signals(result)
    active_ids = {item["id"] for item in active}
    for signal_id, signal in previous_active.items():
        if signal_id not in active_ids:
            resolved[signal_id] = signal
    recurrence_count = int(prior.get("recurrence_count") or 0)
    for signal in active:
        if signal["id"] in resolved:
            recurrence_count += 1
    fingerprint = hashlib.sha256(
        json.dumps(sorted(active_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    repeated = int(prior.get("repeated_outcome_count") or 0) + 1 if fingerprint == prior.get("fingerprint") else 0
    command = str(result.get("command") or "run the deterministic acceptance suite")
    priority = {"dependency": 0, "compile_or_test": 1, "quality": 2}
    focus = min(active, key=lambda item: priority.get(item.get("kind", ""), 9), default={})
    focused_paths = _focus_paths(root, paths, focus) if focus else paths[:1]
    packet = RepairPacket(
        step=step,
        paths=list(dict.fromkeys(paths)),
        validator_command=command,
        active_failures=active,
        resolved_failures=list(resolved.values()),
        repeated_outcome_count=repeated,
        recurrence_count=recurrence_count,
        contracts=_extract_contracts(root, focused_paths),
        focus_failure=focus,
        focus_paths=focused_paths,
    )
    state = packet.to_dict() | {"fingerprint": fingerprint}
    return state, packet
