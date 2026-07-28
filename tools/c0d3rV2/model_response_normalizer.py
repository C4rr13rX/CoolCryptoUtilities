"""Schema-aware translation of heterogeneous model replies into C0d3rV2 actions."""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class NormalizationResult:
    value: Any = None
    valid: bool = False
    errors: list[str] = field(default_factory=list)
    transformations: list[str] = field(default_factory=list)


class ModelResponseNormalizer:
    """Parse, canonicalize, type-coerce, and validate model protocol objects.

    The normalizer is deliberately strict at its output boundary and tolerant
    at its input boundary. It never guesses a missing file path or tool name.
    """

    _KEY_ALIASES = {
        "toolcalls": "tool_calls", "toolcall": "tool_call",
        "fixtoolcalls": "tool_calls", "correctedtoolcalls": "tool_calls",
        "retrytoolcalls": "tool_calls", "requestedtoolcalls": "tool_calls",
        "functioncalls": "tool_calls", "functioncall": "tool_call",
        "arguments": "params", "parameters": "params", "args": "params",
        "filename": "path", "filepath": "path",
        "name": "tool", "type": "action", "intent": "action",
        "result": "output", "message": "output", "text": "output",
        "nextstep": "next_step", "nextaction": "next_step",
        "isdone": "done", "completed": "done", "complete": "done",
        "acceptancecriteria": "acceptance_criteria",
        "recoverypolicy": "recovery_policy",
        "scientificrequest": "scientific_request",
    }
    _ACTION_ALIASES = {
        "tool": "tool_calls", "tools": "tool_calls", "call_tool": "tool_calls",
        "call_tools": "tool_calls", "function_call": "tool_calls",
        "functions": "tool_calls", "respond": "answer", "response": "answer",
        "final": "answer", "done": "complete", "completed": "complete",
        "finish": "complete", "subtasks": "sub_branches", "branches": "sub_branches",
    }
    _BOOL_TRUE = {"true", "yes", "y", "on", "enabled", "done", "complete", "completed", "pass", "passed", "one", "1"}
    _BOOL_FALSE = {"false", "no", "n", "off", "disabled", "pending", "todo", "zero", "0", "fail", "failed"}
    _NUMBERS = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    }

    def __init__(self, tool_schemas: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self.tool_schemas = {str(k): dict(v) for k, v in (tool_schemas or {}).items()}

    @classmethod
    def from_tool_descriptions(cls, descriptions: list[dict[str, Any]]) -> "ModelResponseNormalizer":
        return cls({str(item.get("name")): item.get("params") or {} for item in descriptions if item.get("name")})

    def parse(self, raw: Any) -> NormalizationResult:
        if isinstance(raw, (dict, list)):
            return NormalizationResult(raw, True)
        text = str(raw or "").strip()
        if not text:
            return NormalizationResult(errors=["empty model response"])
        if re.search(r"<tool_call>", text, flags=re.I):
            markup = self._parse_tool_markup(text)
            if markup:
                return NormalizationResult(
                    {"action": "tool_calls", "tool_calls": markup},
                    True,
                    transformations=["parsed_tool_markup"],
                )
        candidates = [text]
        fenced = re.sub(r"^\s*```(?:json|javascript|js|python)?\s*|\s*```\s*$", "", text, flags=re.I | re.S).strip()
        if fenced != text:
            candidates.append(fenced)
        balanced = self._balanced_payload(fenced)
        if balanced and balanced not in candidates:
            candidates.append(balanced)
        errors: list[str] = []
        for candidate in candidates:
            try:
                return NormalizationResult(json.loads(candidate), True, transformations=["parsed_json"])
            except Exception as exc:
                errors.append(type(exc).__name__)
            # Smaller/free coding models commonly emit otherwise valid JSON while
            # placing literal newlines or tabs inside a source-code string.  JSON
            # forbids those unescaped controls, but their location is unambiguous:
            # preserve structural whitespace and escape controls only while the
            # scanner is inside a quoted string.
            escaped_controls = self._escape_controls_in_json_strings(candidate)
            if escaped_controls != candidate:
                try:
                    return NormalizationResult(
                        json.loads(escaped_controls),
                        True,
                        transformations=["escaped_string_control_characters", "parsed_json"],
                    )
                except Exception as exc:
                    errors.append(type(exc).__name__)
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
            closed = self._close_json_delimiters(repaired)
            if closed and closed != repaired:
                try:
                    return NormalizationResult(
                        json.loads(closed), True,
                        transformations=["closed_truncated_json"],
                    )
                except Exception as exc:
                    errors.append(type(exc).__name__)
            try:
                value = ast.literal_eval(repaired)
                if isinstance(value, (dict, list)):
                    return NormalizationResult(value, True, transformations=["parsed_python_literal"])
            except Exception as exc:
                errors.append(type(exc).__name__)
        markup = self._parse_tool_markup(text)
        if markup:
            return NormalizationResult(
                {"action": "tool_calls", "tool_calls": markup},
                True,
                transformations=["parsed_tool_markup"],
            )
        recovered = self._recover_tool_call_array(text)
        if recovered:
            return NormalizationResult(
                {"action": "tool_calls", "tool_calls": recovered},
                True,
                transformations=["recovered_truncated_tool_calls"],
            )
        return NormalizationResult(errors=["unparseable structured response", *errors[-3:]])

    @staticmethod
    def _escape_controls_in_json_strings(text: str) -> str:
        """Escape literal JSON control characters without altering structure.

        This is intentionally a lexical repair, not a best-effort interpretation:
        quotes and backslashes retain normal JSON escaping semantics, while only
        U+0000..U+001F characters occurring inside strings are transformed.
        """
        output: list[str] = []
        in_string = False
        escaped = False
        replacements = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
        for char in text:
            if in_string:
                if escaped:
                    output.append(char)
                    escaped = False
                    continue
                if char == "\\":
                    output.append(char)
                    escaped = True
                    continue
                if char == '"':
                    output.append(char)
                    in_string = False
                    continue
                if ord(char) < 0x20:
                    output.append(replacements.get(char, f"\\u{ord(char):04x}"))
                    continue
                output.append(char)
                continue
            output.append(char)
            if char == '"':
                in_string = True
        return "".join(output)

    def normalize_action(self, raw: Any) -> NormalizationResult:
        parsed = self.parse(raw)
        if not parsed.valid:
            return parsed
        value = self._canonicalize(self._unwrap(parsed.value, parsed.transformations), parsed.transformations)
        if isinstance(value, list):
            if len(value) == 1 and isinstance(value[0], dict) and value[0].get("action"):
                value = value[0]
                parsed.transformations.append("unwrapped_single_action_list")
            elif value and all(isinstance(item, dict) for item in value):
                value = {"action": "tool_calls", "tool_calls": value}
                parsed.transformations.append("wrapped_call_list")
            else:
                return NormalizationResult(value, False, ["top-level list is not a tool-call list"], parsed.transformations)
        if not isinstance(value, dict):
            return NormalizationResult(value, False, ["action is not an object"], parsed.transformations)

        action = str(value.get("action") or "").strip().lower().replace("-", "_").replace(" ", "_")
        if action in self.tool_schemas:
            schema = self.tool_schemas.get(action, {})
            lifted = {
                key: item for key, item in value.items()
                if key in schema
            }
            value = {
                "action": "tool_calls",
                "tool_calls": [{"tool": action, "params": value.get("params") or lifted}],
            }
            if lifted and not value["tool_calls"][0].get("params") == {}:
                parsed.transformations.append(f"lifted top-level {action} parameters")
            action = "tool_calls"
            parsed.transformations.append("translated tool-name action")
        if not action and value.get("tool_calls"):
            action = "tool_calls"
            parsed.transformations.append("inferred tool_calls action")
        if not action and (value.get("tool") or value.get("function")):
            action = "tool_calls"
            value = {"action": action, "tool_calls": [value]}
        action = self._ACTION_ALIASES.get(action, action)
        value["action"] = action

        if action == "tool_call":
            action = "tool_calls"; value["action"] = action
        if action == "tool_calls":
            calls = value.get("tool_calls") or value.get("tool_call") or []
            if isinstance(calls, dict):
                calls = [calls]
            if not calls and (value.get("tool") or value.get("function")):
                calls = [value]
            normalized_calls = []
            for call in calls if isinstance(calls, list) else []:
                normalized = self._normalize_call(call, parsed.transformations)
                if normalized:
                    normalized_calls.append(normalized)
            value = {"action": "tool_calls", "tool_calls": normalized_calls}
            if not normalized_calls:
                return NormalizationResult(value, False, ["tool_calls contains no valid calls"], parsed.transformations)
        elif action in {"answer", "complete"}:
            output = value.get("output")
            if output is None:
                output = value.get("answer") or value.get("summary") or ""
            value["output"] = output if isinstance(output, str) else json.dumps(output, default=str)
        elif action == "sub_branches":
            branches = value.get("sub_branches") or value.get("branches") or []
            value["sub_branches"] = branches if isinstance(branches, list) else [branches]
        else:
            return NormalizationResult(value, False, [f"unsupported action: {action or '<missing>'}"], parsed.transformations)
        return NormalizationResult(value, True, transformations=parsed.transformations)

    def normalize_scrutiny(self, raw: Any) -> NormalizationResult:
        parsed = self.parse(raw)
        if not parsed.valid:
            return parsed
        value = self._canonicalize(self._unwrap(parsed.value, parsed.transformations), parsed.transformations)
        if not isinstance(value, dict):
            return NormalizationResult(value, False, ["scrutiny response is not an object"], parsed.transformations)
        decision = str(value.get("decision") or value.get("action") or "").lower().strip()
        decision = "direct" if decision in {"answer", "respond", "response", "direct_answer"} else "execute" if decision in {"tools", "tool_calls", "work", "plan", "execute_task"} else decision
        if decision not in {"direct", "execute"}:
            return NormalizationResult(value, False, [f"invalid scrutiny decision: {decision or '<missing>'}"], parsed.transformations)
        value["decision"] = decision
        value["branches"] = self._normalize_branches(value.get("branches") or value.get("steps") or value.get("plan") or [], parsed.transformations)
        if decision == "direct":
            value["answer"] = str(value.get("answer") or value.get("output") or "").strip()
            if not value["answer"]:
                return NormalizationResult(value, False, ["direct decision has no answer"], parsed.transformations)
        else:
            value["scientific_request"] = str(value.get("scientific_request") or value.get("request") or "").strip()
        return NormalizationResult(value, True, transformations=parsed.transformations)

    def normalize_plan(self, raw: Any) -> NormalizationResult:
        parsed = self.parse(raw)
        if not parsed.valid:
            return parsed
        value = self._canonicalize(self._unwrap(parsed.value, parsed.transformations), parsed.transformations)
        if not isinstance(value, dict):
            return NormalizationResult(value, False, ["plan is not an object"], parsed.transformations)
        raw_steps = value.get("plan") or value.get("steps") or []
        if isinstance(raw_steps, dict):
            raw_steps = raw_steps.get("steps") or raw_steps.get("plan") or []
        steps = []
        for index, step in enumerate(raw_steps if isinstance(raw_steps, list) else [], 1):
            if isinstance(step, str):
                steps.append({"id": index, "title": step.strip(), "status": "todo"})
                continue
            if not isinstance(step, dict):
                continue
            title = str(step.get("title") or step.get("description") or step.get("task") or step.get("step") or "").strip()
            if not title:
                continue
            status = str(step.get("status") or "todo").lower().strip().replace(" ", "_")
            status = "done" if status in self._BOOL_TRUE else "todo" if status in self._BOOL_FALSE else status
            steps.append({**step, "id": step.get("id", index), "title": title, "status": status})
        value["steps"] = steps
        value["plan"] = steps
        value["done"] = self._coerce(value.get("done", False), "bool", parsed.transformations)
        value["next_step"] = str(value.get("next_step") or "").strip()
        return NormalizationResult(value, bool(steps), [] if steps else ["plan contains no executable steps"], parsed.transformations)

    def _normalize_call(self, call: Any, transformations: list[str]) -> dict[str, Any] | None:
        if not isinstance(call, dict):
            return None
        call = self._canonicalize(call, transformations)
        function = call.get("function")
        if isinstance(function, dict):
            function = self._canonicalize(function, transformations)
            call = {**call, **function}
        tool = str(call.get("tool") or "").strip()
        if not tool:
            return None
        schema = self.tool_schemas.get(tool, {})
        params = call.get("params") or {}
        if not params and schema:
            params = {key: value for key, value in call.items() if key in schema}
            if params:
                transformations.append(f"lifted top-level {tool} parameters")
        if isinstance(params, str):
            parsed = self.parse(params)
            params = parsed.value if parsed.valid and isinstance(parsed.value, dict) else {}
            transformations.extend(parsed.transformations)
        if not isinstance(params, dict):
            params = {}
        normalized_params = {}
        for key, value in params.items():
            if schema and key not in schema:
                transformations.append(f"discarded unknown {tool} parameter {key}")
                continue
            descriptor = str(schema.get(key, ""))
            normalized_params[key] = self._coerce(value, descriptor, transformations)
        if schema and params and not normalized_params:
            transformations.append(f"rejected {tool} call with no recognized parameters")
            return None
        return {"tool": tool, "params": normalized_params}

    def _normalize_branches(self, raw: Any, transformations: list[str]) -> list[dict[str, Any]]:
        if isinstance(raw, dict):
            raw = raw.get("branches") or raw.get("steps") or [raw]
        result = []
        for index, branch in enumerate(raw if isinstance(raw, list) else [], 1):
            if isinstance(branch, str):
                result.append({"id": f"step-{index}", "description": branch})
            elif isinstance(branch, dict):
                branch = self._canonicalize(branch, transformations)
                description = str(branch.get("description") or branch.get("title") or branch.get("task") or "").strip()
                if description:
                    result.append({**branch, "id": str(branch.get("id") or f"step-{index}"), "description": description})
        return result

    def _coerce(self, value: Any, descriptor: str, transformations: list[str]) -> Any:
        expected = descriptor.lower().strip()
        if "bool" in expected:
            if isinstance(value, bool): return value
            token = str(value).lower().strip()
            if token in self._BOOL_TRUE: transformations.append(f"coerced {value!r} to true"); return True
            if token in self._BOOL_FALSE: transformations.append(f"coerced {value!r} to false"); return False
        if any(marker in expected for marker in ("int", "number", "float")):
            if isinstance(value, (int, float)) and not isinstance(value, bool): return value
            token = str(value).lower().strip()
            numeric = self._NUMBERS.get(token, token.replace(",", ""))
            try:
                result = float(numeric) if any(marker in expected for marker in ("number", "float")) else int(float(numeric))
                transformations.append(f"coerced {value!r} to numeric")
                return result
            except (TypeError, ValueError):
                return value
        if ("list" in expected or expected.startswith("[")) and not isinstance(value, list):
            transformations.append("wrapped scalar as list")
            return [value]
        if ("dict" in expected or "object" in expected) and isinstance(value, str):
            parsed = self.parse(value)
            if parsed.valid and isinstance(parsed.value, dict):
                transformations.extend(parsed.transformations); return parsed.value
        return value

    def _canonicalize(self, value: Any, transformations: list[str]) -> Any:
        if isinstance(value, list):
            return [self._canonicalize(item, transformations) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            raw_key = str(key)
            compact = re.sub(r"[^a-z0-9]", "", raw_key.lower())
            canonical = self._KEY_ALIASES.get(compact, re.sub(r"(?<!^)(?=[A-Z])", "_", raw_key).lower().replace("-", "_"))
            if canonical != raw_key:
                transformations.append(f"renamed {raw_key} to {canonical}")
            result[canonical] = self._canonicalize(item, transformations)
        return result

    @staticmethod
    def _unwrap(value: Any, transformations: list[str]) -> Any:
        wrappers = {"response", "data", "payload", "result"}
        while isinstance(value, dict) and len(value) == 1 and next(iter(value)) in wrappers:
            transformations.append(f"unwrapped {next(iter(value))}")
            value = next(iter(value.values()))
        return value

    @staticmethod
    def _balanced_payload(text: str) -> str:
        starts = [(text.find("{"), "{", "}"), (text.find("["), "[", "]")]
        starts = [item for item in starts if item[0] >= 0]
        if not starts:
            return ""
        start, opening, closing = min(starts)
        depth = 0; quoted = False; escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if quoted:
                if escaped: escaped = False
                elif char == "\\": escaped = True
                elif char == '"': quoted = False
                continue
            if char == '"': quoted = True
            elif char == opening: depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0: return text[start:index + 1]
        return ""

    @classmethod
    def _parse_tool_markup(cls, text: str) -> list[dict[str, Any]]:
        """Parse common free-model ``<tool_call>`` dialects without another LLM call."""
        calls: list[dict[str, Any]] = []
        tool_aliases = {
            "execute_command": "executor",
            "run_command": "executor",
            "shell": "executor",
            "terminal": "executor",
            "bash": "executor",
            "powershell": "executor",
            "write_file": "file_write",
            "read_file": "file_read",
        }
        attr_pattern = r"([A-Za-z_]\w*)\s*=\s*(\[[^\]]*\]|\{[^}]*\}|\"[^\"]*\"|'[^']*'|[^,\s/>]+)"

        # Function-style dialect: <tool_call>file_read(path="README.md")
        function_spans: list[tuple[int, int]] = []
        function_pattern = r"<tool_call>\s*([A-Za-z_][\w.-]*)\s*\((.*?)\)\s*(?=\r?$|<tool_call>)"
        for function_call in re.finditer(function_pattern, text, flags=re.I | re.M | re.S):
            tool = tool_aliases.get(function_call.group(1).lower(), function_call.group(1))
            params: dict[str, Any] = {}
            for attr in re.finditer(attr_pattern, function_call.group(2), flags=re.S):
                key, token = attr.group(1), attr.group(2).strip()
                try:
                    params[key] = ast.literal_eval(token)
                except Exception:
                    params[key] = token.strip("\"'")
            calls.append({"tool": tool, "params": params})
            function_spans.append(function_call.span())

        remaining = text
        for start, end in reversed(function_spans):
            remaining = remaining[:start] + remaining[end:]
        chunks = re.split(r"<tool_call>\s*", remaining, flags=re.I)[1:]
        for chunk in chunks:
            header, separator, remainder = chunk.partition(">")
            if not separator:
                continue
            header = header.strip().rstrip("/").strip()
            match = re.match(r"([A-Za-z_][\w.-]*)(.*)", header, flags=re.S)
            if not match:
                continue
            tool = tool_aliases.get(match.group(1).lower(), match.group(1))
            raw_attrs = match.group(2).strip()
            params: dict[str, Any] = {}
            for attr in re.finditer(attr_pattern, raw_attrs, flags=re.S):
                key, token = attr.group(1), attr.group(2).strip()
                try:
                    params[key] = ast.literal_eval(token)
                except Exception:
                    params[key] = token.strip("\"'")
            # Some providers use nested elements: <tool_call>executor><command>...</command>
            body = re.split(r"</?tool_call\s*>", remainder, maxsplit=1, flags=re.I)[0]
            # Kilo-style degraded markup sometimes emits the first parameter as
            # ``<arg_content>new_string: "... </arg_value>`` rather than an
            # arg_key/arg_value pair. Recover it only when it names an explicit
            # parameter; never infer a path or tool.
            for malformed in re.finditer(
                r"<arg_content>\s*([A-Za-z_]\w*)\s*:\s*[\"']?(.*?)</arg_value>",
                body,
                flags=re.I | re.S,
            ):
                params[malformed.group(1).strip()] = malformed.group(2).rstrip("\"'").strip()
            for pair in re.finditer(
                r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
                body,
                flags=re.I | re.S,
            ):
                params[pair.group(1).strip()] = pair.group(2).strip()
            # Recover explicitly named arguments when a gateway omits closing
            # tags; stop at the next named argument or tool boundary.
            for pair in re.finditer(
                r"<arg_key>\s*(.*?)\s*(?:</arg_key>|(?=<arg_value>))\s*"
                r"<arg_value>\s*(.*?)(?:</arg_value>|(?=<arg_key>|<tool_call>|</tool_call>|</invoke>|$))",
                body,
                flags=re.I | re.S,
            ):
                key = pair.group(1).strip()
                value = pair.group(2).strip()
                if key and key not in params:
                    params[key] = value
            for nested in re.finditer(r"<([A-Za-z_]\w*)>\s*(.*?)\s*</\1>", body, flags=re.I | re.S):
                if nested.group(1).lower() not in {"arg_key", "arg_value"}:
                    params[nested.group(1)] = nested.group(2).strip()
            if tool == "executor" and "command" not in params:
                command = re.search(r"<command>\s*(.*?)(?:</command>|$)", body, flags=re.I | re.S)
                if command:
                    params["command"] = command.group(1).strip()
            calls.append({"tool": tool, "params": params})
        return calls

    @staticmethod
    def _recover_tool_call_array(text: str) -> list[dict[str, Any]]:
        """Recover only complete call objects from a truncated tool_calls array."""
        key = re.search(r'"(?:tool_calls|fix_tool_calls)"\s*:\s*\[', text, flags=re.I)
        if not key:
            return []
        calls: list[dict[str, Any]] = []
        depth = 0
        start = -1
        quoted = False
        escaped = False
        for index in range(key.end(), len(text)):
            char = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}" and depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        value = json.loads(text[start:index + 1])
                    except Exception:
                        value = None
                    if isinstance(value, dict) and (
                        value.get("tool") or value.get("name") or value.get("function")
                    ):
                        calls.append(value)
                    start = -1
            elif char == "]" and depth == 0:
                break
        return calls

    @staticmethod
    def _close_json_delimiters(text: str) -> str:
        """Append uniquely implied closing braces/brackets; never repair string content."""
        stack: list[str] = []
        output: list[str] = []
        quoted = False
        escaped = False
        repairs = 0
        pairs = {"{": "}", "[": "]"}
        for char in text:
            output.append(char)
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char in pairs:
                stack.append(char)
            elif char in {"}", "]"}:
                while stack and pairs[stack[-1]] != char and repairs < 4:
                    output.insert(len(output) - 1, pairs[stack.pop()])
                    repairs += 1
                if not stack or pairs[stack[-1]] != char:
                    return ""
                stack.pop()
        if quoted or escaped or (not stack and not repairs) or len(stack) + repairs > 4:
            return ""
        return "".join(output) + "".join(pairs[opening] for opening in reversed(stack))
