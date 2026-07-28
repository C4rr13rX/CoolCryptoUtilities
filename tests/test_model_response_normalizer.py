import json

from tools.c0d3rV2.model_response_normalizer import ModelResponseNormalizer


def normalizer():
    return ModelResponseNormalizer({
        "class_refinement_benchmark": {"count": "int", "attempts": "int"},
        "web_search": {"query": "str", "max_results": "int"},
        "file_read": {"path": "str", "offset": "int", "limit": "int"},
        "file_write": {"path": "str", "content": "str"},
        "example": {"enabled": "bool", "items": "list[str]"},
    })


def test_normalizes_literal_newlines_and_tabs_inside_file_content():
    raw = (
        '{"action":"tool_calls","tool_calls":[{"tool":"file_write",'
        '"params":{"path":"src/example.ts","content":"export class Example {\n'
        '\tvalue = 1;\n}"}}]}'
    )

    result = normalizer().normalize_action(raw)

    assert result.valid is True
    assert "escaped_string_control_characters" in result.transformations
    assert result.value["tool_calls"][0]["params"]["content"] == "export class Example {\n\tvalue = 1;\n}"


def test_control_repair_does_not_escape_structural_json_whitespace():
    raw = '{\n  "action": "answer",\n  "output": "line one\nline two",\n  "reason": "known"\n}'

    result = normalizer().normalize_action(raw)

    assert result.valid is True
    assert result.value["output"] == "line one\nline two"


def test_normalizes_malformed_arg_content_parameter_markup():
    raw = """I will patch it.
<tool_call>file_write>
<arg_content>new_string: \"import { replacement } from './module';
const value = replacement();</arg_value>
<arg_key>old_string</arg_key><arg_value>const value = old();</arg_value>
<arg_key>path</arg_key><arg_value>src/main.ts</arg_value>
</tool_call>"""

    result = ModelResponseNormalizer({
        "file_write": {"path": "str", "old_string": "str", "new_string": "str"},
    }).normalize_action(raw)

    assert result.valid is True
    call = result.value["tool_calls"][0]
    assert call["tool"] == "file_write"
    assert call["params"]["path"] == "src/main.ts"
    assert call["params"]["old_string"] == "const value = old();"
    assert call["params"]["new_string"].startswith("import { replacement }")
    assert call["params"]["new_string"].endswith("replacement();")


def test_fenced_openai_function_call_and_word_number_are_normalized():
    raw = """```json
    {"type":"function_call","functionCall":{"function":{"name":"class_refinement_benchmark","arguments":"{\\"count\\":\\"one\\",\\"attempts\\":\\"2\\"}"}}}
    ```"""
    result = normalizer().normalize_action(raw)
    assert result.valid, result.errors
    assert result.value == {
        "action": "tool_calls",
        "tool_calls": [{"tool": "class_refinement_benchmark", "params": {"count": 1, "attempts": 2}}],
    }


def test_singular_call_aliases_and_boolean_list_coercion():
    result = normalizer().normalize_action({
        "intent": "call tool",
        "toolCall": {"name": "example", "args": {"enabled": "yes", "items": "alpha"}},
    })
    assert result.valid, result.errors
    assert result.value["tool_calls"][0]["params"] == {"enabled": True, "items": ["alpha"]}


def test_wrapped_response_and_trailing_comma_are_parsed():
    result = normalizer().normalize_action('prefix {"response":{"action":"done","summary":"built"},} suffix')
    assert result.valid, result.errors
    assert result.value["action"] == "complete"
    assert result.value["output"] == "built"


def test_scrutiny_camel_case_and_branch_strings_are_normalized():
    result = normalizer().normalize_scrutiny({
        "decision": "execute_task",
        "scientificRequest": "Build bounded module",
        "steps": ["Inspect contract", {"title": "Write implementation"}],
    })
    assert result.valid, result.errors
    assert result.value["decision"] == "execute"
    assert [branch["description"] for branch in result.value["branches"]] == ["Inspect contract", "Write implementation"]


def test_plan_normalizes_numeric_step_descriptions_status_and_done_entity():
    raw = json.dumps({
        "plan": {"steps": [
            {"step": 1, "description": "Create foundation", "status": "completed"},
            {"step": "two", "description": "Run tests", "status": "pending"},
        ]},
        "nextAction": "Run tests",
        "isDone": "zero",
    })
    result = normalizer().normalize_plan(raw)
    assert result.valid, result.errors
    assert result.value["steps"][0]["title"] == "Create foundation"
    assert result.value["steps"][0]["status"] == "done"
    assert result.value["next_step"] == "Run tests"
    assert result.value["done"] is False


def test_invalid_tool_call_is_rejected_instead_of_silently_completing():
    result = normalizer().normalize_action({"action": "tool_calls", "tool_calls": [{"params": {"path": "x"}}]})
    assert not result.valid
    assert "no valid calls" in result.errors[0]


def test_fix_tool_calls_without_action_are_translated_to_actions():
    result = normalizer().normalize_action({
        "fix_tool_calls": [{
            "tool": "file_write",
            "params": {"path": "src/main.ts", "content": "export {};"},
        }],
        "reasoning": "corrected call",
    })
    assert result.valid, result.errors
    assert result.value == {
        "action": "tool_calls",
        "tool_calls": [{
            "tool": "file_write",
            "params": {"path": "src/main.ts", "content": "export {};"},
        }],
    }
    assert "inferred tool_calls action" in result.transformations


def test_pseudo_xml_tool_calls_are_parsed_without_model_repair():
    raw = '''I will inspect and validate.
<tool_call>file_locate query="package.json" project_root="C:/work" />
<tool_call>execute_command><command>npm run build</command>'''
    result = normalizer().normalize_action(raw)
    assert result.valid, result.errors
    assert result.value == {
        "action": "tool_calls",
        "tool_calls": [
            {"tool": "file_locate", "params": {"query": "package.json", "project_root": "C:/work"}},
            {"tool": "executor", "params": {"command": "npm run build"}},
        ],
    }
    assert "parsed_tool_markup" in result.transformations


def test_shell_markup_alias_is_canonicalized_to_executor():
    raw = '''<tool_call>shell>
<arg_key>command</arg_key><arg_value>npm test</arg_value>
</invoke-result>'''
    result = normalizer().normalize_action(raw)
    assert result.valid
    assert result.value["tool_calls"] == [{
        "tool": "executor", "params": {"command": "npm test"},
    }]


def test_unclosed_explicit_argument_markup_is_recovered():
    raw = '''The provider used its legacy tool dialect.
<tool_call>execute_command>
<arg_key>command
<arg_value>npm test
</invoke>'''
    result = normalizer().normalize_action(raw)
    assert result.valid
    assert result.value["tool_calls"] == [{
        "tool": "executor", "params": {"command": "npm test"},
    }]


def test_function_style_tool_markup_is_parsed():
    raw = '''Checking files.
<tool_call>file_locate(query="package.json", project_root="C:/work")
<tool_call>directory_ensure(paths=["src", "tests"])
<tool_call>executor(command="npm run build")'''
    result = normalizer().normalize_action(raw)
    assert result.valid, result.errors
    assert result.value["tool_calls"] == [
        {"tool": "file_locate", "params": {"query": "package.json", "project_root": "C:/work"}},
        {"tool": "directory_ensure", "params": {"paths": ["src", "tests"]}},
        {"tool": "executor", "params": {"command": "npm run build"}},
    ]


def test_arg_key_value_tool_markup_is_parsed():
    raw = '''<tool_call>file_locate>
<arg_key>query</arg_key><arg_value>src/main.ts</arg_value>
<arg_key>project_root</arg_key><arg_value>C:/work</arg_value>'''
    result = normalizer().normalize_action(raw)
    assert result.valid, result.errors
    assert result.value["tool_calls"] == [{
        "tool": "file_locate",
        "params": {"query": "src/main.ts", "project_root": "C:/work"},
    }]


def test_registered_tool_name_used_as_action_is_translated():
    result = normalizer().normalize_action({
        "action": "file_write",
        "params": {"path": "src/a.ts", "content": "export {};"},
    })
    assert result.valid, result.errors
    assert result.value == {
        "action": "tool_calls",
        "tool_calls": [{
            "tool": "file_write",
            "params": {"path": "src/a.ts", "content": "export {};"},
        }],
    }
    assert "translated tool-name action" in result.transformations


def test_complete_tool_calls_are_recovered_from_truncated_json_tail():
    raw = '''{"action":"tool_calls","tool_calls":[
      {"tool":"file_write","params":{"path":"a.ts","content":"export {};"}},
      {"tool":"file_write","params":{"path":"b.ts","content":"export const b=1;"}},
      {"tool":"file_write","params":{"path":"c.ts","content":"unterminated'''
    result = normalizer().normalize_action(raw)
    assert result.valid, result.errors
    assert [call["params"]["path"] for call in result.value["tool_calls"]] == ["a.ts", "b.ts"]
    assert "recovered_truncated_tool_calls" in result.transformations


def test_uniquely_implied_closing_brace_is_repaired():
    raw = '{"action":"tool_calls","tool_calls":[{"tool":"file_write","params":{"path":"a.ts","content":"export {};"}}]'
    result = normalizer().normalize_action(raw)
    assert result.valid, result.errors
    assert result.value["tool_calls"][0]["params"]["path"] == "a.ts"
    assert "closed_truncated_json" in result.transformations


def test_corrupted_markup_with_no_known_parameters_is_rejected():
    response_normalizer = ModelResponseNormalizer({
        "file_locate": {"query": "str", "cwd": "str", "project_root": "str"},
    })
    result = response_normalizer.normalize_action(
        '<tool_call>file_locate nonsense="x" mangled_parameter="y" />'
    )
    assert not result.valid
    assert "no valid calls" in " ".join(result.errors)
    assert any("discarded unknown" in item for item in result.transformations)


def test_registered_action_lifts_schema_parameters_from_top_level():
    result = normalizer().normalize_action({
        "action": "file_read",
        "path": "src/core/scene.ts",
    })
    assert result.valid, result.errors
    assert result.value["tool_calls"] == [{
        "tool": "file_read", "params": {"path": "src/core/scene.ts"},
    }]


def test_tool_call_lifts_schema_parameters_beside_tool_name():
    result = normalizer().normalize_action({
        "action": "tool_calls",
        "tool_calls": [{"tool": "file_read", "path": "src/main.ts"}],
    })
    assert result.valid, result.errors
    assert result.value["tool_calls"] == [{
        "tool": "file_read", "params": {"path": "src/main.ts"},
    }]
