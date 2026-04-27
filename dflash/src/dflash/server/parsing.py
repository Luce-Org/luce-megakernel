"""
Qwen3.x tool-call and reasoning parsers.

Ported from vLLM (Apache-2.0) for behavioral parity with
--tool-call-parser qwen3_coder and --reasoning-parser qwen3.
Pure functions — no FastAPI or subprocess dependencies.
"""
import json
import re
import uuid


TOOL_OPEN_TAG = "<tool_call>"
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"

TOOL_CALL_COMPLETE_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_CALL_FUNCTION_RE = re.compile(
    r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL,
)
# Tolerates unclosed </parameter> by using the next tag or end-of-string as terminator.
TOOL_CALL_PARAMETER_RE = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)


def normalize_stop(stop) -> list[str]:
    """Coerce OpenAI stop field (str | list[str] | None) to list[str]."""
    if not stop:
        return []
    if isinstance(stop, str):
        return [stop]
    return [s for s in stop if isinstance(s, str) and s]


def first_stop_match(text: str, stops: list[str]) -> int:
    """Return the earliest index where any stop sequence appears, or -1."""
    best = -1
    for s in stops:
        i = text.find(s)
        if i != -1 and (best == -1 or i < best):
            best = i
    return best


def parse_reasoning(text: str, thinking_enabled: bool = True) -> tuple[str, str | None]:
    """Extract <think>...</think> reasoning from model output.

    Handles three Qwen3.x flavors:
      - Paired:   both <think> and </think> in generated output.
      - Headless: template prefilled <think>, model only emits </think>.
      - Disabled: thinking disabled; output is pure content.

    Returns (content, reasoning_content).
    """
    parts = text.partition(THINK_OPEN_TAG)
    rest = parts[2] if parts[1] else parts[0]
    if THINK_CLOSE_TAG not in rest:
        if thinking_enabled:
            # Truncated mid-think — treat everything as reasoning.
            return "", (rest.strip() or None)
        else:
            return rest.strip(), None
    reasoning, _, content = rest.partition(THINK_CLOSE_TAG)
    return content.strip(), (reasoning.strip() or None)


def _find_tool_properties(tools, function_name: str) -> dict:
    for t in tools or []:
        fn = t.function if hasattr(t, "function") else t.get("function", {})
        if hasattr(fn, "model_dump"):
            fn = fn.model_dump()
        if fn.get("name") == function_name:
            params = fn.get("parameters", {})
            if isinstance(params, dict):
                return params.get("properties", {})
    return {}


def _convert_param_value(param_value: str, param_name: str, param_config: dict,
                         func_name: str):
    """Coerce a stringified XML parameter value to its JSON-schema type."""
    import ast
    if param_value.lower() == "null":
        return None
    if param_name not in param_config:
        return param_value
    cfg = param_config[param_name]
    if isinstance(cfg, dict) and "type" in cfg:
        ptype = str(cfg["type"]).strip().lower()
    elif isinstance(cfg, dict) and "anyOf" in cfg:
        ptype = "object"
    else:
        ptype = "string"
    if ptype in ("string", "str", "text", "varchar", "char", "enum"):
        return param_value
    if any(ptype.startswith(p) for p in ("int", "uint", "long", "short", "unsigned")):
        try: return int(param_value)
        except (ValueError, TypeError): return param_value
    if ptype.startswith("num") or ptype.startswith("float"):
        try:
            f = float(param_value)
            return f if f - int(f) != 0 else int(f)
        except (ValueError, TypeError):
            return param_value
    if ptype in ("boolean", "bool", "binary"):
        return param_value.lower() == "true"
    if (ptype in ("object", "array", "arr")
            or ptype.startswith("dict") or ptype.startswith("list")):
        try: return json.loads(param_value)
        except (json.JSONDecodeError, TypeError, ValueError): pass
    try: return ast.literal_eval(param_value)
    except (ValueError, SyntaxError, TypeError): return param_value


def parse_tool_calls(text: str, tools=None) -> tuple[str, list[dict]]:
    """Parse Qwen3.x XML tool calls out of model output.

    Returns (cleaned_content, tool_calls_list).
    """
    tool_calls: list[dict] = []
    cleaned_parts: list[str] = []
    cursor = 0
    for m in TOOL_CALL_COMPLETE_RE.finditer(text):
        cleaned_parts.append(text[cursor:m.start()])
        cursor = m.end()
        body = m.group(1)
        fn_match = TOOL_CALL_FUNCTION_RE.search(body)
        if not fn_match:
            continue
        fn_text = fn_match.group(1) or fn_match.group(2) or ""
        end_idx = fn_text.find(">")
        if end_idx == -1:
            continue
        function_name = fn_text[:end_idx].strip()
        params_region = fn_text[end_idx + 1:]
        param_config = _find_tool_properties(tools, function_name)
        args: dict = {}
        for match_text in TOOL_CALL_PARAMETER_RE.findall(params_region):
            eq_idx = match_text.find(">")
            if eq_idx == -1:
                continue
            k = match_text[:eq_idx].strip()
            v = match_text[eq_idx + 1:]
            if v.startswith("\n"): v = v[1:]
            if v.endswith("\n"): v = v[:-1]
            args[k] = _convert_param_value(v, k, param_config, function_name)
        tool_calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    cleaned_parts.append(text[cursor:])
    return "".join(cleaned_parts).strip(), tool_calls
