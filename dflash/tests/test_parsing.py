import json
import pytest
from dflash.server.parsing import (
    normalize_stop,
    first_stop_match,
    parse_reasoning,
    parse_tool_calls,
)


# ── normalize_stop ─────────────────────────────────────────────────

def test_normalize_stop_none():
    assert normalize_stop(None) == []

def test_normalize_stop_empty_list():
    assert normalize_stop([]) == []

def test_normalize_stop_string():
    assert normalize_stop("STOP") == ["STOP"]

def test_normalize_stop_list():
    assert normalize_stop(["a", "b"]) == ["a", "b"]

def test_normalize_stop_filters_non_strings():
    assert normalize_stop(["a", None, 42, "b"]) == ["a", "b"]


# ── first_stop_match ───────────────────────────────────────────────

def test_first_stop_match_not_found():
    assert first_stop_match("hello world", ["STOP"]) == -1

def test_first_stop_match_found():
    assert first_stop_match("hello STOP world", ["STOP"]) == 6

def test_first_stop_match_earliest():
    assert first_stop_match("aXbYc", ["Y", "X"]) == 1

def test_first_stop_match_empty_stops():
    assert first_stop_match("hello", []) == -1


# ── parse_reasoning ────────────────────────────────────────────────

def test_parse_reasoning_paired_tags():
    content, reasoning = parse_reasoning("<think>deep thought</think>the answer")
    assert content == "the answer"
    assert reasoning == "deep thought"

def test_parse_reasoning_headless():
    # Template prefilled <think>; model only emits the close tag.
    content, reasoning = parse_reasoning("deep thought</think>the answer")
    assert content == "the answer"
    assert reasoning == "deep thought"

def test_parse_reasoning_disabled():
    content, reasoning = parse_reasoning("plain content", thinking_enabled=False)
    assert content == "plain content"
    assert reasoning is None

def test_parse_reasoning_truncated():
    # No </think> and thinking enabled — everything is reasoning.
    content, reasoning = parse_reasoning("incomplete thought", thinking_enabled=True)
    assert content == ""
    assert reasoning == "incomplete thought"

def test_parse_reasoning_no_tags():
    content, reasoning = parse_reasoning("plain content", thinking_enabled=True)
    # No </think> → treated as truncated reasoning.
    assert content == ""
    assert reasoning == "plain content"

def test_parse_reasoning_empty_think():
    content, reasoning = parse_reasoning("<think></think>answer")
    assert content == "answer"
    assert reasoning is None  # empty string stripped to None


# ── parse_tool_calls ───────────────────────────────────────────────

def _make_tool_xml(name, params: dict) -> str:
    param_str = "".join(
        f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items()
    )
    return f"<tool_call>\n<function={name}>\n{param_str}</function>\n</tool_call>"


def test_parse_tool_calls_no_calls():
    content, calls = parse_tool_calls("just some text")
    assert content == "just some text"
    assert calls == []

def test_parse_tool_calls_basic():
    xml = _make_tool_xml("get_weather", {"location": "London"})
    content, calls = parse_tool_calls(xml)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["location"] == "London"

def test_parse_tool_calls_multiple():
    xml = (_make_tool_xml("fn_a", {"x": "1"}) + "\n" +
           _make_tool_xml("fn_b", {"y": "2"}))
    _, calls = parse_tool_calls(xml)
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "fn_a"
    assert calls[1]["function"]["name"] == "fn_b"

def test_parse_tool_calls_cleans_surrounding_text():
    xml = "Before. " + _make_tool_xml("fn", {"k": "v"}) + " After."
    content, calls = parse_tool_calls(xml)
    assert "Before." in content
    assert "After." in content
    assert len(calls) == 1

def test_parse_tool_calls_type_coercion_int():
    tools = [{"function": {"name": "fn", "parameters": {
        "properties": {"count": {"type": "integer"}}}}}]
    xml = _make_tool_xml("fn", {"count": "42"})
    _, calls = parse_tool_calls(xml, tools=tools)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["count"] == 42
    assert isinstance(args["count"], int)

def test_parse_tool_calls_type_coercion_bool():
    tools = [{"function": {"name": "fn", "parameters": {
        "properties": {"flag": {"type": "boolean"}}}}}]
    xml = _make_tool_xml("fn", {"flag": "true"})
    _, calls = parse_tool_calls(xml, tools=tools)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["flag"] is True

def test_parse_tool_calls_type_coercion_object():
    tools = [{"function": {"name": "fn", "parameters": {
        "properties": {"data": {"type": "object"}}}}}]
    xml = _make_tool_xml("fn", {"data": '{"key": "val"}'})
    _, calls = parse_tool_calls(xml, tools=tools)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["data"] == {"key": "val"}

def test_parse_tool_calls_unclosed_tag():
    # Malformed — no </tool_call>; should return as plain content.
    text = "<tool_call>\n<function=fn>\n<parameter=x>\nhello\n</parameter>\n</function>\n"
    content, calls = parse_tool_calls(text)
    assert calls == []

def test_parse_tool_calls_ids_are_unique():
    xml = (_make_tool_xml("fn_a", {"x": "1"}) + _make_tool_xml("fn_b", {"y": "2"}))
    _, calls = parse_tool_calls(xml)
    ids = [c["id"] for c in calls]
    assert len(set(ids)) == len(ids)
