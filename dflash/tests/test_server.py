import json
import struct
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from dflash.server import (
    AGENT_CODE_TEXT_PROFILE,
    admission_for_prompt,
    build_app,
    configure_server_environment,
    resolve_server_profile,
)


MODEL = "test-model"


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.decode.return_value = "hello"
    tok.apply_chat_template.return_value = "<prompt>"
    return tok


@pytest.fixture
def app(mock_tokenizer):
    with patch("dflash.server.subprocess.Popen"), \
         patch("dflash.server.os.pipe", return_value=(10, 11)), \
         patch("dflash.server.os.close"):
        return build_app(
            target=Path("target.gguf"),
            draft=Path("draft.safetensors"),
            bin_path=Path("test_dflash"),
            budget=22,
            max_ctx=131072,
            tokenizer=mock_tokenizer,
            stop_ids={2},
            model_name=MODEL,
        )


@pytest.fixture
def client(app):
    return TestClient(app)


def test_agent_code_text_profile_defaults():
    profile = resolve_server_profile("agent-code-text")
    assert profile.max_ctx == 48_000
    assert profile.max_prompt_tokens == 42_000
    assert profile.prefill_ubatch == 384
    assert profile.layer_prefill is False
    assert profile.fa_window == 0
    assert profile.vision is False
    assert profile.prefix_cache is False


def test_agent_code_text_profile_overrides_stay_explicit():
    profile = resolve_server_profile("agent-code-text", prefill_ubatch=256, max_prompt_tokens=40_000)
    assert profile.prefill_ubatch == 256
    assert profile.max_prompt_tokens == 40_000
    assert profile.max_ctx == 48_000


def test_agent_code_text_admission_reserves_prompt_limit():
    rejected = admission_for_prompt(42_001, 1024, AGENT_CODE_TEXT_PROFILE)
    assert rejected.allowed is False
    accepted = admission_for_prompt(42_000, 8_192, AGENT_CODE_TEXT_PROFILE)
    assert accepted.allowed is True
    assert accepted.generation_tokens == 5_980


def test_agent_code_text_admission_rejects_non_positive_generation():
    zero = admission_for_prompt(100, 0, AGENT_CODE_TEXT_PROFILE)
    negative = admission_for_prompt(100, -1, AGENT_CODE_TEXT_PROFILE)
    assert zero.allowed is False
    assert negative.allowed is False


def test_agent_code_text_admission_enforces_reserved_generation_tokens():
    profile = resolve_server_profile("agent-code-text", max_prompt_tokens=47_000)
    rejected = admission_for_prompt(44_000, 8_192, profile)
    accepted = admission_for_prompt(43_884, 8_192, profile)
    assert rejected.allowed is False
    assert accepted.allowed is True
    assert accepted.generation_tokens == 4_096


def test_configure_server_environment_sets_safe_profile_defaults(monkeypatch):
    monkeypatch.delenv("DFLASH27B_PREFILL_UBATCH", raising=False)
    monkeypatch.delenv("DFLASH27B_LAYER_PREFILL", raising=False)
    monkeypatch.delenv("DFLASH27B_FA_WINDOW", raising=False)
    monkeypatch.delenv("DFLASH27B_KV_TQ3", raising=False)
    monkeypatch.delenv("DFLASH27B_KV_K", raising=False)
    monkeypatch.delenv("DFLASH27B_KV_V", raising=False)

    configure_server_environment(AGENT_CODE_TEXT_PROFILE, kv_f16=False)

    import os
    assert os.environ["DFLASH27B_PREFILL_UBATCH"] == "384"
    assert os.environ["DFLASH27B_LAYER_PREFILL"] == "0"
    assert os.environ["DFLASH27B_FA_WINDOW"] == "0"
    assert os.environ["DFLASH27B_KV_TQ3"] == "1"
    assert "DFLASH27B_KV_K" not in os.environ
    assert "DFLASH27B_KV_V" not in os.environ


def test_configure_server_environment_passthroughs_kv_cache_types(monkeypatch):
    monkeypatch.setenv("DFLASH27B_KV_TQ3", "1")
    monkeypatch.delenv("DFLASH27B_KV_K", raising=False)
    monkeypatch.delenv("DFLASH27B_KV_V", raising=False)

    configure_server_environment(
        AGENT_CODE_TEXT_PROFILE,
        kv_f16=False,
        cache_type_k="q4_0",
        cache_type_v="q8_0",
    )

    import os
    assert os.environ["DFLASH27B_KV_K"] == "q4_0"
    assert os.environ["DFLASH27B_KV_V"] == "q8_0"
    assert "DFLASH27B_KV_TQ3" not in os.environ


@pytest.fixture
def agent_profile_app(mock_tokenizer):
    with patch("dflash.server.subprocess.Popen") as popen, \
         patch("dflash.server.os.pipe", return_value=(10, 11)), \
         patch("dflash.server.os.close"):
        app = build_app(
            target=Path("target.gguf"),
            draft=Path("draft.safetensors"),
            bin_path=Path("test_dflash"),
            budget=22,
            tokenizer=mock_tokenizer,
            stop_ids={2},
            model_name=MODEL,
            profile=AGENT_CODE_TEXT_PROFILE,
        )
        app.state.daemon_proc_mock = popen.return_value
        return app


@pytest.fixture
def agent_profile_client(agent_profile_app):
    return TestClient(agent_profile_app)


# ── /v1/models ─────────────────────────────────────────────────────

def test_models_endpoint(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == MODEL


# ── /v1/chat/completions — non-streaming ──────────────────────────

def _token_bytes(*ids):
    return [struct.pack("<i", i) for i in ids]


@patch("dflash.server.os.read")
def test_chat_non_streaming(mock_read, client):
    mock_read.side_effect = _token_bytes(10, -1)
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == MODEL
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["finish_reason"] == "stop"


@patch("dflash.server.os.read")
def test_chat_non_streaming_tool_call(mock_read, client, mock_tokenizer):
    tool_xml = (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=location>\nLondon\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    mock_tokenizer.decode.return_value = tool_xml
    mock_read.side_effect = _token_bytes(10, -1)
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
        }}],
        "stream": False,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    tc = data["choices"][0]["message"]["tool_calls"]
    assert len(tc) == 1
    assert tc[0]["function"]["name"] == "get_weather"
    args = json.loads(tc[0]["function"]["arguments"])
    assert args["location"] == "London"


@patch("dflash.server.os.read")
def test_chat_non_streaming_stop_sequence(mock_read, client, mock_tokenizer):
    mock_tokenizer.decode.return_value = "hello STOP world"
    mock_read.side_effect = _token_bytes(10, -1)
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "stop": "STOP",
        "stream": False,
        # Disable thinking so parse_reasoning doesn't treat plain text as
        # truncated reasoning (which would clear content).
        "chat_template_kwargs": {"enable_thinking": False},
    })
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hello"


def test_chat_context_exceeded(client, mock_tokenizer):
    # encode returns a list long enough to exceed max_ctx
    mock_tokenizer.encode.return_value = list(range(131072))
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })
    assert r.status_code == 400


def test_agent_profile_rejects_openai_image_payload(agent_profile_client):
    r = agent_profile_client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "file:///tmp/x.png"}},
        ]}],
        "stream": False,
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_content"


def test_agent_profile_rejects_openai_non_text_content_block(agent_profile_client):
    r = agent_profile_client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "transcribe"},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        ]}],
        "stream": False,
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_content"


def test_agent_profile_rejects_openai_zero_max_tokens(agent_profile_client):
    r = agent_profile_client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 0,
        "stream": False,
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "context_limit"


def test_agent_profile_rejects_openai_negative_max_tokens(agent_profile_client):
    r = agent_profile_client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": -1,
        "stream": False,
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "context_limit"


def test_agent_profile_rejects_prompt_above_safe_limit(agent_profile_client, mock_tokenizer):
    mock_tokenizer.encode.return_value = list(range(AGENT_CODE_TEXT_PROFILE.max_prompt_tokens + 1))
    r = agent_profile_client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "context_limit"


@patch("dflash.server.os.read")
def test_agent_profile_uses_legacy_run_command_without_prefix_cache(mock_read, agent_profile_client, agent_profile_app):
    mock_read.side_effect = _token_bytes(10, -1)
    r = agent_profile_client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })
    assert r.status_code == 200
    written = agent_profile_app.state.daemon_proc_mock.stdin.write.call_args[0][0].decode()
    assert not written.startswith("RUN_PREFIX ")


# ── /v1/chat/completions — streaming ─────────────────────────────

@patch("dflash.server.os.read")
def test_chat_streaming(mock_read, client):
    mock_read.side_effect = _token_bytes(10, -1)
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert r.status_code == 200
    events = [l for l in r.text.split("\n\n") if l.startswith("data:")]
    assert events[-1] == "data: [DONE]"
    # First event is role header, middle events are content deltas, last is finish.
    payloads = [json.loads(e[len("data: "):]) for e in events[:-1]]
    finish_events = [p for p in payloads if p["choices"][0].get("finish_reason")]
    assert finish_events[-1]["choices"][0]["finish_reason"] == "stop"


# ── tool_choice ───────────────────────────────────────────────────

_INSPECT_TOOL = {"type": "function", "function": {
    "name": "inspect", "description": "Inspect a target.",
    "parameters": {"type": "object", "properties": {"target": {"type": "string"}}},
}}


@patch("dflash.server.os.read")
def test_tool_choice_none_suppresses_tools(mock_read, client, mock_tokenizer):
    mock_read.side_effect = _token_bytes(10, -1)
    mock_tokenizer.decode.return_value = "I cannot call tools."
    client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [_INSPECT_TOOL],
        "tool_choice": "none",
        "stream": False,
    })
    call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
    assert "tools" not in call_kwargs


@patch("dflash.server.os.read")
def test_tool_choice_required_forwarded(mock_read, client, mock_tokenizer):
    mock_read.side_effect = _token_bytes(10, -1)
    mock_tokenizer.decode.return_value = (
        "<tool_call><function=inspect><parameter=target>x</parameter></function></tool_call>"
    )
    client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [_INSPECT_TOOL],
        "tool_choice": "required",
        "stream": False,
    })
    call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
    assert call_kwargs.get("tool_choice") == "required"


@patch("dflash.server.os.read")
def test_tool_choice_named_function_forwarded(mock_read, client, mock_tokenizer):
    mock_read.side_effect = _token_bytes(10, -1)
    mock_tokenizer.decode.return_value = (
        "<tool_call><function=inspect><parameter=target>x</parameter></function></tool_call>"
    )
    tc = {"type": "function", "function": {"name": "inspect"}}
    client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [_INSPECT_TOOL],
        "tool_choice": tc,
        "stream": False,
    })
    call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
    assert call_kwargs.get("tool_choice") == tc


def test_tool_choice_invalid_returns_400(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": "hallucinated_value",
        "stream": False,
    })
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "tool_choice"


@patch("dflash.server.os.read")
def test_tool_choice_required_forwarded_streaming(mock_read, client, mock_tokenizer):
    mock_read.side_effect = _token_bytes(10, -1)
    mock_tokenizer.decode.return_value = ""
    client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [_INSPECT_TOOL],
        "tool_choice": "required",
        "stream": True,
    })
    call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
    assert call_kwargs.get("tool_choice") == "required"


@patch("dflash.server.os.read")
def test_tool_choice_none_suppresses_tools_streaming(mock_read, client, mock_tokenizer):
    mock_read.side_effect = _token_bytes(10, -1)
    mock_tokenizer.decode.return_value = ""
    client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [_INSPECT_TOOL],
        "tool_choice": "none",
        "stream": True,
    })
    call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
    assert "tools" not in call_kwargs


# ── /v1/messages (Anthropic) ───────────────────────────────────────

@patch("dflash.server.os.read")
def test_anthropic_non_streaming(mock_read, client):
    mock_read.side_effect = _token_bytes(10, -1)
    r = client.post("/v1/messages", json={
        "model": MODEL,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["stop_reason"] == "end_turn"
    assert data["content"][0]["type"] == "text"


@patch("dflash.server.os.read")
def test_anthropic_streaming(mock_read, client):
    mock_read.side_effect = _token_bytes(10, -1)
    r = client.post("/v1/messages", json={
        "model": MODEL,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert r.status_code == 200
    event_types = [
        json.loads(l.split("data: ", 1)[1])["type"]
        for l in r.text.split("\n")
        if l.startswith("data:")
    ]
    assert event_types[0] == "message_start"
    assert "content_block_start" in event_types
    assert "message_stop" in event_types


def test_anthropic_context_exceeded(client, mock_tokenizer):
    mock_tokenizer.encode.return_value = list(range(131072))
    r = client.post("/v1/messages", json={
        "model": MODEL,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 400
    assert r.json()["type"] == "error"


def test_agent_profile_rejects_anthropic_image_payload(agent_profile_client):
    r = agent_profile_client.post("/v1/messages", json={
        "model": MODEL,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}},
        ]}],
    })
    assert r.status_code == 400
    assert r.json()["type"] == "error"


def test_agent_profile_rejects_anthropic_non_text_content_block(agent_profile_client):
    r = agent_profile_client.post("/v1/messages", json={
        "model": MODEL,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "read"},
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "x"}},
        ]}],
    })
    assert r.status_code == 400
    assert r.json()["type"] == "error"


def test_agent_profile_rejects_anthropic_zero_max_tokens(agent_profile_client):
    r = agent_profile_client.post("/v1/messages", json={
        "model": MODEL,
        "max_tokens": 0,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 400
    assert r.json()["type"] == "error"
