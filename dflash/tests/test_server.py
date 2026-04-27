import json
import struct
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from dflash.server import build_app


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
