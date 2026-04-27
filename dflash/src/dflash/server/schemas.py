from typing import Any
from pydantic import BaseModel


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON string per OpenAI spec


class ToolCall(BaseModel):
    id: str | None = None
    type: str = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: str
    content: Any | None = None  # str, list, or null when tool_calls present
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolDef(BaseModel):
    type: str = "function"
    function: dict  # {name, description, parameters: {...JSON schema...}}


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 512
    temperature: float | None = None  # accepted, ignored (greedy-only)
    top_p: float | None = None
    tools: list[ToolDef] | None = None
    tool_choice: Any | None = None
    chat_template_kwargs: dict | None = None  # e.g. {"enable_thinking": false}
    stop: Any | None = None  # str or list[str]
    stream_options: dict | None = None  # e.g. {"include_usage": true}


class AnthropicMessage(BaseModel):
    role: str
    content: str | list[dict]


class AnthropicMessagesRequest(BaseModel):
    model: str = ""
    max_tokens: int
    messages: list[AnthropicMessage]
    system: str | list[dict] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
