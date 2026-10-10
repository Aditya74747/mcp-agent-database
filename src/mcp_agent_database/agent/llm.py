"""Provider-agnostic chat + tool-calling interface.

The agent loop only ever sees :class:`Message`, :class:`ToolSpec` and :class:`LLMResponse`.
Each adapter translates those to and from one provider's wire format:

* :class:`OllamaLLM`    - local open-source models via Ollama's native ``/api/chat`` (default)
* :class:`OpenAILLM`    - OpenAI or any OpenAI-compatible server (vLLM, LM Studio, Groq, ...)
* :class:`AnthropicLLM` - Claude via the Messages API (sampling params use model defaults)
* :class:`ScriptedLLM`  - deterministic fake for tests and offline demos
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from mcp_agent_database.config import Settings

log = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant", "tool"]
_THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Set when the model produced arguments that are not a JSON object; the loop reports it back.
    parse_error: str | None = None


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # tool name for role == "tool"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class LLM(Protocol):
    model: str

    async def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse: ...


class LLMError(RuntimeError):
    pass


def _new_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Models return arguments as a dict or a JSON string; normalize and report problems."""
    if raw is None or raw == "":
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as err:
            return {}, f"Arguments were not valid JSON ({err.msg}): {raw[:200]}"
        if isinstance(value, dict):
            return value, None
        return {}, f"Arguments must be a JSON object, got {type(value).__name__}."
    return {}, f"Unsupported argument type {type(raw).__name__}."


def strip_thinking(text: str) -> str:
    return _THINK.sub("", text or "").strip()


# =============================================================================================
# Ollama (native API)
# =============================================================================================
class OllamaLLM:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        think: bool | None = False,
        timeout_s: float = 180.0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.think = think
        self.max_retries = max_retries
        self.client = client or httpx.AsyncClient(base_url=base_url, timeout=timeout_s)

    @staticmethod
    def to_wire(messages: Sequence[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            item: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.role == "assistant" and m.tool_calls:
                item["tool_calls"] = [
                    {"id": c.id, "function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
                ]
            if m.role == "tool":
                item["tool_name"] = m.name
            out.append(item)
        return out

    async def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.to_wire(messages),
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in tools
            ]
        if self.think is not None:
            payload["think"] = self.think

        for attempt in range(self.max_retries + 1):
            try:
                resp = await self.client.post("/api/chat", json=payload)
                if resp.status_code == 404:
                    raise LLMError(f"Ollama model '{self.model}' not found. Run: ollama pull {self.model}")
                resp.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as err:
                if attempt == self.max_retries:
                    raise LLMError(f"Ollama request failed: {err}") from err
                await asyncio.sleep(0.5 * 2**attempt)
        data = resp.json()
        msg = data.get("message", {})
        calls = []
        for raw in msg.get("tool_calls") or []:
            fn = raw.get("function", {})
            args, error = parse_arguments(fn.get("arguments"))
            calls.append(ToolCall(raw.get("id") or _new_id(), fn.get("name", ""), args, error))
        return LLMResponse(
            content=strip_thinking(msg.get("content", "")),
            tool_calls=calls,
            usage=Usage(data.get("prompt_eval_count", 0), data.get("eval_count", 0)),
        )


# =============================================================================================
# OpenAI and OpenAI-compatible servers
# =============================================================================================
class OpenAILLM:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_s: float = 180.0,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as err:  # pragma: no cover
                raise LLMError("Install the extra: uv sync --extra openai") from err
            client = AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url, timeout=timeout_s)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def to_wire(messages: Sequence[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            elif m.role == "assistant" and m.tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": m.content or None,
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                            }
                            for c in m.tool_calls
                        ],
                    }
                )
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    async def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self.to_wire(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in tools
            ]
        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except Exception as err:
            raise LLMError(f"OpenAI-compatible request failed: {err}") from err
        choice = resp.choices[0].message
        calls = []
        for raw in choice.tool_calls or []:
            args, error = parse_arguments(raw.function.arguments)
            calls.append(ToolCall(raw.id or _new_id(), raw.function.name, args, error))
        usage = Usage(resp.usage.prompt_tokens, resp.usage.completion_tokens) if resp.usage else Usage()
        return LLMResponse(content=strip_thinking(choice.content or ""), tool_calls=calls, usage=usage)


# =============================================================================================
# Anthropic (Claude)
# =============================================================================================
class AnthropicLLM:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_tokens: int = 2048,
        timeout_s: float = 180.0,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as err:  # pragma: no cover
                raise LLMError("Install the extra: uv sync --extra anthropic") from err
            client = AsyncAnthropic(api_key=api_key, timeout=timeout_s)
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    @staticmethod
    def to_wire(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Anthropic keeps the system prompt separate and sends tool results as user content blocks."""
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
                # Consecutive tool results must share one user turn.
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                blocks += [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in m.tool_calls
                ]
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            else:
                out.append({"role": "user", "content": m.content})
        return system, out

    async def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        system, wire = self.to_wire(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": wire,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ]
        try:
            resp = await self.client.messages.create(**kwargs)
        except Exception as err:
            raise LLMError(f"Anthropic request failed: {err}") from err
        text, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text.append(block.text)
            elif block.type == "tool_use":
                args, error = parse_arguments(block.input)
                calls.append(ToolCall(block.id, block.name, args, error))
        usage = Usage(resp.usage.input_tokens, resp.usage.output_tokens) if resp.usage else Usage()
        return LLMResponse(content="\n".join(text).strip(), tool_calls=calls, usage=usage)


# =============================================================================================
# Scripted fake (tests, CI, offline demos)
# =============================================================================================
ScriptStep = LLMResponse | Callable[[Sequence[Message]], LLMResponse]


class ScriptedLLM:
    """Replays a fixed list of responses (or callables that inspect the conversation)."""

    def __init__(self, steps: Sequence[ScriptStep], model: str = "scripted") -> None:
        self.steps = list(steps)
        self.model = model
        self.calls: list[list[Message]] = []

    async def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        self.calls.append(list(messages))
        if not self.steps:
            return LLMResponse(content="(script finished)")
        step = self.steps.pop(0)
        return step(messages) if callable(step) else step


def create_llm(settings: Settings) -> LLM:
    provider = settings.llm_provider
    if provider == "ollama":
        return OllamaLLM(
            settings.model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            think=settings.ollama_think,
            timeout_s=settings.llm_timeout_s,
        )
    if provider == "openai":
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        return OpenAILLM(
            settings.model,
            api_key=key,
            base_url=settings.openai_base_url,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout_s=settings.llm_timeout_s,
        )
    if provider == "anthropic":
        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        return AnthropicLLM(
            settings.model,
            api_key=key,
            max_tokens=settings.max_tokens,
            timeout_s=settings.llm_timeout_s,
        )
    raise ValueError(f"Unknown provider {provider!r}")
