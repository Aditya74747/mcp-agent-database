from __future__ import annotations

import json

import httpx
import pytest
import respx

from mcp_agent_database.agent.llm import (
    AnthropicLLM,
    LLMError,
    Message,
    OllamaLLM,
    OpenAILLM,
    ToolCall,
    ToolSpec,
    create_llm,
    parse_arguments,
    strip_thinking,
)
from mcp_agent_database.config import Settings

pytestmark = pytest.mark.anyio

TOOLS = [ToolSpec("sql__run_query", "Run SQL", {"type": "object", "properties": {"sql": {"type": "string"}}})]
CONVERSATION = [
    Message("system", "be helpful"),
    Message("user", "count orders"),
    Message("assistant", "", tool_calls=[ToolCall("c1", "sql__run_query", {"sql": "SELECT 1"})]),
    Message("tool", '{"rows": [[1]]}', tool_call_id="c1", name="sql__run_query"),
]


@pytest.mark.parametrize(
    ("raw", "expected", "error"),
    [
        ({"a": 1}, {"a": 1}, False),
        ('{"a": 1}', {"a": 1}, False),
        ("", {}, False),
        (None, {}, False),
        ("{bad json", {}, True),
        ("[1, 2]", {}, True),
    ],
)
def test_parse_arguments(raw, expected, error) -> None:
    args, err = parse_arguments(raw)
    assert args == expected
    assert (err is not None) is error


def test_strip_thinking() -> None:
    assert strip_thinking("<think>hmm\nlong</think>\nAnswer") == "Answer"


@respx.mock
async def test_ollama_wire_format_and_parsing() -> None:
    route = respx.post("http://ollama.test/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "<think>plan</think>",
                    "tool_calls": [
                        {"function": {"name": "sql__run_query", "arguments": {"sql": "SELECT 2"}}}
                    ],
                },
                "prompt_eval_count": 120,
                "eval_count": 15,
            },
        )
    )
    llm = OllamaLLM("qwen3:8b", base_url="http://ollama.test", think=False)
    resp = await llm.chat(CONVERSATION, TOOLS)

    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "qwen3:8b" and body["stream"] is False and body["think"] is False
    assert body["tools"][0]["function"]["name"] == "sql__run_query"
    assert body["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"sql": "SELECT 1"}
    assert body["messages"][3] == {
        "role": "tool",
        "content": '{"rows": [[1]]}',
        "tool_name": "sql__run_query",
    }

    assert resp.content == ""
    assert resp.tool_calls[0].name == "sql__run_query"
    assert resp.tool_calls[0].arguments == {"sql": "SELECT 2"}
    assert resp.tool_calls[0].id.startswith("call_")
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (120, 15)


@respx.mock
async def test_ollama_missing_model_message() -> None:
    respx.post("http://ollama.test/api/chat").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    with pytest.raises(LLMError, match="ollama pull"):
        await OllamaLLM("nope", base_url="http://ollama.test").chat(CONVERSATION, TOOLS)


@respx.mock
async def test_ollama_retries_transient_errors() -> None:
    route = respx.post("http://ollama.test/api/chat").mock(
        side_effect=[httpx.ConnectError("boom"), httpx.Response(200, json={"message": {"content": "hi"}})]
    )
    resp = await OllamaLLM("m", base_url="http://ollama.test").chat([Message("user", "x")], [])
    assert resp.content == "hi" and route.call_count == 2


class FakeEndpoint:
    """Stands in for `client.chat.completions` / `client.messages`; records the request kwargs."""

    def __init__(self, response) -> None:
        self.response = response
        self.kwargs: dict = {}

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


async def test_openai_compatible_round_trip() -> None:
    from types import SimpleNamespace

    from openai.types.chat import ChatCompletion

    endpoint = FakeEndpoint(
        ChatCompletion.model_validate(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_9",
                                    "type": "function",
                                    "function": {
                                        "name": "sql__run_query",
                                        "arguments": '{"sql": "SELECT 3"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 9, "total_tokens": 59},
            }
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
    resp = await OpenAILLM("m", client=client).chat(CONVERSATION, TOOLS)
    body = endpoint.kwargs
    assert body["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"sql": "SELECT 1"}'
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": '{"rows": [[1]]}'}
    assert resp.tool_calls[0].id == "call_9" and resp.tool_calls[0].arguments == {"sql": "SELECT 3"}
    assert resp.usage.input_tokens == 50


async def test_anthropic_round_trip() -> None:
    from types import SimpleNamespace

    from anthropic.types import Message as AnthropicMessage

    endpoint = FakeEndpoint(
        AnthropicMessage.model_validate(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude",
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "sql__run_query",
                        "input": {"sql": "SELECT 4"},
                    },
                ],
                "usage": {"input_tokens": 70, "output_tokens": 20},
            }
        )
    )
    resp = await AnthropicLLM("claude-test", client=SimpleNamespace(messages=endpoint)).chat(
        CONVERSATION, TOOLS
    )
    body = endpoint.kwargs
    assert body["system"] == "be helpful"
    assert body["tools"][0]["input_schema"]["properties"]["sql"]["type"] == "string"
    assert body["messages"][1]["content"][0] == {
        "type": "tool_use",
        "id": "c1",
        "name": "sql__run_query",
        "input": {"sql": "SELECT 1"},
    }
    assert body["messages"][2]["content"][0]["type"] == "tool_result"
    assert "temperature" not in body
    assert resp.content == "Let me check."
    assert resp.tool_calls[0].arguments == {"sql": "SELECT 4"}


def test_anthropic_merges_consecutive_tool_results() -> None:
    msgs = [
        Message("user", "q"),
        Message("assistant", "", tool_calls=[ToolCall("a", "t", {}), ToolCall("b", "t", {})]),
        Message("tool", "1", tool_call_id="a", name="t"),
        Message("tool", "2", tool_call_id="b", name="t"),
    ]
    _, wire = AnthropicLLM.to_wire(msgs)
    assert len(wire) == 3
    assert [b["tool_use_id"] for b in wire[2]["content"]] == ["a", "b"]


@pytest.mark.parametrize(
    ("provider", "cls"), [("ollama", OllamaLLM), ("openai", OpenAILLM), ("anthropic", AnthropicLLM)]
)
def test_factory(provider, cls) -> None:
    s = Settings(_env_file=None, llm_provider=provider, model="m", openai_api_key="k", anthropic_api_key="k")
    assert isinstance(create_llm(s), cls)
