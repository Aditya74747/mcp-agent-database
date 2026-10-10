from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient
from pydantic import SecretStr

from mcp_agent_database.agent.llm import LLMResponse, ScriptedLLM, ToolCall
from mcp_agent_database.api.app import create_app


def sse_events(lines: Iterator[str]) -> Iterator[tuple[str, dict]]:
    event = "message"
    for line in lines:
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            yield event, json.loads(line[5:].strip())


def scripted(*steps) -> ScriptedLLM:
    return ScriptedLLM(list(steps))


def test_health_tools_and_ui(settings) -> None:
    app = create_app(settings, llm=scripted(), in_process=True)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["tools"] == 16
        tools = {t["name"]: t for t in client.get("/v1/tools").json()}
        assert tools["sql__run_query"]["requires_approval"] is False
        assert tools["workspace__delete_file"]["requires_approval"] is True
        assert "mcp-agent-database" in client.get("/").text


def test_chat_stream_and_session_memory(settings) -> None:
    llm = scripted(
        LLMResponse("", [ToolCall("c1", "sql__run_query", {"sql": "SELECT COUNT(*) FROM products"})]),
        LLMResponse("We sell 40 products."),
        LLMResponse("You asked about products."),
    )
    app = create_app(settings, llm=llm, in_process=True)
    with TestClient(app) as client:
        with client.stream("POST", "/v1/chat", json={"message": "How many products?"}) as resp:
            events = list(sse_events(resp.iter_lines()))
        names = [e for e, _ in events]
        assert names == ["session", "tool_call", "tool_result", "final", "done"]
        assert events[3][1]["content"] == "We sell 40 products."
        session_id = events[0][1]["session_id"]

        with client.stream(
            "POST", "/v1/chat", json={"message": "What did I ask?", "session_id": session_id}
        ) as resp:
            list(resp.iter_lines())
        # Second turn received the first turn's messages as history.
        assert any(m.content == "How many products?" for m in llm.calls[-1])


def test_bearer_token_required_when_configured(settings) -> None:
    settings = settings.model_copy(update={"api_token": SecretStr("s3cret")})
    app = create_app(settings, llm=scripted(), in_process=True)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200  # health stays public for probes
        assert client.get("/v1/tools").status_code == 401
        assert client.get("/v1/tools", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/v1/tools", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_unknown_approval_is_404(settings) -> None:
    with TestClient(create_app(settings, llm=scripted(), in_process=True)) as client:
        assert client.post("/v1/approvals/nope", json={"approved": True}).status_code == 404
        assert client.get("/v1/approvals").json() == []


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.integration
@pytest.mark.parametrize("approve", [True, False])
def test_live_approval_flow(settings, approve: bool) -> None:
    """Start a real server, stream a turn that pauses for approval, approve/deny it over HTTP."""
    llm = scripted(
        LLMResponse("", [ToolCall("w1", "workspace__write_file", {"path": "out.md", "content": "report"})]),
        LLMResponse("Finished."),
    )
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings, llm=llm, in_process=True), port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(f"{base}/health").status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.05)
        seen = []
        with httpx.stream(
            "POST", f"{base}/v1/chat", json={"message": "write the report"}, timeout=20
        ) as resp:
            for event, data in sse_events(resp.iter_lines()):
                seen.append(event)
                if event == "approval_required":
                    assert httpx.get(f"{base}/v1/approvals").json()[0]["id"] == data["id"]
                    r = httpx.post(f"{base}/v1/approvals/{data['id']}", json={"approved": approve})
                    assert r.status_code == 200
                if event == "tool_result":
                    assert data["status"] == ("ok" if approve else "denied")
        assert seen[-2:] == ["final", "done"]
        assert (settings.sandbox_dir / "out.md").exists() is approve
    finally:
        server.should_exit = True
        thread.join(timeout=10)
