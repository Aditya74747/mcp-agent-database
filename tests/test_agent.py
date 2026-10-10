from __future__ import annotations

import json

import pytest

from mcp_agent_database.agent.approval import (
    ApprovalBroker,
    ApprovalPolicy,
    ApprovalRequest,
    AuditLog,
    CallbackApprover,
    StaticApprover,
)
from mcp_agent_database.agent.host import RegisteredTool
from mcp_agent_database.agent.llm import LLMError, LLMResponse, Message, ScriptedLLM, ToolCall
from mcp_agent_database.agent.loop import Agent
from mcp_agent_database.agent.runtime import build_host

pytestmark = pytest.mark.anyio


def call(name: str, **args) -> LLMResponse:
    return LLMResponse("", [ToolCall(f"id-{name}", name, args)])


def agent_for(host, steps, approver=None, settings=None, **kw) -> Agent:
    return Agent(
        ScriptedLLM(steps),
        host,
        approver=approver or StaticApprover(False),
        audit=AuditLog(settings.audit_log if settings else None),
        **kw,
    )


def types(result) -> list[str]:
    return [e.type for e in result.events]


async def test_read_only_tool_runs_without_approval(settings) -> None:
    async with build_host(settings, in_process=True) as host:
        agent = agent_for(
            host, [call("sql__run_query", sql="SELECT COUNT(*) FROM customers"), LLMResponse("250")]
        )
        result = await agent.run("how many customers?")
    assert types(result) == ["tool_call", "tool_result", "final"]
    assert result.answer == "250" and result.steps == 2 and result.tool_errors == 0
    # The model saw the tool result on its second turn.
    tool_msg = agent.llm.calls[1][-1]  # type: ignore[attr-defined]
    assert tool_msg.role == "tool" and json.loads(tool_msg.content)["rows"] == [[250]]


async def test_denied_action_is_not_executed(settings) -> None:
    async with build_host(settings, in_process=True) as host:
        agent = agent_for(
            host, [call("workspace__delete_file", path="todo.txt"), LLMResponse("ok")], settings=settings
        )
        result = await agent.run("delete todo")
    assert types(result) == ["tool_call", "approval_required", "approval_decision", "tool_result", "final"]
    assert result.events[3].data["status"] == "denied"
    assert (settings.sandbox_dir / "todo.txt").exists()
    audit = [json.loads(line) for line in settings.audit_log.read_text().splitlines()]
    approval = next(r for r in audit if r["event"] == "approval")
    assert approval["approved"] is False and approval["tool"] == "workspace__delete_file"


async def test_approved_action_runs(settings) -> None:
    async with build_host(settings, in_process=True) as host:
        agent = agent_for(
            host,
            [call("workspace__write_file", path="r.md", content="hi"), LLMResponse("done")],
            approver=StaticApprover(True),
        )
        result = await agent.run("write")
    assert result.events[-2].data["status"] == "ok"
    assert (settings.sandbox_dir / "r.md").read_text() == "hi"


async def test_trusted_tools_skip_approval(settings) -> None:
    async with build_host(settings, in_process=True) as host:
        agent = agent_for(
            host,
            [call("workspace__write_file", path="t.md", content="x"), LLMResponse("ok")],
            policy=ApprovalPolicy(["workspace__write_file"]),
        )
        result = await agent.run("write")
    assert "approval_required" not in types(result)
    assert (settings.sandbox_dir / "t.md").exists()


async def test_unknown_tool_and_bad_arguments_are_reported_to_model(settings) -> None:
    bad_args = LLMResponse("", [ToolCall("x", "sql__run_query", {}, parse_error="not JSON")])
    async with build_host(settings, in_process=True) as host:
        agent = agent_for(host, [call("sql__drop_everything"), bad_args, LLMResponse("sorry")])
        result = await agent.run("?")
    results = [e.data for e in result.events if e.type == "tool_result"]
    assert "Unknown tool" in results[0]["content"] and "sql__run_query" in results[0]["content"]
    assert "Invalid arguments" in results[1]["content"]
    assert result.tool_errors == 2 and result.answer == "sorry"


async def test_tool_errors_allow_self_correction(settings) -> None:
    async with build_host(settings, in_process=True) as host:
        agent = agent_for(
            host,
            [
                call("sql__run_query", sql="SELECT * FROM order"),  # typo -> DuckDB error
                call("sql__run_query", sql="SELECT COUNT(*) FROM orders"),
                LLMResponse("2500 orders"),
            ],
        )
        result = await agent.run("count orders")
    statuses = [e.data["status"] for e in result.events if e.type == "tool_result"]
    assert statuses == ["error", "ok"] and result.stopped == "final"


async def test_max_steps(settings) -> None:
    async with build_host(settings, in_process=True) as host:
        agent = agent_for(host, [call("sql__list_tables")] * 5, max_steps=3)
        result = await agent.run("loop forever")
    assert result.stopped == "max_steps" and result.steps == 3


async def test_llm_errors_end_the_turn_cleanly(settings) -> None:
    def boom(_messages):
        raise LLMError("model offline")

    async with build_host(settings, in_process=True) as host:
        result = await agent_for(host, [boom]).run("hi")
    assert result.answer == "Error: model offline" and types(result) == ["error"]


async def test_history_and_transcript(settings) -> None:
    async with build_host(settings, in_process=True) as host:
        agent = agent_for(host, [call("sql__list_tables"), LLMResponse("4 tables")])
        transcript: list[Message] = []
        history = [Message("user", "earlier"), Message("assistant", "earlier answer")]
        async for _ in agent.stream("tables?", history, transcript=transcript):
            pass
    assert [m.role for m in transcript] == ["user", "assistant", "tool", "assistant"]
    sent = agent.llm.calls[0]  # type: ignore[attr-defined]
    assert [m.content for m in sent[1:3]] == ["earlier", "earlier answer"]


def _request() -> ApprovalRequest:
    tool = RegisteredTool("ws", "delete_file", "", {}, read_only=False, destructive=True, open_world=False)
    return ApprovalRequest(tool, {"path": "x"}, "reason")


async def test_broker_resolves_and_times_out() -> None:
    import asyncio

    broker = ApprovalBroker(timeout_s=5)
    req = _request()
    task = asyncio.create_task(broker(req))
    await asyncio.sleep(0)
    assert [p["id"] for p in broker.pending()] == [req.id]
    assert broker.resolve(req.id, True)
    assert (await task).approved
    assert not broker.resolve(req.id, True)  # already resolved

    quick = ApprovalBroker(timeout_s=0.05)
    decision = await quick(_request())
    assert not decision.approved and decision.by == "timeout"


async def test_callback_approver() -> None:
    async def yes(_req) -> bool:
        return True

    assert (await CallbackApprover(yes, by="cli")(_request())).by == "cli"


def test_policy_reasons() -> None:
    def tool(**kw):
        base = {"read_only": False, "destructive": False, "open_world": False}
        return RegisteredTool("s", "t", "", {}, **(base | kw))

    policy = ApprovalPolicy()
    assert policy.check(tool(read_only=True)) is None
    assert "modify" in policy.check(tool(destructive=True))
    assert "external" in policy.check(tool(open_world=True))
    assert policy.check(tool()) is not None
