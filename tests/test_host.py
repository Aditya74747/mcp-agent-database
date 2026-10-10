from __future__ import annotations

import json
import sys

import pytest
from mcp import StdioServerParameters
from pydantic import ValidationError

from mcp_agent_database.agent.host import MCPHost, ServerSpec, load_server_specs
from mcp_agent_database.agent.runtime import build_host

pytestmark = pytest.mark.anyio


async def test_registry_namespaces_and_safety_flags(settings) -> None:
    async with build_host(settings, in_process=True) as host:
        assert set(host.clients) == {"sql", "docs", "workspace"}
        run_query = host.get("sql__run_query")
        assert run_query and run_query.read_only and not run_query.destructive
        delete = host.get("workspace__delete_file")
        assert delete and not delete.read_only and delete.destructive
        issue = host.get("workspace__github_create_issue")
        assert issue and not issue.destructive and issue.open_world
        specs = {s.name for s in host.tool_specs()}
        assert "docs__search_docs" in specs and all("__" in name for name in specs)


async def test_missing_annotations_are_treated_as_unsafe() -> None:
    from mcp.server.mcpserver import MCPServer

    bare = MCPServer("bare")

    @bare.tool()
    def mystery() -> str:
        return "?"

    async with MCPHost({"bare": bare}) as host:
        tool = host.get("bare__mystery")
        assert tool and not tool.read_only and tool.destructive and tool.open_world


async def test_call_formats_structured_output(settings) -> None:
    async with build_host(settings, in_process=True) as host:
        outcome = await host.call("sql__run_query", {"sql": "SELECT 42 AS answer"})
        assert not outcome.is_error
        assert json.loads(outcome.text)["rows"] == [[42]]
        bad = await host.call("sql__run_query", {"sql": "DROP TABLE orders"})
        assert bad.is_error and "read-only" in bad.text


async def test_output_is_truncated(settings) -> None:
    host = build_host(settings, in_process=True)
    host.max_output_chars = 100
    async with host:
        outcome = await host.call("docs__read_document", {"doc_id": "refund-policy"})
        assert "[truncated" in outcome.text


async def test_failed_server_does_not_take_down_host(settings) -> None:
    from mcp_agent_database.servers.docs_server import build_server

    broken = StdioServerParameters(command=sys.executable, args=["-c", "import sys; sys.exit(1)"])
    async with MCPHost({"docs": build_server(settings.docs_dir), "broken": broken}) as host:
        assert "docs" in host.clients
        assert "broken" in host.failed


def test_load_specs_expands_variables(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MY_TOKEN", "abc")
    cfg = tmp_path / "servers.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "sql": {"command": "${PYTHON}", "args": ["-m", "x"], "env": {"TOKEN": "${MY_TOKEN}"}},
                    "remote": {"url": "http://localhost:8100/mcp"},
                    "off": {"command": "x", "enabled": False},
                }
            }
        )
    )
    specs = load_server_specs(cfg)
    assert [s.name for s in specs] == ["sql", "remote"]
    assert specs[0].command == sys.executable and specs[0].env == {"TOKEN": "abc"}
    assert specs[1].to_target() == "http://localhost:8100/mcp"


def test_stdio_target_forwards_forge_env(monkeypatch) -> None:
    monkeypatch.setenv("FORGE_DB_PATH", "/data/x.duckdb")
    monkeypatch.setenv("SECRET_UNRELATED", "nope")
    target = ServerSpec(name="sql", command="python").to_target()
    assert isinstance(target, StdioServerParameters)
    assert target.env["FORGE_DB_PATH"] == "/data/x.duckdb"
    assert "SECRET_UNRELATED" not in target.env


@pytest.mark.parametrize("entry", [{}, {"command": "x", "url": "http://y"}])
def test_spec_requires_exactly_one_transport(entry) -> None:
    with pytest.raises(ValidationError):
        ServerSpec(name="s", **entry)


def test_spec_rejects_separator_in_name() -> None:
    with pytest.raises(ValidationError):
        ServerSpec(name="a__b", command="x")
