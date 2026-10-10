"""Spawn the real server processes exactly as an MCP client (Claude Desktop, Claude Code) would."""

from __future__ import annotations

import json

import pytest

from mcp_agent_database.agent.host import MCPHost, load_server_specs

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


async def test_servers_over_stdio(settings, monkeypatch, tmp_path) -> None:
    for key, value in {
        "FORGE_DB_PATH": settings.db_path,
        "FORGE_DOCS_DIR": settings.docs_dir,
        "FORGE_SANDBOX_DIR": settings.sandbox_dir,
    }.items():
        monkeypatch.setenv(key, str(value))
    config = tmp_path / "servers.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    name: {
                        "command": "${PYTHON}",
                        "args": ["-m", f"mcp_agent_database.servers.{name}_server"],
                    }
                    for name in ("sql", "docs", "workspace")
                }
            }
        )
    )
    async with MCPHost.from_specs(load_server_specs(config)) as host:
        assert not host.failed and len(host.tools) == 16
        out = await host.call("sql__run_query", {"sql": "SELECT COUNT(*) AS n FROM orders"})
        assert json.loads(out.text)["rows"] == [[2500]]
        docs = await host.call("docs__search_docs", {"query": "free shipping threshold"})
        assert "shipping-policy" in docs.text
        files = await host.call("workspace__list_files", {})
        assert "todo.txt" in files.text
