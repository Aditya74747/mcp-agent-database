"""Build any of the bundled servers from settings and run it over stdio or Streamable HTTP."""

from __future__ import annotations

import argparse
from typing import Literal

from mcp.server.mcpserver import MCPServer

from mcp_agent_database.config import Settings, get_settings

ServerName = Literal["sql", "docs", "workspace"]
SERVER_NAMES: tuple[ServerName, ...] = ("sql", "docs", "workspace")


def build(name: str, settings: Settings | None = None) -> MCPServer:
    s = settings or get_settings()
    if name == "sql":
        from mcp_agent_database.servers.sql_server import build_server as build_sql

        return build_sql(s.db_path, timeout_s=s.sql_timeout_s, max_rows=s.sql_max_rows)
    if name == "docs":
        from mcp_agent_database.servers.docs_server import build_server as build_docs

        return build_docs(s.docs_dir, embed_model=s.embed_model, ollama_base_url=s.ollama_base_url)
    if name == "workspace":
        from mcp_agent_database.servers.workspace_server import build_server as build_ws

        token = s.github_token.get_secret_value() if s.github_token else None
        return build_ws(
            s.sandbox_dir,
            max_file_bytes=s.max_file_bytes,
            github_token=token,
            github_api_url=s.github_api_url,
        )
    raise ValueError(f"Unknown server '{name}'. Choose from {SERVER_NAMES}.")


def run_server(name: str, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=f"Run the mcp-agent-database '{name}' MCP server.")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args(argv)
    server = build(name)
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)
