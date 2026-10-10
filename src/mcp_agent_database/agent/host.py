"""MCP host: connects to many MCP servers and exposes their tools as one namespaced registry.

Server config uses the same ``{"mcpServers": {...}}`` shape as Claude Desktop / Claude Code,
so the same file works in those clients too. Each entry is either a stdio command or a URL.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, model_validator

from mcp_agent_database.agent.llm import ToolSpec

log = logging.getLogger(__name__)

SEPARATOR = "__"
_ENV_REF = re.compile(r"\$\{([A-Z0-9_]+)\}")
# Env vars forwarded to stdio servers in addition to the SDK's safe defaults (PATH, HOME, ...).
_FORWARDED_PREFIXES = ("FORGE_", "OLLAMA_")

ServerTarget = MCPServer | StdioServerParameters | str


class ServerSpec(BaseModel):
    name: str
    command: str | None = None
    args: list[str] = []
    env: dict[str, str] = {}
    cwd: str | None = None
    url: str | None = None
    enabled: bool = True

    @model_validator(mode="after")
    def _one_transport(self) -> ServerSpec:
        if bool(self.command) == bool(self.url):
            raise ValueError(f"Server '{self.name}' needs exactly one of 'command' or 'url'.")
        if SEPARATOR in self.name or not re.fullmatch(r"[A-Za-z0-9_-]+", self.name):
            raise ValueError(f"Invalid server name '{self.name}' (letters, digits, '-', single '_').")
        return self

    def to_target(self) -> ServerTarget:
        if self.url:
            return self.url
        assert self.command is not None
        forwarded = {k: v for k, v in os.environ.items() if k.startswith(_FORWARDED_PREFIXES)}
        return StdioServerParameters(
            command=self.command, args=self.args, env={**forwarded, **self.env}, cwd=self.cwd
        )


def _expand(value: str) -> str:
    value = value.replace("${PYTHON}", sys.executable)
    return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), value)


def load_server_specs(path: Path) -> list[ServerSpec]:
    """Read a Claude-Desktop-style config. ``${PYTHON}`` and ``${ENV_VAR}`` are expanded."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    specs = []
    for name, entry in raw.get("mcpServers", {}).items():
        entry = dict(entry)
        if "command" in entry:
            entry["command"] = _expand(entry["command"])
        entry["args"] = [_expand(a) for a in entry.get("args", [])]
        entry["env"] = {k: _expand(v) for k, v in entry.get("env", {}).items()}
        spec = ServerSpec(name=name, **entry)
        if spec.enabled:
            specs.append(spec)
    return specs


@dataclass(frozen=True)
class RegisteredTool:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool
    destructive: bool
    open_world: bool
    title: str | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.server}{SEPARATOR}{self.name}"

    def to_spec(self) -> ToolSpec:
        return ToolSpec(self.qualified_name, f"[{self.server}] {self.description}".strip(), self.input_schema)


@dataclass
class ToolOutcome:
    text: str
    is_error: bool
    duration_ms: float
    structured: Any = None


def result_to_text(result: CallToolResult, max_chars: int) -> str:
    """Prefer structured JSON output; fall back to text blocks; truncate for the context window."""
    if result.structured_content is not None and not result.is_error:
        payload = result.structured_content
        if set(payload) == {"result"}:  # SDK wraps non-object return values in {"result": ...}
            payload = payload["result"]
        text = json.dumps(payload, ensure_ascii=False, default=str)
    else:
        parts = []
        for block in result.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
            else:
                parts.append(f"[{block.type} content omitted]")
        text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"
    return text


class MCPHost:
    """Async context manager owning one MCP ``Client`` per server."""

    def __init__(self, servers: dict[str, ServerTarget], *, max_output_chars: int = 12_000) -> None:
        self.targets = servers
        self.max_output_chars = max_output_chars
        self.clients: dict[str, Client] = {}
        self.tools: dict[str, RegisteredTool] = {}
        self.failed: dict[str, str] = {}
        self._stack = AsyncExitStack()

    @classmethod
    def from_specs(cls, specs: list[ServerSpec], **kwargs: Any) -> MCPHost:
        return cls({s.name: s.to_target() for s in specs}, **kwargs)

    async def __aenter__(self) -> MCPHost:
        await self._stack.__aenter__()
        for name, target in self.targets.items():
            try:
                client = await self._stack.enter_async_context(Client(target, read_timeout_seconds=120))
            except Exception as err:  # keep going with the servers that did start
                log.error("MCP server '%s' failed to start: %s", name, err)
                self.failed[name] = str(err)
                continue
            self.clients[name] = client
            await self._register_tools(name, client)
        if not self.clients:
            raise RuntimeError(f"No MCP server could be started: {self.failed}")
        log.info("Connected to %d MCP servers, %d tools", len(self.clients), len(self.tools))
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.__aexit__(*exc)  # type: ignore[arg-type]

    async def _register_tools(self, server: str, client: Client) -> None:
        cursor: str | None = None
        while True:
            page = await client.list_tools(cursor=cursor)
            for tool in page.tools:
                ann = tool.annotations
                # MCP annotations are *hints from the server*. Missing hints are treated as unsafe:
                # unknown read_only -> not read-only; unknown destructive -> destructive (spec default).
                read_only = bool(ann and ann.read_only_hint)
                destructive = not read_only and (ann is None or ann.destructive_hint is not False)
                reg = RegisteredTool(
                    server=server,
                    name=tool.name,
                    description=(tool.description or "").strip(),
                    input_schema=tool.input_schema or {"type": "object", "properties": {}},
                    read_only=read_only,
                    destructive=destructive,
                    open_world=bool(ann is None or ann.open_world_hint is not False),
                    title=tool.title or (ann.title if ann else None),
                )
                self.tools[reg.qualified_name] = reg
            cursor = page.next_cursor
            if not cursor:
                break

    def tool_specs(self) -> list[ToolSpec]:
        return [t.to_spec() for t in self.tools.values()]

    def get(self, qualified_name: str) -> RegisteredTool | None:
        return self.tools.get(qualified_name)

    async def call(self, qualified_name: str, arguments: dict[str, Any]) -> ToolOutcome:
        tool = self.tools.get(qualified_name)
        if tool is None:
            raise KeyError(qualified_name)
        start = time.perf_counter()
        try:
            result = await self.clients[tool.server].call_tool(tool.name, arguments)
        except Exception as err:  # transport / protocol failures become tool errors for the model
            return ToolOutcome(f"Tool call failed: {err}", True, (time.perf_counter() - start) * 1000)
        return ToolOutcome(
            text=result_to_text(result, self.max_output_chars),
            is_error=bool(result.is_error),
            duration_ms=(time.perf_counter() - start) * 1000,
            structured=result.structured_content,
        )
