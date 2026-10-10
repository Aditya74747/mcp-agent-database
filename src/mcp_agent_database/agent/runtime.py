"""Wire settings -> MCP host + LLM + approval policy + agent."""

from __future__ import annotations

import logging

from mcp_agent_database.agent.approval import ApprovalPolicy, Approver, AuditLog, make_approver
from mcp_agent_database.agent.host import MCPHost, ServerTarget, load_server_specs
from mcp_agent_database.agent.llm import LLM, create_llm
from mcp_agent_database.agent.loop import Agent
from mcp_agent_database.config import Settings
from mcp_agent_database.servers._runner import SERVER_NAMES, build

log = logging.getLogger(__name__)


def build_host(settings: Settings, *, in_process: bool = False) -> MCPHost:
    """Servers come from ``mcp_servers.json`` (stdio subprocesses / URLs).

    ``in_process=True`` (or a missing config file) runs the bundled servers inside this process,
    which is handy for tests and quick demos.
    """
    if not in_process and settings.servers_config.exists():
        specs = load_server_specs(settings.servers_config)
        return MCPHost.from_specs(specs, max_output_chars=settings.max_tool_output_chars)
    if not in_process:
        log.warning("%s not found; running bundled servers in-process.", settings.servers_config)
    targets: dict[str, ServerTarget] = {name: build(name, settings) for name in SERVER_NAMES}
    return MCPHost(targets, max_output_chars=settings.max_tool_output_chars)


def build_agent(
    settings: Settings,
    host: MCPHost,
    *,
    interactive_approver: Approver | None = None,
    llm: LLM | None = None,
) -> Agent:
    return Agent(
        llm or create_llm(settings),
        host,
        approver=make_approver(settings.approval_mode, interactive_approver),
        policy=ApprovalPolicy(settings.trusted_tools),
        audit=AuditLog(settings.audit_log),
        max_steps=settings.max_steps,
    )
