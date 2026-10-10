"""Human-in-the-loop approval for sensitive tool calls, plus an append-only audit log.

Policy: a tool runs without asking only if its server marked it read-only (or the operator
listed it in ``trusted_tools``). Everything else - writes, deletes, anything that sends data
outside - pauses the agent until an approver says yes.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from mcp_agent_database.agent.host import RegisteredTool

ApprovalMode = Literal["ask", "auto_approve", "auto_deny"]


@dataclass
class ApprovalRequest:
    tool: RegisteredTool
    arguments: dict[str, Any]
    reason: str
    session_id: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool.qualified_name,
            "title": self.tool.title,
            "arguments": self.arguments,
            "reason": self.reason,
            "destructive": self.tool.destructive,
            "open_world": self.tool.open_world,
        }


@dataclass
class ApprovalDecision:
    approved: bool
    by: str
    note: str = ""


class Approver(Protocol):
    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision: ...


class ApprovalPolicy:
    def __init__(self, trusted_tools: list[str] | None = None) -> None:
        self.trusted = set(trusted_tools or [])

    def check(self, tool: RegisteredTool) -> str | None:
        """Return a human-readable reason if approval is needed, else ``None``."""
        if tool.qualified_name in self.trusted:
            return None
        if tool.read_only:
            return None
        if tool.destructive:
            return "This tool can modify or delete data."
        if tool.open_world:
            return "This tool sends data to an external system."
        return "This tool is not marked read-only."


class StaticApprover:
    """Always approves or always denies (evals, CI, unattended runs)."""

    def __init__(self, approve: bool) -> None:
        self.approve = approve

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(self.approve, by="auto_approve" if self.approve else "auto_deny")


class CallbackApprover:
    """Wraps any async callable, e.g. an interactive CLI prompt."""

    def __init__(self, fn: Callable[[ApprovalRequest], Awaitable[bool]], by: str = "user") -> None:
        self.fn, self.by = fn, by

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(await self.fn(request), by=self.by)


class ApprovalBroker:
    """Parks approval requests until someone resolves them (used by the HTTP API and web UI)."""

    def __init__(self, timeout_s: float = 300.0) -> None:
        self.timeout_s = timeout_s
        self._pending: dict[str, tuple[ApprovalRequest, asyncio.Future[ApprovalDecision]]] = {}

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        future: asyncio.Future[ApprovalDecision] = asyncio.get_running_loop().create_future()
        self._pending[request.id] = (request, future)
        try:
            return await asyncio.wait_for(future, timeout=self.timeout_s)
        except TimeoutError:
            return ApprovalDecision(False, by="timeout", note=f"No answer within {self.timeout_s:.0f}s")
        finally:
            self._pending.pop(request.id, None)

    def pending(self) -> list[dict[str, Any]]:
        return [req.summary() | {"session_id": req.session_id} for req, _ in self._pending.values()]

    def resolve(self, request_id: str, approved: bool, *, by: str = "user", note: str = "") -> bool:
        entry = self._pending.get(request_id)
        if entry is None or entry[1].done():
            return False
        entry[1].set_result(ApprovalDecision(approved, by=by, note=note))
        return True


def make_approver(mode: ApprovalMode, interactive: Approver | None = None) -> Approver:
    if mode == "auto_approve":
        return StaticApprover(True)
    if mode == "auto_deny" or interactive is None:
        return StaticApprover(False)
    return interactive


class AuditLog:
    """JSON-lines audit trail: every tool call, its approval decision and outcome."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **data: Any) -> None:
        if not self.path:
            return
        record = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), "event": event, **data}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=_default, ensure_ascii=False) + "\n")


def _default(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return str(value)
