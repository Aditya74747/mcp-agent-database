"""HTTP API + web UI.

POST /v1/chat                 -> Server-Sent Events stream of agent events for one user turn
GET  /v1/approvals            -> pending approval requests
POST /v1/approvals/{id}       -> approve / deny one request (unblocks the running turn)
GET  /v1/tools, /health       -> introspection
GET  /                        -> single-page chat UI with Approve / Deny buttons
"""

from __future__ import annotations

import asyncio
import json
import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from importlib import resources
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sse_starlette import EventSourceResponse

from mcp_agent_database import __version__
from mcp_agent_database.agent.approval import ApprovalBroker
from mcp_agent_database.agent.host import MCPHost
from mcp_agent_database.agent.llm import LLM, Message
from mcp_agent_database.agent.loop import Agent
from mcp_agent_database.agent.runtime import build_agent, build_host
from mcp_agent_database.config import Settings, get_settings

MAX_HISTORY_MESSAGES = 40


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: str | None = None


class ApprovalBody(BaseModel):
    approved: bool
    note: str = ""


@dataclass
class Session:
    history: list[Message] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class AppState:
    settings: Settings
    host: MCPHost
    agent: Agent
    broker: ApprovalBroker
    sessions: dict[str, Session] = field(default_factory=dict)


_bearer = HTTPBearer(auto_error=False)


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.agentdb
    return state


StateDep = Annotated[AppState, Depends(get_state)]


def require_token(
    st: StateDep, creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]
) -> None:
    token = st.settings.api_token
    if token is None:
        return
    if creds is None or not secrets.compare_digest(creds.credentials, token.get_secret_value()):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def create_app(
    settings: Settings | None = None, *, llm: LLM | None = None, in_process: bool = False
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        broker = ApprovalBroker(timeout_s=settings.approval_timeout_s)
        async with build_host(settings, in_process=in_process) as host:
            agent = build_agent(settings, host, interactive_approver=broker, llm=llm)
            app.state.agentdb = AppState(settings, host, agent, broker)
            yield

    app = FastAPI(title="mcp-agent-database", version=__version__, lifespan=lifespan)
    protected = [Depends(require_token)]

    @app.get("/health")
    async def health(st: StateDep) -> dict[str, Any]:
        return {
            "status": "ok" if not st.host.failed else "degraded",
            "version": __version__,
            "provider": st.settings.llm_provider,
            "model": st.agent.llm.model,
            "servers": sorted(st.host.clients),
            "failed_servers": st.host.failed,
            "tools": len(st.host.tools),
            "approval_mode": st.settings.approval_mode,
        }

    @app.get("/v1/tools", dependencies=protected)
    async def list_tools(st: StateDep) -> list[dict[str, Any]]:
        policy = st.agent.policy
        return [
            {
                "name": t.qualified_name,
                "server": t.server,
                "title": t.title,
                "description": t.description,
                "read_only": t.read_only,
                "requires_approval": policy.check(t) is not None,
                "input_schema": t.input_schema,
            }
            for t in st.host.tools.values()
        ]

    @app.post("/v1/chat", dependencies=protected)
    async def chat(body: ChatRequest, st: StateDep) -> EventSourceResponse:
        session_id = body.session_id or uuid.uuid4().hex[:12]
        session = st.sessions.setdefault(session_id, Session())
        if session.lock.locked():
            raise HTTPException(status_code=409, detail="This session is already running a turn.")

        async def events() -> AsyncIterator[dict[str, str]]:
            async with session.lock:
                yield {"event": "session", "data": json.dumps({"session_id": session_id})}
                transcript: list[Message] = []
                async for event in st.agent.stream(
                    body.message, session.history, session_id=session_id, transcript=transcript
                ):
                    yield {"event": event.type, "data": json.dumps(event.to_dict(), default=str)}
                session.history = (session.history + transcript)[-MAX_HISTORY_MESSAGES:]
                yield {"event": "done", "data": "{}"}

        return EventSourceResponse(events(), ping=15)

    @app.get("/v1/approvals", dependencies=protected)
    async def pending(st: StateDep) -> list[dict[str, Any]]:
        return st.broker.pending()

    @app.post("/v1/approvals/{request_id}", dependencies=protected)
    async def decide(request_id: str, body: ApprovalBody, st: StateDep) -> dict[str, Any]:
        if not st.broker.resolve(request_id, body.approved, by="web", note=body.note):
            raise HTTPException(status_code=404, detail="No pending approval with that id")
        return {"id": request_id, "approved": body.approved}

    @app.delete("/v1/sessions/{session_id}", dependencies=protected)
    async def reset(session_id: str, st: StateDep) -> dict[str, bool]:
        return {"deleted": st.sessions.pop(session_id, None) is not None}

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return (
            resources.files("mcp_agent_database.api")
            .joinpath("static/index.html")
            .read_text(encoding="utf-8")
        )

    return app
