from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mcp_agent_database.config import Settings
from mcp_agent_database.sample_data import build_database, build_sandbox

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def seeded_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One database per test session (building it takes ~1s); copy it when a test writes."""
    return build_database(tmp_path_factory.mktemp("db") / "shop.duckdb")


@pytest.fixture
def settings(tmp_path: Path, seeded_db: Path) -> Settings:
    """Fully isolated settings: private DB copy, docs copy, sandbox and audit log."""
    db = tmp_path / "shop.duckdb"
    shutil.copy(seeded_db, db)
    docs = tmp_path / "docs"
    shutil.copytree(REPO / "data" / "docs", docs)
    sandbox = build_sandbox(tmp_path / "sandbox")
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        db_path=db,
        docs_dir=docs,
        sandbox_dir=sandbox,
        audit_log=tmp_path / "audit.jsonl",
        servers_config=tmp_path / "missing.json",
        approval_mode="ask",
        embed_model=None,
        github_token=None,
        github_api_url="https://api.github.test",
    )


async def call(server, tool: str, args: dict | None = None):
    """Call one tool on an in-process MCP server through a real MCP client session."""
    from mcp import Client

    async with Client(server) as client:
        return await client.call_tool(tool, args or {})


def payload(result):
    """Structured output of a successful tool call (unwrapping the SDK's {"result": ...})."""
    assert not result.is_error, result.content[0].text
    data = result.structured_content
    return data["result"] if isinstance(data, dict) and set(data) == {"result"} else data
