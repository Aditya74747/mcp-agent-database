from __future__ import annotations

import pytest

from mcp_agent_database.servers.docs_server import build_server
from tests.conftest import call, payload

pytestmark = pytest.mark.anyio


async def test_search_returns_cited_passages(settings) -> None:
    out = payload(await call(build_server(settings.docs_dir), "search_docs", {"query": "SEV1 response time"}))
    assert out["mode"] == "bm25"
    first = out["results"][0]
    assert first["doc_id"] == "incident-runbook"
    assert "15 minutes" in first["text"]


async def test_read_and_list_documents(settings) -> None:
    server = build_server(settings.docs_dir)
    docs = payload(await call(server, "list_documents"))
    assert {d["doc_id"] for d in docs} >= {"refund-policy", "data-dictionary", "vendor-faq"}
    doc = payload(await call(server, "read_document", {"doc_id": "shipping-policy"}))
    assert "EUR 150" in doc["text"]
    assert doc["truncated"] is False


async def test_unknown_document_lists_valid_ids(settings) -> None:
    result = await call(build_server(settings.docs_dir), "read_document", {"doc_id": "nope"})
    assert result.is_error
    assert "refund-policy" in result.content[0].text


async def test_reindex_picks_up_new_files(settings) -> None:
    server = build_server(settings.docs_dir)
    (settings.docs_dir / "new.md").write_text("# Parking\nVisitors park on level 3.", encoding="utf-8")
    assert payload(await call(server, "reindex"))["documents"] == 7
    out = payload(await call(server, "search_docs", {"query": "visitor parking"}))
    assert out["results"][0]["doc_id"] == "new"


async def test_empty_query_is_rejected(settings) -> None:
    assert (await call(build_server(settings.docs_dir), "search_docs", {"query": "  "})).is_error


async def test_document_resource(settings) -> None:
    from mcp import Client

    async with Client(build_server(settings.docs_dir)) as client:
        res = await client.read_resource("docs://refund-policy")
    assert "30 days" in res.contents[0].text
