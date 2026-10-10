from __future__ import annotations

import os

import httpx
import pytest
import respx

from mcp_agent_database.servers.workspace_server import build_server
from tests.conftest import call, payload

pytestmark = pytest.mark.anyio
API = "https://api.github.test"


@pytest.mark.parametrize("path", ["../secret.txt", "../../etc/passwd", "notes/../../x", "a/../../../b"])
async def test_paths_cannot_escape_sandbox(settings, path: str) -> None:
    result = await call(build_server(settings.sandbox_dir), "read_file", {"path": path})
    assert result.is_error
    assert "outside the sandbox" in result.content[0].text


async def test_absolute_paths_are_rebased_inside_sandbox(settings) -> None:
    result = await call(build_server(settings.sandbox_dir), "read_file", {"path": "/README.txt"})
    assert "sandbox" in payload(result)["content"]


async def test_symlink_escape_is_blocked(settings, tmp_path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    try:
        os.symlink(secret, settings.sandbox_dir / "link.txt")
    except OSError:  # Windows needs Developer Mode or admin rights to create symlinks
        pytest.skip("this OS does not allow creating symlinks for the current user")
    server = build_server(settings.sandbox_dir)
    assert (await call(server, "read_file", {"path": "link.txt"})).is_error
    listing = payload(await call(server, "list_files", {"recursive": True}))
    assert "link.txt" not in {e["path"] for e in listing}


async def test_list_read_and_search(settings) -> None:
    server = build_server(settings.sandbox_dir)
    files = {e["path"] for e in payload(await call(server, "list_files", {"recursive": True}))}
    assert {"README.txt", "todo.txt", "notes/meeting-2025-11-04.md"} <= files
    hits = payload(await call(server, "search_files", {"text": "q4 summary"}))
    assert hits[0]["path"] == "notes/meeting-2025-11-04.md"


async def test_write_refuses_to_overwrite_by_default(settings) -> None:
    server = build_server(settings.sandbox_dir)
    out = payload(await call(server, "write_file", {"path": "reports/a.md", "content": "hello"}))
    assert out == {"path": "reports/a.md", "bytes": 5}
    again = await call(server, "write_file", {"path": "reports/a.md", "content": "bye"})
    assert again.is_error and "overwrite" in again.content[0].text
    payload(await call(server, "write_file", {"path": "reports/a.md", "content": "bye", "overwrite": True}))
    assert (settings.sandbox_dir / "reports/a.md").read_text() == "bye"


async def test_write_size_limit(settings) -> None:
    server = build_server(settings.sandbox_dir, max_file_bytes=10)
    assert (await call(server, "write_file", {"path": "big.txt", "content": "x" * 11})).is_error


async def test_delete_only_files(settings) -> None:
    server = build_server(settings.sandbox_dir)
    assert (await call(server, "delete_file", {"path": "notes"})).is_error
    payload(await call(server, "delete_file", {"path": "todo.txt"}))
    assert not (settings.sandbox_dir / "todo.txt").exists()


@respx.mock
async def test_github_tools(settings) -> None:
    respx.get(f"{API}/repos/acme/shop").mock(
        return_value=httpx.Response(
            200, json={"full_name": "acme/shop", "stargazers_count": 7, "html_url": "u"}
        )
    )
    respx.get(f"{API}/repos/acme/shop/issues").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"number": 1, "title": "Bug", "state": "open", "html_url": "u1"},
                {"number": 2, "title": "PR", "state": "open", "html_url": "u2", "pull_request": {}},
            ],
        )
    )
    create = respx.post(f"{API}/repos/acme/shop/issues").mock(
        return_value=httpx.Response(201, json={"number": 3, "html_url": "u3"})
    )
    server = build_server(settings.sandbox_dir, github_token="t0k", github_api_url=API)
    assert payload(await call(server, "github_get_repo", {"repo": "acme/shop"}))["stargazers_count"] == 7
    issues = payload(await call(server, "github_list_issues", {"repo": "acme/shop"}))
    assert [i["number"] for i in issues] == [1]  # pull requests are filtered out
    assert (
        payload(await call(server, "github_create_issue", {"repo": "acme/shop", "title": "Hi"}))["number"]
        == 3
    )
    assert create.calls[0].request.headers["Authorization"] == "Bearer t0k"


@respx.mock
async def test_github_errors_are_reported(settings) -> None:
    respx.get(f"{API}/repos/acme/missing").mock(
        return_value=httpx.Response(404, json={"message": "Not Found"})
    )
    server = build_server(settings.sandbox_dir, github_api_url=API)
    result = await call(server, "github_get_repo", {"repo": "acme/missing"})
    assert result.is_error and "404" in result.content[0].text
    assert (await call(server, "github_get_repo", {"repo": "not-a-repo"})).is_error
    no_token = await call(server, "github_create_issue", {"repo": "acme/shop", "title": "x"})
    assert no_token.is_error and "token" in no_token.content[0].text
