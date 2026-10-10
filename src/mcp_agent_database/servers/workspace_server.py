"""Workspace MCP server: a sandboxed folder plus a few GitHub API tools.

The sandbox guarantees that no path (``..``, absolute paths, symlinks) can escape ``root``.
Write / delete / create-issue tools are annotated as non-read-only so the agent host asks
a human before they run.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(
    read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False
)
DELETE = ToolAnnotations(
    read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False
)
GITHUB_READ = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=True)
GITHUB_WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)


class SandboxError(ToolError):
    pass


class Sandbox:
    def __init__(self, root: Path, *, max_file_bytes: int = 200_000) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()
        self.max_file_bytes = max_file_bytes

    def resolve(self, rel: str) -> Path:
        """Map a user-supplied path to an absolute path inside the sandbox, or raise."""
        if "\x00" in rel:
            raise SandboxError("Path contains a NUL byte.")
        candidate = (self.root / rel.strip().lstrip("/")).resolve()  # resolve() follows symlinks
        if candidate != self.root and self.root not in candidate.parents:
            raise SandboxError(f"Path '{rel}' is outside the sandbox.")
        return candidate

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."


class GitHubClient:
    def __init__(self, token: str | None, base_url: str = "https://api.github.com") -> None:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.has_token = bool(token)
        self.http = httpx.Client(base_url=base_url, headers=headers, timeout=20.0)

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        try:
            resp = self.http.request(method, url, **kwargs)
        except httpx.HTTPError as err:
            raise ToolError(f"GitHub request failed: {err}") from err
        if resp.status_code >= 400:
            message = resp.json().get("message", resp.text) if resp.content else resp.reason_phrase
            raise ToolError(f"GitHub API error {resp.status_code}: {message}")
        return resp.json()


def _check_repo(repo: str) -> str:
    parts = repo.strip().split("/")
    if len(parts) != 2 or not all(parts):
        raise ToolError("repo must look like 'owner/name'.")
    return repo.strip()


def build_server(
    sandbox_dir: Path,
    *,
    max_file_bytes: int = 200_000,
    github_token: str | None = None,
    github_api_url: str = "https://api.github.com",
) -> MCPServer:
    box = Sandbox(sandbox_dir, max_file_bytes=max_file_bytes)
    gh = GitHubClient(github_token, github_api_url)
    server = MCPServer(
        "workspace",
        version="0.1.0",
        log_level="WARNING",
        instructions=(
            "A sandboxed working folder plus GitHub access. Paths are relative to the sandbox root. "
            "write_file, delete_file and github_create_issue require human approval."
        ),
    )

    # ------------------------------------------------------------------ files
    @server.tool(title="List files", annotations=READ_ONLY)
    def list_files(path: str = ".", recursive: bool = False) -> list[dict[str, Any]]:
        """List files and folders inside the sandbox."""
        base = box.resolve(path)
        if not base.is_dir():
            raise ToolError(f"'{path}' is not a directory.")
        entries = base.rglob("*") if recursive else base.iterdir()
        out = []
        for entry in sorted(entries):
            if entry.is_symlink():
                continue  # never expose symlinks, they could point outside
            out.append(
                {
                    "path": box.rel(entry),
                    "type": "dir" if entry.is_dir() else "file",
                    "bytes": entry.stat().st_size if entry.is_file() else None,
                }
            )
            if len(out) >= 500:
                break
        return out

    @server.tool(title="Read file", annotations=READ_ONLY)
    def read_file(path: str) -> dict[str, Any]:
        """Read a UTF-8 text file from the sandbox."""
        target = box.resolve(path)
        if not target.is_file():
            raise ToolError(f"File '{path}' not found.")
        data = target.read_bytes()
        if b"\x00" in data[:4096]:
            raise ToolError(f"'{path}' looks like a binary file.")
        truncated = len(data) > box.max_file_bytes
        text = data[: box.max_file_bytes].decode("utf-8", errors="replace")
        return {"path": box.rel(target), "content": text, "truncated": truncated}

    @server.tool(title="Search files", annotations=READ_ONLY)
    def search_files(text: str, glob: str = "*") -> list[dict[str, Any]]:
        """Find lines containing `text` (case-insensitive) in files matching `glob`."""
        needle = text.lower()
        hits = []
        for entry in sorted(box.root.rglob("*")):
            if not entry.is_file() or entry.is_symlink() or not fnmatch.fnmatch(entry.name, glob):
                continue
            try:
                lines = entry.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for no, line in enumerate(lines, start=1):
                if needle in line.lower():
                    hits.append({"path": box.rel(entry), "line": no, "text": line.strip()[:300]})
                    if len(hits) >= 100:
                        return hits
        return hits

    @server.tool(title="Write file", annotations=WRITE)
    def write_file(path: str, content: str, overwrite: bool = False) -> dict[str, Any]:
        """Create (or with overwrite=true, replace) a text file in the sandbox. Requires approval."""
        target = box.resolve(path)
        if target == box.root or target.is_dir():
            raise ToolError("Path points to a directory.")
        if target.exists() and not overwrite:
            raise ToolError(f"'{path}' already exists. Pass overwrite=true to replace it.")
        encoded = content.encode("utf-8")
        if len(encoded) > box.max_file_bytes:
            raise ToolError(f"Content is {len(encoded)} bytes; limit is {box.max_file_bytes}.")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.tmp")
        tmp.write_bytes(encoded)
        os.replace(tmp, target)  # atomic: never leaves a half-written file
        return {"path": box.rel(target), "bytes": len(encoded)}

    @server.tool(title="Delete file", annotations=DELETE)
    def delete_file(path: str) -> dict[str, Any]:
        """Delete one file from the sandbox (folders cannot be deleted). Requires approval."""
        target = box.resolve(path)
        if not target.is_file():
            raise ToolError(f"File '{path}' not found (only files can be deleted).")
        target.unlink()
        return {"deleted": box.rel(target)}

    # ------------------------------------------------------------------ GitHub
    @server.tool(title="Get GitHub repository", annotations=GITHUB_READ)
    def github_get_repo(repo: str) -> dict[str, Any]:
        """Get metadata for a public or accessible repository, e.g. 'modelcontextprotocol/python-sdk'."""
        data = gh.request("GET", f"/repos/{_check_repo(repo)}")
        keys = [
            "full_name",
            "description",
            "stargazers_count",
            "forks_count",
            "open_issues_count",
            "language",
        ]
        return {k: data.get(k) for k in [*keys, "default_branch", "html_url"]}

    @server.tool(title="List GitHub issues", annotations=GITHUB_READ)
    def github_list_issues(repo: str, state: str = "open", limit: int = 10) -> list[dict[str, Any]]:
        """List issues (not pull requests) for a repository."""
        if state not in {"open", "closed", "all"}:
            raise ToolError("state must be open, closed or all.")
        data = gh.request(
            "GET", f"/repos/{_check_repo(repo)}/issues", params={"state": state, "per_page": min(limit, 50)}
        )
        return [
            {"number": i["number"], "title": i["title"], "state": i["state"], "url": i["html_url"]}
            for i in data
            if "pull_request" not in i
        ][:limit]

    @server.tool(title="Create GitHub issue", annotations=GITHUB_WRITE)
    def github_create_issue(repo: str, title: str, body: str = "") -> dict[str, Any]:
        """Open a new issue. Sends data outside the company, so it requires approval."""
        if not gh.has_token:
            raise ToolError("No GitHub token configured (set FORGE_GITHUB_TOKEN).")
        data = gh.request("POST", f"/repos/{_check_repo(repo)}/issues", json={"title": title, "body": body})
        return {"number": data["number"], "url": data["html_url"]}

    return server


def main() -> None:  # pragma: no cover
    from mcp_agent_database.servers._runner import run_server

    run_server("workspace")


if __name__ == "__main__":  # pragma: no cover
    main()
