"""Command line interface: ``uv run mcp-agent-database --help``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

import anyio
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

from mcp_agent_database.agent.approval import ApprovalRequest, CallbackApprover
from mcp_agent_database.agent.llm import Message
from mcp_agent_database.agent.loop import AgentEvent
from mcp_agent_database.agent.runtime import build_agent, build_host
from mcp_agent_database.config import Settings, get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="MCP servers + local-first agent host.")
console = Console()

ProviderOpt = Annotated[str | None, typer.Option("--provider", "-p", help="ollama | openai | anthropic")]
ModelOpt = Annotated[str | None, typer.Option("--model", "-m", help="Model name, e.g. qwen3:8b")]


def _settings(provider: str | None = None, model: str | None = None, **overrides: Any) -> Settings:
    base = get_settings()
    update = {
        k: v for k, v in {"llm_provider": provider, "model": model, **overrides}.items() if v is not None
    }
    return base.model_copy(update=update)


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )


@app.command()
def seed(
    force: Annotated[bool, typer.Option(help="Recreate the database even if it exists.")] = False,
) -> None:
    """Create the sample DuckDB warehouse and sandbox folder."""
    from mcp_agent_database.sample_data import build_database, build_sandbox

    s = get_settings()
    if s.db_path.exists() and not force:
        console.print(f"[yellow]{s.db_path} already exists[/] (use --force to rebuild)")
    else:
        build_database(s.db_path)
        console.print(f"[green]✓[/] database  {s.db_path}")
    build_sandbox(s.sandbox_dir)
    console.print(f"[green]✓[/] sandbox   {s.sandbox_dir}")
    console.print(f"[green]✓[/] docs      {s.docs_dir} ({len(list(s.docs_dir.glob('*.md')))} files)")


@app.command()
def server(
    name: Annotated[str, typer.Argument(help="sql | docs | workspace")],
    transport: Annotated[str, typer.Option(help="stdio | streamable-http")] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8100,
) -> None:
    """Run one MCP server (this is what MCP clients such as Claude Desktop launch)."""
    from mcp_agent_database.servers._runner import run_server

    run_server(name, ["--transport", transport, "--host", host, "--port", str(port)])


@app.command()
def tools(in_process: Annotated[bool, typer.Option("--in-process")] = False) -> None:
    """List every tool the host discovered, and which ones need approval."""

    async def _run() -> None:
        s = get_settings()
        from mcp_agent_database.agent.approval import ApprovalPolicy

        policy = ApprovalPolicy(s.trusted_tools)
        async with build_host(s, in_process=in_process) as host:
            table = Table(title=f"{len(host.tools)} tools from {len(host.clients)} MCP servers")
            for col in ("tool", "access", "description"):
                table.add_column(col)
            for t in host.tools.values():
                access = "[green]read-only[/]" if policy.check(t) is None else "[yellow]needs approval[/]"
                table.add_row(
                    t.qualified_name, access, t.description.splitlines()[0] if t.description else ""
                )
            console.print(table)
            for name, err in host.failed.items():
                console.print(f"[red]✗ {name} failed:[/] {err}")

    anyio.run(_run)


async def _ask_cli(request: ApprovalRequest) -> bool:
    body = json.dumps(request.arguments, indent=2, ensure_ascii=False)
    console.print(
        Panel(
            Syntax(body, "json", theme="ansi_dark", word_wrap=True),
            title=f"[bold yellow]Approval needed[/]: {request.tool.qualified_name}",
            subtitle=request.reason,
            border_style="yellow",
        )
    )
    return await anyio.to_thread.run_sync(lambda: Confirm.ask("Allow this action?", default=False))


def _read_question() -> str:
    return str(Prompt.ask("\n[bold blue]you[/]") or "")


def _render(event: AgentEvent, show_results: bool) -> None:
    d = event.data
    if event.type == "assistant_text" and d["content"]:
        console.print(f"[dim italic]{d['content']}[/]")
    elif event.type == "tool_call":
        args = json.dumps(d["arguments"], ensure_ascii=False)
        console.print(f"[cyan]→ {d['name']}[/] [dim]{args[:160]}[/]")
    elif event.type == "approval_decision":
        mark = "[green]approved[/]" if d["approved"] else "[red]denied[/]"
        console.print(f"  {mark} by {d['by']}")
    elif event.type == "tool_result":
        color = {"ok": "green", "denied": "red"}.get(d["status"], "red")
        console.print(f"  [{color}]{d['status']}[/] [dim]{d['duration_ms']:.0f} ms[/]")
        if show_results or d["status"] == "error":
            console.print(f"  [dim]{d['content'][:600]}[/]")
    elif event.type == "final":
        console.print(Panel(Markdown(d["content"] or "_(empty answer)_"), border_style="green"))
    elif event.type == "error":
        console.print(f"[red]Error:[/] {d['message']}")


@app.command()
def chat(
    provider: ProviderOpt = None,
    model: ModelOpt = None,
    prompt: Annotated[str | None, typer.Option("--prompt", "-q", help="Ask one question and exit.")] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Auto-approve sensitive tools (careful!).")] = False,
    show_results: Annotated[bool, typer.Option("--show-results", help="Print tool outputs.")] = False,
    in_process: Annotated[bool, typer.Option("--in-process", help="Run servers in this process.")] = False,
) -> None:
    """Chat with the agent in your terminal. Sensitive actions ask for confirmation."""
    s = _settings(provider, model, approval_mode="auto_approve" if yes else None)

    async def _run() -> None:
        async with build_host(s, in_process=in_process) as host:
            agent = build_agent(s, host, interactive_approver=CallbackApprover(_ask_cli, by="cli"))
            console.print(
                f"[bold]mcp-agent-database[/] · {s.llm_provider}:{s.model} · {len(host.tools)} tools from "
                f"{', '.join(host.clients)}. Type [bold]/exit[/] to quit, [bold]/reset[/] to clear history."
            )
            history: list[Message] = []
            while True:
                question: str = prompt if prompt else await anyio.to_thread.run_sync(_read_question)
                if question.strip() in {"/exit", "/quit"}:
                    return
                if question.strip() == "/reset":
                    history.clear()
                    console.print("[dim]history cleared[/]")
                    continue
                transcript: list[Message] = []
                async for event in agent.stream(question, history, session_id="cli", transcript=transcript):
                    _render(event, show_results)
                history.extend(transcript)
                if prompt:
                    return

    anyio.run(_run)


@app.command()
def serve(
    host: str = "127.0.0.1", port: int = 8000, provider: ProviderOpt = None, model: ModelOpt = None
) -> None:
    """Start the HTTP API and web UI."""
    import uvicorn

    from mcp_agent_database.api.app import create_app

    console.print(f"[bold]mcp-agent-database[/] UI → http://{host}:{port}")
    uvicorn.run(create_app(_settings(provider, model)), host=host, port=port, log_level="info")


@app.command("eval")
def run_eval(
    provider: ProviderOpt = None,
    model: ModelOpt = None,
    tasks_file: Annotated[Path, typer.Option("--tasks")] = Path("evals/tasks.json"),
    only: Annotated[str | None, typer.Option(help="Comma-separated task ids or categories.")] = None,
    out: Annotated[Path, typer.Option(help="Results folder.")] = Path("evals/results"),
) -> None:
    """Run the benchmark tasks and write JSON + Markdown results."""
    from mcp_agent_database.evals import TaskResult, load_tasks, run_task, save, summarize, to_markdown

    s = _settings(provider, model)
    tasks = load_tasks(tasks_file)
    if only:
        wanted = {w.strip() for w in only.split(",")}
        tasks = [t for t in tasks if t.id in wanted or t.category in wanted]

    async def _run() -> list[TaskResult]:
        results = []
        for task in tasks:
            console.print(f"[cyan]▶ {task.id}[/] ", end="")
            r = await run_task(task, s)
            results.append(r)
            console.print(
                "[green]pass[/]" if r.success else f"[red]fail[/] {r.checks}", f"[dim]{r.duration_s}s[/]"
            )
        return results

    results = anyio.run(_run)
    summary = summarize(results)
    path = save(f"{s.llm_provider}:{s.model}", summary, results, out)
    console.print(Markdown(to_markdown(f"{s.llm_provider}:{s.model}", summary, results)))
    console.print(f"[dim]saved {path} and {path.with_suffix('.md')}[/]")


if __name__ == "__main__":  # pragma: no cover
    app()
