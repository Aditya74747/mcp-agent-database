"""SQL analytics MCP server backed by DuckDB.

Safety model (defence in depth):
1. ``run_query`` only accepts a *single* read-only statement (SELECT / WITH / UNION / DESCRIBE),
   checked by parsing the SQL with sqlglot rather than with regexes.
2. DuckDB's ``enable_external_access`` is switched off, so even a valid SELECT cannot read
   arbitrary files (``read_csv('/etc/passwd')``) or URLs.
3. Every query runs with a timeout (DuckDB ``interrupt``) and a row cap.
4. Writes go through a separate ``execute_statement`` tool that is annotated as destructive,
   so the agent host asks a human before running it.
"""

from __future__ import annotations

import datetime as dt
import decimal
import threading
from pathlib import Path
from typing import Any

import duckdb
import sqlglot
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from sqlglot import exp

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)

_READ_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Describe)
_WRITE_ROOTS = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter, exp.Merge)
_WRITE_NODES = (*_WRITE_ROOTS, exp.Command)


class SQLValidationError(ToolError):
    """Raised for SQL the server refuses to run. The message is shown to the model."""


def _parse_single(sql: str) -> exp.Expr:
    try:
        statements = [s for s in sqlglot.parse(sql, dialect="duckdb") if s is not None]
    except sqlglot.errors.ParseError as err:
        raise SQLValidationError(f"Could not parse SQL: {err}") from err
    if len(statements) != 1:
        raise SQLValidationError(f"Expected exactly one SQL statement, got {len(statements)}.")
    return statements[0]


def validate_read_only(sql: str) -> None:
    """Raise unless ``sql`` is a single statement that cannot modify anything."""
    tree = _parse_single(sql)
    if not isinstance(tree, _READ_ROOTS):
        raise SQLValidationError(
            f"run_query only accepts read-only SELECT queries (got {tree.key.upper()}). "
            "Use execute_statement for changes; it requires human approval."
        )
    if any(True for _ in tree.find_all(*_WRITE_NODES)):
        raise SQLValidationError("Query contains a write or command clause; only reads are allowed.")


def validate_write(sql: str) -> None:
    """Writes are allowed, but nothing that escapes the database (ATTACH, COPY, INSTALL, SET ...)."""
    tree = _parse_single(sql)
    if not isinstance(tree, _WRITE_ROOTS):
        raise SQLValidationError(
            f"execute_statement only accepts INSERT/UPDATE/DELETE/CREATE/DROP/ALTER (got {tree.key.upper()})."
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, dt.date | dt.datetime | dt.time):
        return value.isoformat()
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return value


class Warehouse:
    """Thread-safe wrapper around one DuckDB connection (MCP runs sync tools in worker threads)."""

    def __init__(self, db_path: Path, *, timeout_s: float = 10.0, max_rows: int = 200) -> None:
        if not db_path.exists():
            raise FileNotFoundError(f"{db_path} not found. Run `mcp-agent-database seed` first.")
        self.timeout_s = timeout_s
        self.max_rows = max_rows
        self._lock = threading.Lock()
        self._con = duckdb.connect(str(db_path))
        self._con.execute("SET enable_external_access = false")
        self._con.execute("SET lock_configuration = true")  # the model cannot re-enable it

    def close(self) -> None:
        self._con.close()

    def _run(self, sql: str, params: list[Any] | None = None, fetch: int | None = None) -> dict[str, Any]:
        with self._lock:
            timer = threading.Timer(self.timeout_s, self._con.interrupt)
            timer.start()
            try:
                cur = self._con.execute(sql, params or [])
                if cur.description is None:
                    return {"columns": [], "rows": [], "row_count": 0, "truncated": False}
                columns = [d[0] for d in cur.description]
                rows = cur.fetchmany(fetch + 1) if fetch is not None else cur.fetchall()
            except duckdb.InterruptException as err:
                raise ToolError(
                    f"Query exceeded the {self.timeout_s:.0f}s timeout and was cancelled."
                ) from err
            except duckdb.Error as err:
                raise ToolError(f"DuckDB error: {err}") from err
            finally:
                timer.cancel()
        truncated = fetch is not None and len(rows) > fetch
        rows = rows[:fetch] if fetch is not None else rows
        return {
            "columns": columns,
            "rows": [[_jsonable(v) for v in row] for row in rows],
            "row_count": len(rows),
            "truncated": truncated,
        }

    def tables(self) -> list[str]:
        result = self._run("SELECT table_name FROM information_schema.tables ORDER BY table_name")
        return [row[0] for row in result["rows"]]

    def query(self, sql: str, limit: int) -> dict[str, Any]:
        return self._run(sql, fetch=max(1, min(limit, self.max_rows)))


def build_server(db_path: Path, *, timeout_s: float = 10.0, max_rows: int = 200) -> MCPServer:
    wh = Warehouse(db_path, timeout_s=timeout_s, max_rows=max_rows)
    server = MCPServer(
        "sql",
        version="0.1.0",
        log_level="WARNING",
        instructions=(
            "Analytics over the company's DuckDB warehouse. Call list_tables and describe_table "
            "before writing SQL. Use run_query for reads; execute_statement needs human approval."
        ),
    )

    def _check_table(table: str) -> str:
        if table not in wh.tables():
            raise ToolError(f"Unknown table '{table}'. Available: {', '.join(wh.tables())}")
        return table

    @server.tool(title="List tables", annotations=READ_ONLY)
    def list_tables() -> list[dict[str, Any]]:
        """List all tables with their row counts and column names."""
        out = []
        for table in wh.tables():
            count = wh._run(f'SELECT COUNT(*) FROM "{table}"')["rows"][0][0]
            cols = wh._run(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position",
                [table],
            )["rows"]
            out.append({"table": table, "rows": count, "columns": [c[0] for c in cols]})
        return out

    @server.tool(title="Describe table", annotations=READ_ONLY)
    def describe_table(table: str, sample_rows: int = 3) -> dict[str, Any]:
        """Show column names, types and a few sample rows for one table."""
        table = _check_table(table)
        schema = wh._run(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [table],
        )
        sample = wh._run(f'SELECT * FROM "{table}" LIMIT {max(0, min(sample_rows, 10))}')
        return {
            "table": table,
            "columns": [{"name": r[0], "type": r[1], "nullable": r[2] == "YES"} for r in schema["rows"]],
            "sample": sample,
        }

    @server.tool(title="Run read-only SQL", annotations=READ_ONLY)
    def run_query(sql: str, limit: int = 100) -> dict[str, Any]:
        """Run ONE read-only DuckDB SQL query (SELECT/WITH) and return columns and rows.

        Results are capped at `limit` rows (max 200); `truncated` tells you if more exist,
        so aggregate in SQL instead of fetching raw rows.
        """
        validate_read_only(sql)
        return wh.query(sql, limit)

    @server.tool(title="Execute write statement", annotations=DESTRUCTIVE)
    def execute_statement(sql: str) -> dict[str, Any]:
        """Run ONE statement that changes data (INSERT/UPDATE/DELETE/CREATE/DROP/ALTER).

        Requires human approval. Prefer creating new tables over modifying existing ones.
        """
        validate_write(sql)
        result = wh._run(sql)
        affected = result["rows"][0][0] if result["rows"] else None
        return {"status": "ok", "affected_rows": affected}

    @server.resource("schema://tables", name="schema", description="Full warehouse schema as text.")
    def schema_resource() -> str:
        lines = []
        for table in wh.tables():
            cols = wh._run(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? "
                "ORDER BY ordinal_position",
                [table],
            )["rows"]
            lines.append(f"{table}(" + ", ".join(f"{c} {t}" for c, t in cols) + ")")
        return "\n".join(lines)

    @server.prompt(name="analyze_table", description="Kick off an exploratory analysis of one table.")
    def analyze_table(table: str) -> str:
        return (
            f"Explore the `{table}` table: describe its columns, check row counts and nulls, "
            "and report three interesting findings with the SQL you used."
        )

    return server


def main() -> None:  # pragma: no cover - entry point
    from mcp_agent_database.servers._runner import run_server

    run_server("sql")


if __name__ == "__main__":  # pragma: no cover
    main()
