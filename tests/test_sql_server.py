from __future__ import annotations

import pytest

from mcp_agent_database.servers.sql_server import (
    SQLValidationError,
    build_server,
    validate_read_only,
    validate_write,
)
from tests.conftest import call, payload

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "WITH t AS (SELECT * FROM orders) SELECT COUNT(*) FROM t",
        "SELECT 1 UNION ALL SELECT 2",
        "DESCRIBE orders",
    ],
)
def test_read_only_accepts_queries(sql: str) -> None:
    validate_read_only(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "DROP TABLE orders",
        "INSERT INTO orders VALUES (1, 1, '2025-01-01', 'x')",
        "UPDATE orders SET status = 'x'",
        "SELECT 1; DROP TABLE orders",
        "ATTACH 'other.db'",
        "COPY orders TO 'out.csv'",
        "CREATE TABLE x AS SELECT 1",
        "",
    ],
)
def test_read_only_rejects_writes_and_multi_statements(sql: str) -> None:
    with pytest.raises(SQLValidationError):
        validate_read_only(sql)


@pytest.mark.parametrize("sql", ["ATTACH 'x.db'", "COPY orders TO 'x.csv'", "SET threads = 1", "SELECT 1"])
def test_write_tool_rejects_escape_hatches(sql: str) -> None:
    with pytest.raises(SQLValidationError):
        validate_write(sql)


async def test_list_and_describe(settings) -> None:
    server = build_server(settings.db_path)
    tables = payload(await call(server, "list_tables"))
    assert {t["table"] for t in tables} == {"customers", "products", "orders", "order_items"}
    orders = next(t for t in tables if t["table"] == "orders")
    assert orders["rows"] == 2500

    desc = payload(await call(server, "describe_table", {"table": "customers"}))
    assert [c["name"] for c in desc["columns"]][:2] == ["customer_id", "name"]
    assert desc["sample"]["row_count"] == 3


async def test_describe_unknown_table_is_helpful(settings) -> None:
    result = await call(build_server(settings.db_path), "describe_table", {"table": "nope"})
    assert result.is_error
    assert "Available: customers" in result.content[0].text


async def test_run_query_returns_json_safe_rows(settings) -> None:
    out = payload(
        await call(
            build_server(settings.db_path),
            "run_query",
            {"sql": "SELECT order_date, unit_price FROM orders JOIN order_items USING (order_id) LIMIT 2"},
        )
    )
    assert out["columns"] == ["order_date", "unit_price"]
    date, price = out["rows"][0]
    assert isinstance(date, str) and len(date) == 10
    assert isinstance(price, float)


async def test_run_query_truncates_at_limit(settings) -> None:
    server = build_server(settings.db_path, max_rows=50)
    out = payload(await call(server, "run_query", {"sql": "SELECT * FROM orders", "limit": 500}))
    assert out["row_count"] == 50
    assert out["truncated"] is True


async def test_external_file_access_is_blocked(settings) -> None:
    result = await call(
        build_server(settings.db_path), "run_query", {"sql": "SELECT * FROM read_csv('/etc/passwd')"}
    )
    assert result.is_error
    assert "disabled by configuration" in result.content[0].text


async def test_settings_cannot_be_changed_back(settings) -> None:
    server = build_server(settings.db_path)
    result = await call(server, "execute_statement", {"sql": "SET enable_external_access = true"})
    assert result.is_error


async def test_execute_statement_writes(settings) -> None:
    server = build_server(settings.db_path)
    create = "CREATE TABLE top_customers AS SELECT customer_id FROM customers LIMIT 5"
    assert payload(await call(server, "execute_statement", {"sql": create}))["status"] == "ok"
    rows = payload(await call(server, "run_query", {"sql": "SELECT COUNT(*) FROM top_customers"}))
    assert rows["rows"] == [[5]]


async def test_foreign_keys_protect_integrity(settings) -> None:
    server = build_server(settings.db_path)
    result = await call(server, "execute_statement", {"sql": "DELETE FROM orders WHERE status = 'cancelled'"})
    assert result.is_error
    assert "foreign key" in result.content[0].text


async def test_query_timeout(settings) -> None:
    server = build_server(settings.db_path, timeout_s=0.2)
    slow = "SELECT COUNT(*) FROM range(1000000000) a, range(100) b WHERE a.range % 7 = b.range"
    result = await call(server, "run_query", {"sql": slow})
    assert result.is_error
    assert "timeout" in result.content[0].text


def test_missing_database_fails_fast(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="mcp-agent-database seed"):
        build_server(tmp_path / "missing.duckdb")
