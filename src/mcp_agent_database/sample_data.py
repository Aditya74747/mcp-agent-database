"""Deterministic sample data so demos, tests and evals are reproducible.

Creates a small e-commerce warehouse in DuckDB (customers, products, orders, order_items)
and a sandbox folder for the workspace server. Same seed -> identical data on every machine.
"""

from __future__ import annotations

import csv
import random
import tempfile
from datetime import date, timedelta
from pathlib import Path

import duckdb

COUNTRIES = ["Germany", "France", "India", "United States", "Netherlands", "Spain", "Poland", "Sweden"]
SEGMENTS = ["consumer", "small_business", "enterprise"]
CATEGORIES = {
    "Laptops": (650, 2400),
    "Monitors": (120, 900),
    "Keyboards": (25, 220),
    "Headphones": (30, 450),
    "Storage": (40, 380),
}
STATUSES = ["delivered"] * 80 + ["shipped"] * 8 + ["returned"] * 7 + ["cancelled"] * 5
FIRST = ["Anna", "Ben", "Chen", "Divya", "Elif", "Felix", "Grace", "Hugo", "Ines", "Jonas", "Kavya", "Lena"]
LAST = ["Schmidt", "Martin", "Sharma", "Garcia", "Nowak", "Berg", "Rossi", "Kim", "Patel", "Weber"]

SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL,
    country VARCHAR NOT NULL,
    segment VARCHAR NOT NULL,
    signup_date DATE NOT NULL
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL,
    category VARCHAR NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date DATE NOT NULL,
    status VARCHAR NOT NULL
);
CREATE TABLE order_items (
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL
);
"""


def build_database(path: Path, *, seed: int = 42, n_customers: int = 250, n_orders: int = 2500) -> Path:
    """(Re)create the sample DuckDB database at ``path``."""
    rng = random.Random(seed)  # noqa: S311 - reproducible sample data, not crypto
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    customers = []
    for cid in range(1, n_customers + 1):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        signup = date(2024, 1, 1) + timedelta(days=rng.randrange(0, 540))
        customers.append((cid, name, rng.choice(COUNTRIES), rng.choice(SEGMENTS), signup))

    products = []
    pid = 1
    for category, (low, high) in CATEGORIES.items():
        for i in range(1, 9):
            price = round(rng.uniform(low, high), 2)
            products.append((pid, f"{category[:-1]} {chr(64 + i)}{pid}", category, price))
            pid += 1

    orders, items = [], []
    for oid in range(1, n_orders + 1):
        cust = rng.choice(customers)
        start = max(cust[4], date(2025, 1, 1))
        order_date = start + timedelta(days=rng.randrange(0, max(1, (date(2025, 12, 31) - start).days)))
        orders.append((oid, cust[0], order_date, rng.choice(STATUSES)))
        for product in rng.sample(products, k=rng.choice([1, 1, 1, 2, 2, 3])):
            # Enterprise customers buy in bulk; prices occasionally discounted.
            qty = rng.randint(1, 3) if cust[3] != "enterprise" else rng.randint(2, 12)
            discount = rng.choice([1.0, 1.0, 1.0, 0.9, 0.85])
            items.append((oid, product[0], qty, round(product[3] * discount, 2)))

    con = duckdb.connect(str(path))
    try:
        con.execute(SCHEMA)
        # Bulk-load through CSV: orders of magnitude faster than executemany in DuckDB.
        with tempfile.TemporaryDirectory() as tmp:
            for table, rows in (
                ("customers", customers),
                ("products", products),
                ("orders", orders),
                ("order_items", items),
            ):
                csv_path = Path(tmp) / f"{table}.csv"
                with csv_path.open("w", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerows(rows)
                con.execute(f"COPY {table} FROM '{csv_path.as_posix()}' (HEADER false)")
    finally:
        con.close()
    return path


SANDBOX_FILES = {
    "README.txt": (
        "This is the agent's sandbox. The workspace MCP server can only read and write inside this folder.\n"
    ),
    "notes/meeting-2025-11-04.md": (
        "# Ops sync, 4 Nov 2025\n\n"
        "- Returns spiked for Headphones in October; check with the supplier.\n"
        "- Germany is still the largest market by revenue.\n"
        "- Action: draft a short Q4 summary for leadership (owner: data team).\n"
    ),
    "todo.txt": "1. Clean up old exports\n2. Write Q4 revenue summary\n",
}


def build_sandbox(path: Path) -> Path:
    for rel, content in SANDBOX_FILES.items():
        target = path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(content, encoding="utf-8")
    return path
