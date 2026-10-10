# Data Dictionary: Shop Warehouse (DuckDB)

_Owner: Data Team_

This describes the tables exposed by the **sql** MCP server.

## Tables
- `customers(customer_id, name, country, segment, signup_date)`: one row per customer.
  `segment` is one of `consumer`, `small_business`, `enterprise`.
- `products(product_id, name, category, unit_price)`: catalogue with list prices.
  Categories: Laptops, Monitors, Keyboards, Headphones, Storage.
- `orders(order_id, customer_id, order_date, status)`: one row per order.
  `status` is one of `delivered`, `shipped`, `returned`, `cancelled`.
- `order_items(order_id, product_id, quantity, unit_price)`: line items. `unit_price` here
  is the **price actually paid** (after discounts), which can be lower than the list price.

## Official metric definitions
- **Revenue** = `SUM(order_items.quantity * order_items.unit_price)` over orders whose status is
  `delivered` or `shipped`. Returned and cancelled orders are **excluded**.
- **Return rate** = returned orders / all orders, computed per category or overall.
- **Active customer** = a customer with at least one non-cancelled order in the period.

All monetary values are in EUR.
