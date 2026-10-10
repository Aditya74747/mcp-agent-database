# Refund and Returns Policy

_Owner: Customer Operations · Last reviewed: 2025-09-01_

## Standard refund window
Customers can return most products within **30 days** of delivery for a full refund.
The refund is issued to the original payment method within 5 business days after the
warehouse receives and inspects the item.

## Category exceptions
- **Headphones** can only be returned within **14 days** and must be unopened, for hygiene reasons.
- **Laptops** that have been opened are subject to a **10% restocking fee**.
- Storage devices must be returned with all data wiped; we cannot recover customer data.

## Enterprise customers
Customers in the `enterprise` segment have an extended return window of **45 days** and a
dedicated account manager who approves returns above EUR 5,000.

## How returns appear in the data
A returned order keeps its original rows in `orders` and `order_items`; only `orders.status`
changes to `returned`. Returned and cancelled orders must be excluded from revenue
(see the data dictionary).
