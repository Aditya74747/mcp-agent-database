# Incident Response Runbook

_Owner: Platform Engineering_

## Severity levels
| Level | Example | First response | Updates |
|---|---|---|---|
| SEV1 | Checkout down for all users | **15 minutes** | every 30 minutes |
| SEV2 | Payments failing for one provider | 30 minutes | every hour |
| SEV3 | Slow dashboard, workaround exists | next business day | daily |

## Steps
1. Acknowledge the page and open an incident channel named `inc-<date>-<short-name>`.
2. Assign an incident commander; they coordinate, they do not debug.
3. Mitigate first (roll back, feature-flag off), investigate second.
4. Write a blameless postmortem within **5 business days** for SEV1 and SEV2.
