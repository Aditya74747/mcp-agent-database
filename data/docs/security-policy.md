# Security Policy (excerpt)

_Owner: Security Team_

- API keys and tokens must be rotated every **90 days** and stored in the secrets manager,
  never in source code or `.env` files committed to git.
- Personal data (names, emails, addresses) must not be written to application logs.
- AI agents may read internal data, but any action that **writes, deletes or sends data
  outside the company** requires explicit approval by a human.
- Production database access is read-only by default; write access is granted per ticket.
