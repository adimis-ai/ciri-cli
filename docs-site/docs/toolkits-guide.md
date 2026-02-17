# Toolkits Guide

Toolkits bridge CIRI to external systems or APIs.

- Place toolkit adapters under src/toolkit/ or .ciri/toolkits/
- Provide configuration and credentials via .env or per-tool config
- Prefer small, well-tested adapters that map to a limited surface area

Example toolkit patterns:
- API client wrapper exposing a simple call interface
- Filesystem toolkit providing safe file read/write helpers
- Browser automation adapters using Playwright

Security note: never store secrets in the repo; use environment variables or OS keychains.
