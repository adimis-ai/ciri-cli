# CIRI Copilot Documentation

Welcome to the official CIRI Copilot documentation. This site covers everything a developer or user needs to know: architecture, development setup, how to extend CIRI with skills, toolkits, and subagents, testing, packaging, and contribution guidelines.

Use the sidebar to navigate topics. If a page references code or files, paths are relative to the project root where the repo is checked out.

---

## Quick architecture map

```mermaid
flowchart LR
  A[User Terminal] --> B[CLI (ciri)]
  B --> C[Copilot Graph]
  C --> D[LLM Providers]
  C --> E[Skills & Toolkits]
  C --> F[Subagents]
  C --> G[Local DB (SQLite)]
  style A fill:#f9f,stroke:#333,stroke-width:1px
  style B fill:#b6e3ff,stroke:#333
  style C fill:#ffdcb1,stroke:#333
  style D fill:#d1f7c4,stroke:#333
```

