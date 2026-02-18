# Persistent Workspace Memory

Ciri maintains a long-term understanding of your workspace through a **two-level markdown-based memory system**. This is not a database or vector store — it is a curated set of `.md` files that are injected into every single conversation turn, ensuring Ciri always has your context at hand.

---

## Why Memory Matters

Standard LLM contexts are ephemeral. Without persistent memory, Ciri would forget your stack, your conventions, your domain vocabulary, and your past decisions after every session. The `MemoryMiddleware` solves this by making critical knowledge permanent.

With workspace memory:
- Ciri knows your architecture without you explaining it every time
- She uses your preferred naming conventions, tone, and style automatically
- She remembers the lessons learned from past sessions
- She understands your domain — whether that's fintech, legal, marketing, or engineering

---

## The Two-Level Memory System

```
~/.local/share/ciri/memory/    ← Global memory (cross-project)
├── global-preferences.md      ← Your universal preferences and working style

<project>/
└── .ciri/memory/              ← Workspace memory (this project)
    ├── AGENT.md               ← Master index — Ciri reads this first
    ├── architecture.md        ← System structure, modules, data flow
    ├── conventions.md         ← Code style, naming, tone, formatting rules
    ├── domain.md              ← Business concepts, glossary, key entities
    ├── workflows.md           ← Recurring processes and procedures
    └── <topic>.md             ← Any additional topic files you create
```

**Load order** (additive — both levels are always injected):
1. Core harness memory (`~/.local/share/ciri/memory/`) — global preferences
2. Project harness memory (`.ciri/memory/`) — this workspace's knowledge

All `.md` files from both locations are injected into the system prompt on every turn.

---

## AGENT.md — The Workspace Index

`AGENT.md` is the primary memory file. Ciri is instructed to read it at the start of complex tasks. It serves as an index and quick-reference for everything important about your workspace.

A well-structured `AGENT.md` contains:

```markdown
# Workspace: [Project Name]

## Overview
[1-3 sentences describing what this workspace is and does]

## Directory Structure
- `src/` — [description]
- `docs/` — [description]

## Key Technologies
- Python 3.12 + FastAPI
- PostgreSQL (via SQLAlchemy)
- React + Tailwind on the frontend

## Common Tasks
- Run dev server: `uvicorn src.main:app --reload`
- Run tests: `pytest tests/ -q`
- Deploy: `./scripts/deploy.sh staging`

## Conventions
- All API endpoints use snake_case
- Models use Pydantic v2
- Prefer functional style (avoid classes where possible)
```

---

## How Memory Is Populated

### Automatic (via `/sync`)

The `/sync` command invokes the **Trainer Agent**, which:

1. Scans your project root (excluding `.ciri`)
2. Analyzes your workspace's purpose, structure, and domain
3. Creates or updates memory files with accurate, actionable summaries
4. Writes an `AGENT.md` index if it doesn't exist

```
You > /sync
```

After sync, run `/sync` again whenever your project changes significantly.

### Ciri Updates Memory Automatically

After completing significant work, Ciri is instructed to update relevant memory files:

```
Ciri > I've refactored the authentication module. Updating .ciri/memory/architecture.md
       with the new session management approach...
```

### Manual Editing

Edit any `.ciri/memory/*.md` file directly. Changes take effect on the next conversation turn — no restart needed.

---

## Best Practices

**Keep it concise and actionable.**
Memory files are injected on every turn. Bloated files waste context tokens. Focus on facts Ciri will actually need — not logs or histories.

**Use specific, Ciri-directed language.**
Write memory as instructions to Ciri, not documentation for humans:
- ❌ "The authentication module uses JWT tokens."
- ✅ "When working with auth, always use the `AuthService` class in `src/auth/service.py`. Never bypass the JWT middleware."

**Document "the flavor" of your domain.**
What makes your workspace unique? Your naming conventions, preferred patterns, communication tone, domain vocabulary. This is what transforms Ciri from a generic assistant into a specialist.

**Don't duplicate external docs.**
If you have large external documentation, store a high-level summary + the location, not the full content:
```markdown
## External References
- API docs: https://internal.company.com/api-docs (requires VPN)
- Architecture decision records: `docs/adr/`
- Design system: `packages/design-system/README.md`
```

**Split by topic.**
Don't put everything in `AGENT.md`. Create separate files for architecture, conventions, domain, and workflows. Each file should be under 200 lines.

---

## Memory File Reference

| File | Content | Updated by |
|---|---|---|
| `AGENT.md` | Master index, workspace overview | Trainer Agent on `/sync` |
| `architecture.md` | Module structure, data flow, dependencies | Ciri after refactors |
| `conventions.md` | Style, naming, communication tone | Manual or via `/sync` |
| `domain.md` | Business concepts, glossary, entities | Manual or via `/sync` |
| `workflows.md` | Recurring processes, runbooks | Manual or via `/sync` |
| `<custom>.md` | Any topic you add | You or Ciri |
