# Core Harness Architecture

CIRI uses a **two-level harness** to separate capabilities that belong to your entire development environment from those specific to a single project. This is one of the most important architectural decisions in CIRI — it's what makes skills, toolkits, and subagents reusable across all your workspaces.

---

## The Two Levels

```
~/.local/share/ciri/          ← Core Harness (global, OS-level)
├── skills/                    ← Available in EVERY workspace
│   ├── pdf/
│   ├── docx/
│   ├── pptx/
│   ├── xlsx/
│   ├── webapp-testing/
│   └── <your-custom-global-skills>/
├── toolkits/                  ← Global MCP servers
│   └── <your-global-toolkits>/
├── subagents/                 ← Global specialized agent configs
│   └── <your-global-subagents>/
└── memory/                    ← Cross-project memory (rarely written here)

<your-project>/
└── .ciri/                     ← Project Harness (workspace-level)
    ├── skills/                ← This project only
    │   └── <project-specific-skill>/
    ├── toolkits/              ← This project only
    │   └── <project-toolkit>/
    ├── subagents/             ← This project only
    │   └── <project-subagent>.yaml
    └── memory/                ← This project's persistent memory
        ├── AGENT.md           ← Primary workspace index
        └── <topic>.md
```

On **Linux**, the core harness is at `~/.local/share/ciri/`.
On **macOS**, it is at `~/Library/Application Support/ciri/`.
On **Windows**, it is at `%APPDATA%\ciri\`.

---

## Loading Priority

When Ciri loads skills, toolkits, subagents, or memory, the **core harness always takes precedence** — loaded first, before the project harness:

```
Core harness  →  Project harness
(global)          (local)
```

### Skills

The `SkillsMiddleware` discovers skills in this order:
1. Core harness (`~/.local/share/ciri/skills/`)
2. Project harness (`.ciri/skills/`)

If a skill with the same name exists in both, the **core harness version wins**.

### Toolkits

The `ToolkitInjectionMiddleware` uses an ordered dictionary — first entry wins:
1. Core harness toolkits
2. Project harness toolkits

### SubAgents

The `SubAgentMiddleware` deduplicates by **file stem** (the filename without extension). Core harness subagents registered first cannot be overridden by project-level files with the same name.

### Memory

The `MemoryMiddleware` loads memory **additively** — all `.md` files from both locations are injected into every turn's system prompt:
1. Core harness memory first
2. Project harness memory second (appended)

---

## Default Skills Bootstrap

When Ciri starts for the first time, the `SkillsMiddleware._bootstrap_default_skills()` method copies all built-in skills from `src/skills/` into the **core harness** `~/.local/share/ciri/skills/`. This happens once — existing skills are never overwritten. This is why Ciri has PDF, Office document, and other built-in capabilities available immediately in any new project.

---

## Where the Trainer Agent Writes

When you run `/sync`, the **Trainer Agent** decides where to put new capabilities:

| Type | Default target | Rationale |
|---|---|---|
| **Skill** | Core harness | Skills are domain playbooks — reusable across projects |
| **Toolkit** | Core harness | MCP integrations are typically service-wide |
| **SubAgent** | Core harness | Agent roles are usually general |
| **Memory** | Project harness | Project memory is workspace-specific |

You can instruct the Trainer Agent to place a new component in the project harness if it should be private to that workspace:

```
You > Create a skill for analyzing our internal P&L format — keep it project-specific
```

---

## `get_core_harness_dir()` in Code

The canonical function is in `src/utils.py`:

```python
from .utils import get_core_harness_dir

core_dir = get_core_harness_dir()
# Returns: ~/.local/share/ciri (Linux) | ~/Library/Application Support/ciri (macOS)
# Automatically creates skills/, toolkits/, subagents/, memory/ subdirs on first call
```

---

## Implications for Teams

In a team setup, you may want to commit your `.ciri/` project harness to version control so all team members share the same project-specific skills and memory. The core harness is personal and machine-specific — do not commit it.

```bash
# .gitignore — exclude sensitive runtime state but commit project harness content
.ciri/memory/     # optional: keep private or commit shared memory
# commit .ciri/skills/, .ciri/toolkits/, .ciri/subagents/
```
