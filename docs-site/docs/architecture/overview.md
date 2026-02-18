# Architecture Overview

CIRI is a **local-first, multi-agent orchestration system** built on LangChain and LangGraph. Its core design principle is extensibility: every capability can be added, modified, or replaced at runtime without restarting.

---

## Core Module Map

| Module | File | Responsibility |
|---|---|---|
| **CLI Entrypoint** | `src/__main__.py` | Rich terminal UI, Prompt Toolkit REPL, interrupt rendering, startup sequence |
| **Copilot Graph** | `src/copilot.py` | Assembles the LangGraph agent with all middlewares and subagents |
| **Controller** | `src/controller.py` | Thread management, graph execution, streaming |
| **Backend** | `src/backend.py` | `CiriBackend` — subprocess execution, streaming output callbacks |
| **Database** | `src/db.py` | `CopilotDatabase` — SQLite threads table, thread CRUD |
| **Serializers** | `src/serializers.py` | `LLMConfig` — multi-provider model initialization |
| **Middlewares** | `src/middlewares/` | Context injection, safety gates, dynamic capability loading |
| **Subagents** | `src/subagents/` | 5 built-in specialists (web_researcher, builders, trainer) |
| **Toolkits** | `src/toolkit/` | Core tools: web crawler, script executor, human follow-up |
| **Skills** | `src/skills/` | 17 built-in domain playbooks |
| **Utils** | `src/utils.py` | App dirs, browser detection, gitignore parsing, harness discovery |

---

## Startup Sequence

```mermaid
sequenceDiagram
  autonumber
  participant CLI as src/__main__.py
  participant Copilot as create_copilot()
  participant LLM as LLM Provider
  participant Browser as Chrome/Edge CDP
  participant Builders as Subagent Factory
  participant MW as Middleware Stack
  participant Graph as LangGraph

  CLI->>CLI: ensure_playwright_installed()
  CLI->>CLI: load_all_dotenv() + setup_api_keys()
  CLI->>Copilot: create_copilot(llm_config, ...)
  Copilot->>LLM: llm_config.init_langchain_model()
  Copilot->>Browser: resolve_browser_profile() + launch_browser_with_cdp()
  Copilot->>Builders: build_web_researcher_agent(model, cdp_endpoint)
  Copilot->>Builders: build_skill_builder_agent(web_researcher)
  Copilot->>Builders: build_toolkit_builder_agent(web_researcher)
  Copilot->>Builders: build_subagent_builder_agent(web_researcher)
  Copilot->>Builders: build_trainer_agent(...)
  Copilot->>MW: assemble middleware stack
  Copilot->>Graph: compile state graph
  Graph-->>CLI: CompiledStateGraph
  CLI->>CLI: wrap with CopilotController + start REPL
```

---

## Middleware Stack

The middleware stack wraps every LLM call, injecting context and enforcing safety:

```
User Message
  → InjectNamesMiddleware        (tool + skill + subagent registry)
  → MemoryMiddleware             (workspace .md files)
  → SkillsMiddleware             (domain playbooks)
  → SubAgentMiddleware           (delegation routing)
  → ToolkitInjectionMiddleware   (MCP server tools)
  → HumanInTheLoopMiddleware     (approval gates)
  → TodoListMiddleware           (task tracking)
  → AnthropicPromptCachingMiddleware  (token caching, Anthropic only)
  → LLM Call
  → ToolRetryMiddleware          (retry on failure, skip GraphInterrupt)
  → PatchToolCallsMiddleware     (normalize cross-provider tool call format)
  → SummarizationMiddleware      (compress long contexts)
```

→ [Full Middleware Reference](../internals/middlewares.md)

---

## Two-Level Core Harness

Skills, toolkits, subagents, and memory all follow a **two-level discovery pattern**:

```
~/.local/share/ciri/     ← Core harness (global, all projects)
└── skills/ toolkits/ subagents/ memory/

<project>/.ciri/         ← Project harness (this workspace only)
└── skills/ toolkits/ subagents/ memory/
```

Core harness always loads first. Project harness supplements or overrides on a name-basis.

→ [Core Harness Architecture](core-harness.md)

---

## Data Persistence

| Data | Location | Technology |
|---|---|---|
| Conversation threads | `~/.ciri/data/ciri.db` | SQLite (`threads` table) |
| LangGraph checkpoints | Same database | LangGraph `AsyncSqliteSaver` |
| LangGraph store | Same database | `AsyncSqliteStore` |
| Workspace memory | `.ciri/memory/*.md` | Plain markdown files |
| User settings | `.ciri/settings.json` | JSON |
| API keys | `~/.ciri/.env` + OS keychain | dotenv + `keyring` |

The SQLite database supports **resume-from-checkpoint** — Ciri can continue any thread from exactly where it left off after a restart.

Thread persistence is handled by a local SQLite database.

---

## Streaming Architecture

Ciri streams responses in real time using LangGraph's dual-mode streaming:

```
["updates", "messages"]
  updates  → node state changes + GraphInterrupt signals
  messages → AIMessageChunk tokens (token-by-token text streaming)
```

The `CopilotController.run()` method yields these events to the CLI, which renders them via Rich as they arrive.

---

## Design Goals

1. **Autonomy** — High agency to plan and execute multi-step tasks across any domain
2. **Self-Evolution** — Permanent capability expansion via Skills, Toolkits, and SubAgents
3. **Workspace-Awareness** — Deep integration with local files, memory, and domain context
4. **Safety** — Mandatory Human-in-the-Loop for consequential actions
5. **Extensibility** — Every layer is pluggable: LLM provider, middleware, subagent, skill
6. **Local-First** — No cloud dependency for core function; all data stored locally
