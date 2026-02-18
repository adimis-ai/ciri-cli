# Roadmap

Ciri's development priorities — where we are, where we're going, and what the community can help with.

---

## Current Capabilities (Feb 2026)

Ciri is a capable, production-ready local AI copilot with:

- **Two-level harness system** — global (`~/.local/share/ciri/`) + project (`.ciri/`) hierarchy for skills, toolkits, subagents, and memory
- **17 built-in skills** — PDF, DOCX, PPTX, XLSX, Canvas Design, Algorithmic Art, Frontend Design, Web Artifacts Builder, Theme Factory, Brand Guidelines, Slack GIF Creator, Internal Comms, Doc Co-Authoring, MCP Builder, Webapp Testing, Skill Creator, Subagent Builder
- **5 built-in subagents** — `web_researcher`, `skill_builder`, `toolkit_builder`, `subagent_builder`, `trainer_agent`
- **Script Executor** — isolated Python/JavaScript sandbox with HITL approval and streaming output
- **Web research** — CDP browser integration, BFS deep crawl, crawl4ai-powered extraction
- **14-middleware stack** — memory injection, skills, subagents, toolkits, HITL, retry, caching, summarization
- **`@`-autocomplete triggers** — `@files:`, `@folders:`, `@skills:`, `@toolkits:`, `@subagents:`
- **Multi-provider support** — Anthropic, OpenAI, Google, Groq, Mistral, Ollama, DeepSeek, OpenRouter, and more
- **Thread persistence** — SQLite database
- **Hot-reload** — add skills/toolkits without restart

---

## Near-Term (Q1–Q2 2026)

### `@harness:` Autocomplete Trigger

A new autocomplete trigger that lets you reference specific harness paths directly in your prompt. Completion shows `(Core)` and `(Current)` flags so you know which harness level a resource comes from.

```
You > Load the @harness:skills/competitor-research skill and analyze Notion's pricing
         ├── skills/pdf              (Core)
         ├── skills/docx             (Core)
         ├── skills/competitor-research  (Current)   ← select this
         └── toolkits/github         (Core)
```

This makes it trivial to explicitly invoke a specific harness resource without relying on automatic routing.

---

### Skill Marketplace

A curated registry of community-contributed skills. Browse, install, and update skills from the marketplace without leaving the Ciri prompt:

```
You > /marketplace install legal-contract-review
You > /marketplace search "financial modeling"
You > /marketplace update --all
```

Skills are published with metadata (domain, trigger conditions, dependencies, license). The `skill_builder` subagent will be able to submit skills to the marketplace after a local review step.

---

### Toolkit SDK with Tests and Templates

A first-class SDK for building MCP toolkit servers:

- Scaffold with `ciri scaffold-toolkit my-toolkit`
- FastMCP + Node.js templates with test harnesses included
- Automatic type generation from OpenAPI specs
- Publish directly to the skill marketplace

---

### Multi-Agent Team Runtime (GA)

The team runtime (`team_runtime.py`) is currently experimental. GA milestone:

- Stable API for composing multi-agent workflows
- Parallel subagent execution with result aggregation
- Visual team execution trace in the CLI
- Pre-built team configs for common workflows (research → write → review, analyze → plan → execute)

---

### Desktop Application (Tauri)

A packaged desktop app wrapping the Ciri CLI binary:

- No Python installation required
- Native system tray integration
- Persistent browser connection management
- Settings UI for model, harness, and HITL configuration

---

## Medium-Term (Q3–Q4 2026)

### Enhanced Offline / Local Model Support

- First-class Ollama integration with automatic model discovery
- Local model benchmarking to recommend the best available model for a task
- Offline mode: graceful degradation when no internet is available (local models + local crawl cache)

---

### Workspace Templates

Pre-built `.ciri/` harness configurations for common domains:

- **Software Engineering** — architecture memory, code review skill, GitHub toolkit
- **Marketing Agency** — brand guidelines skill, competitive research skill, content calendar toolkit
- **Legal / Compliance** — contract review skill, clause database skill, deadline tracking toolkit
- **Finance & Accounting** — spreadsheet analysis skill, report generation skill, accounting API toolkit
- **Research & Academia** — citation management toolkit, paper summarization skill, literature review subagent

Apply a template:

```
ciri init --template software-engineering
```

---

### Streaming Improvements

- Live token-by-token streaming in the approval panel before committing
- Stream progress indicators for long-running tool calls (deep web crawls, large file operations)
- Interrupt any running tool call with `Ctrl+C` without losing the thread

---

### CI/CD Multi-Platform Builds

Automated PyInstaller builds for Windows (x64), macOS (arm64 + x64), and Linux (x64) via GitHub Actions. Single-binary distribution with no Python dependency.

---

## Long-Term Vision

- **Ciri as a team member** — connect multiple Ciri instances running in parallel with shared memory and task queues
- **Domain certification** — community-verified skill packages for regulated industries (legal, medical, financial)
- **Webhook integrations** — Ciri listens for external events (GitHub push, Slack DM, calendar event) and reacts autonomously within configured safety boundaries
- **Cross-session learning** — Ciri builds on her own successes and failures across sessions, not just memory files

---

## Contributions Welcome

The roadmap is community-driven. If you want to work on any of these items — or propose a new direction — open an issue on GitHub:

[github.com/adimis-ai/ciri/issues](https://github.com/adimis-ai/ciri/issues)

Label your issue with `roadmap` to start the discussion. Features with strong community demand and clear implementation paths get prioritized.

See [Contributing](contributing.md) to get started.
