# FAQ

Answers to the most common questions about Ciri.

---

## General

### What is Ciri?

Ciri is a local-first, multi-agent AI copilot that adapts to any domain. Unlike a generic chat assistant, Ciri learns your workspace — your stack, conventions, domain vocabulary, and recurring workflows — and becomes a specialist for it. She can help with software engineering, marketing, finance, legal research, document processing, data analysis, brand design, and anything else she is trained to handle.

### Is Ciri only for software developers?

No. Ciri is domain-agnostic. The `/sync` command (Trainer Agent) analyzes your workspace and writes memory files that encode your specific context. A law firm can train Ciri on case workflows. A marketing agency can train her on campaign playbooks. A startup can train her to know the entire product and company. Skills, subagents, and toolkits make her capabilities arbitrarily extensible.

### How is Ciri different from Claude Code or Cursor?

| | Ciri | Claude Code / Cursor |
|---|---|---|
| Domain | Any domain | Primarily software |
| Workspace memory | Persistent `.md` files, injected every turn | Session or project-scoped |
| Self-improvement | Builds new skills/toolkits/subagents via `/sync` | Fixed tool set |
| Custom subagents | Full YAML config + MCP toolkit ecosystem | Limited |
| Local-first | Yes — all state in local SQLite | Cloud-synced |
| Model provider | Any (OpenAI, Anthropic, Groq, Ollama, and 10+ more) | Provider-specific |

---

## Models & Providers

### What model providers does Ciri support?

Ciri works with any provider supported by LangChain, including:

- **Anthropic** — Claude Sonnet 4.6, Claude Opus 4.6, Claude Haiku 4.5
- **OpenAI** — GPT-4o, GPT-4o-mini, o1, o3
- **Google** — Gemini 2.0 Flash, Gemini 1.5 Pro
- **Groq** — Llama 3.3 70B, Mixtral 8x7B (fastest inference)
- **Mistral** — Mistral Large, Mixtral
- **Cohere** — Command R+
- **Together AI** — Llama, Qwen, DeepSeek
- **Fireworks AI** — Fast open-source serving
- **DeepSeek** — DeepSeek-V3, DeepSeek-R1
- **Ollama** — Any local model (no API key needed)
- **OpenRouter** — Universal gateway to 200+ models via a single API key

Set your provider key as an environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
```

### Can I switch models mid-conversation?

Yes. Use `/change-model` at the `You >` prompt to list and select any available model. The change takes effect on the next message — conversation history is preserved.

### Can I run models locally without an internet connection?

Yes. Install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull llama3.3
```

Then start Ciri and select the Ollama model via `/change-model`. No API key or internet connection needed for inference. Note: web browsing and web-research tools still require internet.

---

## Skills

### Where do I add a new skill?

Skills live in the two-level harness:

- **Core harness** — `~/.local/share/ciri/skills/<skill-name>/` — available across all projects
- **Project harness** — `.ciri/skills/<skill-name>/` — available only in this workspace

After creating the skill directory with a valid `SKILL.md`, run `/sync` or just restart Ciri — skills are hot-reloaded.

### What is the required structure for a skill?

Each skill is a directory containing a `SKILL.md` file with YAML frontmatter:

```
my-skill/
└── SKILL.md
```

Minimal `SKILL.md`:

```markdown
---
name: my-skill
description: Use this skill when the user wants to [trigger condition]. It handles [capabilities].
license: MIT
---

# My Skill Guide

## How to Use
[Playbook instructions Ciri follows when this skill is active]
```

The `description` field is what Ciri uses to decide when to activate the skill. Write it as a trigger condition, not a feature list.

### How do I ask Ciri to build a new skill for me?

Just describe what you need:

```
You > Build a skill for analyzing our weekly sales reports from Salesforce exports
```

Ciri delegates to the `skill_builder` subagent, which creates the skill directory and `SKILL.md` with an appropriate playbook. Run `/sync` to register it.

### Can a skill contain additional files?

Yes. You can include `REFERENCE.md`, `FORMS.md`, or any number of supporting `.md` or asset files inside the skill directory. `SKILL.md` is the primary file Ciri reads; additional files can be referenced from the playbook body.

---

## Memory

### Where does Ciri store workspace memory?

Memory files live in two locations:

- `~/.local/share/ciri/memory/` — global preferences (cross-project)
- `.ciri/memory/` — project-specific knowledge (`AGENT.md`, `architecture.md`, etc.)

All `.md` files from both locations are injected into every conversation turn automatically.

### How do I update Ciri's memory?

Three ways:

1. **Automatic** — Run `/sync` to trigger the Trainer Agent, which scans and updates memory files.
2. **Ciri auto-update** — After completing significant work, Ciri updates relevant memory files herself (e.g., after refactoring a module).
3. **Manual** — Edit any `.ciri/memory/*.md` directly. Changes take effect on the next message.

### Will Ciri read my entire codebase every turn?

No. The Trainer Agent reads the codebase during `/sync` and distills it into concise memory files. Those summaries (not raw code) are injected each turn. Keep individual memory files under 200 lines for best performance.

---

## Human-in-the-Loop (HITL)

### Why does Ciri ask for approval before running commands?

Ciri uses the `HumanInTheLoopMiddleware` as a mandatory safety gate. Before executing `execute`, `edit_file`, or `write_file`, Ciri displays the full tool arguments and waits for your explicit approval. This prevents unintended side effects, especially in production environments.

### What are my options when a tool approval prompt appears?

When Ciri pauses for approval, you can:

- **Approve** — Execute as-is
- **Edit** — Modify the JSON arguments before execution
- **Reject** — Cancel this specific tool call; Ciri will replan

### Can I disable HITL for trusted operations?

You can configure `interrupt_on` in `create_copilot()` to change which tool names trigger approval. Setting `all_allowed=True` disables all interrupts — only do this in fully automated/trusted environments. The default interrupt triggers are `execute`, `edit_file`, and `write_file`.

### What is the `execute_script` approval?

When Ciri uses the Script Executor tool (`execute_script`), it shows a special approval panel with the full script content, language, dependencies, and working directory. You can approve, edit the script, or reject it before any code runs. The script runs in a fully isolated venv/node_modules environment.

---

## Data & Privacy

### Does Ciri send my code to the cloud?

Ciri sends conversation context (including any code or documents you share) to the configured LLM provider. The provider processes this for inference. All conversation state (threads, checkpoints) is stored locally in `~/.ciri/data/ciri.db`.

### Where are API keys stored?

API keys are read from environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) or from the OS keychain. Never commit keys to version control.


---

## Browser & Web Research

### How does Ciri access the web?

Ciri uses the `web_crawler` tool for content extraction and can use a real browser session via Chrome DevTools Protocol (CDP). Launch Chrome/Chromium with remote debugging enabled (see [Web Research guide](features/web-research.md)), and Ciri will connect to it.

### Why does Ciri copy my browser profile?

Chrome v136+ requires that remote-debug sessions use a dedicated profile directory. Ciri copies your selected profile to a temporary location and launches CDP on port 9222, preserving your login sessions without modifying the original profile.

### Can Ciri work without a browser?

Yes. The `web_crawler` tool works without CDP using headless crawling. Browser-based research (JavaScript-heavy sites, authenticated sessions) requires the CDP connection, but static content extraction works immediately.

---

## Toolkits (MCP Servers)

### What is a toolkit?

A toolkit is a Model Context Protocol (MCP) server that Ciri connects to as a client. Toolkits expose tools (GitHub, Jira, Slack, databases, internal APIs) as structured function calls. Ciri discovers and connects to toolkits automatically on startup.

### Where do I put toolkits?

- **Core harness** — `~/.local/share/ciri/toolkits/<toolkit-name>/` — available across all projects
- **Project harness** — `.ciri/toolkits/<toolkit-name>/` — available only in this workspace

Each toolkit directory must contain the MCP server manifest and runner. See the [Toolkits Guide](toolkits-guide.md).

### Can Ciri build a toolkit for me?

Yes. Describe the API you need:

```
You > Build a toolkit to connect to our internal metrics API at https://metrics.internal/v2/
```

Ciri delegates to the `toolkit_builder` subagent, which scaffolds the MCP server and writes it to the appropriate harness directory.

---

## Threads & State

### How do threads work?

Each conversation thread is a persistent SQLite record. Threads maintain full message history and LangGraph checkpoint state. Use `/new-thread` to start fresh, `/switch-thread` to resume a previous session, and `/threads` to list all threads.

### Can I export conversation history?

Not via a built-in command currently. You can query the SQLite database directly at `~/.ciri/data/ciri.db` to extract thread data.

---

## Troubleshooting

### Ciri says "model not found" on startup

You haven't set an API key. Set the environment variable for your provider (e.g., `export ANTHROPIC_API_KEY="..."`) and restart.

### Playwright tools fail with "browser not connected"

Launch Chrome with CDP enabled before starting Ciri:

```bash
# Linux / macOS
google-chrome --remote-debugging-port=9222 &
```

Then start `ciri`. See the [Web Research guide](features/web-research.md) for platform-specific instructions.

### Ciri asks for approval 3 times and then fails

This was a known bug (ToolRetryMiddleware retrying GraphInterrupt). If you're on an older version, update to the latest. The fix uses `retry_on=lambda exc: not isinstance(exc, GraphInterrupt)`.

### Skills aren't being discovered after I added a new one

Run `/sync` to trigger re-discovery. If that doesn't help, verify that your `SKILL.md` has valid YAML frontmatter with `name` and `description` fields. Check that the skill directory is in either `~/.local/share/ciri/skills/` or `.ciri/skills/`.

### Ciri doesn't find Git Bash on Windows

Make sure Git for Windows is installed and `bash.exe` is on your PATH (`C:\Program Files\Git\bin`). Alternatively, use WSL — Ciri also detects WSL bash automatically.
