# Getting Started

This guide covers installation, first-run setup, and the essential commands to get Ciri working in your workspace in under five minutes.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.12+** | Required |
| **Git** | For cloning the repo |
| **uv** | Recommended package manager — [install uv](https://docs.astral.sh/uv/) |
| **An LLM API key** | OpenAI, Anthropic, or OpenRouter |
| **Chrome or Edge** *(optional)* | For web research via CDP |

---

## Installation

### Option A — Install from PyPI (recommended)

```bash
pip install ciri-copilot
```

Or with `uv`:

```bash
uv pip install ciri-copilot
```

### Option B — Install from source

```bash
git clone https://github.com/adimis-ai/ciri.git
cd ciri
uv sync --dev
uv pip install -e .
```

---

## Configuration

Ciri supports multiple LLM providers. You can configure your API key in a `.env` file at your project root, or let Ciri prompt you interactively on first run.

### Multi-Provider (via LangChain — default)

```bash
# .env
LLM_GATEWAY_PROVIDER=langchain   # optional; this is the default

# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google
GOOGLE_API_KEY=...
```

### OpenRouter Gateway

OpenRouter gives you access to hundreds of models through a single API key:

```bash
LLM_GATEWAY_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
```

Ciri securely persists your key using your OS keychain (via `keyring`) — you only need to enter it once.

---

## First Run

```bash
ciri
```

On startup you will see a live progress sequence:

```
✓ Playwright browsers ready
✓ API configuration loaded
✓ Available models fetched
✓ Database initialized
✓ LLM agent ready (claude-sonnet-4-6)
✓ Workspace sync complete — 8 skills · 2 toolkits · 5 subagents

You >
```

If any step shows a warning (e.g. CDP port not open), Ciri still starts — browser-based web research will be unavailable but all other features work normally.

### Choosing a Model

On first run (or anytime via `/change-model`), Ciri shows an interactive model selector. Use `Tab` to autocomplete model names. Ciri remembers your choice across sessions.

---

## Workspace Sync

The most important first command is `/sync`. This triggers the **Trainer Agent** to:

1. **Audit** — reads your workspace files, detects existing skills, toolkits, and subagents.
2. **Analyze** — identifies capability gaps for your specific domain.
3. **Build** — creates new Skills (playbooks), Toolkits (MCP integrations), or SubAgents as needed.
4. **Write memory** — populates `.ciri/memory/AGENT.md` with a workspace index Ciri consults on every turn.

```bash
/sync
```

After `/sync`, Ciri understands your project's domain, stack, conventions, and key patterns. Re-run it whenever you add new tools or your project evolves significantly.

---

## Windows Notes

- Use PowerShell and set environment variables with `$env:OPENAI_API_KEY = "..."`.
- Install `uv` via the standalone installer or `pip install uv`.
- Ciri automatically detects **Git Bash** or **WSL bash** for shell tool execution — no manual path configuration needed.
- Playwright is automatically installed on first run.

---

## Browser Integration (Web Research)

For web research using your real browser sessions and cookies:

**Linux/macOS**:
```bash
google-chrome --remote-debugging-port=9222 &
```

**Windows**:
```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
```

Once running, restart Ciri and the CDP connection will be detected automatically. Ciri copies your browser profile to a managed directory to comply with Chrome's CDP security policy (v136+).

→ [Full Web Research Guide](features/web-research.md)

---

## Essential Commands After Installation

| Command | What It Does |
|---|---|
| `/sync` | Analyze workspace, train Ciri, hot-reload skills |
| `/new-thread` | Start a fresh conversation |
| `/change-model` | Switch your LLM (with Tab autocomplete) |
| `/help` | Show all commands and shortcuts |
| `/exit` | Exit cleanly |

→ [Full CLI Reference](cli-reference.md)

---

## Next Steps

- **[CLI Reference](cli-reference.md)** — Master the `/commands`, `@triggers`, and keyboard shortcuts.
- **[Skills Guide](skills-guide.md)** — Understand how Ciri's domain capabilities work.
- **[Core Harness](architecture/core-harness.md)** — The two-level global/project capability system.
- **[Use Cases](use-cases/index.md)** — See Ciri working across software, finance, marketing, and more.
