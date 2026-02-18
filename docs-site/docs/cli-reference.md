# CLI Reference

Ciri's terminal interface is built on **Prompt Toolkit** for rich, interactive input handling. Everything you type at the `You >` prompt is either a natural language request, a slash command, or an `@`-autocomplete trigger.

---

## Slash Commands

Slash commands handle session management, configuration, and workspace operations. They are never sent to the LLM — they are processed locally.

| Command | Description |
|---|---|
| `/new-thread` | Start a fresh conversation thread with a clean state. Previous threads are preserved in the database. |
| `/switch-thread` | Interactively browse and switch to any previous thread. Press `Tab` to cycle through thread names. |
| `/delete-thread` | Permanently remove the current thread and its checkpointed state from the database. |
| `/threads` | Display a table of all historical conversation threads (ID, name, timestamp). |
| `/change-model` | Change the active LLM. Supports `Tab` autocomplete over all available models for the configured provider. |
| `/change-browser-profile` | Switch to a different browser profile for web research when multiple profiles are detected. |
| `/sync` | **The most powerful command.** Invokes the Trainer Agent to analyze your workspace, identify capability gaps, and build new Skills, Toolkits, or SubAgents. Hot-reloads everything without restart. |
| `/help` | Display the full help menu with commands, keyboard shortcuts, and `@`-trigger reference. |
| `/exit` | Gracefully close the session (flushes buffers, closes database). |

---

## `@`-Autocomplete Triggers

Autocomplete triggers inject context into your prompt. Type the trigger prefix followed by `:` and start typing — Ciri will suggest completions as you type. Press `Tab` to cycle through matches.

### File & Folder References

| Trigger | Description | Example |
|---|---|---|
| `@files:<path>` | List all files under the given path, respecting `.gitignore` | `@files:src/utils.py` |
| `@folders:<path>` | List all directories under the path, respecting `.gitignore` | `@folders:src/middlewares` |

Both triggers exclude noisy directories automatically (`.git`, `node_modules`, `.venv`, `__pycache__`, `.ciri`, build artifacts, etc.) even if not in `.gitignore`.

**Usage example:**
```
You > Analyze @files:src/copilot.py for performance bottlenecks
You > What is the structure of @folders:src/middlewares?
```

### Capability References

| Trigger | Description | Source |
|---|---|---|
| `@skills:<prefix>` | List skills from `~/.local/share/ciri/skills/` and `.ciri/skills/` | `@skills:pdf` |
| `@toolkits:<prefix>` | List toolkits from `~/.local/share/ciri/toolkits/` and `.ciri/toolkits/` | `@toolkits:slack` |
| `@subagents:<prefix>` | List subagents from `~/.local/share/ciri/subagents/` and `.ciri/subagents/` | `@subagents:web` |

**Usage example:**
```
You > Run @skills:pdf-form-filler on invoice.pdf
You > Use @toolkits:github to check my open PRs
You > Delegate this to @subagents:web_researcher
```

### Autocomplete Behavior

- Results are **limited to 50 items** for performance.
- Displayed with emoji prefixes: 📄 files · 📁 folders · ⚡ skills · 🔧 toolkits · 🤖 subagents
- Gitignore specs are parsed **recursively** — all `.gitignore` files in the tree are respected.
- Errors (malformed patterns, permission denied) are silently swallowed.

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Tab` | Trigger autocomplete or cycle through suggestions |
| `Alt+Enter` | Insert a newline without submitting (multi-line input) |
| `Ctrl+C` | Stop a currently streaming response |
| `Ctrl+C` ×2 | Exit Ciri immediately |
| `↑` / `↓` | Navigate through your input history |

---

## Human-in-the-Loop (HITL) Prompts

When Ciri wants to take a consequential action, she pauses and presents an approval prompt. This is not optional — it is wired into the `HumanInTheLoopMiddleware` at the graph level.

### Default Interrupt Triggers

| Action | When It Fires |
|---|---|
| `execute` | Any shell command or script execution |
| `edit_file` | Modifying an existing file |
| `write_file` | Creating a new file |

### Response Options

For each proposed action you can:

- **Approve** — Ciri executes as proposed.
- **Edit** — You modify the JSON arguments (e.g., change the command, path, or content) before Ciri executes.
- **Reject** — Ciri does not execute and can propose an alternative.

The HITL panel uses Rich-formatted panels with syntax-highlighted JSON arguments so you can inspect exactly what will happen before approving.

→ [Human-in-the-Loop Deep Dive](features/hitl.md)

---

## Script Execution Approval

The `execute_script` tool runs isolated Python or JavaScript in a sandboxed virtual environment. When triggered, it shows a special interrupt with:

- Syntax-highlighted script content
- Language and dependency list
- Working directory and output directory
- Timeout and cleanup settings

You approve before the sandbox is created and the script runs.

→ [Script Executor Guide](features/script-executor.md)

---

## Thread Management

All conversation threads are stored in the local SQLite database (`~/.ciri/data/ciri.db`). Each thread is a LangGraph checkpoint sequence — meaning you can resume any thread from exactly where it left off, even after restarting Ciri.

```bash
/threads          # list all threads
/new-thread       # creates a new thread, auto-named by timestamp
/switch-thread    # interactive fuzzy-select from your thread list
/delete-thread    # deletes current thread (irreversible)
```

Thread names can be changed in the interactive `/switch-thread` selector.

---

## Tips & Pro-Tricks

**Multi-line input**: Use `Alt+Enter` to write multi-paragraph prompts without accidentally submitting.

**Model switching mid-session**: `/change-model` changes the LLM for all *future* messages in the thread. Past messages remain associated with the previous model.

**gitignore-aware file search**: If `@files:` isn't finding a file, check your `.gitignore` — it may be excluded. Build artifacts, `.venv`, and node_modules are always excluded.

**Inject a file directly**: You can reference a file in your message and Ciri will read it:
```
You > Read @files:docs/ARCHITECTURE.md and summarize the key decisions
```

**Fast subagent delegation**: If you know what subagent handles a task, reference it explicitly to skip the routing decision:
```
You > @subagents:web_researcher — research Q1 2026 SaaS pricing benchmarks
```
