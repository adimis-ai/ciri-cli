# CLI Reference

CIRI provides a powerful interactive interface with specialized slash commands and autocomplete triggers. The terminal is designed to be highly interactive, providing suggestions as you type.

## Interaction Basics

- **Normal Text**: Sent directly to Ciri's core orchestration graph for processing.
- **Slash Commands**: Start with `/`. Used for administrative or configuration tasks.
- **Autocomplete Triggers**: Start with `@`. used to reference files, skills, toolkits, or subagents.
- **Contextual Autocomplete**: Press `Tab` at any time to receive suggestions based on what you've typed (commands, model names, file paths, etc.).

## Slash Commands

| Command | Description |
| :--- | :--- |
| `/new-thread` | Starts a fresh conversation thread with a clean state. |
| `/switch-thread` | Interactively select and switch to a previous thread. |
| `/delete-thread` | Remove the current thread and its checkpointed state. |
| `/threads` | List all historical conversation threads in a table. |
| `/change-model` | Change the active LLM. Supports interactive selection and autocompletion. |
| `/change-browser-profile` | Select a different browser profile for web research if multiple are available. |
| `/sync` | **Critical**: Analyzes your workspace, detects local skills, toolkits, and subagents, and applies them to the current session. |
| `/help` | Displays the help menu and keyboard shortcuts. |
| `/exit` | Gracefully closes the session. |

Triggers allow you to "inject" context into your prompt. Type the trigger followed by a colon and a few letters to see suggestions.

- `@files:<path>`: Reference specific files.
    - *Example*: "Analyze `@files:src/utils.py` for bugs"
- `@folders:<path>`: Reference entire directories.
    - *Example*: "What is the structure of `@folders:src/middlewares`?"
- `@skills:<name>`: Use a predefined Ciri skill.
    - *Example*: "Run `@skills:research_topic` on quantum computing"
- `@toolkits:<name>`: Reference an installed toolkit.
- `@subagents:<name>`: Delegate a task to a specialized subagent.

### Pro-Tip: Deep Autocomplete
Autocomplete respects your `.gitignore`. If you can't find a file via `@files:`, ensure it isn't being ignored by Git.

## Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Tab` | Trigger or cycle through autocomplete suggestions. |
| `Alt+Enter` | Insert a new line without submitting (multi-line input). |
| `Ctrl+C` | Stop a streaming response. |
| `Ctrl+C` (twice) | Exit CIRI immediately. |
| `↑ / ↓` | Navigate through your previous command history. |

## Human-in-the-Loop (HITL)

By default, Ciri requires explicit approval for destructive or external actions:
- `execute`: Running shell scripts or terminal commands.
- `edit_file`: Modifying existing source code.
- `write_file`: Creating new files.

You can **Approve**, **Edit** (modify the proposed arguments), or **Reject** any action request before it runs.

