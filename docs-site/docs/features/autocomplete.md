# Autocomplete & Context Triggers

Ciri's terminal input supports two families of autocomplete:

1. **Slash-command autocomplete** — `/` prefix for CLI commands
2. **`@`-trigger autocomplete** — reference files, folders, skills, toolkits, or subagents

Both use Prompt Toolkit's completion engine (`CiriCompleter`) and activate on `Tab`.

---

## `@`-Trigger Quick Reference

| Trigger | Resolves To | Source |
|---|---|---|
| `@files:<path>` | Matching file paths | Workspace (respects `.gitignore`) |
| `@folders:<path>` | Matching directory paths | Workspace (respects `.gitignore`) |
| `@skills:<prefix>` | Skill names | Core harness + project harness |
| `@toolkits:<prefix>` | Toolkit names | Core harness + project harness |
| `@subagents:<prefix>` | SubAgent names | Core harness + project harness |

---

## File & Folder Triggers

### `@files:<path>`

Lists all files under the given path prefix, filtered by gitignore rules and hardcoded exclusions.

```
You > Explain @files:src/middlewares/skills.py in simple terms
You > Compare @files:docs/proposal-v1.docx with @files:docs/proposal-v2.docx
You > Find all TODOs in @files:src/
```

The completion list is capped at **50 results** for performance. Start typing to narrow it down.

### `@folders:<path>`

Lists directories — useful for giving Ciri a broad context of a module or section.

```
You > Summarize the purpose of @folders:src/subagents
You > What changed in @folders:reports/Q1-2026?
```

### Always-Excluded Paths

These are never shown in `@files:` or `@folders:` completions, regardless of `.gitignore`:

| Category | Excluded Names |
|---|---|
| Version control | `.git`, `.hg`, `.svn`, `.gitignore` |
| Python | `.venv`, `venv`, `env`, `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.tox`, `build`, `dist`, `*.egg-info` |
| Node.js | `node_modules`, `.npm`, `.next`, `.nuxt` |
| IDE/Editor | `.idea`, `.vscode`, `*.swp`, `*.swo` |
| Java/Build | `target`, `.gradle`, `.m2`, `out` |
| PHP | `vendor` |
| Ciri | `.ciri` |

Note: `.ciri` is excluded from file/folder listings but is explicitly searched for `@skills:`, `@toolkits:`, and `@subagents:` completions.

---

## Skills Trigger

### `@skills:<prefix>`

Lists discovered skill names from both the **core harness** and **project harness**.

```
You > Use @skills:pdf to extract form fields from application.pdf
You > Run @skills:competitor-research on Notion and Linear
```

Skills are discovered from:
1. `~/.local/share/ciri/skills/` (core harness — global)
2. `.ciri/skills/` (project harness — local)

Each entry shows the skill name (directory stem). Completions are prefixed with ⚡.

---

## Toolkits Trigger

### `@toolkits:<prefix>`

Lists MCP toolkits available in the current session.

```
You > Use @toolkits:github to list open PRs on our repo
You > Check @toolkits:jira for tickets assigned to me
```

Toolkits are discovered from:
1. `~/.local/share/ciri/toolkits/` (core harness)
2. `.ciri/toolkits/` (project harness)

Completions are prefixed with 🔧.

---

## SubAgents Trigger

### `@subagents:<prefix>`

Lists available subagent configuration names.

```
You > @subagents:web_researcher — find all recent GDPR enforcement actions in the EU
You > Ask @subagents:skill_builder to create a skill for our internal metrics API
```

SubAgents are discovered from:
1. `~/.local/share/ciri/subagents/*.{yaml,yml,json}` (core harness)
2. `.ciri/subagents/*.{yaml,yml,json}` (project harness)

Built-in subagents (`web_researcher`, `skill_builder`, `toolkit_builder`, `subagent_builder`, `trainer_agent`) are always available even without explicit config files. Completions are prefixed with 🤖.

---

## Slash Command Autocomplete

Type `/` and press `Tab` to see all available commands:

```
/new-thread
/switch-thread
/delete-thread
/threads
/change-model
/change-browser-profile
/sync
/help
/exit
```

Model name autocomplete works inside `/change-model`:

```
You > /change-model
Model: claude-[TAB]
  claude-sonnet-4-6
  claude-opus-4-6
  claude-haiku-4-5-20251001
```

---

## Implementation Reference

The autocomplete logic lives in two places:

**`src/utils.py`** — discovery functions:
- `list_files_with_gitignore(root, prefix)` — file listing
- `list_folders_with_gitignore(root, prefix)` — folder listing
- `list_skills(root, prefix)` — skill discovery
- `list_toolkits(root, prefix)` — toolkit discovery
- `list_subagents(root, prefix)` — subagent discovery

**`src/__main__.py`** — `CiriCompleter` class:
- Inherits from `prompt_toolkit.completion.Completer`
- `get_completions(document, complete_event)` — dispatches to the right discovery function
- Adds display prefix emoji for each trigger type
- Handles errors silently via try/except
