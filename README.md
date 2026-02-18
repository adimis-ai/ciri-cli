# CIRI Copilot

[![PyPI version](https://img.shields.io/pypi/v/ciri.svg)](https://pypi.org/project/ciri/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![uv](https://img.shields.io/badge/built%20with-uv-blueviolet)](https://docs.astral.sh/uv/)

**CIRI (Contextual Intelligent Runtime Interface) Copilot** — a local, desktop-class AI copilot that runs as a command-line interface (CLI). It provides interactive chat with AI models, thread-based conversation management, file- and skill-aware autocompletion, and an extensible skills/toolkit system.

This README is intentionally neutral and written for both developers and non-developers: what the project does, how to get started, how to configure it, and key implementation notes and limitations.

---

## Table of Contents

- [What CIRI is (brief)](#what-ciri-is-brief)
- [Features](#features)
- [Who should use it](#who-should-use-it)
- [Prerequisites](#prerequisites)
  - [Windows](#windows)
  - [macOS](#macos)
  - [Linux](#linux)
- [Installation](#installation)
  - [Clone the repo](#clone-the-repo)
  - [Install (global vs development)](#install-global-vs-development)
- [Configuration](#configuration)
  - [OpenRouter API key](#openrouter-api-key)
- [Quickstart](#quickstart)
- [Commands reference (short)](#commands-reference-short)
- [Developer notes](#developer-notes)
- [Limitations & privacy](#limitations--privacy)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## What CIRI is (brief)

CIRI is a local CLI application that helps users interact with AI models and tools from their terminal. It uses OpenRouter (or compatible providers) for model access and aims to balance interactivity, local storage, and extensibility via "skills" and toolkits.

## Features

- **Interactive AI Chat**: Streaming responses with rich terminal formatting.
- **Multi-Provider Support**: Seamless integration with OpenRouter or direct providers (Anthropic, OpenAI, Google, etc.) via LangChain.
- **Multimodal Content**: Support for images, audio, and documents (PDF, CSV, etc.) in conversation.
- **Thread-Based Management**: Save, switch, and delete conversation threads locally.
- **Deep Contextual Autocompletion**: High-performance autocompletion for `@files:`, `@folders:`, `@skills:`, `@toolkits:`, and `@subagents:`.
- **Self-Evolution**: Ciri can analyze its workspace and register new skills, toolkits, and subagents on the fly.
- **Human-in-the-Loop (HITL)**: Approve, reject, or edit tool actions (shell commands, file edits) before they execute.
- **Local Storage**: Checkpoint and conversation history stored in a local SQLite database.
- **Extensible Architecture**: Easily add new skills and toolkits.

## Who should use it

- Non-developers: a lightweight, local AI chat assistant accessible from the terminal.
- Developers: a base to extend with new skills, integrate tools, or customize model usage.

## Prerequisites

Minimum recommended: **Python 3.12+** (project developed and tested on 3.12). Adjust or test if you need earlier versions.

### Windows

- Git
- Python 3.12+ (check "Add Python to PATH" during install)
- uv (https://docs.astral.sh/uv/)

PowerShell (install uv):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS

- Git (Xcode Command Line Tools or Homebrew)
- Python 3.12+ (Homebrew: `brew install python@3.12`)

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Linux (Ubuntu/Debian example)

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install git -y
# If Python 3.12 is not available, consider using the deadsnakes PPA on Ubuntu:
# sudo add-apt-repository ppa:deadsnakes/ppa -y
# sudo apt update
# sudo apt install python3.12 python3.12-venv python3.12-dev -y

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Installation

### From PyPI (recommended)

```bash
pip install ciri
```

Or with [uv](https://docs.astral.sh/uv/) (faster, isolated):

```bash
uv tool install ciri
```

After install, the `ciri` command is available globally.

### From source

Clone the repo:

```bash
git clone https://github.com/adimis-ai/ciri.git
cd ciri
```

Option 1 — Global install from source (recommended for users):

```bash
uv tool install .
```

This places the `ciri` command into your user bin (commonly `~/.local/bin`) and isolates dependencies.

Option 2 — Development / editable (recommended for contributors):

```bash
# create and sync virtual environment with uv
uv sync

# install package in editable mode
uv pip install -e .
```

---

## Configuration

### API Keys

CIRI supports multiple providers. By default, it uses **OpenRouter**, but you can use any provider supported by LangChain (OpenAI, Anthropic, Google, Mistral, etc.).

- **Interactive Setup**: If an API key is missing for your chosen model, CIRI will prompt you to enter it on startup and offer to persist it globally in `~/.ciri/.env` and your shell profile.
- **Environment Variables**: You can also set them manually:
  ```bash
  export OPENROUTER_API_KEY="your-key"
  export ANTHROPIC_API_KEY="your-key"
  # etc.
  ```

### Model Gateway

You can switch between `langchain` (default) and `openrouter` gateways via the `LLM_GATEWAY_PROVIDER` variable.

```bash
export LLM_GATEWAY_PROVIDER="langchain" # Supports provider:model format
```

**Security note:** do not commit API keys to version control.

---

## Quickstart

Start the CLI:

```bash
ciri
```

On first run, you will be guided through model and browser profile selection.

### Common interactions

- **Reference Files**: Type `@files:` then a path fragment.
- **Reference Folders**: Type `@folders:` then a path fragment.
- **Use Skills**: Type `@skills:` to see available local skills.
- **Sync Workspace**: Run `/sync` to let Ciri discover your local setup.
- **Change Model**: Run `/change-model` to switch AI providers/models.
- **Manage Threads**: Use `/threads` to list or `/new-thread` to start fresh.

Example session

```text
You> Hello, analyze the @src/__main__.py file
CIRI> [analysis about the file]

You> /threads
# shows list of threads

You> exit
Goodbye!
```

---

## Commands Reference

| Command | Description |
| :--- | :--- |
| `/threads` | List all conversation threads. |
| `/switch-thread` | Interactively switch to another thread. |
| `/new-thread` | Start a new conversation thread. |
| `/delete-thread` | Delete the current thread history. |
| `/change-model` | Change the active LLM model. |
| `/change-browser-profile` | Switch browser profiles for research. |
| `/sync` | Analyze workspace & register skills/subagents. |
| `/help` | Show the help menu. |
| `/exit` | Exit the CLI. |

**Keyboard shortcuts**

- `Tab` — autocomplete file paths, skills, or model names
- `Ctrl+C` — cancel current operation

---

## Developer notes

**High-level architecture**

- Entry point / CLI: `ciri` starts an interactive REPL-like chat.
- Core Logic: `CopilotController` manages threads and executes the agent graph (supports multimodal inputs).
- Model integration: OpenRouter client used for model calls; streaming and selection handled by runtime code.
- Tools & skills: extensible skills discovered under `.ciri/skills` (skills may include scripts, validators, and metadata).
- Storage: local conversation storage — see code for details.

**Key locations**

- `src/` — main package and CLI entry points (look for `__main__.py` or CLI module)
- `.ciri/skills` — bundled skills and examples
- `tests/` — test suite
- `pyproject.toml` — project metadata and dependencies

**Extending**

- Follow patterns used in `.ciri/skills` to add new skills
- Document inputs/outputs and add tests under `tests/`

**Development tips**

- Use `uv sync` to prepare the development environment
- Install editable: `uv pip install -e .`

---

## Limitations & privacy

- CIRI relies on third-party model providers (OpenRouter). Provider policies, costs, and behavior apply.
- Conversations are stored locally, but model requests are sent over the network to the chosen provider. Avoid sending sensitive data unless you accept the provider's terms.
- Offline use requires configuring or running compatible local models — not provided by default.

---

## Troubleshooting

**Command `ciri` not found**

Cause: user bin (e.g., `~/.local/bin`) not in `PATH`.

Fix (Linux/macOS):

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.profile
# or add to ~/.bashrc or ~/.zshrc and restart the shell
```

**Python version error**

Cause: Python < 3.12 installed. Install Python 3.12+ and ensure `uv` or your environment uses it.

**API key errors**

Cause: invalid or missing OpenRouter API key. Verify at https://openrouter.ai/keys and re-enter when prompted. Remove saved key files if needed (e.g., `~/.ciri/.env`).

**Permission denied when writing data**

Cause: incorrect ownership of `~/.ciri` or other data directories.

Fix (Linux):

```bash
sudo chown -R "$USER":"$USER" ~/.ciri
```

---

## Contributing

Contributions welcome. See `CONTRIBUTING.md`.

Suggested flow:

1. Fork and create a branch
2. Run tests and add tests for new behavior
3. Open a PR with a clear description

---

## License

MIT — see `LICENSE.md`.

---

## Contact

Aditya Mishra — https://github.com/adimis-ai

Project: https://github.com/adimis-ai/ciri

