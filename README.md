# CIRI Copilot

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

- Interactive AI chat with streaming responses
- Thread-based conversation management (save, switch, delete threads)
- File- and skills-aware autocompletion (type `@` for paths, `@skills:` for skills)
- Model selection (choose an available OpenRouter model)
- Human-in-the-loop tool execution (approve, reject, or edit tool actions)
- Local encrypted storage for conversation data
- Extensible skills and toolkit architecture

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

### Clone the repo

Using SSH (recommended if you have SSH keys):

```bash
git clone git@github.com:adimis-ai/ciri.git
cd ciri
```

Or HTTPS:

```bash
git clone https://github.com/adimis-ai/ciri.git
cd ciri
```

### Install (global vs development)

Option 1 — Global (recommended for users):

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

Option 3 — Remote / Private Repo (One-command install):

If you want to install `ciri` on another machine without making the repo public, you can use `uv` with a GitHub Personal Access Token (PAT):

```bash
uv tool install git+https://<YOUR_PAT>@github.com/adimis-ai/ciri.git
```

---

## Configuration

### OpenRouter API key

CIRI uses OpenRouter (or compatible providers) for model access. Obtain an API key from https://openrouter.ai/ and provide it either when prompted on first run or via environment variable / `.env` file.

Set per-session (temporary):

```bash
# Linux/macOS
export OPENROUTER_API_KEY="your-api-key-here"

# Windows PowerShell
$env:OPENROUTER_API_KEY="your-api-key-here"
```

Or store in a `.env` file at the repo root (ignored by git):

```bash
echo 'OPENROUTER_API_KEY=your-api-key-here' > .env
```

**Security note:** do not commit API keys to version control.

---

## Quickstart

Start the CLI:

```bash
ciri
```

On first run you will be prompted for an API key and asked to choose a model. The chat UI supports streaming responses and inline autocompletion.

Common interactions

- Refer to a file: type `@` then a file path fragment to trigger autocomplete
- List threads: `/threads`
- New thread: `/new-thread`
- Change model: `/model [name]`
- Exit: `exit`, `quit`, or `bye`

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

## Commands reference (short)

- `/threads` — list and switch conversation threads
- `/new-thread` — create a new conversation thread
- `/delete-thread` — delete the current thread
- `/model [name]` — switch to a different model

**Keyboard shortcuts**

- `Tab` — autocomplete file paths, skills, or model names
- `Ctrl+C` — cancel current operation

---

## Developer notes

**High-level architecture**

- Entry point / CLI: `ciri` starts an interactive REPL-like chat.
- Model integration: OpenRouter client used for model calls; streaming and selection handled by runtime code.
- Tools & skills: extensible skills discovered under `.ciri/skills` (skills may include scripts, validators, and metadata).
- Storage: local encrypted conversation storage — see code for details.

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
- Conversations are stored locally and encrypted, but model requests are sent over the network to the chosen provider. Avoid sending sensitive data unless you accept the provider's terms.
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

