# Getting Started

This guide will help you set up a local development environment for CIRI.

## Prerequisites

- **Python 3.12+**
- **Git**
- **uv** (https://docs.astral.sh/uv/) — the recommended fast package manager for this repo

## Installation (Linux/macOS)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adimis-ai/ciri.git
   cd ciri
   ```

2. **Sync dependencies**:
   ```bash
   uv sync --dev
   ```

3. **Install the CLI in editable mode**:
   ```bash
   uv pip install -e .
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the project root. At minimum, you need an API key for your chosen provider.

   **If using OpenRouter (Default):**
   ```bash
   OPENROUTER_API_KEY=your_key_here
   ```

   **If using a direct provider (e.g., Anthropic via LangChain):**
   ```bash
   LLM_GATEWAY_PROVIDER=langchain
   ANTHROPIC_API_KEY=your_key_here
   ```

5. **Run CIRI**:
   ```bash
   ciri
   ```

## Windows Notes
- Use PowerShell and set environment variables via `$env:VARIABLE_NAME = "value"`.
- Install `uv` using the standalone installer or `pip`.
- If you encounter browser errors, see the [Web Research guide](features/web-research.md).

## Next Steps
- Run `/sync` inside the CLI to let Ciri analyze your workspace.
- Check out the [Skills Guide](skills-guide.md) to add new capabilities.
