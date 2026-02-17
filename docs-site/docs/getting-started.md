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
   While CIRI can prompt you for missing API keys interactively, you can also set them in a `.env` file at the project root for convenience.

   **Standard Configuration (Multi-Provider via LangChain):**
   CIRI uses `langchain` as the default gateway, supporting OpenAI, Anthropic, Google, and more.
   ```bash
   # Set the gateway (optional, defaults to langchain)
   LLM_GATEWAY_PROVIDER=langchain

   # Set provider-specific keys
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   ```

   **Using OpenRouter Gateway:**
   ```bash
   LLM_GATEWAY_PROVIDER=openrouter
   OPENROUTER_API_KEY=your_key_here
   ```

5. **Run CIRI**:
   ```bash
   ciri
   ```
   On first run, CIRI will walk you through choosing a model and setting up your browser profile.

## Windows Notes
- Use PowerShell and set environment variables via `$env:VARIABLE_NAME = "value"`.
- Install `uv` using the standalone installer or `pip`.
- If you encounter browser errors, see the [Web Research guide](features/web-research.md).

## Next Steps
- Run `/sync` inside the CLI to let Ciri analyze your workspace.
- Check out the [Skills Guide](skills-guide.md) to add new capabilities.
