# Getting Started

This quick start will get you a local development environment for CIRI.

Prerequisites:
- Python 3.12+
- Git
- uv (https://docs.astral.sh/uv/) — package manager used in this repo

Install (Linux/macOS):

1. Clone the repo:

   git clone git@github.com:YOUR_ORG/ciri.git
   cd ciri

2. Install dependencies including dev deps:

   uv sync --dev

3. Install in editable mode:

   uv pip install -e .

4. Create a .env file with your model provider key(s), for example OpenRouter:

   echo 'OPENROUTER_API_KEY=your-key' > .env

5. Run the CLI:

   ciri

Windows notes:
- Use PowerShell and set environment variables via $env:OPENROUTER_API_KEY
- Install uv per README instructions

Next steps: read the Development section to learn how to run tests, linters, and create skills.
