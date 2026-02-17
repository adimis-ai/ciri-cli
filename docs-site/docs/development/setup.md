# Development Setup

This page covers a reproducible development environment.

1. Install OS prerequisites
   - Git, Python 3.12+, and uv (https://astral.sh/uv/) on macOS/Linux

2. Clone and install deps

   git clone git@github.com:YOUR_ORG/ciri.git
   cd ciri
   uv sync --dev

3. Install editable package

   uv pip install -e .

4. Virtual env alternative (venv)

   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e .[dev]

5. Environment variables
- Create a .env file with provider keys (OpenRouter example):
  OPENROUTER_API_KEY=your-key
- Other variables may include: CIRI_DATA_DIR, CIRI_LOG_LEVEL

6. Browser automation
- If you need Playwright, install browsers:
  playwright install

7. Editor integration
- Recommended: enable ruff/black/mypy in your editor or run them via pre-commit
