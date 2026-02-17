# Real-world Use Cases

This page contains several real-world scenarios, with mermaid diagrams and step-by-step flows showing how CIRI can be used in practical contexts.

Use case 1: Research assistant with browser-based crawling and summarization

- Goal: Given a topic, CIRI will browse the web (using Playwright + browser CDP), extract relevant pages, and summarize them into a brief.

Flow:

```mermaid
sequenceDiagram
  participant User
  participant CIRI_CLI
  participant CopilotGraph
  participant WebResearcher
  participant Browser
  participant Model
  User->>CIRI_CLI: Ask "Research topic X"
  CIRI_CLI->>CopilotGraph: Graph receives research command
  CopilotGraph->>WebResearcher: request crawl and extraction
  WebResearcher->>Browser: navigate & scrape pages
  Browser-->>WebResearcher: page HTML
  WebResearcher->>Model: call summarization
  Model-->>CopilotGraph: summarized output
  CopilotGraph-->>CIRI_CLI: present summary
  CIRI_CLI-->>User: show summarized brief
```

Implementation pointers:
- The web researcher agent is built in `src/copilot.py` via `build_web_researcher_agent()` and uses Playwright and the crawler browser config.
- Ensure Playwright browsers are installed (`playwright install`) and CDP is available if using the user's browser profile.

Use case 2: Local codebase assistant that modifies code with human-in-the-loop

- Goal: Make a small code change (refactor a function), run tests, and produce a PR.

```mermaid
sequenceDiagram
  participant Dev
  participant CIRI
  participant CopilotGraph
  participant SkillBuilder
  participant Tests
  Dev->>CIRI: Request "Refactor function X to improve naming"
  CIRI->>CopilotGraph: propose patch via PatchToolCallsMiddleware
  CopilotGraph->>SkillBuilder: create skill patch & open human approval
  Dev-->>CIRI: Approve patch
  CopilotGraph->>Filesystem: write_file
  CopilotGraph->>Tests: run pytest
  Tests-->>CopilotGraph: test results
  CopilotGraph-->>Dev: report test success/failure
```

Notes:
- Human-in-the-loop middleware ensures tool actions (edit_file/write_file) require approval before execution.
- PatchToolCallsMiddleware can create patches for the user to inspect.

Use case 3: Automating release packaging and publishing to PyPI

```mermaid
sequenceDiagram
  participant Maintainer
  participant CIRI
  participant BuildScript
  participant PyPI
  Maintainer->>CIRI: "Build release v1.2.0"
  CIRI->>BuildScript: run python -m build --wheel
  BuildScript-->>CIRI: wheel artifacts
  CIRI->>PyPI: twine upload (or use CI)
  PyPI-->>CIRI: release published
```

Implementation pointers:
- Packaging is handled by hatchling (pyproject.toml). For release automation consider adding a GitHub Actions workflow that runs build and twine in CI with secure secrets.

Each of these use cases can be expanded into a tutorial that walks a new contributor through the exact commands and code locations. If you want, I can autogenerate the tutorial files and add them to the docs as step-by-step guides.
