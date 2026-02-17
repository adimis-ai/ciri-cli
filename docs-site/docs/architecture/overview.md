# Architecture Overview

CIRI is a single-package Python CLI application designed to be extensible with "skills", "toolkits", and "subagents". Key components:

- src/: main codebase and package
  - __main__.py: CLI entrypoint and top-level command loop
  - copilot.py: core copilot behavior and orchestration
  - controller.py: command parsing and controller logic
  - utils.py: helper functions (config, filesystem scanning, skill discovery)
  - skills/: built-in skills and examples
  - toolkit/: toolkit adapters
  - subagents/: long-running or delegated agents
- .ciri/: local user settings and installed skills/toolkits/subagents
  - .ciri/settings.json: example settings
  - .ciri/skills/: user-installed skills

The application relies on LangChain/LangGraph for LLM tooling and state
management. Local encrypted storage (SQLite) is used for threads and checkpoints.

Design goals:
- Local-first, privacy-aware operation
- Extensible via pluggable skills and toolkits
- CLI-first UX with streaming responses and structured tool execution

See the components page for more details about each module.
