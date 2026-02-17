# Architecture - Components

This page documents the major components and how they interact.

- Entry point: src.__main__:main
  - Initializes environment, loads .env, scans for skills/toolkits, and starts the interactive prompt
- Copilot core: src/copilot.py
  - Creates model clients, manages threads, orchestrates tool execution
- Controller: src/controller.py
  - Provides higher-level CLI commands and integrates with prompt_toolkit
- Database: src/db.py
  - Wraps SQLite for storing threads and metadata
- Serializers: src/serializers.py
  - Serializes LLM configuration objects (LLMConfig) and conversation state
- Skills: src/skills/* and .ciri/skills/*
  - Each skill provides metadata and handlers; discoverable via utils.list_skills
- Toolkits & Subagents: src/toolkit/* and src/subagents/*
  - Adapters and example subagents for background tasks or external integrations

Inter-process and library dependencies:
- LangGraph checks/serializers for long-running task orchestration
- LangChain and langchain-openai for model integration
- Playwright for browser automation when using browser-based toolkits
