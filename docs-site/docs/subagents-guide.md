# Subagents Guide

Subagents are specialized agents that run in isolated contexts for background or long-running tasks.

- Implement subagents under src/subagents/ or .ciri/subagents/
- Keep subagents stateless where possible and persist checkpoints to SQLite
- Use langgraph or langchain orchestration patterns for task scheduling

Testing subagents:
- Unit-test controller logic and use integration tests for interactions with langgraph
- Use smaller timeouts for CI and mock external dependencies
