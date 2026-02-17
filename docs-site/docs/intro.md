# CIRI Copilot Documentation

Welcome to the official CIRI Copilot documentation. 

**Ciri** is an autonomous, self-evolving AI copilot that lives inside your workspace. Unlike traditional static tools, Ciri can permanently expand its own capabilities by creating new Skills, Toolkits, and SubAgents as it learns about your environment.

## Quick architecture map

```mermaid
flowchart TD
  User([User Terminal]) <--> CLI[CIRI CLI]
  CLI <--> Graph[Copilot Graph]
  Graph <--> LLM[LLM Providers]
  Graph <--> Tools[Skills & Toolkits]
  Graph <--> Sub[Subagents]
  Graph <--> DB[(Local DB)]
  
  subgraph Self-Evolution
    Trainer[Trainer Agent] --> Builders{Builders}
    Builders --> |Creates| Tools
    Builders --> |Creates| Sub
  end
  Graph <--> Trainer
```

Use the sidebar to navigate topics. For code analysis, use `@files:` or `@folders:` followed by the path.

## Helpful Links
- **[Getting Started](getting-started.md)**: Quick installation and first run.
- **[CLI Reference](cli-reference.md)**: Master the slash commands.
- **[Skills Guide](skills-guide.md)**: Learn how to extend Ciri.
- **[Architecture](architecture/overview.md)**: Deep dive into the internals.


