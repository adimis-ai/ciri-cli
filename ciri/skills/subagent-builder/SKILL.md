---
name: subagent-builder
description: Guide for creating specialized SubAgents. Use when extending the agent's multi-agent system with new roles, specialized orchestration logic, or delegated task handling. Supports both dynamic YAML/JSON subagents and Python-based DeepAgents.
---

# Subagent Builder

This skill provides guidance for designing and implementing effective SubAgents within the Ciri framework.

## Overview

SubAgents are specialized agents that handle specific parts of a task, allowing the main agent to decompose complex problems into manageable sub-tasks. Ciri supports two types of SubAgents:

1.  **Dynamic SubAgents**: Defined in YAML or JSON files in `.ciri/subagents/`. Best for simple, tool-oriented roles.
2.  **Compiled SubAgents (DeepAgents)**: Implemented in Python using the `deepagents` library. Best for complex orchestration, custom logic, or recursive multi-agent flows.

---

# Design Principles

## 1. Role Specialization
Avoid creating general-purpose subagents. A subagent should have a clear, narrow purpose (e.g., "Web Researcher", "SQL Expert", "Security Auditor").

## 2. Trigger Clarity
The `description` of a subagent is its primary triggering mechanism. It must clearly state *when* it should be used and *what* it accomplishes.

## 3. Tool Sandboxing
Grant subagents only the tools they need. This reduces the search space for the LLM and improves reliability.

---

# Implementation: Dynamic SubAgents

Dynamic subagents are discovered automatically by the `SubAgentMiddleware`.

## Directory Structure
Place configuration files in: `.ciri/subagents/` (e.g., `researcher.yaml`, `auditor.json`).

## Configuration Schema
Valid fields for the YAML/JSON file:

-   `name` (string): Unique identifier.
-   `description` (string): Trigger text used by the parent agent.
-   `system_prompt` (string): The core instructions for the subagent.
-   `model` (string, optional): Specific model identifier (e.g., `openai:gpt-4o`).
-   `tools` (list of strings or "all", optional): Names of tools available to this subagent.
-   `interrupt_on` (dict, optional): Conditions for pausing execution.

### Example: `sql_analyst.yaml`
```yaml
name: sql_analyst
description: Use when the user needs to query the database or analyze SQL schemas.
system_prompt: |
  You are an expert SQL Analyst. Your goal is to write optimized queries.
  Always explain your reasoning before executing SQL.
model: "google:gemini-2.0-flash"
tools:
  - "sql_execute"
  - "sql_schema_reader"
```

---

# Implementation: Compiled SubAgents (Python)

Compiled SubAgents use `create_deep_agent` and are returned as a `CompiledSubAgent`.

## Design Pattern (Preferred)
Follow the structure used in `src/subagents/skill_builder.py` or `src/subagents/web_researcher.py`.

### 1. Define the System Prompt
Use a descriptive constant: `SUBAGENT_NAME_SYSTEM_PROMPT`.

### 2. Factory Function
Implement an `async` function to build the agent:
```python
async def build_custom_subagent(
    model: BaseChatModel,
    backend: CiriBackend,
    **kwargs
) -> CompiledSubAgent:
    # 1. Initialize dependencies (e.g., other subagents)
    # 2. Configure tools (e.g., custom scripts or built-ins)
    # 3. Create the deep agent
    agent = create_deep_agent(
        model=model,
        backend=backend,
        name="custom_subagent",
        system_prompt=SYSTEM_PROMPT,
        tools=[...],
        subagents=[...],
        skills=[...],
        middleware=[...]
    )
    # 4. Wrap and return
    return CompiledSubAgent(
        name="custom_subagent",
        runnable=agent,
        description="Detailed WHEN TO USE instructions."
    )
```

---

# Workflow: Building a SubAgent

### Phase 1: Research
1.  Study the domain/task requirements.
2.  Identify reference materials (APIs, documentation).
3.  Use `web_researcher_agent` if the domain is external.

### Phase 2: Design
1.  Draft the `system_prompt`.
2.  Choose the toolset.
3.  Decide between Dynamic (YAML) or Compiled (Python).
    *   *Rule of Thumb*: If it needs custom Python scripts or other subagents, use Compiled.

### Phase 3: Implementation
1.  **Dynamic**: Create the file in `.ciri/subagents/`.
2.  **Compiled**:
    *   Create a new file in `src/subagents/`.
    *   Implement the factory function.
    *   Ensure any custom tools or skills are correctly pathed.

### Phase 4: Verification
1.  Confirm the subagent is discovered by checking logs or testing simple prompts.
2.  Verify tool access.
3.  Test for "over-triggering" or "under-triggering".

