# Graph Execution & Agent Boot

This page traces exactly how Ciri is constructed and executed at runtime — from the moment `ciri` is run in the terminal to the first token streaming back. Reference this when adding middlewares, subagents, or custom tools.

---

## Boot Sequence

```mermaid
sequenceDiagram
  autonumber
  participant Shell as Terminal
  participant Main as src/__main__.py
  participant Copilot as src/copilot.py
  participant LLM as LangChain Model
  participant Browser as Browser/CDP
  participant Builders as Subagent Builders
  participant MW as Middleware Stack
  participant LG as LangGraph

  Shell->>Main: ciri (CLI entrypoint)
  Main->>Main: ensure_playwright_installed()
  Main->>Main: load API config from env
  Main->>Copilot: create_copilot(config)
  Copilot->>LLM: llm_config.init_langchain_model()
  LLM-->>Copilot: model instance
  Copilot->>Browser: resolve_browser_profile() → launch_browser_with_cdp()
  Browser-->>Copilot: cdp_url (or None if no browser)
  Copilot->>Builders: build_web_researcher_agent(model, cdp_url)
  Builders-->>Copilot: web_researcher
  Copilot->>Builders: build_skill_builder_agent(web_researcher)
  Builders-->>Copilot: skill_builder
  Copilot->>Builders: build_toolkit_builder_agent(web_researcher)
  Builders-->>Copilot: toolkit_builder
  Copilot->>Builders: build_subagent_builder_agent(web_researcher)
  Builders-->>Copilot: subagent_builder
  Copilot->>Builders: build_trainer_agent(all_builders)
  Builders-->>Copilot: trainer_agent
  Copilot->>MW: assemble middleware stack (14 layers)
  MW-->>Copilot: configured stack
  Copilot->>LG: compile state graph
  LG-->>Copilot: CompiledStateGraph
  Copilot-->>Main: graph
  Main->>Main: wrap with CiriController
  Main->>Main: start REPL (prompt_toolkit session)
```

---

## Key Entry Points

| Function | File | Purpose |
|---|---|---|
| `main()` | `src/__main__.py` | CLI entry, startup sequence, REPL loop |
| `create_copilot()` | `src/copilot.py` | Assembles the full LangGraph graph |
| `CiriController.stream()` | `src/controller.py` | Streams events from graph for a given input |
| `CiriController.get_state()` | `src/controller.py` | Returns current graph checkpoint state |
| `llm_config.init_langchain_model()` | `src/llm_config.py` | Initializes LangChain model from config |
| `ensure_playwright_installed()` | `src/__main__.py` | Runs `playwright install chromium` silently |

---

## LangGraph State Graph

Ciri uses a `StateGraph` compiled with:

- **State schema** — typed state dict carrying messages, checkpoints, and metadata
- **Nodes** — the main Ciri agent node + tool execution nodes
- **Edges** — conditional routing based on whether the LLM called a tool or finished
- **Checkpointers** — `AsyncSqliteSaver` backed by `~/.ciri/data/ciri.db` for persistence across sessions

The graph runs in **streaming mode** using `astream()` with two streaming modes:

```python
stream_modes = ["updates", "messages"]
```

- `"updates"` — emits node state after each graph step; carries interrupt signals
- `"messages"` — emits `AIMessageChunk` objects for token-by-token streaming

---

## Middleware Wrapping

Each middleware wraps the LLM call via `wrap_model_call`. The stack processes requests inward and responses outward:

```
User input
  → InjectNamesMiddleware     (inject tool/skill/subagent registries)
  → MemoryMiddleware          (inject memory files)
  → SkillsMiddleware          (inject active skill context)
  → SubAgentMiddleware        (inject subagent routing logic)
  → ToolkitInjectionMiddleware (connect MCP servers)
  → HumanInTheLoopMiddleware  (intercept for approval)
  → TodoListMiddleware        (inject task state)
  → LLM call
  → ToolRetryMiddleware       (retry failed tools, skip GraphInterrupt)
  → PatchToolCallsMiddleware  (normalize tool call format)
  → Response
```

Middlewares are assembled in `create_copilot()` via `MiddlewareBuilder`. The order is significant — middlewares added first wrap the outermost layer.

---

## Interrupt Handling

When `HumanInTheLoopMiddleware` triggers, it raises `GraphInterrupt`. This exception surfaces through `astream()` as a special event type:

```python
async for event in graph.astream(input, config, stream_mode=["updates", "messages"]):
    if isinstance(event, GraphInterrupt):
        # Surface approval prompt to user
        handle_interrupts(event)
```

After the user responds (approve/edit/reject), execution resumes with:

```python
await graph.astream(Command(resume=decision), config)
```

`ToolRetryMiddleware` is configured with `retry_on=lambda exc: not isinstance(exc, GraphInterrupt)` to prevent retrying approval prompts.

---

## Thread State and Persistence

Each conversation thread maps to a LangGraph **config** with a unique `thread_id`:

```python
config = {"configurable": {"thread_id": thread_id}}
```

LangGraph's `AsyncSqliteSaver` checkpointer writes the full state (messages + metadata) to `~/.ciri/data/ciri.db` after each graph step. Resuming a thread replays from the latest checkpoint.

The `threads` table in the same database stores human-readable metadata (name, created_at, last_updated) alongside the LangGraph checkpoint data.

---

## Streaming Architecture in the CLI

```mermaid
sequenceDiagram
  participant REPL as REPL (main thread)
  participant Ctrl as CiriController
  participant Graph as LangGraph graph
  participant Rich as Rich Console

  REPL->>Ctrl: stream(input, thread_id)
  Ctrl->>Graph: astream(input, config, stream_mode=["updates","messages"])
  loop Token streaming
    Graph-->>Ctrl: AIMessageChunk
    Ctrl-->>REPL: yield chunk
    REPL->>Rich: print chunk (live update)
  end
  loop Node updates
    Graph-->>Ctrl: node state update
    Ctrl-->>REPL: yield update
    REPL->>REPL: check for GraphInterrupt
  end
  Graph-->>Ctrl: stream complete
  Ctrl-->>REPL: done
```

The REPL runs a synchronous loop calling `controller.stream()`. `CiriController.stream()` runs the async `astream()` in the event loop and yields events synchronously using `asyncio.run_coroutine_threadsafe()` or equivalent.

---

## Adding a New Tool

1. Define the tool function with LangChain `@tool` decorator:

   ```python
   from langchain.tools import tool

   @tool
   def my_new_tool(input: str) -> str:
       """One-sentence description of what this tool does."""
       return f"Result: {input}"
   ```

2. Register it in `src/agent.py` inside `_create_ciri()`:

   ```python
   tools = [
       ...existing tools...,
       my_new_tool,
   ]
   ```

3. If the tool needs HITL approval, add it to `interrupt_on`:

   ```python
   interrupt_on = {"execute": True, "edit_file": True, "write_file": True, "my_new_tool": True}
   ```

4. The `InjectAvailableToolNamesMiddleware` will automatically include it in the registry injection on the next run.

---

## Adding a New Subagent

See the full [Subagents Guide](../subagents-guide.md). At the graph level:

1. Implement the builder function (e.g., `build_my_subagent()`) in `src/subagents/my_subagent.py`
2. Register in `create_copilot()` alongside the other builder calls
3. Pass it to `SubAgentMiddleware` so Ciri can delegate to it
4. `InjectAvailableSubAgentNamesMiddleware` will automatically include it in the name injection

---

## Practical Debugging Tips

**Trace graph compilation:**
```python
# In a test or scratch script
import asyncio
from src.copilot import create_copilot

graph = asyncio.run(create_copilot(...))
print(graph.get_graph().draw_ascii())
```

**Inspect checkpoint state:**
```python
from src.controller import CiriController

ctrl = CiriController(graph)
state = asyncio.run(ctrl.get_state(thread_id="my-thread"))
print(state.values)
```

**Integration test pattern:**
```python
import asyncio
from src.copilot import create_copilot
from src.controller import CiriController

async def test_basic_flow():
    graph = await create_copilot(model="claude-haiku-4-5-20251001")
    ctrl = CiriController(graph)
    thread_id = "test-thread"

    results = []
    async for event in ctrl.astream("Hello", thread_id):
        results.append(event)

    assert any("Hello" in str(r) for r in results)
```
