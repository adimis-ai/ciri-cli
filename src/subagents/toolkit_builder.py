from langgraph.errors import GraphInterrupt
from langgraph.cache.memory import InMemoryCache
from langchain_core.language_models import BaseChatModel
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents.middleware import ToolRetryMiddleware

from ..backend import CiriBackend
from ..prompts import PLAN_AND_RESEARCH_PROMPT
from ..utils import get_default_filesystem_root
from typing import Optional, Any, Callable
from .web_researcher import build_web_researcher_agent, CrawlerBrowserConfig
from ..toolkit import build_script_executor_tool, follow_up_with_human

TOOLKIT_BUILDER_SYSTEM_PROMPT = (
    """You are the **Lead Toolkit Engineer** for the Ciri agent. Your purpose is to create high-quality, robust **MCP (Model Context Protocol) Servers** (also known as "toolkits") that enable the agent to interact with external services and APIs.

## Core Philosophy: Capability over Complexity
You focus on creating tools that are intuitive for an AI to use, robust against errors, and provide clear structured data.

## ⚠️ CRITICAL: Mandates & Constraints
1.  **SKILL USAGE**: You **MUST** use the `mcp-builder` skill for guidance on best practices, project structure, and implementation details.
2.  **RESEARCH FIRST**: Before writing code, you **MUST** use the `web_researcher_agent` to study the target API documentation. You need to know the endpoints, authentication methods, and data models.
3.  **DIRECTORY STRUCTURE**:
    -   All toolkits must be created in `.ciri/toolkits/<toolkit-name>`.
    -   **Python (FastMCP)**:
        -   `pyproject.toml`: Must include `fastmcp`.
        -   `src/main.py`: **MANDATORY** entry point for auto-discovery.
    -   **Node/TypeScript**:
        -   `package.json`: Must include `@modelcontextprotocol/sdk`.
        -   Entry point: Must be defined in `package.json` (usually `dist/index.js` after build).
4.  **MIDDLEWARE COMPATIBILITY**:
    -   Your toolkits MUST be compatible with `ToolkitInjectionMiddleware`.
    -   This means adhering strictly to the directory structure and dependency requirements above.
5.  **NO PLACEHOLDERS**: You build working tools. If you need an API key, ask the user or use a placeholder string that is clearly marked as needing replacement (e.g., `os.getenv("API_KEY")`).

## Workflow: The Toolkit Creation Lifecycle

### Phase 1: Research & Design
-   Use `web_researcher_agent` to find API docs.
-   Plan the tools: naming (verb-noun), inputs (typed), and outputs (structured).
-   Consult `mcp-builder` skill for design patterns.

### Phase 2: Implementation
-   Create the directory: `.ciri/toolkits/<toolkit-name>`.
-   Initialize the project (Python or Node).
-   **Python**: Use `uv init`, add `fastmcp`.
-   **Node**: Use `npm init`, add `@modelcontextprotocol/sdk`.
-   Implement the server logic.
-   **CRITICAL**: Ensure `src/main.py` (Python) or the build output (Node) exists and is executable.

### Phase 3: Verification
-   **Build/Install**: Run `uv sync` or `npm install && npm run build`.
-   **Test**: Use the `script_executor_tool` to run the server in a way that verifies it starts (e.g., check for syntax errors). *Note: Full interaction testing might require the MCP Inspector, but you should at least verify it runs.*

## Tools Strategy

-   **`web_researcher_agent`**: Use for ALL internet research (API docs, libraries).
-   **`execute` (script_executor)**: Use for creating directories, running `uv`/`npm` commands, and verifying the server.
-   **`write_file` / `edit_file`**: Use for creating/modifying code files.
-   **`read_file` / `ls`**: Verify file placement.

You are the bridge between the agent and the world. Build strong bridges.
"""
    + "\n\n"
    + PLAN_AND_RESEARCH_PROMPT
)


async def build_toolkit_builder_agent(
    model: BaseChatModel,
    backend: CiriBackend,
    all_allowed: bool = False,
    browser_name: Optional[str] = None,
    profile_directory: Optional[str] = None,
    headless: Optional[bool] = None,
    crawler_browser_config: Optional[CrawlerBrowserConfig] = None,
    web_researcher_agent: Optional[CompiledSubAgent] = None,
) -> CompiledSubAgent:
    # Create the Web Researcher SubAgent (or reuse pre-built one)
    if web_researcher_agent is None:
        web_researcher_agent = await build_web_researcher_agent(
            model=model,
            headless=headless,
            all_allowed=all_allowed,
            browser_name=browser_name,
            profile_directory=profile_directory,
            crawler_browser_config=crawler_browser_config,
        )

    # Path to the mcp-builder skill
    # We use get_default_filesystem_root() to ensure we find the project root correctly
    mcp_builder_path = (
        get_default_filesystem_root() / ".ciri" / "skills" / "mcp-builder"
    )


    interrupt_on = None
    if not all_allowed:
        interrupt_on = {
            "execute": True,
            "edit_file": True,
            "write_file": True,
        }
        
        
    # Define the Toolkit Builder SubAgent
    toolkit_builder_agent = create_deep_agent(
        model=model,
        backend=backend,
        cache=InMemoryCache(),
        interrupt_on=interrupt_on,
        name="toolkit_builder_agent",
        subagents=[web_researcher_agent],
        system_prompt=TOOLKIT_BUILDER_SYSTEM_PROMPT,
        tools=[build_script_executor_tool(), follow_up_with_human],
        skills=[mcp_builder_path] if mcp_builder_path.exists() else [],
        middleware=[
            ToolRetryMiddleware(
                max_retries=2,
                retry_on=lambda exc: not isinstance(exc, GraphInterrupt),
                on_failure="continue",
                backoff_factor=2.0,
                initial_delay=1.0,
                max_delay=10.0,
                jitter=True,
            ),
        ],
    )

    return CompiledSubAgent(
        name="toolkit_builder_agent",
        runnable=toolkit_builder_agent,
        description=(
            "A specialized SubAgent for creating and managing MCP toolkits (servers).\n"
            "WHEN TO USE: Invoke this agent when the user wants to 'create a toolkit', 'build an MCP server', 'integrate an API', or 'add a new tool source' that involves creating a standalone MCP server.\n"
            "WHY: This agent knows the specific requirements for MCP servers, including `ToolkitInjectionMiddleware` compatibility, directory structure, and dependency management.\n"
            "HOW: Provide a clear task description like 'Create a GitHub toolkit with issue management tools' or 'Build a Stripe integration toolkit'.\n"
            "WHEN NOT TO USE: Do NOT use this for creating internal agent skills (use `skill_builder_agent`) or simple one-off scripts."
        ),
    )
