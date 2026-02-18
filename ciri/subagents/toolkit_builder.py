from pathlib import Path
from langgraph.errors import GraphInterrupt
from langgraph.cache.memory import InMemoryCache
from langchain_core.language_models import BaseChatModel
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents.middleware import ToolRetryMiddleware

from .._retry_helpers import graphinterrupt_aware_failure
from ..backend import CiriBackend
from ..prompts import BUILDER_CORE_PROMPT
from ..utils import get_default_filesystem_root, get_core_harness_dir
from typing import Optional
from .web_researcher import build_web_researcher_agent, CrawlerBrowserConfig
from ..toolkit import build_script_executor_tool, follow_up_with_human

WORKING_DIR_DEFAULT = get_core_harness_dir() / "toolkits"


TOOLKIT_BUILDER_SYSTEM_PROMPT_TEMPLATE = (
    """You are the **Toolkit Engineer** for Ciri. You create MCP (Model Context \
Protocol) servers that give Ciri tool-level access to external APIs and services.

WORKING_DIR: `{working_dir}`

WHAT IS A TOOLKIT?
A standalone MCP server that exposes tools the agent can call. Each toolkit is a
directory in WORKING_DIR with either a Python (FastMCP) or Node.js (@modelcontextprotocol/sdk)
implementation.

MANDATORY PROCESS
1. RESEARCH — Use `web_research_agent` to study the target API: endpoints, auth,
   data models. Consult the `mcp-builder` skill for design patterns.
2. BUILD — Create `{working_dir}/<toolkit-name>/` with:
   - **Python**: `pyproject.toml` (fastmcp dep) + `src/main.py` (MANDATORY entry point)
   - **Node**: `package.json` (@modelcontextprotocol/sdk dep) + built entry point
3. VERIFY — Run `uv sync` or `npm install && npm run build`. Test that the server
   starts without errors via `execute`.

DESIGN RULES
- Tools must be intuitive for an AI: clear names (verb-noun), typed inputs,
  structured outputs.
- Must be compatible with `ToolkitInjectionMiddleware` (correct directory structure).
- No placeholders — use `os.getenv("API_KEY")` for secrets, not hardcoded strings.
- If you need an API key the user hasn't provided, ask via `follow_up_with_human`.
"""
    + "\n\n"
    + BUILDER_CORE_PROMPT
)


async def build_toolkit_builder_agent(
    model: BaseChatModel,
    backend: CiriBackend,
    all_allowed: bool = False,
    cdp_endpoint: Optional[str] = None,
    crawler_browser_config: Optional[CrawlerBrowserConfig] = None,
    web_researcher_agent: Optional[CompiledSubAgent] = None,
    working_dir: Optional[Path] = None,
) -> CompiledSubAgent:
    # Create the Web Researcher SubAgent (or reuse pre-built one)
    if web_researcher_agent is None:
        web_researcher_agent = await build_web_researcher_agent(
            model=model,
            all_allowed=all_allowed,
            cdp_endpoint=cdp_endpoint or "http://localhost:9222",
            crawler_browser_config=crawler_browser_config,
        )

    # Effective working directory
    working_dir = working_dir or WORKING_DIR_DEFAULT

    # Path to the mcp-builder skill — check core harness first, fall back to project harness
    _core_mcp_builder = get_core_harness_dir() / "skills" / "mcp-builder"
    _proj_mcp_builder = (
        get_default_filesystem_root() / ".ciri" / "skills" / "mcp-builder"
    )
    mcp_builder_path = (
        _core_mcp_builder if _core_mcp_builder.exists() else _proj_mcp_builder
    )

    # Format the system prompt with the working directory
    system_prompt = TOOLKIT_BUILDER_SYSTEM_PROMPT_TEMPLATE.format(
        working_dir=working_dir
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
        system_prompt=system_prompt,
        tools=[build_script_executor_tool(), follow_up_with_human],
        skills=[mcp_builder_path] if mcp_builder_path.exists() else [],
        middleware=[
            ToolRetryMiddleware(
                max_retries=2,
                retry_on=lambda exc: not isinstance(exc, GraphInterrupt),
                on_failure=graphinterrupt_aware_failure,
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
            f"Creates MCP toolkit servers (external API integrations) in {working_dir}. "
            "Invoke when: user says 'create a toolkit', 'integrate an API', or "
            "'build an MCP server'. Provide task like 'Create a GitHub toolkit'. "
            "Do NOT use for skills or agent roles."
        ),
    )
