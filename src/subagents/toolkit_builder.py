from langgraph.errors import GraphInterrupt
from langgraph.cache.memory import InMemoryCache
from langchain_core.language_models import BaseChatModel
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents.middleware import ToolRetryMiddleware

from ..backend import CiriBackend
from ..prompts import PLAN_AND_RESEARCH_PROMPT
from ..utils import get_default_filesystem_root
from ..toolkit.web_crawler_tool import BrowserConfig
from .web_researcher import build_web_researcher_agent
from ..toolkit import build_script_executor_tool, follow_up_with_human

TOOLKIT_BUILDER_SYSTEM_PROMPT = """""" + "\n\n" + PLAN_AND_RESEARCH_PROMPT


async def build_toolkit_builder_agent(
    model: BaseChatModel,
    backend: CiriBackend,
    *,
    headless: bool | None = None,
    browser_name: str | None = None,
    profile_directory: str | None = None,
    crawler_browser_config: BrowserConfig | None = None,
) -> CompiledSubAgent:
    # Create the Web Researcher SubAgent
    web_researcher_agent = await build_web_researcher_agent(
        model=model,
        headless=headless,
        browser_name=browser_name,
        profile_directory=profile_directory,
        crawler_browser_config=crawler_browser_config,
    )

    # Path to the mcp-builder skill
    # We use get_default_filesystem_root() to ensure we find the project root correctly
    mcp_builder_path = (
        get_default_filesystem_root() / ".ciri" / "skills" / "mcp-builder"
    )

    # Define the Toolkit Builder SubAgent
    toolkit_builder_agent = create_deep_agent(
        model=model,
        backend=backend,
        cache=InMemoryCache(),
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
        description=(),
    )
