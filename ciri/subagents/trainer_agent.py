from typing import Any, Optional
from pathlib import Path
from langgraph.errors import GraphInterrupt
from langgraph.cache.memory import InMemoryCache
from langchain_core.language_models import BaseChatModel
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents.middleware import ToolRetryMiddleware

from .._retry_helpers import graphinterrupt_aware_failure
from ..backend import CiriBackend
from ..prompts import TRAINER_AGENT_SYSTEM_PROMPT_TEMPLATE
from ..utils import get_default_filesystem_root, get_core_harness_dir
from .web_researcher import build_web_researcher_agent, CrawlerBrowserConfig
from ..toolkit import build_script_executor_tool, follow_up_with_human
from .skill_builder import build_skill_builder_agent
from .subagent_builder import build_subagent_builder_agent
from .toolkit_builder import build_toolkit_builder_agent
from ..middlewares import (
    InjectAvailableToolNamesMiddleware,
    InjectAvailableSubAgentNamesMiddleware,
    InjectAvailableSkillNamesMiddleware,
)

CIRI_DIR_DEFAULT = get_core_harness_dir()


async def build_trainer_agent(
    model: BaseChatModel,
    backend: CiriBackend,
    all_allowed: bool = False,
    cdp_endpoint: Optional[str] = None,
    crawler_browser_config: Optional[CrawlerBrowserConfig] = None,
    web_researcher_agent: Optional[CompiledSubAgent] = None,
    ciri_dir: Optional[Path] = None,  # kept for backward compatibility
    core_harness_dir: Optional[Path] = None,  # preferred: OS-level harness dir
    workspace_memory_dir: Optional[Path] = None,  # workspace-specific .ciri/memory/
) -> CompiledSubAgent:
    # Build or reuse web researcher
    if web_researcher_agent is None:
        web_researcher_agent = await build_web_researcher_agent(
            model=model,
            all_allowed=all_allowed,
            cdp_endpoint=cdp_endpoint,
            crawler_browser_config=crawler_browser_config,
        )

    # Build builder subagents
    skill_builder = await build_skill_builder_agent(
        model=model,
        backend=backend,
        all_allowed=all_allowed,
        cdp_endpoint=cdp_endpoint,
        crawler_browser_config=crawler_browser_config,
        web_researcher_agent=web_researcher_agent,
    )

    subagent_builder = await build_subagent_builder_agent(
        model=model,
        backend=backend,
        all_allowed=all_allowed,
        cdp_endpoint=cdp_endpoint,
        crawler_browser_config=crawler_browser_config,
        web_researcher_agent=web_researcher_agent,
    )

    toolkit_builder = await build_toolkit_builder_agent(
        model=model,
        backend=backend,
        all_allowed=all_allowed,
        cdp_endpoint=cdp_endpoint,
        crawler_browser_config=crawler_browser_config,
        web_researcher_agent=web_researcher_agent,
    )

    # Resolve directories
    # core_harness_dir takes precedence; ciri_dir is the legacy param kept for compat
    resolved_core_harness_dir = core_harness_dir or ciri_dir or CIRI_DIR_DEFAULT
    resolved_workspace_memory_dir = workspace_memory_dir or (
        get_default_filesystem_root() / ".ciri" / "memory"
    )

    # Format system prompt with both directory references
    system_prompt = TRAINER_AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        core_harness_dir=resolved_core_harness_dir,
        workspace_memory_dir=resolved_workspace_memory_dir,
    )

    interrupt_on = None
    if not all_allowed:
        interrupt_on = {
            "execute": True,
            "edit_file": True,
            "write_file": True,
        }

    # Define the Trainer Agent
    trainer_agent = create_deep_agent(
        model=model,
        backend=backend,
        cache=InMemoryCache(),
        interrupt_on=interrupt_on,
        name="trainer_agent",
        subagents=[
            web_researcher_agent,
            skill_builder,
            subagent_builder,
            toolkit_builder,
        ],
        system_prompt=system_prompt,
        tools=[build_script_executor_tool(), follow_up_with_human],
        middleware=[
            InjectAvailableToolNamesMiddleware(),
            InjectAvailableSubAgentNamesMiddleware(),
            InjectAvailableSkillNamesMiddleware(),
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
        name="trainer_agent",
        runnable=trainer_agent,
        description=(
            f"Self-training orchestrator that evolves Ciri's global capabilities in "
            f"{resolved_core_harness_dir} and workspace memory in "
            f"{resolved_workspace_memory_dir}. "
            "Invoke when: user says 'train', 'learn', 'self-improve', 'sync', or "
            "'add a new capability'. Provide a high-level goal (e.g. 'Learn Terraform', "
            "'Sync with this project'). Do NOT use for one-off coding or research."
        ),
    )
