from pathlib import Path
from langgraph.errors import GraphInterrupt
from langgraph.cache.memory import InMemoryCache
from langchain_core.language_models import BaseChatModel
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents.middleware import ToolRetryMiddleware

from .._retry_helpers import graphinterrupt_aware_failure
from ..backend import CiriBackend
from ..prompts import SUBAGENT_BUILDER_SYSTEM_PROMPT_TEMPLATE
from ..utils import get_default_filesystem_root, get_core_harness_dir
from typing import Optional
from .web_researcher import build_web_researcher_agent, CrawlerBrowserConfig
from ..toolkit import build_script_executor_tool, follow_up_with_human
from ..middlewares import (
    InjectAvailableToolNamesMiddleware,
    InjectAvailableSubAgentNamesMiddleware,
    InjectAvailableSkillNamesMiddleware,
)

WORKING_DIR_DEFAULT = get_core_harness_dir() / "subagents"


async def build_subagent_builder_agent(
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

    # Skill paths — check core harness first, fall back to project harness
    _core_skills = get_core_harness_dir() / "skills"
    _proj_skills = get_default_filesystem_root() / ".ciri" / "skills"

    _core_subagent_builder = _core_skills / "subagent-builder"
    _proj_subagent_builder = _proj_skills / "subagent-builder"
    subagent_builder_path = (
        _core_subagent_builder
        if _core_subagent_builder.exists()
        else _proj_subagent_builder
    )

    # Path to the skill-creator skill (often useful when building agents that use skills)
    _core_skill_creator = _core_skills / "skill-creator"
    _proj_skill_creator = _proj_skills / "skill-creator"
    skill_creator_path = (
        _core_skill_creator if _core_skill_creator.exists() else _proj_skill_creator
    )

    # Format the system prompt with the working directory
    system_prompt = SUBAGENT_BUILDER_SYSTEM_PROMPT_TEMPLATE.format(
        working_dir=working_dir
    )

    interrupt_on = None
    if not all_allowed:
        interrupt_on = {
            "execute": True,
            "edit_file": True,
            "write_file": True,
        }

    # Define the SubAgent Builder SubAgent
    subagent_builder_agent = create_deep_agent(
        model=model,
        backend=backend,
        cache=InMemoryCache(),
        interrupt_on=interrupt_on,
        name="subagent_builder_agent",
        subagents=[web_researcher_agent],
        system_prompt=system_prompt,
        tools=[build_script_executor_tool(), follow_up_with_human],
        skills=[p for p in [subagent_builder_path, skill_creator_path] if p.exists()],
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
        name="subagent_builder_agent",
        runnable=subagent_builder_agent,
        description=(
            f"Designs and implements specialized SubAgent roles in {working_dir}. "
            "Invoke when: user says 'create an agent', 'add a specialist role', or "
            "'extend the multi-agent system'. Provide task like 'Create a security "
            "auditor agent'. Do NOT use for general coding."
        ),
    )
