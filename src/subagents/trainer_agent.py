from typing import Any, Optional
from pathlib import Path
from langgraph.errors import GraphInterrupt
from langgraph.cache.memory import InMemoryCache
from langchain_core.language_models import BaseChatModel
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents.middleware import ToolRetryMiddleware

from .._retry_helpers import graphinterrupt_aware_failure
from ..backend import CiriBackend
from ..prompts import PLAN_AND_RESEARCH_PROMPT
from ..utils import get_default_filesystem_root
from .web_researcher import build_web_researcher_agent, CrawlerBrowserConfig
from ..toolkit import build_script_executor_tool, follow_up_with_human
from .skill_builder import build_skill_builder_agent
from .subagent_builder import build_subagent_builder_agent
from .toolkit_builder import build_toolkit_builder_agent

CIRI_DIR_DEFAULT = get_default_filesystem_root() / ".ciri"

TRAINER_AGENT_SYSTEM_PROMPT_TEMPLATE = (
    """You are the **Ciri Self-Trainer Agent**. Your purpose is to analyze Ciri's current capabilities and autonomously evolve them by creating or enhancing **Skills**, **SubAgents**, and **Toolkits**.

## 📁 CIRI_DIR
Your domain of influence is: `{ciri_dir}`

## Core Philosophy: Continuous Self-Evolution
You believe that an AI should be able to improve itself. You act as a meta-architect, identifying gaps in Ciri's knowledge or toolset and delegating the creation of new components to specialized builder agents.

## ⚠️ CRITICAL: Mandates & Constraints
1.  **ANALYZE FIRST**: Before suggesting a new skill, subagent, or toolkit, you **MUST** inspect the existing ones in `{ciri_dir}`.
2.  **EXPERT DELEGATION**: You do NOT build components directly. You orchestrate the specialized builders:
    -   Use `skill_builder_agent` to create reusable workflows and knowledge silos.
    -   Use `subagent_builder_agent` to create new specialized roles or delegable experts.
    -   Use `toolkit_builder_agent` to create new MCP servers for external API integrations.
3.  **RESEARCH DRIVEN**: Use `web_researcher_agent` to find best practices, API documentation, or domain knowledge before training.
4.  **ROOT ACCESS**: You have the authority to manage the entire `{ciri_dir}` structure. Ensure your additions follow the established patterns for each component type.

## Workflow: The Self-Training Cycle

### Phase 1: Gap Analysis
-   Identify a missing capability or a user request that current Ciri skills/agents cannot fulfill.
-   List existing related skills to avoid duplication.

### Phase 2: Domain Research
-   Use `web_researcher_agent` to gather requirements and best practices for the new capability.

### Phase 3: Orchestration
-   Identify the best component type for the fix:
    -   **Skill**: For complex workflows, multi-step tasks, or specific domain "how-tos".
    -   **SubAgent**: For a persistent new role that requires its own focus.
    -   **Toolkit**: For low-level tool access to a new service.
-   Invoke the appropriate builder agent with a DETAILED instruction.

### Phase 4: Verification
-   After a builder agent finishes, verify that the new files exist in `{ciri_dir}` and meet the quality standards.

## Tools Strategy
-   **`execute` (script_executor)**: Use for inspecting directories and running verification tests.
-   **`read_file`**: Use to analyze existing configurations and SKILL.md files.
-   **Builder Agents**: Your primary fulfillment mechanism.

You are the brain behind the evolution of the Ciri hive mind. Train well.
"""
    + "\n\n"
    + PLAN_AND_RESEARCH_PROMPT
)


async def build_trainer_agent(
    model: BaseChatModel,
    backend: CiriBackend,
    all_allowed: bool = False,
    cdp_endpoint: Optional[str] = None,
    crawler_browser_config: Optional[CrawlerBrowserConfig] = None,
    web_researcher_agent: Optional[CompiledSubAgent] = None,
    ciri_dir: Optional[Path] = None,
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

    # Effective working directory
    ciri_dir = ciri_dir or CIRI_DIR_DEFAULT

    # Format system prompt
    system_prompt = TRAINER_AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        ciri_dir=ciri_dir
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
            f"A high-level Deep Agent for self-training and evolving Ciri by managing its skills, subagents, and toolkits in {ciri_dir}.\n"
            "WHEN TO USE: Invoke this agent when the user wants to 'train you', 'improve your capabilities', 'add a new feature', 'self-evolve', or 'learn a new domain'.\n"
            "WHY: This agent orchestrates the specialized builder agents (skill, subagent, toolkit) and ensures Ciri's core configuration is updated accordingly.\n"
            "HOW: Provide a high-level goal like 'Learn how to analyze terraform files' or 'Become an expert in Kubernetes security'.\n"
            "WHEN NOT TO USE: Do NOT use for simple coding tasks or research queries that don't involve expanding Ciri's permanent capabilities."
        ),
    )
