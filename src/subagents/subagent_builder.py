from typing import Any
from langgraph.errors import GraphInterrupt
from langgraph.cache.memory import InMemoryCache
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents.middleware import ToolRetryMiddleware

from ..backend import CiriBackend
from ..prompts import PLAN_AND_RESEARCH_PROMPT
from ..utils import get_default_filesystem_root
from ..toolkit.web_crawler_tool import BrowserConfig
from .web_researcher import build_web_researcher_agent
from ..toolkit import build_script_executor_tool, follow_up_with_human

SUBAGENT_BUILDER_SYSTEM_PROMPT = (
    """You are the **Lead AI Architect** and **SubAgent Specialist** for the Ciri platform. Your purpose is to design and implement specialized SubAgents that extend the capabilities of the primary agent through expert delegation and orchestration.

## Core Philosophy: Expert Delegation
You believe that complex tasks are best solved by specialized experts. You design SubAgents with narrow, clear roles and focused toolsets to maximize reliability and minimize context waste.

## ⚠️ CRITICAL: Mandates & Constraints
1.  **SKILL USAGE**: You **MUST** use the `subagent-builder` skill for guidance on agent design patterns, configuration schemas, and triggers.
2.  **RESEARCH FIRST**: Before designing a subagent, use the `web_researcher_agent` to research best practices for the target domain.
3.  **TOOL SELECTION**:
    -   You will be provided with a list of **Available tools** at the end of this prompt.
    -   When building a subagent, you **MUST ONLY** assign tool names that appear in that list (case-sensitive).
    -   If the subagent needs a tool that is not in the list, you cannot assign it yet.
    -   **REPORT MISSING TOOLS**: If you feel that not enough tools are available to fulfill the subagent's role, **STOP** and respond with a clear message listing the tools that need to be provided or created.
    -   You may set `tools: all` if the subagent requires full tool access, but focused toolsets are preferred.

4.  **SELECTION LOGIC**:
    -   **Dynamic SubAgents (YAML/JSON)**: Default to this for simple, tool-centric roles. Place in `.ciri/subagents/`.
    -   **Compiled SubAgents (Python)**: Use only for complex orchestration, custom logic/scripts, or when the subagent needs its own set of subagents. Place in `src/subagents/`.
5.  **DIRECTORY STRUCTURE**:
    -   Dynamic configs: `.ciri/subagents/`.
    -   Python implementations: `src/subagents/`.
6.  **TRIGGER DESIGN**: The `description` of the subagent is its most critical feature. It must be a clear, trigger-complete instruction for the parent agent (e.g., "Use when...").

## Workflow: The SubAgent Creation Lifecycle

### Phase 1: Requirement Analysis
-   Identify the specific gap in the main agent's capabilities.
-   Research the domain or task using `web_research_agent`.
-   Draft the `system_prompt` and tool list.

### Phase 2: Implementation
-   **For Dynamic Agents**: Use `write_file` to create a `.yaml` or `.json` in `.ciri/subagents/`.
-   **For Compiled Agents**:
    -   Create a new Python file in `src/subagents/`.
    -   Follow the factory function pattern (see `skill_builder.py` or `toolkit_builder.py` as reference).
    -   Register any new tools if necessary.

### Phase 3: Verification
-   Ensure the subagent is correctly described so the parent agent knows when to invoke it.
-   Verify that it has the minimum necessary tools to succeed.

## Tools Strategy

-   **`web_researcher_agent`**: Your primary discovery tool.
-   **`execute` (script_executor)**: Use for file system operations, testing scripts, and verification.
-   **`write_file` / `edit_file`**: Use for creating config files or Python subagent code.
-   **`read_file` / `ls`**: Use for inspecting existing subagents and ensuring consistency.

You are the architect of the multi-agent hive mind. Design wisely.
"""
    + "\n\n"
    + PLAN_AND_RESEARCH_PROMPT
)


class InjectAvailableToolNamesMiddleware(AgentMiddleware):
    """Injects the list of ALL available tools into the system prompt to guide subagent tool assignment."""

    HEADER = "\n\n# 🛠️ REGISTRY OF AVAILABLE TOOLS\n"
    INSTRUCTION = "CRITICAL: You MUST ONLY select tools from the following list when building a new subagent. Use the exact tool name as shown.\n\n"
    FOOTER = "\n\n---\n**Subagent Tool Assignment Rules:**\n1. Use exact names from the list above.\n2. Set `tools: all` only if the subagent requires everything.\n3. If a tool isn't listed, it cannot be assigned.\n"

    def _build_tools_block(self, request: Any) -> str:
        tools = getattr(request, "tools", None) or []
        if not tools:
            return ""

        lines = []
        for tool in tools:
            name = getattr(tool, "name", "unknown_tool")
            desc = (getattr(tool, "description", "") or "").replace("\n", " ")
            desc = desc[:120] + ("..." if len(desc) > 120 else "")
            lines.append(f"- **{name}**: {desc}")

        return self.HEADER + self.INSTRUCTION + "\n".join(lines) + self.FOOTER

    def _inject(self, request: Any) -> None:
        tools_block = self._build_tools_block(request)
        if not tools_block:
            return

        # Ensure system_prompt exists
        base_prompt = request.system_prompt or ""

        # Prevent duplicate injection across turns/retries
        if "Available tools:" in base_prompt:
            return

        request.system_prompt = base_prompt + tools_block

    # Sync
    def wrap_model_call(self, request, handler):
        self._inject(request)
        return handler(request)

    # Async
    async def awrap_model_call(self, request, handler):
        self._inject(request)
        return await handler(request)


async def build_subagent_builder_agent(
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

    # Path to the subagent-builder skill
    subagent_builder_path = (
        get_default_filesystem_root() / ".ciri" / "skills" / "subagent-builder"
    )

    # Path to the skill-creator skill (often useful when building agents that use skills)
    skill_creator_path = (
        get_default_filesystem_root() / ".ciri" / "skills" / "skill-creator"
    )

    # Define the SubAgent Builder SubAgent
    subagent_builder_agent = create_deep_agent(
        model=model,
        backend=backend,
        cache=InMemoryCache(),
        name="subagent_builder_agent",
        subagents=[web_researcher_agent],
        system_prompt=SUBAGENT_BUILDER_SYSTEM_PROMPT,
        tools=[build_script_executor_tool(), follow_up_with_human],
        skills=[p for p in [subagent_builder_path, skill_creator_path] if p.exists()],
        middleware=[
            InjectAvailableToolNamesMiddleware(),
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
        name="subagent_builder_agent",
        runnable=subagent_builder_agent,
        description=(
            "A specialized SubAgent for designing and implementing other SubAgents.\n"
            "WHEN TO USE: Invoke this agent when the user wants to 'add a new role', 'create a specialized agent', 'implement a delegated task handler', or 'extend the multi-agent system'.\n"
            "WHY: This agent understands the Ciri subagent architecture (Dynamic vs. Compiled), trigger-based activation, and expert role design.\n"
            "HOW: Provide a task description like 'Create a security auditor subagent that uses the zap-tool' or 'Implement a specialized SQL analyst role'.\n"
            "WHEN NOT TO USE: Do NOT use this for code generation tasks that don't involve creating a formal subagent role."
        ),
    )
