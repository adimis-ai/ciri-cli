"""Deepagents come with planning, filesystem, and subagents."""

from typing import Any, Optional
from collections.abc import Callable, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    TodoListMiddleware,
)
from langgraph.types import Checkpointer
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.store.base import BaseStore
from langgraph.errors import GraphInterrupt
from langgraph.cache.memory import InMemoryCache
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import SystemMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ResponseFormat
from langchain.agents.middleware import AgentMiddleware, ToolRetryMiddleware
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
from deepagents.middleware.summarization import SummarizationMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware

from .serializers import LLMConfig
from .backend import CiriBackend
from ._retry_helpers import graphinterrupt_aware_failure
from .utils import (
    is_wsl,
    has_display,
    get_chrome_channel,
    resolve_browser_profile,
    detect_browser_profiles,
    launch_browser_with_cdp,
    get_default_filesystem_root,
)
from .prompts import PLAN_AND_RESEARCH_PROMPT
from .toolkit import (
    build_script_executor_tool,
    follow_up_with_human,
    build_crawler_browser_config,
    CrawlerBrowserConfig,
)
from .subagents import (
    build_skill_builder_agent,
    build_web_researcher_agent,
    build_toolkit_builder_agent,
    build_subagent_builder_agent,
    build_trainer_agent,
)
from .middlewares import (
    SkillsMiddleware,
    MemoryMiddleware,
    SubAgentMiddleware,
    ToolkitInjectionMiddleware,
)

BASE_AGENT_PROMPT = """\
You are **Ciri** — an autonomous AI copilot that lives inside the user's workspace.
You combine deep expertise across software engineering, business operations,
marketing, data analysis, and any domain the user works in, with a self-evolving
multi-agent architecture that lets you permanently learn and improve.

IDENTITY & CAPABILITIES
- You operate inside a workspace directory. Treat every task in the context of the
  files, tools, and domain present there.
- You have filesystem tools, a script execution sandbox, web research capabilities,
  and the ability to delegate to specialized sub-agents.
- You can permanently expand your own abilities by creating Skills, Toolkits, and
  SubAgents via the trainer_agent.

DELEGATION STRATEGY — choose the right agent for the job:
- `web_research_agent` — Any task requiring internet lookup, API docs, live data,
  or browsing. It operates the user's real browser (cookies, sessions).
- `skill_builder_agent` — Create a reusable skill package in .ciri/skills/.
- `toolkit_builder_agent` — Create a new MCP server for API integration in .ciri/toolkits/.
- `subagent_builder_agent` — Create a new specialized agent role in .ciri/subagents/.
- `trainer_agent` — When the user says "learn", "train", "self-improve", or /sync.
  It orchestrates the builders above and manages workspace memory.

PERSISTENT MEMORY — two-level system
The MemoryMiddleware auto-loads ALL .md files from TWO locations on every turn:
1. Core harness memory (OS-level, cross-project): globally applicable context.
2. Workspace memory (.ciri/memory/): THIS project's specific understanding.
- Read .ciri/memory/AGENT.md at the start of complex tasks — it's your workspace index.
- After completing significant work (new features, architectural changes, learned
  patterns, resolved issues), UPDATE .ciri/memory/ files to stay current.
- Keep memory files concise, accurate, and actionable. Delete outdated entries.
- If .ciri/memory/AGENT.md doesn't exist yet, suggest the user run /sync to initialize it.
- Never store secrets or credentials in memory files.

CORE PRINCIPLES
1. Understand before acting — read existing files, memory, and context first.
2. Minimal viable change — don't over-engineer or add unrequested features.
3. Delegate when specialized — use sub-agents for their domains.
4. Be transparent — explain reasoning, flag uncertainty, cite sources.
5. Preserve user work — never destructively overwrite without confirmation.
6. Stay in context — consult and update .ciri/memory/ to maintain continuity.
"""


async def create_copilot(
    # Serializable Args
    debug: bool = False,
    name: str | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    browser_name: Optional[str] = None,
    llm_config: LLMConfig | None = None,
    use_headless_browser: Optional[bool] = None,
    browser_profile_directory: Optional[str] = None,
    system_prompt: str | SystemMessage | None = None,
    crawler_browser_config: Optional[CrawlerBrowserConfig] = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    *,
    # Non-Serializable Args
    all_allowed: bool = False,
    cache: BaseCache | None = None,
    store: BaseStore | None = None,
    backend: CiriBackend | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    response_format: ResponseFormat | None = None,
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
) -> CompiledStateGraph:
    if llm_config is None:
        llm_config = LLMConfig(model="openai/gpt-5-mini")

    model = llm_config.init_langchain_model()

    if interrupt_on is None and not all_allowed:
        interrupt_on = {
            "execute": True,
            "edit_file": True,
            "write_file": True,
        }
    if not checkpointer:
        checkpointer = InMemorySaver()
    if not store:
        store = InMemoryStore()
    if not cache:
        cache = InMemoryCache()

    if not backend:
        backend = CiriBackend()

    if not subagents:
        subagents = []

    # --- Launch the user's real browser and get the CDP endpoint ---
    # Chrome v136+ blocks CDP connections to the user's default profile
    # directory for security.  `resolve_browser_profile()` copies the
    # profile to a CIRI-managed directory which is exempt from this
    # restriction.  We MUST use the *copied* user-data-dir for launch.
    cdp_endpoint: str | None = None
    cdp_user_data_dir = None

    if not is_wsl():
        profile_info = resolve_browser_profile(browser_name, browser_profile_directory)
        if profile_info:
            cdp_user_data_dir = profile_info["user_data_dir"]  # copied dir
            browser_profile_directory = profile_info["profile_directory"]
            browser_name = profile_info["browser"]

    try:
        cdp_endpoint = launch_browser_with_cdp(
            browser_name=browser_name,
            user_data_dir=cdp_user_data_dir,
            profile_directory=browser_profile_directory,
        )
    except RuntimeError:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "Could not launch browser with CDP — web browsing tools will be unavailable. "
            "You can manually start Chrome/Edge with --remote-debugging-port=9222 and restart CIRI.",
            exc_info=True,
        )

    if cdp_endpoint and not crawler_browser_config:
        crawler_browser_config = build_crawler_browser_config(
            cdp_url=cdp_endpoint,
        )

    # Build web researcher once and share across subagent builders
    web_researcher = await build_web_researcher_agent(
        model=model,
        all_allowed=all_allowed,
        cdp_endpoint=cdp_endpoint,
        crawler_browser_config=crawler_browser_config,
    )

    # Await subagent builders, passing the shared web_researcher
    subagents.extend(
        [
            web_researcher,
            await build_skill_builder_agent(
                model=model,
                backend=backend,
                all_allowed=all_allowed,
                cdp_endpoint=cdp_endpoint,
                crawler_browser_config=crawler_browser_config,
                web_researcher_agent=web_researcher,
            ),
            await build_toolkit_builder_agent(
                model=model,
                backend=backend,
                all_allowed=all_allowed,
                cdp_endpoint=cdp_endpoint,
                crawler_browser_config=crawler_browser_config,
                web_researcher_agent=web_researcher,
            ),
            await build_subagent_builder_agent(
                model=model,
                backend=backend,
                all_allowed=all_allowed,
                cdp_endpoint=cdp_endpoint,
                crawler_browser_config=crawler_browser_config,
                web_researcher_agent=web_researcher,
            ),
            await build_trainer_agent(
                model=model,
                backend=backend,
                all_allowed=all_allowed,
                cdp_endpoint=cdp_endpoint,
                crawler_browser_config=crawler_browser_config,
                web_researcher_agent=web_researcher,
            ),
        ]
    )

    if (
        model.profile is not None
        and isinstance(model.profile, dict)
        and "max_input_tokens" in model.profile
        and isinstance(model.profile["max_input_tokens"], int)
    ):
        trigger = ("fraction", 0.85)
        keep = ("fraction", 0.10)
        truncate_args_settings = {
            "trigger": ("fraction", 0.85),
            "keep": ("fraction", 0.10),
        }
    else:
        trigger = ("tokens", 170000)
        keep = ("messages", 6)
        truncate_args_settings = {
            "trigger": ("messages", 20),
            "keep": ("messages", 20),
        }

    # Combine system_prompt with BASE_AGENT_PROMPT and PLAN_AND_RESEARCH_PROMPT
    cwd = get_default_filesystem_root()
    full_base_prompt = f"{BASE_AGENT_PROMPT}\n\n[CURRENT WORKING DIRECTORY]: {cwd}\n\n{PLAN_AND_RESEARCH_PROMPT}"
    if system_prompt is None:
        final_system_prompt: str | SystemMessage = full_base_prompt
    elif isinstance(system_prompt, SystemMessage):
        # SystemMessage: append base prompts to content
        if isinstance(system_prompt.content, list):
            new_content = list(system_prompt.content) + [
                {"type": "text", "text": f"\n\n{full_base_prompt}"}
            ]
        else:
            new_content = f"{system_prompt.content}\n\n{full_base_prompt}"
        final_system_prompt = SystemMessage(content=new_content)
    else:
        # String: simple concatenation
        final_system_prompt = system_prompt + "\n\n" + full_base_prompt

    # Pre-instantiate shared middlewares to avoid redundant rescans/syncs
    toolkit_middleware = ToolkitInjectionMiddleware()
    await toolkit_middleware.refresh()

    # SkillsMiddleware is ALWAYS created — it now auto-discovers both the OS-level
    # core harness (get_core_harness_dir()/skills/) and any project .ciri/skills/
    # directories at runtime.  Explicit `skills` paths from the caller are added on top.
    shared_skills_middleware = SkillsMiddleware(backend=backend, sources=skills or [])

    # MemoryMiddleware is ALWAYS created — it now auto-discovers both the OS-level
    # core harness (get_core_harness_dir()/memory/) and project .ciri/memory/ files.
    # Explicit `memory` paths from the caller are added on top.
    shared_memory_middleware = MemoryMiddleware(backend=backend, sources=memory or [])

    # Build middleware stack for subagents
    subagent_middleware: list[AgentMiddleware] = [
        ToolRetryMiddleware(
            max_retries=2,
            retry_on=lambda exc: not isinstance(exc, GraphInterrupt),
            on_failure=graphinterrupt_aware_failure,
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=10.0,
            jitter=True,
        ),
        TodoListMiddleware(),
        toolkit_middleware,
        shared_skills_middleware,
    ]

    subagent_middleware.extend(
        [
            FilesystemMiddleware(backend=backend),
            SummarizationMiddleware(
                model=model,
                backend=backend,
                trigger=trigger,
                keep=keep,
                trim_tokens_to_summarize=None,
                truncate_args_settings=truncate_args_settings,
            ),
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
            PatchToolCallsMiddleware(),
        ]
    )

    # Build main agent middleware stack
    copilot_middleware: list[AgentMiddleware] = [
        ToolRetryMiddleware(
            max_retries=2,
            retry_on=lambda exc: not isinstance(exc, GraphInterrupt),
            on_failure=graphinterrupt_aware_failure,
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=10.0,
            jitter=True,
        ),
        TodoListMiddleware(),
        toolkit_middleware,
        shared_memory_middleware,
        shared_skills_middleware,
    ]
    copilot_middleware.extend(
        [
            FilesystemMiddleware(backend=backend),
            SubAgentMiddleware(
                default_model=model,
                default_tools=tools,
                subagents=subagents,
                default_middleware=subagent_middleware,
                default_interrupt_on=interrupt_on,
                general_purpose_agent=True,
                system_prompt=final_system_prompt,
            ),
            SummarizationMiddleware(
                model=model,
                backend=backend,
                trigger=trigger,
                keep=keep,
                trim_tokens_to_summarize=None,
                truncate_args_settings=truncate_args_settings,
            ),
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
            PatchToolCallsMiddleware(),
        ]
    )
    if middleware:
        copilot_middleware.extend(middleware)
    if interrupt_on is not None:
        copilot_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    merged_tools = list(tools or []) + [
        build_script_executor_tool(output_callback=backend.output_callback),
        follow_up_with_human,
    ]

    return create_agent(
        model,
        name=name,
        store=store,
        debug=debug,
        cache=cache,
        checkpointer=checkpointer,
        middleware=copilot_middleware,
        context_schema=context_schema,
        response_format=response_format,
        system_prompt=final_system_prompt,
        tools=merged_tools,
    ).with_config({"recursion_limit": 1000})
