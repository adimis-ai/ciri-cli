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
)
from .middlewares import (
    SkillsMiddleware,
    MemoryMiddleware,
    SubAgentMiddleware,
    ToolkitInjectionMiddleware,
)

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."


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
        llm_config = LLMConfig(model="openai/gpt-oss-120b:free")

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

    # --- resolve profile & crawler config once for all subagents ---
    profile_info = None
    # Auto-find browser profile unless in WSL2 (which is complex to auto-map)
    if not is_wsl() and not browser_profile_directory:
        profile_info = resolve_browser_profile(browser_name, browser_profile_directory)
        if profile_info:
            browser_profile_directory = profile_info["profile_directory"]
            browser_name = profile_info["browser"]

    effective_headless = (
        use_headless_browser
        if use_headless_browser is not None
        else (not has_display())
    )
    effective_channel = get_chrome_channel()

    if not crawler_browser_config:
        crawler_browser_config = build_crawler_browser_config(
            profile_info=profile_info,
            headless=effective_headless,
            channel=effective_channel,
        )

    # Build web researcher once and share across subagent builders
    web_researcher = await build_web_researcher_agent(
        model=model,
        all_allowed=all_allowed,
        browser_name=browser_name,
        headless=effective_headless,
        profile_directory=browser_profile_directory,
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
                browser_name=browser_name,
                headless=effective_headless,
                profile_directory=browser_profile_directory,
                crawler_browser_config=crawler_browser_config,
                web_researcher_agent=web_researcher,
            ),
            await build_toolkit_builder_agent(
                model=model,
                backend=backend,
                all_allowed=all_allowed,
                browser_name=browser_name,
                headless=effective_headless,
                profile_directory=browser_profile_directory,
                crawler_browser_config=crawler_browser_config,
                web_researcher_agent=web_researcher,
            ),
            await build_subagent_builder_agent(
                model=model,
                backend=backend,
                all_allowed=all_allowed,
                browser_name=browser_name,
                headless=effective_headless,
                profile_directory=browser_profile_directory,
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
    shared_skills_middleware = None
    if skills is not None:
        shared_skills_middleware = SkillsMiddleware(backend=backend, sources=skills)

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
    ]

    if shared_skills_middleware:
        subagent_middleware.append(shared_skills_middleware)

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
    ]
    if memory is not None:
        copilot_middleware.append(MemoryMiddleware(backend=backend, sources=memory))
    if shared_skills_middleware:
        copilot_middleware.append(shared_skills_middleware)
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
        build_script_executor_tool(),
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
