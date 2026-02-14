"""Deepagents come with planning, filesystem, and subagents."""

from collections.abc import Callable, Sequence
from typing import Any

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
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.language_models import BaseChatModel
from langchain.agents.structured_output import ResponseFormat
from langchain.agents.middleware import AgentMiddleware, ToolRetryMiddleware
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
from deepagents.middleware.summarization import SummarizationMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware

from .backend import CiriBackend
from .middlewares import (
    SkillsMiddleware,
    MemoryMiddleware,
    SubAgentMiddleware,
    ToolkitInjectionMiddleware,
)

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."


def create_copilot(
    model: BaseChatModel,
    backend: CiriBackend,
    *,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
    store: BaseStore | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    response_format: ResponseFormat | None = None,
    system_prompt: str | SystemMessage | None = None,
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
) -> CompiledStateGraph:
    if not checkpointer:
        checkpointer = InMemorySaver()
    if not store:
        store = InMemoryStore()
    if not cache:
        cache = InMemoryCache()

    if model is None:
        raise ValueError("Model must be provided")
    elif isinstance(model, str):
        model = init_chat_model(model)

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

    # Pre-instantiate shared middlewares to avoid redundant rescans/syncs
    toolkit_middleware = ToolkitInjectionMiddleware()
    shared_skills_middleware = None
    if skills is not None:
        backend = backend if backend is not None else (lambda rt: StateBackend(rt))
        shared_skills_middleware = SkillsMiddleware(backend=backend, sources=skills)

    # Build middleware stack for subagents
    subagent_middleware: list[AgentMiddleware] = [
        ToolRetryMiddleware(
            max_retries=2,
            retry_on=lambda exc: not isinstance(exc, GraphInterrupt),
            on_failure="continue",
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=10.0,
            jitter=True,
        ),
        TodoListMiddleware(),
        toolkit_middleware,
    ]

    backend = backend if backend is not None else (lambda rt: StateBackend(rt))

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
            on_failure="continue",
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
                subagents=subagents if subagents is not None else [],
                default_middleware=subagent_middleware,
                default_interrupt_on=interrupt_on,
                general_purpose_agent=True,
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

    # Combine system_prompt with BASE_AGENT_PROMPT
    if system_prompt is None:
        final_system_prompt: str | SystemMessage = BASE_AGENT_PROMPT
    elif isinstance(system_prompt, SystemMessage):
        # SystemMessage: append BASE_AGENT_PROMPT to content_blocks
        new_content = [
            *system_prompt.content_blocks,
            {"type": "text", "text": f"\n\n{BASE_AGENT_PROMPT}"},
        ]
        final_system_prompt = SystemMessage(content=new_content)
    else:
        # String: simple concatenation
        final_system_prompt = system_prompt + "\n\n" + BASE_AGENT_PROMPT

    return create_agent(
        model,
        system_prompt=final_system_prompt,
        tools=tools,
        middleware=copilot_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})
