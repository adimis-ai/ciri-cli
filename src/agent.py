from __future__ import annotations
import os
import sys
import shutil
import asyncio
import threading
import yaml
from pathlib import Path
from dotenv import load_dotenv


from functools import cached_property
from langchain.agents import AgentState
from langgraph.types import Checkpointer
from langgraph.errors import GraphInterrupt
from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore
from langgraph.cache.base import BaseCache
from deepagents.backends import FilesystemBackend
from deepagents import SubAgent, CompiledSubAgent
from pydantic import BaseModel, Field, ConfigDict
from langchain.chat_models import init_chat_model, BaseChatModel
from langchain.agents.middleware import AgentMiddleware, ToolRetryMiddleware
from typing import (
    Optional,
    List,
    Dict,
    Literal,
    Union,
    Any,
    Sequence,
    Mapping,
    Tuple,
    Type,
)
from typing_extensions import NotRequired, TypedDict
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents.middleware.shell_tool import (
    RedactionRule,
    ShellToolState,
    ShellToolMiddleware,
)
from deepagents import create_deep_agent
from deepagents.middleware.memory import MemoryState
from deepagents.middleware.skills import SkillsState
from langchain.agents.middleware.todo import PlanningState
from deepagents.middleware.filesystem import FilesystemState


from .middlewares import TeamRuntimeMiddleware
from .toolkit import (
    follow_up_with_human,
    FollowUpInterruptValue,
)
from .subagents.web_researcher import build_web_researcher_agent
from .utils import get_default_filesystem_root, load_all_dotenv, find_windows_bash

load_all_dotenv()

# Constants
CIRI_SYSTEM_PROMPT = """You are CIRI, a female desktop-class personal AI copilot embedded in the user's local environment. You are an autonomous execution agent with tools, persistent memory, filesystem access, and the ability to delegate complex work to subagents.

Never fabricate tool outputs. If a task requires external data, use a tool.

**File safety:** Always read before modifying. Never delete or overwrite without explicit user instruction.

# Operating Principles

- **Plan before complex actions.** For multi-step or irreversible operations, state your plan before executing. For simple tasks, act directly.
- **Report failures transparently.** State what failed, why, and the recommended next step. Never silently ignore errors or fabricate success.
- **Protect user data.** Treat the filesystem as real and permanent. Prefer non-destructive operations.
- **Be concise.** State what you did, found, or need, then stop. No filler, no hedging, no meta-commentary, no emojis.
"""

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# Utility Functions
def _run_coroutine_sync(coro):
    """Run an async coroutine from a sync context, even if an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result = None
    exception = None

    def _target():
        nonlocal result, exception
        try:
            result = asyncio.run(coro)
        except BaseException as exc:
            exception = exc

    thread = threading.Thread(target=_target)
    thread.start()
    thread.join()
    if exception is not None:
        raise exception
    return result


# Configuration Models
class LLMConfig(BaseModel):
    """Configuration for language models."""

    model: str = Field(
        description="The language model to use, e.g. 'openai/gpt-5-mini' or 'openai:gpt-4'."
    )
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @cached_property
    def _parsed_model(self) -> Tuple[str | None, str]:
        if ":" in self.model:  # langchain direct provider:model
            return tuple(self.model.split(":", 1))
        if "/" in self.model:  # openrouter provider/model
            return tuple(self.model.split("/", 1))
        return None, self.model

    @cached_property
    def _is_openrouter(self) -> bool:
        # provider/model AND provider not explicitly specified via provider:model
        return "/" in self.model and ":" not in self.model

    @cached_property
    def _resolved_api_config(self) -> Dict[str, Any]:
        """
        Resolve API config once and cache.
        Avoid repeated env access + dict copies.
        """
        config = dict(self.model_kwargs)  # single copy

        # Fast path if api_key already provided
        if "api_key" in config:
            if self._is_openrouter and "base_url" not in config:
                config["base_url"] = os.getenv(
                    "OPENROUTER_API_BASE_URL", DEFAULT_OPENROUTER_BASE_URL
                )
            return config

        provider, _ = self._parsed_model

        if self._is_openrouter:
            config.setdefault(
                "base_url",
                os.getenv("OPENROUTER_API_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
            )
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY env variable not set.")
            config["api_key"] = api_key
            return config

        # Direct provider key lookup
        if not provider:
            raise ValueError(f"Provider missing in model: {self.model}")

        env_key = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError(f"{env_key} env variable not set for {self.model}")

        config["api_key"] = api_key
        return config

    def init_langchain_model(self) -> BaseChatModel:
        """
        Initialize LangChain chat model.
        Zero redundant env reads or parsing.
        """
        provider, model_name = self._parsed_model
        config = self._resolved_api_config

        if self._is_openrouter:
            # avoid dict rebuild via pop-free filtering
            base_url = config.get("base_url")
            api_key = config["api_key"]

            extra = {
                k: v for k, v in config.items() if k not in ("api_key", "base_url")
            }

            return init_chat_model(
                model=model_name,
                model_provider="openai",
                base_url=base_url,
                api_key=api_key,
                **extra,
            )

        return init_chat_model(model=self.model, **config)


class ShellToolConfig(BaseModel):
    """Configuration for shell tool middleware."""

    env: Optional[Mapping[str, Any]] = None
    shell_command: Optional[Union[Sequence[str], str]] = None
    startup_commands: Optional[Union[tuple, list, str]] = None
    shutdown_commands: Optional[Union[tuple, list, str]] = None
    redaction_rules: tuple[RedactionRule, ...] | list[RedactionRule] | None = None


class SerializableSubAgent(BaseModel):
    """Serializable configuration for subagents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    system_prompt: str
    use_parent_mcp_tools: bool = True
    include_follow_up_with_human_tool: bool = True
    include_shell_tool_middleware: bool = True
    llm_config: Optional[LLMConfig] = None
    interrupt_on: Optional[Dict[str, Any]] = (
        None  # Union[bool, InterruptOnConfig] - Any used for Pydantic compatibility
    )
    shell_tool_config: Optional[ShellToolConfig] = None
    mcp_connections: Optional[Dict[str, Any]] = None


class ActionRequest(TypedDict):
    name: str
    description: str
    arguments: NotRequired[dict[str, Any]]


class ReviewConfig(TypedDict):
    action_name: str
    allowed_decisions: list[Literal["approve", "edit", "reject"]]


class HumanInTheLoopInterrupt(TypedDict):
    review_configs: list[ReviewConfig]
    action_requests: list[ActionRequest]


class InterruptValue(TypedDict):
    value: Union[HumanInTheLoopInterrupt, FollowUpInterruptValue]


class ApprovalDecision(TypedDict):
    type: Literal["approve"]


class EditedAction(TypedDict):
    name: str
    args: dict[str, Any]


class EditDecision(TypedDict):
    type: Literal["edit"]
    edited_action: EditedAction


class RejectDecision(TypedDict):
    type: Literal["reject"]
    message: NotRequired[str]


class ApprovalDecisions(TypedDict):
    decisions: list[ApprovalDecision]


class EditDecisions(TypedDict):
    decisions: list[EditDecision]


class RejectDecisions(TypedDict):
    decisions: list[RejectDecision]


class ResumeCommand(TypedDict):
    resume: Union[ApprovalDecisions, EditDecisions, RejectDecisions]


class CiriState(
    AgentState[Any],
    MemoryState,
    SkillsState,
    ShellToolState,
    FilesystemState,
    PlanningState,
):
    __interrupt__: NotRequired[Optional[List[InterruptValue]]]


# Helper Classes
class FileSystemScanner:
    """Efficiently scans and caches filesystem paths."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for folder in ("memory", "skills", "subagents"):
            (self.root_dir / ".ciri" / folder).mkdir(parents=True, exist_ok=True)

    def scan_memory_paths(self) -> List[str]:
        """Scan and return memory file paths."""
        paths = ["/.ciri/memory/"]
        memory_dir = self.root_dir / ".ciri" / "memory"

        for p in sorted(memory_dir.glob("*.md")):
            if p.is_file():
                paths.append(f"/.ciri/memory/{p.name}")

        return paths

    def scan_skills_paths(self) -> List[str]:
        """Scan and return skills file paths."""
        paths = ["/.ciri/skills/"]
        skills_dir = self.root_dir / ".ciri" / "skills"

        for p in sorted(skills_dir.glob("*/SKILL.md")):
            if p.is_file():
                paths.append(f"/.ciri/skills/{p.parent.name}/SKILL.md")

        return paths

    def scan_subagent_paths(self) -> List[Path]:
        """Scan and return subagent YAML file paths."""
        subagents_dir = self.root_dir / ".ciri" / "subagents"
        return sorted(subagents_dir.glob("*.yaml"))


class MCPClientManager:
    """Manages MCP client initialization and tool retrieval."""

    @staticmethod
    def get_tools(connections: Optional[Dict[str, Any]]) -> List[BaseTool]:
        """Get MCP tools from connections."""
        if not connections:
            return []

        mcp_client = MultiServerMCPClient(connections=connections)
        return _run_coroutine_sync(mcp_client.get_tools())

    @staticmethod
    def merge_connections(
        parent_connections: Optional[Dict[str, Any]],
        child_connections: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Merge parent and child MCP connections."""
        if not parent_connections and not child_connections:
            return None

        return {
            **(parent_connections or {}),
            **(child_connections or {}),
        }


class MiddlewareBuilder:
    """Builds middleware stacks for agents."""

    def __init__(
        self,
        root_dir: Path,
        backend: FilesystemBackend,
        llm_config: LLMConfig,
        memory_paths: List[str],
        skills_paths: List[str],
    ):
        self.root_dir = root_dir
        self.backend = backend
        self.llm_config = llm_config
        self.memory_paths = memory_paths
        self.skills_paths = skills_paths

    def build_shell_tool_middleware(
        self,
        shell_config: Optional[ShellToolConfig] = None,
    ) -> ShellToolMiddleware:
        """Build shell tool middleware."""
        shell_command = getattr(shell_config, "shell_command", None)
        # Default shell_command is /bin/bash which doesn't exist on Windows;
        # the shell session requires a POSIX shell (uses printf, $?), so find
        # Git Bash or WSL bash on Windows.
        if shell_command is None and sys.platform == "win32":
            shell_command = find_windows_bash()
        return ShellToolMiddleware(
            workspace_root=self.root_dir,
            env=getattr(shell_config, "env", None),
            startup_commands=getattr(shell_config, "startup_commands", None),
            shutdown_commands=getattr(shell_config, "shutdown_commands", None),
            redaction_rules=getattr(shell_config, "redaction_rules", None),
            shell_command=shell_command,
        )


class ToolsBuilder:
    """Builds tool collections for agents."""

    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config

    def build_base_tools(
        self,
        mcp_tools: Optional[List[BaseTool]] = None,
        custom_tools: Optional[List[BaseTool]] = None,
        include_follow_up_with_human: bool = True,
    ) -> List[BaseTool]:
        """Build base tool collection."""
        tools = []

        if custom_tools:
            tools.extend(custom_tools)

        if mcp_tools:
            tools.extend(mcp_tools)

        if include_follow_up_with_human:
            tools.append(follow_up_with_human)

        return tools

    def build_subagent_tools(
        self,
        mcp_tools: Optional[List[BaseTool]] = None,
        include_follow_up_with_human: bool = True,
    ) -> List[BaseTool]:
        """Build tool collection for subagents."""
        tools = []

        if mcp_tools:
            tools.extend(mcp_tools)

        if include_follow_up_with_human:
            tools.append(follow_up_with_human)

        return tools


class SubAgentCompiler:
    """Compiles serializable subagent configurations into SubAgent instances."""

    def __init__(
        self,
        root_dir: Path,
        parent_llm_config: LLMConfig,
        parent_shell_config: Optional[ShellToolConfig],
        parent_mcp_connections: Optional[Dict[str, Any]],
    ):
        self.root_dir = root_dir
        self.parent_llm_config = parent_llm_config
        self.parent_shell_config = parent_shell_config
        self.parent_mcp_connections = parent_mcp_connections

        # Scan filesystem once
        scanner = FileSystemScanner(root_dir)
        self.memory_paths = scanner.scan_memory_paths()
        self.skills_paths = scanner.scan_skills_paths()

    def compile(
        self,
        subagent_config: SerializableSubAgent,
    ) -> SubAgent:
        """Compile a single subagent configuration."""
        # Determine connections
        connections = self._get_connections(subagent_config)

        # Get LLM config
        llm_config = subagent_config.llm_config or self.parent_llm_config

        # Get MCP tools
        mcp_tools = MCPClientManager.get_tools(connections)

        # Build tools
        tools_builder = ToolsBuilder(
            llm_config=llm_config,
        )
        tools = tools_builder.build_subagent_tools(
            mcp_tools=mcp_tools,
            include_follow_up_with_human=subagent_config.include_follow_up_with_human_tool,
        )

        # Build middleware
        middleware = self._build_middleware(
            subagent_config=subagent_config,
            llm_config=llm_config,
        )

        return SubAgent(
            name=subagent_config.name,
            description=subagent_config.description,
            interrupt_on=subagent_config.interrupt_on,
            system_prompt=subagent_config.system_prompt,
            model=llm_config.init_langchain_model(),
            middleware=middleware,
            tools=tools,
        )

    def _get_connections(
        self,
        subagent_config: SerializableSubAgent,
    ) -> Optional[Dict[str, Any]]:
        """Get merged MCP connections."""
        if subagent_config.use_parent_mcp_tools:
            return MCPClientManager.merge_connections(
                self.parent_mcp_connections,
                subagent_config.mcp_connections,
            )
        return subagent_config.mcp_connections

    def _build_middleware(
        self,
        subagent_config: SerializableSubAgent,
        llm_config: LLMConfig,
    ) -> List[AgentMiddleware]:
        """Build middleware stack for subagent."""
        middleware_list = []

        # Dummy backend for middleware builder
        backend = FilesystemBackend(root_dir=self.root_dir)

        middleware_builder = MiddlewareBuilder(
            root_dir=self.root_dir,
            backend=backend,
            llm_config=llm_config,
            memory_paths=self.memory_paths,
            skills_paths=self.skills_paths,
        )

        if subagent_config.include_shell_tool_middleware:
            shell_config = subagent_config.shell_tool_config or self.parent_shell_config
            middleware_list.append(
                middleware_builder.build_shell_tool_middleware(
                    shell_config=shell_config
                )
            )

        return middleware_list


# Main Agent Class
class Ciri(BaseModel):
    """CIRI agent configuration and compilation."""

    llm_config: LLMConfig
    instructions: Optional[str] = None
    include_follow_up_with_human_tool: bool = True
    web_search: bool = True
    browser_name: Optional[str] = None
    profile_directory: Optional[str] = None
    headless: Optional[bool] = None
    mcp_connections: Optional[Dict[str, Any]] = None
    shell_tool_config: Optional[ShellToolConfig] = None
    interrupt_on: Optional[Union[bool, Dict[str, Any]]] = (
        None  # Union[bool, InterruptOnConfig] - Any used for Pydantic compatibility
    )

    def _compile_subagents(
        self,
        root_dir: Path,
    ) -> List[Union[SubAgent, CompiledSubAgent]]:
        """Compile all subagent configurations.

        When ``self.web_search`` is True (default), the web-researcher
        sub-agent is automatically included.  It uses the user's real
        browser profile for anti-detection on guarded platforms.
        """
        subagents: List[Union[SubAgent, CompiledSubAgent]] = []

        # --- Web researcher sub-agent ---
        if self.web_search:
            web_researcher = build_web_researcher_agent(
                model=self.llm_config.init_langchain_model(),
                browser_name=self.browser_name,
                profile_directory=self.profile_directory,
                headless=self.headless,
            )
            subagents.append(web_researcher)

        # --- YAML-defined sub-agents ---
        scanner = FileSystemScanner(root_dir)
        subagent_paths = scanner.scan_subagent_paths()

        if subagent_paths:
            compiler = SubAgentCompiler(
                root_dir=root_dir,
                parent_llm_config=self.llm_config,
                parent_shell_config=self.shell_tool_config,
                parent_mcp_connections=self.mcp_connections,
            )
            subagents.extend(
                compiler.compile(
                    SerializableSubAgent.model_validate(
                        yaml.safe_load(path.read_text())
                    ),
                )
                for path in subagent_paths
            )

        return subagents

    def _create_ciri(
        self,
        root_dir: Path,
        store: BaseStore,
        checkpointer: Checkpointer,
        compiled_subagents: List[Union[SubAgent, CompiledSubAgent]],
        cache: Optional[BaseCache] = None,
        context_schema: Optional[Any] = None,
        tools: Optional[List[BaseTool]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        middleware: Optional[List[AgentMiddleware]] = None,
    ):
        """Create the CIRI agent instance (private method)."""
        # Get MCP tools
        mcp_tools = MCPClientManager.get_tools(self.mcp_connections)

        # Scan filesystem
        scanner = FileSystemScanner(root_dir)
        memory_paths = scanner.scan_memory_paths()
        skills_paths = scanner.scan_skills_paths()

        # Create backend
        backend = FilesystemBackend(
            root_dir=root_dir,
        )

        # Build middleware
        middleware_builder = MiddlewareBuilder(
            root_dir=root_dir,
            backend=backend,
            llm_config=self.llm_config,
            memory_paths=memory_paths,
            skills_paths=skills_paths,
        )

        middleware_stack = [
            *(middleware or []),
        ]

        middleware_stack.extend(
            [
                ToolRetryMiddleware(
                    max_retries=2,  # 2 retries after initial attempt
                    retry_on=lambda exc: not isinstance(
                        exc, GraphInterrupt
                    ),  # Retry all except graph interrupts
                    on_failure="continue",  # Return error message to LLM on final failure
                    backoff_factor=2.0,  # Exponential backoff multiplier
                    initial_delay=1.0,  # 1 second initial delay
                    max_delay=10.0,  # Cap at 10 seconds
                    jitter=True,  # Add ±25% jitter
                ),
                middleware_builder.build_shell_tool_middleware(self.shell_tool_config),
            ]
        )

        middleware_stack.extend(
            [
                TeamRuntimeMiddleware(
                    backend=backend,
                    root_dir=root_dir,
                    shell_config=self.shell_tool_config,
                ),
            ]
        )

        # Build tools
        tools_builder = ToolsBuilder(
            llm_config=self.llm_config,
        )
        agent_tools = tools_builder.build_base_tools(
            mcp_tools=mcp_tools,
            custom_tools=tools,
            include_follow_up_with_human=self.include_follow_up_with_human_tool,
        )

        # Build system prompt
        system_prompt = CIRI_SYSTEM_PROMPT
        if self.instructions:
            system_prompt += f"\n# Additional Instructions\n\n{self.instructions}"

        # Create and return agent
        return create_deep_agent(
            name="ciri",
            store=store,
            cache=cache,
            backend=backend,
            memory=memory_paths,
            skills=skills_paths,
            subagents=compiled_subagents,
            interrupt_on=self.interrupt_on,
            checkpointer=checkpointer,
            context_schema=context_schema,
            response_format=response_format,
            system_prompt=system_prompt,
            model=self.llm_config.init_langchain_model(),
            tools=agent_tools,
            middleware=middleware_stack,
        )

    def compile(
        self,
        checkpointer: Checkpointer,
        filesystem_root_dir: Optional[Union[str, Path]] = None,
        *,
        cache: Optional[BaseCache] = None,
        store: Optional[BaseStore] = None,
        context_schema: Optional[Any] = None,
        tools: Optional[List[BaseTool]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        middleware: Optional[List[AgentMiddleware]] = None,
    ):
        if not filesystem_root_dir:
            filesystem_root_dir = get_default_filesystem_root()

        root_dir = Path(filesystem_root_dir).resolve()

        compiled_subagents = self._compile_subagents(
            root_dir=root_dir,
        )

        return self._create_ciri(
            root_dir=root_dir,
            store=store,
            checkpointer=checkpointer,
            compiled_subagents=compiled_subagents,
            cache=cache,
            context_schema=context_schema,
            tools=tools,
            response_format=response_format,
            middleware=middleware,
        )
