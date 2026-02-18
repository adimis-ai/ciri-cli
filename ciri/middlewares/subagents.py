import yaml
import json
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Literal, Union
from deepagents.middleware import (
    SubAgentMiddleware as BaseSubAgentMiddleware,
    SubAgent as DeepAgentSubAgent,
)
from ..utils import get_default_filesystem_root, get_core_harness_dir

logger = logging.getLogger(__name__)


class SubAgentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Unique name for the sub-agent")
    description: str = Field(
        ..., description="Brief description of the sub-agent's purpose"
    )
    system_prompt: str = Field(
        ..., description="System prompt to guide the sub-agent's behavior"
    )
    model: Optional[str] = Field(
        None,
        description="Model to use for this sub-agent (overrides default)",
        examples=[
            "openai:gpt-5-mini",
            "deepseek:deepseek-chat",
            "google:gemini-2.0-flash",
        ],
    )
    interrupt_on: Optional[dict[str, Any]] = Field(
        None, description="Configuration for interrupting the sub-agent's execution"
    )
    tools: Optional[Union[List[str], Literal["all"]]] = Field(
        None,
        description="List of tool names the sub-agent can use, or 'all' to allow all tools",
    )


class SubAgentMiddleware(BaseSubAgentMiddleware):
    def __init__(
        self,
        *,
        default_model,
        default_tools=None,
        default_middleware=None,
        default_interrupt_on=None,
        subagents=None,
        system_prompt=...,
        general_purpose_agent=True,
        task_description=None,
        scan_root: Optional[Union[str, Path]] = None,
    ):
        if subagents is None:
            subagents = []
        else:
            # Ensure subagents is a list if it was a single item or None
            if not isinstance(subagents, list):
                subagents = [subagents]

        # 1. Store configuration for dynamic scanning
        self.root = Path(scan_root) if scan_root else get_default_filesystem_root()
        self.explicit_subagents = subagents
        self.default_tools_config = default_tools
        logger.debug(f"Scanning for subagents in: {self.root}")

        # 2. Initial scan and load
        self._refresh_subagents()

        super().__init__(
            default_model=default_model,
            default_tools=default_tools,
            default_middleware=default_middleware,
            default_interrupt_on=default_interrupt_on,
            subagents=self.subagents,
            system_prompt=system_prompt,
            general_purpose_agent=general_purpose_agent,
            task_description=task_description,
        )
        self.all_available_tools = set()

    def _refresh_subagents(self):
        """Discover, load, and update subagents."""
        # Discover subagent files
        subagent_files = self._discover_subagent_files(self.root)

        # Load and validate subagents
        discovered_subagents = []
        available_tool_names = set()
        if self.default_tools_config:
            available_tool_names = {t.name for t in self.default_tools_config}

        for file_path in subagent_files:
            try:
                subagent_data = self._load_subagent_file(file_path)
                # Validate against SubAgentConfig model
                sub_agent_config = SubAgentConfig(**subagent_data)

                # Check if tools are available
                if sub_agent_config.tools and sub_agent_config.tools != "all":
                    for tool_name in sub_agent_config.tools:
                        if tool_name not in available_tool_names:
                            logger.warning(
                                f"Tool '{tool_name}' defined for subagent '{sub_agent_config.name}' "
                                f"in {file_path} but not found in available tools list."
                            )
                            # We might not want to raise here during refresh as it could crash a running agent
                            # Instead, just log warning.

                discovered_subagents.append(
                    DeepAgentSubAgent(
                        name=sub_agent_config.name,
                        description=sub_agent_config.description,
                        system_prompt=sub_agent_config.system_prompt,
                        model=sub_agent_config.model,
                        interrupt_on=sub_agent_config.interrupt_on,
                        tools=sub_agent_config.tools,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to load subagent from {file_path}: {e}")
                # Continue with other files

        # Merge subagents: Explicitly passed first, then discovered
        final_subagents = list(self.explicit_subagents)
        seen_names = set()
        for s in self.explicit_subagents:
            if isinstance(s, dict):
                seen_names.add(s["name"])
            else:
                seen_names.add(s.name)

        for ds in discovered_subagents:
            ds_name = ds["name"] if isinstance(ds, dict) else ds.name
            if ds_name not in seen_names:
                final_subagents.append(ds)
                seen_names.add(ds_name)
            else:
                # Discovered subagents with same name as explicit ones are skipped
                pass

        self.subagents = final_subagents
        self._subagents_input = final_subagents
        logger.debug(
            f"Refreshed SubAgentMiddleware with {len(final_subagents)} subagents"
        )

    def _discover_subagent_files(self, root: Path) -> List[Path]:
        """Find all subagent config files: core harness first, then project .ciri/subagents dirs.

        De-duplication strategy: name-based (by file stem).
        If a subagent named 'foo' is found in the core harness, any project-level
        'foo.yaml' / 'foo.yml' / 'foo.json' is silently skipped — the core harness
        version wins.  This prevents duplicate roles from accumulating across projects.

        Ordering:
        1. Core harness: get_core_harness_dir() / "subagents" / "*.{yaml,yml,json}"
        2. Project harness: all <root>/**/.ciri/subagents/*.{yaml,yml,json}
        """
        discovered = []
        seen_names: set = set()

        # 1. Core harness subagent files (global defaults)
        try:
            core_subagents_dir = get_core_harness_dir() / "subagents"
            if core_subagents_dir.is_dir():
                for ext in ["*.yaml", "*.yml", "*.json"]:
                    for config_file in sorted(core_subagents_dir.glob(ext)):
                        stem = config_file.stem
                        if stem not in seen_names:
                            discovered.append(config_file)
                            seen_names.add(stem)
        except Exception as e:
            logger.error(f"Error accessing core harness subagents directory: {e}")

        # 2. Project harness subagent files (skip if same name already from core)
        try:
            for ciri_dir in root.rglob(".ciri"):
                # Ensure we are not inside another .ciri folder
                if ciri_dir.is_dir() and not any(
                    p.name == ".ciri" for p in ciri_dir.parents if p != root
                ):
                    subagents_dir = ciri_dir / "subagents"
                    if subagents_dir.is_dir():
                        for ext in ["*.yaml", "*.yml", "*.json"]:
                            for config_file in subagents_dir.glob(ext):
                                stem = config_file.stem
                                if stem not in seen_names:
                                    discovered.append(config_file)
                                    seen_names.add(stem)
                                else:
                                    logger.debug(
                                        f"Skipping project subagent '{stem}' — "
                                        f"already provided by core harness."
                                    )
        except Exception as e:
            logger.error(f"Error while scanning for subagent files: {e}")

        return discovered

    def _load_subagent_file(self, path: Path) -> Dict[str, Any]:
        """Load subagent configuration from a YAML or JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix in [".yaml", ".yml"]:
                return yaml.safe_load(f)
            elif path.suffix == ".json":
                return json.load(f)
            else:
                raise ValueError(f"Unsupported file extension: {path.suffix}")

    async def awrap_model_call(self, request, handler):
        self._refresh_subagents()
        return await self._wrap_model_call_common(request, handler, is_async=True)

    def wrap_model_call(self, request, handler):
        self._refresh_subagents()
        return self._wrap_model_call_common(request, handler, is_async=False)

    def _wrap_model_call_common(self, request, handler, is_async=False):
        available_tools = set()
        if request.tools:
            for tool in request.tools:
                available_tools.add(tool.name)
        self.all_available_tools.update(available_tools)

        if is_async:
            return super().awrap_model_call(request, handler)
        else:
            return super().wrap_model_call(request, handler)
