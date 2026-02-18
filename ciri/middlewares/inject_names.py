import yaml
import json
import logging
from pathlib import Path
from typing import Any, List, Tuple, Optional
from langchain.agents.middleware import AgentMiddleware
from ..utils import get_default_filesystem_root, get_core_harness_dir

logger = logging.getLogger(__name__)


class InjectAvailableToolNamesMiddleware(AgentMiddleware):
    """Injects the list of ALL currently available tool names into the system prompt.

    Reads tool names from the live request, so it reflects whatever tools are
    registered at call time (base tools + any injected by ToolkitInjectionMiddleware).
    """

    HEADER = "\n\n# REGISTRY OF AVAILABLE TOOLS\n"
    INSTRUCTION = (
        "CRITICAL: You MUST ONLY select tools from the following list when building "
        "a new subagent. Use the exact tool name as shown.\n\n"
    )
    FOOTER = (
        "\n\n---\n"
        "Subagent Tool Rules: Use exact names above. "
        "Set `tools: all` only if truly needed. "
        "Missing tools cannot be assigned.\n"
    )

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

        base_prompt = request.system_prompt or ""

        # Prevent duplicate injection across turns/retries
        if "REGISTRY OF AVAILABLE TOOLS" in base_prompt:
            return

        request.system_prompt = base_prompt + tools_block

    def wrap_model_call(self, request, handler):
        self._inject(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self._inject(request)
        return await handler(request)


class InjectAvailableSubAgentNamesMiddleware(AgentMiddleware):
    """Injects a registry of discovered subagent names and descriptions into the system prompt.

    Scans core harness and project .ciri/subagents directories on every call so
    the agent always sees the current set of available subagent roles.
    """

    HEADER = "\n\n# REGISTRY OF AVAILABLE SUBAGENTS\n"
    INSTRUCTION = (
        "Reference the following existing subagents when designing workflows or "
        "checking for duplicate roles. Do not recreate a subagent that already exists.\n\n"
    )
    FOOTER = "\n\n---\n"

    def __init__(self, scan_root: Optional[Path] = None):
        super().__init__()
        self.root = scan_root or get_default_filesystem_root()

    def _discover_subagents(self) -> List[Tuple[str, str]]:
        """Return (name, description) pairs from all discoverable subagent configs."""
        result: List[Tuple[str, str]] = []
        seen_names: set = set()

        def _load(path: Path) -> Optional[Tuple[str, str]]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    if path.suffix in (".yaml", ".yml"):
                        data = yaml.safe_load(f) or {}
                    else:
                        data = json.load(f)
                name = data.get("name", path.stem)
                desc = (data.get("description", "") or "").replace("\n", " ")
                desc = desc[:120] + ("..." if len(desc) > 120 else "")
                return name, desc
            except Exception as e:
                logger.debug(f"Could not load subagent {path}: {e}")
                return None

        # 1. Core harness subagents (highest priority)
        try:
            core_dir = get_core_harness_dir() / "subagents"
            if core_dir.is_dir():
                for ext in ("*.yaml", "*.yml", "*.json"):
                    for f in sorted(core_dir.glob(ext)):
                        stem = f.stem
                        if stem not in seen_names:
                            entry = _load(f)
                            if entry:
                                result.append(entry)
                                seen_names.add(stem)
        except Exception as e:
            logger.debug(f"Error scanning core harness subagents: {e}")

        # 2. Project harness subagents
        try:
            for ciri_dir in self.root.rglob(".ciri"):
                if ciri_dir.is_dir() and not any(
                    p.name == ".ciri" for p in ciri_dir.parents if p != self.root
                ):
                    subagents_dir = ciri_dir / "subagents"
                    if subagents_dir.is_dir():
                        for ext in ("*.yaml", "*.yml", "*.json"):
                            for f in subagents_dir.glob(ext):
                                stem = f.stem
                                if stem not in seen_names:
                                    entry = _load(f)
                                    if entry:
                                        result.append(entry)
                                        seen_names.add(stem)
        except Exception as e:
            logger.debug(f"Error scanning project harness subagents: {e}")

        return result

    def _build_block(self) -> str:
        entries = self._discover_subagents()
        if not entries:
            return ""
        lines = [f"- **{name}**: {desc}" for name, desc in entries]
        return self.HEADER + self.INSTRUCTION + "\n".join(lines) + self.FOOTER

    def _inject(self, request: Any) -> None:
        block = self._build_block()
        if not block:
            return
        base_prompt = request.system_prompt or ""
        if "REGISTRY OF AVAILABLE SUBAGENTS" in base_prompt:
            return
        request.system_prompt = base_prompt + block

    def wrap_model_call(self, request, handler):
        self._inject(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self._inject(request)
        return await handler(request)


class InjectAvailableSkillNamesMiddleware(AgentMiddleware):
    """Injects a registry of discovered skill names and descriptions into the system prompt.

    Reads SKILL.md YAML frontmatter from core harness and project .ciri/skills
    directories on every call so the agent always sees the current skill catalogue.
    """

    HEADER = "\n\n# REGISTRY OF AVAILABLE SKILLS\n"
    INSTRUCTION = (
        "Reference the following existing skills when designing workflows or checking "
        "for duplicates. Extend an existing skill rather than creating a new one when possible.\n\n"
    )
    FOOTER = "\n\n---\n"

    def __init__(self, scan_root: Optional[Path] = None):
        super().__init__()
        self.root = scan_root or get_default_filesystem_root()

    def _read_skill_meta(self, skills_dir: Path) -> List[Tuple[str, str]]:
        """Return (name, description) from each skill directory's SKILL.md frontmatter."""
        entries: List[Tuple[str, str]] = []
        if not skills_dir.is_dir():
            return entries

        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                with open(skill_md, "r", encoding="utf-8") as f:
                    content = f.read()
                # Parse YAML frontmatter between --- delimiters
                if content.startswith("---"):
                    end = content.find("---", 3)
                    if end != -1:
                        fm_text = content[3:end].strip()
                        fm = yaml.safe_load(fm_text) or {}
                        name = fm.get("name", skill_dir.name)
                        desc = (fm.get("description", "") or "").replace("\n", " ")
                        desc = desc[:120] + ("..." if len(desc) > 120 else "")
                        entries.append((name, desc))
                        continue
                # Fallback: directory name, no description
                entries.append((skill_dir.name, ""))
            except Exception as e:
                logger.debug(f"Could not parse SKILL.md in {skill_dir}: {e}")

        return entries

    def _discover_skills(self) -> List[Tuple[str, str]]:
        result: List[Tuple[str, str]] = []
        seen_names: set = set()

        def _add(entries: List[Tuple[str, str]]) -> None:
            for name, desc in entries:
                if name not in seen_names:
                    result.append((name, desc))
                    seen_names.add(name)

        # 1. Core harness skills (highest priority)
        try:
            core_skills_dir = get_core_harness_dir() / "skills"
            _add(self._read_skill_meta(core_skills_dir))
        except Exception as e:
            logger.debug(f"Error scanning core harness skills: {e}")

        # 2. Project harness skills
        try:
            for ciri_dir in self.root.rglob(".ciri"):
                if ciri_dir.is_dir() and not any(
                    p.name == ".ciri" for p in ciri_dir.parents if p != self.root
                ):
                    skills_dir = ciri_dir / "skills"
                    _add(self._read_skill_meta(skills_dir))
        except Exception as e:
            logger.debug(f"Error scanning project harness skills: {e}")

        return result

    def _build_block(self) -> str:
        entries = self._discover_skills()
        if not entries:
            return ""
        lines = [
            f"- **{name}**: {desc}" if desc else f"- **{name}**"
            for name, desc in entries
        ]
        return self.HEADER + self.INSTRUCTION + "\n".join(lines) + self.FOOTER

    def _inject(self, request: Any) -> None:
        block = self._build_block()
        if not block:
            return
        base_prompt = request.system_prompt or ""
        if "REGISTRY OF AVAILABLE SKILLS" in base_prompt:
            return
        request.system_prompt = base_prompt + block

    def wrap_model_call(self, request, handler):
        self._inject(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self._inject(request)
        return await handler(request)
