from pathlib import Path
from typing import Optional
from langgraph.errors import GraphInterrupt
from langgraph.cache.memory import InMemoryCache
from langchain_core.language_models import BaseChatModel
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents.middleware import ToolRetryMiddleware

from ..backend import CiriBackend
from ..prompts import BUILDER_CORE_PROMPT
from ..utils import get_default_filesystem_root, get_core_harness_dir
from .._retry_helpers import graphinterrupt_aware_failure
from .web_researcher import build_web_researcher_agent, CrawlerBrowserConfig
from ..toolkit import build_script_executor_tool, follow_up_with_human
from ..middlewares import (
    InjectAvailableToolNamesMiddleware,
    InjectAvailableSubAgentNamesMiddleware,
    InjectAvailableSkillNamesMiddleware,
)

WORKING_DIR_DEFAULT = get_core_harness_dir() / "skills"


SKILL_BUILDER_SYSTEM_PROMPT_TEMPLATE = (
    """You are the **Skill Engineer** for Ciri. You create reusable Skill \
packages that teach Ciri specialized workflows and domain knowledge.

WORKING_DIR: `{working_dir}`

WHAT IS A SKILL?
A self-contained directory in WORKING_DIR containing a SKILL.md (instructions the
agent follows when the skill is triggered) plus optional scripts/, references/,
and assets/ subdirectories.

PROGRESSIVE DISCLOSURE — minimize token cost:
- Metadata (always loaded): `name` + `description` in SKILL.md YAML frontmatter.
  Description MUST include specific "when to use" triggers.
- Body (loaded on trigger): SKILL.md content. Keep under 500 lines. Focus on
  orchestration steps, not raw data.
- Resources (loaded on demand): scripts/, references/, assets/. Put detailed
  logic, schemas, templates here. Link from SKILL.md with relative paths.

MANDATORY PROCESS
1. INIT — Run `python3 {skill_creator_scripts}/init_skill.py <skill-name> --path {working_dir}`
   via `execute`. This creates `{working_dir}/<skill-name>/SKILL.md` plus
   scripts/, references/, and assets/ subdirectories. You MUST pass `--path {working_dir}`
   or the skill will be created in the wrong location.
2. BUILD — Edit `{working_dir}/<skill-name>/SKILL.md` (created by step 1). Populate
   scripts/, references/ first, then update SKILL.md to link them. Use imperative
   mood ("Run script...", "Analyze..."). SKILL.md is the ONLY accepted instruction
   file — do NOT create README.md or any other top-level markdown file instead.
3. PACKAGE — Run `python3 {skill_creator_scripts}/package_skill.py \
   {working_dir}/<skill-name>` via `execute`. Fix any errors immediately.

SKILL.md RULES
- YAML frontmatter: ONLY `name` and `description` (max 1024 chars).
- Body: Lean orchestration instructions. If content exceeds ~100 lines, move
  details to references/ and link to them.
- Skills can reference available toolkits and subagents by name for multi-step
  workflows. Consult the REGISTRY OF AVAILABLE TOOLS, REGISTRY OF AVAILABLE
  SUBAGENTS, and REGISTRY OF AVAILABLE SKILLS injected at the end of this prompt
  to see what is currently installed.
- The file MUST be named exactly `SKILL.md` — the skills loader only reads this filename.

FORBIDDEN: README.md, INSTALL.md, requirements.txt in skill root. No vague
descriptions like "A skill for git" — always specify trigger phrases.
CRITICAL: Never create README.md as a substitute for SKILL.md. Skills without
SKILL.md are silently ignored and will not be loaded by the agent.

EXAMPLE SKILL.MD:
```markdown
---
name: pdf-processor
description: Manipulate PDF files. Use when asked to "rotate", "merge", "split", or "extract text" from PDFs.
---
# PDF Processor
## Rotate Pages
Run `python3 scripts/rotate_pdf.py --input <file> --angle 90`
## Merge Files
Run `python3 scripts/merge_pdfs.py --files <list>`
## Reference
See [API docs](references/api_docs.md) for advanced options.
```
"""
    + "\n\n"
    + BUILDER_CORE_PROMPT
)


async def build_skill_builder_agent(
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

    # Path to the skill-creator skill — check core harness first, fall back to project harness
    _core_skill_creator = get_core_harness_dir() / "skills" / "skill-creator"
    _proj_skill_creator = (
        get_default_filesystem_root() / ".ciri" / "skills" / "skill-creator"
    )
    skill_creator_path = (
        _core_skill_creator if _core_skill_creator.exists() else _proj_skill_creator
    )

    skill_creator_scripts = skill_creator_path / "scripts"

    # Format the system prompt with the working directory
    system_prompt = SKILL_BUILDER_SYSTEM_PROMPT_TEMPLATE.format(
        working_dir=working_dir,
        skill_creator_scripts=skill_creator_scripts,
    )

    interrupt_on = None
    if not all_allowed:
        interrupt_on = {
            "execute": True,
            "edit_file": True,
            "write_file": True,
        }

    # Define the Skill Builder SubAgent
    skill_builder_agent = create_deep_agent(
        model=model,
        backend=backend,
        cache=InMemoryCache(),
        name="skill_builder_agent",
        interrupt_on=interrupt_on,
        subagents=[web_researcher_agent],
        system_prompt=system_prompt,
        tools=[build_script_executor_tool(), follow_up_with_human],
        skills=[skill_creator_path] if skill_creator_path.exists() else [],
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
        name="skill_builder_agent",
        runnable=skill_builder_agent,
        description=(
            f"Creates and manages reusable Skill packages in {working_dir}. "
            "Invoke when: user says 'create a skill', 'build a capability', or "
            "'add a workflow'. Provide task like 'Create a pdf-processing skill'. "
            "Do NOT use for one-off scripts or general coding."
        ),
    )
