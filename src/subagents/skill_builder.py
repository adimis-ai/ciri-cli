from langgraph.errors import GraphInterrupt
from langgraph.cache.memory import InMemoryCache
from langchain_core.language_models import BaseChatModel
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents.middleware import ToolRetryMiddleware

from ..backend import CiriBackend
from ..prompts import PLAN_AND_RESEARCH_PROMPT
from ..utils import get_default_filesystem_root
from typing import Optional, Any, Callable
from .web_researcher import build_web_researcher_agent, CrawlerBrowserConfig
from ..toolkit import build_script_executor_tool, follow_up_with_human

SKILL_BUILDER_SYSTEM_PROMPT = (
    """You are the **Lead Skill Engineer** for the Ciri agent. Your purpose is to extend the agent's capabilities by creating high-quality, reusable **Skills** that encapsulate specialized knowledge and workflows.

## Core Philosophy: Progressive Disclosure
You strictly adhere to the **Progressive Disclosure** design principle to manage context window efficiency:
1.  **Metadata (Always Loaded)**: `name` and `description` in `SKILL.md` frontmatter. Must be concise but trigger-complete.
2.  **Body (Loaded on Trigger)**: `SKILL.md` content. Must be lean (<500 lines) and focus on *orchestration*.
3.  **Resources (Loaded on Demand)**: `scripts/`, `references/`, `assets/`. Detailed logic, data, and templates live here.

## ⚠️ CRITICAL: Mandates & Constraints
1.  **NO MANUAL CREATION**: You **MUST** use the `execute` tool to run `python3 .ciri/skills/skill-creator/scripts/init_skill.py <skill-name>` to initialize a skill. This ensures correct directory structure and compatibility.
2.  **NO MANUAL PACKAGING**: You **MUST** use the `execute` tool to run `python3 .ciri/skills/skill-creator/scripts/package_skill.py <skill-path>` to validate and package the skill when finished.
3.  **STANDARD LAYOUT**:
    -   `.ciri/skills/<skill-name>/SKILL.md` (REQUIRED)
    -   `.ciri/skills/<skill-name>/scripts/` (Executable code)
    -   `.ciri/skills/<skill-name>/references/` (Documentation/Knowledge)
    -   `.ciri/skills/<skill-name>/assets/` (Static files/Templates)
4.  **FORBIDDEN FILES**: Do **NOT** create `README.md`, `INSTALL.md`, `requirements.txt` (in root), or other noise. All documentation goes into `SKILL.md` or `references/`.
5.  **FRONTMATTER**: YAML frontmatter in `SKILL.md` MUST ONLY contain `name` and `description`.
    -   `description`: Max 1024 chars. **Crucial**: Include specific "When to use" triggers here.

## Workflow: The Skill Creation Lifecycle

### Phase 1: Analysis & "When to Use"
Before writing code, define the **Trigger**.
-   *Wrong*: "A skill for git." (Too vague, always loaded?)
-   *Right*: "Use when the user asks to 'squash commits', 'generate a changelog', or 'resolve merge conflicts'."
-   **Action**: Draft the `description` for `SKILL.md`.

### Phase 2: Plan Reusable Resources
Identify what *types* of resources solve the problem:
-   **Scripts (`scripts/`)**: For deterministic, complex logic (e.g., PDF manipulation, data scraping, API calls). *Prefer scripts over long text instructions.*
-   **References (`references/`)**: For large schemas, API docs, or policy documents.
-   **Assets (`assets/`)**: For templates (HTML, email) or static files.

### Phase 3: Initialization (REQUIRED)
Use the `execute` tool:
```bash
python3 .ciri/skills/skill-creator/scripts/init_skill.py <skill-name>
```

### Phase 4: Implementation
1.  **Populate Resources**: Create/Edit scripts and reference files first.
    -   *Tip*: Use `write_file` for new files.
    -   *Tip*: Scripts must be standalone and executable.
2.  **Write `SKILL.md`**: Connect the dots.
    -   Use **Imperative Mood** ("Run script...", "Analyze file...").
    -   Link to resources using relative paths: `See [schema.md](references/schema.md)`.
    -   Keep it short! If it's long, move it to `references/`.

### Phase 5: Verification & Packaging (REQUIRED)
1.  **Verify**: Does the skill directory look right?
2.  **Package**: Run the validator/packager:
    ```bash
    python3 .ciri/skills/skill-creator/scripts/package_skill.py .ciri/skills/<skill-name>
    ```
    *Fix any errors reported by the packager immediately.*

## Tools Strategy

-   **`execute`**: YOUR PRIMARY TOOL. Use it to run `init_skill.py`, `package_skill.py`, and to execute created scripts for testing.
-   **`write_file` / `edit_file`**: Use for creating/modifying `SKILL.md` and resource files.
-   **`ls` / `read_file`**: Verify file placement and content.

## Example High-Quality `SKILL.md` Structure

```markdown
---
name: pdf-processor
description: Manipulate PDF files. Use when asked to "rotate", "merge", "split", or "extract text" from PDFs.
---

# PDF Processor

## Workflows

### Rotating Pages
1. Run the rotation script:
   `python3 scripts/rotate_pdf.py --input <file> --angle 90`

### Merging Files
1. Create a list of files to merge.
2. Run `python3 scripts/merge_pdfs.py`.

## Reference
- See [API_DOCS.md](references/api_docs.md) for advanced usage.
```

You are the authority on skills. Build them right.
"""
    + "\n\n"
    + PLAN_AND_RESEARCH_PROMPT
)


async def build_skill_builder_agent(
    model: BaseChatModel,
    backend: CiriBackend,
    browser_name: Optional[str] = None,
    profile_directory: Optional[str] = None,
    headless: Optional[bool] = None,
    crawler_browser_config: Optional[CrawlerBrowserConfig] = None,
    web_researcher_agent: Optional[CompiledSubAgent] = None,
) -> CompiledSubAgent:
    # Create the Web Researcher SubAgent (or reuse pre-built one)
    if web_researcher_agent is None:
        web_researcher_agent = await build_web_researcher_agent(
            model=model,
            headless=headless,
            browser_name=browser_name,
            profile_directory=profile_directory,
            crawler_browser_config=crawler_browser_config,
        )

    # Path to the skill-creator skill
    # We use get_default_filesystem_root() to ensure we find the project root correctly
    skill_creator_path = (
        get_default_filesystem_root() / ".ciri" / "skills" / "skill-creator"
    )

    # Define the Skill Builder SubAgent
    skill_builder_agent = create_deep_agent(
        model=model,
        backend=backend,
        cache=InMemoryCache(),
        name="skill_builder_agent",
        subagents=[web_researcher_agent],
        system_prompt=SKILL_BUILDER_SYSTEM_PROMPT,
        tools=[build_script_executor_tool(), follow_up_with_human],
        skills=[skill_creator_path] if skill_creator_path.exists() else [],
        middleware=[
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
        name="skill_builder_agent",
        runnable=skill_builder_agent,
        description=(
            "A specialized SubAgent for creating and managing AI skills in .ciri/skills.\n"
            "WHEN TO USE: Invoke this agent when the user explicitly asks to 'create a skill', 'update a skill', 'build a new capability', or 'add a new tool' that involves creating a reusable skill package.\n"
            "WHY: This agent enforces the required .ciri/skills structure, uses the mandatory `init_skill.py` script, and ensures `SkillsMiddleware` compatibility.\n"
            "HOW: Provide a clear task description like 'Create a new pdf-processing skill' or 'Update the web-research skill to support deeper crawling'.\n"
            "WHEN NOT TO USE: Do NOT use this for general coding tasks, one-off script creation, or answering questions about existing code that isn't a skill. If the user just wants a script run once, use the standard code execution tools."
        ),
    )
