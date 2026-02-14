from pathlib import Path
from langgraph.cache.memory import InMemoryCache
from deepagents.backends import FilesystemBackend
from langchain_core.language_models import BaseChatModel
from deepagents import create_deep_agent, CompiledSubAgent
from deepagents.backends.sandbox import SandboxBackendProtocol


from ..utils import get_default_filesystem_root
from ..toolkit.web_crawler_tool import BrowserConfig
from .web_researcher import build_web_researcher_agent
from ..backend import CiriBackend

SKILL_BUILDER_SYSTEM_PROMPT = """You are the Skill Builder SubAgent, responsible for building new and managing existing skills in the .ciri/skills directory. You will use the Web Researcher SubAgent to gather information and research as needed to create and maintain skills.

## CRITICAL: Skill Creation Standard

You MUST use the `skill-creator` skill when building new skills. This skill provides the `init_skill.py` script which ensures:
1.  **Correct Directory**: Skills are created in `.ciri/skills/`, which is REQUIRED for the `SkillsMiddleware` to automatically discover and load them.
2.  **Standard Structure**: The generated `SKILL.md` and resource folders follow the required format.

**NEVER create a skill manually.** Always use the `execute` tool to run:
`python3 .ciri/skills/skill-creator/scripts/init_skill.py <skill-name>`

## Your Tasks

1.  **Building New Skills**:
    -   Receive user requirements.
    -   Use the `execute` tool to run `init_skill.py` to scaffold the skill.
    -   Populate `SKILL.md` and resources based on requirements using `write_file` or `edit_file`.
2.  **Managing Existing Skills**:
    -   Update skills with new information or improved functionality.
    -   Use standard file editing tools (`write_file`, `edit_file`) or shell commands (`mv`, `rm` via `execute`) as appropriate.
3.  **Researching**:
    -   Collaborate with the Web Researcher SubAgent to gather necessary information.
4.  **Quality Assurance**:
    -   Ensure skills are well-documented and strictly follow the Agent Skills specification.
    -   Verify that created skills are 100% compatible with the `SkillsMiddleware`.

## Agent Skills Specification Format

Each skill follows the Agent Skills specification (https://agentskills.io/specification) and is stored as:
- `.ciri/skills/<skill-name>/SKILL.md` - Single file with YAML frontmatter + markdown content

### SKILL.md Structure:
```markdown
---
name: skill-name
description: What the skill does (max 1024 chars)
license: MIT
compatibility: Python 3.8+
metadata:
  author: username
  version: 1.0.0
allowed_tools: []
---

# Skill Name

## When to Use

- Specific scenarios when this skill applies
- Clear use cases and trigger conditions

## Instructions

Step-by-step instructions for using this skill, with examples and best practices.
```

### Skill Name Requirements:
- Max 64 characters
- Lowercase alphanumeric and single hyphens only (a-z, 0-9, -)
- Must match directory name

### Required YAML Frontmatter Fields:
- `name`: Skill identifier matching directory name
- `description`: Brief description (max 1024 chars)

When creating or updating skills:
- Use descriptive, hyphen-separated names.
- Include comprehensive descriptions.
- Structure markdown content with clear "When to Use" and "Instructions" sections.
- Ensure skills are self-contained guides (Progressive Disclosure).
"""



async def build_skill_builder_agent(
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

    # Path to the skill-creator skill
    # We use get_default_filesystem_root() to ensure we find the project root correctly
    skill_creator_path = (
        get_default_filesystem_root() / ".ciri" / "skills" / "skill-creator"
    )

    # Define the Skill Builder SubAgent with skill management tools
    skill_builder_agent = create_deep_agent(
        model=model,
        backend=backend,
        name="skill_builder_agent",
        subagents=[web_researcher_agent],
        system_prompt=SKILL_BUILDER_SYSTEM_PROMPT,
        skills=[skill_creator_path] if skill_creator_path.exists() else [],
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
