import os
import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchResults
from deepagents.middleware import FilesystemMiddleware
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    TodoListMiddleware,
    SummarizationMiddleware,
    ShellToolMiddleware,
)
from src.toolkit.web_crawler_tool import build_web_crawler_tool

logger = logging.getLogger(__name__)


class SkillGeneratorArgs(BaseModel):
    skill_name: str = Field(
        ...,
        description=(
            "The name of the skill to generate. Must be lowercase alphanumeric with "
            "hyphens only (e.g. 'web-research', 'data-analysis'). Cannot start or end "
            "with a hyphen and no consecutive hyphens allowed."
        ),
    )
    skill_description: str = Field(
        ...,
        description=(
            "A clear, detailed description of what this skill should enable the agent "
            "to do. Include the domain, key workflows, and expected outputs. This will "
            "guide the research and skill file generation."
        ),
    )
    additional_context: Optional[str] = Field(
        default=None,
        description=(
            "Any additional context, constraints, or references that should be "
            "considered when generating this skill."
        ),
    )


SKILL_GENERATOR_SYSTEM_PROMPT = """You are an expert AI Skill Designer. Your job is to research a topic thoroughly and produce a high-quality SKILL.md file that follows the Agent Skills specification. You can both create new skills and update existing ones.

## Your Workflow

1. **Understand the Request**: Analyze the skill name, description, and any additional context provided.
2. **Check Existing Content**: If an existing SKILL.md is provided, read it carefully and understand its current structure, content, and coverage.
3. **Research the Topic**: Use your available search and crawling tools to gather comprehensive, up-to-date information about the skill's domain. Combine web search results with your internal knowledge.
4. **Design / Update the Skill**:
   - For **new skills**: Structure the skill with clear workflows, best practices, examples, and actionable instructions.
   - For **existing skills**: Preserve what's already good, improve weak sections, add missing information, update outdated content, and incorporate the new description/context. Do NOT discard existing valuable content — merge and enhance.
5. **Write the SKILL.md**: Create the skill directory (if needed) and write the SKILL.md file in the correct format.
6. **Verify**: Read back the created/updated file to confirm it was written correctly.

## SKILL.md Format

The SKILL.md file MUST follow this exact format:

```markdown
---
name: <skill-name>
description: <what the skill does, max 1024 chars>
license: MIT
---

# <Skill Title>

## When to Use
- <Condition 1 when this skill applies>
- <Condition 2 when this skill applies>

## Prerequisites
- <Any requirements or setup needed>

## Workflow
1. <Step 1>
2. <Step 2>
3. <Step 3>

## Best Practices
- <Practice 1>
- <Practice 2>

## Examples

### Example 1: <Title>
<Detailed example with context and expected outcome>

## Common Pitfalls
- <Pitfall 1 and how to avoid it>
- <Pitfall 2 and how to avoid it>

## References
- <Source 1>
- <Source 2>
```

## Important Rules

- The skill `name` in frontmatter MUST match the directory name exactly.
- Name must be lowercase alphanumeric with single hyphens (pattern: `^[a-z0-9]+(-[a-z0-9]+)*$`).
- Description must be max 1024 characters.
- The SKILL.md must be comprehensive enough for an AI agent to follow without prior domain knowledge.
- Include real, actionable workflows - not just theoretical descriptions.
- Write the skill to `/.ciri/skills/<skill-name>/SKILL.md`.
- If the skill already exists and its current content is provided, update it by merging new research with the existing content. Preserve valuable existing sections while improving and extending them.

## Research Guidelines

- Use `simple_web_search` for quick factual lookups and discovering relevant URLs.
- Use `web_crawler` to extract detailed content from specific web pages you discover.
- Synthesize information from multiple sources for comprehensive coverage.
- Combine web research with your internal training knowledge for the most accurate result.
- Always cite sources in the References section when using web-sourced information.
"""


class SkillGeneratorMiddleware(AgentMiddleware):
    """Middleware that provides a skill generation tool to the agent.

    Creates an internal research agent equipped with web search (DuckDuckGo),
    web crawling (crawl4ai), filesystem access, shell tools, and summarization
    capabilities. The agent researches the requested topic and produces a
    SKILL.md file in the .ciri/skills/ directory following the Agent Skills
    specification supported by SkillsMiddleware.
    """

    def __init__(
        self,
        root_dir: str,
        backend: FilesystemBackend,
        *,
        shell_config: Optional[dict] = None,
    ):
        super().__init__()
        self.backend = backend
        self.root_dir = root_dir
        self.shell_config = shell_config or {}

    def _get_existing_skills(self) -> list[str]:
        """List existing skill names from .ciri/skills/ directory."""
        skills_dir = os.path.join(self.root_dir, ".ciri", "skills")
        if not os.path.isdir(skills_dir):
            return []
        return [
            d
            for d in os.listdir(skills_dir)
            if os.path.isdir(os.path.join(skills_dir, d))
            and os.path.isfile(os.path.join(skills_dir, d, "SKILL.md"))
        ]

    def _read_existing_skill(self, skill_name: str) -> Optional[str]:
        """Read the full content of an existing SKILL.md file."""
        skill_path = os.path.join(
            self.root_dir, ".ciri", "skills", skill_name, "SKILL.md"
        )
        if not os.path.isfile(skill_path):
            return None
        try:
            with open(skill_path, "r", encoding="utf-8") as f:
                return f.read()
        except (OSError, IOError) as e:
            logger.warning("Failed to read existing skill %s: %s", skill_name, e)
            return None

    def _setup_middleware(self, request: ModelRequest):
        """Set up middleware stack for the internal skill generator agent."""
        return [
            TodoListMiddleware(),
            SummarizationMiddleware(model=request.model),
            FilesystemMiddleware(self.backend),
            ShellToolMiddleware(
                workspace_root=self.root_dir,
                env=self.shell_config.get("env", None),
                execution_policy=self.shell_config.get("execution_policy", None),
                redaction_rules=self.shell_config.get("redaction_rules", None),
                shell_command=self.shell_config.get("shell_command", None),
                shutdown_commands=self.shell_config.get("shutdown_commands", None),
                startup_commands=self.shell_config.get("startup_commands", None),
            ),
        ]

    def _build_research_tools(self) -> list:
        """Build the research tools for the skill generator agent."""
        tools = [
            DuckDuckGoSearchResults(name="simple_web_search"),
            build_web_crawler_tool(),
        ]
        return tools

    def _create_generator_agent(self, request: ModelRequest):
        """Create the internal skill generator agent."""
        middlewares = self._setup_middleware(request)
        research_tools = self._build_research_tools()

        # Remove tools already provided by middleware to avoid duplication
        internal_tools = list(research_tools)
        for middleware in middlewares:
            if hasattr(middleware, "tools"):
                for tool in getattr(middleware, "tools", []):
                    internal_tools = [
                        t for t in internal_tools if t.name != tool.name
                    ]

        return create_agent(
            name="skill_generator",
            model=request.model,
            middleware=middlewares,
            checkpointer=InMemorySaver(),
            tools=internal_tools,
            system_prompt=SKILL_GENERATOR_SYSTEM_PROMPT,
        )

    def _build_generation_prompt(
        self, args: SkillGeneratorArgs, existing_content: Optional[str] = None
    ) -> str:
        """Build the prompt for the skill generator agent."""
        existing_skills = self._get_existing_skills()
        existing_list = (
            ", ".join(f"`{s}`" for s in existing_skills) if existing_skills else "none"
        )

        is_update = existing_content is not None

        if is_update:
            prompt = (
                f"Update the existing skill with the following details:\n\n"
                f"**Skill Name:** `{args.skill_name}`\n"
                f"**Updated Description:** {args.skill_description}\n"
            )
        else:
            prompt = (
                f"Generate a new skill with the following details:\n\n"
                f"**Skill Name:** `{args.skill_name}`\n"
                f"**Description:** {args.skill_description}\n"
            )

        if args.additional_context:
            prompt += f"**Additional Context:** {args.additional_context}\n"

        prompt += f"\n**Existing skills in .ciri/skills/:** {existing_list}\n\n"

        if is_update:
            prompt += (
                f"The skill `{args.skill_name}` already exists. Here is its current "
                f"SKILL.md content:\n\n"
                f"---BEGIN EXISTING SKILL.md---\n"
                f"{existing_content}\n"
                f"---END EXISTING SKILL.md---\n\n"
                f"Read the existing content carefully. Research the topic to find any "
                f"updates or improvements, then rewrite the SKILL.md at "
                f"`/.ciri/skills/{args.skill_name}/SKILL.md` — preserving valuable "
                f"existing content while incorporating the new description, fixing "
                f"outdated information, and adding any missing sections."
            )
        else:
            prompt += (
                f"Research the topic thoroughly using web search and crawling, then "
                f"create the skill directory and SKILL.md file at "
                f"`/.ciri/skills/{args.skill_name}/SKILL.md`."
            )

        return prompt

    def _invoke_generator(
        self,
        request: ModelRequest,
        args: SkillGeneratorArgs,
        existing_content: Optional[str] = None,
    ) -> str:
        """Invoke the skill generator agent synchronously."""
        agent = self._create_generator_agent(request)
        prompt = self._build_generation_prompt(args, existing_content)

        result = agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={"configurable": {"thread_id": "skill_generator_thread"}},
        )

        return self._extract_result(result)

    async def _ainvoke_generator(
        self,
        request: ModelRequest,
        args: SkillGeneratorArgs,
        existing_content: Optional[str] = None,
    ) -> str:
        """Invoke the skill generator agent asynchronously."""
        agent = self._create_generator_agent(request)
        prompt = self._build_generation_prompt(args, existing_content)

        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={"configurable": {"thread_id": "skill_generator_thread"}},
        )

        return self._extract_result(result)

    def _extract_result(self, result: dict) -> str:
        """Extract a readable summary from the generator agent result."""
        messages = result.get("messages", [])

        contributions = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content:
                contributions.append(msg.content)

        if not contributions:
            return "The skill generator completed but produced no output."

        # Return the last AI message as it's typically the summary
        return contributions[-1]

    def _setup_generator_tool(self, request: ModelRequest) -> StructuredTool:
        """Create the StructuredTool that wraps the skill generator agent."""

        def generate_skill(**kwargs) -> str:
            args = SkillGeneratorArgs(**kwargs)

            # Check if skill already exists — if so, read it for updating
            existing_content = self._read_existing_skill(args.skill_name)
            if existing_content is not None:
                logger.info(
                    "Skill '%s' exists — reading current content for update.",
                    args.skill_name,
                )

            return self._invoke_generator(request, args, existing_content)

        async def agenerate_skill(**kwargs) -> str:
            args = SkillGeneratorArgs(**kwargs)

            # Check if skill already exists — if so, read it for updating
            existing_content = self._read_existing_skill(args.skill_name)
            if existing_content is not None:
                logger.info(
                    "Skill '%s' exists — reading current content for update.",
                    args.skill_name,
                )

            return await self._ainvoke_generator(request, args, existing_content)

        return StructuredTool.from_function(
            args_schema=SkillGeneratorArgs,
            func=generate_skill,
            coroutine=agenerate_skill,
            name="generate_skill",
            description=(
                "Research a topic using web search, web crawling, and internal knowledge, "
                "then generate or update a skill file (SKILL.md) in the .ciri/skills/ "
                "directory. The generated skill follows the Agent Skills specification "
                "and can be loaded by SkillsMiddleware for progressive disclosure to "
                "agents. Use this when you want to learn a new capability by creating a "
                "reusable skill that persists across sessions. If the skill already "
                "exists, it will be read fully and updated with new research and context "
                "while preserving valuable existing content."
            ),
        )

    def wrap_model_call(self, request, handler):
        generator_tool = self._setup_generator_tool(request)
        self.tools = [generator_tool]
        updated_request = request.override(tools=[*request.tools, generator_tool])
        return handler(updated_request)

    async def awrap_model_call(self, request, handler):
        generator_tool = self._setup_generator_tool(request)
        self.tools = [generator_tool]
        updated_request = request.override(tools=[*request.tools, generator_tool])
        return await handler(updated_request)
