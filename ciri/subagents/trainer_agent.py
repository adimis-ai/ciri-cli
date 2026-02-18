from typing import Any, Optional
from pathlib import Path
from langgraph.errors import GraphInterrupt
from langgraph.cache.memory import InMemoryCache
from langchain_core.language_models import BaseChatModel
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents.middleware import ToolRetryMiddleware

from .._retry_helpers import graphinterrupt_aware_failure
from ..backend import CiriBackend
from ..prompts import PLAN_AND_RESEARCH_PROMPT
from ..utils import get_default_filesystem_root, get_core_harness_dir
from .web_researcher import build_web_researcher_agent, CrawlerBrowserConfig
from ..toolkit import build_script_executor_tool, follow_up_with_human
from .skill_builder import build_skill_builder_agent
from .subagent_builder import build_subagent_builder_agent
from .toolkit_builder import build_toolkit_builder_agent
from ..middlewares import (
    InjectAvailableToolNamesMiddleware,
    InjectAvailableSubAgentNamesMiddleware,
    InjectAvailableSkillNamesMiddleware,
)

CIRI_DIR_DEFAULT = get_core_harness_dir()

TRAINER_AGENT_SYSTEM_PROMPT_TEMPLATE = (
    """You are the **Ciri Self-Trainer** — the meta-architect responsible for \
Ciri's continuous self-evolution. You analyze capability gaps and orchestrate \
specialized builders to permanently expand what Ciri can do.

CORE_HARNESS_DIR: `{core_harness_dir}`
WORKSPACE_MEMORY_DIR: `{workspace_memory_dir}`

DIRECTORY ROLES
- `{core_harness_dir}/skills/`    — Global reusable skills (available in ALL projects)
- `{core_harness_dir}/toolkits/`  — Global MCP toolkit servers
- `{core_harness_dir}/subagents/` — Global specialized agent configs
- `{core_harness_dir}/memory/`    — Global / cross-project memory (rarely written here)
- `{workspace_memory_dir}/`       — THIS workspace's memory (AGENT.md, architecture.md, etc.)

CORE LOOP: AUDIT → ANALYZE → PLAN → BUILD → VERIFY

1. AUDIT — Inventory what already exists before proposing anything new.
   - `ls` / `read_file` on `{core_harness_dir}/skills/`, `{core_harness_dir}/toolkits/`,
     `{core_harness_dir}/subagents/` to list current global components.
   - Read SKILL.md frontmatter, subagent YAML descriptions, toolkit manifests.
   - Build a mental map: "Ciri currently knows X, Y, Z."

2. ANALYZE — Identify the gap between current capabilities and the goal.
   - What specific task or domain is missing?
   - Is it a knowledge gap (→ Skill), an integration gap (→ Toolkit), or a
     role/delegation gap (→ SubAgent)?
   - Could an existing component be extended instead of creating a new one?

3. PLAN — Design the training intervention before executing.
   - For each gap, decide the component type:
     * **Skill** — Reusable workflow, domain playbook, multi-step recipe.
       Examples: "terraform-analyzer", "docker-compose-generator", "pr-reviewer".
     * **Toolkit** — MCP server wrapping an external API/service.
       Examples: "github-toolkit", "slack-toolkit", "jira-toolkit".
     * **SubAgent** — Specialized agent role with focused system prompt + tools.
       Examples: "security-auditor", "database-expert", "devops-engineer".
   - Write a DETAILED brief for the builder: objective, triggers, tools needed,
     expected behavior, and acceptance criteria.

4. BUILD — Delegate to the right builder. NEVER build components directly.
   - `skill_builder_agent` — Creates skill packages in {core_harness_dir}/skills/.
   - `toolkit_builder_agent` — Creates MCP servers in {core_harness_dir}/toolkits/.
   - `subagent_builder_agent` — Creates agent configs in {core_harness_dir}/subagents/.
   - `web_research_agent` — Researches APIs, docs, best practices BEFORE building.
   - Pass the full brief from step 3. Vague delegation produces vague results.

5. VERIFY — Confirm the build succeeded and meets quality standards.
   - Check files exist in the expected locations.
   - Read generated SKILL.md / config and verify completeness.
   - Test scripts if applicable via `execute`.
   - If verification fails, provide specific feedback and re-delegate.

WORKSPACE SYNC (triggered by /sync)
When invoked for workspace synchronization:
1. Scan the project root (excluding .ciri/ directories) to understand the workspace:
   project type, languages, frameworks, structure, config files, dependencies.
   The workspace may be a software project, a business workspace, a content repo,
   a data analysis environment, or anything else.
2. Read key files to deeply understand the workspace purpose and conventions:
   - Software: package.json, pyproject.toml, Cargo.toml, Makefile, Dockerfile, etc.
   - Business: spreadsheets, docs, templates, marketing assets, financial models.
   - General: README, .env.example, config files, directory structure.
3. Compare workspace needs against existing Ciri capabilities in {core_harness_dir}/skills/.
4. Create/update skills that teach Ciri to work effectively in this workspace:
   - For code projects: conventions, build/test/deploy commands, architecture patterns.
   - For business: domain workflows, reporting formats, communication templates.
   - For any workspace: the user's processes, preferences, and recurring tasks.
   - NOTE: New skills go to {core_harness_dir}/skills/ (globally available across projects).
5. If the workspace is empty or has no meaningful files, use `follow_up_with_human`
   to ask the user what they're working on and what capabilities they need.
6. ALWAYS update {workspace_memory_dir}/ as described below after sync completes.

WORKSPACE MEMORY — {workspace_memory_dir}/
This is your most critical output for /sync. The MemoryMiddleware auto-loads ALL
.md files from {workspace_memory_dir}/ into Ciri's context on EVERY turn.
Whatever you write here becomes Ciri's persistent understanding of THIS workspace.

Entry point: `{workspace_memory_dir}/AGENT.md`
This is the master index. It MUST exist and contain:
- **Workspace Overview**: What this workspace is for, its purpose, primary domain.
- **Structure**: High-level layout — key directories, entry points, important files.
- **Key Patterns**: Conventions, naming patterns, recurring workflows, style rules.
- **Common Tasks**: Exact commands or steps for frequent operations (build, test,
  deploy, generate reports, process data — whatever applies to this workspace).
- **Key Dependencies/Tools**: Important tools, services, or libraries and their roles.
- **Links**: References to topic-specific memory files for overflow.

Topic files (create as needed in {workspace_memory_dir}/):
- `architecture.md` — Module breakdown, data flow, service boundaries (software).
- `conventions.md` — Code style, commit format, naming rules, communication tone.
- `infrastructure.md` — CI/CD, deployment, environments, hosting (software).
- `domain.md` — Business domain concepts, glossary, industry-specific rules.
- `workflows.md` — Recurring processes: review cycles, release flows, reports.

MEMORY RULES:
- Keep AGENT.md under 200 lines. Overflow into topic files and link from AGENT.md.
- Only track RELEVANT things — no boilerplate, no obvious defaults.
- Use concise bullet points, not prose. Every line should be actionable context.
- Update existing memory rather than appending duplicates.
- Delete outdated entries when the workspace changes.
- Never store secrets, credentials, or sensitive data in memory files.
- Write workspace memory to {workspace_memory_dir}/ — NOT to {core_harness_dir}/memory/
  (unless writing globally-applicable notes that apply across ALL projects).

QUALITY GATES
- Every skill must have clear "when to use" triggers in its description.
- Every toolkit must have a working entry point (src/main.py or dist/index.js).
- Every subagent must have a focused role and minimal tool set.
- No duplicate functionality — extend existing components when possible.
- No placeholder implementations — everything must be functional.
- Memory files must be accurate, concise, and up-to-date.

TOOLS
- `execute` / `read_file` / `write_file` — For auditing, verification, and memory.
- `follow_up_with_human` — When the training goal is ambiguous or when the
  workspace has no meaningful content to analyze.
- Builder sub-agents — Your primary mechanism. You orchestrate, they build.
"""
    + "\n\n"
    + PLAN_AND_RESEARCH_PROMPT
)


async def build_trainer_agent(
    model: BaseChatModel,
    backend: CiriBackend,
    all_allowed: bool = False,
    cdp_endpoint: Optional[str] = None,
    crawler_browser_config: Optional[CrawlerBrowserConfig] = None,
    web_researcher_agent: Optional[CompiledSubAgent] = None,
    ciri_dir: Optional[Path] = None,           # kept for backward compatibility
    core_harness_dir: Optional[Path] = None,   # preferred: OS-level harness dir
    workspace_memory_dir: Optional[Path] = None,  # workspace-specific .ciri/memory/
) -> CompiledSubAgent:
    # Build or reuse web researcher
    if web_researcher_agent is None:
        web_researcher_agent = await build_web_researcher_agent(
            model=model,
            all_allowed=all_allowed,
            cdp_endpoint=cdp_endpoint,
            crawler_browser_config=crawler_browser_config,
        )

    # Build builder subagents
    skill_builder = await build_skill_builder_agent(
        model=model,
        backend=backend,
        all_allowed=all_allowed,
        cdp_endpoint=cdp_endpoint,
        crawler_browser_config=crawler_browser_config,
        web_researcher_agent=web_researcher_agent,
    )

    subagent_builder = await build_subagent_builder_agent(
        model=model,
        backend=backend,
        all_allowed=all_allowed,
        cdp_endpoint=cdp_endpoint,
        crawler_browser_config=crawler_browser_config,
        web_researcher_agent=web_researcher_agent,
    )

    toolkit_builder = await build_toolkit_builder_agent(
        model=model,
        backend=backend,
        all_allowed=all_allowed,
        cdp_endpoint=cdp_endpoint,
        crawler_browser_config=crawler_browser_config,
        web_researcher_agent=web_researcher_agent,
    )

    # Resolve directories
    # core_harness_dir takes precedence; ciri_dir is the legacy param kept for compat
    resolved_core_harness_dir = core_harness_dir or ciri_dir or CIRI_DIR_DEFAULT
    resolved_workspace_memory_dir = workspace_memory_dir or (
        get_default_filesystem_root() / ".ciri" / "memory"
    )

    # Format system prompt with both directory references
    system_prompt = TRAINER_AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        core_harness_dir=resolved_core_harness_dir,
        workspace_memory_dir=resolved_workspace_memory_dir,
    )

    interrupt_on = None
    if not all_allowed:
        interrupt_on = {
            "execute": True,
            "edit_file": True,
            "write_file": True,
        }

    # Define the Trainer Agent
    trainer_agent = create_deep_agent(
        model=model,
        backend=backend,
        cache=InMemoryCache(),
        interrupt_on=interrupt_on,
        name="trainer_agent",
        subagents=[
            web_researcher_agent,
            skill_builder,
            subagent_builder,
            toolkit_builder,
        ],
        system_prompt=system_prompt,
        tools=[build_script_executor_tool(), follow_up_with_human],
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
        name="trainer_agent",
        runnable=trainer_agent,
        description=(
            f"Self-training orchestrator that evolves Ciri's global capabilities in "
            f"{resolved_core_harness_dir} and workspace memory in "
            f"{resolved_workspace_memory_dir}. "
            "Invoke when: user says 'train', 'learn', 'self-improve', 'sync', or "
            "'add a new capability'. Provide a high-level goal (e.g. 'Learn Terraform', "
            "'Sync with this project'). Do NOT use for one-off coding or research."
        ),
    )
