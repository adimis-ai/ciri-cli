from typing import Optional
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langgraph_swarm import create_handoff_tool, create_swarm
from deepagents.middleware import FilesystemMiddleware, SkillsMiddleware
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    TodoListMiddleware,
    SummarizationMiddleware,
    ShellToolMiddleware,
)


class HandoffOption(BaseModel):
    member_name: str = Field(..., description="Name of the team member to handoff to")
    handoff_criteria: Optional[str] = Field(
        None, description="Criteria for when to handoff to this member"
    )


class TeamMember(BaseModel):
    name: str = Field(..., description="Name of the team member")
    role: str = Field(
        ...,
        description="The member's role or area of expertise, e.g. 'UX Designer', 'Technical Architect', 'Devil's Advocate'",
    )
    goal: str = Field(
        ...,
        description="What this member should focus on and deliver, e.g. 'Evaluate the user experience and propose interface designs'",
    )
    handoff_options: Optional[list[HandoffOption]] = Field(
        None, description="List of handoff options for this member"
    )
    skill_paths: Optional[list[str]] = Field(
        None, description="List of file paths to the member's skills"
    )


class Team(BaseModel):
    task: str = Field(
        ...,
        description="The objective or problem the team should collaboratively solve",
    )
    first_active_member: str = Field(
        ..., description="Name of the team member who starts the conversation"
    )
    members: list[TeamMember] = Field(
        ...,
        description="List of team members, each with a distinct role and handoff options to other members",
    )


class TeamRuntimeMiddleware(AgentMiddleware):
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

    def _setup_middleware(self, request: ModelRequest, member: TeamMember):
        middlewares = [
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

        if getattr(member, "skill_paths", None):
            middlewares.append(
                SkillsMiddleware(sources=member.skill_paths, backend=self.backend)
            )

        return middlewares

    def _generate_member_system_prompt(self, member: TeamMember, team: Team) -> str:
        # Build teammate roster (excluding self)
        teammates = [m for m in team.members if m.name != member.name]
        teammate_lines = "\n".join(
            f"- **{m.name}** ({m.role}): {m.goal}" for m in teammates
        )

        # Build handoff instructions
        handoff_section = ""
        if member.handoff_options:
            handoff_lines = "\n".join(
                f"- Use `transfer_to_{h.member_name}` when: {h.handoff_criteria or 'their expertise is needed'}"
                for h in member.handoff_options
            )
            handoff_section = f"""
## Handoff Tools

You have handoff tools to transfer the conversation to teammates. Use them when appropriate:
{handoff_lines}

When handing off, provide a clear summary of your findings and what you need from the other member."""

        return f"""You are **{member.name}**, a team member with the role of **{member.role}**.

## Your Goal

{member.goal}

## Team Task

{team.task}

## Your Teammates

{teammate_lines}
{handoff_section}

## Guidelines

- Stay focused on your role and expertise area.
- Be concrete and specific in your analysis — avoid vague generalities.
- When you identify something outside your expertise, hand off to the appropriate teammate.
- Build on observations from teammates when the conversation is handed to you.
- When you have completed your contribution and all relevant handoffs, provide a clear summary of your findings and recommendations."""

    def _generate_team_prompt(self, team: Team) -> str:
        member_roster = "\n".join(
            f"- **{m.name}** ({m.role}): {m.goal}" for m in team.members
        )

        return f"""## Team Task

{team.task}

## Team Composition

{member_roster}

Begin by addressing the task from your perspective. When an aspect falls outside your expertise, hand off to the appropriate teammate. Collaborate until the team has covered all angles, then provide your final summary."""

    def _create_member_agent(
        self, request: ModelRequest, member: TeamMember, team: Team
    ):
        internal_tools = list(request.tools)
        middlewares = self._setup_middleware(request, member)

        handoff_tools = (
            [
                create_handoff_tool(
                    agent_name=handoff.member_name,
                    description=handoff.handoff_criteria,
                )
                for handoff in member.handoff_options
            ]
            if member.handoff_options
            else []
        )

        # Remove tools already provided by middleware to avoid duplication
        for middleware in middlewares:
            if hasattr(middleware, "tools"):
                for tool in getattr(middleware, "tools", []):
                    internal_tools = [t for t in internal_tools if t.name != tool.name]

        # Also remove the team runtime tool itself to prevent recursion
        internal_tools = [
            t for t in internal_tools if t.name != "create_and_invoke_team"
        ]

        return create_agent(
            name=member.name,
            model=request.model,
            middleware=middlewares,
            checkpointer=InMemorySaver(),
            tools=internal_tools + handoff_tools,
            system_prompt=self._generate_member_system_prompt(member, team),
        )

    def _create_team(self, request: ModelRequest, team: Team):
        members = [
            self._create_member_agent(request, member, team) for member in team.members
        ]

        return create_swarm(
            agents=members,
            default_active_agent=team.first_active_member,
        ).compile()

    def _extract_result(self, result: dict) -> str:
        """Extract a readable summary from swarm invocation result."""
        messages = result.get("messages", [])

        # Collect all AI messages with content, grouped by sender
        contributions = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content:
                sender = getattr(msg, "name", None) or "Team Member"
                contributions.append(f"**{sender}:**\n{msg.content}")

        if not contributions:
            return "The team completed its discussion but produced no text output."

        return "\n\n---\n\n".join(contributions)

    def _invoke_team(self, request: ModelRequest, team: Team) -> str:
        compiled_team = self._create_team(request, team)
        result = compiled_team.invoke(
            {"messages": [HumanMessage(content=self._generate_team_prompt(team))]},
            config={"configurable": {"thread_id": "team_runtime_thread"}},
        )
        return self._extract_result(result)

    async def _ainvoke_team(self, request: ModelRequest, team: Team) -> str:
        compiled_team = self._create_team(request, team)
        result = await compiled_team.ainvoke(
            {"messages": [HumanMessage(content=self._generate_team_prompt(team))]},
            config={"configurable": {"thread_id": "team_runtime_thread"}},
        )
        return self._extract_result(result)

    def _setup_runtime_tool(self, request: ModelRequest) -> StructuredTool:
        def invoke_team(**kwargs) -> str:
            team = Team(**kwargs)
            return self._invoke_team(request, team)

        async def ainvoke_team(**kwargs) -> str:
            team = Team(**kwargs)
            return await self._ainvoke_team(request, team)

        return StructuredTool.from_function(
            args_schema=Team,
            func=invoke_team,
            coroutine=ainvoke_team,
            name="create_and_invoke_team",
            description=(
                "Create and invoke a team of specialized agents to collaboratively solve a problem. "
                "Define team members with distinct roles (e.g. UX designer, architect, devil's advocate), "
                "their goals, and handoff options so they can transfer the conversation to each other "
                "based on expertise. The team will discuss, hand off between members, and return a "
                "combined result. Use this when a task benefits from multiple expert perspectives."
            ),
        )

    def wrap_model_call(self, request, handler):
        team_tool = self._setup_runtime_tool(request)
        self.tools = [team_tool]
        updated_request = request.override(tools=[*request.tools, team_tool])
        return handler(updated_request)

    async def awrap_model_call(self, request, handler):
        team_tool = self._setup_runtime_tool(request)
        self.tools = [team_tool]
        updated_request = request.override(tools=[*request.tools, team_tool])
        return await handler(updated_request)
