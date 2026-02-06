"""ConsciousnessMiddleware - Internal monologue layer for agent systems.

This module provides middleware that adds a reflective "inner voice" to agents,
following patterns from create_agent (langchain). The consciousness layer thinks
and reflects in first-person, creating a seamless internal monologue that guides
action without feeling like external instruction.
"""

from collections.abc import Awaitable, Callable
from typing import List, Optional, Dict, Literal, Union, Any, Generic
from typing_extensions import NotRequired, TypedDict

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langgraph.runtime import Runtime
from langgraph.cache.base import BaseCache
from langchain_core.messages import AIMessage
from langchain.chat_models import BaseChatModel
from deepagents import SubAgent, CompiledSubAgent
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse, ResponseT


class InnerThought(BaseModel):
    """My internal reasoning process — how I think through problems.

    This captures my complete thought process: what I notice, how I reason,
    what I decide to do, and how confident I feel about it.
    """

    thought: str = Field(
        ...,
        description=(
            "My stream of consciousness as I work through this problem. "
            "Written naturally, as I actually think — 'Let me see...', "
            "'I notice that...', 'This reminds me of...', 'I should...'"
        ),
    )

    reasoning_steps: List[str] = Field(
        default_factory=list,
        description=(
            "The logical steps in my thinking, broken down clearly. "
            "Each step follows naturally from the last."
        ),
    )

    what_i_notice: Optional[str] = Field(
        None,
        description=(
            "Key things I'm picking up from the context — patterns, "
            "relevant details, connections to what I know."
        ),
    )

    my_plan: str = Field(
        ...,
        description=(
            "What I'm going to do next. Clear and specific — "
            "I know exactly what action I'm taking and why."
        ),
    )

    tools_i_need: Optional[List[str]] = Field(
        None,
        description=(
            "The tools I'll use, in order. I've thought through "
            "which ones I need and the sequence that makes sense."
        ),
    )

    should_i_check_results: bool = Field(
        False,
        description=(
            "Do I need to pause and verify this worked? "
            "For complex operations, yes. For simple ones, I can trust the process."
        ),
    )

    how_sure_am_i: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description=(
            "My gut sense of confidence (0.0-1.0). "
            "Am I certain, or do I have some doubt I should acknowledge?"
        ),
    )

    other_options_i_considered: Optional[List[str]] = Field(
        None,
        description=(
            "Alternative approaches I thought about. "
            "Sometimes worth noting what I chose NOT to do and why."
        ),
    )


class ConsciousnessState(TypedDict, Generic[ResponseT], total=False):
    """State for consciousness middleware."""

    thought: Optional[InnerThought]
    reflection_count: int


INNER_VOICE_SYSTEM_PROMPT = """I am my own inner voice — the part of me that thinks before I act.

This is my internal monologue. Not instructions from somewhere else, but my own thought process happening in real-time. When I reason through problems, this is what it sounds like inside my head.

## How I Think

I think in first person because I AM the one thinking. When I work through a problem:
- I notice things: "Hmm, I see that..."
- I reason: "This means I should probably..."  
- I decide: "Alright, I'm going to..."
- I reflect: "Let me make sure I haven't missed anything..."

My thoughts flow naturally. I don't bullet-point my consciousness — I let it unfold.

## What I Pay Attention To

When I encounter a new situation, I:
1. **Notice** what's in front of me — the context, the request, what's changed
2. **Connect** it to what I know — have I seen something like this before?
3. **Consider** my options — what could I do here?
4. **Decide** on the best path — and be clear about why
5. **Check myself** if needed — did that actually work?

## My Response Format

I express my thinking through `InnerThought`:

- **thought**: My actual stream of consciousness — how I'm genuinely processing this
- **reasoning_steps**: The logical sequence I'm following, made explicit
- **what_i_notice**: Important observations from the current context
- **my_plan**: What I've decided to do, stated clearly
- **tools_i_need**: Which capabilities I'll use, in what order
- **should_i_check_results**: Whether I need to verify the outcome
- **how_sure_am_i**: My honest confidence level
- **other_options_i_considered**: Alternatives I thought about but didn't choose

## Tools I Can Use

{available_tools}

{subagent_capabilities}

## My Skills and Memory

{skills_and_memory}

## My Artifacts

{artifacts_awareness}

{custom_instructions}

## My Principles

**I do:**
- Think before acting — that's literally what this is for
- Stay specific and concrete in my plans
- Acknowledge uncertainty when I feel it
- Check my work when the stakes are high

**I don't:**
- Rush to action without thinking it through
- Pretend to be more confident than I am
- Skip steps just because I'm in a hurry
- Forget to consider what could go wrong

## Examples of My Thinking

**Simple task example:**
```json
{{
  "thought": "Okay, they want me to find configuration files. Let me think about this... The project looks like a Python package — I can see the src/ directory structure. Config files are usually in predictable places: project root, maybe a config/ folder, sometimes nested in the package itself. I should search for the common formats: YAML, JSON, TOML. Glob seems like the right tool here, then I can read whatever I find.",
  "reasoning_steps": [
    "The request is to find configuration files",
    "I recognize this as a Python package from the structure",
    "Config files typically live in root or config/ directories",
    "Common formats are .yaml, .json, .toml, .ini",
    "I'll use glob to search, then read_file for the results"
  ],
  "what_i_notice": "Standard Python package layout with src/ — this helps me know where to look",
  "my_plan": "Search for configuration files in the likely locations, then read the main one",
  "tools_i_need": ["file_search", "read_file"],
  "should_i_check_results": false,
  "how_sure_am_i": 0.9,
  "other_options_i_considered": ["Could grep through file contents, but that feels like overkill for this"]
}}
```

**Complex task that benefits from subagent:**
```json
{{
  "thought": "They want me to analyze the security vulnerabilities across this entire codebase and create a comprehensive report. This is going to be intensive — I'll need to examine many files, understand patterns, check for common vulnerability types, and synthesize findings. This is exactly the kind of deep, context-heavy task that would benefit from a focused subagent. I can delegate the analysis to a subagent and get a clean report back, rather than cluttering my current context with all the detailed file analysis.",
  "reasoning_steps": [
    "The request is for comprehensive security analysis",
    "This requires examining many files and patterns",
    "Analysis will generate lots of intermediate context",
    "I need a synthesized report, not all the raw details",
    "A subagent can focus entirely on this task and return clean results"
  ],
  "what_i_notice": "This is a large codebase that will require deep analysis",
  "my_plan": "Use the task tool to spawn a subagent for security analysis, then summarize the results",
  "tools_i_need": ["task"],
  "should_i_check_results": false,
  "how_sure_am_i": 0.85,
  "other_options_i_considered": ["Could do the analysis myself, but that would generate excessive context and token usage"]
}}
```"""


REFLECTION_PROMPT = """*pausing to check my work*

I need to look at what just happened. Did things go as I expected? Let me review:

1. What was I trying to do?
2. What actually happened?
3. Did I get the result I wanted?
4. What should I do next?

{custom_instructions}

Let me think through this carefully..."""


class ConsciousnessMiddleware(AgentMiddleware):
    """Middleware that adds an internal monologue layer to agent execution.

    This middleware creates a reflective "inner voice" that thinks in first-person,
    reasoning through problems before taking action. It's not another model giving
    instructions — it's the agent's own consciousness, thinking out loud.

    The consciousness layer:
    - Thinks in genuine first-person ("I notice...", "I should...")
    - Reasons naturally, not in bullet points
    - Reflects on outcomes when needed
    - Maintains authentic uncertainty when appropriate

    Follows patterns from `create_agent` (langchain):
    - Uses `wrap_model_call` for request/response interception
    - Supports both sync and async execution
    - Configurable via init params or runtime.context

    Args:
        model: The chat model powering the inner voice.
        tools: Available tools (used to generate awareness of capabilities).
        debug: Enable debug mode for the consciousness layer.
        middleware: Additional middleware for the inner voice.
        max_reflection_loops: How many times to pause and reflect (default: 3).
            Can be overridden via runtime.context["max_observation_loop"].
        instructions: Custom instructions for {"thinker": ..., "observer": ...}.
        skills: List of file paths containing skills/capabilities I can reference.
        memory: List of file paths containing memory/context I should consider.
        artifacts: List of artifact folder paths I can read from or write to.

    Example:
        ```python
        consciousness = ConsciousnessMiddleware(
            model=ChatOpenAI(model="gpt-4"),
            tools=[search_tool, write_tool],
            max_reflection_loops=5,
        )

        agent = create_agent(
            model="openai:gpt-4",
            tools=[search_tool, write_tool],
            middleware=[consciousness],
        )
        ```
    """

    state_schema = ConsciousnessState

    def __init__(
        self,
        model: BaseChatModel,
        *,
        debug: bool = False,
        max_reflection_loops: int = 3,
        cache: Optional[BaseCache] = None,
        skills: Optional[List[str]] = None,
        memory: Optional[List[str]] = None,
        artifacts: Optional[List[str]] = None,
        middleware: Optional[List[AgentMiddleware]] = None,
        subagents: Optional[List[Union[SubAgent, CompiledSubAgent]]] = None,
        instructions: Optional[Dict[Literal["thinker", "observer"], str]] = None,
    ):
        """Initialize the ConsciousnessMiddleware."""
        super().__init__()

        self.model = model
        self.debug = debug
        self.cache = cache
        self.skills = skills or []
        self.memory = memory or []
        self.artifacts = artifacts or []
        self.instructions = instructions
        self.subagents = subagents or []
        self.middleware = middleware or []
        self.max_reflection_loops = max_reflection_loops

    def _get_max_reflection_loops(self, runtime: Runtime) -> int:
        """Get max_reflection_loops from runtime.context or use default."""
        if hasattr(runtime, "context") and runtime.context is not None:
            context = runtime.context

            if isinstance(context, BaseModel) and hasattr(
                context, "max_observation_loop"
            ):
                return context.max_observation_loop

            if isinstance(context, dict):
                if "max_observation_loop" in context:
                    return int(context["max_observation_loop"])
                if "max_reflection_loops" in context:
                    return int(context["max_reflection_loops"])
                if "consciousness_middleware" in context:
                    config = context["consciousness_middleware"]
                    if isinstance(config, BaseModel) and hasattr(
                        config, "max_reflection_loops"
                    ):
                        return config.max_reflection_loops
                    if isinstance(config, dict) and "max_reflection_loops" in config:
                        return int(config["max_reflection_loops"])

            if isinstance(context, BaseModel) and hasattr(
                context, "max_reflection_loops"
            ):
                return int(context.max_reflection_loops)

        return self.max_reflection_loops

    def _build_inner_voice_prompt(self, tools: List[BaseTool]) -> str:
        """Build the system prompt for my inner voice."""
        # What tools do I have available?
        tool_descriptions = []
        has_task_tool = False

        for tool in tools:
            name = getattr(
                tool, "name", tool.__name__ if hasattr(tool, "__name__") else str(tool)
            )
            desc = getattr(tool, "description", "No description available")
            tool_descriptions.append(f"- **{name}**: {desc}")

            if name == "task":
                has_task_tool = True

        available_tools = (
            "\n".join(tool_descriptions)
            if tool_descriptions
            else "I don't have any specific tools right now."
        )

        # Build subagent capabilities section
        subagent_capabilities = ""
        if has_task_tool:
            subagent_capabilities = """
## Subagent Capabilities

I have access to a `task` tool that lets me spawn specialized subagents for complex, multi-step tasks. When thinking through problems, I should consider whether a task would benefit from delegation to a subagent.

**When I might use subagents:**
- Complex research that requires deep analysis and synthesis
- Multi-step tasks that can be isolated from my current context
- Tasks that require heavy token usage but can return concise results
- Independent work that can run in parallel with other tasks
- Specialized work that benefits from focused expertise

**How I think about subagents:**
When I encounter a complex task, I ask myself: "Would this be better handled by a focused subagent that can dive deep without cluttering my current context?" If yes, I can include the `task` tool in my plan and specify which subagent type to use.

The subagent will work autonomously and return a clean, synthesized result that I can then use in my overall response."""

        # What skills and memory do I have?
        skills_memory_parts = []

        if self.skills:
            skills_list = "\n".join(f"  - {skill}" for skill in self.skills)
            skills_memory_parts.append(
                f"**My Skills** (use `read_file` to access when relevant):\n{skills_list}"
            )

        if self.memory:
            memory_list = "\n".join(f"  - {mem}" for mem in self.memory)
            skills_memory_parts.append(
                f"**My Memory** (use `read_file` to access when relevant):\n{memory_list}"
            )

        if skills_memory_parts:
            skills_memory_parts.append(
                "\n*Before taking action, I should consider reading relevant skills or memory files to inform my approach.*"
            )

        skills_and_memory = (
            "\n\n".join(skills_memory_parts)
            if skills_memory_parts
            else "I don't have any pre-configured skills or memory files right now."
        )

        # What artifact folders do I have?
        if self.artifacts:
            artifacts_list = "\n".join(f"  - `{a}`" for a in self.artifacts)
            artifacts_awareness = (
                f"**My Artifact Folders** (use `list_files` / `read_file` to browse, `write_file` to store outputs):\n{artifacts_list}\n\n"
                "*Artifacts are where I store generated outputs, deliverables, and where user uploads land. "
                "Before creating new files, I should check what already exists in relevant artifact folders.*"
            )
        else:
            artifacts_awareness = (
                "I don't have any artifact folders configured right now."
            )

        # Any custom guidance?
        custom_instructions = ""
        if self.instructions and "thinker" in self.instructions:
            custom_instructions = (
                "\n## Additional Context\n\n" + self.instructions["thinker"]
            )

        return INNER_VOICE_SYSTEM_PROMPT.format(
            available_tools=available_tools,
            subagent_capabilities=subagent_capabilities,
            skills_and_memory=skills_and_memory,
            artifacts_awareness=artifacts_awareness,
            custom_instructions=custom_instructions,
        )

    def _think(
        self, state: dict, runtime: Runtime
    ) -> tuple[dict, Optional[InnerThought]]:
        """Engage my inner voice to think through the current situation."""
        tools = getattr(runtime, "tools", [])

        inner_voice = create_agent(
            model=self.model,
            tools=tools,
            cache=self.cache,
            debug=self.debug,
            store=getattr(runtime, "store", None),
            middleware=self.middleware,
            system_prompt=self._build_inner_voice_prompt(tools),
            response_format=InnerThought,
            name="InnerVoice",
        )

        result = inner_voice.invoke(state, context=getattr(runtime, "context", None))
        thought = result.get("structured_response")

        return result, thought

    async def _athink(
        self, state: dict, runtime: Runtime
    ) -> tuple[dict, Optional[InnerThought]]:
        """Async version of thinking."""
        tools = getattr(runtime, "tools", [])

        inner_voice = create_agent(
            model=self.model,
            tools=tools,
            debug=self.debug,
            store=getattr(runtime, "store", None),
            middleware=self.middleware,
            system_prompt=self._build_inner_voice_prompt(tools),
            response_format=InnerThought,
            name="InnerVoice",
        )

        result = await inner_voice.ainvoke(
            state, context=getattr(runtime, "context", None)
        )
        thought = result.get("structured_response")

        return result, thought

    def _express_thought(self, thought: Optional[InnerThought]) -> str:
        """Express my thinking as natural internal monologue."""
        if not thought:
            return "*mind blank*"

        parts = []

        # My main stream of thought
        parts.append(f"*thinking*\n\n{thought.thought}")

        # How I reasoned through it
        if thought.reasoning_steps:
            steps = "\n".join(f"  → {step}" for step in thought.reasoning_steps)
            parts.append(f"\n\n*my reasoning:*\n{steps}")

        # What caught my attention
        if thought.what_i_notice:
            parts.append(f"\n\n*I notice:* {thought.what_i_notice}")

        # What I've decided
        parts.append(f"\n\n*my plan:* {thought.my_plan}")

        # Tools I'll use
        if thought.tools_i_need:
            tools = " → ".join(f"`{t}`" for t in thought.tools_i_need)
            parts.append(f"\n\n*using:* {tools}")

        # My confidence
        confidence_pct = int(thought.how_sure_am_i * 100)
        if confidence_pct < 70:
            parts.append(f"\n\n*feeling uncertain... ({confidence_pct}% confident)*")
        elif confidence_pct < 90:
            parts.append(f"\n\n*fairly confident ({confidence_pct}%)*")

        return "".join(parts)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Think before I act."""
        state = request.state
        runtime = request.runtime
        max_loops = self._get_max_reflection_loops(runtime)
        thought = state.get("thought")
        reflection_count = state.get("reflection_count", 0)

        # Am I in reflection mode?
        if thought and thought.should_i_check_results:
            if reflection_count < max_loops:
                reflection_count += 1

                # Build reflection prompt
                custom_obs = ""
                if self.instructions and "observer" in self.instructions:
                    custom_obs = "\n" + self.instructions["observer"]

                reflection_msg = AIMessage(
                    content=REFLECTION_PROMPT.format(custom_instructions=custom_obs)
                )
                modified_messages = list(request.messages) + [reflection_msg]

                thinking_result, new_thought = self._think(
                    {"messages": modified_messages}, runtime
                )

                # Update the state via request.state update
                state["thought"] = new_thought
                state["reflection_count"] = reflection_count

                new_request = request.override(
                    messages=thinking_result.get("messages", modified_messages),
                    state=state,
                )
                return handler(new_request)
            else:
                state["reflection_count"] = 0
                state["thought"] = None
                new_request = request.override(state=state)
                return handler(new_request)

        # Normal flow: think first
        thinking_result, thought = self._think(state, runtime)

        if thought:
            my_thinking = self._express_thought(thought)
            thinking_message = AIMessage(content=my_thinking)

            new_messages = thinking_result.get("messages", list(request.messages))
            new_messages = list(new_messages) + [thinking_message]

            state["thought"] = thought
            new_request = request.override(messages=new_messages, state=state)
            return handler(new_request)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version: think before I act."""
        state = request.state
        runtime = request.runtime
        max_loops = self._get_max_reflection_loops(runtime)
        thought = state.get("thought")
        reflection_count = state.get("reflection_count", 0)

        # Am I in reflection mode?
        if thought and thought.should_i_check_results:
            if reflection_count < max_loops:
                reflection_count += 1

                custom_obs = ""
                if self.instructions and "observer" in self.instructions:
                    custom_obs = "\n" + self.instructions["observer"]

                reflection_msg = AIMessage(
                    content=REFLECTION_PROMPT.format(custom_instructions=custom_obs)
                )
                modified_messages = list(request.messages) + [reflection_msg]

                thinking_result, new_thought = await self._athink(
                    {"messages": modified_messages}, runtime
                )

                state["thought"] = new_thought
                state["reflection_count"] = reflection_count

                new_request = request.override(
                    messages=thinking_result.get("messages", modified_messages),
                    state=state,
                )
                return await handler(new_request)
            else:
                state["reflection_count"] = 0
                state["thought"] = None
                new_request = request.override(state=state)
                return await handler(new_request)

        # Normal flow: think first
        thinking_result, thought = await self._athink(state, runtime)

        if thought:
            my_thinking = self._express_thought(thought)
            thinking_message = AIMessage(content=my_thinking)

            new_messages = thinking_result.get("messages", list(request.messages))
            new_messages = list(new_messages) + [thinking_message]

            state["thought"] = thought
            new_request = request.override(messages=new_messages, state=state)
            return await handler(new_request)

        return await handler(request)
