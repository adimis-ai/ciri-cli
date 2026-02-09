import os
import json
import asyncio
import sys
import uuid
import subprocess
from pathlib import Path
from typing import Optional, List, Any, Dict, Union, Iterable
import httpx

# Third-party imports
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.syntax import Syntax

# prompt_toolkit for @ autocomplete
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML

# LangGraph / LangChain imports
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessageChunk,
    ToolMessage,
)
from langgraph.types import Command
import aiosqlite

# Local imports
from .agent import Ciri, LLMConfig, ResumeCommand
from .db import CiriDatabase
from .serializers import CiriJsonPlusSerializer
from .utils import get_default_filesystem_root, get_app_data_dir
from dotenv import set_key

console = Console()


# ---------------------------------------------------------------------------
# Startup Initialization
# ---------------------------------------------------------------------------


def ensure_playwright_installed() -> None:
    """Ensure Playwright browsers are installed for web crawling.

    Uses ``python -m playwright`` so it works on any OS regardless of
    whether the ``playwright`` CLI script is on PATH.
    """
    try:
        # `playwright install` is idempotent — fast no-op if already present.
        # Use sys.executable so it works on any OS even if the `playwright`
        # CLI script isn't on PATH.
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            timeout=300,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            console.print(
                "[yellow]Warning: Playwright chromium installation failed. "
                "Web crawling may not work properly.[/yellow]"
            )
            if stderr:
                console.print(f"[dim]{stderr[:200]}[/dim]")
    except FileNotFoundError:
        console.print(
            "[yellow]Warning: Playwright package not found. "
            "Install it with: pip install playwright[/yellow]"
        )
    except subprocess.TimeoutExpired:
        console.print(
            "[yellow]Warning: Playwright installation timed out. "
            "Web crawling may not work properly.[/yellow]"
        )
    except Exception as e:
        console.print(
            f"[yellow]Warning: Failed to ensure Playwright browsers: {e}[/yellow]"
        )


# ---------------------------------------------------------------------------
# @ File/Folder & @skills: Autocomplete
# ---------------------------------------------------------------------------


class CiriCompleter(Completer):
    """Completer that supports:
    - `@` followed by a partial path → file/folder autocomplete from root_dir
    - `@skills:` followed by a partial name → skill name autocomplete from .ciri/skills/
    """

    SKILLS_PREFIX = "skills:"

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir.resolve()
        self._file_cache: Optional[List[str]] = None
        self._skills_cache: Optional[List[str]] = None

    # --- File scanning ---

    def _scan_files(self) -> List[str]:
        """Recursively scan root_dir and cache relative paths."""
        if self._file_cache is not None:
            return self._file_cache
        paths = []
        try:
            for p in sorted(self.root_dir.rglob("*")):
                parts = p.relative_to(self.root_dir).parts
                if any(part.startswith(".") for part in parts):
                    continue
                rel = str(p.relative_to(self.root_dir))
                if p.is_dir():
                    rel += "/"
                paths.append(rel)
        except PermissionError:
            pass
        self._file_cache = paths
        return self._file_cache

    # --- Skills scanning ---

    def _scan_skills(self) -> List[str]:
        """List available skill names from .ciri/skills/ directory."""
        if self._skills_cache is not None:
            return self._skills_cache
        skills_dir = self.root_dir / ".ciri" / "skills"
        skills: List[str] = []
        if skills_dir.is_dir():
            for d in sorted(skills_dir.iterdir()):
                if d.is_dir() and (d / "SKILL.md").is_file():
                    skills.append(d.name)
        self._skills_cache = skills
        return self._skills_cache

    # --- Cache management ---

    def invalidate_cache(self):
        """Clear all cached data so the next completion rescans."""
        self._file_cache = None
        self._skills_cache = None

    # --- Completions ---

    def get_completions(self, document: Document, complete_event):
        text_before = document.text_before_cursor

        # Find the last `@` that starts a reference
        at_idx = text_before.rfind("@")
        if at_idx == -1:
            return

        # Ensure @ is at start or preceded by whitespace (not part of an email)
        if at_idx > 0 and not text_before[at_idx - 1].isspace():
            return

        after_at = text_before[at_idx + 1 :]  # everything after @

        # --- @skills:<partial> → skill name completions ---
        if after_at.lower().startswith(self.SKILLS_PREFIX):
            partial = after_at[len(self.SKILLS_PREFIX) :]
            partial_lower = partial.lower()
            start_position = -len(partial)

            for skill in self._scan_skills():
                if skill.lower().startswith(partial_lower):
                    yield Completion(
                        skill,
                        start_position=start_position,
                        display=skill,
                        display_meta="skill",
                    )
            return

        # If user just typed "@" with no further text, also offer "skills:" as
        # a completion option so they can discover the feature.
        partial = after_at
        partial_lower = partial.lower()
        start_position = -len(partial)

        if self.SKILLS_PREFIX.startswith(partial_lower):
            yield Completion(
                self.SKILLS_PREFIX,
                start_position=start_position,
                display="skills:",
                display_meta="select skill",
            )

        # --- @<partial> → file/folder completions ---
        for path in self._scan_files():
            if path.lower().startswith(partial_lower):
                yield Completion(
                    path,
                    start_position=start_position,
                    display=path,
                    display_meta="dir" if path.endswith("/") else "file",
                )


class ModelCompleter(Completer):
    """Completer for OpenRouter models."""

    def __init__(self, models: List[str]):
        self.models = models

    def get_completions(
        self, document: Document, complete_event
    ) -> Iterable[Completion]:
        text = document.text_before_cursor.lstrip()
        for model in self.models:
            if model.startswith(text):
                yield Completion(model, start_position=-len(text))


async def fetch_openrouter_models() -> List[str]:
    """Fetch available models from OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return []

    url = "https://openrouter.ai/api/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                return [m["id"] for m in data.get("data", [])]
    except Exception as e:
        console.print(f"[dim red]Failed to fetch OpenRouter models: {e}[/dim red]")

    return []


def get_user_input(
    completer: Completer, prompt_text: str = "You> ", default_val: str = ""
) -> str:
    """Get user input with autocomplete via prompt_toolkit."""
    try:
        return pt_prompt(
            prompt_text,
            completer=completer,
            complete_while_typing=True,
            default=default_val,
        )
    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# Message Rendering Helpers
# ---------------------------------------------------------------------------


def render_tool_call(tool_call: dict) -> None:
    """Render a tool call with name and formatted arguments."""
    name = tool_call.get("name", "unknown")
    args = tool_call.get("args", {})
    try:
        args_str = json.dumps(args, indent=2, default=str)
        if len(args_str) > 500:
            args_str = args_str[:500] + "\n  ..."
        console.print(f"\n[bold yellow]Tool Call:[/bold yellow] [cyan]{name}[/cyan]")
        console.print(Syntax(args_str, "json", theme="monokai", line_numbers=False))
    except (TypeError, ValueError):
        console.print(
            f"\n[bold yellow]Tool Call:[/bold yellow] [cyan]{name}[/cyan]([dim]{args}[/dim])"
        )


def render_tool_message(message: ToolMessage) -> None:
    """Render a tool response message."""
    content = str(message.content)
    tool_name = getattr(message, "name", None) or message.tool_call_id or "tool"
    if len(content) > 300:
        content_preview = content[:300] + "..."
    else:
        content_preview = content
    console.print(f"\n[bold magenta]Tool Response ({tool_name}):[/bold magenta]")
    console.print(f"[dim]{content_preview}[/dim]")


def render_human_message(message) -> None:
    """Render a human message (rarely streamed back, but handle it)."""
    content = message.content if hasattr(message, "content") else str(message)
    console.print(f"\n[bold green]You:[/bold green] {content}")


# ---------------------------------------------------------------------------
# Human-in-the-Loop Interrupt Handling & Rendering
# ---------------------------------------------------------------------------


def _render_action_request(
    action_req: dict, action_num: int, show_description: bool = True
) -> None:
    """Beautifully render an action request from HumanInTheLoopMiddleware."""
    name = action_req.get("name", "unknown")
    args = action_req.get("args", {})
    description = action_req.get("description", "")

    # Header with action number and name
    console.print(
        f"\n  [bold cyan]⚙️  Action {action_num}:[/bold cyan] [bold]{name}[/bold]"
    )

    # Show description if available and requested
    if show_description and description:
        console.print(f"  [dim]Description:[/dim]")
        for line in description.split("\n"):
            console.print(f"    [dim]{line}[/dim]")

    # Show args as formatted JSON
    if args:
        console.print(f"  [dim]Arguments:[/dim]")
        try:
            args_str = json.dumps(args, indent=2, default=str)
            # Indent for nested rendering
            args_lines = args_str.split("\n")
            for line in args_lines:
                console.print(f"    {line}")
        except (TypeError, ValueError):
            console.print(f"    {args}")


def _render_hitl_interrupt(
    val: dict, action_requests: list, review_configs: list
) -> None:
    """Beautifully render a HumanInTheLoopMiddleware interrupt (tool approval)."""
    console.print(
        Panel(
            "[bold cyan]🔒 Tool Execution Approval Required[/bold cyan]\n"
            "[dim]The model wants to execute the following tools. Please review:[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # Render each action request
    for i, req in enumerate(action_requests, 1):
        _render_action_request(req, i, show_description=True)

    # Show allowed decisions per tool
    console.print("\n  [dim]Allowed decisions per tool:[/dim]")
    for req, rc in zip(action_requests, review_configs):
        tool_name = req.get("name", "unknown")
        allowed = rc.get("allowed_decisions", [])
        allowed_str = ", ".join(f"[cyan]{d}[/cyan]" for d in allowed)
        console.print(f"    • [bold]{tool_name}[/bold]: {allowed_str}")

    console.print()


def _render_follow_up_interrupt(val: dict) -> None:
    """Beautifully render a human_follow_up interrupt."""
    question = val.get("question", "")
    options = val.get("options")

    console.print(
        Panel(
            f"[bold cyan]❓ Clarification Needed[/bold cyan]\n{question}",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    if options:
        console.print("\n  [dim]Available options:[/dim]")
        for i, opt in enumerate(options, 1):
            console.print(f"    [cyan]{i}.[/cyan] {opt}")
        console.print()


async def handle_interrupts(graph, state, config, completer: CiriCompleter) -> None:
    """Handle all pending interrupts from the graph state with beautiful rendering."""
    snapshot = state.values
    interrupts = snapshot.get("__interrupt__", [])

    if not interrupts:
        return

    for interrupt_val in interrupts:
        # interrupt_val may be an Interrupt object or a dict
        if hasattr(interrupt_val, "value"):
            val = interrupt_val.value
        elif isinstance(interrupt_val, dict):
            val = interrupt_val.get("value")
        else:
            continue

        if not val:
            continue

        # Route to appropriate handler based on interrupt type
        if isinstance(val, dict) and val.get("type") == "human_follow_up":
            _render_follow_up_interrupt(val)
            await _handle_follow_up(graph, val, config, completer)

        elif isinstance(val, dict) and "action_requests" in val:
            # HumanInTheLoopMiddleware interrupt
            action_requests = val.get("action_requests", [])
            review_configs = val.get("review_configs", [])
            _render_hitl_interrupt(val, action_requests, review_configs)
            await _handle_tool_approval(graph, val, config, completer)

        else:
            # Unknown interrupt type — show beautifully formatted raw value
            console.print(
                Panel(
                    "[bold cyan]❔ Unknown Interrupt Type[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
            try:
                interrupt_json = json.dumps(val, indent=2, default=str)
                console.print(
                    Syntax(
                        interrupt_json,
                        "json",
                        theme="monokai",
                        line_numbers=False,
                        padding=1,
                    )
                )
            except (TypeError, ValueError):
                console.print(f"[dim]{val}[/dim]")

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pt_prompt(
                    "\n[cyan]Your response[/cyan]> ", completer=completer
                ),
            )
            await run_graph(graph, Command(resume=response), config, completer)


async def _handle_follow_up(
    graph, val: dict, config: dict, completer: CiriCompleter
) -> None:
    """Handle a human_follow_up interrupt with beautiful option rendering."""
    question = val.get("question", "")
    options = val.get("options")

    # Rendering is done in handle_interrupts via _render_follow_up_interrupt
    # This handler just collects and processes the response

    if options:
        # Get user choice
        choice = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pt_prompt(
                "  [cyan]Enter option number or text[/cyan]> ", completer=completer
            ),
        )

        # If user typed a number, resolve to the option text
        try:
            idx = int(choice.strip()) - 1
            if 0 <= idx < len(options):
                response = options[idx]
            else:
                console.print(
                    "[red]Invalid option number. Using your input as-is.[/red]"
                )
                response = choice
        except ValueError:
            response = choice
    else:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pt_prompt("  [cyan]Your response[/cyan]> ", completer=completer),
        )

    await run_graph(graph, Command(resume=response), config, completer)


async def _handle_tool_approval(
    graph, val: dict, config: dict, completer: CiriCompleter
) -> None:
    """Handle a tool approval interrupt (approve / reject / edit)."""
    action_requests = val.get("action_requests", [])
    review_configs = val.get("review_configs", [])

    # Rendering is done in handle_interrupts via _render_hitl_interrupt
    # This handler just collects and processes the decisions

    # Show decision options summary
    console.print("\n  [dim]Actions available per tool:[/dim]")
    allowed_per_tool = {}
    for req, rc in zip(action_requests, review_configs):
        tool_name = req.get("name", "unknown")
        allowed = set(rc.get("allowed_decisions", []))
        allowed_per_tool[tool_name] = allowed
        allowed_str = " / ".join(f"[cyan]{d}[/cyan]" for d in sorted(allowed))
        console.print(f"    • [bold]{tool_name}[/bold]: {allowed_str}")

    # Get user decision (same for all or per-tool)
    decision = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: pt_prompt(
            "\n  [cyan]Decision (approve/reject/edit)[/cyan]> ",
            completer=completer,
        ),
    )
    decision = decision.strip().lower()

    if decision == "approve":
        decisions = [{"type": "approve"} for _ in action_requests]
        command = Command(resume={"decisions": decisions})

    elif decision == "reject":
        reason = await asyncio.get_event_loop().run_in_executor(
            None, lambda: pt_prompt("  [cyan]Reason for rejection[/cyan]> ")
        )
        decisions = [{"type": "reject", "message": reason} for _ in action_requests]
        command = Command(resume={"decisions": decisions})

    elif decision == "edit":
        console.print("\n  [bold cyan]✏️  Editing Actions[/bold cyan]")
        decisions = []
        for i, req in enumerate(action_requests, 1):
            name = req.get("name", "unknown")
            args = req.get("arguments", req.get("args", {}))

            console.print(f"\n    [bold]Action {i}: {name}[/bold]")
            console.print(f"    [dim]Current arguments:[/dim]")
            try:
                args_str = json.dumps(args, indent=2, default=str)
                for line in args_str.split("\n"):
                    console.print(f"      {line}")
            except (TypeError, ValueError):
                console.print(f"      {args}")

            new_args_str = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pt_prompt(
                    "    [cyan]New arguments (JSON, or press Enter to keep)[/cyan]> "
                ),
            )
            if new_args_str.strip():
                try:
                    new_args = json.loads(new_args_str)
                    console.print(f"    [green]✓ Arguments updated[/green]")
                except json.JSONDecodeError:
                    console.print(
                        "[red]    ✗ Invalid JSON. Keeping original arguments.[/red]"
                    )
                    new_args = args
            else:
                new_args = args

            decisions.append(
                {
                    "type": "edit",
                    "edited_action": {"name": name, "args": new_args},
                }
            )

        command = Command(resume={"decisions": decisions})

    else:
        console.print(f"  [red]Unknown decision '{decision}'. Using 'approve'.[/red]")
        decisions = [{"type": "approve"} for _ in action_requests]
        command = Command(resume={"decisions": decisions})

    await run_graph(graph, command, config, completer)


# ---------------------------------------------------------------------------
# Dual-Mode Streaming
# ---------------------------------------------------------------------------


async def run_graph(graph, inputs, config, completer: Optional[CiriCompleter] = None):
    """Run the graph with dual stream mode and handle output + interrupts."""
    current_ai_message = ""
    prefix_printed = False
    # Accumulate tool calls: {id: {name, args_str}} from tool_call_chunks
    pending_tool_calls: dict[str, dict[str, str]] = {}
    rendered_tool_call_ids: set[str] = set()

    def _flush_pending_tool_calls():
        """Render any accumulated tool calls that are complete."""
        for tc_id, tc_data in list(pending_tool_calls.items()):
            if tc_id in rendered_tool_call_ids:
                continue
            # Parse accumulated args string into dict
            args_str = tc_data.get("args_str", "")
            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                args = {}
            render_tool_call({"name": tc_data["name"], "args": args})
            rendered_tool_call_ids.add(tc_id)

    with console.status("[bold blue]Thinking...", spinner="dots") as status:
        async for stream_type, chunk in graph.astream(
            inputs, config, stream_mode=["updates", "messages"]
        ):
            # ----- Messages stream: token-by-token AI output -----
            if stream_type == "messages":
                message, metadata = chunk

                # AI message chunks (streaming tokens)
                if isinstance(message, BaseMessageChunk):
                    if message.content:
                        # Flush any pending tool calls before printing text
                        if pending_tool_calls:
                            status.stop()
                            _flush_pending_tool_calls()

                        status.stop()
                        if not prefix_printed:
                            console.print("\n[bold cyan]CIRI:[/bold cyan] ", end="")
                            prefix_printed = True

                        content = message.content
                        if isinstance(content, str):
                            console.print(content, end="")
                            current_ai_message += content
                        elif isinstance(content, list):
                            for part in content:
                                if (
                                    isinstance(part, dict)
                                    and part.get("type") == "text"
                                ):
                                    text = part.get("text", "")
                                    console.print(text, end="")
                                    current_ai_message += text

                    # Accumulate tool call chunks incrementally
                    if (
                        hasattr(message, "tool_call_chunks")
                        and message.tool_call_chunks
                    ):
                        for tc_chunk in message.tool_call_chunks:
                            tc_id = tc_chunk.get("id")
                            tc_name = tc_chunk.get("name")
                            tc_args = tc_chunk.get("args", "")
                            if tc_id:
                                if tc_id not in pending_tool_calls:
                                    pending_tool_calls[tc_id] = {
                                        "name": tc_name or "",
                                        "args_str": "",
                                    }
                                if tc_name and not pending_tool_calls[tc_id]["name"]:
                                    pending_tool_calls[tc_id]["name"] = tc_name
                                if tc_args:
                                    pending_tool_calls[tc_id]["args_str"] += tc_args

                # Complete Tool response messages
                elif isinstance(message, ToolMessage):
                    # Flush pending tool calls before showing tool response
                    if pending_tool_calls:
                        status.stop()
                        _flush_pending_tool_calls()
                    status.stop()
                    render_tool_message(message)
                    status.start()

                # Human messages (echoed back in stream, rare)
                elif isinstance(message, HumanMessage):
                    status.stop()
                    render_human_message(message)

            # ----- Updates stream: node state transitions -----
            elif stream_type == "updates":
                # Updates come as {node_name: state_update_dict}
                # We can use these for progress indication
                if isinstance(chunk, dict):
                    for node_name, node_state in chunk.items():
                        if node_name == "__interrupt__":
                            # Interrupt arrived via updates — will be handled after loop
                            pass

    # Flush any remaining pending tool calls at end of stream
    _flush_pending_tool_calls()

    if current_ai_message:
        console.print()  # newline after streamed response

    # Check for pending interrupts after streaming ends
    state = await graph.aget_state(config)
    if state.next:
        if completer:
            await handle_interrupts(graph, state, config, completer)
        else:
            # Fallback without completer (shouldn't normally happen)
            _completer = CiriCompleter(get_default_filesystem_root())
            await handle_interrupts(graph, state, config, _completer)


# ---------------------------------------------------------------------------
# Thread Management Commands
# ---------------------------------------------------------------------------


def handle_threads_command(
    db: CiriDatabase, current_thread_id: str, completer: CiriCompleter
) -> Optional[str]:
    """List all threads and optionally switch. Returns new thread_id or None."""
    threads = db.list_threads()
    if not threads:
        console.print("[yellow]No threads found.[/yellow]")
        return None

    table = Table(title="Threads", show_lines=True)
    table.add_column("#", style="bold", width=4)
    table.add_column("Title", style="cyan")
    table.add_column("Created", style="dim")
    table.add_column("Active", style="green", width=6)

    for i, t in enumerate(threads, 1):
        active = "*" if t["id"] == current_thread_id else ""
        created = t["created_at"][:19].replace("T", " ")
        table.add_row(str(i), t["title"], created, active)

    console.print(table)
    console.print("[dim]Enter a number to switch, or press Enter to stay.[/dim]")

    choice = pt_prompt("Switch to> ", completer=completer).strip()
    if not choice:
        return None

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(threads):
            selected = threads[idx]
            if selected["id"] == current_thread_id:
                console.print("[dim]Already on this thread.[/dim]")
                return None
            console.print(f"[green]Switched to thread:[/green] {selected['title']}")
            return selected["id"]
        else:
            console.print("[red]Invalid selection.[/red]")
    except ValueError:
        console.print("[red]Invalid input.[/red]")
    return None


def handle_new_thread_command(db: CiriDatabase) -> str:
    """Create a new thread and return its id."""
    thread = db.create_thread()
    console.print(f"[green]New thread created:[/green] {thread['id'][:8]}...")
    return thread["id"]


def handle_delete_thread_command(db: CiriDatabase, current_thread_id: str) -> str:
    """Delete the current thread and return a new thread id to switch to."""
    thread = db.get_thread(current_thread_id)
    title = thread["title"] if thread else current_thread_id[:8]
    if not Confirm.ask(f"Delete thread [cyan]{title}[/cyan]?"):
        return current_thread_id

    db.delete_thread(current_thread_id)
    console.print(f"[red]Deleted thread:[/red] {title}")

    # Switch to most recent remaining thread, or create a new one
    threads = db.list_threads()
    if threads:
        new_id = threads[0]["id"]
        console.print(f"[green]Switched to thread:[/green] {threads[0]['title']}")
        return new_id
    else:
        new_thread = db.create_thread()
        console.print(f"[green]Created new thread:[/green] {new_thread['id'][:8]}...")
        return new_thread["id"]


# ---------------------------------------------------------------------------
# Main Interactive Loop
# ---------------------------------------------------------------------------


async def interactive_chat():
    """Main interactive chat loop for CIRI CLI."""
    console.print(
        Panel(
            "[bold cyan]CIRI[/bold cyan] - Desktop Personal AI Copilot\n[dim]Initializing...[/dim]",
            border_style="cyan",
        )
    )

    # Ensure Playwright is installed for web crawling
    ensure_playwright_installed()

    # Ensure required environment variables are set
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("[yellow]OPENROUTER_API_KEY not found in environment.[/yellow]")
        api_key = Prompt.ask(
            "Please enter your [bold]OpenRouter API Key[/bold]", password=True
        )
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
            # Save to global .env
            global_env = get_app_data_dir() / ".env"
            set_key(str(global_env), "OPENROUTER_API_KEY", api_key)
            console.print("[green]API Key set and saved globally.[/green]")
        else:
            console.print("[red]API Key is required to continue. Exiting.[/red]")
            return

    model = os.getenv("CIRI_MODEL")
    available_models = await fetch_openrouter_models()
    model_completer = ModelCompleter(available_models)

    if not model:
        model = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: get_user_input(
                model_completer,
                "Select LLM Model (Tab for options)> ",
                "openai/gpt-5-mini",
            ),
        )
        os.environ["CIRI_MODEL"] = model
        console.print(f"[green]Model set to {model} for this session.[/green]")

    # Initialize encrypted database
    db = CiriDatabase()

    llm_config = LLMConfig(model=model)
    ciri_app = Ciri(llm_config=llm_config)

    async with aiosqlite.connect(db.db_path) as conn:
        checkpointer = AsyncSqliteSaver(conn, serde=CiriJsonPlusSerializer())
        # Compile the agent graph
        graph = ciri_app.compile(checkpointer=checkpointer)

        # Create initial thread
        thread = db.create_thread()
        current_thread_id = thread["id"]
        config = {"configurable": {"thread_id": current_thread_id}}
        is_first_message = True

        # Build file tree completer for @ autocomplete
        root_dir = get_default_filesystem_root()
        completer = CiriCompleter(root_dir)

        console.print(f"[green]Ready![/green] Root directory: [bold]{root_dir}[/bold]")
        console.print(
            "[dim]Tip: @ for file paths, @skills: for skills. Commands: /threads, /new-thread, /delete-thread[/dim]\n"
        )

        try:
            while True:
                try:
                    # Re-scan skills before each prompt so newly generated skills appear
                    completer._skills_cache = None

                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: get_user_input(completer)
                    )

                    stripped = user_input.strip()

                    if stripped.lower() in ("exit", "quit", "bye"):
                        console.print("[cyan]Goodbye![/cyan]")
                        break

                    if not stripped:
                        continue

                    # --- Thread commands ---
                    if stripped == "/threads":
                        new_id = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: handle_threads_command(
                                db, current_thread_id, completer
                            ),
                        )
                        if new_id:
                            current_thread_id = new_id
                            config = {"configurable": {"thread_id": current_thread_id}}
                            is_first_message = False
                        continue

                    if stripped == "/new-thread":
                        current_thread_id = handle_new_thread_command(db)
                        config = {"configurable": {"thread_id": current_thread_id}}
                        is_first_message = True
                        continue

                    if stripped == "/delete-thread":
                        current_thread_id = (
                            await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: handle_delete_thread_command(
                                    db, current_thread_id
                                ),
                            )
                        )
                        config = {"configurable": {"thread_id": current_thread_id}}
                        is_first_message = True
                        continue

                    if stripped.startswith("/model"):
                        parts = stripped.split(maxsplit=1)
                        if len(parts) > 1:
                            new_model = parts[1]
                        else:
                            new_model = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: get_user_input(model_completer, "New model> "),
                            )

                        if new_model:
                            os.environ["CIRI_MODEL"] = new_model
                            # Re-initialize agent with new model
                            llm_config = LLMConfig(model=new_model)
                            ciri_app = Ciri(llm_config=llm_config)
                            graph = ciri_app.compile(checkpointer=checkpointer)
                            console.print(
                                f"[green]Model switched to {new_model}[/green]"
                            )
                        continue

                    # --- Normal message ---
                    # Auto-title thread from first message
                    if is_first_message:
                        title = stripped[:50] + ("..." if len(stripped) > 50 else "")
                        db.rename_thread(current_thread_id, title)
                        is_first_message = False

                    db.touch_thread(current_thread_id)
                    inputs = {"messages": [HumanMessage(content=user_input)]}
                    await run_graph(graph, inputs, config, completer)

                except KeyboardInterrupt:
                    console.print(
                        "\n[yellow]Interrupted by user. Type 'exit' to quit.[/yellow]"
                    )
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
        finally:
            db.close()


def main():
    """EntryPoint for the CIRI CLI."""
    asyncio.run(interactive_chat())


if __name__ == "__main__":
    main()
