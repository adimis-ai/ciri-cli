import os
import json
import asyncio
import sys
import uuid
from pathlib import Path
from typing import Optional, List, Any, Dict, Union

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
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessageChunk,
    ToolMessage,
)
from langgraph.types import Command

# Local imports
from .agent import Ciri, LLMConfig, ResumeCommand
from .db import CiriDatabase
from .utils import get_default_filesystem_root
from .serializers import CiriJsonPlusSerializer

console = Console()


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

        after_at = text_before[at_idx + 1:]  # everything after @

        # --- @skills:<partial> → skill name completions ---
        if after_at.lower().startswith(self.SKILLS_PREFIX):
            partial = after_at[len(self.SKILLS_PREFIX):]
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


def get_user_input(completer: CiriCompleter) -> str:
    """Get user input with @ file and @skills: autocomplete via prompt_toolkit."""
    try:
        return pt_prompt(
            "You> ",
            completer=completer,
            complete_while_typing=False,  # only complete on Tab
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
        console.print(f"\n[bold yellow]Tool Call:[/bold yellow] [cyan]{name}[/cyan]([dim]{args}[/dim])")


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
# Human-in-the-Loop Interrupt Handling
# ---------------------------------------------------------------------------

async def handle_interrupts(graph, state, config, completer: CiriCompleter) -> None:
    """Handle all pending interrupts from the graph state."""
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

        if isinstance(val, dict) and val.get("type") == "human_follow_up":
            await _handle_follow_up(graph, val, config, completer)
        elif isinstance(val, dict) and "action_requests" in val:
            await _handle_tool_approval(graph, val, config, completer)
        else:
            # Unknown interrupt type — show raw value and ask for generic response
            console.print(Panel(
                f"[bold yellow]Interrupt:[/bold yellow]\n{json.dumps(val, indent=2, default=str)}",
                border_style="yellow",
            ))
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: pt_prompt("Your response> ", completer=completer)
            )
            await run_graph(graph, Command(resume=response), config, completer)


async def _handle_follow_up(graph, val: dict, config: dict, completer: CiriCompleter) -> None:
    """Handle a human_follow_up interrupt."""
    question = val.get("question", "")
    options = val.get("options")

    console.print(Panel(
        f"[bold yellow]Follow-up Question:[/bold yellow]\n{question}",
        border_style="yellow",
    ))

    if options:
        # Show numbered options
        for i, opt in enumerate(options, 1):
            console.print(f"  [cyan]{i}.[/cyan] {opt}")
        console.print()

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pt_prompt("Choose an option (number or text)> ", completer=completer),
        )
        # If user typed a number, resolve to the option text
        try:
            idx = int(response.strip()) - 1
            if 0 <= idx < len(options):
                response = options[idx]
        except ValueError:
            pass
    else:
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: pt_prompt("Your response> ", completer=completer)
        )

    await run_graph(graph, Command(resume=response), config, completer)


async def _handle_tool_approval(graph, val: dict, config: dict, completer: CiriCompleter) -> None:
    """Handle a tool approval interrupt (approve / reject / edit)."""
    action_requests = val.get("action_requests", [])
    review_configs = val.get("review_configs", [])

    console.print(Panel(
        "[bold yellow]Tool Execution Approval Required[/bold yellow]",
        border_style="yellow",
    ))

    # Display each action request
    for i, req in enumerate(action_requests):
        name = req.get("name", "unknown")
        args = req.get("arguments", req.get("args", {}))
        console.print(f"\n  [bold]Action {i + 1}:[/bold] [cyan]{name}[/cyan]")
        try:
            args_str = json.dumps(args, indent=2, default=str)
            console.print(Syntax(args_str, "json", theme="monokai", line_numbers=False, padding=1))
        except (TypeError, ValueError):
            console.print(f"  Arguments: {args}")

    # Show allowed decisions from review configs if available
    allowed = {"approve", "reject", "edit"}
    if review_configs:
        for rc in review_configs:
            rc_allowed = set(rc.get("allowed_decisions", []))
            if rc_allowed:
                allowed = allowed & rc_allowed

    choices_str = " / ".join(f"[cyan]{c}[/cyan]" for c in sorted(allowed))
    console.print(f"\nOptions: {choices_str}")

    decision = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: pt_prompt(
            "Decision (approve/reject/edit)> ",
            completer=completer,
        ),
    )
    decision = decision.strip().lower()

    if decision == "approve":
        decisions = [{"type": "approve"} for _ in action_requests]
        command = Command(resume={"decisions": decisions})

    elif decision == "reject":
        reason = await asyncio.get_event_loop().run_in_executor(
            None, lambda: pt_prompt("Reason for rejection> ")
        )
        decisions = [{"type": "reject", "message": reason} for _ in action_requests]
        command = Command(resume={"decisions": decisions})

    elif decision == "edit":
        decisions = []
        for req in action_requests:
            name = req.get("name", "unknown")
            args = req.get("arguments", req.get("args", {}))
            console.print(f"\n[bold]Editing action:[/bold] [cyan]{name}[/cyan]")
            console.print(f"Current args: {json.dumps(args, indent=2, default=str)}")
            new_args_str = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pt_prompt("New args (JSON, or press Enter to keep current)> "),
            )
            if new_args_str.strip():
                try:
                    new_args = json.loads(new_args_str)
                except json.JSONDecodeError:
                    console.print("[red]Invalid JSON. Keeping original args.[/red]")
                    new_args = args
            else:
                new_args = args
            decisions.append({
                "type": "edit",
                "edited_action": {"name": name, "args": new_args},
            })
        command = Command(resume={"decisions": decisions})

    else:
        console.print(f"[red]Unknown decision '{decision}'. Defaulting to approve.[/red]")
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
    seen_tool_call_ids: set[str] = set()

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
                                if isinstance(part, dict) and part.get("type") == "text":
                                    text = part.get("text", "")
                                    console.print(text, end="")
                                    current_ai_message += text

                    # Tool calls in the AI message chunk
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            tc_id = tool_call.get("id")
                            if tc_id and tc_id not in seen_tool_call_ids and tool_call.get("name"):
                                status.stop()
                                render_tool_call(tool_call)
                                seen_tool_call_ids.add(tc_id)

                # Complete Tool response messages
                elif isinstance(message, ToolMessage):
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

def handle_threads_command(db: CiriDatabase, current_thread_id: str, completer: CiriCompleter) -> Optional[str]:
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
    console.print(Panel(
        "[bold cyan]CIRI[/bold cyan] - Desktop Personal AI Copilot\n[dim]Initializing...[/dim]",
        border_style="cyan",
    ))

    # Ensure required environment variables are set
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("[yellow]OPENROUTER_API_KEY not found in environment.[/yellow]")
        api_key = Prompt.ask("Please enter your [bold]OpenRouter API Key[/bold]", password=True)
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
            console.print("[green]API Key set for this session.[/green]")
        else:
            console.print("[red]API Key is required to continue. Exiting.[/red]")
            return

    model = os.getenv("CIRI_MODEL")
    if not model:
        console.print("[yellow]CIRI_MODEL not found in environment.[/yellow]")
        model = Prompt.ask("Please enter the [bold]Model name[/bold]", default="openai/gpt-5-mini")
        os.environ["CIRI_MODEL"] = model
        console.print(f"[green]Model set to {model} for this session.[/green]")

    # Initialize encrypted database
    db = CiriDatabase()

    llm_config = LLMConfig(model=model)
    ciri_app = Ciri(llm_config=llm_config)

    # Checkpointer (encrypted SQLite) and Store
    checkpointer = SqliteSaver(conn=db.connection, serde=CiriJsonPlusSerializer())
    store = InMemoryStore()

    # Compile the agent graph
    graph = ciri_app.compile(store=store, checkpointer=checkpointer)

    # Create initial thread
    thread = db.create_thread()
    current_thread_id = thread["id"]
    config = {"configurable": {"thread_id": current_thread_id}}
    is_first_message = True

    # Build file tree completer for @ autocomplete
    root_dir = get_default_filesystem_root()
    completer = CiriCompleter(root_dir)

    console.print(f"[green]Ready![/green] Root directory: [bold]{root_dir}[/bold]")
    console.print("[dim]Tip: @ for file paths, @skills: for skills. Commands: /threads, /new-thread, /delete-thread[/dim]\n")

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
                        None, lambda: handle_threads_command(db, current_thread_id, completer)
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
                    current_thread_id = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: handle_delete_thread_command(db, current_thread_id)
                    )
                    config = {"configurable": {"thread_id": current_thread_id}}
                    is_first_message = True
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
                console.print("\n[yellow]Interrupted by user. Type 'exit' to quit.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    finally:
        db.close()


def main():
    """EntryPoint for the CIRI CLI."""
    asyncio.run(interactive_chat())


if __name__ == "__main__":
    main()
