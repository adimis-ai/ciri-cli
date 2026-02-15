import os
import sys
import json
import uuid
import httpx
import asyncio
import aiosqlite
import subprocess
from pathlib import Path
from typing import Optional, List, Any, Dict, Union

# Third-party imports
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich.align import Align
from rich import box

# prompt_toolkit for input
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory

# LangGraph / LangChain imports
from langgraph.types import Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ToolMessage,
    SystemMessage,
)

# Copilot
from .db import CopilotDatabase
from .utils import get_app_data_dir, detect_browser_profiles, load_all_dotenv
from .copilot import create_copilot
from .controller import CopilotController
from .serializers import LLMConfig

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BANNER = r"""
   ██████╗ ██╗ ██████╗  ██╗
  ██╔════╝ ██║ ██╔══██╗ ██║
  ██║      ██║ ██████╔╝ ██║
  ██║      ██║ ██╔══██╗ ██║
  ╚██████╗ ██║ ██║  ██║ ██║
   ╚═════╝ ╚═╝ ╚═╝  ╚═╝ ╚═╝
"""

SUB_DESCRIPTION = "Contextual Intelligence and Reasoning Interface"

COMMANDS_HELP = {
    "/new-thread": "Create a new conversation thread",
    "/switch-thread": "Switch to a different thread",
    "/delete-thread": "Delete a thread",
    "/change-model": "Change the active LLM model",
    "/threads": "List all threads",
    "/help": "Show this help message",
    "/exit": "Exit CIRI",
}

DEFAULT_MODEL = "openai/gpt-5-mini"

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Model fetching
# ──────────────────────────────────────────────────────────────────────────────


def _get_models_from_env() -> List[str]:
    """Get model list from LITE_LLM_MODEL_LIST env variable."""
    raw = os.getenv("LITE_LLM_MODEL_LIST", "")
    if not raw.strip():
        return []
    try:
        models = json.loads(raw)
        if isinstance(models, list):
            return [m if isinstance(m, str) else m.get("model_name", str(m)) for m in models]
    except json.JSONDecodeError:
        # Try comma-separated
        return [m.strip() for m in raw.split(",") if m.strip()]
    return []


def _fetch_openrouter_models() -> List[Dict[str, Any]]:
    """Fetch model list from OpenRouter API with full metadata.

    Filters to models with:
    - context_length > 130,000 tokens
    - 'tools' in supported_parameters
    - 'image' in architecture.input_modalities
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    base_url = os.getenv("OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        return []
    try:
        resp = httpx.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])

        filtered = []
        for m in models:
            if "id" not in m:
                continue
            ctx = m.get("context_length") or 0
            if ctx < 130_000:
                continue
            supported = m.get("supported_parameters") or []
            if "tools" not in supported:
                continue
            arch = m.get("architecture") or {}
            input_mods = arch.get("input_modalities") or []
            if "image" not in input_mods:
                continue
            filtered.append(m)

        return filtered
    except Exception:
        return []


def fetch_available_models() -> List[Dict[str, Any]]:
    """Fetch models based on LLM_GATEWAY_PROVIDER env var.

    Returns list of model dicts (with 'id' key) for openrouter,
    or list of {'id': name} dicts for env-based models.
    """
    provider = os.getenv("LLM_GATEWAY_PROVIDER", "openrouter").lower()
    if provider == "openrouter":
        return _fetch_openrouter_models()
    else:
        return [{"id": name} for name in _get_models_from_env()]


# ──────────────────────────────────────────────────────────────────────────────
# Command completer for prompt_toolkit
# ──────────────────────────────────────────────────────────────────────────────


class ModelCompleter(Completer):
    """Autocomplete for model selection with fuzzy matching."""

    def __init__(self, model_ids: List[str]):
        self.model_ids = model_ids

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor.lower()
        if not text:
            for mid in self.model_ids[:20]:
                yield Completion(mid, start_position=0)
            return
        for mid in self.model_ids:
            if text in mid.lower():
                yield Completion(mid, start_position=-len(document.text_before_cursor))


class CiriCompleter(Completer):
    """Autocomplete for /commands."""

    def __init__(self):
        self.commands = list(COMMANDS_HELP.keys())

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/"):
            for cmd in self.commands:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))


# ──────────────────────────────────────────────────────────────────────────────
# CopilotCLI
# ──────────────────────────────────────────────────────────────────────────────


class CopilotCLI:
    def __init__(self):
        self.db: Optional[CopilotDatabase] = None
        self.checkpointer: Optional[AsyncSqliteSaver] = None
        self.controller: Optional[CopilotController] = None
        self.current_thread: Optional[Dict[str, Any]] = None
        self.selected_model: str = DEFAULT_MODEL
        self.selected_browser_profile: Optional[Dict[str, Any]] = None
        self.input_history = InMemoryHistory()
        self.completer = CiriCompleter()
        self.session = PromptSession(
            completer=self.completer,
            history=self.input_history,
        )

    # ── Banner ────────────────────────────────────────────────────────────

    def _render_banner(self):
        """Render the CIRI banner with large ASCII art."""
        banner_text = Text(BANNER, style="bold cyan")
        sub_text = Text(SUB_DESCRIPTION, style="italic bright_white")

        console.print()
        console.print(Align.center(banner_text))
        console.print(Align.center(sub_text))
        console.print()

    # ── Model Selection ───────────────────────────────────────────────────

    async def _select_model(self) -> str:
        """Interactive model selection with autocomplete.

        Shows a simple inline prompt with fuzzy autocomplete from all
        available OpenRouter models (filtered to 130K+ context, tool calls, images).
        """
        console.print(Rule("[bold cyan]Select Model[/]", style="cyan"))
        console.print()

        with console.status("[cyan]Fetching available models...[/]", spinner="dots"):
            models_data = fetch_available_models()

        model_ids = [m["id"] for m in models_data]

        if not model_ids:
            console.print(
                f"  [yellow]Could not fetch model list. Using default:[/] [bold]{DEFAULT_MODEL}[/]"
            )
            console.print()
            use_default = Confirm.ask(
                f"  Use [bold]{DEFAULT_MODEL}[/]?", default=True, console=console
            )
            if use_default:
                return DEFAULT_MODEL
            custom = Prompt.ask("  Enter model name", console=console)
            return custom.strip() or DEFAULT_MODEL

        console.print(
            f"  [dim]{len(model_ids)} models available (130K+ context, tool calls, vision)[/]"
        )
        console.print()

        model_session = PromptSession(
            completer=ModelCompleter(model_ids),
            complete_while_typing=True,
        )

        try:
            choice = await model_session.prompt_async(
                HTML("<ansicyan><b>  Select Model > </b></ansicyan>"),
                default=DEFAULT_MODEL,
            )
        except (EOFError, KeyboardInterrupt):
            choice = DEFAULT_MODEL

        selected = choice.strip() or DEFAULT_MODEL
        console.print(f"  [green]Selected:[/] [bold]{selected}[/]")
        return selected

    # ── Browser Profile Selection ─────────────────────────────────────────

    async def _select_browser_profile(self) -> Optional[Dict[str, Any]]:
        """Interactive browser profile selection."""
        console.print(Rule("[bold cyan]Select Browser Profile[/]", style="cyan"))
        console.print()

        with console.status("[cyan]Detecting browser profiles...[/]", spinner="dots"):
            profiles = detect_browser_profiles()

        if not profiles:
            console.print("  [yellow]No browser profiles detected.[/] Continuing without one.")
            console.print()
            return None

        table = Table(
            title="Browser Profiles",
            box=box.ROUNDED,
            title_style="bold cyan",
            header_style="bold bright_white",
            border_style="dim cyan",
            padding=(0, 1),
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Browser", style="cyan")
        table.add_column("Profile", style="white")
        table.add_column("Name", style="bright_white")

        for i, p in enumerate(profiles, 1):
            table.add_row(str(i), p["browser"], p["profile_directory"], p["display_name"])

        console.print(table)
        console.print()

        choice = Prompt.ask(
            "  Select profile number (or [bold]skip[/])",
            default="skip",
            console=console,
        )

        if choice.lower() == "skip":
            console.print("  [yellow]Skipping browser profile selection.[/]")
            return None

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(profiles):
                selected = profiles[idx]
                console.print(
                    f"  [green]Selected:[/] [bold]{selected['browser']}[/] - {selected['display_name']}"
                )
                return selected
        except ValueError:
            pass

        console.print("  [yellow]Invalid selection, skipping.[/]")
        return None

    # ── Setup ─────────────────────────────────────────────────────────────

    async def setup(self):
        """Initialize database, checkpointer, model, and controller."""
        load_all_dotenv()

        self._render_banner()

        # ── Model selection ──
        self.selected_model = await self._select_model()
        console.print()

        # ── Browser profile selection ──
        self.selected_browser_profile = await self._select_browser_profile()
        console.print()

        # ── Setup Flow ──
        console.print(Rule("[bold cyan]Setup[/]", style="cyan"))
        console.print()

        db_path = get_app_data_dir() / "ciri.db"

        # 1. Database
        with console.status("[cyan]Initializing database...[/]", spinner="dots"):
            self.db = CopilotDatabase(db_path=db_path)
        console.print("  [green]✓[/] Database initialized")

        # 2. Checkpointer
        with console.status("[cyan]Preparing checkpointer...[/]", spinner="dots"):
            conn = await aiosqlite.connect(str(db_path))
            self.checkpointer = AsyncSqliteSaver(conn=conn)
            await self.checkpointer.setup()
        console.print("  [green]✓[/] Checkpointer ready")

        # 3. Create Copilot (Agent Building)
        with console.status("[cyan]Building copilot agent (this may take a moment)...[/]", spinner="dots"):
            # LLM Config
            llm_config = LLMConfig(model=self.selected_model)

            # Browser profile kwargs
            browser_kwargs = {}
            if self.selected_browser_profile:
                browser_kwargs["browser_name"] = self.selected_browser_profile["browser"]
                browser_kwargs["browser_profile_directory"] = self.selected_browser_profile[
                    "profile_directory"
                ]

            # Create copilot
            copilot = await create_copilot(
                name="Ciri",
                llm_config=llm_config,
                checkpointer=self.checkpointer,
                **browser_kwargs,
            )
        console.print("  [green]✓[/] Copilot agent built")

        # 4. Controller
        with console.status("[cyan]Finalizing controller...[/]", spinner="dots"):
            self.controller = CopilotController(graph=copilot, db=self.db)
        console.print("  [green]✓[/] Controller ready")

        # 5. Thread
        with console.status("[cyan]Resuming conversation...[/]", spinner="dots"):
            threads = self.controller.list_threads()
            if threads:
                self.current_thread = threads[0]  # most recently updated
            else:
                self.current_thread = self.controller.create_thread()
        
        thread_title = self.current_thread["title"]
        thread_id = self.current_thread["id"][:8] + "..."
        # Clear and render the main screen
        await asyncio.sleep(0.5)  # Brief pause to let the user see the setup results
        self._render_post_setup_screen()

        # Render existing messages
        await self._render_existing_messages()

    # ── Commands Help ─────────────────────────────────────────────────────

    def _show_commands_help(self):
        """Display available commands in a perfectly aligned 2-column layout using a Table."""
        console.print(Rule("[bold cyan]Available Commands[/]", style="cyan"))
        
        table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
        table.add_column("left_cmd", style="bold cyan", width=16)
        table.add_column("left_desc", style="white")
        table.add_column("right_cmd", style="bold cyan", width=16)
        table.add_column("right_desc", style="white")

        items = list(COMMANDS_HELP.items())
        # Logic to split into two columns: left column first half, right column second half
        mid = (len(items) + 1) // 2
        
        for i in range(mid):
            row = []
            # Left column
            cmd, desc = items[i]
            row.extend([f"  {cmd}", desc])
            
            # Right column
            if i + mid < len(items):
                cmd, desc = items[i + mid]
                row.extend([f"  {cmd}", desc])
            else:
                row.extend(["", ""])
            
            table.add_row(*row)
            
        console.print(table)

    def _render_post_setup_screen(self):
        """Clear terminal and render the main screen after setup completes."""
        if os.name == 'nt':
            os.system('cls')
        else:
            # \033[H: move to home, \033[2J: clear screen, \033[3J: clear scrollback
            sys.stdout.write("\033[H\033[2J\033[3J")
            sys.stdout.flush()
        
        self._render_banner()

        console.print(Rule(style="dim cyan"))

        # Info box: current thread + model
        thread_title = self.current_thread["title"] if self.current_thread else "None"
        thread_id = self.current_thread["id"][:8] + "..." if self.current_thread else ""
        
        info_parts = [
            f"  [bold cyan]Thread:[/] [white]{thread_title}[/] [dim]({thread_id})[/]",
            f"  [bold cyan]Model:[/] [white]{self.selected_model}[/]"
        ]
        console.print(Columns(info_parts, padding=(0, 8)))

        console.print()
        self._show_commands_help()
        console.print(Rule(style="dim cyan"))
        console.print()
        console.print()

    # ── Thread Management ─────────────────────────────────────────────────

    async def _cmd_new_thread(self):
        """Create a new thread."""
        title = Prompt.ask("  Thread title", default="New Thread", console=console)
        self.current_thread = self.controller.create_thread()
        if title != "New Thread":
            self.controller.rename_thread(self.current_thread["id"], title)
            self.current_thread["title"] = title
        console.print(f"  [green]✓[/] Created thread: [bold]{self.current_thread['title']}[/]")
        console.print()
        await self._render_existing_messages()

    async def _cmd_switch_thread(self):
        """Switch to a different thread."""
        threads = self.controller.list_threads()
        if not threads:
            console.print("  [yellow]No threads available.[/]")
            return

        table = Table(
            title="Threads",
            box=box.ROUNDED,
            title_style="bold cyan",
            header_style="bold bright_white",
            border_style="dim cyan",
            padding=(0, 1),
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Title", style="white")
        table.add_column("ID", style="dim")
        table.add_column("Updated", style="dim cyan")

        for i, t in enumerate(threads, 1):
            marker = " [bold green]*[/]" if self.current_thread and t["id"] == self.current_thread["id"] else ""
            table.add_row(str(i), t["title"] + marker, t["id"][:8] + "...", t["updated_at"][:19])

        console.print(table)
        choice = Prompt.ask("  Select thread number", console=console)

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(threads):
                self.current_thread = threads[idx]
                self.controller.touch_thread(self.current_thread["id"])
                console.print(
                    f"  [green]✓[/] Switched to: [bold]{self.current_thread['title']}[/]"
                )
                console.print()
                await self._render_existing_messages()
            else:
                console.print("  [yellow]Invalid selection.[/]")
        except ValueError:
            console.print("  [yellow]Invalid input.[/]")

    async def _cmd_delete_thread(self):
        """Delete a thread."""
        threads = self.controller.list_threads()
        if not threads:
            console.print("  [yellow]No threads to delete.[/]")
            return

        table = Table(box=box.ROUNDED, border_style="dim red", padding=(0, 1))
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Title", style="white")
        table.add_column("ID", style="dim")

        for i, t in enumerate(threads, 1):
            table.add_row(str(i), t["title"], t["id"][:8] + "...")

        console.print(table)
        choice = Prompt.ask("  Select thread number to delete", console=console)

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(threads):
                target = threads[idx]
                if Confirm.ask(
                    f"  Delete [bold red]{target['title']}[/]?", default=False, console=console
                ):
                    self.controller.delete_thread(target["id"])
                    console.print(f"  [red]✗[/] Deleted: {target['title']}")
                    # If we deleted the current thread, switch or create new
                    if self.current_thread and target["id"] == self.current_thread["id"]:
                        remaining = self.controller.list_threads()
                        if remaining:
                            self.current_thread = remaining[0]
                            console.print(
                                f"  [green]✓[/] Switched to: [bold]{self.current_thread['title']}[/]"
                            )
                            console.print()
                            await self._render_existing_messages()
                        else:
                            self.current_thread = self.controller.create_thread()
                            console.print("  [green]✓[/] Created new thread")
                            console.print()
                            await self._render_existing_messages()
            else:
                console.print("  [yellow]Invalid selection.[/]")
        except ValueError:
            console.print("  [yellow]Invalid input.[/]")

    def _cmd_list_threads(self):
        """List all threads."""
        threads = self.controller.list_threads()
        if not threads:
            console.print("  [yellow]No threads.[/]")
            return

        table = Table(
            title="All Threads",
            box=box.ROUNDED,
            title_style="bold cyan",
            header_style="bold bright_white",
            border_style="dim cyan",
            padding=(0, 1),
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Title", style="white")
        table.add_column("ID", style="dim")
        table.add_column("Created", style="dim")
        table.add_column("Updated", style="dim cyan")

        for i, t in enumerate(threads, 1):
            marker = " [bold green]*[/]" if self.current_thread and t["id"] == self.current_thread["id"] else ""
            table.add_row(
                str(i),
                t["title"] + marker,
                t["id"][:8] + "...",
                t["created_at"][:19],
                t["updated_at"][:19],
            )

        console.print(table)

    async def _cmd_change_model(self):
        """Change the active model (requires agent rebuild)."""
        console.print("  [yellow]Note: Changing the model requires rebuilding the agent.[/]")
        new_model = await self._select_model()
        if new_model != self.selected_model:
            self.selected_model = new_model
            console.print(f"  [green]Model set to:[/] [bold]{self.selected_model}[/]")
            console.print("  [yellow]The new model will take effect on the next agent rebuild.[/]")
            console.print("  [dim]Tip: Use /new-thread or restart CIRI to rebuild with the new model.[/]")
        else:
            console.print("  [dim]Model unchanged.[/]")

    # ── Message Rendering ─────────────────────────────────────────────────

    @staticmethod
    def _render_human_message(msg: HumanMessage):
        """Render a HumanMessage."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        console.print()
        console.print(f"[bold bright_green]You >[/] {content}")

    @staticmethod
    def _render_tool_call(tool_call: dict):
        """Render a tool call from an AIMessage in a Panel box."""
        name = tool_call.get("name", "unknown_tool")
        args = tool_call.get("args", {})
        tc_id = tool_call.get("id", "")

        args_text = json.dumps(args, indent=2, default=str) if args else "{}"
        syntax = Syntax(args_text, "json", theme="monokai", line_numbers=False, word_wrap=True)

        console.print()
        console.print(Panel(
            syntax,
            title=f" [bold bright_yellow]Tool Call:[/] [bold]{name}[/] [dim]({tc_id[:8]})[/] ",
            title_align="left",
            border_style="bright_yellow",
            padding=(0, 1),
            box=box.ROUNDED,
        ))

    @staticmethod
    def _render_tool_message(msg: ToolMessage):
        """Render a ToolMessage (tool response) in a Panel box."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        tool_name = getattr(msg, "name", None) or ""
        tc_id = getattr(msg, "tool_call_id", "") or ""

        # Truncate very long outputs
        max_len = 5000  # Increased for better visibility, but still capped
        if len(content) > max_len:
            content = content[:max_len] + f"\n... [dim](truncated, {len(content)} chars total)[/]"

        title_parts = ["[bold bright_green]Tool Response[/]"]
        if tool_name:
            title_parts.append(f"[bold]{tool_name}[/]")
        if tc_id:
            title_parts.append(f"[dim]({tc_id[:8]})[/]")

        console.print()
        console.print(Panel(
            content,
            title=f" {' '.join(title_parts)} ",
            title_align="left",
            border_style="bright_green",
            padding=(0, 1),
            box=box.ROUNDED,
        ))

    @staticmethod
    def _render_system_message(msg: SystemMessage):
        """Render a SystemMessage."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        console.print()
        console.print(f"[bold magenta]System >[/] [dim italic]{content}[/]")

    @staticmethod
    def _render_summary_panel(text: str, title: str = "💭 Thought"):
        """Render a summarization or reasoning block in a beautiful Panel."""
        if not text.strip():
            return
        
        console.print()
        console.print(Panel(
            text,
            title=f" [bold blue]{title}[/] ",
            title_align="left",
            border_style="blue",
            padding=(1, 2),
            style="italic",
            box=box.ROUNDED,
        ))

    @staticmethod
    def _render_ai_complete(content: Union[str, List[Dict[str, Any]]]):
        """Render a complete AI message (non-streamed).
        
        Handles both simple strings and complex content lists with reasoning blocks.
        """
        if isinstance(content, str):
            if content.strip():
                console.print(Markdown(content))
            return

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                
                btype = block.get("type")
                if btype == "reasoning":
                    summary_list = block.get("summary", [])
                    summary_text = ""
                    for s in summary_list:
                        if isinstance(s, dict) and s.get("type") == "summary_text":
                            summary_text += s.get("text", "")
                    
                    if summary_text:
                        CopilotCLI._render_summary_panel(summary_text)
                elif btype == "text":
                    text_content = block.get("text", "")
                    if text_content.strip():
                        console.print(Markdown(text_content))
                elif btype == "summary":
                    # Some models use 'summary' directly
                    text_content = block.get("text", "")
                    if text_content.strip():
                        CopilotCLI._render_summary_panel(text_content, title="Thought")

    async def _render_existing_messages(self):
        """Fetch and render existing messages for the current thread."""
        if not self.current_thread:
            return

        config = {"configurable": {"thread_id": self.current_thread["id"]}}
        state = await self.controller.get_state(config)
        
        if not state or "messages" not in state.values:
            return

        messages = state.values["messages"]
        if not messages:
            return

        console.print(Rule("[dim]Conversation History[/]", style="dim cyan"))
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                self._render_human_message(msg)
            elif isinstance(msg, AIMessage):
                # For history, we render the full content of AI messages
                console.print()
                console.print("[bold cyan]CIRI >[/]")
                self._render_ai_complete(msg.content)
                
                # Also render tool calls if any
                for tc in getattr(msg, "tool_calls", []):
                    self._render_tool_call(tc)
            elif isinstance(msg, ToolMessage):
                self._render_tool_message(msg)
            elif isinstance(msg, SystemMessage):
                self._render_system_message(msg)

        console.print()
        console.print(Rule(style="dim cyan"))
        console.print()

    # ── Interrupt Handling ────────────────────────────────────────────────

    def _handle_follow_up_interrupt(self, interrupt_value: dict) -> list:
        """Handle human_follow_up interrupt - ask user clarification questions."""
        queries = interrupt_value.get("queries", [])
        if not queries:
            return []

        console.print()
        console.print(Panel(
            "[bold]CIRI needs your input[/]",
            title="[bold yellow]Clarification Needed[/]",
            title_align="left",
            border_style="yellow",
            padding=(0, 1),
        ))

        responses = []
        for i, query in enumerate(queries, 1):
            question = query.get("question", "")
            options = query.get("options", [])

            console.print(f"\n  [bold cyan]Q{i}:[/] {question}")

            if options:
                for j, opt in enumerate(options, 1):
                    console.print(f"    [dim]{j}.[/] {opt}")
                console.print()

                answer = Prompt.ask(
                    "  Enter number or type your answer",
                    console=console,
                )

                # Resolve numbered selection
                try:
                    opt_idx = int(answer) - 1
                    if 0 <= opt_idx < len(options):
                        answer = options[opt_idx]
                except ValueError:
                    pass

                responses.append(answer)
            else:
                answer = Prompt.ask("  Your answer", console=console)
                responses.append(answer)

        return responses

    def _handle_script_execution_interrupt(self, interrupt_value: dict) -> Union[str, dict]:
        """Handle script_execution interrupt - approve/reject/edit script."""
        console.print()
        console.print(Panel(
            "[bold]A script wants to execute[/]",
            title="[bold red]Script Execution Approval[/]",
            title_align="left",
            border_style="red",
            padding=(0, 1),
        ))

        lang = interrupt_value.get("language", "python")
        deps = interrupt_value.get("dependencies", [])
        script = interrupt_value.get("script_content", "")
        timeout = interrupt_value.get("timeout", 120)

        console.print(f"  [bold]Language:[/] {lang}")
        if deps:
            console.print(f"  [bold]Dependencies:[/] {', '.join(deps)}")
        console.print(f"  [bold]Timeout:[/] {timeout}s")
        console.print()
        console.print(Syntax(script, lang, theme="monokai", line_numbers=True, word_wrap=True))
        console.print()

        decision = Prompt.ask(
            "  [bold]Approve[/], [bold]Reject[/], or [bold]Edit[/]?",
            choices=["approve", "reject", "edit", "a", "r", "e"],
            default="approve",
            console=console,
        )

        if decision in ("reject", "r"):
            reason = Prompt.ask("  Reason (optional)", default="", console=console)
            return {"status": "rejected", "reason": reason}
        elif decision in ("edit", "e"):
            console.print("  [dim]Enter the edited script (end with an empty line containing only 'EOF'):[/]")
            lines = []
            while True:
                try:
                    line = input()
                    if line.strip() == "EOF":
                        break
                    lines.append(line)
                except EOFError:
                    break
            edited_script = "\n".join(lines)
            if edited_script.strip():
                return {"status": "edited", "script_content": edited_script}
            console.print("  [yellow]Empty script, approving original.[/]")

        return "approved"

    def _handle_hitl_interrupt(self, interrupt_value: dict) -> dict:
        """Handle HumanInTheLoopMiddleware tool approval interrupts."""
        review_configs = interrupt_value.get("review_configs", [])
        action_requests = interrupt_value.get("action_requests", [])

        console.print()
        console.print(Panel(
            "[bold]Tool execution requires your approval[/]",
            title="[bold red]Tool Approval Required[/]",
            title_align="left",
            border_style="red",
            padding=(0, 1),
        ))

        decisions = []
        for i, action in enumerate(action_requests):
            name = action.get("name", "unknown")
            desc = action.get("description", "")
            args = action.get("arguments", {})

            console.print(f"\n  [bold yellow]Action {i+1}:[/] [bold]{name}[/]")
            if desc:
                console.print(f"  [dim]{desc}[/]")
            if args:
                args_text = json.dumps(args, indent=2, default=str)
                console.print(Syntax(args_text, "json", theme="monokai", line_numbers=False, word_wrap=True))

            # Find allowed decisions for this action
            allowed = ["approve", "reject"]
            for rc in review_configs:
                if rc.get("action_name") == name:
                    allowed = rc.get("allowed_decisions", ["approve", "reject"])
                    break

            short_map = {"a": "approve", "r": "reject", "e": "edit"}
            choices = []
            for a in allowed:
                choices.append(a)
                for k, v in short_map.items():
                    if v == a:
                        choices.append(k)

            decision = Prompt.ask(
                f"  Decision ({'/'.join(allowed)})",
                choices=choices,
                default="approve",
                console=console,
            )
            decision = short_map.get(decision, decision)

            if decision == "approve":
                decisions.append({"type": "approve"})
            elif decision == "reject":
                reason = Prompt.ask("  Reason (optional)", default="", console=console)
                d = {"type": "reject"}
                if reason:
                    d["message"] = reason
                decisions.append(d)
            elif decision == "edit":
                console.print("  Enter edited arguments as JSON:")
                raw = Prompt.ask("  ", console=console)
                try:
                    edited_args = json.loads(raw)
                    decisions.append({
                        "type": "edit",
                        "edited_action": {"name": name, "args": edited_args},
                    })
                except json.JSONDecodeError:
                    console.print("  [yellow]Invalid JSON, approving instead.[/]")
                    decisions.append({"type": "approve"})
            else:
                decisions.append({"type": "approve"})

        return {"decisions": decisions}

    def _handle_interrupt(self, interrupt_data: dict) -> Any:
        """Route an interrupt to the appropriate handler and return the resume value."""
        interrupt_type = interrupt_data.get("type", "")

        if interrupt_type == "human_follow_up":
            return self._handle_follow_up_interrupt(interrupt_data)
        elif interrupt_type == "script_execution":
            return self._handle_script_execution_interrupt(interrupt_data)
        elif "review_configs" in interrupt_data or "action_requests" in interrupt_data:
            return self._handle_hitl_interrupt(interrupt_data)
        else:
            # Unknown interrupt - show raw and ask for JSON response
            console.print()
            console.print(Panel(
                Syntax(
                    json.dumps(interrupt_data, indent=2, default=str),
                    "json",
                    theme="monokai",
                ),
                title="[bold yellow]Unknown Interrupt[/]",
                title_align="left",
                border_style="yellow",
                padding=(0, 1),
            ))
            raw = Prompt.ask("  Respond (JSON or text)", console=console)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw

    # ── Streaming ─────────────────────────────────────────────────────────

    async def _stream_response(self, inputs: Union[dict, Command]):
        """Stream a response from the controller, rendering messages in real-time.

        - "messages" mode: AIMessage tokens are typewritten character-by-character
        - "updates" mode: HumanMessage, ToolCall, ToolMessage, SystemMessage
          appear at once; interrupts are collected and processed after
        """
        config = {"configurable": {"thread_id": self.current_thread["id"]}}

        ai_buffer = ""  # accumulates streamed AI tokens
        pending_interrupts = []  # collect interrupts from updates
        streaming_ai = False

        try:
            async for namespace, stream_type, chunk in self.controller.run(
                inputs, config
            ):
                # ── messages mode: token-by-token AI streaming ──
                if stream_type == "messages":
                    message, metadata = chunk

                    if isinstance(message, (AIMessageChunk, AIMessage)):
                        token = ""
                        if isinstance(message.content, str):
                            token = message.content
                        elif isinstance(message.content, list):
                            for block in message.content:
                                if not isinstance(block, dict):
                                    continue
                                
                                btype = block.get("type")
                                if btype == "text":
                                    token += block.get("text", "")
                                elif btype == "reasoning":
                                    # For streaming, we might want to show reasoning as it comes
                                    # but it's complex to handle partial summaries nicely.
                                    # For now, let's just collect it and show it if it's a full block
                                    summary_list = block.get("summary", [])
                                    summary_text = ""
                                    for s in summary_list:
                                        if isinstance(s, dict) and s.get("type") == "summary_text":
                                            summary_text += s.get("text", "")
                                    
                                    if summary_text:
                                        # Flush if we were already streaming text
                                        if streaming_ai:
                                            sys.stdout.write("\n")
                                            sys.stdout.flush()
                                            streaming_ai = False
                                        
                                        CopilotCLI._render_summary_panel(summary_text)
                                        
                                        # Resume CIRI prompt for subsequent text
                                        if not streaming_ai:
                                            console.print("[bold cyan]CIRI >[/] ", end="")
                                            streaming_ai = True

                        if token:
                            if not streaming_ai:
                                streaming_ai = True
                                console.print()
                                console.print("[bold cyan]CIRI >[/] ", end="")

                            # Typewrite: print each token directly to stdout
                            sys.stdout.write(token)
                            sys.stdout.flush()
                            ai_buffer += token

                # ── updates mode: state updates from nodes ──
                elif stream_type == "updates":
                    for node_name, node_value in chunk.items():
                        # Check for __interrupt__
                        if node_name == "__interrupt__":
                            if isinstance(node_value, list):
                                pending_interrupts.extend(node_value)
                            else:
                                pending_interrupts.append(node_value)
                            continue

                        # Process messages in the update
                        if isinstance(node_value, dict) and "messages" in node_value:
                            messages = node_value["messages"]
                            if not isinstance(messages, list):
                                messages = [messages]

                            for msg in messages:
                                # Flush AI buffer before rendering other message types
                                if streaming_ai and ai_buffer.strip():
                                    # End the streaming line
                                    sys.stdout.write("\n")
                                    sys.stdout.flush()
                                    streaming_ai = False
                                    ai_buffer = ""

                                if isinstance(msg, HumanMessage):
                                    self._render_human_message(msg)
                                elif isinstance(msg, AIMessage) and not isinstance(msg, AIMessageChunk):
                                    # Full AI message from updates - render tool_calls only
                                    for tc in getattr(msg, "tool_calls", []):
                                        self._render_tool_call(tc)
                                elif isinstance(msg, ToolMessage):
                                    self._render_tool_message(msg)
                                elif isinstance(msg, SystemMessage):
                                    self._render_system_message(msg)

        finally:
            # Flush remaining AI buffer
            if streaming_ai and ai_buffer.strip():
                sys.stdout.write("\n")
                sys.stdout.flush()

        # Handle any interrupts that were collected
        if pending_interrupts:
            await self._process_interrupts(pending_interrupts)

    async def _process_interrupts(self, interrupts: list):
        """Process collected interrupts and resume the graph."""
        for intr in interrupts:
            # Extract the actual interrupt value
            interrupt_data = intr
            if isinstance(intr, dict) and "value" in intr:
                interrupt_data = intr["value"]

            # Render interrupt indicator
            console.print()
            console.print(Rule("[bold yellow]Interrupt[/]", style="yellow"))

            # Handle the interrupt
            resume_value = self._handle_interrupt(interrupt_data)

            # Resume the graph with the user's response
            console.print()
            console.print("  [dim]Resuming...[/]")

            await self._stream_response(Command(resume=resume_value))

    # ── Main Loop ─────────────────────────────────────────────────────────

    async def run(self):
        """Main chat loop."""
        await self.setup()

        while True:
            try:
                user_input = await self.session.prompt_async(
                    HTML("<ansibrightgreen><b>You > </b></ansibrightgreen>"),
                )
                user_input = user_input.strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n  [dim]Goodbye![/]")
                break

            if not user_input:
                continue

            # ── Handle commands ──
            if user_input.startswith("/"):
                cmd = user_input.split()[0].lower()

                if cmd == "/exit":
                    console.print("  [dim]Goodbye![/]")
                    break
                elif cmd == "/help":
                    self._show_commands_help()
                elif cmd == "/new-thread":
                    await self._cmd_new_thread()
                elif cmd == "/switch-thread":
                    await self._cmd_switch_thread()
                elif cmd == "/delete-thread":
                    await self._cmd_delete_thread()
                elif cmd == "/threads":
                    self._cmd_list_threads()
                elif cmd == "/change-model":
                    await self._cmd_change_model()
                else:
                    console.print(f"  [yellow]Unknown command: {cmd}[/]. Type /help for available commands.")

                console.print()
                continue

            # ── Send message ──
            self.controller.touch_thread(self.current_thread["id"])

            try:
                await self._stream_response({"messages": [HumanMessage(content=user_input)]})
            except KeyboardInterrupt:
                console.print("\n  [yellow]Interrupted.[/]")
            except Exception as e:
                console.print(f"\n  [bold red]Error:[/] {e}")

            console.print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main():
    """Entry point for the CIRI CLI."""
    cli = CopilotCLI()
    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        console.print("\n[dim]Exiting CIRI...[/]")


if __name__ == "__main__":
    main()
