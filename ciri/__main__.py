import os
import sys
import json
import time
import signal
import platform
import subprocess
import httpx
import asyncio
import argparse
import aiosqlite
from pathlib import Path
from dotenv import load_dotenv
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
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings

# LangGraph / LangChain imports
from langgraph.types import Command, Interrupt
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage,
    SystemMessage,
)

# Copilot
from .db import CopilotDatabase
from .utils import (
    get_app_data_dir,
    detect_browser_profiles,
    is_cdp_port_open,
    load_all_dotenv,
    get_default_filesystem_root,
    list_files_with_gitignore,
    list_folders_with_gitignore,
    list_skills,
    list_toolkits,
    list_subagents,
    load_settings,
    save_settings,
    sync_default_skills,
)
from .backend import CiriBackend
from .copilot import create_copilot
from .controller import CopilotController
from .serializers import LLMConfig

load_dotenv()

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
    "/change-browser-profile": "Change the default browser profile",
    "/sync": "Analyze workspace & self-train",
    "/threads": "List all threads",
    "/help": "Show this help message",
    "/exit": "Exit CIRI",
}

KEYBOARD_SHORTCUTS = {
    "Alt+Enter": "Insert new line",
    "Ctrl+C": "Stop streaming",
    "Ctrl+C ×2": "Exit CIRI",
    "↑ / ↓": "Browse history",
}

DEFAULT_MODEL = "openai/gpt-5-mini"  # user-facing format: provider/model

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# API Key Setup
# ──────────────────────────────────────────────────────────────────────────────


def _persist_env_var(name: str, value: str) -> list[str]:
    """Persist an environment variable globally across Mac, Linux, and Windows.

    Writes to:
    1. CIRI's global .env file (~/.ciri/.env) — read by load_all_dotenv() on every startup
    2. Shell profile or Windows registry — so the var is available outside CIRI too

    Returns a list of human-readable messages describing where the variable was saved.
    """
    messages: list[str] = []
    system = platform.system()

    # ── 1. Always write to CIRI's global .env (cross-platform, always works) ──
    global_env = get_app_data_dir() / ".env"
    try:
        global_env.parent.mkdir(parents=True, exist_ok=True)
        env_line = f"{name}={value}"

        if global_env.exists():
            content = global_env.read_text()
            lines = content.splitlines()
            found = False
            new_lines = []
            for line in lines:
                if line.strip().startswith(f"{name}="):
                    new_lines.append(env_line)
                    found = True
                else:
                    new_lines.append(line)
            if not found:
                new_lines.append(env_line)
            global_env.write_text("\n".join(new_lines) + "\n")
        else:
            global_env.write_text(env_line + "\n")

        messages.append(f"Saved to [bold]{global_env}[/] (loaded automatically by CIRI).")
    except OSError as e:
        messages.append(f"[yellow]Warning:[/] Could not write to {global_env}: {e}")

    # ── 2. Also persist to shell profile / Windows registry ──
    if system == "Windows":
        try:
            subprocess.run(
                ["setx", name, value],
                check=True,
                capture_output=True,
            )
            messages.append("Saved via [bold]setx[/] (available in new terminals).")
        except Exception:
            pass  # CIRI .env is the primary store; this is best-effort
    else:
        home = Path.home()
        shell = os.environ.get("SHELL", "")
        export_line = f'\nexport {name}="{value}"\n'

        if "zsh" in shell:
            profile = home / ".zshrc"
        elif "bash" in shell:
            profile = (home / ".bash_profile") if system == "Darwin" else (home / ".bashrc")
        else:
            profile = home / ".profile"

        try:
            if profile.exists():
                content = profile.read_text()
                if f"export {name}=" in content:
                    lines = content.splitlines(keepends=True)
                    new_lines = []
                    for line in lines:
                        if line.strip().startswith(f"export {name}="):
                            new_lines.append(f'export {name}="{value}"\n')
                        else:
                            new_lines.append(line)
                    profile.write_text("".join(new_lines))
                else:
                    with open(profile, "a") as f:
                        f.write(export_line)
            else:
                with open(profile, "a") as f:
                    f.write(export_line)
            messages.append(f"Saved to [bold]{profile}[/] (available in new terminals).")
        except OSError:
            pass  # CIRI .env is the primary store; this is best-effort

    return messages


def _extract_provider_from_model(model: str) -> str | None:
    """Extract the provider name from a model string like 'openai/gpt-5-mini' or 'openai:gpt-5-mini'."""
    if "/" in model:
        return model.split("/", 1)[0]
    if ":" in model:
        return model.split(":", 1)[0]
    return None


def _normalize_model_for_langchain(model: str) -> str:
    """Convert 'provider/model_name' to 'provider:model_name' for LangChain's init_chat_model."""
    if "/" in model:
        return model.replace("/", ":", 1)
    return model


# Well-known provider API key links
_PROVIDER_KEY_URLS: dict[str, str] = {
    "openai": "https://platform.openai.com/api-keys",
    "anthropic": "https://console.anthropic.com/settings/keys",
    "google": "https://aistudio.google.com/apikey",
    "google_genai": "https://aistudio.google.com/apikey",
    "mistralai": "https://console.mistral.ai/api-keys",
    "cohere": "https://dashboard.cohere.com/api-keys",
    "groq": "https://console.groq.com/keys",
    "fireworks": "https://fireworks.ai/account/api-keys",
    "together": "https://api.together.ai/settings/api-keys",
    "deepseek": "https://platform.deepseek.com/api_keys",
    "openrouter": "https://openrouter.ai/keys",
}


def ensure_provider_api_key(model: str) -> None:
    """Ensure the API key for the model's provider is set; prompt and persist if missing.

    For openrouter gateway, checks OPENROUTER_API_KEY.
    For langchain gateway, extracts the provider from the model string and checks
    {PROVIDER}_API_KEY (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY).
    """
    gateway = os.getenv("LLM_GATEWAY_PROVIDER", "langchain")

    if gateway == "openrouter":
        env_key = "OPENROUTER_API_KEY"
        provider_display = "OpenRouter"
        key_url = _PROVIDER_KEY_URLS.get("openrouter", "")
    else:
        # LangChain direct provider — derive key name from model string
        provider = _extract_provider_from_model(model)
        if not provider:
            return  # Can't determine provider; let init_chat_model handle it
        env_key = f"{provider.upper()}_API_KEY"
        provider_display = provider.capitalize()
        key_url = _PROVIDER_KEY_URLS.get(provider.lower(), "")

    api_key = os.getenv(env_key, "").strip()
    if api_key:
        return

    console.print()
    url_hint = (
        f"\nGet one at [link={key_url}]{key_url}[/link]" if key_url else ""
    )
    console.print(
        Panel(
            f"[bold yellow]{env_key}[/] is not set.\n"
            f"You need a {provider_display} API key to use CIRI with this model.{url_hint}",
            title="[bold red]API Key Required[/]",
            border_style="red",
        )
    )
    console.print()

    api_key = Prompt.ask(f"  [bold cyan]Enter your {provider_display} API key[/]").strip()

    if not api_key:
        console.print("  [bold red]No API key provided. Exiting.[/]")
        sys.exit(1)

    # Set for current process immediately
    os.environ[env_key] = api_key

    # Persist globally
    messages = _persist_env_var(env_key, api_key)
    console.print("  [green]✓[/] API key set for this session.")
    for msg in messages:
        console.print(f"  [green]✓[/] {msg}")
    console.print()


# ──────────────────────────────────────────────────────────────────────────────
# Model fetching
# ──────────────────────────────────────────────────────────────────────────────


def _get_models_from_env() -> List[str]:
    """Get model list from LLM_MODEL_LIST or LITE_LLM_MODEL_LIST env variable."""
    raw = os.getenv("LLM_MODEL_LIST") or os.getenv("LITE_LLM_MODEL_LIST", "")
    if not raw.strip():
        return []
    try:
        models = json.loads(raw)
        if isinstance(models, list):
            return [
                m if isinstance(m, str) else m.get("model_name", str(m)) for m in models
            ]
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
    For langchain gateway, LLM_MODEL_LIST is optional — returns empty list
    if not set (user can type any model in provider/model_name format).
    """
    provider = os.getenv("LLM_GATEWAY_PROVIDER", "langchain").lower()
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
    """Autocomplete for /commands and @ triggers (files, folders, skills, toolkits, subagents)."""

    def __init__(self):
        self.commands = list(COMMANDS_HELP.keys())
        self.root = get_default_filesystem_root()

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor

        # Handle /commands
        if text.startswith("/"):
            for cmd in self.commands:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))
            return

        # Handle @files: trigger
        if "@files:" in text:
            prefix = text.split("@files:")[-1]
            try:
                files = list_files_with_gitignore(self.root, prefix)
                for file in files[:50]:  # Limit to 50 results
                    yield Completion(
                        file,
                        start_position=-len(prefix),
                        display=f"📄 {file}",
                    )
            except Exception:
                pass
            return

        # Handle @folders: trigger
        if "@folders:" in text:
            prefix = text.split("@folders:")[-1]
            try:
                folders = list_folders_with_gitignore(self.root, prefix)
                for folder in folders[:50]:  # Limit to 50 results
                    yield Completion(
                        folder,
                        start_position=-len(prefix),
                        display=f"📁 {folder}",
                    )
            except Exception:
                pass
            return

        # Handle @skills: trigger
        if "@skills:" in text:
            prefix = text.split("@skills:")[-1]
            try:
                skills = list_skills(self.root, prefix)
                for skill in skills:
                    yield Completion(
                        skill,
                        start_position=-len(prefix),
                        display=f"⚡ {skill}",
                    )
            except Exception:
                pass
            return

        # Handle @toolkits: trigger
        if "@toolkits:" in text:
            prefix = text.split("@toolkits:")[-1]
            try:
                toolkits = list_toolkits(self.root, prefix)
                for toolkit in toolkits:
                    yield Completion(
                        toolkit,
                        start_position=-len(prefix),
                        display=f"🔧 {toolkit}",
                    )
            except Exception:
                pass
            return

        # Handle @subagents: trigger
        if "@subagents:" in text:
            prefix = text.split("@subagents:")[-1]
            try:
                subagents = list_subagents(self.root, prefix)
                for subagent in subagents:
                    yield Completion(
                        subagent,
                        start_position=-len(prefix),
                        display=f"🤖 {subagent}",
                    )
            except Exception:
                pass
            return


# ──────────────────────────────────────────────────────────────────────────────
# CopilotCLI
# ──────────────────────────────────────────────────────────────────────────────


class CopilotCLI:
    # Seconds within which a second Ctrl+C terminates the session
    _DOUBLE_CTRL_C_THRESHOLD = 1.5

    @staticmethod
    def _create_input_key_bindings() -> KeyBindings:
        """Create key bindings for the chat input.

        - Enter           → submit the input
        - Escape+Enter    → insert a newline  (Alt+Enter on most terminals)

        Note: Ctrl+Enter may or may not send a distinct sequence depending on
        the terminal emulator.  In terminals that use the Kitty keyboard
        protocol or xterm modifyOtherKeys, it works.  Otherwise Alt+Enter
        (or Escape then Enter) is the reliable alternative.
        """
        kb = KeyBindings()

        @kb.add("enter")
        def _submit(event):
            event.current_buffer.validate_and_handle()

        @kb.add("escape", "enter")  # Alt+Enter / Escape then Enter
        def _newline(event):
            event.current_buffer.insert_text("\n")

        return kb

    def __init__(self, all_allowed: bool = False):
        self.all_allowed = all_allowed
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
            key_bindings=self._create_input_key_bindings(),
            multiline=True,
        )
        self.is_new_thread = False  # Track if the current thread needs a title
        self._last_ctrl_c_time: float = 0.0  # For double Ctrl+C detection
        self._streaming = False  # True while streaming a response
        self.settings = load_settings()
        self.selected_model = self.settings.get("model", DEFAULT_MODEL)
        self.selected_browser_profile = self.settings.get("browser_profile")

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

        For langchain gateway: LLM_MODEL_LIST is optional. If provided, models are
        shown as autocomplete suggestions. User can type any model in provider/model_name
        format (e.g. openai/gpt-5-mini, anthropic/claude-sonnet-4-5-20250929).

        For openrouter gateway: fetches models from the OpenRouter API.
        """
        console.print(Rule("[bold cyan]Select Model[/]", style="cyan"))
        console.print()

        gateway = os.getenv("LLM_GATEWAY_PROVIDER", "langchain").lower()
        is_langchain = gateway != "openrouter"

        with console.status("[cyan]Fetching available models...[/]", spinner="dots"):
            models_data = fetch_available_models()

        model_ids = [m["id"] for m in models_data]

        if not model_ids and is_langchain:
            # LangChain gateway with no model list — free-form input
            console.print(
                "  [dim]Enter model as[/] [bold cyan]provider/model_name[/] "
                "[dim](e.g. openai/gpt-5-mini, anthropic/claude-sonnet-4-5-20250929)[/]"
            )
            console.print()

            model_session = PromptSession()
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

        if is_langchain:
            console.print(
                f"  [dim]{len(model_ids)} models in LLM_MODEL_LIST (type any model or pick from suggestions)[/]"
            )
            console.print(
                "  [dim]Format:[/] [bold cyan]provider/model_name[/] "
                "[dim](e.g. openai/gpt-5-mini)[/]"
            )
        else:
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
            console.print(
                "  [yellow]No browser profiles detected.[/] Continuing without one."
            )
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
            table.add_row(
                str(i), p["browser"], p["profile_directory"], p["display_name"]
            )

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
        sync_default_skills()
        load_all_dotenv()

        self._render_banner()

        # ── Model selection ──
        if "model" not in self.settings:
            self.selected_model = await self._select_model()
            self.settings["model"] = self.selected_model
            save_settings(self.settings)

        # ── Ensure provider API key is set for the selected model ──
        ensure_provider_api_key(self.selected_model)
        console.print()

        # ── Browser profile selection ──
        if "browser_profile" not in self.settings:
            self.selected_browser_profile = await self._select_browser_profile()
            self.settings["browser_profile"] = self.selected_browser_profile
            save_settings(self.settings)
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
        with console.status(
            "[cyan]Building copilot agent (this may take a moment)...[/]",
            spinner="dots",
        ):
            # LLM Config
            llm_config = LLMConfig(model=self.selected_model)

            # Browser profile kwargs
            browser_kwargs = {}
            if self.selected_browser_profile:
                browser_kwargs["browser_name"] = self.selected_browser_profile[
                    "browser"
                ]
                browser_kwargs["browser_profile_directory"] = (
                    self.selected_browser_profile["profile_directory"]
                )

            # Create backend with real-time output streaming callback
            def _on_execute_output(line: str):
                console.print(f"  [dim green]|[/] [dim]{line}[/]")

            self.backend = CiriBackend(output_callback=_on_execute_output)

            # Create copilot
            copilot = await create_copilot(
                name="Ciri",
                llm_config=llm_config,
                checkpointer=self.checkpointer,
                all_allowed=self.all_allowed,
                backend=self.backend,
                **browser_kwargs,
            )
        if is_cdp_port_open():
            console.print("  [green]✓[/] Copilot agent built")
        else:
            console.print("  [green]✓[/] Copilot agent built")
            console.print(
                "  [yellow]⚠[/] Browser CDP not available — web browsing tools disabled. "
                "Start Chrome/Edge with [cyan]--remote-debugging-port=9222[/] and restart CIRI."
            )

        # 4. Controller
        with console.status("[cyan]Finalizing controller...[/]", spinner="dots"):
            self.controller = CopilotController(graph=copilot, db=self.db)
        console.print("  [green]✓[/] Controller ready")

        # 5. Thread
        with console.status("[cyan]Resuming conversation...[/]", spinner="dots"):
            last_thread_id = self.settings.get("thread_id")
            if last_thread_id:
                self.current_thread = self.controller.get_thread(last_thread_id)

            if not self.current_thread:
                threads = self.controller.list_threads()
                if threads:
                    self.current_thread = threads[0]  # most recently updated
                else:
                    self.current_thread = self.controller.create_thread()

            if self.current_thread:
                self.settings["thread_id"] = self.current_thread["id"]
                save_settings(self.settings)

        thread_title = self.current_thread["title"]
        thread_id = self.current_thread["id"][:8] + "..."
        # Clear and render the main screen
        await asyncio.sleep(0.5)  # Brief pause to let the user see the setup results
        self._render_post_setup_screen()

        # Render existing messages
        await self._render_existing_messages()

    # ── Commands Help ─────────────────────────────────────────────────────

    def _show_commands_help(self):
        """Display commands and keyboard shortcuts in a 3-column layout."""
        console.print(Rule("[bold cyan]Available Commands & Shortcuts[/]", style="cyan"))

        table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
        table.add_column("left_cmd", style="bold cyan", width=16)
        table.add_column("left_desc", style="white")
        table.add_column("right_cmd", style="bold cyan", width=16)
        table.add_column("right_desc", style="white")
        table.add_column("shortcut_key", style="bold yellow", width=14)
        table.add_column("shortcut_desc", style="white")

        cmd_items = list(COMMANDS_HELP.items())
        shortcut_items = list(KEYBOARD_SHORTCUTS.items())
        cmd_mid = (len(cmd_items) + 1) // 2
        num_rows = max(cmd_mid, len(shortcut_items))

        for i in range(num_rows):
            row = []
            # Left command column
            if i < cmd_mid:
                cmd, desc = cmd_items[i]
                row.extend([f"  {cmd}", desc])
            else:
                row.extend(["", ""])

            # Right command column
            if i + cmd_mid < len(cmd_items):
                cmd, desc = cmd_items[i + cmd_mid]
                row.extend([f"  {cmd}", desc])
            else:
                row.extend(["", ""])

            # Keyboard shortcuts column
            if i < len(shortcut_items):
                key, desc = shortcut_items[i]
                row.extend([f"  {key}", desc])
            else:
                row.extend(["", ""])

            table.add_row(*row)

        console.print(table)

    def _render_post_setup_screen(self):
        """Clear terminal and render the main screen after setup completes."""
        if os.name == "nt":
            os.system("cls")
        else:
            # \033[H: move to home, \033[2J: clear screen, \033[3J: clear scrollback
            sys.stdout.write("\033[H\033[2J\033[3J")
            sys.stdout.flush()

        self._render_banner()

        # Centered info: Model and Thread
        thread_title = self.current_thread["title"] if self.current_thread else "None"
        thread_id = self.current_thread["id"][:8] + "..." if self.current_thread else ""

        info_text = Text()
        info_text.append("Model: ", style="bold cyan")
        info_text.append(f"{self.selected_model}", style="white")
        info_text.append("  ||  ", style="dim")
        info_text.append("Selected Thread: ", style="bold cyan")
        info_text.append(f"{thread_title} ", style="white")
        info_text.append(f"({thread_id})", style="dim")

        if self.all_allowed:
            info_text.append("  ||  ", style="dim")
            info_text.append("Full Access", style="bold red")

        console.print(Align.center(info_text))
        console.print()

        self._show_commands_help()
        console.print(Rule(style="dim cyan"))
        console.print()
        console.print()

    # ── Thread Management ─────────────────────────────────────────────────

    async def _cmd_new_thread(self):
        """Create a new thread without prompting for a title."""
        self.current_thread = self.controller.create_thread()
        self.is_new_thread = True

        self.settings["thread_id"] = self.current_thread["id"]
        save_settings(self.settings)

        # Clear terminal and render post-setup screen
        self._render_post_setup_screen()
        console.print(f"  [green]✓[/] Created new thread")
        console.print()
        # No need to render existing messages for a brand-new thread

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
            marker = (
                " [bold green]*[/]"
                if self.current_thread and t["id"] == self.current_thread["id"]
                else ""
            )
            table.add_row(
                str(i), t["title"] + marker, t["id"][:8] + "...", t["updated_at"][:19]
            )

        console.print(table)
        choice = Prompt.ask("  Select thread number", console=console)

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(threads):
                self.current_thread = threads[idx]
                self.controller.touch_thread(self.current_thread["id"])

                self.settings["thread_id"] = self.current_thread["id"]
                save_settings(self.settings)

                # Clear terminal and render post-setup screen
                self._render_post_setup_screen()
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
                    f"  Delete [bold red]{target['title']}[/]?",
                    default=False,
                    console=console,
                ):
                    self.controller.delete_thread(target["id"])
                    console.print(f"  [red]✗[/] Deleted: {target['title']}")
                    # If we deleted the current thread, switch or create new
                    if (
                        self.current_thread
                        and target["id"] == self.current_thread["id"]
                    ):
                        remaining = self.controller.list_threads()
                        if remaining:
                            self.current_thread = remaining[0]
                            self.settings["thread_id"] = self.current_thread["id"]
                            save_settings(self.settings)
                            self._render_post_setup_screen()
                            console.print(
                                f"  [green]✓[/] Switched to: [bold]{self.current_thread['title']}[/]"
                            )
                            console.print()
                            await self._render_existing_messages()
                        else:
                            self.current_thread = self.controller.create_thread()
                            self.settings["thread_id"] = self.current_thread["id"]
                            save_settings(self.settings)
                            self._render_post_setup_screen()
                            console.print("  [green]✓[/] Created new thread")
                            console.print()
                            # No need to render existing messages for a brand-new thread
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
            marker = (
                " [bold green]*[/]"
                if self.current_thread and t["id"] == self.current_thread["id"]
                else ""
            )
            table.add_row(
                str(i),
                t["title"] + marker,
                t["id"][:8] + "...",
                t["created_at"][:19],
                t["updated_at"][:19],
            )

        console.print(table)

    async def _cmd_change_model(self):
        """Change the active model and auto-rebuild the agent."""
        new_model = await self._select_model()
        if new_model != self.selected_model:
            self.selected_model = new_model
            self.settings["model"] = new_model
            save_settings(self.settings)
            console.print(f"  [green]Model set to:[/] [bold]{self.selected_model}[/]")
            await self._rebuild_agent()
        else:
            console.print("  [dim]Model unchanged.[/]")

    async def _cmd_change_browser_profile(self):
        """Change the default browser profile and auto-rebuild the agent."""
        new_profile = await self._select_browser_profile()
        # Even if new_profile is None (skipped), rebuild if it changed
        if new_profile != self.selected_browser_profile:
            self.selected_browser_profile = new_profile
            self.settings["browser_profile"] = new_profile
            save_settings(self.settings)
            if new_profile:
                console.print(
                    f"  [green]Browser profile set to:[/] [bold]{new_profile['browser']}[/] - {new_profile['display_name']}"
                )
            else:
                console.print("  [green]Browser profile cleared.[/]")
            await self._rebuild_agent()
        else:
            console.print("  [dim]Browser profile unchanged.[/]")

    async def _rebuild_agent(self):
        """Rebuild the copilot agent and controller with the current settings.

        Called automatically after model or browser profile changes.
        Creates a fresh thread so the new agent starts clean.
        """
        console.print()
        console.print(Rule("[bold cyan]Rebuilding Agent[/]", style="cyan"))
        console.print()

        # Browser profile kwargs
        browser_kwargs = {}
        if self.selected_browser_profile:
            browser_kwargs["browser_name"] = self.selected_browser_profile["browser"]
            browser_kwargs["browser_profile_directory"] = (
                self.selected_browser_profile["profile_directory"]
            )

        try:
            with console.status(
                "[cyan]Rebuilding copilot agent...[/]", spinner="dots"
            ):
                llm_config = LLMConfig(model=self.selected_model)

                def _on_execute_output(line: str):
                    console.print(f"  [dim green]|[/] [dim]{line}[/]")

                self.backend = CiriBackend(output_callback=_on_execute_output)

                copilot = await create_copilot(
                    name="Ciri",
                    llm_config=llm_config,
                    checkpointer=self.checkpointer,
                    all_allowed=self.all_allowed,
                    backend=self.backend,
                    **browser_kwargs,
                )
            console.print("  [green]✓[/] Copilot agent rebuilt")

            with console.status("[cyan]Reinitializing controller...[/]", spinner="dots"):
                self.controller = CopilotController(graph=copilot, db=self.db)
            console.print("  [green]✓[/] Controller ready")

            # Start a fresh thread so the new agent begins with a clean slate
            self.current_thread = self.controller.create_thread()
            self.is_new_thread = True
            self.settings["thread_id"] = self.current_thread["id"]
            save_settings(self.settings)
            console.print("  [green]✓[/] New thread started")

        except Exception as e:
            console.print(f"\n  [bold red]Rebuild failed:[/] {e}")
            console.print("  [yellow]The previous agent is still active.[/]")
            return

        console.print()

    # ── /sync Command ─────────────────────────────────────────────────────

    def _scan_workspace_for_sync(self) -> str:
        """Scan the workspace root and build a structured summary for the trainer.

        Excludes .ciri directories (root and nested for monorepos), .git, and
        common non-project directories. Returns a formatted string describing
        the workspace structure, or empty string if nothing meaningful is found.
        """
        root = get_default_filesystem_root()

        # Directories to always skip
        SKIP_DIRS = {
            ".ciri", ".git", ".hg", ".svn",
            "__pycache__", ".pytest_cache", ".mypy_cache", ".tox",
            "node_modules", ".npm", ".next", ".nuxt",
            ".venv", "venv", "env",
            ".idea", ".vscode",
            "build", "dist", "out", "target",
            ".gradle", ".m2", "vendor", "bower_components",
        }

        # Key project indicator files
        PROJECT_INDICATORS = {
            "package.json", "pyproject.toml", "Cargo.toml", "go.mod",
            "pom.xml", "build.gradle", "Makefile", "CMakeLists.txt",
            "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
            "Gemfile", "setup.py", "setup.cfg", "requirements.txt",
            ".github", ".gitlab-ci.yml", "Jenkinsfile",
            "README.md", "README.rst", "README.txt",
            "tsconfig.json", "webpack.config.js", "vite.config.ts",
            ".env.example", "Procfile", "Taskfile.yml",
        }

        top_level_items = []
        project_files_found = []
        dir_tree_lines = []

        try:
            for entry in sorted(root.iterdir()):
                name = entry.name
                if name in SKIP_DIRS or name.startswith(".ciri"):
                    continue

                if entry.is_file():
                    top_level_items.append(f"  {name}")
                    if name in PROJECT_INDICATORS:
                        project_files_found.append(name)
                elif entry.is_dir():
                    # Check if this subdir has its own .ciri (monorepo package)
                    has_own_ciri = (entry / ".ciri").is_dir()
                    marker = " [monorepo-package]" if has_own_ciri else ""

                    # Count immediate children (rough size indicator)
                    try:
                        child_count = sum(
                            1 for c in entry.iterdir()
                            if c.name not in SKIP_DIRS and not c.name.startswith(".")
                        )
                    except PermissionError:
                        child_count = 0

                    dir_tree_lines.append(f"  {name}/{marker} ({child_count} items)")
                    top_level_items.append(f"  {name}/")

                    # Check for project indicators inside subdirs
                    for indicator in PROJECT_INDICATORS:
                        if (entry / indicator).exists():
                            project_files_found.append(f"{name}/{indicator}")
        except PermissionError:
            pass

        return top_level_items, project_files_found, dir_tree_lines

    async def _cmd_sync(self):
        """Analyze the workspace and auto-generate a training message for Ciri."""
        root = get_default_filesystem_root()

        console.print()
        console.print(
            Panel(
                f"[bold]Scanning workspace:[/] {root}",
                style="cyan",
                padding=(0, 1),
            )
        )

        top_level_items, project_files_found, dir_tree_lines = (
            self._scan_workspace_for_sync()
        )

        # If workspace is empty or has no meaningful content
        if not project_files_found and not dir_tree_lines:
            console.print(
                "  [yellow]No meaningful project files detected in the workspace.[/]"
            )
            console.print(
                "  [dim]Ciri will ask you what you'd like to build.[/]"
            )
            console.print()

            sync_message = (
                f"I'm running /sync on my workspace at `{root}`. "
                "The workspace appears to be empty or has no recognizable project "
                "files (no package.json, pyproject.toml, Cargo.toml, Makefile, etc.). "
                "Use the `follow_up_with_human` tool to ask the user:\n"
                "1. What kind of project are they working on or planning to build?\n"
                "2. What languages/frameworks will they use?\n"
                "3. What capabilities should Ciri learn to help them?\n"
                "Then based on their answers, use the trainer_agent to create "
                "appropriate skills, toolkits, or subagents for the project."
            )
        else:
            # Build a rich workspace summary
            console.print(f"  [green]Found {len(project_files_found)} project indicator(s)[/]")
            if dir_tree_lines:
                console.print(f"  [green]Found {len(dir_tree_lines)} directories[/]")
            console.print()

            structure_summary = "## Workspace Structure\n"
            if dir_tree_lines:
                structure_summary += "### Directories\n" + "\n".join(dir_tree_lines) + "\n"

            indicators_summary = ""
            if project_files_found:
                indicators_summary = (
                    "### Project Files Detected\n"
                    + "\n".join(f"  - {f}" for f in project_files_found)
                    + "\n"
                )

            sync_message = (
                f"I'm running /sync on my workspace at `{root}`. Analyze this project "
                "and train me to work on it effectively. Here's what I found:\n\n"
                f"{structure_summary}\n{indicators_summary}\n"
                "**Your task as the trainer_agent:**\n"
                "1. Read the key project files (package.json, pyproject.toml, README, "
                "Dockerfiles, CI configs, etc.) to deeply understand the project.\n"
                "2. Identify the languages, frameworks, build system, test setup, and "
                "deployment patterns.\n"
                "3. Check what skills/toolkits/subagents already exist in .ciri/.\n"
                "4. Create or update skills that teach me:\n"
                "   - The project's conventions (code style, patterns, architecture)\n"
                "   - How to build, test, lint, and deploy\n"
                "   - Domain-specific workflows relevant to this codebase\n"
                "5. If this is a monorepo, create per-package skills as needed.\n"
                "6. If anything is unclear about the project, use `follow_up_with_human` "
                "to ask the user for clarification."
            )

        # ── Prompt for additional training direction (optional) ──────────────
        console.print(
            Panel(
                "[bold]Additional training instructions[/] [dim](optional)[/]\n"
                "[dim]Provide specific goals, focus areas, or constraints for Ciri to\n"
                "prioritise during training. Press [bold]Enter[/] to submit, "
                "[bold]Alt+Enter[/] for a new line, or leave blank to skip.[/]",
                style="cyan",
                padding=(0, 1),
            )
        )
        console.print()
        extra_instructions = await self._prompt_multiline("  Training seeds > ")
        extra_instructions = extra_instructions.strip()
        if extra_instructions:
            sync_message += (
                f"\n\n**Additional instructions from the user:**\n{extra_instructions}"
            )
        console.print()

        # Send as a regular message through the normal flow
        # This ensures it goes through the agent with full streaming/interrupt support
        console.print(
            Panel(
                "[bold cyan]Starting workspace sync...[/]\n"
                "[dim]Ciri will analyze the project and train herself accordingly.[/]",
                style="cyan",
                padding=(0, 1),
            )
        )
        console.print()

        # Auto-title the thread if new
        if self.is_new_thread:
            title = "/sync — Workspace Training"
            self.controller.rename_thread(self.current_thread["id"], title)
            self.current_thread["title"] = title
            self.is_new_thread = False

        # Stream the response like a normal message
        self.controller.touch_thread(self.current_thread["id"])
        try:
            self._streaming = True
            stream_task = asyncio.create_task(
                self._stream_response(
                    {"messages": [HumanMessage(content=sync_message)]}
                )
            )

            loop = asyncio.get_running_loop()

            def _cancel_stream():
                if not stream_task.done():
                    stream_task.cancel()

            try:
                loop.add_signal_handler(signal.SIGINT, _cancel_stream)
            except NotImplementedError:
                pass

            try:
                await stream_task
            except asyncio.CancelledError:
                self._last_ctrl_c_time = time.monotonic()
                console.print("\n  [yellow]Sync interrupted.[/]")
            finally:
                try:
                    loop.remove_signal_handler(signal.SIGINT)
                except NotImplementedError:
                    pass
        except Exception as e:
            console.print(f"\n  [bold red]Error during sync:[/] {e}")
        finally:
            self._streaming = False

    # ── Message Rendering ─────────────────────────────────────────────────

    @staticmethod
    def _render_human_message(msg: HumanMessage):
        """Render a HumanMessage in a compact panel."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        console.print(
            Panel(
                content,
                title=" [bold bright_green]You[/] ",
                title_align="left",
                border_style="bright_green",
                padding=(0, 1),
                box=box.ROUNDED,
            )
        )
        console.print()

    @staticmethod
    def _render_tool_call(tool_call: dict):
        """Render a tool call from an AIMessage in a Panel box."""
        name = tool_call.get("name", "unknown_tool")
        args = tool_call.get("args", {})
        tc_id = tool_call.get("id", "")

        # Custom rendering for specific tools
        if name == "write_todos":
            CopilotCLI._render_todo_list(args)
        elif name == "task":
            CopilotCLI._render_task_call(args)
        elif name == "simple_web_search":
            CopilotCLI._render_web_search_call(args)
        elif name in CopilotCLI._FILESYSTEM_TOOLS:
            CopilotCLI._render_filesystem_tool_call(name, args)
        else:
            args_text = json.dumps(args, indent=2, default=str) if args else "{}"
            syntax = Syntax(
                args_text, "json", theme="monokai", line_numbers=False, word_wrap=True
            )

            console.print(
                Panel(
                    syntax,
                    title=f" [bold bright_yellow]Tool Call:[/] [bold]{name}[/] [dim]({tc_id[:8]})[/] ",
                    title_align="left",
                    border_style="bright_yellow",
                    padding=(0, 1),
                    box=box.ROUNDED,
                )
            )
            console.print()

    @staticmethod
    def _render_web_search_call(args: dict):
        """Render a simple_web_search tool call."""
        query = args.get("query", "")
        console.print(f"[bold cyan]🔍 Searching the web for:[/] [italic]{query}[/]")
        console.print()

    @staticmethod
    def _render_todo_list(args: dict):
        """Render a write_todos tool call as a formatted checklist."""
        todos = args.get("todos", [])
        if not todos:
            return

        lines = []
        for item in todos:
            content = item.get("content", "")
            status = item.get("status", "pending")
            if status == "in_progress":
                indicator = "[bold cyan]◉[/]"
            elif status == "completed":
                indicator = "[bold green]✓[/]"
            else:
                indicator = "[dim]○[/]"
            lines.append(f"  {indicator} {content}")

        body = "\n".join(lines)
        console.print(
            Panel(
                body,
                title=" [bold cyan]📋 Task List[/] ",
                title_align="left",
                border_style="cyan",
                padding=(0, 1),
                box=box.ROUNDED,
            )
        )
        console.print()

    @staticmethod
    def _render_task_call(args: dict):
        """Render a task (subagent) tool call as a friendly delegation message."""
        subagent_type = args.get("subagent_type", "agent")
        description = args.get("description", "")
        console.print(
            f"[bold bright_magenta]🤖 Delegating to[/] [bold]{subagent_type}[/][bold bright_magenta]:[/] {description}"
        )
        console.print()

    # Filesystem tool names for dispatch
    _FILESYSTEM_TOOLS = {
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "execute",
    }

    # Map file extensions to Rich Syntax lexer names
    _EXT_TO_LEXER = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
        ".html": "html",
        ".css": "css",
        ".sh": "bash",
        ".bash": "bash",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".sql": "sql",
        ".xml": "xml",
        ".ini": "ini",
        ".cfg": "ini",
        ".txt": "text",
        ".csv": "text",
        ".lock": "text",
    }

    @staticmethod
    def _render_filesystem_tool_call(name: str, args: dict):
        """Render a filesystem tool call with a friendly, compact format."""
        if name == "ls":
            path = args.get("path", "/")
            console.print(f"[bold yellow]📂 Listing[/] [dim]{path}[/]")
            console.print()

        elif name == "read_file":
            fpath = args.get("file_path", "")
            offset = args.get("offset", 0)
            limit = args.get("limit", 100)
            loc = f" [dim](lines {offset + 1}–{offset + limit})[/]"
            console.print(f"[bold yellow]📄 Reading[/] [dim]{fpath}[/]{loc}")
            console.print()

        elif name == "write_file":
            fpath = args.get("file_path", "")
            content = args.get("content", "")
            lines = content.count("\n") + 1 if content else 0
            console.print(
                f"[bold yellow]📝 Creating[/] [dim]{fpath}[/] [dim]({lines} lines)[/]"
            )
            console.print()

        elif name == "edit_file":
            fpath = args.get("file_path", "")
            old_s = args.get("old_string", "")
            new_s = args.get("new_string", "")
            replace_all = args.get("replace_all", False)
            suffix = " [dim](all occurrences)[/]" if replace_all else ""

            # Build a compact diff view
            diff_lines = []
            for line in old_s.splitlines():
                diff_lines.append(f"[red]- {line}[/]")
            for line in new_s.splitlines():
                diff_lines.append(f"[green]+ {line}[/]")
            # Cap diff preview at 12 lines
            if len(diff_lines) > 12:
                diff_lines = diff_lines[:10] + [
                    f"[dim]  ... ({len(diff_lines) - 10} more lines)[/]"
                ]
            diff_body = "\n".join(diff_lines)

            console.print(
                Panel(
                    diff_body,
                    title=f"[bold yellow]✏️  Editing[/] [dim]{fpath}[/]{suffix} ",
                    title_align="left",
                    border_style="yellow",
                    padding=(0, 1),
                    box=box.ROUNDED,
                )
            )
            console.print()

        elif name == "glob":
            pattern = args.get("pattern", "")
            path = args.get("path", "")
            in_path = f" in [dim]{path}[/]" if path and path != "/" else ""
            console.print(
                f"[bold yellow]🔍 Finding files[/] [bold]{pattern}[/]{in_path}"
            )
            console.print()

        elif name == "grep":
            pattern = args.get("pattern", "")
            path = args.get("path", "")
            glob_filter = args.get("glob", "")
            mode = args.get("output_mode", "files_with_matches")
            parts = [f'  [bold yellow]🔎 Searching[/] [bold]"{pattern}"[/]']
            if path:
                parts.append(f"in [dim]{path}[/]")
            if glob_filter:
                parts.append(f"[dim]({glob_filter})[/]")
            if mode != "files_with_matches":
                parts.append(f"[dim][{mode}][/]")
            console.print(" ".join(parts))
            console.print()

        elif name == "execute":
            command = args.get("command", "")
            console.print(
                Panel(
                    f"[bold]$ {command}[/]",
                    title=" [bold yellow]⚡ Execute[/] ",
                    title_align="left",
                    border_style="yellow",
                    padding=(0, 1),
                    box=box.ROUNDED,
                )
            )
            console.print()

    def _render_filesystem_tool_response(self, msg: ToolMessage):
        """Render a filesystem tool response with appropriate formatting."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        tool_name = getattr(msg, "name", None) or ""

        # Truncate very long outputs
        max_len = 5000
        truncated = False
        if len(content) > max_len:
            content = content[:max_len]
            truncated = True

        if tool_name == "read_file":
            # Detect language from the tool_call context or content
            # Try to guess file extension from first line if it looks like a path reference
            lexer = "text"
            # The content has line numbers like "     1\t..." - still render in a panel
            # but with monospace styling
            display = content
            if truncated:
                display += f"\n[dim]... (truncated)[/]"
            console.print(
                Panel(
                    Syntax(
                        display,
                        lexer,
                        theme="monokai",
                        line_numbers=False,
                        word_wrap=True,
                    ),
                    title=f" [bold bright_green]📄 File Content[/] ",
                    title_align="left",
                    border_style="green",
                    padding=(0, 1),
                    box=box.ROUNDED,
                )
            )

        elif tool_name == "execute":
            # If output was already streamed line-by-line, show compact summary
            if hasattr(self, "backend") and self.backend._last_streamed:
                exit_line = ""
                for line in content.splitlines():
                    if line.startswith("[Command "):
                        exit_line = line
                        break
                if exit_line:
                    console.print(f"  [dim]{exit_line}[/]")
                console.print()
                self.backend._last_streamed = False
            else:
                display = content
                if truncated:
                    display += f"\n... (truncated)"
                console.print(
                    Panel(
                        Syntax(
                            display,
                            "bash",
                            theme="monokai",
                            line_numbers=False,
                            word_wrap=True,
                        ),
                        title=" [bold bright_green]⚡ Output[/] ",
                        title_align="left",
                        border_style="green",
                        padding=(0, 1),
                        box=box.ROUNDED,
                    )
                )

        elif tool_name in ("write_file", "edit_file"):
            # These return short success/error messages
            # Render as a compact status line
            if (
                "error" in content.lower()
                or "cannot" in content.lower()
                or "not found" in content.lower()
            ):
                console.print(f"  [bold red]✗[/] {content}")
                console.print()

        elif tool_name in ("ls", "glob"):
            # File listing - show as compact list
            lines = content.strip().splitlines()
            if truncated:
                lines.append(f"... (truncated)")
            display = "\n".join(lines)
            console.print(
                Panel(
                    display,
                    title=f"[bold bright_green]📂 {'Files' if tool_name == 'glob' else 'Directory'}[/] [dim]({len(lines)} items)[/] ",
                    title_align="left",
                    border_style="green",
                    padding=(0, 1),
                    box=box.ROUNDED,
                )
            )

        elif tool_name == "grep":
            lines = content.strip().splitlines()
            if truncated:
                lines.append("... (truncated)")
            display = "\n".join(lines)
            console.print(
                Panel(
                    display,
                    title=f"[bold bright_green]🔎 Search Results[/] [dim]({len(lines)} matches)[/] ",
                    title_align="left",
                    border_style="green",
                    padding=(0, 1),
                    box=box.ROUNDED,
                )
            )

        else:
            # Fallback for unknown filesystem tools
            if truncated:
                content += f"\n[dim]... (truncated)[/]"
            console.print(
                Panel(
                    content,
                    title=f"[bold bright_green]Tool Response[/] [bold]{tool_name}[/] ",
                    title_align="left",
                    border_style="bright_green",
                    padding=(0, 1),
                    box=box.ROUNDED,
                )
            )
        console.print()

    def _render_tool_message(self, msg: ToolMessage):
        """Render a ToolMessage (tool response) in a Panel box."""
        tool_name = getattr(msg, "name", None) or ""

        # Dispatch tools to custom renderers
        if tool_name in CopilotCLI._FILESYSTEM_TOOLS:
            self._render_filesystem_tool_response(msg)
        elif tool_name == "simple_web_search":
            CopilotCLI._render_web_search_response(msg)
        else:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            tc_id = getattr(msg, "tool_call_id", "") or ""

            # Truncate very long outputs
            max_len = 5000  # Increased for better visibility, but still capped
            if len(content) > max_len:
                content = (
                    content[:max_len]
                    + f"\n... [dim](truncated, {len(content)} chars total)[/]"
                )

            title_parts = ["[bold bright_green]Tool Response[/]"]
            if tool_name:
                title_parts.append(f"[bold]{tool_name}[/]")
            if tc_id:
                title_parts.append(f"[dim]({tc_id[:8]})[/]")

            console.print(
                Panel(
                    content,
                    title=f" {' '.join(title_parts)} ",
                    title_align="left",
                    border_style="bright_green",
                    padding=(0, 1),
                    box=box.ROUNDED,
                )
            )
            console.print()

    @staticmethod
    def _render_web_search_response(msg: ToolMessage):
        """Render a simple_web_search tool response in a beautiful format."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Simple parser for DuckDuckGo string output
        import re

        results = []

        # Pattern to find all snippet:, title:, link: fields
        # Note: We use a lookahead to ensure we don't consume the next field's key
        pattern = r"(snippet|title|link): (.*?)(?=, (?:snippet|title|link): |$)"
        matches = list(re.finditer(pattern, content, re.DOTALL))

        current_result = {}
        for match in matches:
            key = match.group(1)
            value = match.group(2).strip()

            # If we see a snippet and already have one, it's a new result
            if key == "snippet" and "snippet" in current_result:
                results.append(current_result)
                current_result = {}

            current_result[key] = value

        if current_result:
            results.append(current_result)

        if not results:
            # Fallback if parsing fails - render as a simpler panel
            console.print(
                Panel(
                    content,
                    title="[bold bright_green]🔍 Web Search Response[/] ",
                    title_align="left",
                    border_style="bright_green",
                    padding=(0, 1),
                    box=box.ROUNDED,
                )
            )
            return

        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        table.add_column("Result")

        for i, res in enumerate(results):
            title = res.get("title", "No Title")
            link = res.get("link", "#")
            snippet = res.get("snippet", "")

            res_text = Text()
            res_text.append(f"{i+1}. {title}\n", style="bold cyan")
            res_text.append(f"   {link}\n", style="dim underline")
            if snippet:
                # Clean up snippet (sometimes has leading dates/noise)
                if len(snippet) > 300:
                    snippet = snippet[:297] + "..."
                res_text.append(f"   {snippet}\n", style="white")

            table.add_row(res_text)
            if i < len(results) - 1:
                table.add_row(Rule(style="dim cyan"))

        console.print(
            Panel(
                table,
                title=f"[bold bright_green]🔍 Web Search Results[/] [dim]({len(results)} found)[/] ",
                title_align="left",
                border_style="bright_green",
                padding=(1, 2),
                box=box.ROUNDED,
            )
        )
        console.print()

    @staticmethod
    def _render_system_message(msg: SystemMessage):
        """Render a SystemMessage."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        console.print()
        console.print(f"[bold magenta]System >[/] [dim italic]{content}[/]")
        console.print()

    @staticmethod
    def _render_summary_panel(text: str, title: str = "💭 Thought"):
        """Render a summarization or reasoning block in a beautiful Panel."""
        if not text.strip():
            return

        console.print()
        console.print(
            Panel(
                text,
                title=f" [bold blue]{title}[/] ",
                title_align="left",
                border_style="blue",
                padding=(1, 2),
                style="italic",
                box=box.ROUNDED,
            )
        )
        console.print()

    @staticmethod
    def _render_ai_complete(content: Union[str, List[Dict[str, Any]]]):
        """Render a complete AI message (non-streamed).

        Handles both simple strings and complex content lists with reasoning blocks.
        """
        if isinstance(content, str):
            if content.strip():
                console.print(Markdown(content))
                console.print()
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
                        console.print()
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

        # Pre-pass: find the last write_todos tool call ID so we only render it once
        last_todo_tc_id = None
        for msg in messages:
            if isinstance(msg, AIMessage):
                for tc in getattr(msg, "tool_calls", []):
                    if tc.get("name") == "write_todos":
                        last_todo_tc_id = tc.get("id")

        # Collect all write_todos tool call IDs that should be suppressed
        suppressed_todo_tc_ids = set()
        for msg in messages:
            if isinstance(msg, AIMessage):
                for tc in getattr(msg, "tool_calls", []):
                    if (
                        tc.get("name") == "write_todos"
                        and tc.get("id") != last_todo_tc_id
                    ):
                        suppressed_todo_tc_ids.add(tc.get("id"))

        for msg in messages:
            if isinstance(msg, HumanMessage):
                self._render_human_message(msg)
            elif isinstance(msg, AIMessage):
                # For history, we render the full content of AI messages
                self._render_ai_complete(msg.content)

                # Also render tool calls if any (skip follow_up_with_human and old write_todos)
                for tc in getattr(msg, "tool_calls", []):
                    tc_name = tc.get("name")
                    if tc_name == "follow_up_with_human":
                        continue
                    if (
                        tc_name == "write_todos"
                        and tc.get("id") in suppressed_todo_tc_ids
                    ):
                        continue
                    self._render_tool_call(tc)
            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", None)
                # Suppress responses for: follow_up_with_human, write_todos, task
                if tool_name in ("follow_up_with_human", "write_todos", "task"):
                    continue
                # Also suppress responses for old (deduplicated) write_todos calls
                tc_id = getattr(msg, "tool_call_id", None)
                if tc_id in suppressed_todo_tc_ids:
                    continue
                self._render_tool_message(msg)
            elif isinstance(msg, SystemMessage):
                self._render_system_message(msg)

        console.print()
        console.print(Rule(style="dim cyan"))
        console.print()

    # ── Interrupt Handling ────────────────────────────────────────────────

    async def _prompt_multiline(self, label: str, default: str = "") -> str:
        """Prompt for input with multiline support (Alt+Enter for new line).

        Uses a dedicated prompt_toolkit PromptSession that shares the same
        key bindings as the main chat input so Alt+Enter inserts a newline
        and Enter submits.
        """
        session = PromptSession(
            key_bindings=self._create_input_key_bindings(),
            multiline=True,
        )
        try:
            result = await session.prompt_async(
                HTML(f"<ansicyan><b>{label}</b></ansicyan>"),
                default=default,
            )
            return result
        except (EOFError, KeyboardInterrupt):
            return default

    async def _handle_follow_up_interrupt(self, interrupt_value: dict) -> list:
        """Handle human_follow_up interrupt - ask user clarification questions."""
        queries = interrupt_value.get("queries", [])
        if not queries:
            return []

        console.print()
        console.print(
            Panel(
                "[bold]CIRI needs your input[/]",
                title="[bold cyan]Clarification Needed[/]",
                title_align="left",
                border_style="cyan",
                padding=(0, 1),
            )
        )

        responses = []
        for i, query in enumerate(queries, 1):
            question = query.get("question", "")
            options = query.get("options", [])

            console.print(f"\n  [bold cyan]Q{i}:[/] {question}")

            if options:
                for j, opt in enumerate(options, 1):
                    console.print(f"    [dim]{j}.[/] {opt}")
                console.print()

                answer = await self._prompt_multiline("  Enter number or type your answer > ")

                # Resolve numbered selection
                try:
                    opt_idx = int(answer.strip()) - 1
                    if 0 <= opt_idx < len(options):
                        answer = options[opt_idx]
                except ValueError:
                    pass

                responses.append(answer)
            else:
                answer = await self._prompt_multiline("  Your answer > ")
                responses.append(answer)

        return responses

    async def _handle_script_execution_interrupt(
        self, interrupt_value: dict
    ) -> Union[str, dict]:
        """Handle script_execution interrupt - approve/reject/edit script."""
        console.print()
        console.print(
            Panel(
                "[bold]A script wants to execute[/]",
                title="[bold cyan]Script Execution Approval[/]",
                title_align="left",
                border_style="cyan",
                padding=(0, 1),
            )
        )

        lang = interrupt_value.get("language", "python")
        deps = interrupt_value.get("dependencies", [])
        script = interrupt_value.get("script_content", "")
        timeout = interrupt_value.get("timeout", 120)

        console.print(f"  [bold]Language:[/] {lang}")
        if deps:
            console.print(f"  [bold]Dependencies:[/] {', '.join(deps)}")
        console.print(f"  [bold]Timeout:[/] {timeout}s")
        console.print()
        console.print(
            Syntax(script, lang, theme="monokai", line_numbers=True, word_wrap=True)
        )
        console.print()

        decision = Prompt.ask(
            "  [bold]Approve[/], [bold]Reject[/], or [bold]Edit[/]?",
            choices=["approve", "reject", "edit", "a", "r", "e"],
            default="approve",
            console=console,
        )

        if decision in ("reject", "r"):
            reason = await self._prompt_multiline("  Reason (optional) > ")
            return {"status": "rejected", "reason": reason}
        elif decision in ("edit", "e"):
            console.print(
                "  [dim]Edit the script below (Alt+Enter for new line, Enter to submit):[/]"
            )
            edited_script = await self._prompt_multiline("  Script > ", default=script)
            if edited_script.strip():
                return {"status": "edited", "script_content": edited_script}
            console.print("  [yellow]Empty script, approving original.[/]")

        return "approved"

    async def _handle_hitl_interrupt(self, interrupt_value: dict) -> dict:
        """Handle HumanInTheLoopMiddleware tool approval interrupts."""
        review_configs = interrupt_value.get("review_configs", [])
        action_requests = interrupt_value.get("action_requests", [])

        console.print()
        console.print(
            Panel(
                "[bold]Tool execution requires your approval[/]",
                title="[bold cyan]Tool Approval Required[/]",
                title_align="left",
                border_style="cyan",
                padding=(0, 1),
            )
        )

        decisions = []
        for i, action in enumerate(action_requests):
            name = action.get("name", "unknown")
            desc = action.get("description", "")
            args = action.get("arguments", {})

            console.print(f"\n  [bold yellow]Action {i+1}:[/] [bold]{name}[/]")
            if desc:
                console.print(f"  [dim]{desc}[/]")
            if args:
                if name == "write_file":
                    fpath = args.get("file_path", "unknown")
                    content = args.get("content", "")
                    extension = os.path.splitext(fpath)[1]
                    lexer = CopilotCLI._EXT_TO_LEXER.get(extension, "text")

                    console.print(f"[bold cyan]File:[/] [white]{fpath}[/]")
                    if content:
                        console.print(
                            Panel(
                                Syntax(
                                    content,
                                    lexer,
                                    theme="monokai",
                                    line_numbers=True,
                                    word_wrap=True,
                                ),
                                title="[bold green]File Content[/] ",
                                title_align="left",
                                border_style="green",
                                box=box.ROUNDED,
                                padding=(0, 1),
                            )
                        )
                    
                    # Show other args if any
                    other_args = {k: v for k, v in args.items() if k not in ("file_path", "content")}
                    if other_args:
                        args_text = json.dumps(other_args, indent=2, default=str)
                        console.print(f"[dim]Other Arguments:[/]")
                        console.print(
                            Syntax(
                                args_text,
                                "json",
                                theme="monokai",
                                line_numbers=False,
                                word_wrap=True,
                            )
                        )
                else:
                    args_text = json.dumps(args, indent=2, default=str)
                    console.print(
                        Syntax(
                            args_text,
                            "json",
                            theme="monokai",
                            line_numbers=False,
                            word_wrap=True,
                        )
                    )

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
                reason = await self._prompt_multiline("  Reason (optional) > ")
                d = {"type": "reject"}
                if reason:
                    d["message"] = reason
                decisions.append(d)
            elif decision == "edit":
                console.print(
                    "  [dim]Enter edited arguments as JSON (Alt+Enter for new line):[/]"
                )
                default_json = json.dumps(args, indent=2, default=str)
                raw = await self._prompt_multiline("  JSON > ", default=default_json)
                try:
                    edited_args = json.loads(raw)
                    decisions.append(
                        {
                            "type": "edit",
                            "edited_action": {"name": name, "args": edited_args},
                        }
                    )
                except json.JSONDecodeError:
                    console.print("  [yellow]Invalid JSON, approving instead.[/]")
                    decisions.append({"type": "approve"})
            else:
                decisions.append({"type": "approve"})

        return {"decisions": decisions}

    async def _handle_interrupt(self, interrupt_data: dict) -> Any:
        """Route an interrupt to the appropriate handler and return the resume value."""
        interrupt_type = interrupt_data.get("type", "")

        if interrupt_type == "human_follow_up":
            return await self._handle_follow_up_interrupt(interrupt_data)
        elif interrupt_type == "script_execution":
            return await self._handle_script_execution_interrupt(interrupt_data)
        elif "review_configs" in interrupt_data or "action_requests" in interrupt_data:
            return await self._handle_hitl_interrupt(interrupt_data)
        else:
            # Unknown interrupt - show raw and ask for JSON response
            console.print()
            console.print(
                Panel(
                    Syntax(
                        json.dumps(interrupt_data, indent=2, default=str),
                        "json",
                        theme="monokai",
                    ),
                    title="[bold yellow]Unknown Interrupt[/]",
                    title_align="left",
                    border_style="yellow",
                    padding=(0, 1),
                )
            )
            raw = await self._prompt_multiline("  Respond (JSON or text) > ")
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

        # Node-name to indicator map
        INDICATORS = {
            "researcher": "Exploring...",
            "planner": "Planning...",
            "executor": "Executing...",
            "writer": "Writing...",
            "default": "Thinking...",
        }

        try:
            # We use a context manager for the status indicator
            with console.status("[bold cyan]Thinking...[/]", spinner="dots") as status:
                async for namespace, stream_type, chunk in self.controller.run(
                    inputs, config
                ):
                    # ── messages mode: token-by-token AI streaming ──
                    if stream_type == "messages":
                        message, metadata = chunk

                        if isinstance(message, (AIMessageChunk, AIMessage)):
                            # Hide status when starting to stream tokens
                            status.stop()

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
                                            if (
                                                isinstance(s, dict)
                                                and s.get("type") == "summary_text"
                                            ):
                                                summary_text += s.get("text", "")

                                        if summary_text:
                                            # Flush if we were already streaming text
                                            if streaming_ai:
                                                sys.stdout.write("\n")
                                                sys.stdout.flush()
                                                streaming_ai = False

                                            CopilotCLI._render_summary_panel(
                                                summary_text
                                            )

                            if token:
                                if not streaming_ai:
                                    streaming_ai = True
                                    console.print()

                                # Typewrite: print each token directly to stdout
                                sys.stdout.write(token)
                                sys.stdout.flush()
                                ai_buffer += token

                    # ── updates mode: state updates from nodes ──
                    elif stream_type == "updates":
                        for node_name, node_value in chunk.items():
                            # Update status based on node name
                            if not streaming_ai:
                                # Try to find a specific indicator
                                indicator = INDICATORS.get("default")
                                for key in INDICATORS:
                                    if key in node_name.lower():
                                        indicator = INDICATORS[key]
                                        break
                                status.update(f"[bold cyan]{indicator}[/]")
                                if not status._live.is_started:
                                    status.start()

                            # Check for __interrupt__
                            if node_name == "__interrupt__":
                                if isinstance(node_value, (list, tuple)):
                                    pending_interrupts.extend(node_value)
                                else:
                                    pending_interrupts.append(node_value)
                                continue

                            # Process messages in the update
                            if (
                                isinstance(node_value, dict)
                                and "messages" in node_value
                            ):
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
                                    elif isinstance(msg, AIMessage) and not isinstance(
                                        msg, AIMessageChunk
                                    ):
                                        # Full AI message from updates - render tool_calls only
                                        # Skip follow_up_with_human (handled via interrupt UI)
                                        for tc in getattr(msg, "tool_calls", []):
                                            if tc.get("name") != "follow_up_with_human":
                                                self._render_tool_call(tc)
                                    elif isinstance(msg, ToolMessage):
                                        # Skip responses for tools with custom rendering or interrupt UI
                                        tool_name = getattr(msg, "name", None)
                                        if tool_name not in (
                                            "follow_up_with_human",
                                            "write_todos",
                                            "task",
                                        ):
                                            self._render_tool_message(msg)
                                    elif isinstance(msg, SystemMessage):
                                        self._render_system_message(msg)

        finally:
            # Flush remaining AI buffer
            if streaming_ai and ai_buffer.strip():
                sys.stdout.write("\n")
                sys.stdout.flush()

            # Ensure status is stopped
            # status is automatically stopped by the with block

        # Handle any interrupts that were collected
        if pending_interrupts:
            await self._process_interrupts(pending_interrupts)

    async def _process_interrupts(self, interrupts: list):
        """Process collected interrupts and resume the graph."""
        for intr in interrupts:
            # Extract the actual interrupt value from Interrupt dataclass or dict
            if isinstance(intr, Interrupt):
                interrupt_data = intr.value
            elif isinstance(intr, dict) and "value" in intr:
                interrupt_data = intr["value"]
            else:
                interrupt_data = intr

            # Render interrupt indicator
            console.print()
            console.print(Rule("[bold cyan]Interrupt[/]", style="cyan"))

            # Handle the interrupt
            resume_value = await self._handle_interrupt(interrupt_data)

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
                self._last_ctrl_c_time = 0.0  # Reset on successful input
            except EOFError:
                console.print("\n  [dim]Goodbye![/]")
                break
            except KeyboardInterrupt:
                now = time.monotonic()
                if now - self._last_ctrl_c_time < self._DOUBLE_CTRL_C_THRESHOLD:
                    # Double Ctrl+C → terminate
                    console.print("\n  [dim]Goodbye![/]")
                    break
                self._last_ctrl_c_time = now
                console.print(
                    "\n  [dim]Press Ctrl+C again to exit.[/]"
                )
                continue

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
                elif cmd == "/change-browser-profile":
                    await self._cmd_change_browser_profile()
                elif cmd == "/sync":
                    await self._cmd_sync()
                else:
                    console.print(
                        f"[yellow]Unknown command: {cmd}[/]. Type /help for available commands."
                    )

                console.print()
                continue

            # ── Send message ──
            self.controller.touch_thread(self.current_thread["id"])

            # ── Auto-generate title for new thread ──
            if self.is_new_thread:
                # Clean up input for title: first line, max 40 chars
                first_line = user_input.split("\n")[0].strip()
                title = (
                    (first_line[:37] + "...") if len(first_line) > 40 else first_line
                )
                if not title:
                    title = "New Thread"

                self.controller.rename_thread(self.current_thread["id"], title)
                self.current_thread["title"] = title
                self.is_new_thread = False

                # Update info line to show new title
                console.print(f"  [dim]Thread renamed to:[/] [bold cyan]{title}[/]")
                console.print()

            try:
                self._streaming = True
                stream_task = asyncio.create_task(
                    self._stream_response(
                        {"messages": [HumanMessage(content=user_input)]}
                    )
                )

                loop = asyncio.get_running_loop()

                def _cancel_stream():
                    if not stream_task.done():
                        stream_task.cancel()

                try:
                    loop.add_signal_handler(signal.SIGINT, _cancel_stream)
                except NotImplementedError:
                    pass  # Windows doesn't support signal handlers in asyncio

                try:
                    await stream_task
                except asyncio.CancelledError:
                    # Single Ctrl+C during streaming → stop and return to prompt
                    self._last_ctrl_c_time = time.monotonic()
                    console.print("\n  [yellow]Interrupted.[/]")
                finally:
                    try:
                        loop.remove_signal_handler(signal.SIGINT)
                    except NotImplementedError:
                        pass
            except Exception as e:
                console.print(f"\n  [bold red]Error:[/] {e}")
            finally:
                self._streaming = False

            console.print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main():
    """Entry point for the CIRI CLI."""
    parser = argparse.ArgumentParser(
        description="CIRI - Contextual Intelligence and Reasoning Interface"
    )
    parser.add_argument(
        "--all-allowed",
        action="store_true",
        help="Pass all_allowed=True to the copilot (disables some interrupts)",
    )
    args = parser.parse_args()

    cli = CopilotCLI(all_allowed=args.all_allowed)
    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        console.print("\n[dim]Exiting CIRI...[/]")


if __name__ == "__main__":
    main()
