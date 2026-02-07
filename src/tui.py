import os
import sys
import asyncio
import platform
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, List

from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal, Container, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Input,
    Label,
    Static,
    Button,
    LoadingIndicator,
    ListView,
    ListItem,
)
from textual.binding import Binding
from textual.worker import Worker, WorkerState
from textual import on, work
from textual.reactive import reactive
from rich.markdown import Markdown
from rich.text import Text
from rich.panel import Panel

from .agent import Ciri, LLMConfig
from .controllers import CiriController, CiriConfig

# Next-Gen CSS for Ciri
CSS = """
$accent: #00f2ff;
$bg: #0a0a0c;
$surface: #121217;
$panel-bg: rgba(20, 20, 25, 0.8);
$text: #e0e0e0;
$border: #30303b;

Screen {
    background: $bg;
    color: $text;
}

#sidebar {
    width: 35;
    dock: left;
    background: $surface;
    border-right: tall $accent;
    height: 1fr;
}

#sidebar-title {
    padding: 1;
    background: $accent;
    color: $bg;
    text-align: center;
    text-style: bold;
}

#new-thread-btn {
    width: 1fr;
    margin: 1;
}

#thread-list {
    background: $surface;
}

.thread-item {
    padding: 1;
    border-bottom: thin $border;
}

.thread-item:hover {
    background: #1e1e24;
}

.thread-item.selected {
    background: #2563eb;
    color: white;
}

#main-container {
    height: 1fr;
    padding: 1;
}

#chat-history {
    height: 1fr;
    border: none;
    padding: 1;
}

.message {
    margin: 1 0;
    padding: 1 2;
    border-radius: 4;
}

.user-message {
    background: #1e293b;
    border-left: thick $accent;
}

.ai-message {
    background: #111827;
    border-left: thick #8b5cf6;
}

.system-message {
    color: #64748b;
    text-align: center;
    border: none;
}

#input-area {
    height: 6;
    dock: bottom;
    padding: 1;
    background: $panel-bg;
    border-top: tall $border;
}

Input {
    border: tall $border;
    background: $surface;
    color: $text;
}

Input:focus {
    border: tall $accent;
}

#interrupt-panel {
    dock: bottom;
    height: auto;
    background: #1e1e24;
    border-top: thick #ff0055;
    padding: 1;
    display: none;
}

#interrupt-panel > Horizontal {
    align: center middle;
    height: 3;
}

.interrupt-btn {
    margin: 0 1;
}

#loader {
    width: auto;
    height: 1;
    color: $accent;
    display: none;
}
"""


class ThreadItem(ListItem):
    """A widget to represent a thread in the sidebar."""

    def __init__(self, thread_dict: dict):
        super().__init__()
        self.thread_id = thread_dict["id"]
        self.thread_title = thread_dict.get("title") or f"Thread {self.thread_id[:8]}"
        self.add_class("thread-item")

    def compose(self) -> ComposeResult:
        yield Label(self.thread_title)


class Message(Static):
    """A widget to represent a chat message."""

    def __init__(self, content: str, role: str):
        super().__init__()
        self.content = content
        self.role = role
        self.add_class("message")
        self.add_class(f"{role}-message")

    def render(self) -> Markdown:
        return Markdown(self.content or "")


class CiriApp(App):
    """Ciri Textual Interface."""

    CSS = CSS
    TITLE = "CIRI // CO-PILOT"
    SUBTITLE = "Next-Gen AI Interface"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_chat", "Clear Chat", show=True),
        Binding("ctrl+b", "toggle_sidebar", "Toggle Sidebar", show=True),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.controller: Optional[CiriController] = None
        self.current_thread_id: Optional[str] = None
        self.is_streaming = False

    def get_root_dir(self) -> Path:
        """Determine root directory based on OS."""
        system = platform.system()
        user_home = Path.home()

        if system == "Windows":
            root = user_home / "Documents" / "Ciri"
        elif system == "Darwin":
            root = user_home / "Library" / "Application Support" / "Ciri"
        else:  # Linux and others
            root = user_home / ".local" / "share" / "ciri"

        root.mkdir(parents=True, exist_ok=True)
        return root

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label("HISTORY", id="sidebar-title")
                yield Button("New Chat", variant="primary", id="new-thread-btn")
                yield ListView(id="thread-list")

            with Container(id="main-container"):
                with ScrollableContainer(id="chat-history"):
                    yield Message("Initializing Ciri Core...", "system")

                yield LoadingIndicator(id="loader")

                with Vertical(id="interrupt-panel"):
                    yield Label(
                        "Interrupt Detected: What would you like to do?",
                        id="interrupt-label",
                    )
                    with Horizontal():
                        yield Button(
                            "Approve",
                            variant="success",
                            id="btn-approve",
                            classes="interrupt-btn",
                        )
                        yield Button(
                            "Edit",
                            variant="warning",
                            id="btn-edit",
                            classes="interrupt-btn",
                        )
                        yield Button(
                            "Reject",
                            variant="error",
                            id="btn-reject",
                            classes="interrupt-btn",
                        )

                with Horizontal(id="input-area"):
                    yield Input(placeholder="Ask Ciri anything...", id="user-input")
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize CiriController."""
        self.run_worker(self.initialize_agent(), thread=True)

    @work(exclusive=True)
    async def initialize_agent(self) -> None:
        """Background worker to initialize the agent."""
        root_dir = self.get_root_dir()
        self.call_from_thread(self.update_status, f"Root: {root_dir}")

        try:
            config = CiriConfig.from_env()
            if not config.sqlite_url:
                db_path = root_dir / "ciri.db"
                config.sqlite_url = f"sqlite:///{db_path}"

            self.controller = CiriController(config=config)

            # Default Ciri setup
            ciri = Ciri(
                llm_config=LLMConfig(
                    model=os.getenv(
                        "CIRI_MODEL", "openrouter/openai/gpt-4o-mini-2024-07-18"
                    )
                ),
                instructions="You are helpful and concise.",
            )

            self.controller.compile(ciri)
            self.call_from_thread(self.refresh_thread_list)

            threads = self.controller.list_threads()
            if threads:
                self.call_from_thread(self.switch_thread, threads[0]["id"])
            else:
                self.call_from_thread(self.action_new_thread)

            self.call_from_thread(
                self.post_message_to_history,
                "Ciri Core Online. Ready for commands.",
                "system",
            )
        except Exception as e:
            self.call_from_thread(
                self.post_message_to_history, f"Failed to initialize: {e}", "system"
            )

    def update_status(self, text: str) -> None:
        self.sub_title = text

    def refresh_thread_list(self) -> None:
        """Fetch threads from controller and update UI."""
        thread_list_view = self.query_one("#thread-list", ListView)
        thread_list_view.clear()

        threads = self.controller.list_threads()
        for t in threads:
            thread_list_view.append(ThreadItem(t))

    @on(Button.Pressed, "#new-thread-btn")
    def action_new_thread(self) -> None:
        new_id = str(uuid.uuid4())
        self.controller.create_thread(new_id)
        self.refresh_thread_list()
        self.switch_thread(new_id)

    @on(ListView.Selected, "#thread-list")
    def on_thread_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, ThreadItem):
            self.switch_thread(event.item.thread_id)

    def switch_thread(self, thread_id: str) -> None:
        """Switch to a different thread and load its history."""
        self.current_thread_id = thread_id
        history = self.query_one("#chat-history", ScrollableContainer)
        history.clear()
        self.load_history()
        self.update_status(f"Thread: {thread_id[:8]}")

    def post_message_to_history(self, content: str, role: str) -> Message:
        history = self.query_one("#chat-history", ScrollableContainer)
        msg = Message(content, role)
        history.mount(msg)
        msg.scroll_visible()
        return msg

    def load_history(self) -> None:
        """Load conversation history for the current thread."""
        if not self.controller or not self.current_thread_id:
            return

        state = self.controller.compiled_ciri.get_state(
            {"configurable": {"thread_id": self.current_thread_id}}
        )
        if state and state.values and "messages" in state.values:
            for msg in state.values["messages"]:
                role = "ai" if msg.type == "ai" else "user"
                self.post_message_to_history(msg.content, role)

    @on(Input.Submitted, "#user-input")
    async def handle_submit(self, event: Input.Submitted) -> None:
        content = event.value.strip()
        if not content or self.is_streaming:
            return

        event.input.value = ""
        self.post_message_to_history(content, "user")
        self.run_worker(self.process_chat(content), thread=True)

    @work(exclusive=True)
    async def process_chat(self, user_input: str) -> None:
        """Process chat in background thread."""
        self.is_streaming = True
        self.call_from_thread(self.show_loader, True)

        try:
            input_state = {"messages": [{"role": "user", "content": user_input}]}
            config = {"configurable": {"thread_id": self.current_thread_id}}

            for event in self.controller.stream(input_state, config=config):
                self.call_from_thread(self.handle_event, event)

            # After successful interaction, update thread timestamp
            # We don't have a specific title generation logic here, but we update updated_at
            self.controller.update_thread_title(
                self.current_thread_id, f"Chat {datetime.now().strftime('%H:%M')}"
            )
            self.call_from_thread(self.refresh_thread_list)

        except Exception as e:
            self.call_from_thread(self.post_message_to_history, f"Error: {e}", "system")
        finally:
            self.is_streaming = False
            self.call_from_thread(self.show_loader, False)

    def show_loader(self, show: bool) -> None:
        self.query_one("#loader").display = show

    def handle_event(self, event: dict) -> None:
        """Handle stream events from CiriController."""
        event_type = event.get("type")
        data = event.get("data")

        if event_type == "update":
            # Node updates
            for node, output in data.items():
                if "messages" in output:
                    for msg in output["messages"]:
                        if hasattr(msg, "content") and msg.content:
                            self.post_message_to_history(msg.content, "ai")
        elif event_type == "interrupt":
            self.show_interrupt(data)

    def show_interrupt(self, data: Any) -> None:
        """Display the interrupt panel."""
        panel = self.query_one("#interrupt-panel")
        label = self.query_one("#interrupt-label")

        # Display interrupt details
        if isinstance(data, list) and len(data) > 0:
            interrupt_val = data[0].get("value")
            if "action_requests" in interrupt_val:
                req = interrupt_val["action_requests"][0]
                label.update(
                    f"Action Required: **{req['name']}**\n{req['description']}"
                )
            elif "question" in interrupt_val:
                label.update(f"Follow-up Question: {interrupt_val['question']}")

        panel.display = True
        self.query_one("#user-input").disabled = True

    @on(Button.Pressed, "#btn-approve")
    def handle_approve(self) -> None:
        self.submit_resume({"decisions": [{"type": "approve"}]})

    @on(Button.Pressed, "#btn-reject")
    def handle_reject(self) -> None:
        self.submit_resume(
            {"decisions": [{"type": "reject", "message": "User rejected the action."}]}
        )

    @on(Button.Pressed, "#btn-edit")
    def handle_edit(self) -> None:
        self.post_message_to_history(
            "Edit functionality not yet fully implemented in TUI.", "system"
        )

    def submit_resume(self, resume_data: Any) -> None:
        self.query_one("#interrupt-panel").display = False
        self.query_one("#user-input").disabled = False
        self.run_worker(self.resume_agent(resume_data), thread=True)

    @work(exclusive=True)
    async def resume_agent(self, resume_data: Any) -> None:
        self.is_streaming = True
        self.call_from_thread(self.show_loader, True)
        try:
            config = {"configurable": {"thread_id": self.current_thread_id}}
            for event in self.controller.stream({"resume": resume_data}, config=config):
                self.call_from_thread(self.handle_event, event)
        except Exception as e:
            self.call_from_thread(
                self.post_message_to_history, f"Resume Error: {e}", "system"
            )
        finally:
            self.is_streaming = False
            self.call_from_thread(self.show_loader, False)

    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar")
        sidebar.display = not sidebar.display

    def action_clear_chat(self) -> None:
        self.query_one("#chat-history", ScrollableContainer).clear()


if __name__ == "__main__":
    app = CiriApp()
    app.run()
