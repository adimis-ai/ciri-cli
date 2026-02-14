import os
import sys
import json
import uuid
import httpx
import asyncio
import aiosqlite
import subprocess
from pathlib import Path
from typing import Optional, List, Any, Dict, Union, Iterable

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
from langgraph.types import Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessageChunk,
    ToolMessage,
)

# Copilot
from .db import CopilotDatabase
from .utils import get_app_data_dir
from .copilot import create_copilot
from .controller import CopilotController


"""
in CopilotCLI.setup method add a modern big cli banner named like:
```
CIRI: Contextual Intelligence and Reasoning Interface
======================================================
<Setup logs...>
======================================================
<Messages List...>
======================================================
<Chat Input Box / Interrupt Value>
```
"""
class CopilotCLI:
    def __init__(self):
        self.db: Optional[CopilotDatabase] = None
        self.checkpointer: Optional[AsyncSqliteSaver] = None
        self.controller: Optional[CopilotController] = None

    async def setup(self):
        db_path = get_app_data_dir() / "ciri.db"
        
        # Setup thread management DB
        self.db = CopilotDatabase(db_path=db_path)
        
        # Setup LangGraph checkpointer
        self.checkpointer = AsyncSqliteSaver.from_conn_string(str(db_path))
        
        copilot = await create_copilot(
            name="Ciri",
            checkpointer=self.checkpointer
        )

        # Setup controller
        self.controller = CopilotController(graph=copilot, db=self.db)
