from langgraph.graph import MessagesState
from langgraph.types import StateSnapshot, Command
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from typing import Any, Dict, List, Optional, AsyncGenerator, Union

from .db import CiriDatabase
from .serializers import CiriSerializer


class CiriController:
    """
    Controller for CIRI that handles graph execution and thread management.
    Separates the core logic from the user interface.
    """

    def __init__(self, graph: CompiledStateGraph, db: Optional[CiriDatabase] = None):
        """
        Initialize the controller.

        Args:
            graph: The compiled LangGraph instance.
            db: The CiriDatabase instance for thread management.
        """
        self.graph = graph
        self.db = db

    async def run(
        self,
        inputs: Union[MessagesState, Command],
        config: RunnableConfig,
        *,
        context: Optional[Any] = None,
        subgraphs: Optional[bool] = True,
        serialize: bool = False,
    ) -> AsyncGenerator[tuple[tuple[str, ...], str, Any], None]:
        """
        Execute the graph and yield stream events.

        Args:
            inputs: Graph inputs (e.g., {"messages": [...]} or Command(resume=...)).
            config: Execution config (e.g., {"configurable": {"thread_id": "..."}}).
            context: Optional context to pass to the graph.
            subgraphs: Whether to include subgraphs in the stream.
            serialize: Whether to serialize output chunks to JSON-compatible formats.

        Yields:
            tuple: (namespace, stream_type, chunk)
        """
        # Deserialization of inputs if they are messages in dict form
        if isinstance(inputs, dict) and "messages" in inputs:
            inputs["messages"] = [
                CiriSerializer.deserialize_any_message(m) if isinstance(m, dict) else m
                for m in inputs["messages"]
            ]

        async for namespace, stream_type, chunk in self.graph.astream(
            inputs,
            config,
            stream_mode=["updates", "messages"],
            subgraphs=subgraphs,
            context=context,
        ):
            if serialize:
                if stream_type == "messages":
                    # chunk is (message, metadata)
                    message, metadata = chunk
                    yield namespace, stream_type, (
                        CiriSerializer.serialize_any_message(message),
                        metadata,
                    )
                elif stream_type == "updates":
                    # chunk is {node: update}
                    serialized_chunk = {
                        node: CiriSerializer._serialize_value(val)
                        for node, val in chunk.items()
                    }
                    yield namespace, stream_type, serialized_chunk
                else:
                    yield namespace, stream_type, CiriSerializer._serialize_value(chunk)
            else:
                yield namespace, stream_type, chunk

    async def get_state(self, config: RunnableConfig, serialize: bool = False) -> Any:
        """Get the current state of the graph."""
        state = await self.graph.aget_state(config)
        if serialize:
            return CiriSerializer.serialize_state_snapshot(state)
        return state

    async def get_state_history(
        self,
        config: RunnableConfig,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[str] = None,
        limit: Optional[int] = None,
        serialize: bool = False,
    ) -> AsyncGenerator[Union[StateSnapshot, Dict[str, Any]], None]:
        """Get the current state of the graph."""
        async for snapshot in self.graph.aget_state_history(
            config, filter, before, limit
        ):
            if serialize:
                yield CiriSerializer.serialize_state_snapshot(snapshot)
            else:
                yield snapshot

    # --- Thread Management ---

    def list_threads(self) -> List[Dict[str, Any]]:
        """List all threads."""
        if not self.db:
            return []
        return self.db.list_threads()

    def create_thread(self) -> Dict[str, Any]:
        """Create a new thread."""
        if not self.db:
            raise ValueError("Database not initialized")
        return self.db.create_thread()

    def delete_thread(self, thread_id: str) -> None:
        """Delete a thread."""
        if not self.db:
            return
        self.db.delete_thread(thread_id)

    def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get a thread by ID."""
        if not self.db:
            return None
        return self.db.get_thread(thread_id)

    def rename_thread(self, thread_id: str, title: str) -> None:
        """Rename a thread."""
        if not self.db:
            return
        self.db.rename_thread(thread_id, title)

    def touch_thread(self, thread_id: str) -> None:
        """Update the touch time for a thread."""
        if not self.db:
            return
        self.db.touch_thread(thread_id)
