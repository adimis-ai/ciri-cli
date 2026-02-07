import os
import sqlite3
import logging
from dotenv import load_dotenv
from typing import Optional, Any, Iterator
from dataclasses import dataclass


from langgraph.graph import MessagesState
from langgraph.store.sqlite import SqliteStore
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from .agent import Ciri, ResumeCommand
from .serializers import CiriJsonPlusSerializer

load_dotenv()
logger = logging.getLogger(__name__)


def _parse_sqlite_url(url: str) -> str:
    """Parse SQLite URL and return the database file path.

    Args:
        url: SQLite URL in format like 'sqlite://path/to/db.db' or just 'path/to/db.db'

    Returns:
        str: The database file path
    """
    if url.startswith("sqlite://"):
        # Extract the path after sqlite://
        db_path = url[9:]  # Remove 'sqlite://' prefix
        # Remove query parameters if present
        if "?" in db_path:
            db_path = db_path.split("?")[0]
    else:
        db_path = url

    # Convert relative path to absolute path
    if not os.path.isabs(db_path):
        # Make it relative to the project root (src-copilot directory)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_root, db_path)

    # Ensure the directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Created database directory: {db_dir}")

    return db_path


@dataclass
class CiriConfig:
    """Configuration for CiriController."""

    sqlite_url: Optional[str] = None
    embedding_dims: int = 1024
    embedding_index_fields: str = "$"

    @classmethod
    def from_env(cls) -> "CiriConfig":
        """Create configuration from environment variables."""
        return cls(
            sqlite_url=os.getenv("SQLITE_URL"),
            embedding_dims=int(os.getenv("EMBEDDING_DIMS", "1024")),
            embedding_index_fields=os.getenv("EMBEDDING_INDEX_FIELDS", "$"),
        )

    def validate(self) -> None:
        """Validate configuration values."""
        if not self.sqlite_url:
            raise ValueError(
                "SQLite URL is required. Set SQLITE_URL environment variable or provide sqlite_url."
            )


class _CacheManager:
    """Manages singleton instances for embeddings, store, and checkpointer."""

    def __init__(self):
        self._embeddings = None
        self._store = None
        self._store_context = None
        self._checkpointer = None
        self._checkpointer_context = None
        self._initialized = False

    def get_embeddings(self):
        """Get or create the cached embeddings instance."""
        if self._embeddings is None:
            from .utils import initialize_embeddings

            logger.info(
                "Initializing embeddings (this may take a moment on first load)..."
            )
            self._embeddings = initialize_embeddings()
            logger.info("Embeddings initialized and cached")
        return self._embeddings

    def get_store(self, sqlite_url: str, config: CiriConfig) -> SqliteStore:
        """Get or create the cached SqliteStore instance."""
        if self._store is None or self._store_context != sqlite_url:
            logger.info("Initializing SqliteStore...")
            try:
                db_path = _parse_sqlite_url(sqlite_url)
                logger.info(f"Using SQLite database at: {db_path}")
                conn = sqlite3.connect(
                    db_path,
                    check_same_thread=False,
                    isolation_level=None,  # autocommit mode
                )
                self._store = SqliteStore(
                    conn,
                    # index={
                    #     "embed": self.get_embeddings(),
                    #     "dims": config.embedding_dims,
                    #     "fields": config.embedding_index_fields.split(","),
                    # },
                )
                # Setup the store after creation
                self._store.setup()
                self._store_context = sqlite_url
                logger.info("SqliteStore initialized and cached")
            except Exception as e:
                logger.error(f"Failed to initialize SqliteStore: {e}")
                raise
        return self._store

    def get_checkpointer(self, sqlite_url: str) -> SqliteSaver:
        """Get or create the cached SqliteSaver instance."""
        if self._checkpointer is None or self._checkpointer_context != sqlite_url:
            logger.info("Initializing SqliteSaver...")
            try:
                db_path = _parse_sqlite_url(sqlite_url)
                logger.info(f"Using SQLite database at: {db_path}")
                conn = sqlite3.connect(
                    db_path,
                    check_same_thread=False,
                )
                # Use our custom serializer that can handle Send objects
                custom_serde = CiriJsonPlusSerializer(pickle_fallback=False)
                self._checkpointer = SqliteSaver(conn, serde=custom_serde)
                # Setup the checkpointer after creation
                self._checkpointer.setup()
                self._checkpointer_context = sqlite_url
                logger.info("SqliteSaver initialized with custom serializer and cached")
            except Exception as e:
                logger.error(f"Failed to initialize SqliteSaver: {e}")
                raise
        return self._checkpointer

    def setup_components(
        self, sqlite_url: str, config: CiriConfig
    ) -> tuple[SqliteStore, SqliteSaver]:
        """Setup and return cached SqliteStore and SqliteSaver instances."""
        if not self._initialized:
            store = self.get_store(sqlite_url, config)
            checkpointer = self.get_checkpointer(sqlite_url)
            self._initialized = True
            return store, checkpointer
        else:
            return self._store, self._checkpointer


# Global cache manager instance
_cache_manager = _CacheManager()


class CiriController:
    """Controller for managing Ciri agent instances and their lifecycle."""

    def __init__(
        self,
        config: Optional[CiriConfig] = None,
        sqlite_url: Optional[str] = None,
    ):
        """Initialize the CiriController.

        Args:
            config: Configuration instance. If None, will create from env vars.
        """
        # Handle backward compatibility
        if config is None:
            config = CiriConfig.from_env()
            if sqlite_url is not None:
                config.sqlite_url = sqlite_url

        config.validate()

        self.config = config
        self.ciri: Optional[Ciri] = None
        self.compiled_ciri: Optional[CompiledStateGraph] = None
        self._store: Optional[SqliteStore] = None
        self._checkpointer: Optional[SqliteSaver] = None
        self._cache: Optional[InMemoryCache] = None

        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize store, checkpointer, and cache components."""
        store, checkpointer = _cache_manager.setup_components(
            self.config.sqlite_url, self.config
        )
        self._store = store
        self._checkpointer = checkpointer
        self._cache = InMemoryCache()

    def _ensure_compiled(self) -> None:
        """Ensure the controller is compiled before operations."""
        if not self.compiled_ciri:
            raise ValueError(
                "CiriController is not compiled. Call compile() before using this method."
            )

    def _prepare_input(self, input: Any) -> Any:
        """Prepare and validate input for agent operations."""
        if isinstance(input, dict) and "resume" in input:
            return Command(resume=input["resume"])
        return input

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures resources are cleaned up."""
        self.close()
        return False

    def close(self) -> None:
        """Clean up resources and references.

        Properly close database connections and stop background threads.
        """
        if self._store is not None:
            try:
                if hasattr(self._store, "stop_ttl_sweeper"):
                    self._store.stop_ttl_sweeper()
                # Close the database connection
                if hasattr(self._store, "conn") and self._store.conn:
                    self._store.conn.close()
            except Exception as e:
                logger.warning(f"Error cleaning up SqliteStore: {e}")
            self._store = None

        if self._checkpointer is not None:
            try:
                # Close the database connection
                if hasattr(self._checkpointer, "conn") and self._checkpointer.conn:
                    self._checkpointer.conn.close()
            except Exception as e:
                logger.warning(f"Error cleaning up SqliteSaver: {e}")
            self._checkpointer = None

        self._cache = None
        self.compiled_ciri = None

    def compile(
        self,
        ciri: Ciri,
        *,
        debug: bool = False,
        context_schema: Optional[Any] = None,
        tools: Optional[list] = None,
        response_format: Optional[Any] = None,
        middleware: Optional[list] = None,
        consciousness_middleware: Optional[list] = None,
    ) -> None:
        """Compile the Ciri agent with configured components.

        Args:
            ciri: The Ciri agent instance to compile.
            debug: Enable debug mode for the agent.
            context_schema: Optional context schema for the agent.
            tools: Optional additional tools to include.
            response_format: Optional response format (Pydantic model).
            middleware: Optional additional middleware.
            consciousness_middleware: Optional middleware for consciousness layer.
        """
        self.ciri = ciri

        # Compile the Ciri agent with store, checkpointer, and optional cache
        self.compiled_ciri = self.ciri.compile(
            store=self._store,
            checkpointer=self._checkpointer,
            debug=debug,
            cache=self._cache,
            context_schema=context_schema,
            tools=tools,
            response_format=response_format,
            middleware=middleware,
            consciousness_middleware=consciousness_middleware,
        )

    def stream(
        self,
        input: MessagesState | ResumeCommand | None,
        config: RunnableConfig | None = None,
        *,
        context: Any = None,
        subgraphs: bool = True,
        stream_mode: list[str] | str | None = None,
    ) -> Iterator[dict]:
        """Stream responses from the compiled Ciri agent.

        Args:
            input: Input for the agent.
            config: Configuration for the run.
            context: Additional context.
            subgraphs: Include subgraph information.
            stream_mode: Streaming mode(s).

        Yields:
            Formatted stream items with type, data, and optional namespace.
        """
        self._ensure_compiled()

        input = self._prepare_input(input)

        # Default stream mode if not provided
        if stream_mode is None:
            stream_mode = ["updates", "values"]

        for stream_item in self.compiled_ciri.stream(
            input=input,
            config=config,
            context=context,
            subgraphs=subgraphs,
            stream_mode=stream_mode,
        ):
            yield self._format_stream_item(stream_item, subgraphs, stream_mode)

    def _format_stream_item(
        self, stream_item: Any, subgraphs: bool, stream_mode: list[str] | str
    ) -> dict:
        """Format a stream item into a consistent output format."""
        try:
            # Handle different tuple structures based on subgraphs and stream_mode
            if subgraphs and isinstance(stream_mode, list):
                # 3-tuple: (namespace, mode, payload)
                namespace, mode, chunk = stream_item
            elif isinstance(stream_mode, list):
                # 2-tuple: (mode, payload)
                mode, chunk = stream_item
                namespace = None
            elif subgraphs:
                # 2-tuple: (namespace, payload)
                namespace, chunk = stream_item
                mode = stream_mode
            else:
                # Just payload
                chunk = stream_item
                mode = stream_mode
                namespace = None

            if mode == "updates":
                if chunk is not None and "__interrupt__" in chunk:
                    return {
                        "type": "interrupt",
                        "data": chunk["__interrupt__"],
                        "namespace": namespace,
                    }
                else:
                    return {
                        "type": "update",
                        "data": chunk,
                        "namespace": namespace,
                    }
            elif mode == "values":
                return {
                    "type": "messages",
                    "data": chunk,
                    "namespace": namespace,
                }

            # Fallback for unknown modes
            return {
                "type": "unknown",
                "data": chunk,
                "namespace": namespace,
            }
        except Exception as e:
            # If unpacking or formatting fails, return an error
            logger.error(f"Failed to format stream item: {e}, item: {stream_item}")
            return {
                "type": "error",
                "message": f"Failed to format stream item: {str(e)}",
                "raw_item": str(stream_item),
            }

    def invoke(
        self,
        input: MessagesState | ResumeCommand | None,
        config: RunnableConfig | None = None,
        *,
        context: Any = None,
    ) -> Any:
        """Invoke the compiled Ciri agent synchronously.

        Args:
            input: Input for the agent.
            config: Configuration for the run.
            context: Additional context.

        Returns:
            The agent's response.
        """
        self._ensure_compiled()
        input = self._prepare_input(input)

        return self.compiled_ciri.invoke(
            input=input,
            config=config,
            context=context,
        )

    def history(
        self,
        thread_id: str,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[Any]:
        """Get the history of states for a thread.

        Args:
            thread_id: The thread ID to get history for.
            filter: Optional filter criteria.
            before: Get history before this config.
            limit: Maximum number of states to return.

        Yields:
            Historical state snapshots.
        """
        self._ensure_compiled()

        for snapshot in self.compiled_ciri.get_state_history(
            limit=limit,
            filter=filter,
            before=before,
            config={"configurable": {"thread_id": thread_id}},
        ):
            # Only log if logger is enabled (respects logging suppression)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Fetched Snapshot: {snapshot}")
            yield snapshot
