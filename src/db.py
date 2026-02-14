import uuid
import sqlite3
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from .utils import get_app_data_dir

logger = logging.getLogger(__name__)


class CopilotDatabase:
    """SQLite database for CIRI persistence and thread management."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = get_app_data_dir() / "ciri.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._setup_threads_table()

    def _setup_threads_table(self):
        with self._lock:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'New Thread',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            self.connection.commit()

    def create_thread(self, title: str = "New Thread") -> Dict[str, Any]:
        thread_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.connection.execute(
                "INSERT INTO threads (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (thread_id, title, now, now),
            )
            self.connection.commit()
        return {"id": thread_id, "title": title, "created_at": now, "updated_at": now}

    def list_threads(self) -> List[Dict[str, Any]]:
        with self._lock:
            cursor = self.connection.execute(
                "SELECT id, title, created_at, updated_at FROM threads ORDER BY updated_at DESC"
            )
            return [
                {
                    "id": row[0],
                    "title": row[1],
                    "created_at": row[2],
                    "updated_at": row[3],
                }
                for row in cursor.fetchall()
            ]

    def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cursor = self.connection.execute(
                "SELECT id, title, created_at, updated_at FROM threads WHERE id = ?",
                (thread_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3],
        }

    def rename_thread(self, thread_id: str, title: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.connection.execute(
                "UPDATE threads SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, thread_id),
            )
            self.connection.commit()

    def touch_thread(self, thread_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.connection.execute(
                "UPDATE threads SET updated_at = ? WHERE id = ?",
                (now, thread_id),
            )
            self.connection.commit()

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            self.connection.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            # Also clean up checkpoints for this thread
            for table in ("checkpoints", "checkpoint_writes", "checkpoint_blobs"):
                try:
                    self.connection.execute(
                        f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,)
                    )
                except Exception:
                    pass  # Table may not exist yet
            self.connection.commit()

    def close(self):
        with self._lock:
            self.connection.close()
