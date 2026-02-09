import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import keyring
from cryptography.fernet import Fernet

from .utils import get_app_data_dir

logger = logging.getLogger(__name__)

SERVICE_NAME = "ciri"
KEY_USERNAME = "db_key"


def _get_or_create_db_key() -> str:
    """Retrieve the database encryption key from keyring, or generate and store one."""
    key = keyring.get_password(SERVICE_NAME, KEY_USERNAME)
    if key is None:
        key = Fernet.generate_key().decode("utf-8")
        keyring.set_password(SERVICE_NAME, KEY_USERNAME, key)
        logger.info("Generated and stored new database encryption key")
    return key


def _open_encrypted_connection(db_path: str):
    """Open an encrypted SQLite connection via pysqlcipher3."""
    from pysqlcipher3 import dbapi2 as sqlcipher

    conn = sqlcipher.connect(db_path)
    key = _get_or_create_db_key()
    conn.execute(f"PRAGMA key = '{key}'")
    # Verify the key works by reading the schema
    conn.execute("SELECT count(*) FROM sqlite_master")
    return conn


class CiriDatabase:
    """Encrypted SQLite database for CIRI persistence and thread management."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = get_app_data_dir() / "ciri.db"
        self.db_path = str(db_path)
        self.connection = _open_encrypted_connection(self.db_path)
        self._setup_threads_table()

    def _setup_threads_table(self):
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
        self.connection.execute(
            "INSERT INTO threads (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (thread_id, title, now, now),
        )
        self.connection.commit()
        return {"id": thread_id, "title": title, "created_at": now, "updated_at": now}

    def list_threads(self) -> List[Dict[str, Any]]:
        cursor = self.connection.execute(
            "SELECT id, title, created_at, updated_at FROM threads ORDER BY updated_at DESC"
        )
        return [
            {"id": row[0], "title": row[1], "created_at": row[2], "updated_at": row[3]}
            for row in cursor.fetchall()
        ]

    def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.connection.execute(
            "SELECT id, title, created_at, updated_at FROM threads WHERE id = ?",
            (thread_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {"id": row[0], "title": row[1], "created_at": row[2], "updated_at": row[3]}

    def rename_thread(self, thread_id: str, title: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            "UPDATE threads SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, thread_id),
        )
        self.connection.commit()

    def touch_thread(self, thread_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            "UPDATE threads SET updated_at = ? WHERE id = ?",
            (now, thread_id),
        )
        self.connection.commit()

    def delete_thread(self, thread_id: str) -> None:
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
        self.connection.close()
