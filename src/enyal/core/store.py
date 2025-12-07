"""Context store implementation using SQLite with sqlite-vec."""

import json
import logging
import os
import sqlite3
import struct
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import numpy as np
from numpy.typing import NDArray

from enyal.embeddings.engine import EmbeddingEngine
from enyal.models.context import (
    ContextEntry,
    ContextStats,
    ContextType,
    ScopeLevel,
    SourceType,
)

logger = logging.getLogger(__name__)

# SQL Schema
SCHEMA_SQL = """
-- Main context entries table
CREATE TABLE IF NOT EXISTS context_entries (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    content_type TEXT NOT NULL,
    scope_level TEXT NOT NULL,
    scope_path TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    accessed_at TEXT,
    access_count INTEGER NOT NULL DEFAULT 0,
    source_type TEXT,
    source_ref TEXT,
    tags TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    is_deprecated INTEGER NOT NULL DEFAULT 0
);

-- Vector embeddings (sqlite-vec virtual table)
CREATE VIRTUAL TABLE IF NOT EXISTS context_vectors USING vec0(
    entry_id TEXT PRIMARY KEY,
    embedding float[384]
);

-- Full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS context_fts USING fts5(
    content,
    content='context_entries',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS context_fts_insert AFTER INSERT ON context_entries BEGIN
    INSERT INTO context_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS context_fts_delete AFTER DELETE ON context_entries BEGIN
    INSERT INTO context_fts(context_fts, rowid, content)
    VALUES('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS context_fts_update AFTER UPDATE ON context_entries BEGIN
    INSERT INTO context_fts(context_fts, rowid, content)
    VALUES('delete', old.rowid, old.content);
    INSERT INTO context_fts(rowid, content) VALUES (new.rowid, new.content);
END;

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_context_scope ON context_entries(scope_level, scope_path);
CREATE INDEX IF NOT EXISTS idx_context_type ON context_entries(content_type);
CREATE INDEX IF NOT EXISTS idx_context_updated ON context_entries(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_context_confidence ON context_entries(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_context_active ON context_entries(is_deprecated) WHERE is_deprecated = 0;
"""


def serialize_embedding(embedding: NDArray[np.float32]) -> bytes:
    """Serialize a numpy embedding to bytes for sqlite-vec."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def deserialize_embedding(data: bytes) -> NDArray[np.float32]:
    """Deserialize bytes back to numpy embedding."""
    count = len(data) // 4  # 4 bytes per float32
    return np.array(struct.unpack(f"{count}f", data), dtype=np.float32)


class ContextStore:
    """
    Thread-safe context store with SQLite and sqlite-vec.

    Uses WAL mode for concurrent reads and application-level locking for writes.
    """

    def __init__(self, db_path: str | Path):
        """
        Initialize the context store.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path).expanduser()
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            # Load sqlite-vec extension
            conn.enable_load_extension(True)
            try:
                import sqlite_vec

                sqlite_vec.load(conn)
            except Exception as e:
                logger.warning(f"Could not load sqlite-vec extension: {e}")
                logger.warning("Vector search will not be available")

            # Create schema
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _read_transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for read transactions (no locking needed with WAL)."""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            pass  # No commit needed for reads

    @contextmanager
    def _write_transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for write transactions (serialized with lock)."""
        with self._write_lock:
            conn = self._get_connection()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def remember(
        self,
        content: str,
        content_type: ContextType | str = ContextType.FACT,
        scope_level: ScopeLevel | str = ScopeLevel.PROJECT,
        scope_path: str | None = None,
        source_type: SourceType | str | None = None,
        source_ref: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        confidence: float = 1.0,
    ) -> str:
        """
        Store new context in memory.

        Args:
            content: The context/knowledge to store.
            content_type: Type of context (fact, preference, decision, etc.).
            scope_level: Hierarchical scope (global, workspace, project, file).
            scope_path: Path for non-global scopes.
            source_type: Source of this context.
            source_ref: Reference to the source.
            tags: Tags for categorization.
            metadata: Additional metadata.
            confidence: Initial confidence score (0.0-1.0).

        Returns:
            The ID of the created entry.
        """
        entry = ContextEntry(
            content=content,
            content_type=ContextType(content_type) if isinstance(content_type, str) else content_type,
            scope_level=ScopeLevel(scope_level) if isinstance(scope_level, str) else scope_level,
            scope_path=scope_path,
            source_type=SourceType(source_type) if isinstance(source_type, str) and source_type else None,
            source_ref=source_ref,
            tags=tags or [],
            metadata=metadata or {},
            confidence=confidence,
        )

        # Generate embedding
        embedding = EmbeddingEngine.embed(content)

        with self._write_transaction() as conn:
            # Insert metadata
            conn.execute(
                """
                INSERT INTO context_entries (
                    id, content, content_type, scope_level, scope_path,
                    confidence, created_at, updated_at, accessed_at,
                    access_count, source_type, source_ref, tags, metadata, is_deprecated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.content,
                    entry.content_type.value,
                    entry.scope_level.value,
                    entry.scope_path,
                    entry.confidence,
                    entry.created_at.isoformat(),
                    entry.updated_at.isoformat(),
                    entry.accessed_at.isoformat() if entry.accessed_at else None,
                    entry.access_count,
                    entry.source_type.value if entry.source_type else None,
                    entry.source_ref,
                    json.dumps(entry.tags),
                    json.dumps(entry.metadata),
                    int(entry.is_deprecated),
                ),
            )

            # Insert vector embedding
            conn.execute(
                "INSERT INTO context_vectors (entry_id, embedding) VALUES (?, ?)",
                (entry.id, serialize_embedding(embedding)),
            )

        logger.info(f"Stored context entry: {entry.id}")
        return entry.id

    def recall(
        self,
        query: str,
        limit: int = 10,
        scope_level: ScopeLevel | str | None = None,
        scope_path: str | None = None,
        content_type: ContextType | str | None = None,
        min_confidence: float = 0.3,
        include_deprecated: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search for relevant context using semantic similarity.

        Args:
            query: Natural language search query.
            limit: Maximum number of results.
            scope_level: Filter by scope level.
            scope_path: Filter by scope path (prefix match).
            content_type: Filter by content type.
            min_confidence: Minimum confidence threshold.
            include_deprecated: Include deprecated entries.

        Returns:
            List of matching entries with relevance scores.
        """
        # Generate query embedding
        query_embedding = EmbeddingEngine.embed(query)

        with self._read_transaction() as conn:
            # Build the query with filters
            # First, get vector matches
            vector_results = conn.execute(
                """
                SELECT entry_id, distance
                FROM context_vectors
                WHERE embedding MATCH ?
                  AND k = ?
                """,
                (serialize_embedding(query_embedding), limit * 2),  # Overfetch for filtering
            ).fetchall()

            if not vector_results:
                return []

            # Get entry IDs and distances
            entry_ids = [r["entry_id"] for r in vector_results]
            distances = {r["entry_id"]: r["distance"] for r in vector_results}

            # Build filter conditions
            conditions = ["id IN ({})".format(",".join("?" * len(entry_ids)))]
            params: list[Any] = list(entry_ids)

            if not include_deprecated:
                conditions.append("is_deprecated = 0")

            if min_confidence > 0:
                conditions.append("confidence >= ?")
                params.append(min_confidence)

            if scope_level:
                sl = ScopeLevel(scope_level) if isinstance(scope_level, str) else scope_level
                conditions.append("scope_level = ?")
                params.append(sl.value)

            if scope_path:
                conditions.append("(scope_path = ? OR scope_path LIKE ?)")
                params.extend([scope_path, f"{scope_path}%"])

            if content_type:
                ct = ContextType(content_type) if isinstance(content_type, str) else content_type
                conditions.append("content_type = ?")
                params.append(ct.value)

            # Fetch matching entries
            query_sql = f"""
                SELECT * FROM context_entries
                WHERE {' AND '.join(conditions)}
            """
            rows = conn.execute(query_sql, params).fetchall()

            # Update access timestamps
            now = datetime.utcnow().isoformat()
            accessed_ids = [dict(r)["id"] for r in rows]
            if accessed_ids:
                conn.execute(
                    f"""
                    UPDATE context_entries
                    SET accessed_at = ?, access_count = access_count + 1
                    WHERE id IN ({','.join('?' * len(accessed_ids))})
                    """,
                    [now, *accessed_ids],
                )
                conn.commit()

            # Build results with scores
            results = []
            for row in rows:
                row_dict = dict(row)
                entry_id = row_dict["id"]
                distance = distances.get(entry_id, float("inf"))

                # Convert distance to score (lower distance = higher score)
                # Using 1 / (1 + distance) to normalize to 0-1 range
                score = 1.0 / (1.0 + distance)

                results.append({
                    "entry": self._row_to_entry(row_dict),
                    "distance": distance,
                    "score": score,
                })

            # Sort by score and limit
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]

    def forget(self, entry_id: str, hard_delete: bool = False) -> bool:
        """
        Remove or deprecate a context entry.

        Args:
            entry_id: The ID of the entry to remove.
            hard_delete: If True, permanently delete. Otherwise soft-delete.

        Returns:
            True if entry was found and modified.
        """
        with self._write_transaction() as conn:
            if hard_delete:
                # Delete from vectors first (foreign key)
                conn.execute("DELETE FROM context_vectors WHERE entry_id = ?", (entry_id,))
                result = conn.execute("DELETE FROM context_entries WHERE id = ?", (entry_id,))
            else:
                result = conn.execute(
                    "UPDATE context_entries SET is_deprecated = 1, updated_at = ? WHERE id = ?",
                    (datetime.utcnow().isoformat(), entry_id),
                )

            return result.rowcount > 0

    def get(self, entry_id: str) -> ContextEntry | None:
        """
        Get a specific context entry by ID.

        Args:
            entry_id: The ID of the entry.

        Returns:
            The entry if found, None otherwise.
        """
        with self._read_transaction() as conn:
            row = conn.execute(
                "SELECT * FROM context_entries WHERE id = ?", (entry_id,)
            ).fetchone()

            if row:
                return self._row_to_entry(dict(row))
            return None

    def update(
        self,
        entry_id: str,
        content: str | None = None,
        confidence: float | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Update an existing context entry.

        Args:
            entry_id: The ID of the entry to update.
            content: New content (will regenerate embedding).
            confidence: New confidence score.
            tags: New tags (replaces existing).
            metadata: New metadata (replaces existing).

        Returns:
            True if entry was found and updated.
        """
        updates = ["updated_at = ?"]
        params: list[Any] = [datetime.utcnow().isoformat()]

        if content is not None:
            updates.append("content = ?")
            params.append(content)

        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)

        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))

        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        params.append(entry_id)

        with self._write_transaction() as conn:
            result = conn.execute(
                f"UPDATE context_entries SET {', '.join(updates)} WHERE id = ?",
                params,
            )

            # Update embedding if content changed
            if content is not None and result.rowcount > 0:
                embedding = EmbeddingEngine.embed(content)
                conn.execute(
                    "UPDATE context_vectors SET embedding = ? WHERE entry_id = ?",
                    (serialize_embedding(embedding), entry_id),
                )

            return result.rowcount > 0

    def stats(self) -> ContextStats:
        """
        Get statistics about the context store.

        Returns:
            Statistics including counts, sizes, and distributions.
        """
        with self._read_transaction() as conn:
            # Total and active counts
            total = conn.execute("SELECT COUNT(*) FROM context_entries").fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM context_entries WHERE is_deprecated = 0"
            ).fetchone()[0]
            deprecated = total - active

            # By type
            by_type = {}
            for row in conn.execute(
                "SELECT content_type, COUNT(*) as cnt FROM context_entries GROUP BY content_type"
            ):
                by_type[row["content_type"]] = row["cnt"]

            # By scope
            by_scope = {}
            for row in conn.execute(
                "SELECT scope_level, COUNT(*) as cnt FROM context_entries GROUP BY scope_level"
            ):
                by_scope[row["scope_level"]] = row["cnt"]

            # Average confidence
            avg_conf = conn.execute(
                "SELECT AVG(confidence) FROM context_entries WHERE is_deprecated = 0"
            ).fetchone()[0] or 0.0

            # Date range
            oldest = conn.execute(
                "SELECT MIN(created_at) FROM context_entries"
            ).fetchone()[0]
            newest = conn.execute(
                "SELECT MAX(created_at) FROM context_entries"
            ).fetchone()[0]

            # Storage size
            storage_size = os.path.getsize(self.db_path) if self.db_path.exists() else 0

            return ContextStats(
                total_entries=total,
                active_entries=active,
                deprecated_entries=deprecated,
                entries_by_type=by_type,
                entries_by_scope=by_scope,
                avg_confidence=avg_conf,
                storage_size_bytes=storage_size,
                oldest_entry=datetime.fromisoformat(oldest) if oldest else None,
                newest_entry=datetime.fromisoformat(newest) if newest else None,
            )

    def _row_to_entry(self, row: dict[str, Any]) -> ContextEntry:
        """Convert a database row to a ContextEntry."""
        return ContextEntry(
            id=row["id"],
            content=row["content"],
            content_type=ContextType(row["content_type"]),
            scope_level=ScopeLevel(row["scope_level"]),
            scope_path=row["scope_path"],
            confidence=row["confidence"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            accessed_at=datetime.fromisoformat(row["accessed_at"]) if row["accessed_at"] else None,
            access_count=row["access_count"],
            source_type=SourceType(row["source_type"]) if row["source_type"] else None,
            source_ref=row["source_ref"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            is_deprecated=bool(row["is_deprecated"]),
        )

    def close(self) -> None:
        """Close all database connections."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
