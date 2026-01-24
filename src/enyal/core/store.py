"""Context store implementation using SQLite with sqlite-vec."""

import json
import logging
import os
import sqlite3
import struct
import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from enyal.embeddings.engine import EmbeddingEngine
from enyal.embeddings.models import ModelConfig
from enyal.models.context import (
    ContextEdge,
    ContextEntry,
    ContextStats,
    ContextType,
    EdgeType,
    ScopeLevel,
    SourceType,
)

logger = logging.getLogger(__name__)

# SQL Schema (everything except context_vectors, which is managed by migration)
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

-- Knowledge graph edges
CREATE TABLE IF NOT EXISTS context_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL CHECK (edge_type IN (
        'relates_to', 'supersedes', 'depends_on', 'conflicts_with'
    )),
    confidence REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    UNIQUE(source_id, target_id, edge_type),
    FOREIGN KEY (source_id) REFERENCES context_entries(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

-- Edge indexes
CREATE INDEX IF NOT EXISTS idx_edges_source ON context_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON context_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON context_edges(edge_type);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_context_scope ON context_entries(scope_level, scope_path);
CREATE INDEX IF NOT EXISTS idx_context_type ON context_entries(content_type);
CREATE INDEX IF NOT EXISTS idx_context_updated ON context_entries(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_context_confidence ON context_entries(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_context_active ON context_entries(is_deprecated) WHERE is_deprecated = 0;

-- Entry version history
CREATE TABLE IF NOT EXISTS context_versions (
    id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    changed_at TEXT NOT NULL,
    change_type TEXT NOT NULL CHECK (change_type IN ('created', 'updated', 'restored')),
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_versions_entry ON context_versions(entry_id);
CREATE INDEX IF NOT EXISTS idx_versions_changed ON context_versions(changed_at DESC);

-- Usage analytics
CREATE TABLE IF NOT EXISTS context_analytics (
    id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('recall', 'update', 'link', 'impact')),
    event_at TEXT NOT NULL,
    query TEXT,
    result_rank INTEGER,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_analytics_entry ON context_analytics(entry_id);
CREATE INDEX IF NOT EXISTS idx_analytics_event ON context_analytics(event_type);
CREATE INDEX IF NOT EXISTS idx_analytics_time ON context_analytics(event_at DESC);
"""


def serialize_embedding(embedding: NDArray[np.float32]) -> bytes:
    """Serialize a numpy embedding to bytes for sqlite-vec."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def deserialize_embedding(data: bytes) -> NDArray[np.float32]:
    """Deserialize bytes back to numpy embedding."""
    count = len(data) // 4  # 4 bytes per float32
    return np.array(struct.unpack(f"{count}f", data), dtype=np.float32)


def _escape_fts_query(query: str) -> str:
    """Escape FTS5 special characters for literal matching.

    Wraps query in quotes for phrase matching, escaping any internal quotes.
    """
    # Remove characters that break FTS5 query syntax
    cleaned = query.replace('"', " ").replace("*", " ").replace(":", " ")
    cleaned = " ".join(cleaned.split())  # Normalize whitespace
    if not cleaned:
        return '""'
    return f'"{cleaned}"'


class ContextStore:
    """
    Thread-safe context store with SQLite and sqlite-vec.

    Uses WAL mode for concurrent reads and application-level locking for writes.
    """

    def __init__(self, db_path: str | Path, engine: EmbeddingEngine | None = None):
        """
        Initialize the context store.

        Args:
            db_path: Path to the SQLite database file.
            engine: EmbeddingEngine instance. If None, creates one from env config.
        """
        self.db_path = Path(db_path).expanduser()
        self._engine = engine or EmbeddingEngine(ModelConfig.from_env())
        self._write_lock = threading.RLock()  # Reentrant lock for nested transactions
        self._local = threading.local()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema and handle migrations."""
        from enyal.core.migration import MigrationManager, MigrationStatus

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            # Load sqlite-vec extension
            conn.enable_load_extension(True)
            try:
                import sqlite_vec  # type: ignore[import-untyped]

                sqlite_vec.load(conn)
            except Exception as e:
                logger.warning(f"Could not load sqlite-vec extension: {e}")
                logger.warning("Vector search will not be available")

            # Create base schema (everything except context_vectors)
            conn.executescript(SCHEMA_SQL)
            conn.commit()

            # Handle vectors table via migration manager
            manager = MigrationManager(self._engine)
            status = manager.check_status(conn)

            if status == MigrationStatus.FRESH:
                manager.ensure_fresh_schema(conn)
                conn.commit()
            elif status in (MigrationStatus.LEGACY, MigrationStatus.NEEDS_MIGRATION):
                result = manager.migrate(conn)
                if not result.success:
                    raise RuntimeError(
                        f"Embedding migration failed: {result.error}. "
                        f"Database may need manual intervention."
                    )
                conn.commit()
            # MigrationStatus.CURRENT → no-op

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            new_conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            new_conn.row_factory = sqlite3.Row
            new_conn.execute("PRAGMA journal_mode=WAL")
            new_conn.execute("PRAGMA busy_timeout=5000")
            new_conn.execute("PRAGMA synchronous=NORMAL")
            new_conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            new_conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = new_conn
        result: sqlite3.Connection = self._local.conn
        return result

    @contextmanager
    def _read_transaction(self) -> Generator[sqlite3.Connection]:
        """Context manager for read transactions (no locking needed with WAL)."""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            pass  # No commit needed for reads

    @contextmanager
    def _write_transaction(self) -> Generator[sqlite3.Connection]:
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
        check_duplicate: bool = False,
        duplicate_threshold: float = 0.85,
        on_duplicate: str = "reject",
        # Graph relationship parameters
        auto_link: bool = False,
        auto_link_threshold: float = 0.85,
        relates_to: list[str] | None = None,
        supersedes: str | None = None,
        depends_on: list[str] | None = None,
        # Conflict detection
        detect_conflicts: bool = False,
        conflict_threshold: float = 0.9,
        # Supersedes suggestion
        suggest_supersedes: bool = False,
        supersedes_threshold: float = 0.95,
        auto_supersede: bool = False,
    ) -> str | dict[str, Any]:
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
            check_duplicate: Check for similar existing entries before storing.
            duplicate_threshold: Similarity threshold for duplicate detection (0-1).
            on_duplicate: Action when duplicate found:
                - "reject": Return existing entry ID without storing
                - "merge": Merge into existing entry (combine tags, update confidence)
                - "store": Store anyway as new entry
            auto_link: Automatically create RELATES_TO edges for similar entries.
            auto_link_threshold: Similarity threshold for auto-linking (0-1).
            relates_to: Entry IDs to create RELATES_TO edges with.
            supersedes: Entry ID that this entry supersedes.
            depends_on: Entry IDs that this entry depends on.
            detect_conflicts: Detect and flag potential contradictions with existing entries.
            conflict_threshold: Similarity threshold for conflict detection (0-1).
            suggest_supersedes: Suggest entries that this new entry might supersede.
            supersedes_threshold: Similarity threshold for supersedes suggestion (0-1).
            auto_supersede: Automatically create SUPERSEDES edges for very similar entries.

        Returns:
            Entry ID (str) if no detection features enabled.
            Dict with entry_id, action, and detection info if check_duplicate, detect_conflicts,
            or suggest_supersedes is True.
        """
        entry = ContextEntry(
            content=content,
            content_type=ContextType(content_type)
            if isinstance(content_type, str)
            else content_type,
            scope_level=ScopeLevel(scope_level) if isinstance(scope_level, str) else scope_level,
            scope_path=scope_path,
            source_type=SourceType(source_type)
            if isinstance(source_type, str) and source_type
            else None,
            source_ref=source_ref,
            tags=tags or [],
            metadata=metadata or {},
            confidence=confidence,
        )

        # Check for duplicates if requested
        if check_duplicate:
            similar = self.find_similar(
                content=content,
                threshold=duplicate_threshold,
                limit=1,
            )

            if similar:
                existing = similar[0]

                if on_duplicate == "reject":
                    return {
                        "entry_id": existing["entry_id"],
                        "action": "existing",
                        "duplicate_of": existing["entry_id"],
                        "similarity": existing["similarity"],
                    }

                elif on_duplicate == "merge":
                    # Merge tags (union) and update confidence (max)
                    existing_entry = existing["entry"]
                    merged_tags = list(set((existing_entry.tags or []) + (tags or [])))
                    merged_confidence = max(existing_entry.confidence, confidence)

                    self.update(
                        entry_id=existing["entry_id"],
                        confidence=merged_confidence,
                        tags=merged_tags,
                    )

                    return {
                        "entry_id": existing["entry_id"],
                        "action": "merged",
                        "duplicate_of": existing["entry_id"],
                        "similarity": existing["similarity"],
                    }

                # on_duplicate == "store" falls through to normal insert

        # Generate embedding
        embedding = self._engine.embed(content)

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

            # Create explicit edges if provided (INSIDE transaction)
            if supersedes:
                self.link(entry.id, supersedes, EdgeType.SUPERSEDES)
            if depends_on:
                for dep_id in depends_on:
                    self.link(entry.id, dep_id, EdgeType.DEPENDS_ON)
            if relates_to:
                for rel_id in relates_to:
                    self.link(entry.id, rel_id, EdgeType.RELATES_TO)

        # Auto-generate RELATES_TO edges based on similarity (OUTSIDE write transaction)
        if auto_link:
            similar = self.find_similar(
                content=content,
                threshold=auto_link_threshold,
                limit=5,
                exclude_deprecated=True,
            )
            for match in similar:
                if match["entry_id"] != entry.id:
                    self.link(
                        entry.id,
                        match["entry_id"],
                        EdgeType.RELATES_TO,
                        confidence=match["similarity"],
                        metadata={"auto_generated": True},
                    )

        # Detect potential conflicts if requested
        potential_conflicts: list[dict[str, Any]] = []
        if detect_conflicts:
            similar = self.find_similar(
                content=content,
                threshold=conflict_threshold,
                limit=5,
                exclude_deprecated=True,
            )
            for match in similar:
                existing = match["entry"]
                # Heuristic: Same scope + type + high similarity + contradiction = conflict
                if (
                    existing.scope_level == entry.scope_level
                    and existing.content_type == entry.content_type
                    and existing.id != entry.id
                    and self._appears_contradictory(content, existing.content)
                ):
                    self.link(
                        entry.id,
                        existing.id,
                        EdgeType.CONFLICTS_WITH,
                        metadata={"detected_at": "store", "similarity": match["similarity"]},
                    )
                    potential_conflicts.append(
                        {
                            "entry_id": existing.id,
                            "content": existing.content[:100],
                            "similarity": match["similarity"],
                        }
                    )

        # Suggest or auto-create supersedes relationships
        supersedes_candidates: list[dict[str, Any]] = []
        if suggest_supersedes or auto_supersede:
            similar = self.find_similar(
                content=content,
                threshold=supersedes_threshold,
                limit=3,
                exclude_deprecated=True,
            )
            for match in similar:
                existing = match["entry"]
                if (
                    existing.id != entry.id
                    and existing.scope_level == entry.scope_level
                    and existing.content_type == entry.content_type
                ):
                    if auto_supersede:
                        self.link(
                            entry.id,
                            existing.id,
                            EdgeType.SUPERSEDES,
                            metadata={"auto_detected": True, "similarity": match["similarity"]},
                        )
                    supersedes_candidates.append(
                        {
                            "entry_id": existing.id,
                            "content": existing.content[:100],
                            "similarity": match["similarity"],
                            "auto_superseded": auto_supersede,
                        }
                    )

        logger.info(f"Stored context entry: {entry.id}")

        # Create initial version record
        self._create_version(entry, "created", version=1)

        # Build return value based on what was requested
        # Maintain backward compatibility: only return dict if explicitly requested
        if check_duplicate or detect_conflicts or suggest_supersedes:
            return {
                "entry_id": entry.id,
                "action": "created",
                "duplicate_of": None,
                "similarity": None,
                "potential_conflicts": potential_conflicts if detect_conflicts else [],
                "supersedes_candidates": supersedes_candidates if suggest_supersedes else [],
            }

        # Default: return just the entry ID (backward compatible)
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
        query_embedding = self._engine.embed_query(query)

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
                WHERE {" AND ".join(conditions)}
            """
            rows = conn.execute(query_sql, params).fetchall()

            # Update access timestamps
            now = datetime.now(UTC).replace(tzinfo=None).isoformat()
            accessed_ids = [dict(r)["id"] for r in rows]
            if accessed_ids:
                conn.execute(
                    f"""
                    UPDATE context_entries
                    SET accessed_at = ?, access_count = access_count + 1
                    WHERE id IN ({",".join("?" * len(accessed_ids))})
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

                results.append(
                    {
                        "entry": self._row_to_entry(row_dict),
                        "distance": distance,
                        "score": score,
                    }
                )

            # Sort by score and limit
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]

    def fts_search(
        self,
        query: str,
        limit: int = 20,
        include_deprecated: bool = False,
    ) -> list[dict[str, Any]]:
        """Search using FTS5 full-text search.

        Args:
            query: Search query text.
            limit: Maximum results to return.
            include_deprecated: Include deprecated entries.

        Returns:
            List of dicts with entry_id and bm25_score.
            BM25 scores are negative; more negative = better match.
        """
        escaped_query = _escape_fts_query(query)

        with self._read_transaction() as conn:
            try:
                sql = """
                    SELECT ce.id as entry_id, bm25(context_fts) as bm25_score
                    FROM context_fts
                    JOIN context_entries ce ON ce.rowid = context_fts.rowid
                    WHERE context_fts MATCH ?
                      AND (? = 1 OR ce.is_deprecated = 0)
                    ORDER BY bm25(context_fts)
                    LIMIT ?
                """
                results = conn.execute(
                    sql, (escaped_query, int(include_deprecated), limit)
                ).fetchall()
                return [{"entry_id": r["entry_id"], "bm25_score": r["bm25_score"]} for r in results]
            except sqlite3.OperationalError:
                # FTS query failed (e.g., empty query) - return empty
                logger.debug(f"FTS search failed for query: {query}")
                return []

    def find_similar(
        self,
        content: str,
        threshold: float = 0.85,
        limit: int = 5,
        exclude_deprecated: bool = True,
    ) -> list[dict[str, Any]]:
        """Find entries similar to the given content.

        Args:
            content: Text to compare against existing entries.
            threshold: Minimum similarity score (0-1) to return.
            limit: Maximum results to return.
            exclude_deprecated: Exclude deprecated entries.

        Returns:
            List of dicts with entry_id, similarity, and entry.
        """
        embedding = self._engine.embed(content)

        with self._read_transaction() as conn:
            results = conn.execute(
                """
                SELECT entry_id, distance
                FROM context_vectors
                WHERE embedding MATCH ? AND k = ?
                """,
                (serialize_embedding(embedding), limit + 5),  # Overfetch for filtering
            ).fetchall()

            similar = []
            for row in results:
                distance = row["distance"]
                # Convert distance to similarity score (0-1, higher is better)
                similarity = 1.0 / (1.0 + distance)

                if similarity >= threshold:
                    entry = self.get(row["entry_id"])
                    if entry and (not exclude_deprecated or not entry.is_deprecated):
                        similar.append(
                            {
                                "entry_id": row["entry_id"],
                                "similarity": similarity,
                                "entry": entry,
                            }
                        )
                        if len(similar) >= limit:
                            break

            return similar

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
                    (datetime.now(UTC).replace(tzinfo=None).isoformat(), entry_id),
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
            row = conn.execute("SELECT * FROM context_entries WHERE id = ?", (entry_id,)).fetchone()

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
        params: list[Any] = [datetime.now(UTC).replace(tzinfo=None).isoformat()]

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
                embedding = self._engine.embed(content)
                conn.execute(
                    "UPDATE context_vectors SET embedding = ? WHERE entry_id = ?",
                    (serialize_embedding(embedding), entry_id),
                )

            updated = result.rowcount > 0

        # Create version record if update succeeded
        if updated:
            entry = self.get(entry_id)
            if entry:
                self._create_version(entry, "updated")

        return updated

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
            avg_conf = (
                conn.execute(
                    "SELECT AVG(confidence) FROM context_entries WHERE is_deprecated = 0"
                ).fetchone()[0]
                or 0.0
            )

            # Date range
            oldest = conn.execute("SELECT MIN(created_at) FROM context_entries").fetchone()[0]
            newest = conn.execute("SELECT MAX(created_at) FROM context_entries").fetchone()[0]

            # Storage size
            storage_size = os.path.getsize(self.db_path) if self.db_path.exists() else 0

            # Edge statistics
            total_edges = conn.execute("SELECT COUNT(*) FROM context_edges").fetchone()[0]

            edges_by_type: dict[str, int] = {}
            for row in conn.execute(
                "SELECT edge_type, COUNT(*) as cnt FROM context_edges GROUP BY edge_type"
            ):
                edges_by_type[row["edge_type"]] = row["cnt"]

            connected_entries = conn.execute(
                """
                SELECT COUNT(DISTINCT id) FROM context_entries
                WHERE id IN (
                    SELECT source_id FROM context_edges
                    UNION
                    SELECT target_id FROM context_edges
                )
                """
            ).fetchone()[0]

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
                # Graph statistics
                total_edges=total_edges,
                edges_by_type=edges_by_type,
                connected_entries=connected_entries,
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

    def _row_to_edge(self, row: dict[str, Any]) -> ContextEdge:
        """Convert a database row to a ContextEdge."""
        return ContextEdge(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            edge_type=EdgeType(row["edge_type"]),
            confidence=row["confidence"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def link(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType | str,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Create a relationship between two context entries.

        Args:
            source_id: ID of the source entry.
            target_id: ID of the target entry.
            edge_type: Type of relationship.
            confidence: Confidence score (0-1).
            metadata: Additional metadata.

        Returns:
            Edge ID if created, None if duplicate or entries don't exist.
        """
        # Convert string to EdgeType if needed
        if isinstance(edge_type, str):
            edge_type = EdgeType(edge_type)

        edge = ContextEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            confidence=confidence,
            metadata=metadata or {},
        )

        with self._write_transaction() as conn:
            # Verify both entries exist
            source_exists = conn.execute(
                "SELECT 1 FROM context_entries WHERE id = ?", (source_id,)
            ).fetchone()
            target_exists = conn.execute(
                "SELECT 1 FROM context_entries WHERE id = ?", (target_id,)
            ).fetchone()

            if not source_exists or not target_exists:
                logger.warning(
                    f"Cannot create edge: entry not found (source={source_id}, target={target_id})"
                )
                return None

            try:
                conn.execute(
                    """
                    INSERT INTO context_edges (
                        id, source_id, target_id, edge_type, confidence, created_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge.id,
                        edge.source_id,
                        edge.target_id,
                        edge.edge_type.value,
                        edge.confidence,
                        edge.created_at.isoformat(),
                        json.dumps(edge.metadata),
                    ),
                )
                logger.info(
                    f"Created edge: {edge.source_id} --{edge.edge_type.value}--> {edge.target_id}"
                )
                return edge.id
            except sqlite3.IntegrityError:
                # Duplicate edge
                logger.debug(f"Edge already exists: {source_id} -> {target_id} ({edge_type})")
                return None

    def unlink(self, edge_id: str) -> bool:
        """
        Remove a relationship by edge ID.

        Args:
            edge_id: The ID of the edge to remove.

        Returns:
            True if edge was found and removed.
        """
        with self._write_transaction() as conn:
            result = conn.execute("DELETE FROM context_edges WHERE id = ?", (edge_id,))
            if result.rowcount > 0:
                logger.info(f"Removed edge: {edge_id}")
                return True
            return False

    def get_edge(self, edge_id: str) -> ContextEdge | None:
        """
        Get a specific edge by ID.

        Args:
            edge_id: The ID of the edge.

        Returns:
            The edge if found, None otherwise.
        """
        with self._read_transaction() as conn:
            row = conn.execute("SELECT * FROM context_edges WHERE id = ?", (edge_id,)).fetchone()
            if row:
                return self._row_to_edge(dict(row))
            return None

    def get_edges(
        self,
        entry_id: str,
        direction: str = "both",
        edge_type: EdgeType | str | None = None,
    ) -> list[ContextEdge]:
        """
        Get edges connected to an entry.

        Args:
            entry_id: The entry to get edges for.
            direction: "outgoing", "incoming", or "both".
            edge_type: Optional filter by edge type.

        Returns:
            List of edges connected to the entry.

        Raises:
            ValueError: If direction is invalid.
        """
        # Validate direction
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'outgoing', 'incoming', or 'both'"
            )

        with self._read_transaction() as conn:
            conditions = []
            params: list[Any] = []

            if direction == "outgoing":
                conditions.append("source_id = ?")
                params.append(entry_id)
            elif direction == "incoming":
                conditions.append("target_id = ?")
                params.append(entry_id)
            else:  # both
                conditions.append("(source_id = ? OR target_id = ?)")
                params.extend([entry_id, entry_id])

            if edge_type:
                et = EdgeType(edge_type) if isinstance(edge_type, str) else edge_type
                conditions.append("edge_type = ?")
                params.append(et.value)

            query = f"""
                SELECT * FROM context_edges
                WHERE {" AND ".join(conditions)}
                ORDER BY created_at DESC
            """
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_edge(dict(row)) for row in rows]

    def traverse(
        self,
        start_id: str,
        edge_types: list[EdgeType | str] | None = None,
        direction: str = "outgoing",
        max_depth: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Traverse the graph from a starting node.

        Args:
            start_id: Entry ID to start traversal from.
            edge_types: Filter by edge types (None = all).
            direction: "outgoing" or "incoming".
            max_depth: Maximum traversal depth (1-4).

        Returns:
            List of dicts with entry, depth, path, edge_type, and confidence.

        Raises:
            ValueError: If direction is invalid.
        """
        # Validate direction
        if direction not in ("outgoing", "incoming"):
            raise ValueError(
                f"Invalid direction for traverse: {direction}. Must be 'outgoing' or 'incoming'"
            )

        max_depth = min(max(1, max_depth), 4)  # Clamp to 1-4

        # Build edge type filter
        type_filter = ""
        type_params: list[str] = []
        if edge_types:
            type_params = [
                EdgeType(et).value if isinstance(et, str) else et.value for et in edge_types
            ]
            type_placeholders = ",".join("?" * len(type_params))
            type_filter = f"AND edge_type IN ({type_placeholders})"

        # Direction determines which column to follow
        if direction == "outgoing":
            start_col, next_col = "source_id", "target_id"
        else:
            start_col, next_col = "target_id", "source_id"

        with self._read_transaction() as conn:
            # Use window function (ROW_NUMBER) to get deterministic results
            # This ensures we get the edge_type/confidence from the shortest path
            query = f"""
                WITH RECURSIVE traverse_chain AS (
                    -- Base case: direct connections
                    SELECT
                        {next_col} as entry_id,
                        1 as depth,
                        {next_col} as path,
                        edge_type,
                        confidence
                    FROM context_edges
                    WHERE {start_col} = ? {type_filter}

                    UNION ALL

                    -- Recursive case
                    SELECT
                        e.{next_col},
                        tc.depth + 1,
                        tc.path || ',' || e.{next_col},
                        e.edge_type,
                        e.confidence
                    FROM context_edges e
                    JOIN traverse_chain tc ON e.{start_col} = tc.entry_id
                    WHERE tc.depth < ?
                        AND tc.path NOT LIKE '%' || e.{next_col} || '%'
                        {type_filter.replace("edge_type", "e.edge_type") if type_filter else ""}
                ),
                ranked AS (
                    -- Use window function to pick the shortest path for each entry
                    SELECT
                        entry_id,
                        depth,
                        path,
                        edge_type,
                        confidence,
                        ROW_NUMBER() OVER (PARTITION BY entry_id ORDER BY depth) as rn
                    FROM traverse_chain
                )
                SELECT entry_id, depth as min_depth, path, edge_type, confidence
                FROM ranked
                WHERE rn = 1
                ORDER BY min_depth, entry_id
            """

            # Build params: start_id, [types], max_depth, [types again for recursive]
            params: list[Any] = [start_id]
            params.extend(type_params)
            params.append(max_depth)
            params.extend(type_params)

            rows = conn.execute(query, params).fetchall()

            results = []
            for row in rows:
                entry = self.get(row["entry_id"])
                if entry:
                    results.append(
                        {
                            "entry": entry,
                            "depth": row["min_depth"],
                            "path": row["path"].split(","),
                            "edge_type": row["edge_type"],
                            "confidence": row["confidence"],
                        }
                    )
            return results

    def get_superseded_ids(self) -> set[str]:
        """
        Get IDs of all entries that have been superseded by another entry.

        An entry is superseded if there exists a SUPERSEDES edge pointing TO it.
        (source --SUPERSEDES--> target means target is superseded)

        Returns:
            Set of entry IDs that are superseded.
        """
        with self._read_transaction() as conn:
            rows = conn.execute(
                "SELECT DISTINCT target_id FROM context_edges WHERE edge_type = ?",
                (EdgeType.SUPERSEDES.value,),
            ).fetchall()
            return {row["target_id"] for row in rows}

    def get_conflicted_ids(self) -> set[str]:
        """
        Get IDs of all entries that have unresolved conflicts.

        Returns:
            Set of entry IDs involved in CONFLICTS_WITH relationships.
        """
        with self._read_transaction() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT entry_id FROM (
                    SELECT source_id as entry_id FROM context_edges WHERE edge_type = ?
                    UNION
                    SELECT target_id as entry_id FROM context_edges WHERE edge_type = ?
                )
                """,
                (EdgeType.CONFLICTS_WITH.value, EdgeType.CONFLICTS_WITH.value),
            ).fetchall()
            return {row["entry_id"] for row in rows}

    def is_superseded(self, entry_id: str) -> bool:
        """Check if an entry has been superseded."""
        with self._read_transaction() as conn:
            row = conn.execute(
                "SELECT 1 FROM context_edges WHERE target_id = ? AND edge_type = ? LIMIT 1",
                (entry_id, EdgeType.SUPERSEDES.value),
            ).fetchone()
            return row is not None

    def get_superseding_entry(self, entry_id: str) -> str | None:
        """Get the ID of the entry that supersedes this one, if any."""
        with self._read_transaction() as conn:
            row = conn.execute(
                "SELECT source_id FROM context_edges WHERE target_id = ? AND edge_type = ? LIMIT 1",
                (entry_id, EdgeType.SUPERSEDES.value),
            ).fetchone()
            return row["source_id"] if row else None

    def health_check(self) -> dict[str, Any]:
        """
        Get comprehensive graph health statistics.

        Returns:
            Dict with health metrics including stale, orphan, conflict, and supersedes info.
        """
        now = datetime.now(UTC).replace(tzinfo=None)
        stale_threshold = now - timedelta(days=180)  # 6 months
        low_confidence_threshold = 0.5

        with self._read_transaction() as conn:
            # Basic counts
            total_entries = conn.execute(
                "SELECT COUNT(*) FROM context_entries WHERE is_deprecated = 0"
            ).fetchone()[0]
            total_edges = conn.execute("SELECT COUNT(*) FROM context_edges").fetchone()[0]

            # Superseded entries
            superseded_count = conn.execute(
                """
                SELECT COUNT(DISTINCT target_id) FROM context_edges
                WHERE edge_type = ?
                """,
                (EdgeType.SUPERSEDES.value,),
            ).fetchone()[0]

            # Conflicted entries (entries involved in CONFLICTS_WITH edges)
            conflicted_count = conn.execute(
                """
                SELECT COUNT(DISTINCT entry_id) FROM (
                    SELECT source_id as entry_id FROM context_edges WHERE edge_type = ?
                    UNION
                    SELECT target_id as entry_id FROM context_edges WHERE edge_type = ?
                )
                """,
                (EdgeType.CONFLICTS_WITH.value, EdgeType.CONFLICTS_WITH.value),
            ).fetchone()[0]

            # Stale entries (not updated in 6 months)
            stale_count = conn.execute(
                """
                SELECT COUNT(*) FROM context_entries
                WHERE is_deprecated = 0 AND datetime(updated_at) < ?
                """,
                (stale_threshold.isoformat(),),
            ).fetchone()[0]

            # Orphan entries (no edges at all)
            orphan_count = conn.execute(
                """
                SELECT COUNT(*) FROM context_entries ce
                WHERE is_deprecated = 0
                AND NOT EXISTS (
                    SELECT 1 FROM context_edges e
                    WHERE e.source_id = ce.id OR e.target_id = ce.id
                )
                """
            ).fetchone()[0]

            # Low confidence entries
            low_confidence_count = conn.execute(
                """
                SELECT COUNT(*) FROM context_entries
                WHERE is_deprecated = 0 AND confidence < ?
                """,
                (low_confidence_threshold,),
            ).fetchone()[0]

            # Never accessed entries
            never_accessed_count = conn.execute(
                """
                SELECT COUNT(*) FROM context_entries
                WHERE is_deprecated = 0 AND access_count = 0
                """
            ).fetchone()[0]

        return {
            "total_entries": total_entries,
            "total_edges": total_edges,
            "superseded_entries": superseded_count,
            "unresolved_conflicts": conflicted_count
            // 2,  # Divide by 2 since each conflict involves 2 entries
            "stale_entries": stale_count,
            "orphan_entries": orphan_count,
            "low_confidence_entries": low_confidence_count,
            "never_accessed_entries": never_accessed_count,
            "health_score": self._calculate_health_score(
                total_entries, superseded_count, conflicted_count // 2, stale_count, orphan_count
            ),
        }

    def _calculate_health_score(
        self,
        total: int,
        superseded: int,
        conflicts: int,
        stale: int,
        orphans: int,
    ) -> float:
        """Calculate overall health score (0-1)."""
        if total == 0:
            return 1.0

        # Penalties
        superseded_penalty = (superseded / total) * 0.2
        conflict_penalty = (conflicts / total) * 0.3
        stale_penalty = (stale / total) * 0.3
        orphan_penalty = (orphans / total) * 0.2

        score = 1.0 - superseded_penalty - conflict_penalty - stale_penalty - orphan_penalty
        return max(0.0, min(1.0, score))

    def get_stale_entries(self, days_old: int = 180, limit: int = 20) -> list[ContextEntry]:
        """Get entries that haven't been updated recently."""
        threshold = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days_old)
        with self._read_transaction() as conn:
            rows = conn.execute(
                """
                SELECT id FROM context_entries
                WHERE is_deprecated = 0 AND datetime(updated_at) < ?
                ORDER BY updated_at ASC
                LIMIT ?
                """,
                (threshold.isoformat(), limit),
            ).fetchall()
            entries = [self.get(row["id"]) for row in rows]
            return [e for e in entries if e is not None]

    def get_orphan_entries(self, limit: int = 20) -> list[ContextEntry]:
        """Get entries with no graph connections."""
        with self._read_transaction() as conn:
            rows = conn.execute(
                """
                SELECT id FROM context_entries ce
                WHERE is_deprecated = 0
                AND NOT EXISTS (
                    SELECT 1 FROM context_edges e
                    WHERE e.source_id = ce.id OR e.target_id = ce.id
                )
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            entries = [self.get(row["id"]) for row in rows]
            return [e for e in entries if e is not None]

    def get_conflicted_entries(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get entries involved in unresolved conflicts with their conflict partners."""
        with self._read_transaction() as conn:
            rows = conn.execute(
                """
                SELECT e.source_id, e.target_id, e.confidence, e.metadata
                FROM context_edges e
                WHERE e.edge_type = ?
                LIMIT ?
                """,
                (EdgeType.CONFLICTS_WITH.value, limit),
            ).fetchall()

            results = []
            for row in rows:
                source = self.get(row["source_id"])
                target = self.get(row["target_id"])
                if source and target:
                    results.append(
                        {
                            "entry1": source,
                            "entry2": target,
                            "confidence": row["confidence"],
                        }
                    )
            return results

    def _create_version(
        self,
        entry: ContextEntry,
        change_type: str,
        version: int | None = None,
    ) -> str:
        """Create a version history record for an entry."""
        if version is None:
            # Get next version number
            with self._read_transaction() as conn:
                row = conn.execute(
                    "SELECT MAX(version) FROM context_versions WHERE entry_id = ?", (entry.id,)
                ).fetchone()
                version = (row[0] or 0) + 1

        version_id = str(uuid.uuid4())
        now = datetime.now(UTC).replace(tzinfo=None)

        with self._write_transaction() as conn:
            conn.execute(
                """
                INSERT INTO context_versions (
                    id, entry_id, version, content, content_type,
                    confidence, tags, changed_at, change_type, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    entry.id,
                    version,
                    entry.content,
                    entry.content_type.value,
                    entry.confidence,
                    json.dumps(entry.tags),
                    now.isoformat(),
                    change_type,
                    json.dumps(entry.metadata),
                ),
            )
        return version_id

    def get_history(
        self,
        entry_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get version history for an entry.

        Args:
            entry_id: The entry to get history for.
            limit: Maximum number of versions to return.

        Returns:
            List of version records, newest first.
        """
        with self._read_transaction() as conn:
            rows = conn.execute(
                """
                SELECT id, version, content, content_type, confidence,
                       tags, changed_at, change_type, metadata
                FROM context_versions
                WHERE entry_id = ?
                ORDER BY version DESC
                LIMIT ?
                """,
                (entry_id, limit),
            ).fetchall()

            return [
                {
                    "version_id": row["id"],
                    "version": row["version"],
                    "content": row["content"],
                    "content_type": row["content_type"],
                    "confidence": row["confidence"],
                    "tags": json.loads(row["tags"]),
                    "changed_at": row["changed_at"],
                    "change_type": row["change_type"],
                    "metadata": json.loads(row["metadata"]),
                }
                for row in rows
            ]

    def track_usage(
        self,
        entry_id: str,
        event_type: str,
        query: str | None = None,
        result_rank: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Track a usage event for analytics."""
        event_id = str(uuid.uuid4())
        now = datetime.now(UTC).replace(tzinfo=None)

        with self._write_transaction() as conn:
            conn.execute(
                """
                INSERT INTO context_analytics (
                    id, entry_id, event_type, event_at, query, result_rank, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    entry_id,
                    event_type,
                    now.isoformat(),
                    query,
                    result_rank,
                    json.dumps(metadata or {}),
                ),
            )

    def get_analytics(
        self,
        entry_id: str | None = None,
        event_type: str | None = None,
        days: int = 30,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Get usage analytics.

        Args:
            entry_id: Filter by specific entry (optional).
            event_type: Filter by event type (optional).
            days: How many days of history to include.
            limit: Maximum events to return.

        Returns:
            Dict with analytics data.
        """
        since = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)

        query = """
            SELECT entry_id, event_type, COUNT(*) as count
            FROM context_analytics
            WHERE datetime(event_at) >= ?
        """
        params: list[Any] = [since.isoformat()]

        if entry_id:
            query += " AND entry_id = ?"
            params.append(entry_id)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        query += " GROUP BY entry_id, event_type ORDER BY count DESC LIMIT ?"
        params.append(limit)

        with self._read_transaction() as conn:
            rows = conn.execute(query, params).fetchall()

            # Also get top accessed entries
            top_rows = conn.execute(
                """
                SELECT entry_id, COUNT(*) as recall_count
                FROM context_analytics
                WHERE event_type = 'recall' AND datetime(event_at) >= ?
                GROUP BY entry_id
                ORDER BY recall_count DESC
                LIMIT 10
                """,
                (since.isoformat(),),
            ).fetchall()

            return {
                "period_days": days,
                "events_by_type": [
                    {"entry_id": r["entry_id"], "event_type": r["event_type"], "count": r["count"]}
                    for r in rows
                ],
                "top_recalled": [
                    {"entry_id": r["entry_id"], "recall_count": r["recall_count"]} for r in top_rows
                ],
            }

    def _appears_contradictory(self, new_content: str, existing_content: str) -> bool:
        """
        Simple heuristic to detect potential contradictions.

        Returns True if content appears to contradict.
        """
        new_lower = new_content.lower()
        existing_lower = existing_content.lower()

        # Check for negation patterns
        negation_words = [
            "not",
            "never",
            "don't",
            "doesn't",
            "shouldn't",
            "won't",
            "cannot",
            "disable",
        ]
        new_has_negation = any(word in new_lower for word in negation_words)
        existing_has_negation = any(word in existing_lower for word in negation_words)

        # If one has negation and other doesn't, might be contradiction
        if new_has_negation != existing_has_negation:
            return True

        # Check for opposing values in common patterns
        # e.g., "use React" vs "use Vue"
        use_pattern_words = ["use ", "prefer ", "chose ", "selected "]
        for pattern in use_pattern_words:
            if pattern in new_lower and pattern in existing_lower:
                # Extract what follows the pattern
                new_choice = new_lower.split(pattern)[-1].split()[0] if pattern in new_lower else ""
                existing_choice = (
                    existing_lower.split(pattern)[-1].split()[0]
                    if pattern in existing_lower
                    else ""
                )
                if new_choice and existing_choice and new_choice != existing_choice:
                    return True

        return False

    def close(self) -> None:
        """Close all database connections."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
