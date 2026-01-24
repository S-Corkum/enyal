"""Database migration for embedding model changes."""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enyal.embeddings.engine import EmbeddingEngine

logger = logging.getLogger(__name__)

# Bump this when embedding strategy changes (e.g., adding normalization)
# without changing the model name or dimension.
EMBEDDING_VERSION = "1"


class MigrationStatus(Enum):
    """Status of the database schema relative to current model config."""

    FRESH = "fresh"  # No existing database / no vectors table
    LEGACY = "legacy"  # Has vectors table but no schema_meta (old install)
    NEEDS_MIGRATION = "needs_migration"  # schema_meta exists but model/dim changed
    CURRENT = "current"  # schema_meta matches current config


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    success: bool
    entries_migrated: int
    old_model: str
    old_dimension: int
    new_model: str
    new_dimension: int
    duration_seconds: float
    error: str | None = None


SCHEMA_META_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class MigrationManager:
    """Manages embedding model migrations for the context store.

    The migration protocol is safe: all new embeddings are computed in memory
    before any database modifications occur. Text content and edges are never
    touched during migration.
    """

    def __init__(self, engine: EmbeddingEngine) -> None:
        """Initialize with an embedding engine instance.

        Args:
            engine: The configured EmbeddingEngine to use for re-embedding.
        """
        self._engine = engine

    def check_status(self, conn: sqlite3.Connection) -> MigrationStatus:
        """Detect the current schema state.

        Args:
            conn: An active database connection.

        Returns:
            MigrationStatus indicating what action is needed.
        """
        # Check if schema_meta table exists
        meta_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_meta'"
        ).fetchone()

        if not meta_exists:
            # Check if context_vectors table exists (legacy install)
            vectors_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='context_vectors'"
            ).fetchone()
            if vectors_exists:
                return MigrationStatus.LEGACY
            return MigrationStatus.FRESH

        # schema_meta exists - check if model matches
        row = conn.execute("SELECT value FROM schema_meta WHERE key = 'embedding_model'").fetchone()

        if not row:
            return MigrationStatus.LEGACY

        stored_model = row[0]
        current_model = self._engine.config.name

        if stored_model != current_model:
            return MigrationStatus.NEEDS_MIGRATION

        # Also check dimension
        dim_row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedding_dimension'"
        ).fetchone()

        if dim_row:
            stored_dim = int(dim_row[0])
            if stored_dim != self._engine.config.dimension:
                return MigrationStatus.NEEDS_MIGRATION

        # Check embedding version (catches strategy changes like normalization)
        ver_row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedding_version'"
        ).fetchone()

        stored_version = ver_row[0] if ver_row else "0"
        if stored_version != EMBEDDING_VERSION:
            return MigrationStatus.NEEDS_MIGRATION

        return MigrationStatus.CURRENT

    def migrate(self, conn: sqlite3.Connection) -> MigrationResult:
        """Migrate embeddings to the current model configuration.

        Safe protocol:
        1. Read all entry_id + content pairs
        2. Compute all new embeddings in memory
        3. Drop old vectors table
        4. Create new vectors table with correct dimensions
        5. Insert all new embeddings
        6. Update schema_meta

        Args:
            conn: An active database connection (caller manages transaction).

        Returns:
            MigrationResult with details of the migration.
        """
        import struct

        import numpy as np

        start_time = time.time()
        new_model = self._engine.config.name
        new_dim = self._engine.config.dimension

        # Detect old model info
        old_model = "unknown"
        old_dim = 384  # default assumption for legacy

        meta_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_meta'"
        ).fetchone()

        if meta_exists:
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'embedding_model'"
            ).fetchone()
            if row:
                old_model = row[0]
            dim_row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'embedding_dimension'"
            ).fetchone()
            if dim_row:
                old_dim = int(dim_row[0])
        else:
            old_model = "all-MiniLM-L6-v2"

        logger.info(f"Starting migration: {old_model} ({old_dim}d) -> {new_model} ({new_dim}d)")

        try:
            # Step 1: Read all entries that have vectors
            entries = conn.execute(
                """
                SELECT cv.entry_id, ce.content
                FROM context_vectors cv
                JOIN context_entries ce ON ce.id = cv.entry_id
                """
            ).fetchall()

            total = len(entries)
            logger.info(f"Found {total} entries to re-embed")

            if total == 0:
                # No entries to migrate - just recreate table
                conn.execute("DROP TABLE IF EXISTS context_vectors")
                conn.execute(
                    f"""
                    CREATE VIRTUAL TABLE context_vectors USING vec0(
                        entry_id TEXT PRIMARY KEY,
                        embedding float[{new_dim}]
                    )
                    """
                )
                self._write_schema_meta(conn)
                duration = time.time() - start_time
                return MigrationResult(
                    success=True,
                    entries_migrated=0,
                    old_model=old_model,
                    old_dimension=old_dim,
                    new_model=new_model,
                    new_dimension=new_dim,
                    duration_seconds=duration,
                )

            # Step 2: Compute all new embeddings in memory
            entry_ids = [row[0] for row in entries]
            contents = [row[1] for row in entries]

            # Batch embed with progress logging
            all_embeddings: list[np.ndarray] = []
            batch_size = 32
            for i in range(0, total, batch_size):
                batch = contents[i : i + batch_size]
                batch_embeddings = self._engine.embed_batch(batch, task="document")
                all_embeddings.append(batch_embeddings)

                processed = min(i + batch_size, total)
                if processed % 100 == 0 or processed == total:
                    logger.info(f"Re-embedding progress: {processed}/{total}")

            embeddings = np.vstack(all_embeddings)

            # Step 3: Drop old vectors table
            conn.execute("DROP TABLE IF EXISTS context_vectors")

            # Step 4: Create new vectors table
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE context_vectors USING vec0(
                    entry_id TEXT PRIMARY KEY,
                    embedding float[{new_dim}]
                )
                """
            )

            # Step 5: Insert all new embeddings
            for idx, entry_id in enumerate(entry_ids):
                embedding = embeddings[idx]
                serialized = struct.pack(f"{new_dim}f", *embedding)
                conn.execute(
                    "INSERT INTO context_vectors (entry_id, embedding) VALUES (?, ?)",
                    (entry_id, serialized),
                )

            # Step 6: Update schema_meta
            self._write_schema_meta(conn)

            duration = time.time() - start_time
            logger.info(f"Migration complete: {total} entries in {duration:.1f}s")

            return MigrationResult(
                success=True,
                entries_migrated=total,
                old_model=old_model,
                old_dimension=old_dim,
                new_model=new_model,
                new_dimension=new_dim,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Migration failed: {e}")
            return MigrationResult(
                success=False,
                entries_migrated=0,
                old_model=old_model,
                old_dimension=old_dim,
                new_model=new_model,
                new_dimension=new_dim,
                duration_seconds=duration,
                error=str(e),
            )

    def ensure_fresh_schema(self, conn: sqlite3.Connection) -> None:
        """Create vectors table and schema_meta for a fresh install.

        Args:
            conn: An active database connection.
        """
        dim = self._engine.config.dimension
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS context_vectors USING vec0(
                entry_id TEXT PRIMARY KEY,
                embedding float[{dim}]
            )
            """
        )
        conn.executescript(SCHEMA_META_SQL)
        self._write_schema_meta(conn)

    def _write_schema_meta(self, conn: sqlite3.Connection) -> None:
        """Write current model configuration to schema_meta."""
        from datetime import UTC, datetime

        now = datetime.now(UTC).replace(tzinfo=None).isoformat()

        # Ensure schema_meta table exists
        conn.execute(
            SCHEMA_META_SQL.replace("CREATE TABLE IF NOT EXISTS", "CREATE TABLE IF NOT EXISTS")
        )

        conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value, updated_at) VALUES (?, ?, ?)",
            ("embedding_model", self._engine.config.name, now),
        )
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value, updated_at) VALUES (?, ?, ?)",
            ("embedding_dimension", str(self._engine.config.dimension), now),
        )
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value, updated_at) VALUES (?, ?, ?)",
            ("embedding_version", EMBEDDING_VERSION, now),
        )
