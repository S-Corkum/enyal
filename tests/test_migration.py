"""Tests for the embedding model migration system."""

import sqlite3
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from enyal.core.migration import EMBEDDING_VERSION, MigrationManager, MigrationStatus
from enyal.embeddings.models import MODEL_REGISTRY


@pytest.fixture
def temp_db_path() -> Path:
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_migration.db"


@pytest.fixture
def mock_engine_768() -> MagicMock:
    """Create a mock engine with 768-dim nomic config."""
    engine = MagicMock()
    engine.config = MODEL_REGISTRY["nomic-ai/nomic-embed-text-v1.5"]
    engine.embed_batch.return_value = np.random.rand(0, 768).astype(np.float32)
    return engine


@pytest.fixture
def mock_engine_384() -> MagicMock:
    """Create a mock engine with 384-dim MiniLM config."""
    engine = MagicMock()
    engine.config = MODEL_REGISTRY["all-MiniLM-L6-v2"]
    engine.embed_batch.return_value = np.random.rand(0, 384).astype(np.float32)
    return engine


def _create_legacy_db(db_path: Path, num_entries: int = 3) -> None:
    """Create a legacy database with 384-dim vectors and no schema_meta."""
    conn = sqlite3.connect(str(db_path))

    # Load sqlite-vec
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except Exception:
        pytest.skip("sqlite-vec not available")

    # Create tables
    conn.executescript("""
        CREATE TABLE context_entries (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            content_type TEXT NOT NULL DEFAULT 'fact',
            scope_level TEXT NOT NULL DEFAULT 'project',
            scope_path TEXT,
            confidence REAL NOT NULL DEFAULT 1.0,
            created_at TEXT NOT NULL DEFAULT '2024-01-01T00:00:00',
            updated_at TEXT NOT NULL DEFAULT '2024-01-01T00:00:00',
            accessed_at TEXT,
            access_count INTEGER NOT NULL DEFAULT 0,
            source_type TEXT,
            source_ref TEXT,
            tags TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}',
            is_deprecated INTEGER NOT NULL DEFAULT 0
        );
    """)

    conn.execute("""
        CREATE VIRTUAL TABLE context_vectors USING vec0(
            entry_id TEXT PRIMARY KEY,
            embedding float[384]
        )
    """)

    # Create edges table
    conn.executescript("""
        CREATE TABLE context_edges (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            created_at TEXT NOT NULL DEFAULT '2024-01-01T00:00:00',
            metadata TEXT DEFAULT '{}'
        );
    """)

    # Insert test entries with 384-dim embeddings
    for i in range(num_entries):
        entry_id = f"entry-{i}"
        content = f"Test content number {i}"
        conn.execute(
            "INSERT INTO context_entries (id, content) VALUES (?, ?)",
            (entry_id, content),
        )
        embedding = np.random.rand(384).astype(np.float32)
        serialized = struct.pack("384f", *embedding)
        conn.execute(
            "INSERT INTO context_vectors (entry_id, embedding) VALUES (?, ?)",
            (entry_id, serialized),
        )

    # Add an edge to verify preservation
    if num_entries >= 2:
        conn.execute(
            "INSERT INTO context_edges (id, source_id, target_id, edge_type) VALUES (?, ?, ?, ?)",
            ("edge-1", "entry-0", "entry-1", "relates_to"),
        )

    conn.commit()
    conn.close()


def _create_db_with_meta(
    db_path: Path,
    model_name: str,
    dimension: int,
    embedding_version: str | None = None,
) -> None:
    """Create a database with schema_meta tracking a specific model."""
    conn = sqlite3.connect(str(db_path))

    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except Exception:
        pytest.skip("sqlite-vec not available")

    conn.executescript("""
        CREATE TABLE context_entries (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            content_type TEXT NOT NULL DEFAULT 'fact',
            scope_level TEXT NOT NULL DEFAULT 'project',
            scope_path TEXT,
            confidence REAL NOT NULL DEFAULT 1.0,
            created_at TEXT NOT NULL DEFAULT '2024-01-01T00:00:00',
            updated_at TEXT NOT NULL DEFAULT '2024-01-01T00:00:00',
            accessed_at TEXT,
            access_count INTEGER NOT NULL DEFAULT 0,
            source_type TEXT,
            source_ref TEXT,
            tags TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}',
            is_deprecated INTEGER NOT NULL DEFAULT 0
        );
    """)

    conn.execute(f"""
        CREATE VIRTUAL TABLE context_vectors USING vec0(
            entry_id TEXT PRIMARY KEY,
            embedding float[{dimension}]
        )
    """)

    conn.executescript("""
        CREATE TABLE schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)

    conn.execute(
        "INSERT INTO schema_meta (key, value, updated_at) VALUES (?, ?, ?)",
        ("embedding_model", model_name, "2024-01-01T00:00:00"),
    )
    conn.execute(
        "INSERT INTO schema_meta (key, value, updated_at) VALUES (?, ?, ?)",
        ("embedding_dimension", str(dimension), "2024-01-01T00:00:00"),
    )
    if embedding_version is not None:
        conn.execute(
            "INSERT INTO schema_meta (key, value, updated_at) VALUES (?, ?, ?)",
            ("embedding_version", embedding_version, "2024-01-01T00:00:00"),
        )

    conn.commit()
    conn.close()


class TestMigrationStatus:
    """Tests for migration status detection."""

    def test_fresh_status_no_tables(self, temp_db_path: Path, mock_engine_768: MagicMock) -> None:
        """Test FRESH status when no tables exist."""
        conn = sqlite3.connect(str(temp_db_path))
        manager = MigrationManager(mock_engine_768)

        status = manager.check_status(conn)
        assert status == MigrationStatus.FRESH
        conn.close()

    def test_legacy_status_no_meta(self, temp_db_path: Path, mock_engine_768: MagicMock) -> None:
        """Test LEGACY status when vectors exist but no schema_meta."""
        _create_legacy_db(temp_db_path)
        conn = sqlite3.connect(str(temp_db_path))

        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            pytest.skip("sqlite-vec not available")

        manager = MigrationManager(mock_engine_768)
        status = manager.check_status(conn)
        assert status == MigrationStatus.LEGACY
        conn.close()

    def test_needs_migration_model_changed(self, temp_db_path: Path, mock_engine_768: MagicMock) -> None:
        """Test NEEDS_MIGRATION when model name differs."""
        _create_db_with_meta(temp_db_path, "all-MiniLM-L6-v2", 384)
        conn = sqlite3.connect(str(temp_db_path))

        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            pytest.skip("sqlite-vec not available")

        manager = MigrationManager(mock_engine_768)
        status = manager.check_status(conn)
        assert status == MigrationStatus.NEEDS_MIGRATION
        conn.close()

    def test_needs_migration_version_changed(self, temp_db_path: Path, mock_engine_768: MagicMock) -> None:
        """Test NEEDS_MIGRATION when embedding version differs."""
        _create_db_with_meta(
            temp_db_path, "nomic-ai/nomic-embed-text-v1.5", 768, embedding_version="0"
        )
        conn = sqlite3.connect(str(temp_db_path))

        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            pytest.skip("sqlite-vec not available")

        manager = MigrationManager(mock_engine_768)
        status = manager.check_status(conn)
        assert status == MigrationStatus.NEEDS_MIGRATION
        conn.close()

    def test_needs_migration_no_version(self, temp_db_path: Path, mock_engine_768: MagicMock) -> None:
        """Test NEEDS_MIGRATION when embedding_version key is missing."""
        _create_db_with_meta(temp_db_path, "nomic-ai/nomic-embed-text-v1.5", 768)
        conn = sqlite3.connect(str(temp_db_path))

        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            pytest.skip("sqlite-vec not available")

        manager = MigrationManager(mock_engine_768)
        status = manager.check_status(conn)
        assert status == MigrationStatus.NEEDS_MIGRATION
        conn.close()

    def test_current_status_matches(self, temp_db_path: Path, mock_engine_768: MagicMock) -> None:
        """Test CURRENT status when meta matches engine config."""
        _create_db_with_meta(
            temp_db_path, "nomic-ai/nomic-embed-text-v1.5", 768,
            embedding_version=EMBEDDING_VERSION,
        )
        conn = sqlite3.connect(str(temp_db_path))

        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            pytest.skip("sqlite-vec not available")

        manager = MigrationManager(mock_engine_768)
        status = manager.check_status(conn)
        assert status == MigrationStatus.CURRENT
        conn.close()


class TestMigrationExecution:
    """Tests for migration execution."""

    def test_migrate_legacy_to_new(self, temp_db_path: Path, mock_engine_768: MagicMock) -> None:
        """Test migrating a legacy 384-dim DB to 768-dim."""
        _create_legacy_db(temp_db_path, num_entries=3)

        conn = sqlite3.connect(str(temp_db_path))
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            pytest.skip("sqlite-vec not available")

        # Mock embed_batch to return 768-dim embeddings
        mock_engine_768.embed_batch.return_value = np.random.rand(3, 768).astype(np.float32)

        manager = MigrationManager(mock_engine_768)
        result = manager.migrate(conn)
        conn.commit()

        assert result.success is True
        assert result.entries_migrated == 3
        assert result.old_model == "all-MiniLM-L6-v2"
        assert result.old_dimension == 384
        assert result.new_model == "nomic-ai/nomic-embed-text-v1.5"
        assert result.new_dimension == 768
        assert result.error is None

        # Verify schema_meta was written
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedding_model'"
        ).fetchone()
        assert row[0] == "nomic-ai/nomic-embed-text-v1.5"

        dim_row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedding_dimension'"
        ).fetchone()
        assert dim_row[0] == "768"

        ver_row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedding_version'"
        ).fetchone()
        assert ver_row[0] == EMBEDDING_VERSION

        conn.close()

    def test_migrate_empty_db(self, temp_db_path: Path, mock_engine_768: MagicMock) -> None:
        """Test migrating a legacy DB with no entries."""
        conn = sqlite3.connect(str(temp_db_path))
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            pytest.skip("sqlite-vec not available")

        # Create minimal legacy structure
        conn.executescript("""
            CREATE TABLE context_entries (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE context_vectors USING vec0(
                entry_id TEXT PRIMARY KEY,
                embedding float[384]
            )
        """)
        conn.commit()

        manager = MigrationManager(mock_engine_768)
        result = manager.migrate(conn)
        conn.commit()

        assert result.success is True
        assert result.entries_migrated == 0
        conn.close()

    def test_migrate_preserves_content(self, temp_db_path: Path, mock_engine_768: MagicMock) -> None:
        """Test that migration preserves entry content."""
        _create_legacy_db(temp_db_path, num_entries=2)

        conn = sqlite3.connect(str(temp_db_path))
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            pytest.skip("sqlite-vec not available")

        mock_engine_768.embed_batch.return_value = np.random.rand(2, 768).astype(np.float32)

        manager = MigrationManager(mock_engine_768)
        result = manager.migrate(conn)
        conn.commit()

        assert result.success is True

        # Verify content is preserved
        rows = conn.execute("SELECT id, content FROM context_entries ORDER BY id").fetchall()
        assert len(rows) == 2
        assert rows[0][0] == "entry-0"
        assert rows[0][1] == "Test content number 0"
        assert rows[1][0] == "entry-1"
        assert rows[1][1] == "Test content number 1"

        conn.close()

    def test_migrate_preserves_edges(self, temp_db_path: Path, mock_engine_768: MagicMock) -> None:
        """Test that migration preserves knowledge graph edges."""
        _create_legacy_db(temp_db_path, num_entries=2)

        conn = sqlite3.connect(str(temp_db_path))
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            pytest.skip("sqlite-vec not available")

        mock_engine_768.embed_batch.return_value = np.random.rand(2, 768).astype(np.float32)

        manager = MigrationManager(mock_engine_768)
        result = manager.migrate(conn)
        conn.commit()

        assert result.success is True

        # Verify edges are preserved
        edges = conn.execute("SELECT * FROM context_edges").fetchall()
        assert len(edges) == 1
        assert edges[0][1] == "entry-0"  # source_id
        assert edges[0][2] == "entry-1"  # target_id

        conn.close()

    def test_migrate_model_change(self, temp_db_path: Path, mock_engine_768: MagicMock) -> None:
        """Test migrating when model changes (with schema_meta)."""
        _create_db_with_meta(temp_db_path, "all-MiniLM-L6-v2", 384)

        conn = sqlite3.connect(str(temp_db_path))
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            pytest.skip("sqlite-vec not available")

        # Add an entry
        conn.execute(
            "INSERT INTO context_entries (id, content) VALUES (?, ?)",
            ("test-1", "test content"),
        )
        embedding = np.random.rand(384).astype(np.float32)
        conn.execute(
            "INSERT INTO context_vectors (entry_id, embedding) VALUES (?, ?)",
            ("test-1", struct.pack("384f", *embedding)),
        )
        conn.commit()

        mock_engine_768.embed_batch.return_value = np.random.rand(1, 768).astype(np.float32)

        manager = MigrationManager(mock_engine_768)
        result = manager.migrate(conn)
        conn.commit()

        assert result.success is True
        assert result.entries_migrated == 1
        assert result.old_model == "all-MiniLM-L6-v2"
        assert result.new_model == "nomic-ai/nomic-embed-text-v1.5"

        conn.close()

    def test_migrate_failure_returns_error(self, temp_db_path: Path, mock_engine_768: MagicMock) -> None:
        """Test that migration failure returns error in result."""
        _create_legacy_db(temp_db_path, num_entries=2)

        conn = sqlite3.connect(str(temp_db_path))
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            pytest.skip("sqlite-vec not available")

        # Make embed_batch raise an error
        mock_engine_768.embed_batch.side_effect = RuntimeError("Model load failed")

        manager = MigrationManager(mock_engine_768)
        result = manager.migrate(conn)

        assert result.success is False
        assert result.error == "Model load failed"
        assert result.entries_migrated == 0

        conn.close()


class TestFreshSchema:
    """Tests for fresh schema creation."""

    def test_ensure_fresh_schema_creates_vectors(self, temp_db_path: Path, mock_engine_768: MagicMock) -> None:
        """Test that ensure_fresh_schema creates the vectors table."""
        conn = sqlite3.connect(str(temp_db_path))
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            pytest.skip("sqlite-vec not available")

        manager = MigrationManager(mock_engine_768)
        manager.ensure_fresh_schema(conn)
        conn.commit()

        # Verify table exists
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='context_vectors'"
        ).fetchone()
        assert row is not None

        # Verify schema_meta was written
        model_row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedding_model'"
        ).fetchone()
        assert model_row[0] == "nomic-ai/nomic-embed-text-v1.5"

        dim_row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedding_dimension'"
        ).fetchone()
        assert dim_row[0] == "768"

        conn.close()

    def test_ensure_fresh_schema_384(self, temp_db_path: Path, mock_engine_384: MagicMock) -> None:
        """Test fresh schema with 384-dim config."""
        conn = sqlite3.connect(str(temp_db_path))
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            pytest.skip("sqlite-vec not available")

        manager = MigrationManager(mock_engine_384)
        manager.ensure_fresh_schema(conn)
        conn.commit()

        dim_row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedding_dimension'"
        ).fetchone()
        assert dim_row[0] == "384"

        conn.close()


class TestMigrationProgress:
    """Tests for migration progress logging."""

    def test_large_batch_logs_progress(self, temp_db_path: Path, mock_engine_768: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
        """Test that large migrations log progress."""
        # Create a DB with enough entries to trigger progress logging
        conn = sqlite3.connect(str(temp_db_path))
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            pytest.skip("sqlite-vec not available")

        conn.executescript("""
            CREATE TABLE context_entries (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE context_vectors USING vec0(
                entry_id TEXT PRIMARY KEY,
                embedding float[384]
            )
        """)

        # Insert 100 entries
        for i in range(100):
            conn.execute(
                "INSERT INTO context_entries (id, content) VALUES (?, ?)",
                (f"entry-{i}", f"content {i}"),
            )
            embedding = np.random.rand(384).astype(np.float32)
            conn.execute(
                "INSERT INTO context_vectors (entry_id, embedding) VALUES (?, ?)",
                (f"entry-{i}", struct.pack("384f", *embedding)),
            )
        conn.commit()

        # Mock embed_batch to return correct shapes for batches
        def mock_batch(texts, **_kwargs):
            return np.random.rand(len(texts), 768).astype(np.float32)

        mock_engine_768.embed_batch.side_effect = mock_batch

        manager = MigrationManager(mock_engine_768)

        import logging
        with caplog.at_level(logging.INFO, logger="enyal.core.migration"):
            result = manager.migrate(conn)
            conn.commit()

        assert result.success is True
        assert result.entries_migrated == 100

        # Check progress logging occurred
        progress_logs = [r for r in caplog.records if "progress" in r.message.lower()]
        assert len(progress_logs) > 0

        conn.close()
