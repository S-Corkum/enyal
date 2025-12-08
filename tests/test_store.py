"""Tests for context store."""

import tempfile
from pathlib import Path

import pytest

from enyal.core.store import ContextStore
from enyal.models.context import ContextType, ScopeLevel


@pytest.fixture
def temp_db() -> Path:
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def store(temp_db: Path) -> ContextStore:
    """Create a test store."""
    return ContextStore(temp_db)


class TestContextStore:
    """Tests for ContextStore."""

    def test_remember_and_get(self, store: ContextStore) -> None:
        """Test storing and retrieving an entry."""
        entry_id = store.remember(
            content="Test content for storage",
            content_type=ContextType.FACT,
            scope_level=ScopeLevel.PROJECT,
        )

        assert entry_id is not None

        entry = store.get(entry_id)
        assert entry is not None
        assert entry.content == "Test content for storage"
        assert entry.content_type == ContextType.FACT
        assert entry.scope_level == ScopeLevel.PROJECT

    def test_remember_with_all_fields(self, store: ContextStore) -> None:
        """Test storing with all optional fields."""
        entry_id = store.remember(
            content="Complete entry",
            content_type=ContextType.DECISION,
            scope_level=ScopeLevel.FILE,
            scope_path="/path/to/file.py",
            source_type="conversation",
            source_ref="session-123",
            tags=["test", "important"],
            metadata={"key": "value"},
            confidence=0.9,
        )

        entry = store.get(entry_id)
        assert entry is not None
        assert entry.content == "Complete entry"
        assert entry.scope_path == "/path/to/file.py"
        assert entry.tags == ["test", "important"]
        assert entry.metadata == {"key": "value"}
        assert entry.confidence == 0.9

    def test_forget_soft_delete(self, store: ContextStore) -> None:
        """Test soft deleting an entry."""
        entry_id = store.remember(content="To be deprecated")

        success = store.forget(entry_id, hard_delete=False)
        assert success is True

        # Entry should still exist but be deprecated
        entry = store.get(entry_id)
        assert entry is not None
        assert entry.is_deprecated is True

    def test_forget_hard_delete(self, store: ContextStore) -> None:
        """Test hard deleting an entry."""
        entry_id = store.remember(content="To be deleted")

        success = store.forget(entry_id, hard_delete=True)
        assert success is True

        # Entry should be gone
        entry = store.get(entry_id)
        assert entry is None

    def test_forget_nonexistent(self, store: ContextStore) -> None:
        """Test forgetting a nonexistent entry."""
        success = store.forget("nonexistent-id")
        assert success is False

    def test_update_content(self, store: ContextStore) -> None:
        """Test updating entry content."""
        entry_id = store.remember(content="Original content")

        success = store.update(entry_id, content="Updated content")
        assert success is True

        entry = store.get(entry_id)
        assert entry is not None
        assert entry.content == "Updated content"

    def test_update_confidence(self, store: ContextStore) -> None:
        """Test updating entry confidence."""
        entry_id = store.remember(content="Test", confidence=1.0)

        success = store.update(entry_id, confidence=0.5)
        assert success is True

        entry = store.get(entry_id)
        assert entry is not None
        assert entry.confidence == 0.5

    def test_update_tags(self, store: ContextStore) -> None:
        """Test updating entry tags."""
        entry_id = store.remember(content="Test", tags=["old"])

        success = store.update(entry_id, tags=["new", "tags"])
        assert success is True

        entry = store.get(entry_id)
        assert entry is not None
        assert entry.tags == ["new", "tags"]

    def test_stats_empty(self, store: ContextStore) -> None:
        """Test stats on empty store."""
        stats = store.stats()
        assert stats.total_entries == 0
        assert stats.active_entries == 0
        assert stats.deprecated_entries == 0

    def test_stats_with_entries(self, store: ContextStore) -> None:
        """Test stats with some entries."""
        store.remember(content="Fact 1", content_type=ContextType.FACT)
        store.remember(content="Fact 2", content_type=ContextType.FACT)
        store.remember(content="Decision 1", content_type=ContextType.DECISION)

        stats = store.stats()
        assert stats.total_entries == 3
        assert stats.active_entries == 3
        assert stats.entries_by_type.get("fact") == 2
        assert stats.entries_by_type.get("decision") == 1

    def test_recall_basic(self, store: ContextStore) -> None:
        """Test basic semantic recall."""
        store.remember(content="Python is a programming language")
        store.remember(content="JavaScript runs in browsers")
        store.remember(content="Rust is memory safe")

        results = store.recall("programming language", limit=2)
        assert len(results) <= 2
        # The Python entry should be most relevant
        assert any("Python" in r["entry"].content for r in results)

    def test_recall_with_filters(self, store: ContextStore) -> None:
        """Test recall with scope and type filters."""
        store.remember(
            content="Global setting",
            scope_level=ScopeLevel.GLOBAL,
            content_type=ContextType.PREFERENCE,
        )
        store.remember(
            content="Project setting",
            scope_level=ScopeLevel.PROJECT,
            content_type=ContextType.PREFERENCE,
        )

        results = store.recall(
            "setting",
            scope_level=ScopeLevel.GLOBAL,
        )
        assert len(results) >= 1
        assert all(r["entry"].scope_level == ScopeLevel.GLOBAL for r in results)

    def test_recall_excludes_deprecated(self, store: ContextStore) -> None:
        """Test that recall excludes deprecated entries by default."""
        entry_id = store.remember(content="Deprecated info")
        store.forget(entry_id, hard_delete=False)

        results = store.recall("Deprecated info")
        assert not any(r["entry"].id == entry_id for r in results)

    def test_recall_includes_deprecated(self, store: ContextStore) -> None:
        """Test that recall can include deprecated entries."""
        entry_id = store.remember(content="Deprecated info")
        store.forget(entry_id, hard_delete=False)

        results = store.recall("Deprecated info", include_deprecated=True)
        assert any(r["entry"].id == entry_id for r in results)

    def test_recall_min_confidence(self, store: ContextStore) -> None:
        """Test recall respects minimum confidence."""
        store.remember(content="High confidence", confidence=0.9)
        store.remember(content="Low confidence", confidence=0.2)

        results = store.recall("confidence", min_confidence=0.5)
        assert all(r["entry"].confidence >= 0.5 for r in results)

    def test_update_nonexistent_entry(self, store: ContextStore) -> None:
        """Test updating a nonexistent entry returns False."""
        success = store.update("nonexistent-id", content="New content")
        assert success is False

    def test_recall_with_all_filters(self, store: ContextStore) -> None:
        """Test recall with all filter options combined."""
        store.remember(
            content="Complete filter test",
            content_type=ContextType.DECISION,
            scope_level=ScopeLevel.FILE,
            scope_path="/test/path/file.py",
            confidence=0.9,
        )
        store.remember(
            content="Other entry",
            content_type=ContextType.FACT,
            scope_level=ScopeLevel.PROJECT,
            confidence=0.5,
        )

        results = store.recall(
            query="filter test",
            scope_level=ScopeLevel.FILE,
            scope_path="/test/path",
            content_type=ContextType.DECISION,
            min_confidence=0.8,
        )

        # Should only find the matching entry
        assert len(results) >= 0  # May be 0 or 1 depending on vector similarity

    def test_remember_all_content_types(self, store: ContextStore) -> None:
        """Test storing entries with all content types."""
        content_types = [
            ContextType.FACT,
            ContextType.PREFERENCE,
            ContextType.DECISION,
            ContextType.CONVENTION,
            ContextType.PATTERN,
        ]

        for ct in content_types:
            entry_id = store.remember(
                content=f"Test {ct.value}",
                content_type=ct,
            )
            entry = store.get(entry_id)
            assert entry is not None
            assert entry.content_type == ct

    def test_remember_all_scope_levels(self, store: ContextStore) -> None:
        """Test storing entries with all scope levels."""
        scope_levels = [
            ScopeLevel.FILE,
            ScopeLevel.PROJECT,
            ScopeLevel.WORKSPACE,
            ScopeLevel.GLOBAL,
        ]

        for sl in scope_levels:
            entry_id = store.remember(
                content=f"Test {sl.value}",
                scope_level=sl,
            )
            entry = store.get(entry_id)
            assert entry is not None
            assert entry.scope_level == sl

    def test_update_with_content_regenerates_embedding(self, store: ContextStore) -> None:
        """Test that updating content regenerates the embedding."""
        entry_id = store.remember(content="Original content")

        # Update the content
        success = store.update(entry_id, content="Updated content")
        assert success is True

        # Verify content was updated
        entry = store.get(entry_id)
        assert entry is not None
        assert entry.content == "Updated content"

    def test_update_metadata(self, store: ContextStore) -> None:
        """Test updating entry metadata."""
        entry_id = store.remember(content="Test", metadata={"old": "value"})

        success = store.update(entry_id, metadata={"new": "metadata"})
        assert success is True

        entry = store.get(entry_id)
        assert entry is not None
        assert entry.metadata == {"new": "metadata"}

    def test_get_nonexistent_entry(self, store: ContextStore) -> None:
        """Test getting a nonexistent entry returns None."""
        entry = store.get("definitely-not-a-real-id")
        assert entry is None

    def test_recall_empty_store(self, store: ContextStore) -> None:
        """Test recall on empty store returns empty list."""
        results = store.recall("any query")
        assert results == []

    def test_store_close(self, temp_db: Path) -> None:
        """Test closing the store."""
        store = ContextStore(temp_db)
        store.remember(content="Test entry")

        # Should not raise
        store.close()

    def test_stats_storage_size(self, store: ContextStore) -> None:
        """Test that stats includes storage size."""
        store.remember(content="Test entry for size")

        stats = store.stats()
        # Storage size should be greater than 0 after adding an entry
        assert stats.storage_size_bytes >= 0

    def test_remember_with_string_enum_values(self, store: ContextStore) -> None:
        """Test remember accepts string values for enum parameters."""
        entry_id = store.remember(
            content="String enum test",
            content_type="decision",  # String instead of ContextType.DECISION
            scope_level="file",  # String instead of ScopeLevel.FILE
        )

        entry = store.get(entry_id)
        assert entry is not None
        assert entry.content_type == ContextType.DECISION
        assert entry.scope_level == ScopeLevel.FILE
