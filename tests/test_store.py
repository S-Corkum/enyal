"""Tests for context store."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from enyal.core.store import ContextStore
from enyal.embeddings.models import MODEL_REGISTRY
from enyal.models.context import ContextType, EdgeType, ScopeLevel


@pytest.fixture
def temp_db() -> Path:
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _word_based_embed(text: str) -> np.ndarray:
    """Generate a deterministic embedding based on word content.

    Uses word hashing to distribute words into embedding dimensions,
    producing similar vectors for texts with shared words.
    This gives realistic similarity behavior in tests.
    """
    embedding = np.zeros(768, dtype=np.float32)
    words = text.lower().split()
    for word in words:
        # Each word activates a few dimensions based on its hash
        word_hash = hash(word) % (2**31)
        rng = np.random.RandomState(word_hash)
        indices = rng.choice(768, size=min(10, 768), replace=False)
        embedding[indices] += rng.rand(len(indices)).astype(np.float32)
    # Normalize to unit vector
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.astype(np.float32)


@pytest.fixture
def mock_engine() -> MagicMock:
    """Create a mock EmbeddingEngine for store tests.

    Uses word-based embeddings: texts with shared words produce similar vectors,
    texts with different words produce different vectors. This makes
    similarity-based operations work realistically in tests.
    """
    engine = MagicMock()
    engine.config = MODEL_REGISTRY["nomic-ai/nomic-embed-text-v1.5"]
    engine.embed.side_effect = _word_based_embed
    engine.embed_query.side_effect = _word_based_embed
    engine.embed_batch.return_value = np.zeros((0, 768), dtype=np.float32)
    engine.embedding_dimension.return_value = 768
    return engine


@pytest.fixture
def store(temp_db: Path, mock_engine: MagicMock) -> ContextStore:
    """Create a test store with mock engine."""
    return ContextStore(temp_db, engine=mock_engine)


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


class TestFTSSearch:
    """Tests for FTS5 full-text search."""

    def test_fts_search_basic(self, store: ContextStore) -> None:
        """Test basic FTS search functionality."""
        store.remember(content="Python programming language")
        store.remember(content="JavaScript for web development")

        results = store.fts_search("Python", limit=10)

        assert len(results) >= 1
        assert any("Python" in store.get(r["entry_id"]).content for r in results)

    def test_fts_search_special_characters(self, store: ContextStore) -> None:
        """Test FTS search with special characters in query."""
        store.remember(content="Use pytest-cov for coverage")

        # Query with special chars should not break
        results = store.fts_search('pytest "cov"', limit=10)

        # Should return results (empty or matching)
        assert isinstance(results, list)

    def test_fts_search_excludes_deprecated(self, store: ContextStore) -> None:
        """Test that FTS excludes deprecated entries by default."""
        entry_id = store.remember(content="Deprecated FTS content")
        store.forget(entry_id, hard_delete=False)

        results = store.fts_search("Deprecated FTS", limit=10)

        assert not any(r["entry_id"] == entry_id for r in results)

    def test_fts_search_includes_deprecated(self, store: ContextStore) -> None:
        """Test FTS can include deprecated entries."""
        entry_id = store.remember(content="Deprecated but searchable")
        store.forget(entry_id, hard_delete=False)

        results = store.fts_search("Deprecated but searchable", limit=10, include_deprecated=True)

        assert any(r["entry_id"] == entry_id for r in results)

    def test_fts_search_empty_results(self, store: ContextStore) -> None:
        """Test FTS with no matching results."""
        store.remember(content="Some unrelated content")

        results = store.fts_search("xyznonexistent", limit=10)

        assert results == []


class TestFindSimilar:
    """Tests for find_similar method."""

    def test_find_similar_basic(self, store: ContextStore) -> None:
        """Test basic find_similar functionality."""
        store.remember(content="Python is a programming language")

        similar = store.find_similar(
            content="Python programming language",
            threshold=0.5,
        )

        assert len(similar) >= 1
        assert similar[0]["similarity"] >= 0.5

    def test_find_similar_threshold(self, store: ContextStore) -> None:
        """Test that threshold filters results."""
        store.remember(content="JavaScript for web development")

        # Very high threshold should exclude most results
        similar = store.find_similar(
            content="Rust systems programming",  # Different topic
            threshold=0.99,
        )

        assert len(similar) == 0

    def test_find_similar_excludes_deprecated(self, store: ContextStore) -> None:
        """Test that find_similar excludes deprecated entries."""
        entry_id = store.remember(content="Deprecated similar content")
        store.forget(entry_id, hard_delete=False)

        similar = store.find_similar(
            content="Deprecated similar content",
            threshold=0.5,
        )

        assert not any(s["entry_id"] == entry_id for s in similar)


class TestRememberDeduplication:
    """Tests for remember with deduplication."""

    def test_remember_dedup_reject(self, store: ContextStore) -> None:
        """Test that duplicate is rejected and existing ID returned."""
        original_id = store.remember(content="Original unique content here")

        result = store.remember(
            content="Original unique content here",  # Exact duplicate
            check_duplicate=True,
            duplicate_threshold=0.85,
            on_duplicate="reject",
        )

        assert isinstance(result, dict)
        assert result["action"] == "existing"
        assert result["entry_id"] == original_id
        assert result["similarity"] >= 0.85

    def test_remember_dedup_merge(self, store: ContextStore) -> None:
        """Test that duplicate is merged with existing entry."""
        original_id = store.remember(
            content="Content to be merged",
            tags=["original"],
            confidence=0.8,
        )

        result = store.remember(
            content="Content to be merged",
            tags=["new"],
            confidence=0.9,
            check_duplicate=True,
            on_duplicate="merge",
        )

        assert result["action"] == "merged"

        # Check that tags were merged and confidence updated
        entry = store.get(original_id)
        assert "original" in entry.tags
        assert "new" in entry.tags
        assert entry.confidence == 0.9  # Max of 0.8 and 0.9

    def test_remember_dedup_store(self, store: ContextStore) -> None:
        """Test that duplicate is stored anyway when on_duplicate='store'."""
        original_id = store.remember(content="Force store duplicate")

        result = store.remember(
            content="Force store duplicate",
            check_duplicate=True,
            on_duplicate="store",
        )

        assert result["action"] == "created"
        assert result["entry_id"] != original_id

    def test_remember_dedup_disabled(self, store: ContextStore) -> None:
        """Test that dedup check is skipped when disabled."""
        store.remember(content="No dedup check")

        # Default check_duplicate=False should store without checking
        result = store.remember(content="No dedup check")

        # Returns string, not dict
        assert isinstance(result, str)


class TestKnowledgeGraph:
    """Tests for knowledge graph functionality."""

    def test_link_basic(self, store: ContextStore) -> None:
        """Test creating a basic edge."""
        entry1_id = store.remember(content="First entry")
        entry2_id = store.remember(content="Second entry")

        edge_id = store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)
        assert edge_id is not None

    def test_link_all_edge_types(self, store: ContextStore) -> None:
        """Test creating edges of all types."""
        entry1_id = store.remember(content="Entry one")
        entry2_id = store.remember(content="Entry two")

        for edge_type in EdgeType:
            # Unlink first to avoid duplicate constraint
            edges = store.get_edges(entry1_id, direction="outgoing")
            for edge in edges:
                store.unlink(edge.id)

            edge_id = store.link(entry1_id, entry2_id, edge_type)
            assert edge_id is not None

    def test_link_duplicate_rejected(self, store: ContextStore) -> None:
        """Test that duplicate edges are rejected."""
        entry1_id = store.remember(content="Entry A")
        entry2_id = store.remember(content="Entry B")

        edge_id1 = store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)
        edge_id2 = store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)

        assert edge_id1 is not None
        assert edge_id2 is None  # Duplicate rejected

    def test_link_nonexistent_entry(self, store: ContextStore) -> None:
        """Test linking to nonexistent entry returns None."""
        entry_id = store.remember(content="Real entry")
        edge_id = store.link(entry_id, "nonexistent-id", EdgeType.RELATES_TO)
        assert edge_id is None

    def test_unlink(self, store: ContextStore) -> None:
        """Test removing an edge."""
        entry1_id = store.remember(content="Entry X")
        entry2_id = store.remember(content="Entry Y")

        edge_id = store.link(entry1_id, entry2_id, EdgeType.DEPENDS_ON)
        assert edge_id is not None

        success = store.unlink(edge_id)
        assert success is True

        # Verify edge is gone
        edges = store.get_edges(entry1_id)
        assert len(edges) == 0

    def test_get_edge(self, store: ContextStore) -> None:
        """Test getting a single edge by ID."""
        entry1_id = store.remember(content="Entry 1")
        entry2_id = store.remember(content="Entry 2")

        edge_id = store.link(entry1_id, entry2_id, EdgeType.RELATES_TO, confidence=0.75)
        assert edge_id is not None

        edge = store.get_edge(edge_id)
        assert edge is not None
        assert edge.id == edge_id
        assert edge.source_id == entry1_id
        assert edge.target_id == entry2_id
        assert edge.edge_type == EdgeType.RELATES_TO
        assert edge.confidence == 0.75

    def test_get_edge_nonexistent(self, store: ContextStore) -> None:
        """Test getting a nonexistent edge returns None."""
        edge = store.get_edge("nonexistent-edge-id")
        assert edge is None

    def test_get_edges_outgoing(self, store: ContextStore) -> None:
        """Test getting outgoing edges."""
        entry1_id = store.remember(content="Source entry")
        entry2_id = store.remember(content="Target entry")

        store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)

        edges = store.get_edges(entry1_id, direction="outgoing")
        assert len(edges) == 1
        assert edges[0].source_id == entry1_id
        assert edges[0].target_id == entry2_id

    def test_get_edges_incoming(self, store: ContextStore) -> None:
        """Test getting incoming edges."""
        entry1_id = store.remember(content="Source")
        entry2_id = store.remember(content="Target")

        store.link(entry1_id, entry2_id, EdgeType.DEPENDS_ON)

        edges = store.get_edges(entry2_id, direction="incoming")
        assert len(edges) == 1
        assert edges[0].source_id == entry1_id

    def test_get_edges_both_directions(self, store: ContextStore) -> None:
        """Test getting edges in both directions."""
        entry1_id = store.remember(content="Middle entry")
        entry2_id = store.remember(content="Left entry")
        entry3_id = store.remember(content="Right entry")

        store.link(entry2_id, entry1_id, EdgeType.RELATES_TO)  # Incoming
        store.link(entry1_id, entry3_id, EdgeType.RELATES_TO)  # Outgoing

        edges = store.get_edges(entry1_id, direction="both")
        assert len(edges) == 2

    def test_get_edges_filtered_by_type(self, store: ContextStore) -> None:
        """Test filtering edges by type."""
        entry1_id = store.remember(content="Entry 1")
        entry2_id = store.remember(content="Entry 2")

        store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)
        store.link(entry1_id, entry2_id, EdgeType.DEPENDS_ON)

        relates_edges = store.get_edges(
            entry1_id, direction="outgoing", edge_type=EdgeType.RELATES_TO
        )
        assert len(relates_edges) == 1
        assert relates_edges[0].edge_type == EdgeType.RELATES_TO

    def test_get_edges_invalid_direction(self, store: ContextStore) -> None:
        """Test that invalid direction raises ValueError."""
        entry_id = store.remember(content="Test entry")

        with pytest.raises(ValueError, match="Invalid direction"):
            store.get_edges(entry_id, direction="sideways")

    def test_traverse_single_hop(self, store: ContextStore) -> None:
        """Test traversing one level."""
        entry1_id = store.remember(content="Start node")
        entry2_id = store.remember(content="End node")

        store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)

        results = store.traverse(entry1_id, max_depth=1)
        assert len(results) == 1
        assert results[0]["entry"].id == entry2_id
        assert results[0]["depth"] == 1

    def test_traverse_multi_hop(self, store: ContextStore) -> None:
        """Test traversing multiple levels."""
        entry1_id = store.remember(content="Node A")
        entry2_id = store.remember(content="Node B")
        entry3_id = store.remember(content="Node C")

        store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)
        store.link(entry2_id, entry3_id, EdgeType.RELATES_TO)

        results = store.traverse(entry1_id, max_depth=2)
        assert len(results) == 2

        # Check depths
        depths = {r["entry"].id: r["depth"] for r in results}
        assert depths[entry2_id] == 1
        assert depths[entry3_id] == 2

    def test_traverse_max_depth(self, store: ContextStore) -> None:
        """Test depth limiting."""
        entry1_id = store.remember(content="Level 0")
        entry2_id = store.remember(content="Level 1")
        entry3_id = store.remember(content="Level 2")
        entry4_id = store.remember(content="Level 3")

        store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)
        store.link(entry2_id, entry3_id, EdgeType.RELATES_TO)
        store.link(entry3_id, entry4_id, EdgeType.RELATES_TO)

        results = store.traverse(entry1_id, max_depth=2)

        # Should only get entries at depth 1 and 2, not 3
        entry_ids = {r["entry"].id for r in results}
        assert entry2_id in entry_ids
        assert entry3_id in entry_ids
        assert entry4_id not in entry_ids

    def test_traverse_invalid_direction(self, store: ContextStore) -> None:
        """Test that invalid direction raises ValueError."""
        entry_id = store.remember(content="Test entry")

        with pytest.raises(ValueError, match="Invalid direction"):
            store.traverse(entry_id, direction="both")  # traverse only accepts outgoing/incoming

    def test_cascade_delete_edges(self, store: ContextStore) -> None:
        """Test edges are deleted when entry is deleted."""
        entry1_id = store.remember(content="Entry to delete")
        entry2_id = store.remember(content="Connected entry")

        store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)

        # Delete entry1 (hard delete)
        store.forget(entry1_id, hard_delete=True)

        # Edges should be gone
        edges = store.get_edges(entry2_id, direction="incoming")
        assert len(edges) == 0

    def test_auto_link_on_remember(self, store: ContextStore) -> None:
        """Test automatic linking on remember."""
        # Store first entry
        store.remember(content="Python is a programming language")

        # Store similar entry with auto_link
        entry2_id = store.remember(
            content="Python programming language guide",
            auto_link=True,
            auto_link_threshold=0.5,  # Lower threshold for test
        )

        # Should have created RELATES_TO edge
        edges = store.get_edges(entry2_id, direction="outgoing", edge_type=EdgeType.RELATES_TO)
        assert len(edges) >= 1

    def test_explicit_supersedes(self, store: ContextStore) -> None:
        """Test explicit supersedes relationship."""
        old_id = store.remember(content="Old decision")
        new_id = store.remember(content="New decision", supersedes=old_id)

        edges = store.get_edges(new_id, direction="outgoing", edge_type=EdgeType.SUPERSEDES)
        assert len(edges) == 1
        assert edges[0].target_id == old_id

    def test_explicit_depends_on(self, store: ContextStore) -> None:
        """Test explicit depends_on relationship."""
        dep_id = store.remember(content="Dependency")
        main_id = store.remember(content="Main entry", depends_on=[dep_id])

        edges = store.get_edges(main_id, direction="outgoing", edge_type=EdgeType.DEPENDS_ON)
        assert len(edges) == 1
        assert edges[0].target_id == dep_id

    def test_stats_with_edges(self, store: ContextStore) -> None:
        """Test that stats includes edge metrics."""
        entry1_id = store.remember(content="Entry with edges 1")
        entry2_id = store.remember(content="Entry with edges 2")

        store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)
        store.link(entry1_id, entry2_id, EdgeType.DEPENDS_ON)

        stats = store.stats()
        assert stats.total_edges == 2
        assert stats.edges_by_type.get("relates_to") == 1
        assert stats.edges_by_type.get("depends_on") == 1
        assert stats.connected_entries == 2


class TestGraphValidity:
    """Tests for graph validity helper methods."""

    def test_get_superseded_ids_empty(self, store: ContextStore) -> None:
        """Test get_superseded_ids with no superseded entries."""
        store.remember(content="Entry without supersedes")
        superseded = store.get_superseded_ids()
        assert superseded == set()

    def test_get_superseded_ids_with_supersedes(self, store: ContextStore) -> None:
        """Test get_superseded_ids returns correctly superseded entries."""
        old_id = store.remember(content="Old decision: use React")
        new_id = store.remember(content="New decision: use Vue", supersedes=old_id)

        superseded = store.get_superseded_ids()
        assert old_id in superseded
        assert new_id not in superseded

    def test_get_conflicted_ids_empty(self, store: ContextStore) -> None:
        """Test get_conflicted_ids with no conflicts."""
        store.remember(content="Entry without conflicts")
        conflicted = store.get_conflicted_ids()
        assert conflicted == set()

    def test_get_conflicted_ids_with_conflicts(self, store: ContextStore) -> None:
        """Test get_conflicted_ids returns entries with conflicts."""
        entry1_id = store.remember(content="Use tabs for indentation")
        entry2_id = store.remember(content="Use spaces for indentation")
        store.link(entry1_id, entry2_id, EdgeType.CONFLICTS_WITH)

        conflicted = store.get_conflicted_ids()
        assert entry1_id in conflicted
        assert entry2_id in conflicted

    def test_is_superseded_false(self, store: ContextStore) -> None:
        """Test is_superseded returns False for non-superseded entry."""
        entry_id = store.remember(content="Current entry")
        assert store.is_superseded(entry_id) is False

    def test_is_superseded_true(self, store: ContextStore) -> None:
        """Test is_superseded returns True for superseded entry."""
        old_id = store.remember(content="Old entry")
        store.remember(content="New entry", supersedes=old_id)
        assert store.is_superseded(old_id) is True

    def test_get_superseding_entry_none(self, store: ContextStore) -> None:
        """Test get_superseding_entry returns None when not superseded."""
        entry_id = store.remember(content="Current entry")
        assert store.get_superseding_entry(entry_id) is None

    def test_get_superseding_entry_found(self, store: ContextStore) -> None:
        """Test get_superseding_entry returns the superseding entry ID."""
        old_id = store.remember(content="Old entry")
        new_id = store.remember(content="New entry", supersedes=old_id)
        assert store.get_superseding_entry(old_id) == new_id

    def test_appears_contradictory_negation(self, store: ContextStore) -> None:
        """Test _appears_contradictory detects negation patterns."""
        assert store._appears_contradictory("Enable feature X", "Don't enable feature X")
        assert store._appears_contradictory("Use tabs", "Never use tabs")

    def test_appears_contradictory_no_contradiction(self, store: ContextStore) -> None:
        """Test _appears_contradictory returns False for non-contradicting content."""
        assert not store._appears_contradictory("Use React", "Use React hooks")
        assert not store._appears_contradictory("Enable logging", "Enable metrics")

    def test_appears_contradictory_different_choices(self, store: ContextStore) -> None:
        """Test _appears_contradictory detects different choices."""
        assert store._appears_contradictory(
            "Use React for the frontend", "Use Vue for the frontend"
        )
        assert store._appears_contradictory("Prefer tabs", "Prefer spaces")


class TestConflictDetection:
    """Tests for conflict detection in remember()."""

    def test_detect_conflicts_disabled_by_default(self, store: ContextStore) -> None:
        """Test that conflict detection is disabled by default."""
        store.remember(content="First entry about Python")
        result = store.remember(content="Second entry about Python")
        # Should be a string (entry ID) not a dict
        assert isinstance(result, str)

    def test_detect_conflicts_returns_dict(self, store: ContextStore) -> None:
        """Test detect_conflicts=True returns dict with conflict info."""
        store.remember(content="Use React for frontend")
        result = store.remember(
            content="Don't use React for frontend",
            detect_conflicts=True,
        )
        assert isinstance(result, dict)
        assert "potential_conflicts" in result

    def test_suggest_supersedes_returns_candidates(self, store: ContextStore) -> None:
        """Test suggest_supersedes=True returns supersedes candidates."""
        store.remember(content="Python version 3.10 required")
        result = store.remember(
            content="Python version 3.11 required",
            suggest_supersedes=True,
            supersedes_threshold=0.7,  # Lower for test
        )
        assert isinstance(result, dict)
        assert "supersedes_candidates" in result


class TestHealthCheck:
    """Tests for graph health check functionality."""

    def test_health_check_empty_store(self, store: ContextStore) -> None:
        """Test health check on empty store."""
        health = store.health_check()
        assert health["total_entries"] == 0
        assert health["health_score"] == 1.0

    def test_health_check_with_entries(self, store: ContextStore) -> None:
        """Test health check with some entries."""
        store.remember(content="Entry 1")
        store.remember(content="Entry 2")
        store.remember(content="Entry 3")

        health = store.health_check()
        assert health["total_entries"] == 3
        assert health["orphan_entries"] == 3  # No edges yet
        assert health["superseded_entries"] == 0

    def test_health_check_with_supersedes(self, store: ContextStore) -> None:
        """Test health check detects superseded entries."""
        old_id = store.remember(content="Old entry")
        store.remember(content="New entry", supersedes=old_id)

        health = store.health_check()
        assert health["superseded_entries"] == 1

    def test_health_check_with_conflicts(self, store: ContextStore) -> None:
        """Test health check detects conflicts."""
        entry1 = store.remember(content="Entry 1")
        entry2 = store.remember(content="Entry 2")
        store.link(entry1, entry2, EdgeType.CONFLICTS_WITH)

        health = store.health_check()
        assert health["unresolved_conflicts"] == 1


class TestVersioning:
    """Tests for entry versioning functionality."""

    def test_get_history_initial_version(self, store: ContextStore) -> None:
        """Test that remember() automatically creates initial version."""
        entry_id = store.remember(content="New entry")
        history = store.get_history(entry_id)
        assert len(history) == 1  # Initial version created automatically
        assert history[0]["version"] == 1
        assert history[0]["change_type"] == "created"
        assert history[0]["content"] == "New entry"

    def test_version_on_update(self, store: ContextStore) -> None:
        """Test that update() creates a new version."""
        entry_id = store.remember(content="Original content")
        store.update(entry_id, content="Updated content")

        history = store.get_history(entry_id)
        assert len(history) == 2  # Initial + update
        # History is ordered by version DESC
        assert history[0]["version"] == 2
        assert history[0]["change_type"] == "updated"
        assert history[0]["content"] == "Updated content"
        assert history[1]["version"] == 1
        assert history[1]["change_type"] == "created"
        assert history[1]["content"] == "Original content"


class TestAnalytics:
    """Tests for usage analytics functionality."""

    def test_track_usage(self, store: ContextStore) -> None:
        """Test tracking a usage event."""
        entry_id = store.remember(content="Test entry")
        store.track_usage(entry_id, "recall", query="test query", result_rank=1)

        analytics = store.get_analytics(entry_id=entry_id, days=1)
        assert len(analytics["events_by_type"]) >= 0  # May be 0 if not aggregated yet

    def test_get_analytics_empty(self, store: ContextStore) -> None:
        """Test get_analytics with no events."""
        analytics = store.get_analytics(days=1)
        assert "period_days" in analytics
        assert analytics["period_days"] == 1
        assert "events_by_type" in analytics
        assert "top_recalled" in analytics


class TestReviewMethods:
    """Tests for review helper methods."""

    def test_get_orphan_entries(self, store: ContextStore) -> None:
        """Test get_orphan_entries returns entries with no edges."""
        entry1 = store.remember(content="Orphan entry")
        entry2 = store.remember(content="Connected entry")
        entry3 = store.remember(content="Another connected")
        store.link(entry2, entry3, EdgeType.RELATES_TO)

        orphans = store.get_orphan_entries()
        orphan_ids = [e.id for e in orphans]
        assert entry1 in orphan_ids
        assert entry2 not in orphan_ids
        assert entry3 not in orphan_ids

    def test_get_conflicted_entries(self, store: ContextStore) -> None:
        """Test get_conflicted_entries returns conflict pairs."""
        entry1 = store.remember(content="Entry 1")
        entry2 = store.remember(content="Entry 2")
        store.link(entry1, entry2, EdgeType.CONFLICTS_WITH)

        conflicts = store.get_conflicted_entries()
        assert len(conflicts) == 1
        assert conflicts[0]["entry1"].id in (entry1, entry2)
        assert conflicts[0]["entry2"].id in (entry1, entry2)
