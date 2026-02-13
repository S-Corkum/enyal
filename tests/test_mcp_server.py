"""Tests for MCP server module."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from enyal.models.context import (
    ContextEntry,
    ContextSearchResult,
    ContextStats,
    ContextType,
    ScopeLevel,
)


# Patch FastMCP at import time since it's used at module level
@pytest.fixture(scope="module", autouse=True)
def mock_fastmcp():
    """Mock FastMCP at module level before importing server."""
    with patch("fastmcp.FastMCP") as mock:
        mock.return_value = MagicMock()
        yield mock


# Import server module after patching
@pytest.fixture
def server_module():
    """Import the server module with FastMCP mocked."""
    # Remove cached module if present
    modules_to_remove = [k for k in sys.modules if k.startswith("enyal.mcp")]
    for mod in modules_to_remove:
        del sys.modules[mod]

    with patch("fastmcp.FastMCP") as mock_fmcp:
        mock_mcp = MagicMock()
        mock_fmcp.return_value = mock_mcp
        # Make the decorator return the original function
        mock_mcp.tool.return_value = lambda f: f

        import enyal.mcp.server as server_module

        yield server_module

        # Reset the module-level globals
        server_module._store = None
        server_module._retrieval = None


class TestGetStore:
    """Tests for get_store function."""

    def test_get_store_initialization(self, server_module) -> None:
        """Test that get_store initializes and returns a store."""
        with (
            patch.object(server_module, "ContextStore") as mock_store_class,
            patch("enyal.embeddings.engine.EmbeddingEngine"),
            patch("enyal.embeddings.models.ModelConfig"),
        ):
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            server_module._store = None

            result = server_module.get_store()

            mock_store_class.assert_called_once()
            assert result == mock_store

    def test_get_store_uses_env_var(self, server_module) -> None:
        """Test that get_store uses ENYAL_DB_PATH env var."""
        with (
            patch.dict(os.environ, {"ENYAL_DB_PATH": "/custom/db/path.db"}),
            patch.object(server_module, "ContextStore") as mock_store_class,
            patch("enyal.embeddings.engine.EmbeddingEngine"),
            patch("enyal.embeddings.models.ModelConfig"),
        ):
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            server_module._store = None

            server_module.get_store()

            assert mock_store_class.call_args[0][0] == "/custom/db/path.db"

    def test_get_store_caches_instance(self, server_module) -> None:
        """Test that get_store returns cached instance on subsequent calls."""
        with (
            patch.object(server_module, "ContextStore") as mock_store_class,
            patch("enyal.embeddings.engine.EmbeddingEngine"),
            patch("enyal.embeddings.models.ModelConfig"),
        ):
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            server_module._store = None

            result1 = server_module.get_store()
            result2 = server_module.get_store()

            # Should only be called once due to caching
            mock_store_class.assert_called_once()
            assert result1 is result2


class TestGetRetrieval:
    """Tests for get_retrieval function."""

    def test_get_retrieval_initialization(self, server_module) -> None:
        """Test that get_retrieval initializes and returns a retrieval engine."""
        with (
            patch.object(server_module, "ContextStore"),
            patch.object(server_module, "RetrievalEngine") as mock_retrieval_class,
            patch("enyal.embeddings.engine.EmbeddingEngine"),
            patch("enyal.embeddings.models.ModelConfig"),
        ):
            mock_retrieval = MagicMock()
            mock_retrieval_class.return_value = mock_retrieval
            server_module._store = None
            server_module._retrieval = None

            result = server_module.get_retrieval()

            mock_retrieval_class.assert_called_once()
            assert result == mock_retrieval

    def test_get_retrieval_caches_instance(self, server_module) -> None:
        """Test that get_retrieval returns cached instance on subsequent calls."""
        with (
            patch.object(server_module, "ContextStore"),
            patch.object(server_module, "RetrievalEngine") as mock_retrieval_class,
            patch("enyal.embeddings.engine.EmbeddingEngine"),
            patch("enyal.embeddings.models.ModelConfig"),
        ):
            mock_retrieval = MagicMock()
            mock_retrieval_class.return_value = mock_retrieval
            server_module._store = None
            server_module._retrieval = None

            result1 = server_module.get_retrieval()
            result2 = server_module.get_retrieval()

            # Should only be called once due to caching
            mock_retrieval_class.assert_called_once()
            assert result1 is result2


class TestRememberInput:
    """Tests for RememberInput model."""

    def test_remember_input_defaults(self, server_module) -> None:
        """Test RememberInput default values."""
        input_data = server_module.RememberInput(content="Test content")

        assert input_data.content == "Test content"
        assert input_data.content_type == "fact"
        assert input_data.scope == "project"
        assert input_data.scope_path is None
        assert input_data.source is None
        assert input_data.tags == []

    def test_remember_input_all_fields(self, server_module) -> None:
        """Test RememberInput with all fields."""
        input_data = server_module.RememberInput(
            content="Test content",
            content_type="decision",
            scope="file",
            scope_path="/path/to/file.py",
            source="conversation-123",
            tags=["tag1", "tag2"],
        )

        assert input_data.content_type == "decision"
        assert input_data.scope == "file"
        assert input_data.scope_path == "/path/to/file.py"
        assert input_data.source == "conversation-123"
        assert input_data.tags == ["tag1", "tag2"]


class TestRecallInput:
    """Tests for RecallInput model."""

    def test_recall_input_defaults(self, server_module) -> None:
        """Test RecallInput default values."""
        input_data = server_module.RecallInput(query="test query")

        assert input_data.query == "test query"
        assert input_data.limit == 10
        assert input_data.scope is None
        assert input_data.scope_path is None
        assert input_data.content_type is None
        assert input_data.min_confidence == 0.3

    def test_recall_input_validation(self, server_module) -> None:
        """Test RecallInput validation."""
        input_data = server_module.RecallInput(
            query="test",
            limit=50,
            scope="project",
            content_type="fact",
            min_confidence=0.7,
        )

        assert input_data.limit == 50
        assert input_data.scope == "project"
        assert input_data.min_confidence == 0.7


class TestUpdateInput:
    """Tests for UpdateInput model."""

    def test_update_input_validation(self, server_module) -> None:
        """Test UpdateInput with various fields."""
        input_data = server_module.UpdateInput(
            entry_id="test-id",
            content="Updated content",
            confidence=0.8,
            tags=["new", "tags"],
        )

        assert input_data.entry_id == "test-id"
        assert input_data.content == "Updated content"
        assert input_data.confidence == 0.8
        assert input_data.tags == ["new", "tags"]

    def test_update_input_minimal(self, server_module) -> None:
        """Test UpdateInput with only required fields."""
        input_data = server_module.UpdateInput(entry_id="test-id")

        assert input_data.entry_id == "test-id"
        assert input_data.content is None
        assert input_data.confidence is None
        assert input_data.tags is None


class TestEnyalRemember:
    """Tests for enyal_remember tool."""

    def test_enyal_remember_success(self, server_module) -> None:
        """Test successful remember operation."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.remember.return_value = "new-entry-id-123"
            mock_get_store.return_value = mock_store

            input_data = server_module.RememberInput(
                content="Test knowledge to store",
                content_type="fact",
                scope="project",
            )

            result = server_module.enyal_remember(input_data)

            assert result["success"] is True
            assert result["entry_id"] == "new-entry-id-123"
            assert "message" in result

    def test_enyal_remember_with_all_options(self, server_module) -> None:
        """Test remember with all options."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.remember.return_value = "new-entry-id"
            mock_get_store.return_value = mock_store

            input_data = server_module.RememberInput(
                content="Complex entry",
                content_type="decision",
                scope="file",
                scope_path="/path/to/file.py",
                source="session-abc",
                tags=["important", "architecture"],
            )

            result = server_module.enyal_remember(input_data)

            assert result["success"] is True
            mock_store.remember.assert_called_once_with(
                content="Complex entry",
                content_type=ContextType.DECISION,
                scope_level=ScopeLevel.FILE,
                scope_path="/path/to/file.py",
                source_type="conversation",
                source_ref="session-abc",
                tags=["important", "architecture"],
                check_duplicate=False,
                duplicate_threshold=0.85,
                on_duplicate="reject",
                # Graph parameters
                auto_link=False,
                auto_link_threshold=0.85,
                relates_to=None,
                supersedes=None,
                depends_on=None,
                # Conflict/supersedes detection
                detect_conflicts=False,
                suggest_supersedes=False,
                auto_supersede=False,
            )

    def test_enyal_remember_error(self, server_module) -> None:
        """Test remember operation with error."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.remember.side_effect = Exception("Database error")
            mock_get_store.return_value = mock_store

            input_data = server_module.RememberInput(content="Test content")

            result = server_module.enyal_remember(input_data)

            assert result["success"] is False
            assert "error" in result
            assert "Database error" in result["error"]


class TestEnyalRecall:
    """Tests for enyal_recall tool."""

    def test_enyal_recall_success(self, server_module, sample_entry: ContextEntry) -> None:
        """Test successful recall operation."""
        mock_result = ContextSearchResult(entry=sample_entry, distance=0.25, score=0.8)

        with patch.object(server_module, "get_retrieval") as mock_get_retrieval:
            mock_retrieval = MagicMock()
            mock_retrieval.search.return_value = [mock_result]
            mock_get_retrieval.return_value = mock_retrieval

            input_data = server_module.RecallInput(query="test query")

            result = server_module.enyal_recall(input_data)

            assert result["success"] is True
            assert result["count"] == 1
            assert len(result["results"]) == 1
            assert result["results"][0]["content"] == "Test content for unit tests"

    def test_enyal_recall_with_filters(self, server_module, sample_entry: ContextEntry) -> None:
        """Test recall with filters."""
        mock_result = ContextSearchResult(entry=sample_entry, distance=0.25, score=0.8)

        with patch.object(server_module, "get_retrieval") as mock_get_retrieval:
            mock_retrieval = MagicMock()
            mock_retrieval.search.return_value = [mock_result]
            mock_get_retrieval.return_value = mock_retrieval

            input_data = server_module.RecallInput(
                query="test query",
                limit=5,
                scope="project",
                content_type="fact",
                min_confidence=0.5,
            )

            result = server_module.enyal_recall(input_data)

            assert result["success"] is True
            mock_retrieval.search.assert_called_once_with(
                query="test query",
                limit=5,
                scope_level=ScopeLevel.PROJECT,
                scope_path=None,
                content_type=ContextType.FACT,
                min_confidence=0.5,
                # Validity parameters
                exclude_superseded=True,
                flag_conflicts=True,
                freshness_boost=0.1,
            )

    def test_enyal_recall_empty_results(self, server_module) -> None:
        """Test recall with no results."""
        with patch.object(server_module, "get_retrieval") as mock_get_retrieval:
            mock_retrieval = MagicMock()
            mock_retrieval.search.return_value = []
            mock_get_retrieval.return_value = mock_retrieval

            input_data = server_module.RecallInput(query="nonexistent query")

            result = server_module.enyal_recall(input_data)

            assert result["success"] is True
            assert result["count"] == 0
            assert result["results"] == []

    def test_enyal_recall_error(self, server_module) -> None:
        """Test recall operation with error."""
        with patch.object(server_module, "get_retrieval") as mock_get_retrieval:
            mock_retrieval = MagicMock()
            mock_retrieval.search.side_effect = Exception("Search error")
            mock_get_retrieval.return_value = mock_retrieval

            input_data = server_module.RecallInput(query="test query")

            result = server_module.enyal_recall(input_data)

            assert result["success"] is False
            assert "error" in result
            assert result["results"] == []


class TestEnyalForget:
    """Tests for enyal_forget tool."""

    def test_enyal_forget_success(self, server_module) -> None:
        """Test successful forget operation (soft delete)."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.forget.return_value = True
            mock_get_store.return_value = mock_store

            input_data = server_module.ForgetInput(entry_id="test-entry-id", hard_delete=False)

            result = server_module.enyal_forget(input_data)

            assert result["success"] is True
            assert "deprecated" in result["message"]
            mock_store.forget.assert_called_once_with("test-entry-id", hard_delete=False)

    def test_enyal_forget_hard_delete(self, server_module) -> None:
        """Test forget with hard delete."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.forget.return_value = True
            mock_get_store.return_value = mock_store

            input_data = server_module.ForgetInput(entry_id="test-entry-id", hard_delete=True)

            result = server_module.enyal_forget(input_data)

            assert result["success"] is True
            assert "permanently deleted" in result["message"]
            mock_store.forget.assert_called_once_with("test-entry-id", hard_delete=True)

    def test_enyal_forget_not_found(self, server_module) -> None:
        """Test forget when entry not found."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.forget.return_value = False
            mock_get_store.return_value = mock_store

            input_data = server_module.ForgetInput(entry_id="nonexistent-id")

            result = server_module.enyal_forget(input_data)

            assert result["success"] is False
            assert "not found" in result["error"]

    def test_enyal_forget_error(self, server_module) -> None:
        """Test forget operation with error."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.forget.side_effect = Exception("Database error")
            mock_get_store.return_value = mock_store

            input_data = server_module.ForgetInput(entry_id="test-id")

            result = server_module.enyal_forget(input_data)

            assert result["success"] is False
            assert "error" in result


class TestEnyalUpdate:
    """Tests for enyal_update tool."""

    def test_enyal_update_success(self, server_module) -> None:
        """Test successful update operation."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.update.return_value = True
            mock_get_store.return_value = mock_store

            input_data = server_module.UpdateInput(
                entry_id="test-entry-id",
                content="Updated content",
                confidence=0.9,
                tags=["new-tag"],
            )

            result = server_module.enyal_update(input_data)

            assert result["success"] is True
            assert "updated" in result["message"]
            mock_store.update.assert_called_once_with(
                entry_id="test-entry-id",
                content="Updated content",
                confidence=0.9,
                tags=["new-tag"],
            )

    def test_enyal_update_not_found(self, server_module) -> None:
        """Test update when entry not found."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.update.return_value = False
            mock_get_store.return_value = mock_store

            input_data = server_module.UpdateInput(entry_id="nonexistent-id", content="New content")

            result = server_module.enyal_update(input_data)

            assert result["success"] is False
            assert "not found" in result["error"]

    def test_enyal_update_error(self, server_module) -> None:
        """Test update operation with error."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.update.side_effect = Exception("Update error")
            mock_get_store.return_value = mock_store

            input_data = server_module.UpdateInput(entry_id="test-id", content="New content")

            result = server_module.enyal_update(input_data)

            assert result["success"] is False
            assert "error" in result


class TestEnyalStats:
    """Tests for enyal_stats tool."""

    def test_enyal_stats_success(self, server_module, sample_stats: ContextStats) -> None:
        """Test successful stats operation."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.stats.return_value = sample_stats
            mock_get_store.return_value = mock_store

            result = server_module.enyal_stats()

            assert result["success"] is True
            assert "stats" in result
            assert result["stats"]["total_entries"] == 100
            assert result["stats"]["active_entries"] == 90
            assert result["stats"]["deprecated_entries"] == 10

    def test_enyal_stats_error(self, server_module) -> None:
        """Test stats operation with error."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.stats.side_effect = Exception("Stats error")
            mock_get_store.return_value = mock_store

            result = server_module.enyal_stats()

            assert result["success"] is False
            assert "error" in result


class TestEnyalGet:
    """Tests for enyal_get tool."""

    def test_enyal_get_success(self, server_module, sample_entry: ContextEntry) -> None:
        """Test successful get operation."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get.return_value = sample_entry
            mock_get_store.return_value = mock_store

            result = server_module.enyal_get("test-entry-id")

            assert result["success"] is True
            assert "entry" in result
            assert result["entry"]["content"] == "Test content for unit tests"
            assert result["entry"]["type"] == "fact"

    def test_enyal_get_with_source(
        self, server_module, sample_entry_with_source: ContextEntry
    ) -> None:
        """Test get with entry that has source information."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get.return_value = sample_entry_with_source
            mock_get_store.return_value = mock_store

            result = server_module.enyal_get("test-entry-id")

            assert result["success"] is True
            assert result["entry"]["source_type"] == "conversation"
            assert result["entry"]["source_ref"] == "session-123"

    def test_enyal_get_not_found(self, server_module) -> None:
        """Test get when entry not found."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get.return_value = None
            mock_get_store.return_value = mock_store

            result = server_module.enyal_get("nonexistent-id")

            assert result["success"] is False
            assert "not found" in result["error"]

    def test_enyal_get_error(self, server_module) -> None:
        """Test get operation with error."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get.side_effect = Exception("Database error")
            mock_get_store.return_value = mock_store

            result = server_module.enyal_get("test-id")

            assert result["success"] is False
            assert "error" in result


class TestEnyalRecallByScope:
    """Tests for enyal_recall_by_scope tool."""

    def test_enyal_recall_by_scope_success(self, server_module, sample_entry: ContextEntry) -> None:
        """Test successful recall by scope operation."""
        mock_result = ContextSearchResult(entry=sample_entry, distance=0.25, score=0.8)

        with patch.object(server_module, "get_retrieval") as mock_get_retrieval:
            mock_retrieval = MagicMock()
            mock_retrieval.search_by_scope.return_value = [mock_result]
            mock_get_retrieval.return_value = mock_retrieval

            input_data = server_module.RecallByScopeInput(
                query="test query",
                file_path="/path/to/file.py",
            )

            result = server_module.enyal_recall_by_scope(input_data)

            assert result["success"] is True
            assert result["count"] == 1
            mock_retrieval.search_by_scope.assert_called_once()

    def test_enyal_recall_by_scope_error(self, server_module) -> None:
        """Test recall by scope with error."""
        with patch.object(server_module, "get_retrieval") as mock_get_retrieval:
            mock_retrieval = MagicMock()
            mock_retrieval.search_by_scope.side_effect = Exception("Scope error")
            mock_get_retrieval.return_value = mock_retrieval

            input_data = server_module.RecallByScopeInput(
                query="test",
                file_path="/path/to/file.py",
            )

            result = server_module.enyal_recall_by_scope(input_data)

            assert result["success"] is False
            assert "error" in result


class TestEnyalRememberDedup:
    """Tests for enyal_remember with deduplication."""

    def test_enyal_remember_dedup_reject(self, server_module) -> None:
        """Test remember with duplicate rejection."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.remember.return_value = {
                "entry_id": "existing-id",
                "action": "existing",
                "duplicate_of": "existing-id",
                "similarity": 0.92,
            }
            mock_get_store.return_value = mock_store

            input_data = server_module.RememberInput(
                content="Duplicate content",
                check_duplicate=True,
                on_duplicate="reject",
            )

            result = server_module.enyal_remember(input_data)

            assert result["success"] is True
            assert result["action"] == "existing"
            assert result["duplicate_of"] == "existing-id"
            assert "similarity" in result["message"]

    def test_enyal_remember_dedup_created(self, server_module) -> None:
        """Test remember creates new entry when no duplicate."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.remember.return_value = {
                "entry_id": "new-entry-id",
                "action": "created",
                "duplicate_of": None,
                "similarity": None,
            }
            mock_get_store.return_value = mock_store

            input_data = server_module.RememberInput(
                content="Unique content",
                check_duplicate=True,
            )

            result = server_module.enyal_remember(input_data)

            assert result["success"] is True
            assert result["action"] == "created"
            assert result["entry_id"] == "new-entry-id"


class TestEnyalRememberMerged:
    """Tests for enyal_remember with merged and detection results."""

    def test_enyal_remember_merged(self, server_module) -> None:
        """Test remember with merge action."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.remember.return_value = {
                "entry_id": "existing-id",
                "action": "merged",
                "duplicate_of": "existing-id",
                "similarity": 0.95,
            }
            mock_get_store.return_value = mock_store

            input_data = server_module.RememberInput(
                content="Merged content",
                check_duplicate=True,
                on_duplicate="merge",
            )

            result = server_module.enyal_remember(input_data)

            assert result["success"] is True
            assert result["action"] == "merged"
            assert "similarity" in result["message"]

    def test_enyal_remember_with_detect_conflicts(self, server_module) -> None:
        """Test remember with conflict detection enabled."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.remember.return_value = {
                "entry_id": "new-id",
                "action": "created",
                "duplicate_of": None,
                "similarity": None,
                "potential_conflicts": [
                    {"entry_id": "conflict-1", "content": "Conflicting", "similarity": 0.91}
                ],
            }
            mock_get_store.return_value = mock_store

            input_data = server_module.RememberInput(
                content="New content",
                detect_conflicts=True,
            )

            result = server_module.enyal_remember(input_data)

            assert result["success"] is True
            assert result["action"] == "created"
            assert "potential_conflicts" in result
            assert len(result["potential_conflicts"]) == 1

    def test_enyal_remember_with_suggest_supersedes(self, server_module) -> None:
        """Test remember with supersedes suggestion enabled."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.remember.return_value = {
                "entry_id": "new-id",
                "action": "created",
                "duplicate_of": None,
                "similarity": None,
                "supersedes_candidates": [
                    {"entry_id": "old-1", "content": "Old entry", "similarity": 0.96}
                ],
            }
            mock_get_store.return_value = mock_store

            input_data = server_module.RememberInput(
                content="New content",
                suggest_supersedes=True,
            )

            result = server_module.enyal_remember(input_data)

            assert result["success"] is True
            assert "supersedes_candidates" in result
            assert len(result["supersedes_candidates"]) == 1


class TestEnyalLink:
    """Tests for enyal_link tool."""

    def test_enyal_link_success(self, server_module) -> None:
        """Test successful link creation."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.link.return_value = "edge-123"
            mock_get_store.return_value = mock_store

            input_data = server_module.LinkInput(
                source_id="entry-1",
                target_id="entry-2",
                relation="relates_to",
                confidence=0.9,
                reason="They are related",
            )

            result = server_module.enyal_link(input_data)

            assert result["success"] is True
            assert result["edge_id"] == "edge-123"
            assert "relates_to" in result["message"]

    def test_enyal_link_failure(self, server_module) -> None:
        """Test link when entries don't exist."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.link.return_value = None
            mock_get_store.return_value = mock_store

            input_data = server_module.LinkInput(
                source_id="nonexistent-1",
                target_id="nonexistent-2",
                relation="relates_to",
            )

            result = server_module.enyal_link(input_data)

            assert result["success"] is False
            assert "error" in result

    def test_enyal_link_no_reason(self, server_module) -> None:
        """Test link creation without a reason."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.link.return_value = "edge-456"
            mock_get_store.return_value = mock_store

            input_data = server_module.LinkInput(
                source_id="entry-1",
                target_id="entry-2",
                relation="depends_on",
            )

            result = server_module.enyal_link(input_data)

            assert result["success"] is True
            mock_store.link.assert_called_once()
            # metadata should be empty dict when no reason
            call_kwargs = mock_store.link.call_args
            assert call_kwargs[1]["metadata"] == {}

    def test_enyal_link_value_error(self, server_module) -> None:
        """Test link with invalid relation type."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_get_store.return_value = MagicMock()

            input_data = server_module.LinkInput(
                source_id="entry-1",
                target_id="entry-2",
                relation="invalid_type",
            )

            result = server_module.enyal_link(input_data)

            assert result["success"] is False
            assert "Invalid relation type" in result["error"]

    def test_enyal_link_error(self, server_module) -> None:
        """Test link with unexpected error."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.link.side_effect = RuntimeError("DB error")
            mock_get_store.return_value = mock_store

            input_data = server_module.LinkInput(
                source_id="entry-1",
                target_id="entry-2",
                relation="relates_to",
            )

            result = server_module.enyal_link(input_data)

            assert result["success"] is False
            assert "DB error" in result["error"]


class TestEnyalUnlink:
    """Tests for enyal_unlink tool."""

    def test_enyal_unlink_success(self, server_module) -> None:
        """Test successful unlink."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.unlink.return_value = True
            mock_get_store.return_value = mock_store

            result = server_module.enyal_unlink("edge-123")

            assert result["success"] is True
            assert "edge-123" in result["message"]

    def test_enyal_unlink_not_found(self, server_module) -> None:
        """Test unlink when edge not found."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.unlink.return_value = False
            mock_get_store.return_value = mock_store

            result = server_module.enyal_unlink("nonexistent-edge")

            assert result["success"] is False
            assert "not found" in result["error"]

    def test_enyal_unlink_error(self, server_module) -> None:
        """Test unlink with error."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.unlink.side_effect = Exception("Unlink error")
            mock_get_store.return_value = mock_store

            result = server_module.enyal_unlink("edge-123")

            assert result["success"] is False
            assert "Unlink error" in result["error"]


class TestEnyalEdges:
    """Tests for enyal_edges tool."""

    def test_enyal_edges_success(self, server_module, sample_edge) -> None:
        """Test successful edges retrieval."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_edges.return_value = [sample_edge]
            mock_get_store.return_value = mock_store

            input_data = server_module.EdgesInput(
                entry_id="test-entry",
                direction="both",
            )

            result = server_module.enyal_edges(input_data)

            assert result["success"] is True
            assert result["count"] == 1
            assert len(result["edges"]) == 1
            assert result["edges"][0]["relation"] == "relates_to"

    def test_enyal_edges_with_filter(self, server_module) -> None:
        """Test edges with relation type filter."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_edges.return_value = []
            mock_get_store.return_value = mock_store

            input_data = server_module.EdgesInput(
                entry_id="test-entry",
                direction="outgoing",
                relation_type="supersedes",
            )

            result = server_module.enyal_edges(input_data)

            assert result["success"] is True
            assert result["count"] == 0

    def test_enyal_edges_value_error(self, server_module) -> None:
        """Test edges with invalid direction."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_edges.side_effect = ValueError("Invalid direction")
            mock_get_store.return_value = mock_store

            input_data = server_module.EdgesInput(
                entry_id="test-entry",
                direction="invalid",
            )

            result = server_module.enyal_edges(input_data)

            assert result["success"] is False
            assert "edges" in result
            assert result["edges"] == []

    def test_enyal_edges_error(self, server_module) -> None:
        """Test edges with unexpected error."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_edges.side_effect = RuntimeError("DB error")
            mock_get_store.return_value = mock_store

            input_data = server_module.EdgesInput(entry_id="test-entry")

            result = server_module.enyal_edges(input_data)

            assert result["success"] is False
            assert "edges" in result


class TestEnyalTraverse:
    """Tests for enyal_traverse tool."""

    def test_enyal_traverse_success(self, server_module, sample_entry) -> None:
        """Test successful graph traversal."""
        mock_search_result = ContextSearchResult(
            entry=sample_entry, distance=0.1, score=0.95
        )
        traverse_result = {
            "entry": sample_entry,
            "depth": 1,
            "path": [sample_entry.id],
            "edge_type": "relates_to",
            "confidence": 0.9,
        }

        with (
            patch.object(server_module, "get_store") as mock_get_store,
            patch.object(server_module, "get_retrieval") as mock_get_retrieval,
        ):
            mock_store = MagicMock()
            mock_store.traverse.return_value = [traverse_result]
            mock_get_store.return_value = mock_store

            mock_retrieval = MagicMock()
            mock_retrieval.search.return_value = [mock_search_result]
            mock_get_retrieval.return_value = mock_retrieval

            input_data = server_module.TraverseInput(
                start_query="test query",
                max_depth=2,
            )

            result = server_module.enyal_traverse(input_data)

            assert result["success"] is True
            assert "start_entry" in result
            assert result["count"] == 1

    def test_enyal_traverse_no_start(self, server_module) -> None:
        """Test traverse when no starting entry found."""
        with (
            patch.object(server_module, "get_store"),
            patch.object(server_module, "get_retrieval") as mock_get_retrieval,
        ):
            mock_retrieval = MagicMock()
            mock_retrieval.search.return_value = []
            mock_get_retrieval.return_value = mock_retrieval

            input_data = server_module.TraverseInput(start_query="nonexistent")

            result = server_module.enyal_traverse(input_data)

            assert result["success"] is False
            assert "No entry found" in result["error"]

    def test_enyal_traverse_with_relation_types(self, server_module, sample_entry) -> None:
        """Test traverse with relation type filter."""
        mock_search_result = ContextSearchResult(
            entry=sample_entry, distance=0.1, score=0.95
        )

        with (
            patch.object(server_module, "get_store") as mock_get_store,
            patch.object(server_module, "get_retrieval") as mock_get_retrieval,
        ):
            mock_store = MagicMock()
            mock_store.traverse.return_value = []
            mock_get_store.return_value = mock_store

            mock_retrieval = MagicMock()
            mock_retrieval.search.return_value = [mock_search_result]
            mock_get_retrieval.return_value = mock_retrieval

            input_data = server_module.TraverseInput(
                start_query="test",
                relation_types=["depends_on", "relates_to"],
            )

            result = server_module.enyal_traverse(input_data)

            assert result["success"] is True

    def test_enyal_traverse_value_error(self, server_module, sample_entry) -> None:
        """Test traverse with invalid relation type."""
        mock_search_result = ContextSearchResult(
            entry=sample_entry, distance=0.1, score=0.95
        )

        with (
            patch.object(server_module, "get_store"),
            patch.object(server_module, "get_retrieval") as mock_get_retrieval,
        ):
            mock_retrieval = MagicMock()
            mock_retrieval.search.return_value = [mock_search_result]
            mock_get_retrieval.return_value = mock_retrieval

            input_data = server_module.TraverseInput(
                start_query="test",
                relation_types=["invalid_type"],
            )

            result = server_module.enyal_traverse(input_data)

            assert result["success"] is False

    def test_enyal_traverse_error(self, server_module) -> None:
        """Test traverse with unexpected error."""
        with (
            patch.object(server_module, "get_store"),
            patch.object(server_module, "get_retrieval") as mock_get_retrieval,
        ):
            mock_retrieval = MagicMock()
            mock_retrieval.search.side_effect = RuntimeError("Search error")
            mock_get_retrieval.return_value = mock_retrieval

            input_data = server_module.TraverseInput(start_query="test")

            result = server_module.enyal_traverse(input_data)

            assert result["success"] is False


class TestEnyalImpact:
    """Tests for enyal_impact tool."""

    def test_enyal_impact_with_entry_id(self, server_module, sample_entry) -> None:
        """Test impact analysis using entry_id."""
        with (
            patch.object(server_module, "get_store") as mock_get_store,
            patch.object(server_module, "get_retrieval"),
        ):
            mock_store = MagicMock()
            mock_store.get.return_value = sample_entry
            mock_store.traverse.return_value = []
            mock_get_store.return_value = mock_store

            input_data = server_module.ImpactInput(entry_id="test-entry-id")

            result = server_module.enyal_impact(input_data)

            assert result["success"] is True
            assert "target" in result
            assert "impact" in result
            assert result["impact"]["direct_dependencies"] == 0
            assert result["impact"]["transitive_dependencies"] == 0
            assert result["impact"]["related_entries"] == 0

    def test_enyal_impact_with_query(self, server_module, sample_entry) -> None:
        """Test impact analysis using query."""
        mock_search_result = ContextSearchResult(
            entry=sample_entry, distance=0.1, score=0.95
        )

        with (
            patch.object(server_module, "get_store") as mock_get_store,
            patch.object(server_module, "get_retrieval") as mock_get_retrieval,
        ):
            mock_store = MagicMock()
            mock_store.traverse.return_value = []
            mock_get_store.return_value = mock_store

            mock_retrieval = MagicMock()
            mock_retrieval.search.return_value = [mock_search_result]
            mock_get_retrieval.return_value = mock_retrieval

            input_data = server_module.ImpactInput(query="test query")

            result = server_module.enyal_impact(input_data)

            assert result["success"] is True
            assert result["target"]["content"] == sample_entry.content

    def test_enyal_impact_with_dependencies(self, server_module, sample_entry) -> None:
        """Test impact with actual dependencies found."""
        dep_entry = ContextEntry(
            content="Dependent entry",
            content_type=ContextType.FACT,
            scope_level=ScopeLevel.PROJECT,
        )

        with (
            patch.object(server_module, "get_store") as mock_get_store,
            patch.object(server_module, "get_retrieval"),
        ):
            mock_store = MagicMock()
            mock_store.get.return_value = sample_entry
            # First traverse call: depends_on (direct dep at depth 1)
            # Second traverse call: relates_to (related at confidence 0.9)
            mock_store.traverse.side_effect = [
                [{"entry": dep_entry, "depth": 1, "edge_type": "depends_on", "confidence": 1.0}],
                [{"entry": dep_entry, "depth": 1, "edge_type": "relates_to", "confidence": 0.9}],
            ]
            mock_get_store.return_value = mock_store

            input_data = server_module.ImpactInput(entry_id="test-id")

            result = server_module.enyal_impact(input_data)

            assert result["success"] is True
            assert result["impact"]["direct_dependencies"] == 1
            assert result["impact"]["related_entries"] == 1

    def test_enyal_impact_entry_not_found(self, server_module) -> None:
        """Test impact when entry_id not found."""
        with (
            patch.object(server_module, "get_store") as mock_get_store,
            patch.object(server_module, "get_retrieval"),
        ):
            mock_store = MagicMock()
            mock_store.get.return_value = None
            mock_get_store.return_value = mock_store

            input_data = server_module.ImpactInput(entry_id="nonexistent-id")

            result = server_module.enyal_impact(input_data)

            assert result["success"] is False
            assert "not found" in result["error"]

    def test_enyal_impact_query_not_found(self, server_module) -> None:
        """Test impact when query returns no results."""
        with (
            patch.object(server_module, "get_store"),
            patch.object(server_module, "get_retrieval") as mock_get_retrieval,
        ):
            mock_retrieval = MagicMock()
            mock_retrieval.search.return_value = []
            mock_get_retrieval.return_value = mock_retrieval

            input_data = server_module.ImpactInput(query="nonexistent query")

            result = server_module.enyal_impact(input_data)

            assert result["success"] is False
            assert "No entry found" in result["error"]

    def test_enyal_impact_no_input(self, server_module) -> None:
        """Test impact with neither entry_id nor query."""
        with (
            patch.object(server_module, "get_store"),
            patch.object(server_module, "get_retrieval"),
        ):
            input_data = server_module.ImpactInput()

            result = server_module.enyal_impact(input_data)

            assert result["success"] is False
            assert "Provide either" in result["error"]

    def test_enyal_impact_error(self, server_module) -> None:
        """Test impact with unexpected error."""
        with (
            patch.object(server_module, "get_store") as mock_get_store,
            patch.object(server_module, "get_retrieval"),
        ):
            mock_store = MagicMock()
            mock_store.get.side_effect = RuntimeError("DB error")
            mock_get_store.return_value = mock_store

            input_data = server_module.ImpactInput(entry_id="test-id")

            result = server_module.enyal_impact(input_data)

            assert result["success"] is False


class TestEnyalHealth:
    """Tests for enyal_health tool."""

    def test_enyal_health_success(self, server_module) -> None:
        """Test successful health check."""
        health_data = {
            "total_entries": 50,
            "total_edges": 20,
            "superseded_entries": 2,
            "unresolved_conflicts": 0,
            "stale_entries": 5,
            "orphan_entries": 10,
            "low_confidence_entries": 3,
            "never_accessed_entries": 8,
            "health_score": 0.85,
        }

        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.health_check.return_value = health_data
            mock_get_store.return_value = mock_store

            result = server_module.enyal_health()

            assert result["success"] is True
            assert result["health"] == health_data
            assert "recommendations" in result

    def test_enyal_health_error(self, server_module) -> None:
        """Test health check with error."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.health_check.side_effect = Exception("Health error")
            mock_get_store.return_value = mock_store

            result = server_module.enyal_health()

            assert result["success"] is False
            assert "Health error" in result["error"]


class TestGetHealthRecommendations:
    """Tests for _get_health_recommendations helper."""

    def test_healthy_graph(self, server_module) -> None:
        """Test recommendations for healthy graph."""
        health = {
            "superseded_entries": 2,
            "unresolved_conflicts": 0,
            "stale_entries": 5,
            "orphan_entries": 5,
            "total_entries": 100,
            "health_score": 0.9,
        }

        result = server_module._get_health_recommendations(health)

        assert result == ["Graph health is good!"]

    def test_superseded_entries(self, server_module) -> None:
        """Test recommendation for many superseded entries."""
        health = {
            "superseded_entries": 15,
            "unresolved_conflicts": 0,
            "stale_entries": 5,
            "orphan_entries": 5,
            "total_entries": 100,
            "health_score": 0.8,
        }

        result = server_module._get_health_recommendations(health)

        assert any("superseded" in r for r in result)

    def test_unresolved_conflicts(self, server_module) -> None:
        """Test recommendation for conflicts."""
        health = {
            "superseded_entries": 0,
            "unresolved_conflicts": 3,
            "stale_entries": 5,
            "orphan_entries": 5,
            "total_entries": 100,
            "health_score": 0.8,
        }

        result = server_module._get_health_recommendations(health)

        assert any("conflicting" in r for r in result)

    def test_many_stale_entries(self, server_module) -> None:
        """Test recommendation for stale entries."""
        health = {
            "superseded_entries": 0,
            "unresolved_conflicts": 0,
            "stale_entries": 25,
            "orphan_entries": 5,
            "total_entries": 100,
            "health_score": 0.8,
        }

        result = server_module._get_health_recommendations(health)

        assert any("stale" in r for r in result)

    def test_many_orphans(self, server_module) -> None:
        """Test recommendation for many orphan entries."""
        health = {
            "superseded_entries": 0,
            "unresolved_conflicts": 0,
            "stale_entries": 5,
            "orphan_entries": 40,
            "total_entries": 100,
            "health_score": 0.8,
        }

        result = server_module._get_health_recommendations(health)

        assert any("linking" in r for r in result)

    def test_low_health_score(self, server_module) -> None:
        """Test recommendation for low health score."""
        health = {
            "superseded_entries": 0,
            "unresolved_conflicts": 0,
            "stale_entries": 5,
            "orphan_entries": 5,
            "total_entries": 100,
            "health_score": 0.5,
        }

        result = server_module._get_health_recommendations(health)

        assert any("maintenance" in r for r in result)


class TestEnyalReview:
    """Tests for enyal_review tool."""

    def test_enyal_review_all(self, server_module, sample_entry) -> None:
        """Test review with all categories."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_stale_entries.return_value = [sample_entry]
            mock_store.get_orphan_entries.return_value = [sample_entry]
            mock_store.get_conflicted_entries.return_value = [
                {"entry1": sample_entry, "entry2": sample_entry, "confidence": 0.9}
            ]
            mock_get_store.return_value = mock_store

            input_data = server_module.ReviewInput(category="all", limit=10)

            result = server_module.enyal_review(input_data)

            assert result["success"] is True
            assert "stale_entries" in result
            assert "orphan_entries" in result
            assert "conflicted_entries" in result

    def test_enyal_review_stale(self, server_module, sample_entry) -> None:
        """Test review stale category only."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_stale_entries.return_value = [sample_entry]
            mock_get_store.return_value = mock_store

            input_data = server_module.ReviewInput(category="stale")

            result = server_module.enyal_review(input_data)

            assert result["success"] is True
            assert "stale_entries" in result
            assert "orphan_entries" not in result

    def test_enyal_review_orphan(self, server_module, sample_entry) -> None:
        """Test review orphan category only."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_orphan_entries.return_value = []
            mock_get_store.return_value = mock_store

            input_data = server_module.ReviewInput(category="orphan")

            result = server_module.enyal_review(input_data)

            assert result["success"] is True
            assert "orphan_entries" in result
            assert "stale_entries" not in result

    def test_enyal_review_conflicts(self, server_module, sample_entry) -> None:
        """Test review conflicts category."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_conflicted_entries.return_value = []
            mock_get_store.return_value = mock_store

            input_data = server_module.ReviewInput(category="conflicts")

            result = server_module.enyal_review(input_data)

            assert result["success"] is True
            assert "conflicted_entries" in result

    def test_enyal_review_error(self, server_module) -> None:
        """Test review with error."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_stale_entries.side_effect = Exception("Review error")
            mock_get_store.return_value = mock_store

            input_data = server_module.ReviewInput(category="all")

            result = server_module.enyal_review(input_data)

            assert result["success"] is False
            assert "Review error" in result["error"]


class TestEnyalHistory:
    """Tests for enyal_history tool."""

    def test_enyal_history_success(self, server_module, sample_entry) -> None:
        """Test successful history retrieval."""
        history_records = [
            {
                "version_id": "v1",
                "version": 1,
                "content": "Original content",
                "content_type": "fact",
                "confidence": 0.9,
                "tags": [],
                "changed_at": "2024-01-01T00:00:00",
                "change_type": "created",
                "metadata": {},
            }
        ]

        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_history.return_value = history_records
            mock_store.get.return_value = sample_entry
            mock_get_store.return_value = mock_store

            input_data = server_module.HistoryInput(entry_id="test-entry-id")

            result = server_module.enyal_history(input_data)

            assert result["success"] is True
            assert result["entry_id"] == "test-entry-id"
            assert result["version_count"] == 1

    def test_enyal_history_not_found(self, server_module) -> None:
        """Test history when entry not found."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_history.return_value = []
            mock_store.get.return_value = None
            mock_get_store.return_value = mock_store

            input_data = server_module.HistoryInput(entry_id="nonexistent-id")

            result = server_module.enyal_history(input_data)

            assert result["success"] is False
            assert "not found" in result["error"]

    def test_enyal_history_error(self, server_module) -> None:
        """Test history with error."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_history.side_effect = Exception("History error")
            mock_get_store.return_value = mock_store

            input_data = server_module.HistoryInput(entry_id="test-id")

            result = server_module.enyal_history(input_data)

            assert result["success"] is False
            assert "History error" in result["error"]


class TestEnyalAnalytics:
    """Tests for enyal_analytics tool."""

    def test_enyal_analytics_success(self, server_module) -> None:
        """Test successful analytics retrieval."""
        analytics_data = {
            "period_days": 30,
            "events_by_type": [],
            "top_recalled": [],
        }

        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_analytics.return_value = analytics_data
            mock_get_store.return_value = mock_store

            input_data = server_module.AnalyticsInput(days=30)

            result = server_module.enyal_analytics(input_data)

            assert result["success"] is True
            assert result["analytics"] == analytics_data

    def test_enyal_analytics_with_filters(self, server_module) -> None:
        """Test analytics with entry_id and event_type filters."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_analytics.return_value = {"period_days": 7, "events_by_type": [], "top_recalled": []}
            mock_get_store.return_value = mock_store

            input_data = server_module.AnalyticsInput(
                entry_id="specific-entry",
                event_type="recall",
                days=7,
            )

            result = server_module.enyal_analytics(input_data)

            assert result["success"] is True
            mock_store.get_analytics.assert_called_once_with(
                entry_id="specific-entry",
                event_type="recall",
                days=7,
            )

    def test_enyal_analytics_error(self, server_module) -> None:
        """Test analytics with error."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_analytics.side_effect = Exception("Analytics error")
            mock_get_store.return_value = mock_store

            input_data = server_module.AnalyticsInput()

            result = server_module.enyal_analytics(input_data)

            assert result["success"] is False
            assert "Analytics error" in result["error"]


class TestVerifyStartupHealth:
    """Tests for _verify_startup_health function."""

    def test_verify_startup_health_success(self, server_module, sample_stats) -> None:
        """Test successful startup health check."""
        mock_store = MagicMock()
        mock_store.stats.return_value = sample_stats

        # Should not raise
        server_module._verify_startup_health(mock_store)

        mock_store.stats.assert_called_once()

    def test_verify_startup_health_failure(self, server_module) -> None:
        """Test startup health check failure."""
        mock_store = MagicMock()
        mock_store.stats.side_effect = RuntimeError("DB corrupted")

        with pytest.raises(RuntimeError, match="DB corrupted"):
            server_module._verify_startup_health(mock_store)


class TestCleanup:
    """Tests for _cleanup function."""

    def test_cleanup_with_store(self, server_module) -> None:
        """Test cleanup releases resources."""
        mock_store = MagicMock()
        mock_lock = MagicMock()
        server_module._store = mock_store
        server_module._retrieval = MagicMock()
        server_module._process_lock = mock_lock

        server_module._cleanup()

        mock_store.checkpoint_wal.assert_called_once()
        mock_store.close.assert_called_once()
        mock_lock.release.assert_called_once()
        assert server_module._store is None
        assert server_module._retrieval is None
        assert server_module._process_lock is None

    def test_cleanup_no_store(self, server_module) -> None:
        """Test cleanup when no store initialized."""
        server_module._store = None
        server_module._retrieval = None
        server_module._process_lock = None

        # Should not raise
        server_module._cleanup()

    def test_cleanup_checkpoint_error(self, server_module) -> None:
        """Test cleanup continues despite checkpoint error."""
        mock_store = MagicMock()
        mock_store.checkpoint_wal.side_effect = Exception("WAL error")
        server_module._store = mock_store
        server_module._retrieval = MagicMock()
        server_module._process_lock = None

        # Should not raise, should still call close
        server_module._cleanup()

        mock_store.close.assert_called_once()

    def test_cleanup_close_error(self, server_module) -> None:
        """Test cleanup continues despite close error."""
        mock_store = MagicMock()
        mock_store.close.side_effect = Exception("Close error")
        server_module._store = mock_store
        server_module._retrieval = MagicMock()
        server_module._process_lock = None

        # Should not raise
        server_module._cleanup()

        assert server_module._store is None


class TestInputModels:
    """Tests for additional input model validation."""

    def test_link_input_defaults(self, server_module) -> None:
        """Test LinkInput default values."""
        input_data = server_module.LinkInput(
            source_id="s1", target_id="t1", relation="relates_to"
        )
        assert input_data.confidence == 1.0
        assert input_data.reason is None

    def test_edges_input_defaults(self, server_module) -> None:
        """Test EdgesInput default values."""
        input_data = server_module.EdgesInput(entry_id="e1")
        assert input_data.direction == "both"
        assert input_data.relation_type is None

    def test_traverse_input_defaults(self, server_module) -> None:
        """Test TraverseInput default values."""
        input_data = server_module.TraverseInput(start_query="test")
        assert input_data.direction == "outgoing"
        assert input_data.max_depth == 2
        assert input_data.relation_types is None

    def test_impact_input_defaults(self, server_module) -> None:
        """Test ImpactInput default values."""
        input_data = server_module.ImpactInput()
        assert input_data.entry_id is None
        assert input_data.query is None
        assert input_data.max_depth == 3

    def test_review_input_defaults(self, server_module) -> None:
        """Test ReviewInput default values."""
        input_data = server_module.ReviewInput()
        assert input_data.category == "all"
        assert input_data.limit == 10

    def test_history_input(self, server_module) -> None:
        """Test HistoryInput creation."""
        input_data = server_module.HistoryInput(entry_id="e1", limit=5)
        assert input_data.entry_id == "e1"
        assert input_data.limit == 5

    def test_analytics_input_defaults(self, server_module) -> None:
        """Test AnalyticsInput default values."""
        input_data = server_module.AnalyticsInput()
        assert input_data.entry_id is None
        assert input_data.event_type is None
        assert input_data.days == 30

    def test_recall_by_scope_input(self, server_module) -> None:
        """Test RecallByScopeInput creation."""
        input_data = server_module.RecallByScopeInput(
            query="test", file_path="/path/to/file.py"
        )
        assert input_data.exclude_superseded is True
        assert input_data.flag_conflicts is True
        assert input_data.freshness_boost == 0.1
