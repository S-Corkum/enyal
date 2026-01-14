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
        with patch.object(server_module, "ContextStore") as mock_store_class:
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
        ):
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            server_module._store = None

            server_module.get_store()

            mock_store_class.assert_called_once_with("/custom/db/path.db")

    def test_get_store_caches_instance(self, server_module) -> None:
        """Test that get_store returns cached instance on subsequent calls."""
        with patch.object(server_module, "ContextStore") as mock_store_class:
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
