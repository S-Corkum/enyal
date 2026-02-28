"""Tests for MCP server module."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastmcp.exceptions import ToolError
from pydantic import ValidationError

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
        """Test RememberInput default values — flipped defaults for 0.8.0."""
        input_data = server_module.RememberInput(content="Test content")

        assert input_data.content == "Test content"
        assert input_data.content_type == "fact"
        assert input_data.scope == "project"
        assert input_data.scope_path is None
        assert input_data.source is None
        assert input_data.tags == []
        # Flipped defaults (was False, now True)
        assert input_data.check_duplicate is True
        assert input_data.auto_link is True
        assert input_data.detect_conflicts is True

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

    def test_recall_input_with_query(self, server_module) -> None:
        """Test RecallInput with query."""
        input_data = server_module.RecallInput(query="test query")

        assert input_data.query == "test query"
        assert input_data.limit == 10
        assert input_data.scope is None
        assert input_data.file_path is None
        assert input_data.tags is None

    def test_recall_input_with_tags(self, server_module) -> None:
        """Test RecallInput with tags only."""
        input_data = server_module.RecallInput(tags=["python", "testing"])

        assert input_data.query is None
        assert input_data.tags == ["python", "testing"]
        assert input_data.match_all is False

    def test_recall_input_with_query_and_file_path(self, server_module) -> None:
        """Test RecallInput with query and file_path for scope-aware search."""
        input_data = server_module.RecallInput(
            query="conventions",
            file_path="/path/to/file.py",
        )
        assert input_data.query == "conventions"
        assert input_data.file_path == "/path/to/file.py"

    def test_recall_input_requires_query_or_tags(self, server_module) -> None:
        """Test RecallInput validation: must provide query or tags."""
        with pytest.raises(ValidationError, match="Must provide"):
            server_module.RecallInput()

    def test_recall_input_validation(self, server_module) -> None:
        """Test RecallInput with filters."""
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


class TestForgetInput:
    """Tests for ForgetInput model."""

    def test_forget_input_defaults(self, server_module) -> None:
        """Test ForgetInput default values."""
        input_data = server_module.ForgetInput(entry_id="test-id")
        assert input_data.hard_delete is False
        assert input_data.restore is False

    def test_forget_input_restore_and_hard_delete_mutual_exclusion(self, server_module) -> None:
        """Test that restore and hard_delete are mutually exclusive."""
        with pytest.raises(ValidationError, match="Cannot use both"):
            server_module.ForgetInput(entry_id="test-id", restore=True, hard_delete=True)


class TestGetInput:
    """Tests for GetInput model."""

    def test_get_input_defaults(self, server_module) -> None:
        """Test GetInput default values."""
        input_data = server_module.GetInput(entry_id="test-id")
        assert input_data.entry_id == "test-id"
        assert input_data.include_history is False
        assert input_data.history_limit == 10

    def test_get_input_with_history(self, server_module) -> None:
        """Test GetInput with history enabled."""
        input_data = server_module.GetInput(
            entry_id="test-id", include_history=True, history_limit=5
        )
        assert input_data.include_history is True
        assert input_data.history_limit == 5


class TestLinkInput:
    """Tests for LinkInput model."""

    def test_link_input_create_defaults(self, server_module) -> None:
        """Test LinkInput create action defaults."""
        input_data = server_module.LinkInput(
            source_id="s1", target_id="t1", relation="relates_to"
        )
        assert input_data.action == "create"
        assert input_data.confidence == 1.0
        assert input_data.reason is None

    def test_link_input_remove(self, server_module) -> None:
        """Test LinkInput remove action."""
        input_data = server_module.LinkInput(action="remove", edge_id="edge-123")
        assert input_data.action == "remove"
        assert input_data.edge_id == "edge-123"

    def test_link_input_create_requires_fields(self, server_module) -> None:
        """Test that create action requires source_id, target_id, relation."""
        with pytest.raises(ValidationError, match="create action requires"):
            server_module.LinkInput(action="create", source_id="s1")

    def test_link_input_remove_requires_edge_id(self, server_module) -> None:
        """Test that remove action requires edge_id."""
        with pytest.raises(ValidationError, match="remove action requires"):
            server_module.LinkInput(action="remove")


class TestTraverseInput:
    """Tests for TraverseInput model."""

    def test_traverse_input_with_start_query(self, server_module) -> None:
        """Test TraverseInput with start_query."""
        input_data = server_module.TraverseInput(start_query="test")
        assert input_data.direction == "outgoing"
        assert input_data.max_depth == 2
        assert input_data.relation_types is None

    def test_traverse_input_with_entry_id(self, server_module) -> None:
        """Test TraverseInput with entry_id for edge lookup."""
        input_data = server_module.TraverseInput(entry_id="test-id", direction="both")
        assert input_data.entry_id == "test-id"
        assert input_data.direction == "both"

    def test_traverse_input_requires_entry_id_or_start_query(self, server_module) -> None:
        """Test validation: must provide entry_id or start_query."""
        with pytest.raises(ValidationError, match="Must provide"):
            server_module.TraverseInput()

    def test_traverse_input_both_direction_requires_entry_id(self, server_module) -> None:
        """Test validation: direction='both' only valid with entry_id."""
        with pytest.raises(ValidationError, match="direction='both'"):
            server_module.TraverseInput(start_query="test", direction="both")


class TestStatusInput:
    """Tests for StatusInput model."""

    def test_status_input_defaults(self, server_module) -> None:
        """Test StatusInput default values."""
        input_data = server_module.StatusInput()
        assert input_data.view == "summary"
        assert input_data.category == "all"
        assert input_data.limit == 10
        assert input_data.days == 30

    def test_status_input_analytics(self, server_module) -> None:
        """Test StatusInput for analytics view."""
        input_data = server_module.StatusInput(
            view="analytics", entry_id="e1", event_type="recall", days=7
        )
        assert input_data.view == "analytics"
        assert input_data.entry_id == "e1"
        assert input_data.event_type == "recall"
        assert input_data.days == 7


class TestTransferInput:
    """Tests for TransferInput model."""

    def test_transfer_input_export(self, server_module) -> None:
        """Test TransferInput export direction."""
        input_data = server_module.TransferInput(direction="export")
        assert input_data.direction == "export"
        assert input_data.scope is None

    def test_transfer_input_import_requires_data(self, server_module) -> None:
        """Test that import direction requires data."""
        with pytest.raises(ValidationError, match="import direction requires"):
            server_module.TransferInput(direction="import")

    def test_transfer_input_import_with_data(self, server_module) -> None:
        """Test TransferInput import with data."""
        input_data = server_module.TransferInput(
            direction="import", data={"entries": [], "edges": []}
        )
        assert input_data.direction == "import"
        assert input_data.data is not None


class TestImpactInput:
    """Tests for ImpactInput model."""

    def test_impact_input_defaults(self, server_module) -> None:
        """Test ImpactInput default values."""
        input_data = server_module.ImpactInput()
        assert input_data.entry_id is None
        assert input_data.query is None
        assert input_data.max_depth == 3


class TestEnyalRemember:
    """Tests for enyal_remember tool."""

    def test_enyal_remember_success(self, server_module) -> None:
        """Test successful remember operation."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.remember.return_value = {
                "entry_id": "new-entry-id-123",
                "action": "created",
                "duplicate_of": None,
                "similarity": None,
            }
            mock_get_store.return_value = mock_store

            input_data = server_module.RememberInput(
                content="Test knowledge to store",
                content_type="fact",
                scope="project",
            )

            result = server_module.enyal_remember(input_data)

            assert result.success is True
            assert result.entry_id == "new-entry-id-123"
            assert hasattr(result, "message")

    def test_enyal_remember_with_all_options(self, server_module) -> None:
        """Test remember with all options — verifies flipped defaults."""
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
                content="Complex entry",
                content_type="decision",
                scope="file",
                scope_path="/path/to/file.py",
                source="session-abc",
                tags=["important", "architecture"],
            )

            result = server_module.enyal_remember(input_data)

            assert result.success is True
            mock_store.remember.assert_called_once_with(
                content="Complex entry",
                content_type=ContextType.DECISION,
                scope_level=ScopeLevel.FILE,
                scope_path="/path/to/file.py",
                source_type="conversation",
                source_ref="session-abc",
                tags=["important", "architecture"],
                check_duplicate=True,  # Flipped default
                duplicate_threshold=0.85,
                on_duplicate="reject",
                # Graph parameters
                auto_link=True,  # Flipped default
                auto_link_threshold=0.85,
                relates_to=None,
                supersedes=None,
                depends_on=None,
                # Conflict/supersedes detection
                detect_conflicts=True,  # Flipped default
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

            with pytest.raises(Exception, match="Database error"):
                server_module.enyal_remember(input_data)


class TestEnyalRecall:
    """Tests for enyal_recall tool — covers all 3 modes."""

    def test_enyal_recall_semantic_search(self, server_module, sample_entry: ContextEntry) -> None:
        """Test standard semantic search (query only)."""
        mock_result = ContextSearchResult(entry=sample_entry, distance=0.25, score=0.8)

        with patch.object(server_module, "get_retrieval") as mock_get_retrieval:
            mock_retrieval = MagicMock()
            mock_retrieval.search.return_value = [mock_result]
            mock_get_retrieval.return_value = mock_retrieval

            with patch.object(server_module, "get_store"):
                input_data = server_module.RecallInput(query="test query")

                result = server_module.enyal_recall(input_data)

                assert result.success is True
                assert result.count == 1
                assert len(result.results) == 1
                assert result.results[0].content == "Test content for unit tests"

    def test_enyal_recall_scope_aware(self, server_module, sample_entry: ContextEntry) -> None:
        """Test scope-aware search (query + file_path)."""
        mock_result = ContextSearchResult(entry=sample_entry, distance=0.25, score=0.8)

        with patch.object(server_module, "get_retrieval") as mock_get_retrieval:
            mock_retrieval = MagicMock()
            mock_retrieval.search_by_scope.return_value = [mock_result]
            mock_get_retrieval.return_value = mock_retrieval

            with patch.object(server_module, "get_store"):
                input_data = server_module.RecallInput(
                    query="test query",
                    file_path="/path/to/file.py",
                )

                result = server_module.enyal_recall(input_data)

                assert result.success is True
                assert result.count == 1
                mock_retrieval.search_by_scope.assert_called_once()

    def test_enyal_recall_tags_only(self, server_module, sample_entry: ContextEntry) -> None:
        """Test tag-only search."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.search_by_tags.return_value = [sample_entry]
            mock_get_store.return_value = mock_store

            with patch.object(server_module, "get_retrieval"):
                input_data = server_module.RecallInput(tags=["test", "sample"])

                result = server_module.enyal_recall(input_data)

                assert result.success is True
                assert result.count == 1
                mock_store.search_by_tags.assert_called_once_with(
                    tags=["test", "sample"], match_all=False, limit=10
                )

    def test_enyal_recall_query_and_tags_post_filter(self, server_module, sample_entry: ContextEntry) -> None:
        """Test query + tags: semantic search with tag post-filtering."""
        mock_result = ContextSearchResult(entry=sample_entry, distance=0.25, score=0.8)

        with patch.object(server_module, "get_retrieval") as mock_get_retrieval:
            mock_retrieval = MagicMock()
            mock_retrieval.search.return_value = [mock_result]
            mock_get_retrieval.return_value = mock_retrieval

            with patch.object(server_module, "get_store"):
                # sample_entry has tags ["test", "sample"]
                input_data = server_module.RecallInput(
                    query="test query",
                    tags=["test"],
                )

                result = server_module.enyal_recall(input_data)

                assert result.success is True
                assert result.count == 1  # Kept because entry has "test" tag

    def test_enyal_recall_query_and_tags_filter_out(self, server_module, sample_entry: ContextEntry) -> None:
        """Test query + tags: entry filtered out when tags don't match."""
        mock_result = ContextSearchResult(entry=sample_entry, distance=0.25, score=0.8)

        with patch.object(server_module, "get_retrieval") as mock_get_retrieval:
            mock_retrieval = MagicMock()
            mock_retrieval.search.return_value = [mock_result]
            mock_get_retrieval.return_value = mock_retrieval

            with patch.object(server_module, "get_store"):
                input_data = server_module.RecallInput(
                    query="test query",
                    tags=["nonexistent-tag"],
                )

                result = server_module.enyal_recall(input_data)

                assert result.success is True
                assert result.count == 0  # Filtered out

    def test_enyal_recall_with_filters(self, server_module, sample_entry: ContextEntry) -> None:
        """Test recall with filters."""
        mock_result = ContextSearchResult(entry=sample_entry, distance=0.25, score=0.8)

        with patch.object(server_module, "get_retrieval") as mock_get_retrieval:
            mock_retrieval = MagicMock()
            mock_retrieval.search.return_value = [mock_result]
            mock_get_retrieval.return_value = mock_retrieval

            with patch.object(server_module, "get_store"):
                input_data = server_module.RecallInput(
                    query="test query",
                    limit=5,
                    scope="project",
                    content_type="fact",
                    min_confidence=0.5,
                )

                result = server_module.enyal_recall(input_data)

                assert result.success is True
                mock_retrieval.search.assert_called_once_with(
                    query="test query",
                    limit=5,
                    scope_level=ScopeLevel.PROJECT,
                    scope_path=None,
                    content_type=ContextType.FACT,
                    min_confidence=0.5,
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

            with patch.object(server_module, "get_store"):
                input_data = server_module.RecallInput(query="nonexistent query")

                result = server_module.enyal_recall(input_data)

                assert result.success is True
                assert result.count == 0
                assert result.results == []

    def test_enyal_recall_error(self, server_module) -> None:
        """Test recall operation with error."""
        with patch.object(server_module, "get_retrieval") as mock_get_retrieval:
            mock_retrieval = MagicMock()
            mock_retrieval.search.side_effect = Exception("Search error")
            mock_get_retrieval.return_value = mock_retrieval

            with patch.object(server_module, "get_store"):
                input_data = server_module.RecallInput(query="test query")

                with pytest.raises(Exception, match="Search error"):
                    server_module.enyal_recall(input_data)


class TestEnyalGet:
    """Tests for enyal_get tool."""

    def test_enyal_get_success(self, server_module, sample_entry: ContextEntry) -> None:
        """Test successful get operation."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get.return_value = sample_entry
            mock_store.get_edges.return_value = []
            mock_get_store.return_value = mock_store

            input_data = server_module.GetInput(entry_id="test-entry-id")
            result = server_module.enyal_get(input_data)

            assert result.success is True
            assert result.entry is not None
            assert result.entry["content"] == "Test content for unit tests"
            assert result.entry["type"] == "fact"
            assert result.history is None

    def test_enyal_get_with_history(self, server_module, sample_entry: ContextEntry) -> None:
        """Test get with include_history=True."""
        history_records = [
            {"version": 1, "content": "Original", "change_type": "created"}
        ]

        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get.return_value = sample_entry
            mock_store.get_edges.return_value = []
            mock_store.get_history.return_value = history_records
            mock_get_store.return_value = mock_store

            input_data = server_module.GetInput(
                entry_id="test-entry-id", include_history=True
            )
            result = server_module.enyal_get(input_data)

            assert result.success is True
            assert result.history == history_records
            assert result.version_count == 1
            mock_store.get_history.assert_called_once_with("test-entry-id", limit=10)

    def test_enyal_get_with_source(
        self, server_module, sample_entry_with_source: ContextEntry
    ) -> None:
        """Test get with entry that has source information."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get.return_value = sample_entry_with_source
            mock_store.get_edges.return_value = []
            mock_get_store.return_value = mock_store

            input_data = server_module.GetInput(entry_id="test-entry-id")
            result = server_module.enyal_get(input_data)

            assert result.success is True
            assert result.entry["source_type"] == "conversation"
            assert result.entry["source_ref"] == "session-123"

    def test_enyal_get_not_found(self, server_module) -> None:
        """Test get when entry not found."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get.return_value = None
            mock_get_store.return_value = mock_store

            input_data = server_module.GetInput(entry_id="nonexistent-id")
            with pytest.raises(ToolError, match="not found"):
                server_module.enyal_get(input_data)

    def test_enyal_get_error(self, server_module) -> None:
        """Test get operation with error."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get.side_effect = Exception("Database error")
            mock_get_store.return_value = mock_store

            input_data = server_module.GetInput(entry_id="test-id")
            with pytest.raises(Exception, match="Database error"):
                server_module.enyal_get(input_data)


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

            assert result.success is True
            assert "deprecated" in result.message
            mock_store.forget.assert_called_once_with("test-entry-id", hard_delete=False)

    def test_enyal_forget_hard_delete(self, server_module) -> None:
        """Test forget with hard delete."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.forget.return_value = True
            mock_get_store.return_value = mock_store

            input_data = server_module.ForgetInput(entry_id="test-entry-id", hard_delete=True)

            result = server_module.enyal_forget(input_data)

            assert result.success is True
            assert "permanently deleted" in result.message
            mock_store.forget.assert_called_once_with("test-entry-id", hard_delete=True)

    def test_enyal_forget_restore(self, server_module) -> None:
        """Test forget with restore=True."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.restore.return_value = True
            mock_get_store.return_value = mock_store

            input_data = server_module.ForgetInput(entry_id="test-entry-id", restore=True)

            result = server_module.enyal_forget(input_data)

            assert result.success is True
            assert "restored" in result.message
            mock_store.restore.assert_called_once_with("test-entry-id")

    def test_enyal_forget_restore_not_found(self, server_module) -> None:
        """Test restore when entry not found or not deprecated."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.restore.return_value = False
            mock_get_store.return_value = mock_store

            input_data = server_module.ForgetInput(entry_id="test-id", restore=True)

            with pytest.raises(ToolError, match="not found or not deprecated"):
                server_module.enyal_forget(input_data)

    def test_enyal_forget_not_found(self, server_module) -> None:
        """Test forget when entry not found."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.forget.return_value = False
            mock_get_store.return_value = mock_store

            input_data = server_module.ForgetInput(entry_id="nonexistent-id")

            with pytest.raises(ToolError, match="not found"):
                server_module.enyal_forget(input_data)

    def test_enyal_forget_error(self, server_module) -> None:
        """Test forget operation with error."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.forget.side_effect = Exception("Database error")
            mock_get_store.return_value = mock_store

            input_data = server_module.ForgetInput(entry_id="test-id")

            with pytest.raises(Exception, match="Database error"):
                server_module.enyal_forget(input_data)


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

            assert result.success is True
            assert "updated" in result.message
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

            with pytest.raises(ToolError, match="not found"):
                server_module.enyal_update(input_data)

    def test_enyal_update_error(self, server_module) -> None:
        """Test update operation with error."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.update.side_effect = Exception("Update error")
            mock_get_store.return_value = mock_store

            input_data = server_module.UpdateInput(entry_id="test-id", content="New content")

            with pytest.raises(Exception, match="Update error"):
                server_module.enyal_update(input_data)


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

            assert result.success is True
            assert result.action == "existing"
            assert result.duplicate_of == "existing-id"
            assert "similarity" in result.message

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

            assert result.success is True
            assert result.action == "created"
            assert result.entry_id == "new-entry-id"


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

            assert result.success is True
            assert result.action == "merged"
            assert "similarity" in result.message

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

            assert result.success is True
            assert result.action == "created"
            assert len(result.potential_conflicts) == 1

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

            assert result.success is True
            assert len(result.supersedes_candidates) == 1


class TestEnyalLink:
    """Tests for enyal_link tool."""

    def test_enyal_link_create_success(self, server_module) -> None:
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

            assert result.success is True
            assert result.edge_id == "edge-123"
            assert "relates_to" in result.message

    def test_enyal_link_create_failure(self, server_module) -> None:
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

            with pytest.raises(ToolError):
                server_module.enyal_link(input_data)

    def test_enyal_link_remove_success(self, server_module) -> None:
        """Test successful link removal."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.unlink.return_value = True
            mock_get_store.return_value = mock_store

            input_data = server_module.LinkInput(action="remove", edge_id="edge-123")

            result = server_module.enyal_link(input_data)

            assert result.success is True
            assert "edge-123" in result.message

    def test_enyal_link_remove_not_found(self, server_module) -> None:
        """Test link remove when edge not found."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.unlink.return_value = False
            mock_get_store.return_value = mock_store

            input_data = server_module.LinkInput(action="remove", edge_id="nonexistent")

            with pytest.raises(ToolError, match="not found"):
                server_module.enyal_link(input_data)

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

            assert result.success is True
            mock_store.link.assert_called_once()
            call_kwargs = mock_store.link.call_args
            assert call_kwargs[1]["metadata"] == {}

    def test_enyal_link_value_error(self, server_module) -> None:
        """Test link with invalid relation type raises validation error."""
        with pytest.raises(ValidationError):
            server_module.LinkInput(
                source_id="entry-1",
                target_id="entry-2",
                relation="invalid_type",
            )

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

            with pytest.raises(RuntimeError, match="DB error"):
                server_module.enyal_link(input_data)


class TestEnyalTraverse:
    """Tests for enyal_traverse tool — covers traversal and edge lookup modes."""

    def test_enyal_traverse_success(self, server_module, sample_entry) -> None:
        """Test successful graph traversal via start_query."""
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

            assert result.success is True
            assert result.start_entry is not None
            assert result.count == 1
            assert result.edges is None  # Not in traversal mode

    def test_enyal_traverse_edge_lookup(self, server_module, sample_edge) -> None:
        """Test direct edge lookup via entry_id."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_edges.return_value = [sample_edge]
            mock_get_store.return_value = mock_store

            input_data = server_module.TraverseInput(
                entry_id="test-entry",
                direction="both",
            )

            result = server_module.enyal_traverse(input_data)

            assert result.success is True
            assert result.count == 1
            assert result.edges is not None
            assert len(result.edges) == 1
            assert result.edges[0].relation == "relates_to"
            assert result.results == []  # No traversal results

    def test_enyal_traverse_edge_lookup_with_filter(self, server_module) -> None:
        """Test edge lookup with relation type filter."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_edges.return_value = []
            mock_get_store.return_value = mock_store

            input_data = server_module.TraverseInput(
                entry_id="test-entry",
                direction="outgoing",
                relation_type="supersedes",
            )

            result = server_module.enyal_traverse(input_data)

            assert result.success is True
            assert result.count == 0

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

            with pytest.raises(ToolError, match="No entry found"):
                server_module.enyal_traverse(input_data)

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

            assert result.success is True

    def test_enyal_traverse_value_error(self, server_module, sample_entry) -> None:
        """Test traverse with invalid relation type raises validation error."""
        with pytest.raises(ValidationError):
            server_module.TraverseInput(
                start_query="test",
                relation_types=["invalid_type"],
            )

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

            with pytest.raises(RuntimeError, match="Search error"):
                server_module.enyal_traverse(input_data)


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

            assert result.success is True
            assert result.target is not None
            assert result.impact is not None
            assert result.impact["direct_dependencies"] == 0
            assert result.impact["transitive_dependencies"] == 0
            assert result.impact["related_entries"] == 0

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

            assert result.success is True
            assert result.target.content == sample_entry.content

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
            mock_store.traverse.side_effect = [
                [{"entry": dep_entry, "depth": 1, "edge_type": "depends_on", "confidence": 1.0}],
                [{"entry": dep_entry, "depth": 1, "edge_type": "relates_to", "confidence": 0.9}],
            ]
            mock_get_store.return_value = mock_store

            input_data = server_module.ImpactInput(entry_id="test-id")

            result = server_module.enyal_impact(input_data)

            assert result.success is True
            assert result.impact["direct_dependencies"] == 1
            assert result.impact["related_entries"] == 1

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

            with pytest.raises(ToolError, match="not found"):
                server_module.enyal_impact(input_data)

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

            with pytest.raises(ToolError, match="No entry found"):
                server_module.enyal_impact(input_data)

    def test_enyal_impact_no_input(self, server_module) -> None:
        """Test impact with neither entry_id nor query."""
        with (
            patch.object(server_module, "get_store"),
            patch.object(server_module, "get_retrieval"),
        ):
            input_data = server_module.ImpactInput()

            with pytest.raises(ToolError, match="Provide either"):
                server_module.enyal_impact(input_data)

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

            with pytest.raises(RuntimeError, match="DB error"):
                server_module.enyal_impact(input_data)


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


class TestEnyalStatus:
    """Tests for enyal_status tool — covers all 4 views."""

    def test_enyal_status_summary(self, server_module, sample_stats: ContextStats) -> None:
        """Test summary view."""
        health_data = {
            "total_entries": 50,
            "superseded_entries": 2,
            "unresolved_conflicts": 0,
            "stale_entries": 5,
            "orphan_entries": 10,
            "health_score": 0.85,
        }

        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.stats.return_value = sample_stats
            mock_store.health_check.return_value = health_data
            mock_get_store.return_value = mock_store

            input_data = server_module.StatusInput(view="summary")
            result = server_module.enyal_status(input_data)

            assert result.success is True
            assert result.view == "summary"
            assert result.stats is not None
            assert result.stats["total_entries"] == 100
            assert result.health is not None
            assert result.recommendations is not None

    def test_enyal_status_health(self, server_module) -> None:
        """Test health view."""
        health_data = {
            "total_entries": 50,
            "superseded_entries": 2,
            "unresolved_conflicts": 0,
            "stale_entries": 5,
            "orphan_entries": 10,
            "health_score": 0.85,
        }

        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.health_check.return_value = health_data
            mock_get_store.return_value = mock_store

            input_data = server_module.StatusInput(view="health")
            result = server_module.enyal_status(input_data)

            assert result.success is True
            assert result.view == "health"
            assert result.health == health_data
            assert result.recommendations is not None

    def test_enyal_status_review_all(self, server_module, sample_entry) -> None:
        """Test review view with all categories."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_stale_entries.return_value = [sample_entry]
            mock_store.get_orphan_entries.return_value = [sample_entry]
            mock_store.get_conflicted_entries.return_value = [
                {"entry1": sample_entry, "entry2": sample_entry, "confidence": 0.9}
            ]
            mock_get_store.return_value = mock_store

            input_data = server_module.StatusInput(view="review", category="all", limit=10)
            result = server_module.enyal_status(input_data)

            assert result.success is True
            assert result.view == "review"
            assert len(result.stale_entries) > 0
            assert len(result.orphan_entries) > 0
            assert len(result.conflicted_entries) > 0

    def test_enyal_status_review_stale(self, server_module, sample_entry) -> None:
        """Test review view with stale category only."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_stale_entries.return_value = [sample_entry]
            mock_get_store.return_value = mock_store

            input_data = server_module.StatusInput(view="review", category="stale")
            result = server_module.enyal_status(input_data)

            assert result.success is True
            assert len(result.stale_entries) > 0
            assert result.orphan_entries == []

    def test_enyal_status_analytics(self, server_module) -> None:
        """Test analytics view."""
        analytics_data = {
            "period_days": 30,
            "events_by_type": [],
            "top_recalled": [],
        }

        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_analytics.return_value = analytics_data
            mock_get_store.return_value = mock_store

            input_data = server_module.StatusInput(view="analytics", days=30)
            result = server_module.enyal_status(input_data)

            assert result.success is True
            assert result.view == "analytics"
            assert result.analytics == analytics_data

    def test_enyal_status_analytics_with_filters(self, server_module) -> None:
        """Test analytics view with filters."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_analytics.return_value = {"period_days": 7}
            mock_get_store.return_value = mock_store

            input_data = server_module.StatusInput(
                view="analytics", entry_id="specific-entry", event_type="recall", days=7
            )
            result = server_module.enyal_status(input_data)

            assert result.success is True
            mock_store.get_analytics.assert_called_once_with(
                entry_id="specific-entry",
                event_type="recall",
                days=7,
            )

    def test_enyal_status_error(self, server_module) -> None:
        """Test status with error."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.stats.side_effect = Exception("Stats error")
            mock_get_store.return_value = mock_store

            input_data = server_module.StatusInput(view="summary")

            with pytest.raises(Exception, match="Stats error"):
                server_module.enyal_status(input_data)


class TestEnyalTransfer:
    """Tests for enyal_transfer tool."""

    def test_enyal_transfer_export(self, server_module) -> None:
        """Test export direction."""
        export_data = {"entries": [{"id": "e1"}], "edges": []}

        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.export_entries.return_value = export_data
            mock_get_store.return_value = mock_store

            input_data = server_module.TransferInput(direction="export")
            result = server_module.enyal_transfer(input_data)

            assert result.success is True
            assert result.direction == "export"
            assert result.count == 1
            assert result.data == export_data

    def test_enyal_transfer_export_with_filters(self, server_module) -> None:
        """Test export with scope filters."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.export_entries.return_value = {"entries": [], "edges": []}
            mock_get_store.return_value = mock_store

            input_data = server_module.TransferInput(
                direction="export", scope="project", include_deprecated=True
            )
            result = server_module.enyal_transfer(input_data)

            assert result.success is True
            mock_store.export_entries.assert_called_once_with(
                scope_level=ScopeLevel.PROJECT,
                scope_path=None,
                include_deprecated=True,
            )

    def test_enyal_transfer_import(self, server_module) -> None:
        """Test import direction."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.import_entries.return_value = {
                "entries_imported": 5,
                "edges_imported": 3,
                "entries_skipped": 1,
            }
            mock_get_store.return_value = mock_store

            import_data = {"entries": [{"id": "e1"}], "edges": []}
            input_data = server_module.TransferInput(
                direction="import", data=import_data
            )
            result = server_module.enyal_transfer(input_data)

            assert result.success is True
            assert result.direction == "import"
            assert result.entries_imported == 5
            assert result.edges_imported == 3
            assert result.entries_skipped == 1

    def test_enyal_transfer_error(self, server_module) -> None:
        """Test transfer with error."""
        with patch.object(server_module, "get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.export_entries.side_effect = Exception("Export error")
            mock_get_store.return_value = mock_store

            input_data = server_module.TransferInput(direction="export")

            with pytest.raises(Exception, match="Export error"):
                server_module.enyal_transfer(input_data)


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
