"""Shared pytest fixtures for Enyal tests."""

import tempfile
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from enyal.models.context import (
    ContextEntry,
    ContextSearchResult,
    ContextStats,
    ContextType,
    ScopeLevel,
    SourceType,
)


@pytest.fixture
def temp_db() -> Generator[Path]:
    """Create a temporary database path for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def mock_embedding() -> NDArray[np.float32]:
    """Return a mock 384-dimensional embedding."""
    return np.random.rand(384).astype(np.float32)


@pytest.fixture
def mock_embedding_engine(mock_embedding: NDArray[np.float32]) -> Generator[MagicMock]:
    """Mock the EmbeddingEngine to avoid model loading in tests."""
    with patch("enyal.embeddings.engine.EmbeddingEngine") as mock_class:
        mock_instance = mock_class
        mock_instance.embed.return_value = mock_embedding
        mock_instance.embed_batch.return_value = np.zeros((0, 384), dtype=np.float32)
        mock_instance.embedding_dimension.return_value = 384
        mock_instance.is_loaded.return_value = False
        mock_instance._model = None
        yield mock_instance


@pytest.fixture
def sample_entry() -> ContextEntry:
    """Create a sample ContextEntry for tests."""
    return ContextEntry(
        content="Test content for unit tests",
        content_type=ContextType.FACT,
        scope_level=ScopeLevel.PROJECT,
        scope_path="/test/project",
        tags=["test", "sample"],
        metadata={"source": "unit_test"},
        confidence=0.9,
    )


@pytest.fixture
def sample_entry_with_source() -> ContextEntry:
    """Create a sample ContextEntry with source information."""
    return ContextEntry(
        content="Test content with source",
        content_type=ContextType.DECISION,
        scope_level=ScopeLevel.FILE,
        scope_path="/test/project/file.py",
        source_type=SourceType.CONVERSATION,
        source_ref="session-123",
        tags=["decision", "architecture"],
        confidence=0.85,
    )


@pytest.fixture
def sample_search_result(sample_entry: ContextEntry) -> ContextSearchResult:
    """Create a sample ContextSearchResult for tests."""
    return ContextSearchResult(
        entry=sample_entry,
        distance=0.25,
        score=0.8,
    )


@pytest.fixture
def sample_stats() -> ContextStats:
    """Create sample ContextStats for tests."""
    return ContextStats(
        total_entries=100,
        active_entries=90,
        deprecated_entries=10,
        entries_by_type={"fact": 50, "decision": 30, "preference": 10, "convention": 10},
        entries_by_scope={"project": 60, "global": 30, "file": 10},
        avg_confidence=0.85,
        storage_size_bytes=1024 * 1024,  # 1MB
        oldest_entry=datetime(2024, 1, 1),
        newest_entry=datetime(2024, 12, 1),
    )


@pytest.fixture
def mock_store(sample_entry: ContextEntry, sample_stats: ContextStats) -> MagicMock:
    """Create a mock ContextStore for unit tests."""
    store = MagicMock()
    store.remember.return_value = "test-entry-id-123"
    store.get.return_value = sample_entry
    store.forget.return_value = True
    store.update.return_value = True
    store.recall.return_value = [
        {
            "entry": sample_entry,
            "distance": 0.25,
            "score": 0.8,
        }
    ]
    store.stats.return_value = sample_stats
    return store


@pytest.fixture
def mock_retrieval_engine(sample_search_result: ContextSearchResult) -> MagicMock:
    """Create a mock RetrievalEngine for unit tests."""
    retrieval = MagicMock()
    retrieval.search.return_value = [sample_search_result]
    retrieval.search_by_scope.return_value = [sample_search_result]
    retrieval.get_related.return_value = []
    return retrieval


@pytest.fixture
def cli_args_remember() -> dict[str, Any]:
    """Common CLI arguments for remember command."""
    return {
        "content": "Test content to remember",
        "type": "fact",
        "scope": "project",
        "scope_path": None,
        "tags": None,
        "db": None,
        "json": False,
    }


@pytest.fixture
def cli_args_recall() -> dict[str, Any]:
    """Common CLI arguments for recall command."""
    return {
        "query": "test query",
        "limit": 10,
        "type": None,
        "scope": None,
        "scope_path": None,
        "min_confidence": 0.3,
        "db": None,
        "json": False,
    }


@pytest.fixture
def cli_args_forget() -> dict[str, Any]:
    """Common CLI arguments for forget command."""
    return {
        "entry_id": "test-entry-id",
        "hard": False,
        "db": None,
        "json": False,
    }


@pytest.fixture
def cli_args_get() -> dict[str, Any]:
    """Common CLI arguments for get command."""
    return {
        "entry_id": "test-entry-id",
        "db": None,
        "json": False,
    }


@pytest.fixture
def cli_args_stats() -> dict[str, Any]:
    """Common CLI arguments for stats command."""
    return {
        "db": None,
        "json": False,
    }


@pytest.fixture
def duplicate_entry() -> ContextEntry:
    """Create an entry similar to sample_entry for dedup testing."""
    return ContextEntry(
        id="duplicate-entry-id",
        content="Test content for unit tests - slightly different",  # Similar to sample
        content_type=ContextType.FACT,
        scope_level=ScopeLevel.PROJECT,
        scope_path="/test/project",
        tags=["test", "duplicate"],
        confidence=0.85,
    )


@pytest.fixture
def mock_fts_results() -> list[dict[str, Any]]:
    """Mock FTS search results."""
    return [
        {"entry_id": "fts-match-1", "bm25_score": -5.5},
        {"entry_id": "fts-match-2", "bm25_score": -3.2},
    ]
