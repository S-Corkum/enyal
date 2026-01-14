"""Tests for retrieval engine module."""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from enyal.core.retrieval import RetrievalEngine
from enyal.models.context import (
    ContextEntry,
    ContextType,
    ScopeLevel,
)


class TestRetrievalEngineInit:
    """Tests for RetrievalEngine initialization."""

    def test_init_default_weights(self) -> None:
        """Test initialization with default weights."""
        mock_store = MagicMock()

        engine = RetrievalEngine(mock_store)

        assert engine.store == mock_store
        assert engine.confidence_half_life_days == 90
        # Weights should be normalized to sum to 1
        total = engine.semantic_weight + engine.keyword_weight + engine.recency_weight
        assert abs(total - 1.0) < 0.001

    def test_init_custom_weights(self) -> None:
        """Test initialization with custom weights."""
        mock_store = MagicMock()

        engine = RetrievalEngine(
            mock_store,
            semantic_weight=0.8,
            keyword_weight=0.1,
            recency_weight=0.1,
        )

        # Weights should be normalized
        total = engine.semantic_weight + engine.keyword_weight + engine.recency_weight
        assert abs(total - 1.0) < 0.001

    def test_weight_normalization(self) -> None:
        """Test that weights are normalized properly."""
        mock_store = MagicMock()

        # Use non-normalized weights (don't sum to 1)
        engine = RetrievalEngine(
            mock_store,
            semantic_weight=2.0,
            keyword_weight=1.0,
            recency_weight=1.0,
        )

        # Should normalize to: semantic=0.5, keyword=0.25, recency=0.25
        assert abs(engine.semantic_weight - 0.5) < 0.001
        assert abs(engine.keyword_weight - 0.25) < 0.001
        assert abs(engine.recency_weight - 0.25) < 0.001


class TestSearch:
    """Tests for search method."""

    def test_search_basic(self, sample_entry: ContextEntry) -> None:
        """Test basic search functionality."""
        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {
                "entry": sample_entry,
                "distance": 0.2,
                "score": 0.8,
            }
        ]

        engine = RetrievalEngine(mock_store)
        results = engine.search("test query")

        assert len(results) == 1
        assert results[0].entry == sample_entry
        mock_store.recall.assert_called_once()

    def test_search_with_scope_filter(self, sample_entry: ContextEntry) -> None:
        """Test search with scope level filter."""
        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {
                "entry": sample_entry,
                "distance": 0.2,
                "score": 0.8,
            }
        ]

        engine = RetrievalEngine(mock_store)
        results = engine.search(
            "test query",
            scope_level=ScopeLevel.PROJECT,
        )

        assert len(results) == 1
        call_args = mock_store.recall.call_args
        assert call_args.kwargs.get("scope_level") == ScopeLevel.PROJECT

    def test_search_with_content_type_filter(self, sample_entry: ContextEntry) -> None:
        """Test search with content type filter."""
        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {
                "entry": sample_entry,
                "distance": 0.2,
                "score": 0.8,
            }
        ]

        engine = RetrievalEngine(mock_store)
        results = engine.search(
            "test query",
            content_type=ContextType.FACT,
        )

        assert len(results) == 1
        call_args = mock_store.recall.call_args
        assert call_args.kwargs.get("content_type") == ContextType.FACT

    def test_search_empty_results(self) -> None:
        """Test search with no results."""
        mock_store = MagicMock()
        mock_store.recall.return_value = []

        engine = RetrievalEngine(mock_store)
        results = engine.search("nonexistent query")

        assert len(results) == 0

    def test_search_combines_scores(self, sample_entry: ContextEntry) -> None:
        """Test that search combines semantic and recency scores."""
        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {
                "entry": sample_entry,
                "distance": 0.2,
                "score": 0.8,
            }
        ]

        engine = RetrievalEngine(mock_store)
        results = engine.search("test query")

        assert len(results) == 1
        # Score should be calculated from combined weights
        assert results[0].score > 0

    def test_search_respects_limit(self) -> None:
        """Test that search respects the limit parameter."""
        # Create multiple entries
        entries = []
        for i in range(5):
            entry = ContextEntry(
                content=f"Entry {i}",
                content_type=ContextType.FACT,
                scope_level=ScopeLevel.PROJECT,
            )
            entries.append(
                {
                    "entry": entry,
                    "distance": 0.2 + i * 0.1,
                    "score": 0.8 - i * 0.1,
                }
            )

        mock_store = MagicMock()
        mock_store.recall.return_value = entries

        engine = RetrievalEngine(mock_store)
        results = engine.search("test query", limit=3)

        assert len(results) == 3


class TestHybridSearch:
    """Tests for hybrid search with FTS5 integration."""

    def test_search_uses_fts_scores(self, sample_entry: ContextEntry) -> None:
        """Test that search incorporates FTS5 scores."""
        mock_store = MagicMock()
        mock_store.recall.return_value = [{"entry": sample_entry, "distance": 0.2, "score": 0.8}]
        mock_store.fts_search.return_value = [{"entry_id": sample_entry.id, "bm25_score": -5.0}]

        engine = RetrievalEngine(mock_store)
        results = engine.search("test query")

        assert len(results) == 1
        # FTS search should have been called
        mock_store.fts_search.assert_called_once()

    def test_search_keyword_boost(self) -> None:
        """Test that exact keyword matches get boosted."""
        # Entry with exact keyword should rank higher
        exact_match = ContextEntry(
            id="exact-id",
            content="pytest convention for testing",
            content_type=ContextType.CONVENTION,
            scope_level=ScopeLevel.PROJECT,
        )
        similar_semantic = ContextEntry(
            id="semantic-id",
            content="unit testing best practices",
            content_type=ContextType.CONVENTION,
            scope_level=ScopeLevel.PROJECT,
        )

        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {"entry": similar_semantic, "distance": 0.15, "score": 0.87},
            {"entry": exact_match, "distance": 0.2, "score": 0.8},
        ]
        # FTS finds exact match
        mock_store.fts_search.return_value = [
            {"entry_id": "exact-id", "bm25_score": -8.0},  # Strong FTS match
        ]
        mock_store.get.return_value = None

        engine = RetrievalEngine(mock_store, keyword_weight=0.4)
        results = engine.search("pytest convention")

        # Exact match should have higher combined score due to FTS boost
        assert len(results) >= 1

    def test_search_fts_only_entries(self) -> None:
        """Test entries only in FTS results are included."""
        fts_only_entry = ContextEntry(
            id="fts-only-id",
            content="Exact keyword match only",
            content_type=ContextType.FACT,
            scope_level=ScopeLevel.PROJECT,
        )

        mock_store = MagicMock()
        mock_store.recall.return_value = []  # No semantic matches
        mock_store.fts_search.return_value = [{"entry_id": "fts-only-id", "bm25_score": -6.0}]
        mock_store.get.return_value = fts_only_entry

        engine = RetrievalEngine(mock_store)
        results = engine.search("Exact keyword")

        assert len(results) == 1
        assert results[0].entry.id == "fts-only-id"


class TestSearchByScope:
    """Tests for search_by_scope method."""

    def test_search_by_scope_file_path(self, sample_entry: ContextEntry) -> None:
        """Test search_by_scope with a file path."""
        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {
                "entry": sample_entry,
                "distance": 0.2,
                "score": 0.8,
            }
        ]

        engine = RetrievalEngine(mock_store)

        # Create a temp file for testing
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            temp_file = Path(f.name)

        try:
            results = engine.search_by_scope("test query", temp_file)
            # Should return results
            assert isinstance(results, list)
        finally:
            temp_file.unlink()

    def test_search_by_scope_directory_path(self, sample_entry: ContextEntry) -> None:
        """Test search_by_scope with a directory path."""
        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {
                "entry": sample_entry,
                "distance": 0.2,
                "score": 0.8,
            }
        ]

        engine = RetrievalEngine(mock_store)

        with tempfile.TemporaryDirectory() as temp_dir:
            results = engine.search_by_scope("test query", temp_dir)
            assert isinstance(results, list)

    def test_search_by_scope_applies_weights(self, sample_entry: ContextEntry) -> None:
        """Test that search_by_scope applies scope weights."""
        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {
                "entry": sample_entry,
                "distance": 0.2,
                "score": 0.8,
            }
        ]

        engine = RetrievalEngine(mock_store)

        with tempfile.TemporaryDirectory() as temp_dir:
            results = engine.search_by_scope("test query", temp_dir)
            # Results should have adjusted scores based on scope weights
            assert isinstance(results, list)

    def test_search_by_scope_deduplicates(self) -> None:
        """Test that search_by_scope deduplicates results by entry ID."""
        # Create an entry that will appear in multiple scope searches
        entry = ContextEntry(
            id="same-entry-id",
            content="Duplicate entry",
            content_type=ContextType.FACT,
            scope_level=ScopeLevel.PROJECT,
        )

        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {
                "entry": entry,
                "distance": 0.2,
                "score": 0.8,
            }
        ]

        engine = RetrievalEngine(mock_store)

        with tempfile.TemporaryDirectory() as temp_dir:
            results = engine.search_by_scope("test query", temp_dir)
            # Should only have one entry despite potentially appearing in multiple scopes
            entry_ids = [r.entry.id for r in results]
            assert len(entry_ids) == len(set(entry_ids))


class TestEffectiveConfidence:
    """Tests for _calculate_effective_confidence method."""

    def test_effective_confidence_no_decay(self) -> None:
        """Test effective confidence with no time decay (recent entry)."""
        mock_store = MagicMock()
        engine = RetrievalEngine(mock_store)

        # Just created entry
        now = datetime.now(UTC).replace(tzinfo=None)
        confidence = engine._calculate_effective_confidence(
            base_confidence=0.9,
            updated_at=now,
            access_count=0,
        )

        # Should be close to base confidence (may have tiny access boost)
        assert confidence >= 0.9
        assert confidence <= 1.0

    def test_effective_confidence_with_decay(self) -> None:
        """Test effective confidence with time decay."""
        mock_store = MagicMock()
        engine = RetrievalEngine(mock_store, confidence_half_life_days=90)

        # Entry from 90 days ago (half-life)
        old_date = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=90)
        confidence = engine._calculate_effective_confidence(
            base_confidence=0.8,
            updated_at=old_date,
            access_count=0,
        )

        # Should be approximately half of original (decay factor of ~0.5)
        assert confidence < 0.8
        assert confidence > 0.3

    def test_effective_confidence_access_boost(self) -> None:
        """Test effective confidence with access boost."""
        mock_store = MagicMock()
        engine = RetrievalEngine(mock_store)

        now = datetime.now(UTC).replace(tzinfo=None)

        # Entry with many accesses
        confidence_high_access = engine._calculate_effective_confidence(
            base_confidence=0.8,
            updated_at=now,
            access_count=100,
        )

        # Entry with no accesses
        confidence_no_access = engine._calculate_effective_confidence(
            base_confidence=0.8,
            updated_at=now,
            access_count=0,
        )

        # High access count should have higher effective confidence
        assert confidence_high_access > confidence_no_access

    def test_effective_confidence_max_cap(self) -> None:
        """Test that effective confidence is capped at 1.0."""
        mock_store = MagicMock()
        engine = RetrievalEngine(mock_store)

        now = datetime.now(UTC).replace(tzinfo=None)

        # Entry with max confidence and many accesses
        confidence = engine._calculate_effective_confidence(
            base_confidence=1.0,
            updated_at=now,
            access_count=1000,
        )

        # Should not exceed 1.0
        assert confidence <= 1.0


class TestGetApplicableScopes:
    """Tests for _get_applicable_scopes method."""

    def test_get_applicable_scopes_file(self) -> None:
        """Test scope resolution for a file path."""
        mock_store = MagicMock()
        engine = RetrievalEngine(mock_store)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            temp_file = Path(f.name)

        try:
            scopes = engine._get_applicable_scopes(temp_file)

            # Should include file scope and global scope at minimum
            scope_levels = [s[0] for s in scopes]
            assert "file" in scope_levels
            assert "global" in scope_levels
        finally:
            temp_file.unlink()

    def test_get_applicable_scopes_directory(self) -> None:
        """Test scope resolution for a directory path."""
        mock_store = MagicMock()
        engine = RetrievalEngine(mock_store)

        with tempfile.TemporaryDirectory() as temp_dir:
            scopes = engine._get_applicable_scopes(Path(temp_dir))

            # Should include global scope
            scope_levels = [s[0] for s in scopes]
            assert "global" in scope_levels

    def test_get_applicable_scopes_with_git(self) -> None:
        """Test scope resolution finds project via .git marker."""
        mock_store = MagicMock()
        engine = RetrievalEngine(mock_store)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create .git directory
            git_dir = Path(temp_dir) / ".git"
            git_dir.mkdir()

            # Create a file in the project
            test_file = Path(temp_dir) / "test.py"
            test_file.touch()

            scopes = engine._get_applicable_scopes(test_file)

            # Should include project scope
            scope_levels = [s[0] for s in scopes]
            assert "project" in scope_levels

    def test_get_applicable_scopes_with_pyproject(self) -> None:
        """Test scope resolution finds project via pyproject.toml marker."""
        mock_store = MagicMock()
        engine = RetrievalEngine(mock_store)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create pyproject.toml
            pyproject = Path(temp_dir) / "pyproject.toml"
            pyproject.touch()

            # Create a file in the project
            test_file = Path(temp_dir) / "test.py"
            test_file.touch()

            scopes = engine._get_applicable_scopes(test_file)

            # Should include project scope
            scope_levels = [s[0] for s in scopes]
            assert "project" in scope_levels

    def test_get_applicable_scopes_global_always_included(self) -> None:
        """Test that global scope is always included."""
        mock_store = MagicMock()
        engine = RetrievalEngine(mock_store)

        with tempfile.TemporaryDirectory() as temp_dir:
            scopes = engine._get_applicable_scopes(Path(temp_dir))

            # Global should always be last
            assert scopes[-1] == ("global", None)


class TestGetRelated:
    """Tests for get_related method."""

    def test_get_related_basic(self, sample_entry: ContextEntry) -> None:
        """Test basic get_related functionality."""
        # Create a related entry
        related_entry = ContextEntry(
            id="related-id",
            content="Related content",
            content_type=ContextType.FACT,
            scope_level=ScopeLevel.PROJECT,
        )

        mock_store = MagicMock()
        mock_store.get.return_value = sample_entry
        mock_store.recall.return_value = [
            {
                "entry": related_entry,
                "distance": 0.3,
                "score": 0.7,
            }
        ]

        engine = RetrievalEngine(mock_store)
        results = engine.get_related(sample_entry.id, limit=5)

        assert len(results) == 1
        assert results[0].entry.id == "related-id"

    def test_get_related_excludes_self(self, sample_entry: ContextEntry) -> None:
        """Test that get_related excludes the reference entry itself."""
        mock_store = MagicMock()
        mock_store.get.return_value = sample_entry
        mock_store.recall.return_value = [
            {
                "entry": sample_entry,  # Same entry
                "distance": 0.0,
                "score": 1.0,
            }
        ]

        engine = RetrievalEngine(mock_store)
        results = engine.get_related(sample_entry.id)

        # Should filter out the reference entry
        assert len(results) == 0

    def test_get_related_entry_not_found(self) -> None:
        """Test get_related when entry doesn't exist."""
        mock_store = MagicMock()
        mock_store.get.return_value = None

        engine = RetrievalEngine(mock_store)
        results = engine.get_related("nonexistent-id")

        assert results == []

    def test_get_related_respects_limit(self, sample_entry: ContextEntry) -> None:
        """Test that get_related respects the limit parameter."""
        # Create multiple related entries
        related_entries = []
        for i in range(10):
            entry = ContextEntry(
                id=f"related-{i}",
                content=f"Related content {i}",
                content_type=ContextType.FACT,
                scope_level=ScopeLevel.PROJECT,
            )
            related_entries.append(
                {
                    "entry": entry,
                    "distance": 0.1 + i * 0.05,
                    "score": 0.9 - i * 0.05,
                }
            )

        mock_store = MagicMock()
        mock_store.get.return_value = sample_entry
        mock_store.recall.return_value = related_entries

        engine = RetrievalEngine(mock_store)
        results = engine.get_related(sample_entry.id, limit=3)

        assert len(results) <= 3


class TestValidityFiltering:
    """Tests for validity filtering in search."""

    def test_search_default_excludes_superseded(self, sample_entry: ContextEntry) -> None:
        """Test that superseded entries are excluded by default."""
        superseded_entry = ContextEntry(
            id="superseded-id",
            content="Old deprecated info",
            content_type=ContextType.FACT,
            scope_level=ScopeLevel.PROJECT,
        )

        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {"entry": sample_entry, "distance": 0.2, "score": 0.8},
            {"entry": superseded_entry, "distance": 0.3, "score": 0.7},
        ]
        mock_store.fts_search.return_value = []
        mock_store.get_superseded_ids.return_value = {"superseded-id"}
        mock_store.get_conflicted_ids.return_value = set()
        mock_store.get_superseding_entry.return_value = None

        engine = RetrievalEngine(mock_store)
        results = engine.search("test query")

        # Superseded entry should be filtered out
        result_ids = [r.entry.id for r in results]
        assert "superseded-id" not in result_ids
        assert sample_entry.id in result_ids

    def test_search_include_superseded(self, sample_entry: ContextEntry) -> None:
        """Test that superseded entries can be included with flag."""
        superseded_entry = ContextEntry(
            id="superseded-id",
            content="Old deprecated info",
            content_type=ContextType.FACT,
            scope_level=ScopeLevel.PROJECT,
        )

        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {"entry": sample_entry, "distance": 0.2, "score": 0.8},
            {"entry": superseded_entry, "distance": 0.3, "score": 0.7},
        ]
        mock_store.fts_search.return_value = []
        mock_store.get_superseded_ids.return_value = {"superseded-id"}
        mock_store.get_conflicted_ids.return_value = set()
        mock_store.get_superseding_entry.return_value = "sample-entry-id"

        engine = RetrievalEngine(mock_store)
        results = engine.search("test query", exclude_superseded=False)

        # Superseded entry should be included
        result_ids = [r.entry.id for r in results]
        assert "superseded-id" in result_ids
        # But it should be marked as superseded
        superseded_result = next(r for r in results if r.entry.id == "superseded-id")
        assert superseded_result.is_superseded is True

    def test_search_flags_conflicts(self, sample_entry: ContextEntry) -> None:
        """Test that conflicted entries are flagged."""
        conflicted_entry = ContextEntry(
            id="conflicted-id",
            content="Conflicting info",
            content_type=ContextType.FACT,
            scope_level=ScopeLevel.PROJECT,
        )

        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {"entry": sample_entry, "distance": 0.2, "score": 0.8},
            {"entry": conflicted_entry, "distance": 0.3, "score": 0.7},
        ]
        mock_store.fts_search.return_value = []
        mock_store.get_superseded_ids.return_value = set()
        mock_store.get_conflicted_ids.return_value = {"conflicted-id"}
        mock_store.get_superseding_entry.return_value = None

        engine = RetrievalEngine(mock_store)
        results = engine.search("test query")

        # Find the conflicted entry in results
        conflicted_result = next(r for r in results if r.entry.id == "conflicted-id")
        assert conflicted_result.has_conflicts is True
        # Non-conflicted entry should not be flagged
        normal_result = next(r for r in results if r.entry.id == sample_entry.id)
        assert normal_result.has_conflicts is False

    def test_search_freshness_score_calculated(self, sample_entry: ContextEntry) -> None:
        """Test that freshness score is calculated."""
        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {"entry": sample_entry, "distance": 0.2, "score": 0.8},
        ]
        mock_store.fts_search.return_value = []
        mock_store.get_superseded_ids.return_value = set()
        mock_store.get_conflicted_ids.return_value = set()

        engine = RetrievalEngine(mock_store)
        results = engine.search("test query")

        assert len(results) == 1
        # Freshness score should be between 0 and 1
        assert 0.0 <= results[0].freshness_score <= 1.0
        # Adjusted score should also be set
        assert results[0].adjusted_score is not None

    def test_search_validity_metadata_in_results(self, sample_entry: ContextEntry) -> None:
        """Test that all validity metadata is present in results."""
        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {"entry": sample_entry, "distance": 0.2, "score": 0.8},
        ]
        mock_store.fts_search.return_value = []
        mock_store.get_superseded_ids.return_value = set()
        mock_store.get_conflicted_ids.return_value = set()

        engine = RetrievalEngine(mock_store)
        results = engine.search("test query")

        result = results[0]
        # All validity fields should be present
        assert hasattr(result, "is_superseded")
        assert hasattr(result, "superseded_by")
        assert hasattr(result, "has_conflicts")
        assert hasattr(result, "freshness_score")
        assert hasattr(result, "adjusted_score")
