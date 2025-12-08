"""Tests for data models."""

from datetime import datetime

import pytest

from enyal.models.context import (
    ContextEntry,
    ContextSearchResult,
    ContextStats,
    ContextType,
    ScopeLevel,
    SourceType,
)


class TestContextEntry:
    """Tests for ContextEntry model."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        entry = ContextEntry(content="Test content")

        assert entry.content == "Test content"
        assert entry.content_type == ContextType.FACT
        assert entry.scope_level == ScopeLevel.PROJECT
        assert entry.confidence == 1.0
        assert entry.access_count == 0
        assert entry.is_deprecated is False
        assert entry.tags == []
        assert entry.metadata == {}
        assert entry.id is not None

    def test_all_content_types(self) -> None:
        """Test all content types are valid."""
        for ct in ContextType:
            entry = ContextEntry(content="Test", content_type=ct)
            assert entry.content_type == ct

    def test_all_scope_levels(self) -> None:
        """Test all scope levels are valid."""
        for sl in ScopeLevel:
            entry = ContextEntry(content="Test", scope_level=sl)
            assert entry.scope_level == sl

    def test_confidence_bounds(self) -> None:
        """Test confidence must be between 0 and 1."""
        entry = ContextEntry(content="Test", confidence=0.5)
        assert entry.confidence == 0.5

        with pytest.raises(ValueError):
            ContextEntry(content="Test", confidence=-0.1)

        with pytest.raises(ValueError):
            ContextEntry(content="Test", confidence=1.1)

    def test_with_source(self) -> None:
        """Test entry with source information."""
        entry = ContextEntry(
            content="Test content",
            source_type=SourceType.CONVERSATION,
            source_ref="session-123",
        )

        assert entry.source_type == SourceType.CONVERSATION
        assert entry.source_ref == "session-123"


class TestContextSearchResult:
    """Tests for ContextSearchResult model."""

    def test_creation(self) -> None:
        """Test search result creation."""
        entry = ContextEntry(content="Test")
        result = ContextSearchResult(
            entry=entry,
            distance=0.5,
            score=0.8,
        )

        assert result.entry == entry
        assert result.distance == 0.5
        assert result.score == 0.8

    def test_frozen(self) -> None:
        """Test that search results are immutable."""
        entry = ContextEntry(content="Test")
        result = ContextSearchResult(entry=entry, distance=0.5, score=0.8)

        with pytest.raises(TypeError):  # Pydantic frozen models raise TypeError
            result.score = 0.9  # type: ignore[misc]


class TestContextStats:
    """Tests for ContextStats model."""

    def test_creation(self) -> None:
        """Test stats creation."""
        stats = ContextStats(
            total_entries=100,
            active_entries=90,
            deprecated_entries=10,
            entries_by_type={"fact": 50, "decision": 40, "preference": 10},
            entries_by_scope={"project": 60, "global": 40},
            avg_confidence=0.85,
            storage_size_bytes=1024 * 1024,
            oldest_entry=datetime(2024, 1, 1),
            newest_entry=datetime(2024, 12, 1),
        )

        assert stats.total_entries == 100
        assert stats.active_entries == 90
        assert stats.deprecated_entries == 10
        assert stats.avg_confidence == 0.85
        assert stats.storage_size_bytes == 1024 * 1024


class TestEnums:
    """Tests for enum types."""

    def test_context_type_values(self) -> None:
        """Test ContextType string values."""
        assert ContextType.FACT.value == "fact"
        assert ContextType.PREFERENCE.value == "preference"
        assert ContextType.DECISION.value == "decision"
        assert ContextType.CONVENTION.value == "convention"
        assert ContextType.PATTERN.value == "pattern"

    def test_scope_level_values(self) -> None:
        """Test ScopeLevel string values."""
        assert ScopeLevel.FILE.value == "file"
        assert ScopeLevel.PROJECT.value == "project"
        assert ScopeLevel.WORKSPACE.value == "workspace"
        assert ScopeLevel.GLOBAL.value == "global"

    def test_source_type_values(self) -> None:
        """Test SourceType string values."""
        assert SourceType.CONVERSATION.value == "conversation"
        assert SourceType.FILE.value == "file"
        assert SourceType.COMMIT.value == "commit"
        assert SourceType.MANUAL.value == "manual"
