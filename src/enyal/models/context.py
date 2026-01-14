"""Data models for context entries."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Return current UTC time as a naive datetime (for backward compatibility)."""
    return datetime.now(UTC).replace(tzinfo=None)


class ContextType(StrEnum):
    """Types of context that can be stored."""

    FACT = "fact"
    PREFERENCE = "preference"
    DECISION = "decision"
    CONVENTION = "convention"
    PATTERN = "pattern"


class ScopeLevel(StrEnum):
    """Hierarchical scope levels from most to least specific."""

    FILE = "file"
    PROJECT = "project"
    WORKSPACE = "workspace"
    GLOBAL = "global"


class SourceType(StrEnum):
    """Source of context information."""

    CONVERSATION = "conversation"
    FILE = "file"
    COMMIT = "commit"
    MANUAL = "manual"


class EdgeType(StrEnum):
    """Types of relationships between context entries."""

    RELATES_TO = "relates_to"
    SUPERSEDES = "supersedes"
    DEPENDS_ON = "depends_on"
    CONFLICTS_WITH = "conflicts_with"


class ContextEntry(BaseModel):
    """A single context entry in Enyal's memory."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(description="The actual knowledge/context")
    content_type: ContextType = Field(default=ContextType.FACT)
    scope_level: ScopeLevel = Field(default=ScopeLevel.PROJECT)
    scope_path: str | None = Field(default=None, description="Path for non-global scopes")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    accessed_at: datetime | None = Field(default=None)
    access_count: int = Field(default=0, ge=0)
    source_type: SourceType | None = Field(default=None)
    source_ref: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_deprecated: bool = Field(default=False)

    model_config = {"frozen": False, "extra": "forbid"}


class ContextEdge(BaseModel):
    """A relationship between two context entries."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str = Field(description="ID of the source entry")
    target_id: str = Field(description="ID of the target entry")
    edge_type: EdgeType = Field(description="Type of relationship")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False, "extra": "forbid"}


class ContextSearchResult(BaseModel):
    """A search result with relevance information and validity metadata."""

    entry: ContextEntry
    distance: float = Field(description="Vector distance (lower is more similar)")
    score: float = Field(description="Combined relevance score (higher is better)")
    # Validity metadata
    is_superseded: bool = Field(default=False, description="Entry has been superseded by another")
    superseded_by: str | None = Field(default=None, description="ID of superseding entry")
    has_conflicts: bool = Field(default=False, description="Entry has unresolved conflicts")
    freshness_score: float = Field(default=1.0, description="Time-based freshness (0-1)")
    adjusted_score: float | None = Field(default=None, description="Score after validity adjustments")

    model_config = {"frozen": True}


class ContextStats(BaseModel):
    """Statistics about the context store."""

    total_entries: int
    active_entries: int
    deprecated_entries: int
    entries_by_type: dict[str, int]
    entries_by_scope: dict[str, int]
    avg_confidence: float
    storage_size_bytes: int
    oldest_entry: datetime | None
    newest_entry: datetime | None
    # Graph statistics
    total_edges: int = 0
    edges_by_type: dict[str, int] = Field(default_factory=dict)
    connected_entries: int = 0

    model_config = {"frozen": True}
