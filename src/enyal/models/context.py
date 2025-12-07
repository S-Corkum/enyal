"""Data models for context entries."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


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


class ContextEntry(BaseModel):
    """A single context entry in Enyal's memory."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(description="The actual knowledge/context")
    content_type: ContextType = Field(default=ContextType.FACT)
    scope_level: ScopeLevel = Field(default=ScopeLevel.PROJECT)
    scope_path: str | None = Field(default=None, description="Path for non-global scopes")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_at: datetime | None = Field(default=None)
    access_count: int = Field(default=0, ge=0)
    source_type: SourceType | None = Field(default=None)
    source_ref: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_deprecated: bool = Field(default=False)

    model_config = {"frozen": False, "extra": "forbid"}


class ContextSearchResult(BaseModel):
    """A search result with relevance information."""

    entry: ContextEntry
    distance: float = Field(description="Vector distance (lower is more similar)")
    score: float = Field(description="Combined relevance score (higher is better)")

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

    model_config = {"frozen": True}
