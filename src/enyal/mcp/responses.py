"""Typed response models for Enyal MCP tools.

FastMCP uses these to auto-generate outputSchema and structuredContent
in MCP tool responses, giving LLMs predictable response structures.
"""

from typing import Any

from pydantic import BaseModel, Field

# === Shared building blocks ===


class EntryBrief(BaseModel):
    """Minimal entry representation for list views."""

    id: str
    content: str
    type: str
    scope: str
    scope_path: str | None = None
    confidence: float
    tags: list[str]
    created_at: str
    updated_at: str


class EntrySearchResult(EntryBrief):
    """Entry with search/validity metadata."""

    score: float
    is_superseded: bool = False
    superseded_by: str | None = None
    has_conflicts: bool = False
    freshness_score: float
    adjusted_score: float | None = None


class EdgeBrief(BaseModel):
    """Edge representation."""

    id: str
    source_id: str
    target_id: str
    relation: str
    confidence: float
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConflictCandidate(BaseModel):
    """Potential conflict entry."""

    entry_id: str
    content: str
    similarity: float


# === Tool responses (10 tools = 10 response models) ===


class RememberResponse(BaseModel):
    """Response from enyal_remember."""

    success: bool
    entry_id: str
    action: str  # "created", "existing", "merged"
    message: str
    duplicate_of: str | None = None
    similarity: float | None = None
    potential_conflicts: list[ConflictCandidate] = Field(default_factory=list)
    supersedes_candidates: list[ConflictCandidate] = Field(default_factory=list)


class RecallResponse(BaseModel):
    """Response from enyal_recall."""

    success: bool
    count: int
    results: list[EntrySearchResult]


class ForgetResponse(BaseModel):
    """Response from enyal_forget."""

    success: bool
    message: str


class UpdateResponse(BaseModel):
    """Response from enyal_update."""

    success: bool
    message: str


class GetResponse(BaseModel):
    """Response from enyal_get."""

    success: bool
    entry: dict[str, Any]
    edges: dict[str, Any]
    history: list[dict[str, Any]] | None = None
    version_count: int | None = None


class LinkResponse(BaseModel):
    """Response from enyal_link."""

    success: bool
    edge_id: str | None = None
    message: str


class TraverseResponse(BaseModel):
    """Response from enyal_traverse."""

    success: bool
    start_entry: EntryBrief | None = None
    count: int
    results: list[dict[str, Any]]
    edges: list[EdgeBrief] | None = None


class ImpactResponse(BaseModel):
    """Response from enyal_impact."""

    success: bool
    target: EntryBrief | None = None
    impact: dict[str, Any]
    direct_dependencies: list[dict[str, Any]]
    transitive_dependencies: list[dict[str, Any]]
    related: list[dict[str, Any]]


class StatusResponse(BaseModel):
    """Response from enyal_status. Consolidates health/stats/review/analytics."""

    success: bool
    view: str
    stats: dict[str, Any] | None = None
    health: dict[str, Any] | None = None
    recommendations: list[str] | None = None
    stale_entries: list[dict[str, Any]] = Field(default_factory=list)
    orphan_entries: list[dict[str, Any]] = Field(default_factory=list)
    conflicted_entries: list[dict[str, Any]] = Field(default_factory=list)
    analytics: dict[str, Any] | None = None


class TransferResponse(BaseModel):
    """Response from enyal_transfer. Consolidates export/import."""

    success: bool
    direction: str
    count: int | None = None
    data: dict[str, Any] | None = None
    entries_imported: int | None = None
    edges_imported: int | None = None
    entries_skipped: int | None = None
    message: str
