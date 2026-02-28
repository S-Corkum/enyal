"""Typed response models for Enyal MCP tools.

FastMCP uses these to auto-generate outputSchema and structuredContent
in MCP tool responses, giving LLMs predictable response structures.
"""

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
    metadata: dict = Field(default_factory=dict)


class ConflictCandidate(BaseModel):
    """Potential conflict entry."""

    entry_id: str
    content: str
    similarity: float


# === Tool responses ===


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
    """Response from enyal_recall and enyal_recall_by_scope."""

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


class StatsResponse(BaseModel):
    """Response from enyal_stats."""

    success: bool
    stats: dict


class GetResponse(BaseModel):
    """Response from enyal_get."""

    success: bool
    entry: dict
    edges: dict


class LinkResponse(BaseModel):
    """Response from enyal_link."""

    success: bool
    edge_id: str
    message: str


class UnlinkResponse(BaseModel):
    """Response from enyal_unlink."""

    success: bool
    message: str


class EdgesResponse(BaseModel):
    """Response from enyal_edges."""

    success: bool
    count: int
    edges: list[EdgeBrief]


class TraverseResponse(BaseModel):
    """Response from enyal_traverse."""

    success: bool
    start_entry: EntryBrief | None = None
    count: int
    results: list[dict]


class ImpactResponse(BaseModel):
    """Response from enyal_impact."""

    success: bool
    target: EntryBrief | None = None
    impact: dict
    direct_dependencies: list[dict]
    transitive_dependencies: list[dict]
    related: list[dict]


class HealthResponse(BaseModel):
    """Response from enyal_health."""

    success: bool
    health: dict
    recommendations: list[str]


class ReviewResponse(BaseModel):
    """Response from enyal_review."""

    success: bool
    stale_entries: list[dict] = Field(default_factory=list)
    orphan_entries: list[dict] = Field(default_factory=list)
    conflicted_entries: list[dict] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    """Response from enyal_history."""

    success: bool
    entry_id: str
    current_content: str
    version_count: int
    history: list[dict]


class AnalyticsResponse(BaseModel):
    """Response from enyal_analytics."""

    success: bool
    analytics: dict


class RestoreResponse(BaseModel):
    """Response from enyal_restore."""

    success: bool
    message: str


class SearchTagsResponse(BaseModel):
    """Response from enyal_search_tags."""

    success: bool
    count: int
    results: list[EntryBrief]


class ExportResponse(BaseModel):
    """Response from enyal_export."""

    success: bool
    count: int
    data: dict


class ImportResponse(BaseModel):
    """Response from enyal_import."""

    success: bool
    entries_imported: int
    edges_imported: int
    entries_skipped: int
    message: str
