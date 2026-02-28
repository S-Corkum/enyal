"""MCP server implementation for Enyal."""

import atexit
import logging
import logging.handlers
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, model_validator

from enyal.core.process_lock import ProcessLock
from enyal.core.retrieval import RetrievalEngine
from enyal.core.store import ContextStore
from enyal.mcp.responses import (
    ConflictCandidate,
    EdgeBrief,
    EntryBrief,
    EntrySearchResult,
    ForgetResponse,
    GetResponse,
    ImpactResponse,
    LinkResponse,
    RecallResponse,
    RememberResponse,
    StatusResponse,
    TransferResponse,
    TraverseResponse,
    UpdateResponse,
)
from enyal.models.context import ContextType, EdgeType, ScopeLevel

logger = logging.getLogger(__name__)

# Global store instance (initialized lazily, eagerly via lifespan when possible)
_store: ContextStore | None = None
_retrieval: RetrievalEngine | None = None
_process_lock: ProcessLock | None = None


def _create_store() -> ContextStore:
    """Create a new ContextStore instance from environment configuration."""
    from enyal.embeddings.engine import EmbeddingEngine
    from enyal.embeddings.models import ModelConfig

    db_path = os.environ.get("ENYAL_DB_PATH", "~/.enyal/context.db")
    config = ModelConfig.from_env()
    engine = EmbeddingEngine(config)
    store = ContextStore(db_path, engine=engine)
    logger.info(f"Initialized context store at: {db_path} (model: {config.name})")
    return store


def _verify_startup_health(store: ContextStore) -> None:
    """Verify the store is functional on startup.

    Runs a minimal health check to ensure the database is accessible,
    sqlite-vec is loaded, and the store can serve queries.
    """
    try:
        stats = store.stats()
        logger.info(
            f"Startup health check passed: {stats.active_entries} entries, "
            f"{stats.total_edges} edges"
        )
    except Exception:
        logger.exception("Startup health check failed")
        raise


def _cleanup() -> None:
    """Cleanup handler for graceful shutdown."""
    global _store, _retrieval, _process_lock
    if _store is not None:
        try:
            _store.checkpoint_wal()
        except Exception:
            logger.warning("Failed to checkpoint WAL on shutdown", exc_info=True)
        try:
            _store.close()
        except Exception:
            logger.warning("Failed to close store on shutdown", exc_info=True)
        _store = None
        _retrieval = None
    if _process_lock is not None:
        _process_lock.release()
        _process_lock = None


@asynccontextmanager
async def app_lifespan(_server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Manage server lifecycle: initialization and cleanup.

    Startup:
        - Acquire process lock to prevent duplicate instances
        - Initialize store and retrieval engine eagerly
        - Run startup health check

    Shutdown:
        - Checkpoint WAL for clean database state
        - Close database connections
        - Release process lock
    """
    global _store, _retrieval, _process_lock

    # 1. Acquire process lock
    db_path = Path(os.environ.get("ENYAL_DB_PATH", "~/.enyal/context.db")).expanduser()
    _process_lock = ProcessLock(db_path)
    if not _process_lock.acquire():
        logger.warning(
            "Could not acquire process lock. Another Enyal instance may be running. "
            "Proceeding anyway, but concurrent access may cause issues."
        )

    # 2. Initialize store and retrieval eagerly
    try:
        _store = _create_store()
        _retrieval = RetrievalEngine(_store)
        _verify_startup_health(_store)
    except Exception:
        logger.exception("Failed to initialize Enyal server during lifespan startup")
        # Don't hard-fail - tools will return individual errors

    # 3. Register atexit as backup cleanup (in case lifespan shutdown doesn't run)
    atexit.register(_cleanup)

    yield {}

    # 4. Cleanup (unregister atexit first to avoid double cleanup)
    atexit.unregister(_cleanup)
    _cleanup()
    logger.info("Enyal server shutdown complete")


# Initialize MCP server with lifespan management
mcp = FastMCP(
    name="enyal",
    lifespan=app_lifespan,
)


def get_store() -> ContextStore:
    """Get or create the context store instance.

    Prefers the instance initialized by the lifespan. Falls back to
    lazy initialization if lifespan hasn't run yet (e.g., during testing).
    """
    global _store
    if _store is None:
        _store = _create_store()
    return _store


def _truncate_content(content: str, max_length: int | None) -> str:
    """Truncate content to max_length, appending '...' if truncated."""
    if max_length is None or len(content) <= max_length:
        return content
    return content[: max_length - 3] + "..."


def get_retrieval() -> RetrievalEngine:
    """Get or create the retrieval engine instance."""
    global _retrieval
    if _retrieval is None:
        _retrieval = RetrievalEngine(get_store())
    return _retrieval


# =============================================================================
# Input Models (10 tools)
# =============================================================================


class RememberInput(BaseModel):
    """Input for the remember tool."""

    content: str = Field(description="The context/knowledge to store")
    content_type: Literal["fact", "preference", "decision", "convention", "pattern"] = Field(
        default="fact",
        description="Type: fact, preference, decision, convention, pattern",
    )
    scope: Literal["global", "workspace", "project", "file"] = Field(
        default="project",
        description="Scope: global, workspace, project, file",
    )
    scope_path: str | None = Field(
        default=None,
        description="Path for workspace/project/file scope",
    )
    source: str | None = Field(
        default=None,
        description="Source reference (file path, conversation ID, etc.)",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization",
    )
    check_duplicate: bool = Field(
        default=True,
        description="Check for similar existing entries before storing",
    )
    duplicate_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for duplicate detection (0-1)",
    )
    on_duplicate: Literal["reject", "merge", "store"] = Field(
        default="reject",
        description="Action when duplicate found: reject (return existing), merge, store",
    )
    # Graph relationship parameters
    auto_link: bool = Field(
        default=True,
        description="Automatically create RELATES_TO edges for similar entries",
    )
    auto_link_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for auto-linking",
    )
    relates_to: list[str] | None = Field(
        default=None,
        description="Entry IDs to create RELATES_TO edges with",
    )
    supersedes: str | None = Field(
        default=None,
        description="Entry ID that this entry supersedes",
    )
    depends_on: list[str] | None = Field(
        default=None,
        description="Entry IDs that this entry depends on",
    )
    # Conflict and supersedes detection
    detect_conflicts: bool = Field(
        default=True,
        description="Detect and flag potential contradictions with existing entries",
    )
    suggest_supersedes: bool = Field(
        default=False,
        description="Suggest entries that this new entry might supersede",
    )
    auto_supersede: bool = Field(
        default=False,
        description="Automatically create SUPERSEDES edges for very similar entries",
    )


class RecallInput(BaseModel):
    """Input for the recall tool. Supports semantic search, scope-aware search, and tag search."""

    query: str | None = Field(
        default=None,
        description="Natural language search query",
    )
    file_path: str | None = Field(
        default=None,
        description="Current file path for scope-weighted search (file > project > workspace > global)",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Search by tags. Use alone for exact tag matching, or with query to post-filter semantic results.",
    )
    match_all: bool = Field(
        default=False,
        description="When using tags: True = entries must have ALL tags, False = any matching tag",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results",
    )
    scope: Literal["global", "workspace", "project", "file"] | None = Field(
        default=None,
        description="Filter by scope: global, workspace, project, file",
    )
    scope_path: str | None = Field(
        default=None,
        description="Filter by scope path (prefix match)",
    )
    content_type: Literal["fact", "preference", "decision", "convention", "pattern"] | None = Field(
        default=None,
        description="Filter by type: fact, preference, decision, convention, pattern",
    )
    min_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    )
    # Validity parameters
    exclude_superseded: bool = Field(
        default=True,
        description="Exclude entries that have been superseded by newer entries",
    )
    flag_conflicts: bool = Field(
        default=True,
        description="Mark entries that have unresolved conflicts",
    )
    freshness_boost: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="How much to boost recent entries (0=ignore time, 1=heavy time weight)",
    )
    max_content_length: int | None = Field(
        default=None,
        ge=50,
        le=10000,
        description="Truncate entry content to this length in results (saves tokens). None = full content.",
    )

    @model_validator(mode="after")
    def validate_query_or_tags(self) -> "RecallInput":
        """Must provide query or tags (or both)."""
        if self.query is None and self.tags is None:
            raise ValueError("Must provide 'query' or 'tags' (or both)")
        return self


class GetInput(BaseModel):
    """Input for the get tool."""

    entry_id: str = Field(description="ID of the entry to retrieve")
    include_history: bool = Field(
        default=False,
        description="Include version history for the entry",
    )
    history_limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum history versions to return",
    )


class ForgetInput(BaseModel):
    """Input for the forget tool."""

    entry_id: str = Field(description="ID of the entry to remove or restore")
    hard_delete: bool = Field(
        default=False,
        description="Permanently delete (True) or soft-delete (False)",
    )
    restore: bool = Field(
        default=False,
        description="Restore a previously soft-deleted entry",
    )

    @model_validator(mode="after")
    def validate_restore_and_hard_delete(self) -> "ForgetInput":
        """restore and hard_delete are mutually exclusive."""
        if self.restore and self.hard_delete:
            raise ValueError("Cannot use both 'restore' and 'hard_delete'")
        return self


class UpdateInput(BaseModel):
    """Input for updating an entry."""

    entry_id: str = Field(description="ID of the entry to update")
    content: str | None = Field(
        default=None,
        description="New content (regenerates embedding)",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="New confidence score",
    )
    tags: list[str] | None = Field(
        default=None,
        description="New tags (replaces existing)",
    )


class LinkInput(BaseModel):
    """Input for the link tool. Supports create and remove actions."""

    action: Literal["create", "remove"] = Field(
        default="create",
        description="Action: 'create' a new edge or 'remove' an existing one",
    )
    # For create action
    source_id: str | None = Field(
        default=None,
        description="ID of the source entry (required for create)",
    )
    target_id: str | None = Field(
        default=None,
        description="ID of the target entry (required for create)",
    )
    relation: Literal["relates_to", "supersedes", "depends_on", "conflicts_with"] | None = Field(
        default=None,
        description="Relationship type (required for create)",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this relationship",
    )
    reason: str | None = Field(
        default=None,
        description="Optional reason for this relationship",
    )
    # For remove action
    edge_id: str | None = Field(
        default=None,
        description="ID of the edge to remove (required for remove)",
    )

    @model_validator(mode="after")
    def validate_action_params(self) -> "LinkInput":
        """Validate required params per action."""
        if self.action == "create":
            if not self.source_id or not self.target_id or not self.relation:
                raise ValueError(
                    "create action requires 'source_id', 'target_id', and 'relation'"
                )
        elif self.action == "remove" and not self.edge_id:
            raise ValueError("remove action requires 'edge_id'")
        return self


class TraverseInput(BaseModel):
    """Input for graph navigation. Supports traversal and direct edge lookup."""

    entry_id: str | None = Field(
        default=None,
        description="Entry ID for direct edge lookup (replaces enyal_edges)",
    )
    start_query: str | None = Field(
        default=None,
        description="Query to find starting node for graph traversal",
    )
    relation_types: list[Literal["relates_to", "supersedes", "depends_on", "conflicts_with"]] | None = Field(
        default=None,
        description="Filter by relationship types",
    )
    direction: Literal["outgoing", "incoming", "both"] = Field(
        default="outgoing",
        description="Direction: outgoing, incoming, or both (both only valid with entry_id)",
    )
    max_depth: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Maximum traversal depth",
    )
    relation_type: Literal["relates_to", "supersedes", "depends_on", "conflicts_with"] | None = Field(
        default=None,
        description="Filter edges by single relation type (for edge lookup mode)",
    )

    @model_validator(mode="after")
    def validate_entry_id_or_start_query(self) -> "TraverseInput":
        """Must provide entry_id or start_query."""
        if self.entry_id is None and self.start_query is None:
            raise ValueError("Must provide 'entry_id' or 'start_query'")
        if self.direction == "both" and self.entry_id is None:
            raise ValueError("direction='both' is only valid with 'entry_id' (edge lookup mode)")
        return self


class ImpactInput(BaseModel):
    """Input for the impact tool."""

    entry_id: str | None = Field(
        default=None,
        description="Entry ID to analyze impact for",
    )
    query: str | None = Field(
        default=None,
        description="Or query to find the entry",
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        le=4,
        description="Maximum depth for impact analysis",
    )


class StatusInput(BaseModel):
    """Input for the status tool. Consolidates health/stats/review/analytics."""

    view: Literal["summary", "health", "review", "analytics"] = Field(
        default="summary",
        description="View: summary (stats+health), health (detailed), review (entries needing attention), analytics (usage patterns)",
    )
    category: Literal["all", "stale", "orphan", "conflicts"] = Field(
        default="all",
        description="Category for review view: all, stale, orphan, conflicts",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum entries to return (for review/analytics)",
    )
    entry_id: str | None = Field(
        default=None,
        description="Filter analytics by specific entry",
    )
    event_type: Literal["recall", "update", "link", "impact"] | None = Field(
        default=None,
        description="Filter analytics by event type",
    )
    days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days of history for analytics",
    )


class TransferInput(BaseModel):
    """Input for the transfer tool. Consolidates export/import."""

    direction: Literal["export", "import"] = Field(
        description="Direction: export or import"
    )
    # Export params
    scope: Literal["global", "workspace", "project", "file"] | None = Field(
        default=None, description="Filter by scope level (for export)"
    )
    scope_path: str | None = Field(
        default=None, description="Filter by scope path (for export)"
    )
    include_deprecated: bool = Field(
        default=False, description="Include deprecated entries (for export)"
    )
    # Import params
    data: dict[str, Any] | None = Field(
        default=None, description="Export data to import (required for import)"
    )
    skip_duplicates: bool = Field(
        default=True, description="Skip entries whose IDs already exist (for import)"
    )

    @model_validator(mode="after")
    def validate_import_data(self) -> "TransferInput":
        """Import requires data."""
        if self.direction == "import" and self.data is None:
            raise ValueError("import direction requires 'data'")
        return self


# =============================================================================
# Tool Functions (10 tools)
# =============================================================================


@mcp.tool(
    title="Store Knowledge",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
    ),
)
def enyal_remember(input: RememberInput) -> RememberResponse:
    """Store knowledge that persists across sessions.

    Call this when a convention is established, a decision is made, a pattern is
    discovered, or a user states a preference. Duplicates are auto-checked,
    conflicts auto-detected, and related entries auto-linked by default.
    """
    store = get_store()
    result = store.remember(
        content=input.content,
        content_type=ContextType(input.content_type),
        scope_level=ScopeLevel(input.scope),
        scope_path=input.scope_path,
        source_type="conversation",
        source_ref=input.source,
        tags=input.tags,
        check_duplicate=input.check_duplicate,
        duplicate_threshold=input.duplicate_threshold,
        on_duplicate=input.on_duplicate,
        # Graph parameters
        auto_link=input.auto_link,
        auto_link_threshold=input.auto_link_threshold,
        relates_to=input.relates_to,
        supersedes=input.supersedes,
        depends_on=input.depends_on,
        # Conflict/supersedes detection
        detect_conflicts=input.detect_conflicts,
        suggest_supersedes=input.suggest_supersedes,
        auto_supersede=input.auto_supersede,
    )

    action = result["action"]
    entry_id = result["entry_id"]

    if action == "existing":
        message = f"Found similar existing entry (similarity: {result['similarity']:.2%})"
    elif action == "merged":
        message = f"Merged with existing entry (similarity: {result['similarity']:.2%})"
    else:
        message = f"Stored context: {input.content[:50]}..."

    potential_conflicts = [
        ConflictCandidate(**c) for c in result.get("potential_conflicts", [])
    ]
    supersedes_candidates = [
        ConflictCandidate(**c) for c in result.get("supersedes_candidates", [])
    ]

    return RememberResponse(
        success=True,
        entry_id=entry_id,
        action=action,
        duplicate_of=result.get("duplicate_of"),
        similarity=result.get("similarity"),
        message=message,
        potential_conflicts=potential_conflicts,
        supersedes_candidates=supersedes_candidates,
    )


@mcp.tool(
    title="Search Memory",
    annotations=ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
)
def enyal_recall(input: RecallInput) -> RecallResponse:
    """Search memory for relevant context. Three modes:

    1. query alone: semantic search across all entries
    2. query + file_path: scope-weighted search (file > project > workspace > global)
    3. tags alone: exact tag matching (match_all controls AND vs OR)
    4. query + tags: semantic search, then post-filter by tags

    Use at session start, before decisions, and when recalling conventions.
    """
    store = get_store()
    retrieval = get_retrieval()
    max_len = input.max_content_length

    # Mode 1: Tags-only search
    if input.query is None and input.tags is not None:
        entries = store.search_by_tags(
            tags=input.tags, match_all=input.match_all, limit=input.limit
        )
        return RecallResponse(
            success=True,
            count=len(entries),
            results=[
                EntrySearchResult(
                    id=e.id,
                    content=_truncate_content(e.content, max_len),
                    type=e.content_type.value,
                    scope=e.scope_level.value,
                    scope_path=e.scope_path,
                    confidence=e.confidence,
                    score=1.0,
                    tags=e.tags,
                    created_at=e.created_at.isoformat(),
                    updated_at=e.updated_at.isoformat(),
                    is_superseded=False,
                    has_conflicts=False,
                    freshness_score=1.0,
                )
                for e in entries
            ],
        )

    # Mode 2: Scope-aware search (query + file_path)
    if input.query is not None and input.file_path is not None:
        results = retrieval.search_by_scope(
            query=input.query,
            file_path=input.file_path,
            limit=input.limit,
            min_confidence=input.min_confidence,
            exclude_superseded=input.exclude_superseded,
            flag_conflicts=input.flag_conflicts,
            freshness_boost=input.freshness_boost,
        )
    # Mode 3: Standard semantic search (query only, or query + tags for post-filter)
    else:
        assert input.query is not None  # validated by model_validator
        results = retrieval.search(
            query=input.query,
            limit=input.limit,
            scope_level=ScopeLevel(input.scope) if input.scope else None,
            scope_path=input.scope_path,
            content_type=ContextType(input.content_type) if input.content_type else None,
            min_confidence=input.min_confidence,
            exclude_superseded=input.exclude_superseded,
            flag_conflicts=input.flag_conflicts,
            freshness_boost=input.freshness_boost,
        )

    # Post-filter by tags if both query and tags provided
    if input.tags is not None:
        tag_set = set(input.tags)
        if input.match_all:
            results = [r for r in results if tag_set.issubset(set(r.entry.tags))]
        else:
            results = [r for r in results if tag_set.intersection(set(r.entry.tags))]

    return RecallResponse(
        success=True,
        count=len(results),
        results=[
            EntrySearchResult(
                id=r.entry.id,
                content=_truncate_content(r.entry.content, max_len),
                type=r.entry.content_type.value,
                scope=r.entry.scope_level.value,
                scope_path=r.entry.scope_path,
                confidence=r.entry.confidence,
                score=round(r.score, 4),
                tags=r.entry.tags,
                created_at=r.entry.created_at.isoformat(),
                updated_at=r.entry.updated_at.isoformat(),
                is_superseded=r.is_superseded,
                superseded_by=r.superseded_by,
                has_conflicts=r.has_conflicts,
                freshness_score=round(r.freshness_score, 4),
                adjusted_score=round(r.adjusted_score, 4) if r.adjusted_score else None,
            )
            for r in results
        ],
    )


@mcp.tool(
    title="Get Entry Details",
    annotations=ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
)
def enyal_get(input: GetInput) -> GetResponse:
    """Get a specific context entry by ID.

    Returns full details including metadata and relationships.
    Set include_history=True to also get version history.
    """
    store = get_store()
    entry = store.get(input.entry_id)
    if not entry:
        raise ToolError(f"Entry {input.entry_id} not found")

    edges = store.get_edges(input.entry_id, direction="both")
    outgoing_edges = [e for e in edges if e.source_id == input.entry_id]
    incoming_edges = [e for e in edges if e.target_id == input.entry_id]

    history = None
    version_count = None
    if input.include_history:
        history = store.get_history(input.entry_id, limit=input.history_limit)
        version_count = len(history)

    return GetResponse(
        success=True,
        entry={
            "id": entry.id,
            "content": entry.content,
            "type": entry.content_type.value,
            "scope": entry.scope_level.value,
            "scope_path": entry.scope_path,
            "confidence": entry.confidence,
            "tags": entry.tags,
            "metadata": entry.metadata,
            "source_type": entry.source_type.value if entry.source_type else None,
            "source_ref": entry.source_ref,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "accessed_at": entry.accessed_at.isoformat() if entry.accessed_at else None,
            "access_count": entry.access_count,
            "is_deprecated": entry.is_deprecated,
        },
        edges={
            "outgoing": [
                {
                    "id": e.id,
                    "target_id": e.target_id,
                    "relation": e.edge_type.value,
                    "confidence": e.confidence,
                }
                for e in outgoing_edges
            ],
            "incoming": [
                {
                    "id": e.id,
                    "source_id": e.source_id,
                    "relation": e.edge_type.value,
                    "confidence": e.confidence,
                }
                for e in incoming_edges
            ],
        },
        history=history,
        version_count=version_count,
    )


@mcp.tool(
    title="Update Entry",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
)
def enyal_update(input: UpdateInput) -> UpdateResponse:
    """Update an existing context entry.

    Use this to correct or refine stored content, adjust confidence scores,
    or update tags. If content is updated, the embedding is automatically regenerated.
    """
    store = get_store()
    success = store.update(
        entry_id=input.entry_id,
        content=input.content,
        confidence=input.confidence,
        tags=input.tags,
    )
    if not success:
        raise ToolError(f"Entry {input.entry_id} not found")
    return UpdateResponse(success=True, message=f"Entry {input.entry_id} updated")


@mcp.tool(
    title="Remove or Restore Entry",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False
    ),
)
def enyal_forget(input: ForgetInput) -> ForgetResponse:
    """Remove, deprecate, or restore a context entry.

    Default: soft-delete (deprecated, excluded from search, restorable).
    hard_delete=True: permanent removal.
    restore=True: un-deprecate a previously soft-deleted entry.
    """
    store = get_store()

    if input.restore:
        success = store.restore(input.entry_id)
        if not success:
            raise ToolError(f"Entry {input.entry_id} not found or not deprecated")
        return ForgetResponse(success=True, message=f"Entry {input.entry_id} restored")

    success = store.forget(input.entry_id, hard_delete=input.hard_delete)
    if not success:
        raise ToolError(f"Entry {input.entry_id} not found")
    action = "permanently deleted" if input.hard_delete else "deprecated"
    return ForgetResponse(success=True, message=f"Entry {input.entry_id} has been {action}")


@mcp.tool(
    title="Manage Relationships",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
)
def enyal_link(input: LinkInput) -> LinkResponse:
    """Create or remove a relationship between context entries.

    action='create': Connect two entries with a typed relationship.
    action='remove': Delete an existing edge by ID.

    Relationship types:
    - relates_to: General semantic relationship
    - supersedes: This entry replaces an older one
    - depends_on: This entry requires another
    - conflicts_with: These entries contradict each other
    """
    store = get_store()

    if input.action == "remove":
        assert input.edge_id is not None  # validated by model_validator
        success = store.unlink(input.edge_id)
        if not success:
            raise ToolError(f"Edge {input.edge_id} not found")
        return LinkResponse(success=True, message=f"Removed edge {input.edge_id}")

    # action == "create" — fields validated by model_validator
    assert input.source_id is not None
    assert input.target_id is not None
    assert input.relation is not None
    edge_id = store.link(
        source_id=input.source_id,
        target_id=input.target_id,
        edge_type=EdgeType(input.relation),
        confidence=input.confidence,
        metadata={"reason": input.reason} if input.reason else {},
    )

    if not edge_id:
        raise ToolError("Could not create edge (entries may not exist or edge already exists)")
    return LinkResponse(
        success=True, edge_id=edge_id, message=f"Created {input.relation} relationship"
    )


@mcp.tool(
    title="Navigate Graph",
    annotations=ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
)
def enyal_traverse(input: TraverseInput) -> TraverseResponse:
    """Navigate the knowledge graph. Two modes:

    1. entry_id: Direct edge lookup — returns all edges for an entry (replaces enyal_edges)
    2. start_query: Graph traversal — finds entry via search, then walks the graph

    Use direction='both' with entry_id to get all connections.
    """
    store = get_store()

    # Mode 1: Direct edge lookup
    if input.entry_id is not None:
        edges = store.get_edges(
            entry_id=input.entry_id,
            direction=input.direction,
            edge_type=EdgeType(input.relation_type) if input.relation_type else None,
        )

        return TraverseResponse(
            success=True,
            count=len(edges),
            results=[],
            edges=[
                EdgeBrief(
                    id=e.id,
                    source_id=e.source_id,
                    target_id=e.target_id,
                    relation=e.edge_type.value,
                    confidence=e.confidence,
                    created_at=e.created_at.isoformat(),
                    metadata=e.metadata,
                )
                for e in edges
            ],
        )

    # Mode 2: Graph traversal via search
    assert input.start_query is not None  # validated by model_validator
    retrieval = get_retrieval()
    search_results = retrieval.search(query=input.start_query, limit=1)
    if not search_results:
        raise ToolError(f"No entry found matching: {input.start_query}")

    start_entry = search_results[0].entry
    edge_types: list[EdgeType | str] | None = (
        [EdgeType(et) for et in input.relation_types] if input.relation_types else None
    )

    results = store.traverse(
        start_id=start_entry.id,
        edge_types=edge_types,
        direction=input.direction,
        max_depth=input.max_depth,
    )

    return TraverseResponse(
        success=True,
        start_entry=EntryBrief(
            id=start_entry.id,
            content=start_entry.content,
            type=start_entry.content_type.value,
            scope=start_entry.scope_level.value,
            scope_path=start_entry.scope_path,
            confidence=start_entry.confidence,
            tags=start_entry.tags,
            created_at=start_entry.created_at.isoformat(),
            updated_at=start_entry.updated_at.isoformat(),
        ),
        count=len(results),
        results=[
            {
                "id": r["entry"].id,
                "content": r["entry"].content,
                "depth": r["depth"],
                "relation": r["edge_type"],
                "confidence": r["confidence"],
            }
            for r in results
        ],
    )


@mcp.tool(
    title="Analyze Impact",
    annotations=ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
)
def enyal_impact(input: ImpactInput) -> ImpactResponse:
    """Analyze what would be affected if an entry changes.

    Finds all entries that depend on the specified entry (directly or
    transitively), helping you understand the impact of potential changes.
    """
    store = get_store()
    retrieval = get_retrieval()

    if input.entry_id:
        target = store.get(input.entry_id)
        if not target:
            raise ToolError(f"Entry {input.entry_id} not found")
    elif input.query:
        search_results = retrieval.search(query=input.query, limit=1)
        if not search_results:
            raise ToolError(f"No entry found matching: {input.query}")
        target = search_results[0].entry
    else:
        raise ToolError("Provide either entry_id or query")

    depends_on_results = store.traverse(
        start_id=target.id,
        edge_types=[EdgeType.DEPENDS_ON],
        direction="incoming",
        max_depth=input.max_depth,
    )

    relates_to_results = store.traverse(
        start_id=target.id,
        edge_types=[EdgeType.RELATES_TO],
        direction="incoming",
        max_depth=input.max_depth,
    )

    direct_deps = [r for r in depends_on_results if r["depth"] == 1]
    transitive_deps = [r for r in depends_on_results if r["depth"] > 1]
    related = [r for r in relates_to_results if r["confidence"] >= 0.8]

    return ImpactResponse(
        success=True,
        target=EntryBrief(
            id=target.id,
            content=target.content,
            type=target.content_type.value,
            scope=target.scope_level.value,
            scope_path=target.scope_path,
            confidence=target.confidence,
            tags=target.tags,
            created_at=target.created_at.isoformat(),
            updated_at=target.updated_at.isoformat(),
        ),
        impact={
            "direct_dependencies": len(direct_deps),
            "transitive_dependencies": len(transitive_deps),
            "related_entries": len(related),
        },
        direct_dependencies=[
            {"id": r["entry"].id, "content": r["entry"].content} for r in direct_deps
        ],
        transitive_dependencies=[
            {"id": r["entry"].id, "content": r["entry"].content, "depth": r["depth"]}
            for r in transitive_deps
        ],
        related=[
            {
                "id": r["entry"].id,
                "content": r["entry"].content,
                "confidence": r["confidence"],
            }
            for r in related
        ],
    )


def _get_health_recommendations(health: dict[str, Any]) -> list[str]:
    """Generate recommendations based on health metrics."""
    recommendations = []

    if health["superseded_entries"] > 10:
        recommendations.append(
            f"Consider cleaning up {health['superseded_entries']} superseded entries"
        )
    if health["unresolved_conflicts"] > 0:
        recommendations.append(f"Resolve {health['unresolved_conflicts']} conflicting entries")
    if health["stale_entries"] > 20:
        recommendations.append(f"Review {health['stale_entries']} stale entries (>6 months old)")
    if health["orphan_entries"] > health["total_entries"] * 0.3:
        recommendations.append(
            "Many entries have no connections - consider linking related entries"
        )
    if health["health_score"] < 0.7:
        recommendations.append("Overall health is low - maintenance recommended")

    return recommendations or ["Graph health is good!"]


@mcp.tool(
    title="Memory Status",
    annotations=ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
)
def enyal_status(input: StatusInput) -> StatusResponse:
    """Get memory status, health, review items, or usage analytics.

    Views:
    - summary: Combined stats + health score (default, start here)
    - health: Detailed health metrics with recommendations
    - review: Entries needing attention (stale, orphan, conflicts)
    - analytics: Usage patterns and most-recalled entries
    """
    store = get_store()

    if input.view == "summary":
        stats = store.stats()
        health = store.health_check()
        return StatusResponse(
            success=True,
            view="summary",
            stats={
                "total_entries": stats.total_entries,
                "active_entries": stats.active_entries,
                "deprecated_entries": stats.deprecated_entries,
                "entries_by_type": stats.entries_by_type,
                "entries_by_scope": stats.entries_by_scope,
                "avg_confidence": round(stats.avg_confidence, 3),
                "storage_size_mb": round(stats.storage_size_bytes / (1024 * 1024), 2),
                "oldest_entry": stats.oldest_entry.isoformat() if stats.oldest_entry else None,
                "newest_entry": stats.newest_entry.isoformat() if stats.newest_entry else None,
                "total_edges": stats.total_edges,
                "edges_by_type": stats.edges_by_type,
                "connected_entries": stats.connected_entries,
            },
            health=health,
            recommendations=_get_health_recommendations(health),
        )

    if input.view == "health":
        health = store.health_check()
        return StatusResponse(
            success=True,
            view="health",
            health=health,
            recommendations=_get_health_recommendations(health),
        )

    if input.view == "review":
        stale_entries: list[dict[str, Any]] = []
        orphan_entries: list[dict[str, Any]] = []
        conflicted_entries: list[dict[str, Any]] = []

        if input.category in ("all", "stale"):
            stale = store.get_stale_entries(limit=input.limit)
            stale_entries = [
                {
                    "id": e.id,
                    "content": e.content[:100],
                    "updated_at": e.updated_at.isoformat(),
                    "confidence": e.confidence,
                }
                for e in stale
            ]

        if input.category in ("all", "orphan"):
            orphans = store.get_orphan_entries(limit=input.limit)
            orphan_entries = [
                {
                    "id": e.id,
                    "content": e.content[:100],
                    "created_at": e.created_at.isoformat(),
                }
                for e in orphans
            ]

        if input.category in ("all", "conflicts"):
            conflicts = store.get_conflicted_entries(limit=input.limit)
            conflicted_entries = [
                {
                    "entry1_id": c["entry1"].id,
                    "entry1_content": c["entry1"].content[:100],
                    "entry2_id": c["entry2"].id,
                    "entry2_content": c["entry2"].content[:100],
                }
                for c in conflicts
            ]

        return StatusResponse(
            success=True,
            view="review",
            stale_entries=stale_entries,
            orphan_entries=orphan_entries,
            conflicted_entries=conflicted_entries,
        )

    # view == "analytics"
    analytics = store.get_analytics(
        entry_id=input.entry_id,
        event_type=input.event_type,
        days=input.days,
    )
    return StatusResponse(
        success=True,
        view="analytics",
        analytics=analytics,
    )


@mcp.tool(
    title="Transfer Knowledge",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
)
def enyal_transfer(input: TransferInput) -> TransferResponse:
    """Export or import knowledge for backup, migration, or sharing.

    direction='export': Export entries and edges as structured data.
    direction='import': Import previously exported data.
    """
    store = get_store()

    if input.direction == "export":
        data = store.export_entries(
            scope_level=ScopeLevel(input.scope) if input.scope else None,
            scope_path=input.scope_path,
            include_deprecated=input.include_deprecated,
        )
        return TransferResponse(
            success=True,
            direction="export",
            count=len(data.get("entries", [])),
            data=data,
            message=f"Exported {len(data.get('entries', []))} entries",
        )

    # direction == "import"
    assert input.data is not None  # validated by model_validator
    result = store.import_entries(data=input.data, skip_duplicates=input.skip_duplicates)
    return TransferResponse(
        success=True,
        direction="import",
        entries_imported=result["entries_imported"],
        edges_imported=result["edges_imported"],
        entries_skipped=result["entries_skipped"],
        message=f"Imported {result['entries_imported']} entries and {result['edges_imported']} edges",
    )


# === MCP Prompts ===


@mcp.prompt("session_start")
def session_start_prompt(project_path: str = "") -> str:
    """Start a new session by recalling project conventions and recent decisions."""
    return (
        f"Recall project conventions and recent decisions for the current session.\n"
        f"Project path: {project_path or 'current directory'}\n\n"
        f"Use enyal_recall with query='conventions decisions patterns' "
        f"and file_path='{project_path}' to get scope-weighted context."
    )


@mcp.prompt("maintenance")
def maintenance_prompt() -> str:
    """Run knowledge graph maintenance: check health, review issues, resolve conflicts."""
    return (
        "Run knowledge graph maintenance:\n"
        "1. Call enyal_status with view='health' to check overall graph health score\n"
        "2. If score < 0.8, call enyal_status with view='review' and category='conflicts' to find contradictions\n"
        "3. Call enyal_status with view='review' and category='stale' to find outdated entries\n"
        "4. Call enyal_status with view='review' and category='orphan' to find disconnected entries\n"
        "5. For each issue found, suggest fixes (update, forget, or link as appropriate)"
    )


@mcp.prompt("before_commit")
def before_commit_prompt(project_path: str = "") -> str:  # noqa: ARG001
    """Recall commit conventions before making a git commit."""
    return (
        "Before committing, recall the project's commit conventions.\n"
        "Use enyal_recall with query='commit convention message format' "
        "and scope='project'."
    )


# === MCP Resources ===


@mcp.resource("enyal://health")
def health_resource() -> str:
    """Current knowledge graph health status."""
    import json as _json

    store = get_store()
    health = store.health_check()
    return _json.dumps(health, indent=2, default=str)


@mcp.resource("enyal://stats")
def stats_resource() -> str:
    """Knowledge graph statistics."""
    import json as _json

    store = get_store()
    stats = store.stats()
    return _json.dumps(stats.model_dump(mode="json"), indent=2, default=str)


# Entry point for running the server
def main() -> None:
    """Run the MCP server."""
    import sys

    # Configure logging to stderr (captured by MCP transport)
    log_level = os.environ.get("ENYAL_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    # Also log to file for persistent diagnostics
    db_dir = Path(os.environ.get("ENYAL_DB_PATH", "~/.enyal/context.db")).expanduser().parent
    db_dir.mkdir(parents=True, exist_ok=True)
    log_file = db_dir / "enyal.log"
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_file), maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    except Exception as e:
        logger.warning(f"Could not set up file logging at {log_file}: {e}")

    # Configure SSL settings BEFORE any model loading
    # This is critical for corporate networks with SSL inspection
    from enyal.core.ssl_config import (
        configure_http_backend,
        configure_ssl_environment,
        get_ssl_config,
    )

    ssl_config = get_ssl_config()
    configure_ssl_environment(ssl_config)
    configure_http_backend(ssl_config)

    # Mark SSL as configured so the embedding engine doesn't reconfigure redundantly
    import enyal.embeddings.engine as _engine_module

    _engine_module._ssl_configured = True

    # ── SSL Diagnostic Logging ──────────────────────────────────────────
    # Log comprehensive SSL state for debugging corporate network issues.
    import ssl as _ssl_mod

    logger.info(
        f"SSL config: verify={ssl_config.verify}, "
        f"cert_file={ssl_config.cert_file}, "
        f"offline={ssl_config.offline_mode}, "
        f"OpenSSL={getattr(_ssl_mod, 'OPENSSL_VERSION', 'unknown')}, "
        f"Python={sys.version.split()[0]}"
    )
    if ssl_config.cert_file:
        logger.info(f"SSL: Using CA bundle: {ssl_config.cert_file}")
    if not ssl_config.verify:
        logger.warning("SSL: Verification disabled (insecure)")
    if ssl_config.offline_mode:
        logger.info("SSL: Offline mode enabled")
    if ssl_config.model_path:
        logger.info(f"SSL: Using local model: {ssl_config.model_path}")
    if ssl_config.hf_endpoint:
        logger.info(f"SSL: Using custom HF endpoint: {ssl_config.hf_endpoint}")

    # Run background SSL probe (non-blocking, just logs results)
    if not ssl_config.offline_mode and not ssl_config.model_path:
        try:
            from enyal.core.ssl_config import ssl_diagnostic_probe

            probe = ssl_diagnostic_probe()
            if probe.get("success"):
                logger.debug("SSL probe: connectivity OK")
            else:
                logger.warning(f"SSL probe failed: {probe.get('error', 'unknown')}")
                if probe.get("suggestion"):
                    logger.warning(f"SSL suggestion: {probe['suggestion']}")
        except Exception as e:
            logger.debug(f"SSL probe skipped: {e}")

    mcp.run()
