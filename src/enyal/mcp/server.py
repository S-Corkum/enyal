"""MCP server implementation for Enyal."""

import logging
import os
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from enyal.core.retrieval import RetrievalEngine
from enyal.core.store import ContextStore
from enyal.models.context import ContextType, EdgeType, ScopeLevel

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP(
    name="enyal",
)

# Global store instance (initialized lazily)
_store: ContextStore | None = None
_retrieval: RetrievalEngine | None = None


def get_store() -> ContextStore:
    """Get or create the context store instance."""
    global _store
    if _store is None:
        db_path = os.environ.get("ENYAL_DB_PATH", "~/.enyal/context.db")
        _store = ContextStore(db_path)
        logger.info(f"Initialized context store at: {db_path}")
    return _store


def get_retrieval() -> RetrievalEngine:
    """Get or create the retrieval engine instance."""
    global _retrieval
    if _retrieval is None:
        _retrieval = RetrievalEngine(get_store())
    return _retrieval


# Tool input models
class RememberInput(BaseModel):
    """Input for the remember tool."""

    content: str = Field(description="The context/knowledge to store")
    content_type: str = Field(
        default="fact",
        description="Type: fact, preference, decision, convention, pattern",
    )
    scope: str = Field(
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
        default=False,
        description="Check for similar existing entries before storing",
    )
    duplicate_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for duplicate detection (0-1)",
    )
    on_duplicate: str = Field(
        default="reject",
        description="Action when duplicate found: reject (return existing), merge, store",
    )
    # Graph relationship parameters
    auto_link: bool = Field(
        default=False,
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
        default=False,
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
    """Input for the recall tool."""

    query: str = Field(description="Natural language search query")
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results",
    )
    scope: str | None = Field(
        default=None,
        description="Filter by scope: global, workspace, project, file",
    )
    scope_path: str | None = Field(
        default=None,
        description="Filter by scope path (prefix match)",
    )
    content_type: str | None = Field(
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


class RecallByScopeInput(BaseModel):
    """Input for the recall_by_scope tool."""

    query: str = Field(description="Natural language search query")
    file_path: str = Field(description="Current file path for automatic scope resolution")
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results",
    )
    min_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    )
    # Validity parameters
    exclude_superseded: bool = Field(default=True, description="Exclude superseded entries")
    flag_conflicts: bool = Field(default=True, description="Flag conflicted entries")
    freshness_boost: float = Field(default=0.1, ge=0.0, le=1.0, description="Freshness weight")


class ForgetInput(BaseModel):
    """Input for the forget tool."""

    entry_id: str = Field(description="ID of the entry to remove")
    hard_delete: bool = Field(
        default=False,
        description="Permanently delete (True) or soft-delete (False)",
    )


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
    """Input for the link tool."""

    source_id: str = Field(description="ID of the source entry")
    target_id: str = Field(description="ID of the target entry")
    relation: str = Field(
        description="Relationship type: relates_to, supersedes, depends_on, conflicts_with"
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


class EdgesInput(BaseModel):
    """Input for the edges tool."""

    entry_id: str = Field(description="ID of the entry to get edges for")
    direction: str = Field(
        default="both",
        description="Direction: outgoing, incoming, or both",
    )
    relation_type: str | None = Field(
        default=None,
        description="Filter by relationship type",
    )


class TraverseInput(BaseModel):
    """Input for the traverse tool."""

    start_query: str = Field(description="Query to find starting node(s)")
    relation_types: list[str] | None = Field(
        default=None,
        description="Filter by relationship types",
    )
    direction: str = Field(
        default="outgoing",
        description="Direction: outgoing or incoming",
    )
    max_depth: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Maximum traversal depth",
    )


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


@mcp.tool()
def enyal_remember(input: RememberInput) -> dict[str, Any]:
    """
    Store new context in Enyal's memory.

    Use this to save facts, preferences, decisions, conventions,
    or patterns that should persist across sessions.

    Examples:
    - Facts: "The database uses PostgreSQL 15 with PostGIS"
    - Preferences: "User prefers tabs over spaces"
    - Decisions: "We chose React over Vue for the frontend"
    - Conventions: "All API endpoints follow REST naming"
    - Patterns: "Error handling uses Result<T, E> pattern"

    When check_duplicate is True, the system will look for similar existing
    entries before storing. The on_duplicate parameter controls the behavior:
    - "reject": Return the existing entry ID without storing a duplicate
    - "merge": Combine tags and update confidence of the existing entry
    - "store": Store as a new entry regardless of similarity
    """
    store = get_store()

    try:
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

        # Handle both return types (str or dict)
        if isinstance(result, dict):
            action = result["action"]
            entry_id = result["entry_id"]

            if action == "existing":
                message = f"Found similar existing entry (similarity: {result['similarity']:.2%})"
            elif action == "merged":
                message = f"Merged with existing entry (similarity: {result['similarity']:.2%})"
            else:
                message = f"Stored context: {input.content[:50]}..."

            response = {
                "success": True,
                "entry_id": entry_id,
                "action": action,
                "duplicate_of": result.get("duplicate_of"),
                "similarity": result.get("similarity"),
                "message": message,
            }

            # Include conflict/supersedes info if detection was enabled
            if input.detect_conflicts:
                response["potential_conflicts"] = result.get("potential_conflicts", [])
            if input.suggest_supersedes:
                response["supersedes_candidates"] = result.get("supersedes_candidates", [])

            return response
        else:
            # Legacy string return (check_duplicate=False)
            return {
                "success": True,
                "entry_id": result,
                "action": "created",
                "message": f"Stored context: {input.content[:50]}...",
            }
    except Exception as e:
        logger.exception("Error storing context")
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
def enyal_recall(input: RecallInput) -> dict[str, Any]:
    """
    Search Enyal's memory for relevant context.

    Uses semantic search to find entries similar to your query,
    with optional filtering by scope and content type.

    Returns entries sorted by relevance with confidence scores.
    """
    retrieval = get_retrieval()

    try:
        results = retrieval.search(
            query=input.query,
            limit=input.limit,
            scope_level=ScopeLevel(input.scope) if input.scope else None,
            scope_path=input.scope_path,
            content_type=ContextType(input.content_type) if input.content_type else None,
            min_confidence=input.min_confidence,
            # Validity parameters
            exclude_superseded=input.exclude_superseded,
            flag_conflicts=input.flag_conflicts,
            freshness_boost=input.freshness_boost,
        )

        return {
            "success": True,
            "count": len(results),
            "results": [
                {
                    "id": r.entry.id,
                    "content": r.entry.content,
                    "type": r.entry.content_type.value,
                    "scope": r.entry.scope_level.value,
                    "scope_path": r.entry.scope_path,
                    "confidence": r.entry.confidence,
                    "score": round(r.score, 4),
                    "tags": r.entry.tags,
                    "created_at": r.entry.created_at.isoformat(),
                    "updated_at": r.entry.updated_at.isoformat(),
                    # Validity metadata
                    "is_superseded": r.is_superseded,
                    "superseded_by": r.superseded_by,
                    "has_conflicts": r.has_conflicts,
                    "freshness_score": round(r.freshness_score, 4),
                    "adjusted_score": round(r.adjusted_score, 4) if r.adjusted_score else None,
                }
                for r in results
            ],
        }
    except Exception as e:
        logger.exception("Error recalling context")
        return {
            "success": False,
            "error": str(e),
            "results": [],
        }


@mcp.tool()
def enyal_recall_by_scope(input: RecallByScopeInput) -> dict[str, Any]:
    """
    Search Enyal's memory with automatic scope resolution.

    Searches from most specific (file) to most general (global) scope,
    returning results weighted by scope specificity. Use this when you
    want context relevant to the current file or project you're working in.

    The file_path is used to automatically determine applicable scopes:
    - file: The exact file path
    - project: Detected via .git, pyproject.toml, package.json, etc.
    - workspace: User's projects/code directory
    - global: Always included

    Results from more specific scopes are weighted higher than general ones.
    """
    retrieval = get_retrieval()

    try:
        results = retrieval.search_by_scope(
            query=input.query,
            file_path=input.file_path,
            limit=input.limit,
            min_confidence=input.min_confidence,
            exclude_superseded=input.exclude_superseded,
            flag_conflicts=input.flag_conflicts,
            freshness_boost=input.freshness_boost,
        )

        return {
            "success": True,
            "count": len(results),
            "results": [
                {
                    "id": r.entry.id,
                    "content": r.entry.content,
                    "type": r.entry.content_type.value,
                    "scope": r.entry.scope_level.value,
                    "scope_path": r.entry.scope_path,
                    "confidence": r.entry.confidence,
                    "score": round(r.score, 4),
                    "tags": r.entry.tags,
                    "created_at": r.entry.created_at.isoformat(),
                    "updated_at": r.entry.updated_at.isoformat(),
                    # Validity metadata
                    "is_superseded": r.is_superseded,
                    "superseded_by": r.superseded_by,
                    "has_conflicts": r.has_conflicts,
                    "freshness_score": round(r.freshness_score, 4),
                    "adjusted_score": round(r.adjusted_score, 4) if r.adjusted_score else None,
                }
                for r in results
            ],
        }
    except Exception as e:
        logger.exception("Error recalling context by scope")
        return {
            "success": False,
            "error": str(e),
            "results": [],
        }


@mcp.tool()
def enyal_forget(input: ForgetInput) -> dict[str, Any]:
    """
    Remove or deprecate context from memory.

    By default, entries are soft-deleted (deprecated) and can be restored.
    Use hard_delete=True to permanently remove an entry.

    Soft-deleted entries are excluded from search results but preserved
    for audit purposes.
    """
    store = get_store()

    try:
        success = store.forget(input.entry_id, hard_delete=input.hard_delete)
        if success:
            action = "permanently deleted" if input.hard_delete else "deprecated"
            return {
                "success": True,
                "message": f"Entry {input.entry_id} has been {action}",
            }
        else:
            return {
                "success": False,
                "error": f"Entry {input.entry_id} not found",
            }
    except Exception as e:
        logger.exception("Error forgetting context")
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
def enyal_update(input: UpdateInput) -> dict[str, Any]:
    """
    Update an existing context entry.

    Use this to:
    - Correct or refine stored content
    - Adjust confidence scores
    - Update tags

    If content is updated, the embedding is automatically regenerated.
    """
    store = get_store()

    try:
        success = store.update(
            entry_id=input.entry_id,
            content=input.content,
            confidence=input.confidence,
            tags=input.tags,
        )
        if success:
            return {
                "success": True,
                "message": f"Entry {input.entry_id} updated",
            }
        else:
            return {
                "success": False,
                "error": f"Entry {input.entry_id} not found",
            }
    except Exception as e:
        logger.exception("Error updating context")
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
def enyal_stats() -> dict[str, Any]:
    """
    Get usage statistics and health metrics.

    Returns counts by scope, content type, confidence distribution,
    storage metrics, and date ranges.
    """
    store = get_store()

    try:
        stats = store.stats()
        return {
            "success": True,
            "stats": {
                "total_entries": stats.total_entries,
                "active_entries": stats.active_entries,
                "deprecated_entries": stats.deprecated_entries,
                "entries_by_type": stats.entries_by_type,
                "entries_by_scope": stats.entries_by_scope,
                "avg_confidence": round(stats.avg_confidence, 3),
                "storage_size_mb": round(stats.storage_size_bytes / (1024 * 1024), 2),
                "oldest_entry": stats.oldest_entry.isoformat() if stats.oldest_entry else None,
                "newest_entry": stats.newest_entry.isoformat() if stats.newest_entry else None,
                # Graph statistics
                "total_edges": stats.total_edges,
                "edges_by_type": stats.edges_by_type,
                "connected_entries": stats.connected_entries,
            },
        }
    except Exception as e:
        logger.exception("Error getting stats")
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
def enyal_get(entry_id: str) -> dict[str, Any]:
    """
    Get a specific context entry by ID.

    Returns full details of the entry including all metadata and relationships.
    """
    store = get_store()

    try:
        entry = store.get(entry_id)
        if entry:
            # Get edges for this entry
            edges = store.get_edges(entry_id, direction="both")
            outgoing_edges = [e for e in edges if e.source_id == entry_id]
            incoming_edges = [e for e in edges if e.target_id == entry_id]

            return {
                "success": True,
                "entry": {
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
                "edges": {
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
            }
        else:
            return {
                "success": False,
                "error": f"Entry {entry_id} not found",
            }
    except Exception as e:
        logger.exception("Error getting entry")
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
def enyal_link(input: LinkInput) -> dict[str, Any]:
    """
    Create a relationship between two context entries.

    Use this to explicitly connect related entries. Relationship types:
    - relates_to: General semantic relationship
    - supersedes: This entry replaces an older one
    - depends_on: This entry requires another
    - conflicts_with: These entries contradict each other
    """
    store = get_store()

    try:
        edge_id = store.link(
            source_id=input.source_id,
            target_id=input.target_id,
            edge_type=EdgeType(input.relation),
            confidence=input.confidence,
            metadata={"reason": input.reason} if input.reason else {},
        )

        if edge_id:
            return {
                "success": True,
                "edge_id": edge_id,
                "message": f"Created {input.relation} relationship",
            }
        else:
            return {
                "success": False,
                "error": "Could not create edge (entries may not exist or edge already exists)",
            }
    except ValueError as e:
        return {"success": False, "error": f"Invalid relation type: {e}"}
    except Exception as e:
        logger.exception("Error creating edge")
        return {"success": False, "error": str(e)}


@mcp.tool()
def enyal_unlink(edge_id: str) -> dict[str, Any]:
    """
    Remove a relationship between entries.

    Use this to delete an edge that was created with enyal_link.
    """
    store = get_store()

    try:
        success = store.unlink(edge_id)
        if success:
            return {"success": True, "message": f"Removed edge {edge_id}"}
        else:
            return {"success": False, "error": f"Edge {edge_id} not found"}
    except Exception as e:
        logger.exception("Error removing edge")
        return {"success": False, "error": str(e)}


@mcp.tool()
def enyal_edges(input: EdgesInput) -> dict[str, Any]:
    """
    Get relationships for a context entry.

    Returns all edges connected to the specified entry, optionally
    filtered by direction and relationship type.
    """
    store = get_store()

    try:
        edges = store.get_edges(
            entry_id=input.entry_id,
            direction=input.direction,
            edge_type=EdgeType(input.relation_type) if input.relation_type else None,
        )

        return {
            "success": True,
            "count": len(edges),
            "edges": [
                {
                    "id": e.id,
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relation": e.edge_type.value,
                    "confidence": e.confidence,
                    "created_at": e.created_at.isoformat(),
                    "metadata": e.metadata,
                }
                for e in edges
            ],
        }
    except ValueError as e:
        return {"success": False, "error": str(e), "edges": []}
    except Exception as e:
        logger.exception("Error getting edges")
        return {"success": False, "error": str(e), "edges": []}


@mcp.tool()
def enyal_traverse(input: TraverseInput) -> dict[str, Any]:
    """
    Traverse the knowledge graph from a starting point.

    Finds the starting entry via semantic search, then walks the graph
    following the specified relationship types up to max_depth levels.
    """
    store = get_store()
    retrieval = get_retrieval()

    try:
        # Find starting node via search
        search_results = retrieval.search(query=input.start_query, limit=1)
        if not search_results:
            return {
                "success": False,
                "error": f"No entry found matching: {input.start_query}",
            }

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

        return {
            "success": True,
            "start_entry": {
                "id": start_entry.id,
                "content": start_entry.content,
            },
            "count": len(results),
            "results": [
                {
                    "id": r["entry"].id,
                    "content": r["entry"].content,
                    "depth": r["depth"],
                    "relation": r["edge_type"],
                    "confidence": r["confidence"],
                }
                for r in results
            ],
        }
    except ValueError as e:
        return {"success": False, "error": str(e), "results": []}
    except Exception as e:
        logger.exception("Error traversing graph")
        return {"success": False, "error": str(e), "results": []}


@mcp.tool()
def enyal_impact(input: ImpactInput) -> dict[str, Any]:
    """
    Analyze what would be affected if an entry changes.

    Finds all entries that depend on the specified entry (directly or
    transitively), helping you understand the impact of potential changes.
    """
    store = get_store()
    retrieval = get_retrieval()

    try:
        # Find target entry
        if input.entry_id:
            target = store.get(input.entry_id)
            if not target:
                return {"success": False, "error": f"Entry {input.entry_id} not found"}
        elif input.query:
            search_results = retrieval.search(query=input.query, limit=1)
            if not search_results:
                return {"success": False, "error": f"No entry found matching: {input.query}"}
            target = search_results[0].entry
        else:
            return {"success": False, "error": "Provide either entry_id or query"}

        # Traverse INCOMING depends_on and relates_to edges
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

        # Group by depth
        direct_deps = [r for r in depends_on_results if r["depth"] == 1]
        transitive_deps = [r for r in depends_on_results if r["depth"] > 1]
        related = [r for r in relates_to_results if r["confidence"] >= 0.8]

        return {
            "success": True,
            "target": {
                "id": target.id,
                "content": target.content,
            },
            "impact": {
                "direct_dependencies": len(direct_deps),
                "transitive_dependencies": len(transitive_deps),
                "related_entries": len(related),
            },
            "direct_dependencies": [
                {"id": r["entry"].id, "content": r["entry"].content} for r in direct_deps
            ],
            "transitive_dependencies": [
                {"id": r["entry"].id, "content": r["entry"].content, "depth": r["depth"]}
                for r in transitive_deps
            ],
            "related": [
                {
                    "id": r["entry"].id,
                    "content": r["entry"].content,
                    "confidence": r["confidence"],
                }
                for r in related
            ],
        }
    except Exception as e:
        logger.exception("Error analyzing impact")
        return {"success": False, "error": str(e)}


@mcp.tool()
def enyal_health() -> dict[str, Any]:
    """
    Get comprehensive health metrics for the knowledge graph.

    Returns statistics about:
    - Superseded entries that should be cleaned up
    - Unresolved conflicts needing attention
    - Stale entries not updated recently
    - Orphan entries with no connections
    - Low confidence entries
    - Never accessed entries
    - Overall health score (0-1)
    """
    store = get_store()

    try:
        health = store.health_check()
        return {
            "success": True,
            "health": health,
            "recommendations": _get_health_recommendations(health),
        }
    except Exception as e:
        logger.exception("Error checking health")
        return {"success": False, "error": str(e)}


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


class ReviewInput(BaseModel):
    """Input for the review tool."""

    category: str = Field(
        default="all",
        description="Category to review: all, stale, orphan, conflicts",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum entries to return",
    )


class HistoryInput(BaseModel):
    """Input for the history tool."""

    entry_id: str = Field(description="Entry ID to get history for")
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum versions to return",
    )


class AnalyticsInput(BaseModel):
    """Input for the analytics tool."""

    entry_id: str | None = Field(
        default=None,
        description="Filter by specific entry (optional)",
    )
    event_type: str | None = Field(
        default=None,
        description="Filter by event type: recall, update, link, impact",
    )
    days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days of history to include",
    )


@mcp.tool()
def enyal_review(input: ReviewInput) -> dict[str, Any]:
    """
    Review entries that need attention.

    Categories:
    - all: Summary of all categories
    - stale: Entries not updated in 6+ months
    - orphan: Entries with no graph connections
    - conflicts: Entries with unresolved conflicts
    """
    store = get_store()

    try:
        result: dict[str, Any] = {"success": True}

        if input.category in ("all", "stale"):
            stale = store.get_stale_entries(limit=input.limit)
            result["stale_entries"] = [
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
            result["orphan_entries"] = [
                {
                    "id": e.id,
                    "content": e.content[:100],
                    "created_at": e.created_at.isoformat(),
                }
                for e in orphans
            ]

        if input.category in ("all", "conflicts"):
            conflicts = store.get_conflicted_entries(limit=input.limit)
            result["conflicted_entries"] = [
                {
                    "entry1_id": c["entry1"].id,
                    "entry1_content": c["entry1"].content[:100],
                    "entry2_id": c["entry2"].id,
                    "entry2_content": c["entry2"].content[:100],
                }
                for c in conflicts
            ]

        return result
    except Exception as e:
        logger.exception("Error reviewing entries")
        return {"success": False, "error": str(e)}


@mcp.tool()
def enyal_history(input: HistoryInput) -> dict[str, Any]:
    """
    Get version history for an entry.

    Shows how an entry has changed over time, including content changes,
    confidence updates, and tag modifications.
    """
    store = get_store()

    try:
        history = store.get_history(input.entry_id, limit=input.limit)
        entry = store.get(input.entry_id)

        if not entry:
            return {
                "success": False,
                "error": f"Entry {input.entry_id} not found",
            }

        return {
            "success": True,
            "entry_id": input.entry_id,
            "current_content": entry.content[:100],
            "version_count": len(history),
            "history": history,
        }
    except Exception as e:
        logger.exception("Error getting history")
        return {"success": False, "error": str(e)}


@mcp.tool()
def enyal_analytics(input: AnalyticsInput) -> dict[str, Any]:
    """
    Get usage analytics for the knowledge graph.

    Shows which entries are most frequently recalled, how often
    entries are updated, and usage patterns over time.
    """
    store = get_store()

    try:
        analytics = store.get_analytics(
            entry_id=input.entry_id,
            event_type=input.event_type,
            days=input.days,
        )

        return {
            "success": True,
            "analytics": analytics,
        }
    except Exception as e:
        logger.exception("Error getting analytics")
        return {"success": False, "error": str(e)}


# Entry point for running the server
def main() -> None:
    """Run the MCP server."""
    import sys

    # Configure logging
    log_level = os.environ.get("ENYAL_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

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

    if ssl_config.cert_file:
        logger.info(f"SSL: Using CA bundle: {ssl_config.cert_file}")
    if not ssl_config.verify:
        logger.warning("SSL: Verification disabled (insecure)")
    if ssl_config.offline_mode:
        logger.info("SSL: Offline mode enabled")
    if ssl_config.model_path:
        logger.info(f"SSL: Using local model: {ssl_config.model_path}")

    # Optionally preload the embedding model
    if os.environ.get("ENYAL_PRELOAD_MODEL", "").lower() == "true":
        from enyal.embeddings.engine import EmbeddingEngine

        logger.info("Preloading embedding model...")
        EmbeddingEngine.preload()
        logger.info("Embedding model preloaded")

    # Run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
