"""MCP server implementation for Enyal."""

import logging
import os
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from enyal.core.retrieval import RetrievalEngine
from enyal.core.store import ContextStore
from enyal.models.context import ContextType, ScopeLevel

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP(
    name="enyal",
    version="0.1.0",
    description="Persistent, queryable memory for AI coding agents",
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
    """
    store = get_store()

    try:
        entry_id = store.remember(
            content=input.content,
            content_type=ContextType(input.content_type),
            scope_level=ScopeLevel(input.scope),
            scope_path=input.scope_path,
            source_type="conversation",
            source_ref=input.source,
            tags=input.tags,
        )
        return {
            "success": True,
            "entry_id": entry_id,
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

    Returns full details of the entry including all metadata.
    """
    store = get_store()

    try:
        entry = store.get(entry_id)
        if entry:
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
