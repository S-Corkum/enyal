"""Retrieval engine with hybrid search (semantic + keyword + recency)."""

import math
from datetime import datetime
from pathlib import Path

from enyal.core.store import ContextStore
from enyal.models.context import (
    ContextSearchResult,
    ContextType,
    ScopeLevel,
)


class RetrievalEngine:
    """
    High-level retrieval engine with hybrid search.

    Combines semantic search (vector similarity), keyword search (FTS5),
    and recency weighting for optimal context retrieval.
    """

    def __init__(
        self,
        store: ContextStore,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.3,
        recency_weight: float = 0.1,
        confidence_half_life_days: int = 90,
    ):
        """
        Initialize the retrieval engine.

        Args:
            store: The underlying context store.
            semantic_weight: Weight for semantic similarity (0-1).
            keyword_weight: Weight for keyword match (0-1).
            recency_weight: Weight for recency (0-1).
            confidence_half_life_days: Days until confidence decays to half.
        """
        self.store = store
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.recency_weight = recency_weight
        self.confidence_half_life_days = confidence_half_life_days

        # Normalize weights
        total = semantic_weight + keyword_weight + recency_weight
        self.semantic_weight /= total
        self.keyword_weight /= total
        self.recency_weight /= total

    def search(
        self,
        query: str,
        limit: int = 10,
        scope_level: ScopeLevel | str | None = None,
        scope_path: str | None = None,
        content_type: ContextType | str | None = None,
        min_confidence: float = 0.3,
        include_deprecated: bool = False,
    ) -> list[ContextSearchResult]:
        """
        Perform hybrid search combining semantic and keyword matching.

        Args:
            query: Natural language search query.
            limit: Maximum number of results.
            scope_level: Filter by scope level.
            scope_path: Filter by scope path.
            content_type: Filter by content type.
            min_confidence: Minimum confidence threshold.
            include_deprecated: Include deprecated entries.

        Returns:
            List of search results with combined scores.
        """
        # Get semantic results
        semantic_results = self.store.recall(
            query=query,
            limit=limit * 2,  # Overfetch for merging
            scope_level=scope_level,
            scope_path=scope_path,
            content_type=content_type,
            min_confidence=min_confidence,
            include_deprecated=include_deprecated,
        )

        if not semantic_results:
            return []

        # Calculate combined scores
        now = datetime.utcnow()
        results = []

        for result in semantic_results:
            entry = result["entry"]
            semantic_score = result["score"]

            # Calculate recency score (exponential decay)
            days_since_update = (now - entry.updated_at).days
            recency_score = math.exp(-days_since_update / self.confidence_half_life_days)

            # Calculate effective confidence (with decay)
            effective_confidence = self._calculate_effective_confidence(
                entry.confidence,
                entry.updated_at,
                entry.access_count,
            )

            # Combine scores
            # Note: keyword_weight is applied to semantic_score as a placeholder
            # until FTS5 hybrid search is implemented
            combined_score = (
                self.semantic_weight * semantic_score
                + self.keyword_weight * semantic_score  # TODO: Add FTS5 score
                + self.recency_weight * recency_score
            ) * effective_confidence

            results.append(
                ContextSearchResult(
                    entry=entry,
                    distance=result["distance"],
                    score=combined_score,
                )
            )

        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def search_by_scope(
        self,
        query: str,
        file_path: str | Path,
        limit: int = 10,
        min_confidence: float = 0.3,
    ) -> list[ContextSearchResult]:
        """
        Search with automatic scope resolution based on file path.

        Searches from most specific (file) to most general (global) scope,
        returning results weighted by scope specificity.

        Args:
            query: Natural language search query.
            file_path: Current file path for scope resolution.
            limit: Maximum number of results.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of search results from all applicable scopes.
        """
        scopes = self._get_applicable_scopes(Path(file_path))
        all_results: list[ContextSearchResult] = []
        scope_weights = {"file": 1.0, "project": 0.9, "workspace": 0.8, "global": 0.7}

        for scope_level, scope_path in scopes:
            results = self.search(
                query=query,
                limit=limit,
                scope_level=scope_level,
                scope_path=scope_path,
                min_confidence=min_confidence,
            )

            # Apply scope weight
            weight = scope_weights.get(scope_level, 0.5)
            for result in results:
                # Create new result with adjusted score
                all_results.append(
                    ContextSearchResult(
                        entry=result.entry,
                        distance=result.distance,
                        score=result.score * weight,
                    )
                )

        # Deduplicate by entry ID, keeping highest score
        seen: dict[str, ContextSearchResult] = {}
        for result in all_results:
            entry_id = result.entry.id
            if entry_id not in seen or result.score > seen[entry_id].score:
                seen[entry_id] = result

        # Sort and limit
        final_results = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return final_results[:limit]

    def _calculate_effective_confidence(
        self,
        base_confidence: float,
        updated_at: datetime,
        access_count: int,
    ) -> float:
        """
        Calculate effective confidence with time decay and access boost.

        Args:
            base_confidence: Original confidence score.
            updated_at: Last update timestamp.
            access_count: Number of times accessed.

        Returns:
            Effective confidence score (0-1).
        """
        now = datetime.utcnow()
        days_since_update = (now - updated_at).days

        # Exponential decay with half-life
        decay_factor = math.pow(0.5, days_since_update / self.confidence_half_life_days)

        # Access frequency boost (logarithmic)
        access_boost = min(0.2, math.log1p(access_count) * 0.05)

        return min(1.0, base_confidence * decay_factor + access_boost)

    def _get_applicable_scopes(self, file_path: Path) -> list[tuple[str, str | None]]:
        """
        Get applicable scopes from most to least specific.

        Args:
            file_path: The current file path.

        Returns:
            List of (scope_level, scope_path) tuples.
        """
        scopes: list[tuple[str, str | None]] = []
        file_path = file_path.resolve()

        # File scope
        if file_path.is_file():
            scopes.append(("file", str(file_path)))
            path = file_path.parent
        else:
            path = file_path

        # Project scope (look for project markers)
        project_markers = {".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"}
        current = path
        while current != current.parent:
            if any((current / marker).exists() for marker in project_markers):
                scopes.append(("project", str(current)))
                break
            current = current.parent

        # Workspace scope (user's projects directory)
        home = Path.home()
        common_workspace_names = {"projects", "code", "dev", "workspace", "repos"}
        for name in common_workspace_names:
            workspace = home / name
            if workspace.exists() and str(file_path).startswith(str(workspace)):
                scopes.append(("workspace", str(workspace)))
                break

        # Global scope (always included)
        scopes.append(("global", None))

        return scopes

    def get_related(
        self,
        entry_id: str,
        limit: int = 5,
    ) -> list[ContextSearchResult]:
        """
        Find entries related to a given entry.

        Args:
            entry_id: The ID of the reference entry.
            limit: Maximum number of related entries.

        Returns:
            List of related entries.
        """
        entry = self.store.get(entry_id)
        if not entry:
            return []

        # Search using the entry's content
        results = self.search(
            query=entry.content,
            limit=limit + 1,  # +1 to exclude self
        )

        # Filter out the reference entry
        return [r for r in results if r.entry.id != entry_id][:limit]
