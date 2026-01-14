# Knowledge Graph Implementation Plan

> **Branch:** `feature/knowledge-graph`
> **Created:** 2026-01-14
> **Status:** Ready for Implementation

## Research Notes (Pre-Plan Verification)

### Files Verified
- `src/enyal/models/context.py`: ContextType (line 16), ScopeLevel (line 26), SourceType (line 35), ContextEntry (line 44), ContextSearchResult (line 66), ContextStats (line 76)
- `src/enyal/core/store.py`: SCHEMA_SQL (lines 30-86), ContextStore class (line 113), remember() (line 191), recall() (line 334), find_similar() (line 490), stats() (line 640)
- `src/enyal/core/retrieval.py`: RetrievalEngine class (line 16), search() (line 54), search_by_scope() (line 183)
- `src/enyal/mcp/server.py`: Input models (lines 45-162), MCP tools (lines 164-491)
- `tests/test_store.py`: TestContextStore, TestFTSSearch, TestFindSimilar, TestRememberDeduplication classes
- `tests/conftest.py`: temp_db, mock_embedding, sample_entry fixtures
- `docs/ARCHITECTURE.md`: Full architecture documentation (1228 lines)

### Project Structure
```
src/enyal/
├── core/
│   ├── store.py      # ContextStore - will add edge methods
│   ├── retrieval.py  # RetrievalEngine - optional enhancement
│   └── ssl_config.py # SSL configuration (no changes)
├── models/
│   └── context.py    # Will add EdgeType, ContextEdge
├── mcp/
│   └── server.py     # Will add graph MCP tools
└── embeddings/
    └── engine.py     # EmbeddingEngine (no changes)
```

### Dependencies Confirmed
- `sqlite3` (stdlib) - for edge table
- `pydantic>=2.0.0` - for ContextEdge model
- `uuid4` - already imported in context.py

### Similar Implementations
- Duplicate detection in `store.py:remember()` (lines 247-284) - uses find_similar()
- FTS search in `store.py:fts_search()` (lines 451-488) - query pattern
- Recursive patterns not yet in codebase - will use SQL recursive CTE

---

## Implementation Steps

### Step 1: Add EdgeType Enum and ContextEdge Model

**File:** `src/enyal/models/context.py` (verified via Read)

**Action:** Edit

**Details:**
- Add `EdgeType` enum after `SourceType` (after line 42)
- Add `ContextEdge` model after `ContextEntry` (after line 64)
- Enhance `ContextStats` with graph metrics (lines 76-89)

**Code to Add:**

```python
# After line 42 (after SourceType class)
class EdgeType(StrEnum):
    """Types of relationships between context entries."""

    RELATES_TO = "relates_to"
    SUPERSEDES = "supersedes"
    DEPENDS_ON = "depends_on"
    CONFLICTS_WITH = "conflicts_with"


# After line 64 (after ContextEntry class)
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
```

**ContextStats Enhancement (modify lines 76-89):**
```python
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
    # NEW: Graph statistics
    total_edges: int = 0
    edges_by_type: dict[str, int] = Field(default_factory=dict)
    connected_entries: int = 0

    model_config = {"frozen": True}
```

**Depends On:** None

**Verify:** Run `python -c "from enyal.models.context import EdgeType, ContextEdge; print('OK')"`

**Grounding:** Read of context.py confirmed enum pattern at lines 16-42, model pattern at lines 44-64

---

### Step 2: Add Edge Table Schema

**File:** `src/enyal/core/store.py` (verified via Read)

**Action:** Edit

**Details:**
- Add `context_edges` table definition to SCHEMA_SQL (after line 62, before indexes)
- Add indexes for edge queries

**Code to Add (insert after line 62, before "-- Performance indexes"):**

```sql
-- Knowledge graph edges
CREATE TABLE IF NOT EXISTS context_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL CHECK (edge_type IN (
        'relates_to', 'supersedes', 'depends_on', 'conflicts_with'
    )),
    confidence REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    UNIQUE(source_id, target_id, edge_type),
    FOREIGN KEY (source_id) REFERENCES context_entries(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

-- Edge indexes
CREATE INDEX IF NOT EXISTS idx_edges_source ON context_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON context_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON context_edges(edge_type);
```

**Depends On:** Step 1 (EdgeType enum for validation reference)

**Verify:** Create temp store, verify table exists: `sqlite3 test.db ".schema context_edges"`

**Grounding:** Read of store.py confirmed SCHEMA_SQL pattern at lines 30-86

---

### Step 3: Add Import for EdgeType and ContextEdge

**File:** `src/enyal/core/store.py` (verified via Read)

**Action:** Edit

**Details:**
- Update imports at line 19-25 to include EdgeType, ContextEdge

**Current (lines 19-25):**
```python
from enyal.models.context import (
    ContextEntry,
    ContextStats,
    ContextType,
    ScopeLevel,
    SourceType,
)
```

**New:**
```python
from enyal.models.context import (
    ContextEdge,
    ContextEntry,
    ContextStats,
    ContextType,
    EdgeType,
    ScopeLevel,
    SourceType,
)
```

**Depends On:** Step 1

**Verify:** Import check passes

**Grounding:** Read of store.py confirmed import location at lines 19-25

---

### Step 4: Implement link() Method

**File:** `src/enyal/core/store.py` (verified via Read)

**Action:** Edit

**Details:**
- Add `link()` method after `stats()` (after line 694)
- Add `_row_to_edge()` helper method

**Code to Add:**

```python
def link(
    self,
    source_id: str,
    target_id: str,
    edge_type: EdgeType | str,
    confidence: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    """
    Create a relationship between two context entries.

    Args:
        source_id: ID of the source entry.
        target_id: ID of the target entry.
        edge_type: Type of relationship.
        confidence: Confidence score (0-1).
        metadata: Additional metadata.

    Returns:
        Edge ID if created, None if duplicate or entries don't exist.
    """
    edge = ContextEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=EdgeType(edge_type) if isinstance(edge_type, str) else edge_type,
        confidence=confidence,
        metadata=metadata or {},
    )

    with self._write_transaction() as conn:
        # Verify both entries exist
        source_exists = conn.execute(
            "SELECT 1 FROM context_entries WHERE id = ?", (source_id,)
        ).fetchone()
        target_exists = conn.execute(
            "SELECT 1 FROM context_entries WHERE id = ?", (target_id,)
        ).fetchone()

        if not source_exists or not target_exists:
            logger.warning(f"Cannot create edge: entry not found")
            return None

        try:
            conn.execute(
                """
                INSERT INTO context_edges (
                    id, source_id, target_id, edge_type, confidence, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge.id,
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type.value,
                    edge.confidence,
                    edge.created_at.isoformat(),
                    json.dumps(edge.metadata),
                ),
            )
            logger.info(f"Created edge: {edge.source_id} --{edge.edge_type.value}--> {edge.target_id}")
            return edge.id
        except sqlite3.IntegrityError:
            # Duplicate edge
            logger.debug(f"Edge already exists: {source_id} -> {target_id} ({edge_type})")
            return None

def _row_to_edge(self, row: dict[str, Any]) -> ContextEdge:
    """Convert a database row to a ContextEdge."""
    return ContextEdge(
        id=row["id"],
        source_id=row["source_id"],
        target_id=row["target_id"],
        edge_type=EdgeType(row["edge_type"]),
        confidence=row["confidence"],
        created_at=datetime.fromisoformat(row["created_at"]),
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
    )
```

**Depends On:** Steps 1, 2, 3

**Verify:** Unit test `test_link_basic`

**Grounding:** Pattern from remember() at lines 191-332, _row_to_entry() at lines 696-714

---

### Step 5: Implement unlink() Method

**File:** `src/enyal/core/store.py` (verified via Read)

**Action:** Edit

**Details:**
- Add `unlink()` method after `link()`

**Code to Add:**

```python
def unlink(self, edge_id: str) -> bool:
    """
    Remove a relationship by edge ID.

    Args:
        edge_id: The ID of the edge to remove.

    Returns:
        True if edge was found and removed.
    """
    with self._write_transaction() as conn:
        result = conn.execute(
            "DELETE FROM context_edges WHERE id = ?", (edge_id,)
        )
        return result.rowcount > 0
```

**Depends On:** Step 4

**Verify:** Unit test `test_unlink`

**Grounding:** Pattern from forget() at lines 541-563

---

### Step 6: Implement get_edges() Method

**File:** `src/enyal/core/store.py` (verified via Read)

**Action:** Edit

**Details:**
- Add `get_edges()` method after `unlink()`

**Code to Add:**

```python
def get_edges(
    self,
    entry_id: str,
    direction: str = "both",
    edge_type: EdgeType | str | None = None,
) -> list[ContextEdge]:
    """
    Get edges connected to an entry.

    Args:
        entry_id: The entry to get edges for.
        direction: "outgoing", "incoming", or "both".
        edge_type: Optional filter by edge type.

    Returns:
        List of edges connected to the entry.
    """
    with self._read_transaction() as conn:
        conditions = []
        params: list[Any] = []

        if direction == "outgoing":
            conditions.append("source_id = ?")
            params.append(entry_id)
        elif direction == "incoming":
            conditions.append("target_id = ?")
            params.append(entry_id)
        else:  # both
            conditions.append("(source_id = ? OR target_id = ?)")
            params.extend([entry_id, entry_id])

        if edge_type:
            et = EdgeType(edge_type) if isinstance(edge_type, str) else edge_type
            conditions.append("edge_type = ?")
            params.append(et.value)

        query = f"""
            SELECT * FROM context_edges
            WHERE {" AND ".join(conditions)}
            ORDER BY created_at DESC
        """
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_edge(dict(row)) for row in rows]
```

**Depends On:** Step 4

**Verify:** Unit tests `test_get_edges_outgoing`, `test_get_edges_incoming`

**Grounding:** Pattern from recall() at lines 334-449

---

### Step 7: Implement traverse() Method

**File:** `src/enyal/core/store.py` (verified via Read)

**Action:** Edit

**Details:**
- Add `traverse()` method using recursive CTE

**Code to Add:**

```python
def traverse(
    self,
    start_id: str,
    edge_types: list[EdgeType | str] | None = None,
    direction: str = "outgoing",
    max_depth: int = 2,
) -> list[dict[str, Any]]:
    """
    Traverse the graph from a starting node.

    Args:
        start_id: Entry ID to start traversal from.
        edge_types: Filter by edge types (None = all).
        direction: "outgoing" or "incoming".
        max_depth: Maximum traversal depth (1-4).

    Returns:
        List of dicts with entry, depth, and path information.
    """
    max_depth = min(max(1, max_depth), 4)  # Clamp to 1-4

    # Build edge type filter
    type_filter = ""
    if edge_types:
        types = [
            EdgeType(et).value if isinstance(et, str) else et.value
            for et in edge_types
        ]
        type_placeholders = ",".join("?" * len(types))
        type_filter = f"AND edge_type IN ({type_placeholders})"
    else:
        types = []

    # Direction determines which column to follow
    if direction == "outgoing":
        start_col, next_col = "source_id", "target_id"
    else:
        start_col, next_col = "target_id", "source_id"

    with self._read_transaction() as conn:
        query = f"""
            WITH RECURSIVE traverse_chain AS (
                -- Base case: direct connections
                SELECT
                    {next_col} as entry_id,
                    1 as depth,
                    {next_col} as path,
                    edge_type,
                    confidence
                FROM context_edges
                WHERE {start_col} = ? {type_filter}

                UNION ALL

                -- Recursive case
                SELECT
                    e.{next_col},
                    tc.depth + 1,
                    tc.path || ',' || e.{next_col},
                    e.edge_type,
                    e.confidence
                FROM context_edges e
                JOIN traverse_chain tc ON e.{start_col} = tc.entry_id
                WHERE tc.depth < ?
                    AND tc.path NOT LIKE '%' || e.{next_col} || '%'
                    {type_filter.replace('edge_type', 'e.edge_type') if type_filter else ''}
            )
            SELECT DISTINCT
                entry_id,
                MIN(depth) as min_depth,
                path,
                edge_type,
                confidence
            FROM traverse_chain
            GROUP BY entry_id
            ORDER BY min_depth, entry_id
        """

        # Build params: start_id, [types], max_depth, [types again for recursive]
        params: list[Any] = [start_id]
        params.extend(types)
        params.append(max_depth)
        params.extend(types)

        rows = conn.execute(query, params).fetchall()

        results = []
        for row in rows:
            entry = self.get(row["entry_id"])
            if entry:
                results.append({
                    "entry": entry,
                    "depth": row["min_depth"],
                    "path": row["path"].split(","),
                    "edge_type": row["edge_type"],
                    "confidence": row["confidence"],
                })
        return results
```

**Depends On:** Steps 4, 6

**Verify:** Unit tests `test_traverse_single_hop`, `test_traverse_multi_hop`

**Grounding:** SQL recursive CTE pattern (new to codebase), similar to impact analysis design from planning

---

### Step 8: Enhance remember() with Edge Parameters

**File:** `src/enyal/core/store.py` (verified via Read)

**Action:** Edit

**Details:**
- Add new parameters to `remember()` signature (line 191)
- Add edge creation logic after vector insertion (after line 322)

**Current Signature (lines 191-205):**
```python
def remember(
    self,
    content: str,
    content_type: ContextType | str = ContextType.FACT,
    scope_level: ScopeLevel | str = ScopeLevel.PROJECT,
    scope_path: str | None = None,
    source_type: SourceType | str | None = None,
    source_ref: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    confidence: float = 1.0,
    check_duplicate: bool = False,
    duplicate_threshold: float = 0.85,
    on_duplicate: str = "reject",
) -> str | dict[str, Any]:
```

**New Signature (add after line 204, before closing paren):**
```python
    on_duplicate: str = "reject",
    # NEW: Graph relationship parameters
    auto_link: bool = False,
    auto_link_threshold: float = 0.85,
    relates_to: list[str] | None = None,
    supersedes: str | None = None,
    depends_on: list[str] | None = None,
) -> str | dict[str, Any]:
```

**Add Edge Creation Logic (after line 322, after vector insertion):**
```python
        # Create explicit edges if provided
        edges_created: list[str] = []
        if supersedes:
            edge_id = self.link(entry.id, supersedes, EdgeType.SUPERSEDES)
            if edge_id:
                edges_created.append(edge_id)
        if depends_on:
            for dep_id in depends_on:
                edge_id = self.link(entry.id, dep_id, EdgeType.DEPENDS_ON)
                if edge_id:
                    edges_created.append(edge_id)
        if relates_to:
            for rel_id in relates_to:
                edge_id = self.link(entry.id, rel_id, EdgeType.RELATES_TO)
                if edge_id:
                    edges_created.append(edge_id)

        # Auto-generate RELATES_TO edges based on similarity
        if auto_link:
            similar = self.find_similar(
                content=content,
                threshold=auto_link_threshold,
                limit=5,
                exclude_deprecated=True,
            )
            for match in similar:
                if match["entry_id"] != entry.id:
                    edge_id = self.link(
                        entry.id,
                        match["entry_id"],
                        EdgeType.RELATES_TO,
                        confidence=match["similarity"],
                        metadata={"auto_generated": True},
                    )
                    if edge_id:
                        edges_created.append(edge_id)
```

**Depends On:** Steps 4-7

**Verify:** Unit tests `test_auto_link_on_remember`, `test_explicit_supersedes`

**Grounding:** Read of remember() at lines 191-332, find_similar() at lines 490-539

---

### Step 9: Enhance stats() with Graph Metrics

**File:** `src/enyal/core/store.py` (verified via Read)

**Action:** Edit

**Details:**
- Add graph statistics queries to `stats()` method (lines 640-694)

**Add After Line 682 (before return statement):**
```python
            # Edge statistics
            total_edges = conn.execute(
                "SELECT COUNT(*) FROM context_edges"
            ).fetchone()[0]

            edges_by_type: dict[str, int] = {}
            for row in conn.execute(
                "SELECT edge_type, COUNT(*) as cnt FROM context_edges GROUP BY edge_type"
            ):
                edges_by_type[row["edge_type"]] = row["cnt"]

            connected_entries = conn.execute("""
                SELECT COUNT(DISTINCT id) FROM context_entries
                WHERE id IN (
                    SELECT source_id FROM context_edges
                    UNION
                    SELECT target_id FROM context_edges
                )
            """).fetchone()[0]
```

**Update Return Statement (line 684):**
```python
            return ContextStats(
                # ... existing fields ...
                total_edges=total_edges,
                edges_by_type=edges_by_type,
                connected_entries=connected_entries,
            )
```

**Depends On:** Steps 1, 2

**Verify:** Unit test `test_stats_with_edges`

**Grounding:** Pattern from existing stats() at lines 640-694

---

### Step 10: Add MCP Input Models

**File:** `src/enyal/mcp/server.py` (verified via Read)

**Action:** Edit

**Details:**
- Add input models after existing models (after line 162)

**Code to Add:**

```python
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
```

**Depends On:** None

**Verify:** Import check passes

**Grounding:** Pattern from existing input models at lines 45-162

---

### Step 11: Add MCP Tools

**File:** `src/enyal/mcp/server.py` (verified via Read)

**Action:** Edit

**Details:**
- Add imports for EdgeType at top of file
- Add MCP tools after existing tools (after line 491)

**Update Import (around line 12):**
```python
from enyal.models.context import ContextType, EdgeType, ScopeLevel
```

**Add Tools:**

```python
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
        edge_types = (
            [EdgeType(et) for et in input.relation_types]
            if input.relation_types
            else None
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
                {"id": r["entry"].id, "content": r["entry"].content}
                for r in direct_deps
            ],
            "transitive_dependencies": [
                {"id": r["entry"].id, "content": r["entry"].content, "depth": r["depth"]}
                for r in transitive_deps
            ],
            "related": [
                {"id": r["entry"].id, "content": r["entry"].content, "confidence": r["confidence"]}
                for r in related
            ],
        }
    except Exception as e:
        logger.exception("Error analyzing impact")
        return {"success": False, "error": str(e)}
```

**Depends On:** Steps 4-9

**Verify:** MCP server starts, tools listed

**Grounding:** Pattern from existing tools at lines 164-491

---

### Step 12: Update enyal_remember Tool

**File:** `src/enyal/mcp/server.py` (verified via Read)

**Action:** Edit

**Details:**
- Update RememberInput model (lines 45-83) with graph params
- Update enyal_remember tool (lines 165-234) to pass new params

**Add to RememberInput (after line 82):**
```python
    on_duplicate: str = Field(
        default="reject",
        description="Action when duplicate found: reject, merge, store",
    )
    # NEW: Graph relationship parameters
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
```

**Update enyal_remember Tool Call (around line 188):**
```python
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
            # NEW
            auto_link=input.auto_link,
            auto_link_threshold=input.auto_link_threshold,
            relates_to=input.relates_to,
            supersedes=input.supersedes,
            depends_on=input.depends_on,
        )
```

**Depends On:** Step 8

**Verify:** Remember with auto_link=True creates edges

**Grounding:** Read of RememberInput at lines 45-83, enyal_remember at lines 165-234

---

### Step 13: Add Graph Test Fixtures

**File:** `tests/conftest.py` (verified via Read)

**Action:** Edit

**Details:**
- Add imports for EdgeType, ContextEdge
- Add edge fixtures after existing fixtures (after line 214)

**Update Imports (around line 14):**
```python
from enyal.models.context import (
    ContextEdge,
    ContextEntry,
    ContextSearchResult,
    ContextStats,
    ContextType,
    EdgeType,
    ScopeLevel,
    SourceType,
)
```

**Add Fixtures:**

```python
@pytest.fixture
def sample_edge() -> ContextEdge:
    """Create a sample ContextEdge for tests."""
    return ContextEdge(
        source_id="source-entry-id",
        target_id="target-entry-id",
        edge_type=EdgeType.RELATES_TO,
        confidence=0.9,
        metadata={"test": True},
    )


@pytest.fixture
def sample_edge_supersedes() -> ContextEdge:
    """Create a supersedes edge for tests."""
    return ContextEdge(
        source_id="new-entry-id",
        target_id="old-entry-id",
        edge_type=EdgeType.SUPERSEDES,
        confidence=1.0,
    )


@pytest.fixture
def sample_edge_depends_on() -> ContextEdge:
    """Create a depends_on edge for tests."""
    return ContextEdge(
        source_id="dependent-entry-id",
        target_id="dependency-entry-id",
        edge_type=EdgeType.DEPENDS_ON,
        confidence=1.0,
    )
```

**Depends On:** Step 1

**Verify:** Fixtures importable in tests

**Grounding:** Pattern from existing fixtures at lines 24-214

---

### Step 14: Add Graph Tests to test_store.py

**File:** `tests/test_store.py` (verified via Read)

**Action:** Edit

**Details:**
- Add TestKnowledgeGraph class at end of file (after line 489)

**Code to Add:**

```python
class TestKnowledgeGraph:
    """Tests for knowledge graph functionality."""

    def test_link_basic(self, store: ContextStore) -> None:
        """Test creating a basic edge."""
        entry1_id = store.remember(content="First entry")
        entry2_id = store.remember(content="Second entry")

        edge_id = store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)
        assert edge_id is not None

    def test_link_all_edge_types(self, store: ContextStore) -> None:
        """Test creating edges of all types."""
        entry1_id = store.remember(content="Entry one")
        entry2_id = store.remember(content="Entry two")

        for edge_type in EdgeType:
            # Unlink first to avoid duplicate constraint
            edges = store.get_edges(entry1_id, direction="outgoing")
            for edge in edges:
                store.unlink(edge.id)

            edge_id = store.link(entry1_id, entry2_id, edge_type)
            assert edge_id is not None

    def test_link_duplicate_rejected(self, store: ContextStore) -> None:
        """Test that duplicate edges are rejected."""
        entry1_id = store.remember(content="Entry A")
        entry2_id = store.remember(content="Entry B")

        edge_id1 = store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)
        edge_id2 = store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)

        assert edge_id1 is not None
        assert edge_id2 is None  # Duplicate rejected

    def test_link_nonexistent_entry(self, store: ContextStore) -> None:
        """Test linking to nonexistent entry returns None."""
        entry_id = store.remember(content="Real entry")
        edge_id = store.link(entry_id, "nonexistent-id", EdgeType.RELATES_TO)
        assert edge_id is None

    def test_unlink(self, store: ContextStore) -> None:
        """Test removing an edge."""
        entry1_id = store.remember(content="Entry X")
        entry2_id = store.remember(content="Entry Y")

        edge_id = store.link(entry1_id, entry2_id, EdgeType.DEPENDS_ON)
        assert edge_id is not None

        success = store.unlink(edge_id)
        assert success is True

        # Verify edge is gone
        edges = store.get_edges(entry1_id)
        assert len(edges) == 0

    def test_get_edges_outgoing(self, store: ContextStore) -> None:
        """Test getting outgoing edges."""
        entry1_id = store.remember(content="Source entry")
        entry2_id = store.remember(content="Target entry")

        store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)

        edges = store.get_edges(entry1_id, direction="outgoing")
        assert len(edges) == 1
        assert edges[0].source_id == entry1_id
        assert edges[0].target_id == entry2_id

    def test_get_edges_incoming(self, store: ContextStore) -> None:
        """Test getting incoming edges."""
        entry1_id = store.remember(content="Source")
        entry2_id = store.remember(content="Target")

        store.link(entry1_id, entry2_id, EdgeType.DEPENDS_ON)

        edges = store.get_edges(entry2_id, direction="incoming")
        assert len(edges) == 1
        assert edges[0].source_id == entry1_id

    def test_get_edges_both_directions(self, store: ContextStore) -> None:
        """Test getting edges in both directions."""
        entry1_id = store.remember(content="Middle entry")
        entry2_id = store.remember(content="Left entry")
        entry3_id = store.remember(content="Right entry")

        store.link(entry2_id, entry1_id, EdgeType.RELATES_TO)  # Incoming
        store.link(entry1_id, entry3_id, EdgeType.RELATES_TO)  # Outgoing

        edges = store.get_edges(entry1_id, direction="both")
        assert len(edges) == 2

    def test_get_edges_filtered_by_type(self, store: ContextStore) -> None:
        """Test filtering edges by type."""
        entry1_id = store.remember(content="Entry 1")
        entry2_id = store.remember(content="Entry 2")

        store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)
        store.link(entry1_id, entry2_id, EdgeType.DEPENDS_ON)

        relates_edges = store.get_edges(
            entry1_id, direction="outgoing", edge_type=EdgeType.RELATES_TO
        )
        assert len(relates_edges) == 1
        assert relates_edges[0].edge_type == EdgeType.RELATES_TO

    def test_traverse_single_hop(self, store: ContextStore) -> None:
        """Test traversing one level."""
        entry1_id = store.remember(content="Start node")
        entry2_id = store.remember(content="End node")

        store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)

        results = store.traverse(entry1_id, max_depth=1)
        assert len(results) == 1
        assert results[0]["entry"].id == entry2_id
        assert results[0]["depth"] == 1

    def test_traverse_multi_hop(self, store: ContextStore) -> None:
        """Test traversing multiple levels."""
        entry1_id = store.remember(content="Node A")
        entry2_id = store.remember(content="Node B")
        entry3_id = store.remember(content="Node C")

        store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)
        store.link(entry2_id, entry3_id, EdgeType.RELATES_TO)

        results = store.traverse(entry1_id, max_depth=2)
        assert len(results) == 2

        # Check depths
        depths = {r["entry"].id: r["depth"] for r in results}
        assert depths[entry2_id] == 1
        assert depths[entry3_id] == 2

    def test_traverse_max_depth(self, store: ContextStore) -> None:
        """Test depth limiting."""
        entry1_id = store.remember(content="Level 0")
        entry2_id = store.remember(content="Level 1")
        entry3_id = store.remember(content="Level 2")
        entry4_id = store.remember(content="Level 3")

        store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)
        store.link(entry2_id, entry3_id, EdgeType.RELATES_TO)
        store.link(entry3_id, entry4_id, EdgeType.RELATES_TO)

        results = store.traverse(entry1_id, max_depth=2)

        # Should only get entries at depth 1 and 2, not 3
        entry_ids = {r["entry"].id for r in results}
        assert entry2_id in entry_ids
        assert entry3_id in entry_ids
        assert entry4_id not in entry_ids

    def test_cascade_delete_edges(self, store: ContextStore) -> None:
        """Test edges are deleted when entry is deleted."""
        entry1_id = store.remember(content="Entry to delete")
        entry2_id = store.remember(content="Connected entry")

        store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)

        # Delete entry1 (hard delete)
        store.forget(entry1_id, hard_delete=True)

        # Edges should be gone
        edges = store.get_edges(entry2_id, direction="incoming")
        assert len(edges) == 0

    def test_auto_link_on_remember(self, store: ContextStore) -> None:
        """Test automatic linking on remember."""
        # Store first entry
        store.remember(content="Python is a programming language")

        # Store similar entry with auto_link
        entry2_id = store.remember(
            content="Python programming language guide",
            auto_link=True,
            auto_link_threshold=0.5,  # Lower threshold for test
        )

        # Should have created RELATES_TO edge
        edges = store.get_edges(entry2_id, direction="outgoing", edge_type=EdgeType.RELATES_TO)
        assert len(edges) >= 1

    def test_explicit_supersedes(self, store: ContextStore) -> None:
        """Test explicit supersedes relationship."""
        old_id = store.remember(content="Old decision")
        new_id = store.remember(content="New decision", supersedes=old_id)

        edges = store.get_edges(new_id, direction="outgoing", edge_type=EdgeType.SUPERSEDES)
        assert len(edges) == 1
        assert edges[0].target_id == old_id

    def test_explicit_depends_on(self, store: ContextStore) -> None:
        """Test explicit depends_on relationship."""
        dep_id = store.remember(content="Dependency")
        main_id = store.remember(content="Main entry", depends_on=[dep_id])

        edges = store.get_edges(main_id, direction="outgoing", edge_type=EdgeType.DEPENDS_ON)
        assert len(edges) == 1
        assert edges[0].target_id == dep_id

    def test_stats_with_edges(self, store: ContextStore) -> None:
        """Test that stats includes edge metrics."""
        entry1_id = store.remember(content="Entry with edges 1")
        entry2_id = store.remember(content="Entry with edges 2")

        store.link(entry1_id, entry2_id, EdgeType.RELATES_TO)
        store.link(entry1_id, entry2_id, EdgeType.DEPENDS_ON)

        stats = store.stats()
        assert stats.total_edges == 2
        assert stats.edges_by_type.get("relates_to") == 1
        assert stats.edges_by_type.get("depends_on") == 1
        assert stats.connected_entries == 2
```

**Depends On:** Steps 1-9, 13

**Verify:** `pytest tests/test_store.py::TestKnowledgeGraph -v`

**Grounding:** Pattern from existing test classes in test_store.py

---

### Step 15: Run Tests and Verify

**Action:** Bash

**Command:** `cd /Users/seancorkum/projects/ai-assistance/enyal && uv run pytest tests/ -v`

**Depends On:** Steps 1-14

**Verify:** All tests pass, including new graph tests

**Grounding:** pytest.ini at pyproject.toml lines 103-106

---

### Step 16: Update ARCHITECTURE.md

**File:** `docs/ARCHITECTURE.md` (verified via Read)

**Action:** Edit

**Details:**
- Add Section 10: Knowledge Graph Layer (after Section 9, before Appendix A)

**Insert Before Appendix A (around line 1182):**

```markdown
## 10. Knowledge Graph Layer

### Design Decision: SQLite Edge Table

**Selected:** `context_edges` table with `ON DELETE CASCADE`

**Rationale:**
- Zero new dependencies (stays in SQLite)
- Unified storage with entries and vectors
- Automatic cleanup when entries are deleted
- Supports recursive queries via SQL CTEs

### Edge Types

| Type | Purpose | Direction |
|------|---------|-----------|
| `relates_to` | Semantic relationship (auto-generated) | Bidirectional conceptually |
| `supersedes` | Entry A replaces entry B | A → B |
| `depends_on` | Entry A requires entry B | A → B |
| `conflicts_with` | Entries contradict each other | Bidirectional |

### Schema

```sql
CREATE TABLE context_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES context_entries(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES context_entries(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    UNIQUE(source_id, target_id, edge_type)
);
```

### Auto-Linking

When `auto_link=True` is passed to `remember()`:
1. After storing the entry, find similar entries via `find_similar()`
2. For each entry with similarity >= threshold, create `RELATES_TO` edge
3. Edge confidence is set to the similarity score
4. Metadata includes `{"auto_generated": true}`

This builds the graph automatically without user friction.

### Graph Traversal

Uses SQL recursive CTEs for efficient traversal:

```sql
WITH RECURSIVE traverse_chain AS (
    SELECT target_id as entry_id, 1 as depth
    FROM context_edges WHERE source_id = ?
    UNION ALL
    SELECT e.target_id, tc.depth + 1
    FROM context_edges e
    JOIN traverse_chain tc ON e.source_id = tc.entry_id
    WHERE tc.depth < ?
)
SELECT * FROM traverse_chain;
```

### Impact Analysis

`enyal_impact` traverses INCOMING `depends_on` edges to find all entries that would be affected by changing the target entry.

### MCP Tools

| Tool | Description |
|------|-------------|
| `enyal_link` | Create explicit relationship |
| `enyal_unlink` | Remove a relationship |
| `enyal_edges` | Get edges for an entry |
| `enyal_traverse` | Walk the graph |
| `enyal_impact` | Find affected entries |

### Statistics

`enyal_stats` now includes:
- `total_edges`: Count of all edges
- `edges_by_type`: Breakdown by relationship type
- `connected_entries`: Entries with at least one edge
```

**Depends On:** Steps 1-14

**Verify:** Documentation renders correctly

**Grounding:** Pattern from existing sections in ARCHITECTURE.md

---

## Verification Checklist

- [ ] All existing tests pass (`pytest tests/`)
- [ ] New graph tests pass (`pytest tests/test_store.py::TestKnowledgeGraph`)
- [ ] MCP server starts without errors (`python -m enyal.mcp.server`)
- [ ] `enyal_stats` shows edge metrics
- [ ] `enyal_link` creates edges successfully
- [ ] `enyal_traverse` walks the graph correctly
- [ ] `enyal_impact` finds dependencies
- [ ] `enyal_remember` with `auto_link=True` creates edges
- [ ] Edge cascade delete works (delete entry removes its edges)
- [ ] ARCHITECTURE.md section is accurate

---

## Success Criteria

1. **Backward Compatible:** All existing functionality unchanged
2. **Graph Foundation:** Edges can be created, queried, and deleted
3. **Auto-Linking:** Similar entries automatically connected
4. **Impact Analysis:** Dependencies can be traced
5. **Observability:** Stats show graph health
6. **Documentation:** Architecture doc updated
