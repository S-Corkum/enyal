# Enyal Architecture Document

> **Version:** 1.0.0-draft
> **Last Updated:** December 2024
> **Status:** Pending Review

## Executive Summary

Enyal is a persistent, queryable memory system for AI coding agents. It transforms ephemeral conversation context into durable institutional knowledge that survives session restarts. This document details the architectural decisions for storage, vector search, embeddings, concurrency, and MCP integration.

---

## 1. Storage Engine Selection

### Decision: SQLite with WAL Mode

**Selected:** SQLite (via Python 3.13 stdlib `sqlite3`)

### Evaluation Matrix

| Criteria | SQLite | DuckDB | LMDB | RocksDB |
|----------|--------|--------|------|---------|
| Write Performance (small updates) | Good | Fair | Good | Excellent |
| Read Performance | Excellent | Excellent | Excellent | Good |
| Cross-Platform Reliability | Excellent | Good | Good | Fair |
| Python 3.13 Compatibility | Guaranteed | Likely | Likely | Uncertain |
| Single-File Portability | Yes | Yes | No (2 files) | No (directory) |
| ACID Compliance | Full | Full | Full | Full |
| Embedded Operation | Yes | Yes | Yes | Yes |

### Detailed Analysis

#### SQLite (Selected)

**Strengths:**
- Built into Python 3.13 stdlib—zero external dependency risk
- Single `.db` file for all data—trivial backup/restore/migration
- WAL (Write-Ahead Logging) mode enables concurrent reads during writes
- Battle-tested in billions of deployments worldwide
- Integrates with sqlite-vec for unified relational + vector storage
- Excellent cross-platform support (macOS Intel/ARM, Windows, Linux)

**Trade-offs:**
- Single-writer model (one write transaction at a time)
- Write throughput limited vs. LSM-tree databases under heavy load

**Why this trade-off is acceptable:**
Enyal's write pattern is moderate (10-100 writes per session, 1-10KB each). This is well within SQLite's comfort zone. The simplicity and reliability outweigh marginal write performance gains from alternatives.

#### DuckDB (Rejected)

**Why not:**
- Optimized for OLAP (analytical queries on large datasets), not OLTP (frequent small writes)
- Bulk insert-oriented design doesn't match our update patterns
- Larger binary size and dependency footprint
- Less mature ecosystem for our use case

#### LMDB (Rejected)

**Why not:**
- Two files required (data + lock)—complicates backup/restore
- Fixed `map_size` must be set upfront; growth requires re-opening with larger size
- `MapFullError` handling adds complexity
- Single-writer model similar to SQLite but without the ecosystem benefits

#### RocksDB (Rejected)

**Why not:**
- Directory-based storage (multiple SST files)—worst for portability
- Cross-platform builds are problematic (often requires compilation)
- `python-rocksdb` has uncertain Python 3.13 compatibility (Cython extension)
- Overkill for our write volume; designed for write-heavy workloads

### Configuration

```python
import sqlite3

def create_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")        # Enable WAL for concurrency
    conn.execute("PRAGMA busy_timeout=5000")       # 5s timeout for lock contention
    conn.execute("PRAGMA synchronous=NORMAL")      # Balance durability/performance
    conn.execute("PRAGMA cache_size=-64000")       # 64MB cache
    conn.execute("PRAGMA foreign_keys=ON")         # Enforce referential integrity
    return conn
```

---

## 2. Vector Search Approach

### Decision: sqlite-vec (SQLite Extension)

**Selected:** sqlite-vec

### Evaluation Matrix

| Criteria | sqlite-vec | usearch | hnswlib | faiss-cpu |
|----------|------------|---------|---------|-----------|
| Integration with SQLite | Native | Separate file | Separate file | Separate file |
| Atomic Transactions | Yes | No | No | No |
| Binary Size | ~200KB | ~2MB | ~1MB | ~50MB+ |
| Cross-Platform | Excellent | Good | Good | Fair |
| Index Build Time | Good | Excellent | Good | Good |
| Query Speed (10k vectors) | ~5ms | ~1ms | ~2ms | ~3ms |
| Incremental Updates | Yes | Yes | Yes | Limited |

### Detailed Analysis

#### sqlite-vec (Selected)

**Strengths:**
- **Unified storage:** Vectors live in the same SQLite database as metadata
- **Atomic transactions:** Insert metadata and vector in single transaction
- **Single-file backup:** One file contains everything
- **Pure C:** Runs anywhere SQLite runs—perfect cross-platform support
- **Small footprint:** ~200KB extension
- **Multiple vector types:** float32, int8, binary quantization
- **Hybrid search ready:** Combine with FTS5 for keyword+semantic search

**Trade-offs:**
- Newer project (less battle-tested than faiss/hnswlib)
- May have performance limits at extreme scale (1M+ vectors)

**Why this trade-off is acceptable:**
For a context store with <100k entries, sqlite-vec easily meets the <50ms query requirement (~5ms for 10k vectors). The simplicity of unified storage outweighs marginal performance differences at our scale.

#### Separate Index Libraries (hnswlib/usearch/faiss) — Rejected

**Why not:**
- **Sync complexity:** Must keep separate index file in sync with SQLite
- **No atomic transactions:** Metadata update can succeed while vector update fails
- **Backup complexity:** Multiple files to manage
- **Added abstraction layer:** More code, more bugs

**When to reconsider:**
If Enyal scales to 1M+ vectors and query latency becomes problematic, we could add a secondary hnswlib/usearch index as a read-optimized cache, with sqlite-vec as the source of truth.

### Schema Design

```sql
-- Vector index table (sqlite-vec virtual table)
CREATE VIRTUAL TABLE context_vectors USING vec0(
    entry_id TEXT PRIMARY KEY,     -- Foreign key to context_entries
    embedding float[384]           -- 384-dimensional from all-MiniLM-L6-v2
);

-- Example KNN query
SELECT
    ce.id,
    ce.content,
    cv.distance
FROM context_vectors cv
JOIN context_entries ce ON ce.id = cv.entry_id
WHERE cv.embedding MATCH ?  -- Query vector as parameter
  AND k = 10                -- Top 10 results
ORDER BY cv.distance;
```

### Binary Quantization (Future Optimization)

For scaling to 1M+ entries, sqlite-vec supports binary quantization:

```sql
-- Coarse index with 32x storage reduction
CREATE VIRTUAL TABLE context_vectors_coarse USING vec0(
    entry_id TEXT PRIMARY KEY,
    embedding_coarse bit[384]  -- Binary quantized
);

-- Two-stage search: coarse filter → fine rerank
WITH coarse_matches AS (
    SELECT entry_id, embedding
    FROM context_vectors
    WHERE embedding_coarse MATCH vec_quantize_binary(?)
      AND k = 100  -- Overfetch for reranking
)
SELECT entry_id, vec_distance_L2(embedding, ?) as distance
FROM coarse_matches
ORDER BY distance
LIMIT 10;
```

---

## 3. Data Model Design

### Core Entities

```
┌─────────────────────────────────────────────────────────────────┐
│                        context_entries                          │
├─────────────────────────────────────────────────────────────────┤
│ id (PK)          │ UUID                                         │
│ content          │ The actual knowledge/context                 │
│ content_type     │ fact, preference, decision, convention       │
│ scope_level      │ global, workspace, project, file             │
│ scope_path       │ Path for non-global scopes                   │
│ confidence       │ 0.0-1.0, decays over time                    │
│ created_at       │ ISO timestamp                                │
│ updated_at       │ ISO timestamp                                │
│ accessed_at      │ Last retrieval time                          │
│ access_count     │ Usage frequency                              │
│ source_type      │ conversation, file, commit, manual           │
│ source_ref       │ Reference to origin                          │
│ tags             │ JSON array                                   │
│ metadata         │ JSON object for extensibility                │
│ is_deprecated    │ Soft delete flag                             │
└─────────────────────────────────────────────────────────────────┘
         │
         │ 1:1
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       context_vectors                           │
├─────────────────────────────────────────────────────────────────┤
│ entry_id (PK)    │ Foreign key to context_entries              │
│ embedding        │ float[384] vector                           │
└─────────────────────────────────────────────────────────────────┘
         │
         │ FTS5 index
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        context_fts                              │
├─────────────────────────────────────────────────────────────────┤
│ content          │ Full-text searchable content                │
└─────────────────────────────────────────────────────────────────┘
```

### Schema DDL

```sql
-- Main context entries table
CREATE TABLE context_entries (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    content_type TEXT NOT NULL CHECK (content_type IN (
        'fact', 'preference', 'decision', 'convention', 'pattern'
    )),
    scope_level TEXT NOT NULL CHECK (scope_level IN (
        'global', 'workspace', 'project', 'file'
    )),
    scope_path TEXT,
    confidence REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    accessed_at TEXT,
    access_count INTEGER NOT NULL DEFAULT 0,
    source_type TEXT CHECK (source_type IN (
        'conversation', 'file', 'commit', 'manual', NULL
    )),
    source_ref TEXT,
    tags TEXT DEFAULT '[]',      -- JSON array
    metadata TEXT DEFAULT '{}',  -- JSON object
    is_deprecated INTEGER NOT NULL DEFAULT 0
);

-- Vector embeddings (sqlite-vec)
CREATE VIRTUAL TABLE context_vectors USING vec0(
    entry_id TEXT PRIMARY KEY,
    embedding float[384]
);

-- Full-text search (FTS5)
CREATE VIRTUAL TABLE context_fts USING fts5(
    content,
    content='context_entries',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER context_fts_insert AFTER INSERT ON context_entries BEGIN
    INSERT INTO context_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER context_fts_delete AFTER DELETE ON context_entries BEGIN
    INSERT INTO context_fts(context_fts, rowid, content)
    VALUES('delete', old.rowid, old.content);
END;

CREATE TRIGGER context_fts_update AFTER UPDATE ON context_entries BEGIN
    INSERT INTO context_fts(context_fts, rowid, content)
    VALUES('delete', old.rowid, old.content);
    INSERT INTO context_fts(rowid, content) VALUES (new.rowid, new.content);
END;

-- Performance indexes
CREATE INDEX idx_context_scope ON context_entries(scope_level, scope_path);
CREATE INDEX idx_context_type ON context_entries(content_type);
CREATE INDEX idx_context_updated ON context_entries(updated_at DESC);
CREATE INDEX idx_context_confidence ON context_entries(confidence DESC);
CREATE INDEX idx_context_active ON context_entries(is_deprecated) WHERE is_deprecated = 0;
```

### Hierarchical Scoping

Scopes form a hierarchy from most specific to most general:

```
file       →  /Users/dev/myapp/src/auth.py
project    →  /Users/dev/myapp/
workspace  →  /Users/dev/
global     →  (no path)
```

**Query Resolution:**
1. Find entries matching the most specific scope
2. Fall back to broader scopes if needed
3. Combine results with scope-weighted relevance

```python
def get_applicable_scopes(file_path: str) -> list[tuple[str, str]]:
    """Return scopes from most to least specific."""
    from pathlib import Path

    scopes = [("file", file_path)]

    path = Path(file_path).parent
    # Walk up to find project root (has .git, pyproject.toml, etc.)
    while path != path.parent:
        if (path / ".git").exists() or (path / "pyproject.toml").exists():
            scopes.append(("project", str(path)))
            break
        path = path.parent

    # Workspace is user's projects directory
    home = Path.home()
    if "projects" in file_path.lower():
        workspace = str(home / "projects")
        scopes.append(("workspace", workspace))

    scopes.append(("global", None))
    return scopes
```

### Confidence Decay

Context confidence decays over time to surface stale information:

```python
import math
from datetime import datetime, timedelta

def calculate_confidence(
    base_confidence: float,
    created_at: datetime,
    updated_at: datetime,
    access_count: int,
    half_life_days: int = 90
) -> float:
    """Calculate current confidence with time decay."""
    now = datetime.utcnow()
    days_since_update = (now - updated_at).days

    # Exponential decay with half-life
    decay_factor = math.pow(0.5, days_since_update / half_life_days)

    # Access frequency boost (log scale)
    access_boost = min(0.2, math.log1p(access_count) * 0.05)

    return min(1.0, base_confidence * decay_factor + access_boost)
```

---

## 4. Concurrency Model

### Decision: WAL Mode + Application-Level Write Serialization

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Server Process                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Agent 1   │  │   Agent 2   │  │   Agent 3   │             │
│  │  (Reader)   │  │  (Reader)   │  │  (Writer)   │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│  ┌──────────────────────────────────────────────────┐          │
│  │              Connection Pool                      │          │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐          │          │
│  │  │ Conn 1  │  │ Conn 2  │  │ Conn 3  │          │          │
│  │  │ (Read)  │  │ (Read)  │  │ (Write) │          │          │
│  │  └────┬────┘  └────┬────┘  └────┬────┘          │          │
│  └───────┼───────────┼───────────┼──────────────────┘          │
│          │           │           │                              │
│          ▼           ▼           ▼                              │
│  ┌──────────────────────────────────────────────────┐          │
│  │           SQLite WAL Mode Database                │          │
│  │  ┌─────────────────────────────────────────────┐ │          │
│  │  │  Readers: Unlimited concurrent              │ │          │
│  │  │  Writers: Serialized (one at a time)        │ │          │
│  │  └─────────────────────────────────────────────┘ │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
import sqlite3
import threading
from contextlib import contextmanager
from typing import Generator, Any

class ContextStore:
    """Thread-safe context store with WAL mode and write serialization."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript(SCHEMA_DDL)

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def read_transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Read transaction - no locking needed with WAL."""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            pass  # No commit needed for reads

    @contextmanager
    def write_transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Write transaction - serialized with lock."""
        with self._write_lock:
            conn = self._get_connection()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def recall(self, query: str, limit: int = 10) -> list[dict]:
        """Semantic search - read operation."""
        with self.read_transaction() as conn:
            # Vector search + metadata join
            results = conn.execute("""
                SELECT ce.*, cv.distance
                FROM context_vectors cv
                JOIN context_entries ce ON ce.id = cv.entry_id
                WHERE cv.embedding MATCH ?
                  AND k = ?
                  AND ce.is_deprecated = 0
                ORDER BY cv.distance
            """, (query_embedding, limit)).fetchall()
            return [dict(r) for r in results]

    def remember(self, content: str, **metadata) -> str:
        """Store context - write operation."""
        with self.write_transaction() as conn:
            entry_id = generate_uuid()
            embedding = self._embed(content)

            # Insert metadata
            conn.execute("""
                INSERT INTO context_entries (id, content, ...)
                VALUES (?, ?, ...)
            """, (entry_id, content, ...))

            # Insert vector (same transaction!)
            conn.execute("""
                INSERT INTO context_vectors (entry_id, embedding)
                VALUES (?, ?)
            """, (entry_id, embedding))

            return entry_id
```

### Race Condition Prevention

1. **WAL Mode:** Readers never block writers, writers never block readers
2. **Write Lock:** Application-level lock serializes writes
3. **busy_timeout:** SQLite waits up to 5s for locks before failing
4. **Transaction Isolation:** Each transaction sees consistent snapshot

### Handling Multi-Process Access

If multiple Enyal MCP servers access the same database:

```python
# SQLite handles multi-process via file locking
# WAL mode uses shared memory for coordination
# busy_timeout handles contention

conn.execute("PRAGMA locking_mode=NORMAL")  # Default, allows multi-process
conn.execute("PRAGMA busy_timeout=10000")   # Longer timeout for multi-process
```

---

## 5. Local Embedding Strategy

### Decision: sentence-transformers with all-MiniLM-L6-v2

### Model Selection

| Model | Dimensions | Size | Quality | Speed |
|-------|------------|------|---------|-------|
| all-MiniLM-L6-v2 | 384 | 80MB | Good | Fast |
| all-mpnet-base-v2 | 768 | 420MB | Better | Slower |
| paraphrase-MiniLM-L3-v2 | 384 | 60MB | Fair | Fastest |

**Selected: all-MiniLM-L6-v2**

**Rationale:**
- 384 dimensions is optimal balance of quality and storage
- 80MB model loads quickly (under 1s cold start)
- Excellent for semantic similarity on technical content
- Well-tested in production systems

### Storage Calculation

```
100,000 entries × 384 dims × 4 bytes/float = 153.6 MB
```

This fits comfortably within the 500MB memory target.

### Implementation

```python
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingEngine:
    """Lazy-loaded embedding engine."""

    _model: SentenceTransformer | None = None
    _model_name = "all-MiniLM-L6-v2"

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """Lazy load model on first use."""
        if cls._model is None:
            cls._model = SentenceTransformer(cls._model_name)
        return cls._model

    @classmethod
    def embed(cls, text: str) -> np.ndarray:
        """Generate embedding for text."""
        model = cls.get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    @classmethod
    def embed_batch(cls, texts: list[str]) -> np.ndarray:
        """Batch embedding for efficiency."""
        model = cls.get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, batch_size=32)
        return embeddings.astype(np.float32)
```

### Cold Start Mitigation

1. **Lazy Loading:** Model loads only when first embedding is requested
2. **Model Caching:** Loaded model persists in memory for session lifetime
3. **Warm-up Option:** Optional pre-loading during server startup

```python
async def server_lifespan(server):
    """Optional warm-up during server startup."""
    if os.getenv("ENYAL_PRELOAD_MODEL", "false").lower() == "true":
        EmbeddingEngine.get_model()  # Pre-load
    yield
```

### ONNX Optimization (Future)

For production performance optimization:

```python
from sentence_transformers import SentenceTransformer

# Export to ONNX
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("model_onnx", create_onnx=True)

# Use ONNX runtime for inference
import onnxruntime as ort

session = ort.InferenceSession("model_onnx/model.onnx")
# ~2x faster inference
```

---

## 6. MCP Server Implementation

### Decision: FastMCP

### Comparison

| Aspect | FastMCP | Official SDK (Low-Level) |
|--------|---------|-------------------------|
| API Style | Decorator-based | Handler registration |
| Boilerplate | Minimal | Substantial |
| Transport | Built-in stdio/SSE | Manual setup |
| Learning Curve | Low | Medium |
| Flexibility | High (can access low-level) | Full control |

**Selected: FastMCP**

**Rationale:**
- Clean, Pythonic decorator API matches our design philosophy
- stdio transport is default—perfect for Claude Code integration
- Less code to maintain
- Can still access MCP primitives when needed

### Server Implementation

```python
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import Optional

mcp = FastMCP(
    name="enyal",
    version="1.0.0",
    description="Persistent memory for AI coding agents"
)

class RememberInput(BaseModel):
    content: str = Field(description="The context to remember")
    content_type: str = Field(
        default="fact",
        description="Type: fact, preference, decision, convention, pattern"
    )
    scope: str = Field(
        default="project",
        description="Scope: global, workspace, project, file"
    )
    scope_path: Optional[str] = Field(
        default=None,
        description="Path for workspace/project/file scope"
    )
    source: Optional[str] = Field(
        default=None,
        description="Source reference (file, conversation, etc.)"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization"
    )

class RecallInput(BaseModel):
    query: str = Field(description="Search query (semantic)")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    scope: Optional[str] = Field(
        default=None,
        description="Filter by scope level"
    )
    content_type: Optional[str] = Field(
        default=None,
        description="Filter by content type"
    )
    min_confidence: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Minimum confidence threshold"
    )

@mcp.tool()
def enyal_remember(input: RememberInput) -> dict:
    """
    Store new context in Enyal's memory.

    Use this to save facts, preferences, decisions, conventions,
    or patterns that should persist across sessions.
    """
    store = get_context_store()
    entry_id = store.remember(
        content=input.content,
        content_type=input.content_type,
        scope_level=input.scope,
        scope_path=input.scope_path,
        source_type="conversation",
        source_ref=input.source,
        tags=input.tags
    )
    return {"success": True, "entry_id": entry_id}

@mcp.tool()
def enyal_recall(input: RecallInput) -> list[dict]:
    """
    Search Enyal's memory for relevant context.

    Uses semantic search to find entries similar to the query,
    with optional filtering by scope and content type.
    """
    store = get_context_store()
    results = store.recall(
        query=input.query,
        limit=input.limit,
        scope_level=input.scope,
        content_type=input.content_type,
        min_confidence=input.min_confidence
    )
    return results

@mcp.tool()
def enyal_forget(entry_id: str, hard_delete: bool = False) -> dict:
    """
    Remove or deprecate context from memory.

    By default, entries are soft-deleted (deprecated) and can be restored.
    Use hard_delete=True to permanently remove.
    """
    store = get_context_store()
    success = store.forget(entry_id, hard_delete=hard_delete)
    return {"success": success, "action": "deleted" if hard_delete else "deprecated"}

@mcp.tool()
def enyal_stats() -> dict:
    """
    Get usage statistics and health metrics.

    Returns counts by scope, content type, confidence distribution,
    and storage metrics.
    """
    store = get_context_store()
    return store.get_stats()

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "enyal": {
      "command": "python",
      "args": ["-m", "enyal.mcp"],
      "env": {
        "ENYAL_DB_PATH": "~/.enyal/context.db"
      }
    }
  }
}
```

---

## 7. Questions and Answers

### Q1: Why SQLite over alternatives?

SQLite was chosen because:
1. **Zero dependency risk:** Built into Python 3.13 stdlib
2. **Unified storage:** Integrates with sqlite-vec for combined relational + vector storage
3. **Single-file portability:** Trivial backup/restore (copy one file)
4. **Proven reliability:** Most deployed database in the world
5. **Sufficient performance:** Our write pattern (10-100 writes/session) is well within SQLite's capabilities
6. **Cross-platform:** Works identically on macOS, Windows, Linux

DuckDB is OLAP-optimized (wrong workload), LMDB requires two files and fixed map_size, RocksDB has cross-platform build issues and directory-based storage.

### Q2: What's the embedding dimension and why?

**384 dimensions** using all-MiniLM-L6-v2.

Rationale:
- Storage efficient: 100k entries × 384 × 4 bytes = 153MB
- Quality sufficient: Captures semantic meaning for technical content
- Fast inference: ~5ms per embedding
- Model size: 80MB, loads in <1s

768 dimensions (mpnet) would double storage for marginal quality improvement. 256 dimensions (truncated) would reduce quality noticeably.

### Q3: How do you handle embedding model cold-start latency?

1. **Lazy loading:** Model loads only when first embedding is needed
2. **Memory persistence:** Once loaded, model stays in memory for session
3. **Optional pre-loading:** `ENYAL_PRELOAD_MODEL=true` for background warm-up

Cold start is ~800ms on first embedding. Subsequent embeddings are ~5ms. For MCP servers (long-running), this is a one-time cost.

### Q4: What happens with 1M context entries?

At 1M entries:
- **Storage:** 384 dims × 1M × 4 bytes = 1.5GB (manageable)
- **Query latency:** sqlite-vec may slow to ~50-100ms (still acceptable)
- **Memory:** May need to tune SQLite cache, consider mmap

**Mitigation strategies if needed:**
1. Binary quantization: 32x storage reduction (48MB for 1M entries)
2. Secondary hnswlib index: As read-optimized cache
3. Partitioning: Separate databases per workspace

The architecture supports these optimizations without breaking changes.

### Q5: How would you add encryption-at-rest later?

SQLite supports encryption via:

1. **SQLCipher extension:** Drop-in replacement with AES-256
   ```python
   from pysqlcipher3 import dbapi2 as sqlite
   conn = sqlite.connect("encrypted.db")
   conn.execute("PRAGMA key='password'")
   ```

2. **SQLite3 Multiple Ciphers:** Multiple cipher options
   ```python
   conn.execute("PRAGMA cipher='aes256cbc'")
   conn.execute("PRAGMA key='password'")
   ```

**Non-breaking integration:**
- Connection factory accepts optional encryption key
- Existing databases can be migrated with `sqlcipher_export()`
- No schema changes required

```python
def create_connection(db_path: str, encryption_key: str | None = None):
    if encryption_key:
        from pysqlcipher3 import dbapi2 as sqlite
        conn = sqlite.connect(db_path)
        conn.execute(f"PRAGMA key='{encryption_key}'")
    else:
        import sqlite3
        conn = sqlite3.connect(db_path)
    # ... rest of setup
```

---

## 8. Performance Targets

| Metric | Target | Expected |
|--------|--------|----------|
| Cold start (first query) | <500ms | ~800ms (model load) |
| Warm query latency | <50ms p99 | ~10ms |
| Memory footprint (100k entries) | <500MB | ~300MB |
| Write latency | <100ms | ~20ms |
| Database size (100k entries) | <500MB | ~200MB |

### Benchmark Plan

```python
import time
import statistics

def benchmark_recall(store, queries, iterations=100):
    latencies = []
    for _ in range(iterations):
        for query in queries:
            start = time.perf_counter()
            store.recall(query, limit=10)
            latencies.append(time.perf_counter() - start)

    return {
        "p50": statistics.median(latencies) * 1000,
        "p95": statistics.quantiles(latencies, n=20)[18] * 1000,
        "p99": statistics.quantiles(latencies, n=100)[98] * 1000,
    }
```

---

## 9. Future Considerations

### Not In Scope (Current)

1. **Multi-user/team sharing:** Would require auth, conflict resolution
2. **Cloud sync:** Would require service infrastructure
3. **Auto-extraction from code:** Complex NLP/code analysis
4. **Context summarization:** LLM integration for compression

### Extension Points

The architecture supports future additions:

1. **Plugin system:** Custom content types, extractors
2. **Export/import:** JSON/YAML format for portability
3. **Analytics dashboard:** Visualization of context usage
4. **Conflict resolution:** For multi-agent scenarios

---

## Appendix A: Technology Versions

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.13+ | Required for latest features |
| SQLite | 3.45+ | Via stdlib, WAL2 if available |
| sqlite-vec | 0.1.x | Latest stable |
| sentence-transformers | 3.x | For embeddings |
| FastMCP | 2.x | MCP server framework |

## Appendix B: File Layout

```
~/.enyal/
├── context.db          # Main SQLite database (includes vectors)
├── context.db-wal      # WAL file (transient)
├── context.db-shm      # Shared memory (transient)
├── config.toml         # User configuration
└── logs/
    └── enyal.log       # Application logs
```

---

*Document prepared for architecture review. Awaiting approval before Phase 2 implementation.*
