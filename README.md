# Enyal

**Persistent, queryable memory for AI coding agents.**

Enyal gives AI agents like Claude Code durable context that survives session restarts. Every conversation becomes accumulated institutional knowledge—facts, preferences, decisions, and conventions that persist and grow.

## Features

- **Persistent Memory**: Context survives restarts, crashes, and process termination
- **Semantic Search**: Find relevant context using natural language queries (384-dim embeddings via all-MiniLM-L6-v2)
- **Knowledge Graph**: Link related entries with relationships (supersedes, depends_on, conflicts_with, relates_to)
- **Validity Tracking**: Automatically filter superseded entries and flag conflicts
- **Entry Versioning**: Full history of changes with automatic version creation
- **Usage Analytics**: Track how context is accessed and used over time
- **Health Monitoring**: Get insights into stale, orphan, and conflicting entries
- **Hierarchical Scoping**: Global → workspace → project → file context inheritance
- **Fully Offline**: Zero network calls during operation
- **Cross-Platform**: macOS (Intel + Apple Silicon), Linux, and Windows
- **MCP Compatible**: Works with Claude Code, Cursor, Windsurf, Kiro, and any MCP client

## Quick Start

Get up and running in under 2 minutes:

### 1. Install

```bash
# Using uv (recommended)
uv pip install enyal --system

# Or using pip
pip install enyal
```

### 2. Configure Your MCP Client

**Universal configuration** (works with Claude Code, Cursor, Windsurf, Kiro):

```json
{
  "mcpServers": {
    "enyal": {
      "command": "uvx",
      "args": ["enyal", "serve"]
    }
  }
}
```

**For macOS Intel users** (requires Python 3.11 or 3.12):

```json
{
  "mcpServers": {
    "enyal": {
      "command": "uvx",
      "args": ["--python", "3.12", "enyal", "serve"]
    }
  }
}
```

### 3. Start Using

```
You: Remember that this project uses pytest for all testing
Assistant: [calls enyal_remember] Stored context about testing framework

You: What testing framework should I use?
Assistant: [calls enyal_recall] Based on stored context, this project uses pytest.
```

## Platform Support

| Platform | Python 3.11 | Python 3.12 | Python 3.13 |
|----------|-------------|-------------|-------------|
| macOS Apple Silicon | Supported | Supported | Supported |
| macOS Intel | Supported | Supported | Not supported* |
| Linux | Supported | Supported | Supported |
| Windows | Supported | Supported | Supported |

*macOS Intel + Python 3.13 is not supported due to PyTorch ecosystem constraints.

## Installation Methods

### Method 1: uv (Recommended)

```bash
# Install globally
uv pip install enyal --system

# Run server
enyal serve

# With model preloading for faster first query
enyal serve --preload
```

### Method 2: pip

```bash
# Install globally
pip install enyal

# Run server
enyal serve
```

### Method 3: pipx

```bash
# Install in isolated environment
pipx install enyal

# Run server
enyal serve
```

### Method 4: uvx (Run without installing)

For ephemeral execution without permanent installation:

```bash
# Most platforms (auto-selects Python)
uvx enyal serve

# macOS Intel (explicit Python version)
uvx --python 3.12 enyal serve
```

Note: `uvx` runs the package in a temporary environment each time. For persistent installation, use Method 1, 2, or 3.

## MCP Integration

Enyal works with any MCP-compatible client. The configuration is the same across platforms—only the command may vary for macOS Intel.

### Claude Code

**File locations:**
- Project: `.mcp.json` (in project root)
- User: `~/.claude/.mcp.json`

**Standard configuration:**
```json
{
  "mcpServers": {
    "enyal": {
      "command": "uvx",
      "args": ["enyal", "serve"],
      "env": {
        "ENYAL_DB_PATH": "~/.enyal/context.db"
      }
    }
  }
}
```

**macOS Intel configuration:**
```json
{
  "mcpServers": {
    "enyal": {
      "command": "uvx",
      "args": ["--python", "3.12", "enyal", "serve"],
      "env": {
        "ENYAL_DB_PATH": "~/.enyal/context.db"
      }
    }
  }
}
```

**CLI setup:**
```bash
# Standard
claude mcp add-json enyal '{"command":"uvx","args":["enyal","serve"]}'

# macOS Intel
claude mcp add-json enyal '{"command":"uvx","args":["--python","3.12","enyal","serve"]}'
```

### Claude Desktop

**File locations:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

**Configuration:**
```json
{
  "mcpServers": {
    "enyal": {
      "command": "uvx",
      "args": ["enyal", "serve"],
      "env": {
        "ENYAL_DB_PATH": "~/.enyal/context.db"
      }
    }
  }
}
```

### Cursor

**File locations:**
- Global: `~/.cursor/mcp.json`
- Project: `.cursor/mcp.json`

**Configuration:**
```json
{
  "mcpServers": {
    "enyal": {
      "command": "uvx",
      "args": ["enyal", "serve"],
      "env": {
        "ENYAL_DB_PATH": "~/.enyal/context.db"
      }
    }
  }
}
```

**UI setup:** File → Preferences → Cursor Settings → MCP

### Windsurf

**File location:** `~/.codeium/windsurf/mcp_config.json`

**Configuration:**
```json
{
  "mcpServers": {
    "enyal": {
      "command": "uvx",
      "args": ["enyal", "serve"],
      "env": {
        "ENYAL_DB_PATH": "~/.enyal/context.db"
      }
    }
  }
}
```

**UI setup:** Windsurf Settings → Cascade → MCP, or use the Plugin Store

### Kiro

**File locations:**
- Global: `~/.kiro/settings/mcp.json`
- Project: `.kiro/settings/mcp.json`

**Configuration:**
```json
{
  "mcpServers": {
    "enyal": {
      "command": "uvx",
      "args": ["enyal", "serve"],
      "env": {
        "ENYAL_DB_PATH": "~/.enyal/context.db"
      },
      "autoApprove": ["enyal_recall", "enyal_stats", "enyal_get"]
    }
  }
}
```

**UI setup:** Click the Kiro ghost tab → MCP Servers → "+"

See [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) for detailed platform-specific guides.

## Available Tools

### Core Tools

| Tool | Description |
|------|-------------|
| **enyal_remember** | Store new context with optional duplicate detection, conflict detection, and auto-linking |
| **enyal_recall** | Semantic search with validity filtering (excludes superseded entries by default) |
| **enyal_recall_by_scope** | Scope-aware search that automatically finds context relevant to the current file/project |
| **enyal_forget** | Remove or deprecate context (soft-delete by default, hard-delete optional) |
| **enyal_update** | Update existing entries (content, confidence, tags) - automatically creates version |
| **enyal_get** | Retrieve a specific entry by ID with full metadata |
| **enyal_stats** | Get usage statistics and health metrics |

### Knowledge Graph Tools

| Tool | Description |
|------|-------------|
| **enyal_link** | Create relationships between entries (relates_to, supersedes, depends_on, conflicts_with) |
| **enyal_unlink** | Remove a relationship between entries |
| **enyal_edges** | Get all relationships for an entry |
| **enyal_traverse** | Walk the knowledge graph from an entry |
| **enyal_impact** | Find all entries that depend on a given entry |

### Intelligence Tools

| Tool | Description |
|------|-------------|
| **enyal_health** | Get graph health metrics (stale, orphan, conflicting entries) |
| **enyal_review** | Get entries needing review (stale, orphan, or conflicted) |
| **enyal_history** | Get version history for an entry |
| **enyal_analytics** | Get usage analytics (recall frequency, top accessed entries) |

### Content Types

| Type | Use For | Example |
|------|---------|---------|
| `fact` | Objective information | "The database uses PostgreSQL 15" |
| `preference` | User/team preferences | "Prefer tabs over spaces" |
| `decision` | Recorded decisions | "Chose React over Vue for frontend" |
| `convention` | Coding standards | "All API endpoints follow REST naming" |
| `pattern` | Code patterns | "Error handling uses Result<T, E> pattern" |

### Scope Levels

| Scope | Applies To | Example Path |
|-------|------------|--------------|
| `global` | All projects | (none) |
| `workspace` | Directory of projects | `/Users/dev/projects` |
| `project` | Single project | `/Users/dev/myproject` |
| `file` | Specific file | `/Users/dev/myproject/src/auth.py` |

### Relationship Types

| Type | Use For | Example |
|------|---------|---------|
| `relates_to` | General semantic relationship | "Testing guide" relates to "pytest conventions" |
| `supersedes` | Entry A replaces entry B | New decision supersedes old decision |
| `depends_on` | Entry A requires entry B | Feature depends on architecture decision |
| `conflicts_with` | Entries contradict each other | "Use tabs" conflicts with "Use spaces" |

## CLI Usage

Enyal provides a command-line interface for direct interaction:

```bash
# Store context
enyal remember "Always use pytest for testing" --type convention --scope project

# Search context
enyal recall "testing framework" --limit 5

# Get entry details
enyal get <entry-id>

# View statistics
enyal stats

# Remove context
enyal forget <entry-id>

# Run MCP server
enyal serve --preload
```

**Options:**
- `--db PATH` — Custom database path
- `--json` — Output in JSON format

See [docs/CLI.md](docs/CLI.md) for complete CLI reference.

## Python Library

```python
from enyal.core.store import ContextStore
from enyal.core.retrieval import RetrievalEngine
from enyal.models.context import ContextType, ScopeLevel

# Initialize store
store = ContextStore("~/.enyal/context.db")
retrieval = RetrievalEngine(store)

# Remember something
entry_id = store.remember(
    content="Always use pytest for testing in this project",
    content_type=ContextType.CONVENTION,
    scope_level=ScopeLevel.PROJECT,
    scope_path="/Users/dev/myproject",
    tags=["testing", "pytest"]
)

# Remember with duplicate detection
result = store.remember(
    content="Use pytest for all testing",  # Similar to existing
    check_duplicate=True,           # Enable duplicate checking
    duplicate_threshold=0.85,       # Similarity threshold
    on_duplicate="reject"           # "reject", "merge", or "store"
)
# Returns dict: {"entry_id": "...", "action": "existing", "similarity": 0.92}

# Recall relevant context (hybrid semantic + keyword search)
results = retrieval.search(
    query="how should I write tests?",
    limit=5,
    min_confidence=0.5
)

for result in results:
    print(f"{result.score:.2f}: {result.entry.content}")

# Scope-aware search (file → project → workspace → global)
results = retrieval.search_by_scope(
    query="testing conventions",
    file_path="/Users/dev/myproject/src/auth.py",
    limit=5
)

# Find similar entries (useful for deduplication checks)
similar = store.find_similar(
    content="pytest testing conventions",
    threshold=0.8,
    limit=3
)

# Update context (automatically creates a version record)
store.update(entry_id, confidence=0.9, tags=["testing", "pytest", "unit-tests"])

# Get specific entry
entry = store.get(entry_id)

# Get statistics
stats = store.stats()
print(f"Total entries: {stats.total_entries}")
```

### Knowledge Graph

```python
from enyal.models.context import EdgeType

# Create entries
old_decision = store.remember(content="Use Python 3.10", content_type="decision")
new_decision = store.remember(content="Use Python 3.13", content_type="decision")

# Link entries (new supersedes old)
store.link(new_decision, old_decision, EdgeType.SUPERSEDES)

# Search with validity filtering (superseded entries excluded by default)
results = retrieval.search("Python version", exclude_superseded=True)

# Include superseded entries with metadata
results = retrieval.search("Python version", exclude_superseded=False)
for r in results:
    if r.is_superseded:
        print(f"SUPERSEDED: {r.entry.content} (by {r.superseded_by})")

# Traverse the graph
related = store.traverse(new_decision, max_depth=2)

# Find what depends on an entry
dependents = store.traverse(new_decision, direction="incoming",
                           edge_types=[EdgeType.DEPENDS_ON])
```

### Versioning & History

```python
# Every remember() creates an initial version
entry_id = store.remember(content="Initial approach", content_type="decision")

# Every update() creates a new version
store.update(entry_id, content="Revised approach")
store.update(entry_id, content="Final approach")

# Get version history
history = store.get_history(entry_id)
for version in history:
    print(f"v{version['version']}: {version['change_type']} - {version['content']}")
# Output:
# v3: updated - Final approach
# v2: updated - Revised approach
# v1: created - Initial approach
```

### Analytics & Health

```python
# Track usage (called automatically during recall)
store.track_usage(entry_id, "recall", query="approach", result_rank=1)

# Get analytics
analytics = store.get_analytics(days=30)
print(f"Top recalled: {analytics['top_recalled']}")

# Health check
health = store.health_check()
print(f"Health score: {health['health_score']:.0%}")
print(f"Stale entries: {health['stale_entries']}")
print(f"Orphan entries: {health['orphan_entries']}")
print(f"Conflicts: {health['unresolved_conflicts']}")

# Get entries needing review
stale = store.get_stale_entries(days_old=180)
orphans = store.get_orphan_entries()
conflicts = store.get_conflicted_entries()
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENYAL_DB_PATH` | `~/.enyal/context.db` | Database file location |
| `ENYAL_PRELOAD_MODEL` | `false` | Pre-load embedding model at startup |
| `ENYAL_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `ENYAL_SSL_CERT_FILE` | (system) | Path to CA certificate bundle (for corporate networks) |
| `ENYAL_SSL_VERIFY` | `true` | Enable/disable SSL verification (set `false` only as last resort) |
| `ENYAL_MODEL_PATH` | (none) | Path to local pre-downloaded model |
| `ENYAL_OFFLINE_MODE` | `false` | Prevent network calls (use with cached/local model) |

### Database Location

The default database is stored at `~/.enyal/context.db`. This single SQLite file contains:
- All context entries and metadata
- Vector embeddings for semantic search
- Full-text search index

## Troubleshooting

### Installation Fails on macOS Intel

**Symptom:** Error about torch/PyTorch wheels not found

**Cause:** PyTorch doesn't provide wheels for macOS Intel + Python 3.13

**Solution:** Use Python 3.11 or 3.12:
```bash
# Install with specific Python version
uv pip install enyal --python 3.12 --system
```

### MCP Server Not Connecting

1. **Check uvx is available:**
   ```bash
   uvx --version
   ```

2. **Test server manually:**
   ```bash
   uvx enyal serve
   # Should start without errors, waiting for MCP protocol
   ```

3. **Enable debug logging:**
   ```json
   {
     "mcpServers": {
       "enyal": {
         "command": "uvx",
         "args": ["enyal", "serve", "--log-level", "DEBUG"]
       }
     }
   }
   ```

4. **Check server status:**
   - Claude Code: `/mcp` command
   - Cursor: Settings → MCP → check status
   - Windsurf: Cascade → Plugins
   - Kiro: Ghost tab → MCP Servers

### Slow First Query

The first query loads the embedding model (~80MB). This takes ~1-2 seconds. Subsequent queries are fast (~34ms).

**To pre-load the model at startup:**
```json
{
  "mcpServers": {
    "enyal": {
      "command": "uvx",
      "args": ["enyal", "serve", "--preload"]
    }
  }
}
```

### Database Locked Error

If you see "database is locked" errors, ensure only one MCP server instance is running per database file. Use different `ENYAL_DB_PATH` values for different projects if needed.

### Permission Errors

On macOS/Linux, ensure the database directory exists and is writable:
```bash
mkdir -p ~/.enyal
chmod 755 ~/.enyal
```

### SSL Certificate Errors (Corporate Networks)

**Symptom:** Error containing "SSL: CERTIFICATE_VERIFY_FAILED" or "self signed certificate in certificate chain"

**Cause:** Corporate networks with SSL inspection (Zscaler, BlueCoat, etc.) inject enterprise CA certificates that Python doesn't recognize by default.

**Quick Fix:**
```bash
# Option 1: Point to your corporate CA bundle (recommended)
export ENYAL_SSL_CERT_FILE=/path/to/corporate-ca-bundle.crt
enyal model download

# Option 2: Pre-download model on unrestricted network, then use offline
export ENYAL_OFFLINE_MODE=true
```

**For MCP configuration:**
```json
{
  "mcpServers": {
    "enyal": {
      "command": "uvx",
      "args": ["enyal", "serve"],
      "env": {
        "ENYAL_SSL_CERT_FILE": "/path/to/corporate-ca-bundle.crt"
      }
    }
  }
}
```

**Check your SSL configuration:**
```bash
enyal model status
```

See [docs/SSL_TROUBLESHOOTING.md](docs/SSL_TROUBLESHOOTING.md) for detailed troubleshooting guide.

## Architecture

Enyal uses a unified SQLite database with:

- **Relational storage** for metadata and attributes
- **sqlite-vec** for vector similarity search (384-dim embeddings)
- **FTS5** for keyword search
- **Knowledge graph** with typed edges (supersedes, depends_on, conflicts_with, relates_to)
- **Version history** for change tracking
- **Usage analytics** for access patterns
- **WAL mode** for concurrent access

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design decisions.

## Development

```bash
# Clone repository
git clone https://github.com/seancorkum/enyal.git
cd enyal

# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Type checking
uv run mypy src/enyal

# Linting
uv run ruff check src/enyal
```

## Performance

Benchmarked on Intel Mac with Python 3.12:

| Metric | Target (p95) | Measured (p95) | Status |
|--------|--------------|----------------|--------|
| Cold start (model load + first query) | <2000ms | ~1500ms | ✓ |
| Warm query latency | <50ms | ~34ms | ✓ |
| Write latency | <50ms | ~34ms | ✓ |
| Concurrent reads (4 threads) | <150ms | ~85ms | ✓ |
| Memory (100k entries estimated) | <500MB | ~35MB | ✓ |

**Embedding model:** [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (22M params, 384 dimensions)

Run benchmarks:
```bash
uv run python benchmarks/benchmark_performance.py
```

## License

MIT
