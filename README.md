# Enyal

**Persistent, queryable memory for AI coding agents.**

Enyal gives AI agents like Claude Code durable context that survives session restarts. Every conversation becomes accumulated institutional knowledge—facts, preferences, decisions, and conventions that persist and grow.

## Features

- **Persistent Memory**: Context survives restarts, crashes, and process termination
- **Semantic Search**: Find relevant context using natural language queries (384-dim embeddings via all-MiniLM-L6-v2)
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

| Tool | Description |
|------|-------------|
| **enyal_remember** | Store new context with optional duplicate detection and merging |
| **enyal_recall** | Semantic search for relevant context with filtering by scope and type |
| **enyal_recall_by_scope** | Scope-aware search that automatically finds context relevant to the current file/project |
| **enyal_forget** | Remove or deprecate context (soft-delete by default, hard-delete optional) |
| **enyal_update** | Update existing entries (content, confidence, tags) |
| **enyal_get** | Retrieve a specific entry by ID with full metadata |
| **enyal_stats** | Get usage statistics and health metrics |

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

# Update context
store.update(entry_id, confidence=0.9, tags=["testing", "pytest", "unit-tests"])

# Get specific entry
entry = store.get(entry_id)

# Get statistics
stats = store.stats()
print(f"Total entries: {stats.total_entries}")
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
