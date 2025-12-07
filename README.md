# Enyal

**Persistent, queryable memory for AI coding agents.**

Enyal gives AI agents like Claude Code durable context that survives session restarts. Every conversation becomes accumulated institutional knowledge—facts, preferences, decisions, and conventions that persist and grow.

## Features

- **Persistent Memory**: Context survives restarts, crashes, and process termination
- **Semantic Search**: Find relevant context using natural language queries
- **Hierarchical Scoping**: Global → workspace → project → file context inheritance
- **Fully Offline**: Zero network calls during operation
- **Cross-Platform**: macOS (Intel + Apple Silicon) and Windows 10/11
- **MCP Compatible**: Works with any MCP client (Claude Desktop, Claude Code, etc.)

## Installation

```bash
# Using uv (recommended)
uv add enyal

# Using pip
pip install enyal
```

## Quick Start

### As an MCP Server

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json`):

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

### Available Tools

- **enyal_remember**: Store new context with metadata
- **enyal_recall**: Semantic search for relevant context
- **enyal_forget**: Remove or deprecate context
- **enyal_stats**: Usage and health metrics

### As a Python Library

```python
from enyal import ContextStore, ContextType, ScopeLevel

# Initialize store
store = ContextStore("~/.enyal/context.db")

# Remember something
entry_id = store.remember(
    content="Always use pytest for testing in this project",
    content_type=ContextType.CONVENTION,
    scope_level=ScopeLevel.PROJECT,
    scope_path="/Users/dev/myproject",
    tags=["testing", "pytest"]
)

# Recall relevant context
results = store.recall(
    query="how should I write tests?",
    limit=5,
    min_confidence=0.5
)

for result in results:
    print(f"{result.score:.2f}: {result.entry.content}")
```

## Architecture

Enyal uses a unified SQLite database with:

- **Relational storage** for metadata and attributes
- **sqlite-vec** for vector similarity search
- **FTS5** for keyword search
- **WAL mode** for concurrent access

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design decisions.

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENYAL_DB_PATH` | `~/.enyal/context.db` | Database file location |
| `ENYAL_PRELOAD_MODEL` | `false` | Pre-load embedding model at startup |
| `ENYAL_LOG_LEVEL` | `INFO` | Logging level |

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

Run benchmarks:
```bash
uv run python benchmarks/benchmark_performance.py
```

## License

MIT
