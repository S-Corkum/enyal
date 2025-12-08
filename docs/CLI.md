# Enyal CLI Reference

The Enyal command-line interface provides direct access to the context store for testing, debugging, and scripting.

## Installation

```bash
# Install with pip
pip install enyal

# Install with pipx (isolated environment)
pipx install enyal

# Install with uv
uv add enyal
```

## Global Options

These options are available for all commands:

| Option | Description |
|--------|-------------|
| `--db PATH` | Path to database file (default: `~/.enyal/context.db`) |
| `--json` | Output in JSON format for scripting |
| `-h, --help` | Show help message |

## Commands

### remember

Store new context in the database.

```bash
enyal remember <content> [options]
```

**Arguments:**
- `content` (required): The context to store

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--type` | `-t` | `fact` | Content type: `fact`, `preference`, `decision`, `convention`, `pattern` |
| `--scope` | `-s` | `project` | Scope level: `global`, `workspace`, `project`, `file` |
| `--scope-path` | | (none) | Path for scope context |
| `--tags` | | (none) | Comma-separated tags |

**Examples:**

```bash
# Store a simple fact
enyal remember "The database uses PostgreSQL 15"

# Store a convention with tags
enyal remember "Always use pytest for testing" --type convention --tags testing,pytest

# Store a project-scoped decision
enyal remember "Chose React over Vue for frontend" \
  --type decision \
  --scope project \
  --scope-path /Users/dev/myproject

# Store with JSON output
enyal remember "Use tabs for indentation" --type preference --json
```

**JSON Output:**
```json
{
  "success": true,
  "entry_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### recall

Search for relevant context using semantic search.

```bash
enyal recall <query> [options]
```

**Arguments:**
- `query` (required): Natural language search query

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--limit` | `-n` | `10` | Maximum number of results |
| `--type` | `-t` | (any) | Filter by content type |
| `--scope` | `-s` | (any) | Filter by scope level |
| `--scope-path` | | (none) | Filter by scope path (prefix match) |
| `--min-confidence` | | `0.3` | Minimum confidence threshold (0.0-1.0) |

**Examples:**

```bash
# Basic search
enyal recall "testing framework"

# Limit results
enyal recall "database configuration" --limit 3

# Filter by type
enyal recall "coding standards" --type convention

# Filter by scope
enyal recall "project setup" --scope project --scope-path /Users/dev/myproject

# High confidence only
enyal recall "important decisions" --min-confidence 0.8

# JSON output for scripting
enyal recall "API design" --json
```

**Standard Output:**
```
1. [convention] (score: 0.847)
   Always use pytest for testing in this project
   ID: 550e8400-e29b-41d4-a716-446655440000
   Tags: testing, pytest

2. [fact] (score: 0.723)
   The test suite uses pytest-asyncio for async tests
   ID: 550e8400-e29b-41d4-a716-446655440001
   Tags: testing, async
```

**JSON Output:**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "content": "Always use pytest for testing in this project",
    "type": "convention",
    "scope": "project",
    "score": 0.847,
    "confidence": 1.0
  },
  {
    "id": "550e8400-e29b-41d4-a716-446655440001",
    "content": "The test suite uses pytest-asyncio for async tests",
    "type": "fact",
    "scope": "project",
    "score": 0.723,
    "confidence": 1.0
  }
]
```

---

### get

Retrieve a specific entry by ID.

```bash
enyal get <entry_id>
```

**Arguments:**
- `entry_id` (required): UUID of the entry to retrieve

**Examples:**

```bash
# Get entry details
enyal get 550e8400-e29b-41d4-a716-446655440000

# JSON output
enyal get 550e8400-e29b-41d4-a716-446655440000 --json
```

**Standard Output:**
```
ID:         550e8400-e29b-41d4-a716-446655440000
Content:    Always use pytest for testing in this project
Type:       convention
Scope:      project
Scope path: /Users/dev/myproject
Confidence: 100.00%
Tags:       testing, pytest
Created:    2024-12-08T10:30:00
Updated:    2024-12-08T10:30:00
Accessed:   5 times
```

**JSON Output:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "content": "Always use pytest for testing in this project",
  "type": "convention",
  "scope": "project",
  "scope_path": "/Users/dev/myproject",
  "confidence": 1.0,
  "tags": ["testing", "pytest"],
  "created_at": "2024-12-08T10:30:00",
  "updated_at": "2024-12-08T10:30:00",
  "access_count": 5,
  "is_deprecated": false
}
```

---

### forget

Remove or deprecate context from the database.

```bash
enyal forget <entry_id> [options]
```

**Arguments:**
- `entry_id` (required): UUID of the entry to remove

**Options:**

| Option | Description |
|--------|-------------|
| `--hard` | Permanently delete instead of soft-delete (deprecate) |

**Examples:**

```bash
# Soft delete (deprecate) - entry is excluded from search but preserved
enyal forget 550e8400-e29b-41d4-a716-446655440000

# Hard delete - permanently remove
enyal forget 550e8400-e29b-41d4-a716-446655440000 --hard

# With JSON output
enyal forget 550e8400-e29b-41d4-a716-446655440000 --json
```

**Standard Output:**
```
Entry 550e8400-e29b-41d4-a716-446655440000 has been deprecated
```

**JSON Output:**
```json
{
  "success": true
}
```

---

### stats

Display usage statistics and health metrics.

```bash
enyal stats
```

**Examples:**

```bash
# Standard output
enyal stats

# JSON output
enyal stats --json
```

**Standard Output:**
```
Enyal Context Store Statistics
========================================
Total entries:      42
Active entries:     40
Deprecated entries: 2
Average confidence: 94.50%
Storage size:       156.3 KB

By type:
  convention: 15
  decision: 8
  fact: 12
  pattern: 5
  preference: 2

By scope:
  file: 3
  global: 5
  project: 30
  workspace: 4
```

**JSON Output:**
```json
{
  "total_entries": 42,
  "active_entries": 40,
  "deprecated_entries": 2,
  "entries_by_type": {
    "convention": 15,
    "decision": 8,
    "fact": 12,
    "pattern": 5,
    "preference": 2
  },
  "entries_by_scope": {
    "file": 3,
    "global": 5,
    "project": 30,
    "workspace": 4
  },
  "avg_confidence": 0.945,
  "storage_size_bytes": 160051
}
```

---

### serve

Run the MCP server for integration with AI coding assistants.

```bash
enyal serve [options]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--preload` | Pre-load embedding model at startup |
| `--log-level` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

**Examples:**

```bash
# Basic server (model loads on first use)
enyal serve

# Pre-load model for faster first query
enyal serve --preload

# Debug mode
enyal serve --log-level DEBUG

# Custom database
enyal serve --db ~/projects/myproject/.enyal/context.db
```

**Note:** The serve command runs the MCP server using stdio transport. It's designed to be invoked by MCP clients (Claude Code, Cursor, etc.) rather than run directly.

---

## Environment Variables

The CLI respects these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENYAL_DB_PATH` | `~/.enyal/context.db` | Database file location |
| `ENYAL_PRELOAD_MODEL` | `false` | Pre-load embedding model |
| `ENYAL_LOG_LEVEL` | `INFO` | Logging level |

**Example:**
```bash
export ENYAL_DB_PATH=~/projects/.enyal/context.db
enyal stats  # Uses custom database path
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Error (entry not found, invalid input, etc.) |

---

## Scripting Examples

### Batch Import

```bash
#!/bin/bash
# Import conventions from a file

while IFS= read -r convention; do
  enyal remember "$convention" --type convention --json
done < conventions.txt
```

### Export All Context

```bash
#!/bin/bash
# Export all context as JSON

enyal recall "" --limit 1000 --json > context_backup.json
```

### Check for Specific Context

```bash
#!/bin/bash
# Check if a testing convention exists

result=$(enyal recall "testing framework" --type convention --json)
if echo "$result" | jq -e 'length > 0' > /dev/null; then
  echo "Testing convention found"
else
  echo "No testing convention - consider adding one"
fi
```

### Integration with Git Hooks

```bash
#!/bin/bash
# .git/hooks/pre-commit
# Check conventions before commit

conventions=$(enyal recall "commit conventions" --type convention --json)
# ... validate commit message against conventions
```

### Project Setup Script

```bash
#!/bin/bash
# Initialize enyal with project conventions

PROJECT_PATH=$(pwd)

enyal remember "Use pytest for all testing" \
  --type convention \
  --scope project \
  --scope-path "$PROJECT_PATH" \
  --tags testing

enyal remember "Follow PEP 8 style guide" \
  --type convention \
  --scope project \
  --scope-path "$PROJECT_PATH" \
  --tags style

enyal remember "Chose FastAPI for backend framework" \
  --type decision \
  --scope project \
  --scope-path "$PROJECT_PATH" \
  --tags backend,api

echo "Project conventions initialized"
enyal stats
```

---

## Tips

### Use JSON for Scripting
Always use `--json` flag when parsing output programmatically:
```bash
enyal recall "query" --json | jq '.[0].content'
```

### Organize with Tags
Use consistent tags for easier filtering:
```bash
enyal remember "..." --tags backend,api,rest
enyal remember "..." --tags frontend,react,hooks
```

### Scope Appropriately
- `global`: Cross-project preferences (editor settings, personal conventions)
- `workspace`: Shared conventions across related projects
- `project`: Project-specific decisions and conventions
- `file`: File-specific notes and context

### Regular Maintenance
Periodically review and clean up outdated context:
```bash
# Check stats
enyal stats

# Review old entries
enyal recall "old decision" --min-confidence 0.5

# Deprecate outdated entries
enyal forget <entry-id>
```
