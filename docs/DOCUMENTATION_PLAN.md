# Enyal Documentation Enhancement Plan

> **Version:** 1.0.0
> **Created:** December 2024
> **Status:** Ready for Implementation

## Executive Summary

This plan outlines comprehensive enhancements to the Enyal project documentation to improve accuracy, expand platform coverage, and provide better onboarding for users across Claude Code, Cursor, Windsurf, and Kiro.

---

## 1. Documentation Accuracy Audit

### 1.1 Issues Identified in README.md

| Line | Issue | Fix Required |
|------|-------|--------------|
| 30 | Claude Desktop path is macOS-only | Add Windows path |
| 37-38 | Only shows `python -m enyal.mcp` | Add `enyal serve` and `uvx` options |
| 46-52 | Lists only 4 tools | Add `enyal_update` and `enyal_get` |
| 56-79 | Python example may use internal API | Verify public API usage |

### 1.2 Verified Correct Information

- Database path default: `~/.enyal/context.db` ✓
- Environment variables: `ENYAL_DB_PATH`, `ENYAL_PRELOAD_MODEL`, `ENYAL_LOG_LEVEL` ✓
- Performance metrics match implementation ✓
- Architecture link is valid ✓

### 1.3 Entry Points (from pyproject.toml)

```toml
# CLI entry point
enyal = "enyal.cli.main:main"

# MCP server entry point (via mcp.servers)
enyal = "enyal.mcp.server:mcp"
```

**Valid MCP server invocation methods:**
1. `python -m enyal.mcp` (current docs) ✓
2. `enyal serve` (CLI command) ✓
3. `uvx enyal serve` (once published to PyPI)

---

## 2. MCP Integration Guides

### 2.1 Claude Code CLI Integration

**Source:** [Claude Code MCP Documentation](https://docs.claude.com/en/docs/claude-code/mcp)

**Configuration File Locations:**
- Project-level: `.mcp.json` (in project root)
- User-level: `~/.claude/.mcp.json`
- Settings: `~/.claude.json`

**Example Configuration:**

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

**Alternative using uvx (recommended for installed packages):**

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

**CLI Commands:**
```bash
# Add via CLI
claude mcp add-json enyal '{"command":"python","args":["-m","enyal.mcp"]}'

# List configured servers
claude mcp list

# Check server status (within Claude Code)
/mcp
```

### 2.2 Claude Desktop Integration

**Source:** [Model Context Protocol - Local Servers](https://modelcontextprotocol.io/docs/develop/connect-local-servers)

**Configuration File Locations:**
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

**Example Configuration:**

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

### 2.3 Cursor IDE Integration

**Source:** [Cursor MCP Documentation](https://docs.cursor.com/context/model-context-protocol)

**Configuration File Locations:**
- **Global:** `~/.cursor/mcp.json`
- **Project:** `.cursor/mcp.json`

**Example Configuration:**

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

**Setup via UI:**
1. Navigate to File → Preferences → Cursor Settings
2. Select the "MCP" option
3. Add server configuration

**Notes:**
- Resources are not yet supported in Cursor (tools only)
- SSH/remote development environments may have limitations

### 2.4 Windsurf IDE Integration

**Source:** [Windsurf Cascade MCP Integration](https://docs.windsurf.com/windsurf/cascade/mcp)

**Configuration File Location:**
- `~/.codeium/windsurf/mcp_config.json`

**Example Configuration:**

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

**Setup via UI:**
1. Click Windsurf Settings (bottom right) or `Cmd+Shift+P` / `Ctrl+Shift+P`
2. Type "Open Windsurf Settings"
3. Navigate to Cascade → MCP section
4. Enable MCP and add configuration

**Alternative: Plugin Store**
- Click Plugins icon in Cascade panel top-right
- Search for or manually add MCP server

### 2.5 Kiro IDE Integration

**Source:** [Kiro MCP Configuration](https://kiro.dev/docs/mcp/configuration/)

**Configuration File Locations:**
- **Global:** `~/.kiro/settings/mcp.json`
- **Project:** `.kiro/settings/mcp.json`

**Example Configuration:**

```json
{
  "mcpServers": {
    "enyal": {
      "command": "python",
      "args": ["-m", "enyal.mcp"],
      "env": {
        "ENYAL_DB_PATH": "${HOME}/.enyal/context.db"
      },
      "disabled": false,
      "autoApprove": ["enyal_recall", "enyal_stats"]
    }
  }
}
```

**Setup via UI:**
1. Click the Kiro "ghost" tab
2. Look for "MCP Servers" in the list
3. Click "+" to add a new MCP server

**Notes:**
- Kiro supports environment variable expansion (`${VAR}` syntax)
- Auto-approve can be configured per-tool
- Supports both local (stdio) and remote (HTTP) servers

---

## 3. Complete Tool Reference

### Current Tool Documentation Gap

The README only documents 4 tools. Here is the complete list of 6 tools:

| Tool | Description | Status in README |
|------|-------------|------------------|
| `enyal_remember` | Store new context with metadata | ✓ Documented |
| `enyal_recall` | Semantic search for relevant context | ✓ Documented |
| `enyal_forget` | Remove or deprecate context | ✓ Documented |
| `enyal_update` | Update existing context entry | ✗ Missing |
| `enyal_stats` | Usage and health metrics | ✓ Documented |
| `enyal_get` | Get specific entry by ID | ✗ Missing |

### Tool Documentation to Add

**enyal_update:**
```
Update an existing context entry.

Parameters:
- entry_id (required): ID of the entry to update
- content (optional): New content (regenerates embedding)
- confidence (optional): New confidence score (0.0-1.0)
- tags (optional): New tags (replaces existing)

Use cases:
- Correct or refine stored content
- Adjust confidence scores as context evolves
- Update tag categorization
```

**enyal_get:**
```
Get a specific context entry by ID.

Parameters:
- entry_id (required): ID of the entry to retrieve

Returns full details including all metadata:
- Content and type
- Scope and path
- Confidence score
- Tags and metadata
- Timestamps (created, updated, accessed)
- Access count
- Deprecation status
```

---

## 4. README.md Restructure

### Proposed Structure

```markdown
# Enyal

**Persistent, queryable memory for AI coding agents.**

## Features
(expand to include all 6 tools)

## Quick Start (NEW - < 2 min to first use)

### Install
```bash
pip install enyal  # or: uv add enyal
```

### Configure (Claude Code example)
Create `.mcp.json` in your project:
```json
{
  "mcpServers": {
    "enyal": {
      "command": "python",
      "args": ["-m", "enyal.mcp"]
    }
  }
}
```

### Use
```
You: @enyal remember this project uses pytest for testing
You: @enyal what testing framework do we use?
```

## Installation
- uv (recommended)
- pip
- pipx (CLI-only)

## MCP Integration
### Claude Code
### Claude Desktop
### Cursor
### Windsurf
### Kiro

## Available Tools
(all 6 tools with descriptions)

## CLI Usage (NEW)
(link to docs/CLI.md)

## Python Library Usage
(keep existing, verify imports)

## Configuration
(keep existing, expand env vars table)

## Troubleshooting (NEW)
- Common issues and solutions
- Debug logging
- Health check command

## Architecture
(link to ARCHITECTURE.md)

## Development
(keep existing)

## Performance
(keep existing)

## License
```

---

## 5. New Documentation Files

### 5.1 docs/INTEGRATIONS.md

Detailed integration guide for each platform including:
- Complete configuration examples
- Platform-specific notes and limitations
- Troubleshooting per platform
- Version compatibility matrix
- Screenshots where helpful

### 5.2 docs/CLI.md

Full CLI reference including:
- All commands with options
- JSON output format documentation
- Exit codes
- Scripting examples
- Automation patterns

**Commands to document:**
```bash
enyal remember <content> [options]
enyal recall <query> [options]
enyal forget <entry_id> [options]
enyal get <entry_id> [options]
enyal stats [options]
enyal serve [options]
```

### 5.3 docs/EXAMPLES.md

Practical workflow examples:
1. **Project Conventions Workflow**
   - Storing coding standards
   - Recalling during code review

2. **Decision Log Workflow**
   - Recording architecture decisions
   - Querying past decisions

3. **Multi-Project Knowledge Base**
   - Using workspace/global scopes
   - Cross-project context sharing

4. **Team Onboarding Workflow**
   - Capturing institutional knowledge
   - Automated context for new contributors

---

## 6. Implementation Checklist

### Phase 1: README.md Updates (Priority: High)

- [ ] Add Quick Start section (< 2 min)
- [ ] Add missing tools (enyal_update, enyal_get)
- [ ] Add Claude Code CLI configuration
- [ ] Update Claude Desktop paths (add Windows)
- [ ] Add Cursor IDE configuration
- [ ] Add Windsurf IDE configuration
- [ ] Add Kiro IDE configuration
- [ ] Add CLI Usage section
- [ ] Add Troubleshooting section
- [ ] Verify Python library example

### Phase 2: New Documentation Files (Priority: Medium)

- [ ] Create docs/INTEGRATIONS.md
- [ ] Create docs/CLI.md
- [ ] Create docs/EXAMPLES.md

### Phase 3: Verification (Priority: High)

- [ ] Test all configuration examples
- [ ] Verify CLI commands work as documented
- [ ] Test Python library example
- [ ] Review performance metrics accuracy
- [ ] Validate all external links

---

## 7. Configuration Examples Summary

### Quick Reference - All Platforms

**Claude Code (.mcp.json):**
```json
{"mcpServers":{"enyal":{"command":"python","args":["-m","enyal.mcp"]}}}
```

**Claude Desktop (claude_desktop_config.json):**
```json
{"mcpServers":{"enyal":{"command":"python","args":["-m","enyal.mcp"]}}}
```

**Cursor (.cursor/mcp.json):**
```json
{"mcpServers":{"enyal":{"command":"python","args":["-m","enyal.mcp"]}}}
```

**Windsurf (mcp_config.json):**
```json
{"mcpServers":{"enyal":{"command":"python","args":["-m","enyal.mcp"]}}}
```

**Kiro (.kiro/settings/mcp.json):**
```json
{"mcpServers":{"enyal":{"command":"python","args":["-m","enyal.mcp"]}}}
```

> **Note:** All platforms use the same JSON schema. The key difference is the file location and optional platform-specific features (e.g., Kiro's `autoApprove`).

---

## 8. Research Sources

### Official Documentation

- **Claude Code MCP:** https://docs.claude.com/en/docs/claude-code/mcp
- **MCP Protocol:** https://modelcontextprotocol.io/
- **Cursor MCP:** https://docs.cursor.com/context/model-context-protocol
- **Windsurf MCP:** https://docs.windsurf.com/windsurf/cascade/mcp
- **Kiro MCP:** https://kiro.dev/docs/mcp/configuration/

### Model Information

- **sentence-transformers/all-MiniLM-L6-v2:** https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
  - 384 dimensions
  - 22M parameters
  - Max sequence length: 256 tokens
  - Training: 1B+ sentence pairs

---

## Appendix A: Content Types Reference

| Type | Description | Example |
|------|-------------|---------|
| `fact` | Objective information | "The database uses PostgreSQL 15" |
| `preference` | User or team preferences | "Prefer tabs over spaces" |
| `decision` | Recorded decisions | "Chose React over Vue for frontend" |
| `convention` | Coding conventions | "All API endpoints follow REST naming" |
| `pattern` | Code patterns | "Error handling uses Result<T, E> pattern" |

## Appendix B: Scope Levels Reference

| Scope | Description | Example Path |
|-------|-------------|--------------|
| `global` | Applies everywhere | (no path) |
| `workspace` | User's projects directory | `/Users/dev/projects` |
| `project` | Single project | `/Users/dev/myproject` |
| `file` | Specific file | `/Users/dev/myproject/src/auth.py` |

---

*Document prepared for documentation enhancement implementation.*
