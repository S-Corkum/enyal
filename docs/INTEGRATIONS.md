# Enyal MCP Integration Guide

This guide provides detailed instructions for integrating Enyal with various AI coding assistants that support the Model Context Protocol (MCP).

## Overview

Enyal runs as an MCP server that provides persistent memory capabilities to AI agents. All MCP clients use a similar JSON configuration format—the main difference is where the configuration file is located.

**Universal Configuration Format:**
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

## Claude Code

[Claude Code](https://docs.claude.com/en/docs/claude-code) is Anthropic's official CLI for agentic coding.

### Configuration Locations

| Scope | Location | Use Case |
|-------|----------|----------|
| Project | `.mcp.json` | Team-shared config in version control |
| User | `~/.claude/.mcp.json` | Personal servers across all projects |

### Setup Methods

#### Method 1: Configuration File

Create `.mcp.json` in your project root:

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

#### Method 2: CLI Command

```bash
claude mcp add-json enyal '{"command":"python","args":["-m","enyal.mcp"],"env":{"ENYAL_DB_PATH":"~/.enyal/context.db"}}'
```

#### Method 3: Interactive CLI

```bash
claude mcp add enyal --command python --args "-m" "enyal.mcp"
```

### Verify Installation

```bash
# List configured servers
claude mcp list

# Get details for enyal
claude mcp get enyal
```

Inside Claude Code, use the `/mcp` command to check server status.

### Usage

Once configured, Claude Code will automatically have access to Enyal tools:

```
You: @enyal remember this project uses TypeScript strict mode
Claude: I'll store that context for you.
[Calls enyal_remember]
✓ Stored: "this project uses TypeScript strict mode"

You: What TypeScript configuration should I use?
Claude: Let me check what I know about this project's TypeScript setup.
[Calls enyal_recall]
Based on stored context, this project uses TypeScript strict mode...
```

### Advanced Configuration

**With OAuth (if using authenticated services):**
```json
{
  "mcpServers": {
    "enyal": {
      "command": "python",
      "args": ["-m", "enyal.mcp"],
      "env": {
        "ENYAL_DB_PATH": "~/.enyal/context.db",
        "ENYAL_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

### Troubleshooting

**Server not appearing:**
1. Restart Claude Code after adding configuration
2. Check that the `.mcp.json` file is valid JSON
3. Verify Python 3.11+ is available: `python --version`

**Permission errors:**
```bash
# Ensure user-level config directory exists
mkdir -p ~/.claude
```

---

## Claude Desktop

[Claude Desktop](https://claude.ai/download) is Anthropic's desktop application.

### Configuration Locations

| Platform | Location |
|----------|----------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

### Setup

#### macOS

1. Open Terminal
2. Create or edit the configuration file:

```bash
# Create directory if it doesn't exist
mkdir -p ~/Library/Application\ Support/Claude

# Edit configuration
nano ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

3. Add the configuration:

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

4. Restart Claude Desktop

#### Windows

1. Press `Win+R`, type `%APPDATA%\Claude`, press Enter
2. Create or edit `claude_desktop_config.json`
3. Add the configuration:

```json
{
  "mcpServers": {
    "enyal": {
      "command": "python",
      "args": ["-m", "enyal.mcp"],
      "env": {
        "ENYAL_DB_PATH": "%USERPROFILE%\\.enyal\\context.db"
      }
    }
  }
}
```

4. Restart Claude Desktop

### Windows-Specific Notes

- Use `%USERPROFILE%` instead of `~` for home directory
- Ensure Python is in your system PATH
- If using a virtual environment, specify the full path to python:

```json
{
  "mcpServers": {
    "enyal": {
      "command": "C:\\Users\\YourName\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
      "args": ["-m", "enyal.mcp"]
    }
  }
}
```

### Verify Installation

After restarting Claude Desktop, look for the MCP tools icon in the chat interface. Enyal tools should appear when you click it.

---

## Cursor

[Cursor](https://cursor.com) is an AI-native code editor.

### Configuration Locations

| Scope | Location |
|-------|----------|
| Global | `~/.cursor/mcp.json` |
| Project | `.cursor/mcp.json` |

### Setup

#### Method 1: Configuration File

Create `~/.cursor/mcp.json` for global access:

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

Or create `.cursor/mcp.json` in your project for project-specific access.

#### Method 2: Settings UI

1. Open Cursor
2. Go to **File → Preferences → Cursor Settings**
3. Select **MCP** in the sidebar
4. Add your server configuration

### Verify Installation

1. Open Cursor Settings → MCP
2. Check that "enyal" appears in the server list
3. Verify the status shows as "Connected" or "Running"

### Usage with Composer

Cursor's Composer Agent automatically uses MCP tools when relevant:

```
You: What conventions does this project follow?
Composer: [Uses enyal_recall to search for conventions]
Based on stored context, this project follows...
```

### Current Limitations

- **Resources not supported:** Cursor currently only supports MCP tools, not resources
- **SSH limitations:** MCP servers may not work properly over SSH connections

### Troubleshooting

**Tools not appearing:**
1. Check that the MCP server shows as "Running" in settings
2. Restart Cursor after configuration changes
3. Verify JSON syntax in configuration file

---

## Windsurf

[Windsurf](https://windsurf.com) is Codeium's AI-powered IDE.

### Configuration Location

`~/.codeium/windsurf/mcp_config.json`

### Setup Methods

#### Method 1: Configuration File

Create or edit `~/.codeium/windsurf/mcp_config.json`:

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

#### Method 2: Settings UI

1. Open Windsurf
2. Click **Windsurf Settings** (bottom right) or press `Cmd+Shift+P` / `Ctrl+Shift+P`
3. Type "Open Windsurf Settings"
4. Navigate to **Cascade → MCP**
5. Enable MCP and add your server

#### Method 3: Plugin Store

1. Click the **Plugins** icon in the Cascade panel (top right)
2. Search for MCP servers or add manually
3. Configure the server settings

### Verify Installation

1. Open the Cascade panel
2. Click the Plugins icon
3. Verify "enyal" appears in the MCP servers list

### Built-in MCP Servers

Windsurf comes with some pre-configured MCP servers:
- **Context7** - Library documentation
- **GitHub** - Repository integration

Enyal complements these by providing persistent project-specific memory.

### Usage with Cascade

```
You: Remember that we're using FastAPI with SQLModel for the backend
Cascade: [Calls enyal_remember] I've stored that context.

You: What backend framework are we using?
Cascade: [Calls enyal_recall] Based on stored context, you're using FastAPI with SQLModel for the backend.
```

### Security Best Practices

**Never hardcode secrets in config files:**

```json
{
  "mcpServers": {
    "enyal": {
      "command": "python",
      "args": ["-m", "enyal.mcp"],
      "env": {
        "ENYAL_DB_PATH": "${HOME}/.enyal/context.db"
      }
    }
  }
}
```

### Team Administration

Team admins can:
- Toggle MCP access for the team
- Whitelist approved MCP servers
- Block non-whitelisted servers

---

## Kiro

[Kiro](https://kiro.dev) is AWS's spec-driven AI IDE.

### Configuration Locations

| Scope | Location |
|-------|----------|
| Global | `~/.kiro/settings/mcp.json` |
| Project | `.kiro/settings/mcp.json` |

### Setup Methods

#### Method 1: Configuration File

Create `~/.kiro/settings/mcp.json`:

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
      "autoApprove": ["enyal_recall", "enyal_stats", "enyal_get"]
    }
  }
}
```

#### Method 2: UI Setup

1. Click the **Kiro ghost tab** in the sidebar
2. Find **MCP Servers** in the list
3. Click **"+"** to add a new server
4. Configure the server settings

### Configuration Options

Kiro supports additional configuration options:

| Option | Type | Description |
|--------|------|-------------|
| `disabled` | boolean | Temporarily disable the server |
| `autoApprove` | string[] | Tools that don't require approval |
| `disabledTools` | string[] | Tools to disable |

**Example with all options:**
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
      "autoApprove": ["enyal_recall", "enyal_stats", "enyal_get"],
      "disabledTools": []
    }
  }
}
```

### Environment Variable Expansion

Kiro supports environment variable expansion using `${VAR}` syntax:

```json
{
  "env": {
    "ENYAL_DB_PATH": "${HOME}/.enyal/context.db",
    "ENYAL_LOG_LEVEL": "${ENYAL_LOG_LEVEL:-INFO}"
  }
}
```

### Remote MCP Servers

Kiro also supports remote MCP servers via HTTP:

```json
{
  "mcpServers": {
    "remote-enyal": {
      "url": "https://your-server.com/mcp",
      "headers": {
        "Authorization": "Bearer ${API_TOKEN}"
      }
    }
  }
}
```

### CLI Integration

Kiro CLI uses the same configuration:

```bash
# Your .kiro folder config works in both IDE and CLI
kiro chat  # Uses same MCP servers
```

### Security Best Practices

1. Use environment variable references (`${VAR}`) for secrets
2. Never commit `mcp.json` files with credentials to git
3. Add `.kiro/settings/mcp.json` to `.gitignore`
4. Use `autoApprove` sparingly—only for trusted read-only tools
5. Review tool permissions before approving

---

## Troubleshooting Guide

### Common Issues

#### Python Not Found

**Symptom:** MCP server fails to start, "python not found" error

**Solutions:**
1. Verify Python 3.11+ is installed: `python --version`
2. Use full path to Python executable:
   ```json
   {
     "command": "/usr/local/bin/python3"
   }
   ```
3. On Windows, ensure Python is in PATH or use full path:
   ```json
   {
     "command": "C:\\Python312\\python.exe"
   }
   ```

#### Enyal Not Installed

**Symptom:** ModuleNotFoundError for enyal

**Solutions:**
1. Install enyal: `pip install enyal`
2. If using virtual environments, ensure the MCP config uses the correct Python
3. Verify installation: `python -c "import enyal; print('OK')"`

#### Database Permission Errors

**Symptom:** Cannot create database file

**Solutions:**
```bash
# Create database directory
mkdir -p ~/.enyal
chmod 755 ~/.enyal
```

#### Slow Startup

**Symptom:** First query takes several seconds

**Explanation:** The embedding model (~80MB) loads on first use

**Solution:** Pre-load the model:
```json
{
  "env": {
    "ENYAL_PRELOAD_MODEL": "true"
  }
}
```

#### Database Locked

**Symptom:** "database is locked" errors

**Solutions:**
1. Ensure only one MCP server instance per database
2. Use separate databases for different projects:
   ```json
   {
     "env": {
       "ENYAL_DB_PATH": "~/.enyal/project-name.db"
     }
   }
   ```

### Debug Mode

Enable debug logging to troubleshoot issues:

```json
{
  "mcpServers": {
    "enyal": {
      "command": "python",
      "args": ["-m", "enyal.mcp"],
      "env": {
        "ENYAL_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

Check logs at `~/.claude/debug.log` (Claude Code) or the IDE's developer console.

---

## Version Compatibility

| Platform | Minimum Version | Notes |
|----------|-----------------|-------|
| Claude Code | 1.0+ | Full support |
| Claude Desktop | 1.0+ | Full support |
| Cursor | 0.40+ | Tools only (no resources) |
| Windsurf | 1.0+ | Full support |
| Kiro | 1.0+ | Full support with auto-approve |

| Enyal | Python | Notes |
|-------|--------|-------|
| 0.1.0+ | 3.11+ | Required for modern type hints |
| 0.1.0+ | 3.12+ | Recommended for performance |
| 0.1.0+ | 3.13 | macOS ARM64 only (no Intel) |

---

## Quick Reference

### Configuration File Locations

| Platform | Config Path |
|----------|-------------|
| Claude Code (project) | `.mcp.json` |
| Claude Code (user) | `~/.claude/.mcp.json` |
| Claude Desktop (macOS) | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Claude Desktop (Windows) | `%APPDATA%\Claude\claude_desktop_config.json` |
| Cursor (global) | `~/.cursor/mcp.json` |
| Cursor (project) | `.cursor/mcp.json` |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` |
| Kiro (global) | `~/.kiro/settings/mcp.json` |
| Kiro (project) | `.kiro/settings/mcp.json` |

### Minimal Configuration

Copy-paste ready configuration for any platform:

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

### Full Configuration

With all options:

```json
{
  "mcpServers": {
    "enyal": {
      "command": "python",
      "args": ["-m", "enyal.mcp"],
      "env": {
        "ENYAL_DB_PATH": "~/.enyal/context.db",
        "ENYAL_PRELOAD_MODEL": "true",
        "ENYAL_LOG_LEVEL": "INFO"
      }
    }
  }
}
```
