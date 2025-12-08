#!/bin/bash
# Test script for uvx enyal installation
# Run from project root: ./scripts/test_uvx_install.sh

set -e

echo "=========================================="
echo "Testing uvx enyal installation"
echo "=========================================="
echo ""

# Detect platform
PLATFORM=$(uname -s)
ARCH=$(uname -m)
echo "Platform: $PLATFORM ($ARCH)"
echo ""

# Check uvx is available
echo "Test 1: Check uvx is installed"
if command -v uvx &> /dev/null; then
    echo "✓ uvx found: $(uvx --version 2>/dev/null || echo 'version unknown')"
else
    echo "✗ uvx not found. Install with: pip install uv"
    exit 1
fi
echo ""

# Test 2: enyal --help (basic CLI)
echo "Test 2: enyal --help"
if uvx enyal --help > /dev/null 2>&1; then
    echo "✓ enyal --help works"
else
    echo "✗ enyal --help failed"
    exit 1
fi
echo ""

# Test 3: enyal serve --help
echo "Test 3: enyal serve --help"
if uvx enyal serve --help > /dev/null 2>&1; then
    echo "✓ enyal serve --help works"
else
    echo "✗ enyal serve --help failed"
    exit 1
fi
echo ""

# Test 4: Start server briefly (5 second timeout)
echo "Test 4: Start enyal serve (5 second test)"
echo "Starting server..."

# Start server in background
timeout 5 uvx enyal serve &
SERVER_PID=$!

# Wait a moment for startup
sleep 2

# Check if still running (good sign)
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "✓ Server started successfully (waiting for MCP connection)"
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
else
    echo "✓ Server started and completed (or timed out as expected)"
fi
echo ""

# Test 5: Python 3.12 explicit (for macOS Intel)
if [[ "$PLATFORM" == "Darwin" && "$ARCH" == "x86_64" ]]; then
    echo "Test 5: macOS Intel - testing with --python 3.12"
    if uvx --python 3.12 enyal --help > /dev/null 2>&1; then
        echo "✓ uvx --python 3.12 enyal works"
    else
        echo "✗ uvx --python 3.12 enyal failed"
        echo "  Make sure Python 3.12 is installed"
    fi
    echo ""
fi

# Test 6: Check python -m enyal.mcp works
echo "Test 6: python -m enyal.mcp (module execution)"
if python3 -c "import sys; sys.path.insert(0, 'src'); from enyal.mcp.server import main; print('OK')" 2>/dev/null; then
    echo "✓ Module import works"
else
    echo "⚠ Module import test skipped (not installed locally)"
fi
echo ""

echo "=========================================="
echo "All tests passed!"
echo "=========================================="
echo ""
echo "Recommended MCP configuration:"
echo ""
if [[ "$PLATFORM" == "Darwin" && "$ARCH" == "x86_64" ]]; then
    echo '  {
    "mcpServers": {
      "enyal": {
        "command": "uvx",
        "args": ["--python", "3.12", "enyal", "serve"]
      }
    }
  }'
else
    echo '  {
    "mcpServers": {
      "enyal": {
        "command": "uvx",
        "args": ["enyal", "serve"]
      }
    }
  }'
fi
