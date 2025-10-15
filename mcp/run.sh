#!/bin/bash

# MCP Home Finder Server Startup Script

echo "🏠 Starting MCP Home Finder Server..."

# Check if we're in the right directory
if [ ! -f "__main__.py" ]; then
    echo "❌ Error: __main__.py not found. Please run this script from the mcp/ directory."
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Load environment variables
if [ -f "../variables.env" ]; then
    echo "📋 Loading environment variables from ../variables.env..."
    source ../variables.env
else
    echo "❌ Error: ../variables.env file not found"
    echo "   Please copy variables.env.template to variables.env and configure it"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Check if MCP server configuration is set
MCP_HOST=${MCP_HOST:-localhost}
MCP_PORT=${MCP_PORT:-8000}

# Kill any process running on the MCP port
echo "🔍 Checking for processes on port ${MCP_PORT}..."
if lsof -ti:${MCP_PORT} > /dev/null 2>&1; then
    echo "⚠️  Found process(es) running on port ${MCP_PORT}, killing them..."
    lsof -ti:${MCP_PORT} | xargs kill -9
    sleep 2
    echo "✅ Port ${MCP_PORT} is now free"
else
    echo "✅ Port ${MCP_PORT} is available"
fi

echo "🚀 Starting MCP server..."
echo "   Server will be available at: http://${MCP_HOST}:${MCP_PORT}"
echo "   Press Ctrl+C to stop the server"
echo ""

# Start the MCP server
python __main__.py
