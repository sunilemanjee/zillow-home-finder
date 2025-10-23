#!/bin/bash

# Offline Evaluation Launcher Script
# This script sets up the environment and runs the offline evaluator

set -e  # Exit on any error

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🏠 Zillow Home Finder - Offline Evaluation Service"
echo "=================================================="
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""

# Source environment variables
echo "📋 Loading environment variables..."
if [ -f "$PROJECT_ROOT/variables.env" ]; then
    source "$PROJECT_ROOT/variables.env"
    echo "✅ Environment variables loaded"
else
    echo "❌ Error: variables.env not found at $PROJECT_ROOT/variables.env"
    exit 1
fi

# Check required environment variables
echo "🔍 Checking required environment variables..."
required_vars=(
    "ELASTICSEARCH_URL"
    "ELASTICSEARCH_API_KEY"
    "EVAL_SOURCE_INDEX_NAME"
    "EVAL_INDEX_NAME"
    "LLM_URL"
    "LLM_MODEL"
    "LLM_API_KEY"
)

missing_vars=()
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo "❌ Error: Missing required environment variables:"
    printf '   - %s\n' "${missing_vars[@]}"
    exit 1
fi

echo "✅ All required environment variables are set"
echo ""

# Set up virtual environment
VENV_DIR="$SCRIPT_DIR/venv"
echo "🐍 Setting up Python virtual environment..."

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "📦 Installing Python dependencies..."
    pip install -r "$REQUIREMENTS_FILE"
    echo "✅ Dependencies installed"
else
    echo "❌ Error: requirements.txt not found at $REQUIREMENTS_FILE"
    exit 1
fi

echo ""

# Check if source index exists
echo "🔍 Checking if source index exists..."
python3 -c "
import sys
sys.path.append('$PROJECT_ROOT/mcp')
from elasticsearch import Elasticsearch
import os

es_client = Elasticsearch(
    hosts=[os.getenv('ELASTICSEARCH_URL')],
    api_key=os.getenv('ELASTICSEARCH_API_KEY'),
    verify_certs=True
)

source_index = os.getenv('EVAL_SOURCE_INDEX_NAME', 'eval_source')
if not es_client.indices.exists(index=source_index):
    print(f'❌ Source index {source_index} does not exist')
    print('Please run: python3 create_eval_source_index.py')
    sys.exit(1)
else:
    print(f'✅ Source index {source_index} exists')
"

if [ $? -ne 0 ]; then
    echo ""
    echo "💡 To create the source index, run:"
    echo "   cd $SCRIPT_DIR"
    echo "   python3 create_eval_source_index.py"
    exit 1
fi

echo ""

# Start the offline evaluator
echo "🚀 Starting offline evaluator..."
echo "   - Polling every 5 seconds for unevaluated queries"
echo "   - Press Ctrl+C to stop"
echo ""

# Run the offline evaluator with proper logging
python3 "$SCRIPT_DIR/offline_evaluator.py" 2>&1 | while IFS= read -r line; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
done

echo ""
echo "👋 Offline evaluator stopped"
