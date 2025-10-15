#!/bin/bash

# Zillow Home Finder - Chainlit App Startup Script

echo "🏠 Starting Zillow Home Finder Chainlit App..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if variables.env exists in parent directory
if [ ! -f "../variables.env" ]; then
    echo "⚠️  Warning: variables.env not found in parent directory"
    echo "   Please ensure your LLM configuration is properly set up"
fi

# Kill any existing processes on port 8022
echo "🔄 Checking for existing processes on port 8022..."
if lsof -ti:8022 > /dev/null 2>&1; then
    echo "⚠️  Found existing processes on port 8022, killing them..."
    lsof -ti:8022 | xargs kill -9
    sleep 2
    echo "✅ Port 8022 cleared"
else
    echo "✅ Port 8022 is available"
fi

# Start the Chainlit app
echo "🚀 Starting Chainlit app on port 8022..."
echo "   Open your browser to: http://localhost:8022"
echo "   Press Ctrl+C to stop the server"
echo ""

chainlit run app.py --port 8022
