#!/bin/bash

# Zillow Elasticsearch Data Ingestion Script
# This script sets up a Python virtual environment and runs the data ingestion
# Usage: ./ingest-data.sh [-fresh] [-template-only]

set -e  # Exit on any error

# Parse command line arguments
FRESH_DOWNLOAD=false
TEMPLATE_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -fresh)
            FRESH_DOWNLOAD=true
            shift
            ;;
        -template-only)
            TEMPLATE_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: $0 [-fresh] [-template-only]"
            echo "  -fresh: Delete existing CSV file and re-download from source"
            echo "  -template-only: Only update the search template, skip data ingestion"
            exit 1
            ;;
    esac
done

echo "Starting Zillow Elasticsearch Data Ingestion..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Load environment variables
if [ -f "../variables.env" ]; then
    echo "Loading environment variables from ../variables.env..."
    source ../variables.env
elif [ -f "variables.env" ]; then
    echo "Loading environment variables from variables.env..."
    source variables.env
else
    echo "Error: variables.env file not found in current directory or parent directory"
    exit 1
fi

# Handle fresh download flag
if [ "$FRESH_DOWNLOAD" = true ]; then
    echo "Fresh download requested - removing existing CSV file..."
    if [ -f "$EXTRACTED_FILE_NAME" ]; then
        rm "$EXTRACTED_FILE_NAME"
        echo "Removed existing CSV file: $EXTRACTED_FILE_NAME"
    else
        echo "No existing CSV file found to remove"
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Run the ingestion script
if [ "$TEMPLATE_ONLY" = true ]; then
    echo "Running search template update only..."
    python ingest_data.py --template-only
else
    echo "Running data ingestion script..."
    if [ "$FRESH_DOWNLOAD" = true ]; then
        FRESH_DOWNLOAD=true python ingest_data.py
    else
        python ingest_data.py
    fi
fi

# Deactivate virtual environment
echo "Deactivating virtual environment..."
deactivate

echo "Data ingestion setup completed successfully!"
