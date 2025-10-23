#!/usr/bin/env python3
"""
Script to create the EVAL_SOURCE_INDEX_NAME with proper mappings.
This index stores unevaluated query data for offline processing.
"""

import os
import json
import sys
from pathlib import Path
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

def main():
    """Create the eval source index with proper mappings."""
    
    # Load environment variables
    env_path = Path(__file__).parent.parent / 'variables.env'
    load_dotenv(env_path)
    
    # Get configuration from environment
    es_url = os.getenv('ELASTICSEARCH_URL')
    es_api_key = os.getenv('ELASTICSEARCH_API_KEY')
    eval_source_index = os.getenv('EVAL_SOURCE_INDEX_NAME', 'eval_source')
    
    if not es_url or not es_api_key:
        print("Error: Missing ELASTICSEARCH_URL or ELASTICSEARCH_API_KEY")
        sys.exit(1)
    
    # Initialize Elasticsearch client
    es_client = Elasticsearch(
        hosts=[es_url],
        api_key=es_api_key,
        verify_certs=True
    )
    
    # Load index mappings
    mappings_path = Path(__file__).parent / 'eval_source_index_mappings.json'
    with open(mappings_path, 'r') as f:
        index_config = json.load(f)
    
    try:
        # Check if index already exists
        if es_client.indices.exists(index=eval_source_index):
            print(f"Index '{eval_source_index}' already exists.")
            response = input("Do you want to delete and recreate it? (y/N): ")
            if response.lower() == 'y':
                es_client.indices.delete(index=eval_source_index)
                print(f"Deleted existing index '{eval_source_index}'")
            else:
                print("Aborted.")
                return
        
        # Create the index
        es_client.indices.create(
            index=eval_source_index,
            body=index_config
        )
        
        print(f"✅ Successfully created index '{eval_source_index}'")
        print(f"📊 Index configuration:")
        print(f"   - Query field: text with standard analyzer")
        print(f"   - Parsed params: object with nested properties")
        print(f"   - Session tracking: session_id, trace_id, span_id")
        print(f"   - Evaluation status: evaluated (boolean)")
        print(f"   - Conversation history: nested array")
        
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
