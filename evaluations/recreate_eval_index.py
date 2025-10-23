#!/usr/bin/env python3
"""
Script to recreate the evaluation index with Azure AI Evaluation SDK schema.
This will delete the existing 'evals' index and create a new one with proper mapping.
"""

import os
import json
import sys
from pathlib import Path
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from variables.env"""
    env_path = Path(__file__).parent.parent / 'variables.env'
    load_dotenv(env_path)
    
    # Get Elasticsearch configuration
    es_url = os.getenv('ELASTICSEARCH_O11Y_URL')
    es_api_key = os.getenv('ELASTICSEARCH_O11Y_API_KEY')
    eval_index_name = os.getenv('EVAL_INDEX_NAME', 'evals')
    
    if not es_url or not es_api_key:
        print("❌ Error: Missing Elasticsearch configuration")
        print("Required environment variables:")
        print("- ELASTICSEARCH_O11Y_URL")
        print("- ELASTICSEARCH_O11Y_API_KEY")
        sys.exit(1)
    
    return es_url, es_api_key, eval_index_name

def create_elasticsearch_client(es_url, es_api_key):
    """Create Elasticsearch client"""
    try:
        client = Elasticsearch(
            hosts=[es_url],
            api_key=es_api_key,
            verify_certs=True,
            request_timeout=30
        )
        
        # Test connection
        info = client.info()
        print(f"✅ Connected to Elasticsearch: {info['cluster_name']}")
        return client
    except Exception as e:
        print(f"❌ Error connecting to Elasticsearch: {e}")
        sys.exit(1)

def load_index_mapping():
    """Load the new index mapping from JSON file"""
    mapping_file = Path(__file__).parent / 'eval_index_mappings.json'
    
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        print(f"✅ Loaded index mapping from {mapping_file}")
        return mapping
    except Exception as e:
        print(f"❌ Error loading index mapping: {e}")
        sys.exit(1)

def recreate_index(client, index_name, mapping):
    """Delete existing index and create new one with mapping"""
    
    # Check if index exists
    if client.indices.exists(index=index_name):
        print(f"⚠️  Index '{index_name}' exists. Deleting...")
        try:
            client.indices.delete(index=index_name)
            print(f"✅ Deleted existing index '{index_name}'")
        except Exception as e:
            print(f"❌ Error deleting index: {e}")
            return False
    else:
        print(f"ℹ️  Index '{index_name}' does not exist")
    
    # Create new index with mapping
    print(f"🔄 Creating new index '{index_name}' with Azure SDK schema...")
    try:
        client.indices.create(
            index=index_name,
            body=mapping
        )
        print(f"✅ Created new index '{index_name}' with Azure SDK mapping")
        return True
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        return False

def verify_index(client, index_name):
    """Verify the new index was created correctly"""
    try:
        # Get index mapping
        mapping = client.indices.get_mapping(index=index_name)
        print(f"✅ Index '{index_name}' mapping verified")
        
        # Get index settings
        settings = client.indices.get_settings(index=index_name)
        print(f"✅ Index '{index_name}' settings verified")
        
        # Test document insertion
        test_doc = {
            "query": "test query",
            "parsed_params": {"bedrooms": 2},
            "response": '{"bedrooms": 2}',
            "relevance_score": 4.5,
            "groundedness_score": 4.0,
            "f1_score": 0.9,
            "timestamp": "2024-01-01T00:00:00Z",
            "trace_id": "test_trace",
            "span_id": "test_span"
        }
        
        result = client.index(index=index_name, body=test_doc)
        print(f"✅ Test document inserted with ID: {result['_id']}")
        
        # Clean up test document
        client.delete(index=index_name, id=result['_id'])
        print("✅ Test document cleaned up")
        
        return True
    except Exception as e:
        print(f"❌ Error verifying index: {e}")
        return False

def main():
    """Main execution function"""
    print("🚀 Azure AI Evaluation SDK Index Recreation")
    print("=" * 50)
    
    # Load configuration
    es_url, es_api_key, eval_index_name = load_environment()
    print(f"📋 Target index: {eval_index_name}")
    print(f"📋 Elasticsearch URL: {es_url}")
    
    # Create Elasticsearch client
    client = create_elasticsearch_client(es_url, es_api_key)
    
    # Load index mapping
    mapping = load_index_mapping()
    
    # Recreate index
    if recreate_index(client, eval_index_name, mapping):
        print("✅ Index recreation completed successfully")
        
        # Verify index
        if verify_index(client, eval_index_name):
            print("🎉 Index recreation and verification completed!")
            print("\n📊 Next steps:")
            print("1. Deploy updated code with Azure SDK evaluators")
            print("2. Test evaluation flow")
            print("3. Monitor Elasticsearch for new evaluation data")
        else:
            print("❌ Index verification failed")
            sys.exit(1)
    else:
        print("❌ Index recreation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
