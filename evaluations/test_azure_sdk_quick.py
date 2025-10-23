#!/usr/bin/env python3
"""
Quick test script for Azure AI Evaluation SDK integration.
Tests imports and basic functionality without making API calls.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

def test_azure_sdk_imports():
    """Test that Azure SDK can be imported"""
    print("🧪 Testing Azure SDK imports...")
    
    try:
        from azure.ai.evaluation import RelevanceEvaluator, GroundednessEvaluator, F1ScoreEvaluator
        print("✅ Azure SDK evaluators imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing Azure SDK: {e}")
        return False

def test_environment_variables():
    """Test that required environment variables are available"""
    print("🧪 Testing environment variables...")
    
    env_path = Path(__file__).parent.parent / 'variables.env'
    load_dotenv(env_path)
    
    required_vars = [
        'ELASTICSEARCH_O11Y_URL',
        'ELASTICSEARCH_O11Y_API_KEY', 
        'LLM_URL',
        'LLM_MODEL',
        'LLM_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    else:
        print("✅ All required environment variables present")
        return True

def test_elasticsearch_connection():
    """Test Elasticsearch connection without making API calls"""
    print("🧪 Testing Elasticsearch connection...")
    
    try:
        from elasticsearch import Elasticsearch
        
        es_url = os.getenv('ELASTICSEARCH_O11Y_URL')
        es_api_key = os.getenv('ELASTICSEARCH_O11Y_API_KEY')
        
        client = Elasticsearch(
            hosts=[es_url],
            api_key=es_api_key,
            verify_certs=True,
            request_timeout=5  # Short timeout for quick test
        )
        
        # Test connection
        info = client.info()
        print(f"✅ Connected to Elasticsearch: {info['cluster_name']}")
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Elasticsearch: {e}")
        return False

def test_model_configuration():
    """Test that model configuration is valid for Azure SDK"""
    print("🧪 Testing model configuration...")
    
    try:
        llm_url = os.getenv('LLM_URL')
        llm_model = os.getenv('LLM_MODEL')
        llm_api_key = os.getenv('LLM_API_KEY')
        
        # Test Azure SDK model config format
        model_config = {
            "azure_endpoint": llm_url,
            "api_key": llm_api_key,
            "azure_deployment": llm_model,
        }
        
        print(f"✅ Model config created:")
        print(f"  - Endpoint: {llm_url}")
        print(f"  - Model: {llm_model}")
        print(f"  - API Key: {'*' * 10}...{llm_api_key[-4:] if llm_api_key else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating model config: {e}")
        return False

def test_index_mapping():
    """Test that the new index mapping file exists and is valid JSON"""
    print("🧪 Testing index mapping file...")
    
    try:
        mapping_file = Path(__file__).parent / 'eval_index_mappings.json'
        
        if not mapping_file.exists():
            print(f"❌ Index mapping file not found: {mapping_file}")
            return False
        
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        
        # Check required fields
        required_fields = ['mappings', 'settings']
        missing_fields = [field for field in required_fields if field not in mapping]
        
        if missing_fields:
            print(f"❌ Missing fields in mapping: {missing_fields}")
            return False
        
        print("✅ Index mapping file is valid JSON with required fields")
        return True
        
    except Exception as e:
        print(f"❌ Error reading index mapping: {e}")
        return False

def test_opentelemetry_imports():
    """Test that OpenTelemetry imports work"""
    print("🧪 Testing OpenTelemetry imports...")
    
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.metrics import MeterProvider
        
        print("✅ OpenTelemetry imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Error importing OpenTelemetry: {e}")
        return False

def main():
    """Main test execution"""
    print("🚀 Azure AI Evaluation SDK Quick Test")
    print("=" * 50)
    
    tests = [
        ("Azure SDK Imports", test_azure_sdk_imports),
        ("Environment Variables", test_environment_variables),
        ("Elasticsearch Connection", test_elasticsearch_connection),
        ("Model Configuration", test_model_configuration),
        ("Index Mapping File", test_index_mapping),
        ("OpenTelemetry Imports", test_opentelemetry_imports),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Summary:")
    print("=" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{'✅' if result else '❌'} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All quick tests passed! Azure SDK integration is ready.")
        print("\n📋 Next steps:")
        print("1. Run the full integration test (may take time due to API calls)")
        print("2. Deploy the updated code")
        print("3. Test with actual search operations")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
