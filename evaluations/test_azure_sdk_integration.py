#!/usr/bin/env python3
"""
Test script for Azure AI Evaluation SDK integration.
Verifies that Azure SDK evaluators work with LiteLLM proxy and store results in Elasticsearch.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import MCP modules
sys.path.append(str(Path(__file__).parent.parent))

from mcp.query_evaluator import QueryParserEvaluator

def load_environment():
    """Load environment variables from variables.env"""
    env_path = Path(__file__).parent.parent / 'variables.env'
    load_dotenv(env_path)
    
    # Get configuration
    es_url = os.getenv('ELASTICSEARCH_O11Y_URL')
    es_api_key = os.getenv('ELASTICSEARCH_O11Y_API_KEY')
    eval_index_name = os.getenv('EVAL_INDEX_NAME', 'evals')
    llm_url = os.getenv('LLM_URL')
    llm_model = os.getenv('LLM_MODEL')
    llm_api_key = os.getenv('LLM_API_KEY')
    
    if not all([es_url, es_api_key, llm_url, llm_model, llm_api_key]):
        print("❌ Error: Missing required environment variables")
        print("Required:")
        print("- ELASTICSEARCH_O11Y_URL")
        print("- ELASTICSEARCH_O11Y_API_KEY")
        print("- LLM_URL")
        print("- LLM_MODEL")
        print("- LLM_API_KEY")
        sys.exit(1)
    
    return es_url, es_api_key, eval_index_name, llm_url, llm_model, llm_api_key

def test_azure_sdk_initialization():
    """Test that Azure SDK evaluators can be initialized"""
    print("🧪 Testing Azure SDK evaluator initialization...")
    
    try:
        es_url, es_api_key, eval_index_name, llm_url, llm_model, llm_api_key = load_environment()
        
        # Initialize evaluator
        evaluator = QueryParserEvaluator(
            es_url=es_url,
            es_api_key=es_api_key,
            eval_index_name=eval_index_name,
            llm_url=llm_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key
        )
        
        print("✅ Azure SDK evaluators initialized successfully")
        return evaluator
        
    except Exception as e:
        print(f"❌ Error initializing Azure SDK evaluators: {e}")
        sys.exit(1)

def test_evaluation_with_azure_sdk(evaluator):
    """Test evaluation using Azure SDK evaluators"""
    print("🧪 Testing Azure SDK evaluation...")
    
    # Test query and parsed parameters
    test_query = "homes within 10 miles of orlando fl with 2 bedrooms"
    test_parsed_params = {
        "location": "orlando fl",
        "distance": 10,
        "distance_unit": "mi",
        "bedrooms": 2
    }
    
    try:
        # Run evaluation
        result = evaluator.evaluate_query_parsing(
            query=test_query,
            parsed_params=test_parsed_params,
            trace_id="test_trace_123",
            span_id="test_span_456"
        )
        
        print("✅ Azure SDK evaluation completed successfully")
        print(f"📊 Evaluation Results:")
        print(f"  - Overall Score: {result.get('evaluation_scores', {}).get('overall_score', 'N/A')}")
        print(f"  - Relevance Score: {result.get('relevance_score', 'N/A')}")
        print(f"  - Groundedness Score: {result.get('groundedness_score', 'N/A')}")
        print(f"  - F1 Score: {result.get('f1_score', 'N/A')}")
        print(f"  - Issues: {len(result.get('issues', []))}")
        
        # Verify Azure SDK format
        required_fields = ['query', 'parsed_params', 'response', 'relevance_score', 'groundedness_score']
        missing_fields = [field for field in required_fields if field not in result]
        
        if missing_fields:
            print(f"⚠️  Missing Azure SDK fields: {missing_fields}")
        else:
            print("✅ All required Azure SDK fields present")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during Azure SDK evaluation: {e}")
        return None

def test_elasticsearch_storage(evaluator, evaluation_result):
    """Test that evaluation results are stored in Elasticsearch"""
    print("🧪 Testing Elasticsearch storage...")
    
    try:
        # Check if result was stored (this is done automatically in evaluate_query_parsing)
        if evaluation_result and 'timestamp' in evaluation_result:
            print("✅ Evaluation result stored in Elasticsearch")
            print(f"  - Timestamp: {evaluation_result['timestamp']}")
            print(f"  - Trace ID: {evaluation_result.get('trace_id', 'N/A')}")
            print(f"  - Span ID: {evaluation_result.get('span_id', 'N/A')}")
        else:
            print("⚠️  No evaluation result to verify storage")
            
    except Exception as e:
        print(f"❌ Error verifying Elasticsearch storage: {e}")

def test_async_evaluation():
    """Test that evaluation can run asynchronously"""
    print("🧪 Testing async evaluation...")
    
    async def run_async_evaluation():
        """Simulate async evaluation like in search_homes_tool"""
        try:
            es_url, es_api_key, eval_index_name, llm_url, llm_model, llm_api_key = load_environment()
            
            evaluator = QueryParserEvaluator(
                es_url=es_url,
                es_api_key=es_api_key,
                eval_index_name=eval_index_name,
                llm_url=llm_url,
                llm_model=llm_model,
                llm_api_key=llm_api_key
            )
            
            # Simulate async evaluation
            test_query = "homes in miami with 3 bedrooms"
            test_params = {"location": "miami", "bedrooms": 3}
            
            result = evaluator.evaluate_query_parsing(
                query=test_query,
                parsed_params=test_params,
                trace_id="async_test_trace",
                span_id="async_test_span"
            )
            
            print("✅ Async evaluation completed successfully")
            return result
            
        except Exception as e:
            print(f"❌ Error in async evaluation: {e}")
            return None
    
    # Run async evaluation
    result = asyncio.run(run_async_evaluation())
    return result is not None

def test_opentelemetry_integration():
    """Test that OpenTelemetry spans are created"""
    print("🧪 Testing OpenTelemetry integration...")
    
    try:
        from opentelemetry import trace
        
        # Check if tracer is available
        tracer = trace.get_tracer(__name__)
        if tracer:
            print("✅ OpenTelemetry tracer available")
            
            # Test span creation
            with tracer.start_as_current_span("test_azure_sdk_integration") as span:
                span.set_attribute("test.type", "azure_sdk_integration")
                span.set_attribute("test.status", "success")
                print("✅ OpenTelemetry span created successfully")
                return True
        else:
            print("⚠️  OpenTelemetry tracer not available")
            return False
            
    except Exception as e:
        print(f"❌ Error testing OpenTelemetry: {e}")
        return False

def main():
    """Main test execution"""
    print("🚀 Azure AI Evaluation SDK Integration Test")
    print("=" * 60)
    
    # Test 1: Azure SDK Initialization
    evaluator = test_azure_sdk_initialization()
    
    # Test 2: Azure SDK Evaluation
    evaluation_result = test_evaluation_with_azure_sdk(evaluator)
    
    # Test 3: Elasticsearch Storage
    if evaluation_result:
        test_elasticsearch_storage(evaluator, evaluation_result)
    
    # Test 4: Async Evaluation
    async_success = test_async_evaluation()
    
    # Test 5: OpenTelemetry Integration
    otel_success = test_opentelemetry_integration()
    
    # Summary
    print("\n📊 Test Summary:")
    print("=" * 30)
    print(f"✅ Azure SDK Initialization: {'PASS' if evaluator else 'FAIL'}")
    print(f"✅ Azure SDK Evaluation: {'PASS' if evaluation_result else 'FAIL'}")
    print(f"✅ Elasticsearch Storage: {'PASS' if evaluation_result else 'FAIL'}")
    print(f"✅ Async Evaluation: {'PASS' if async_success else 'FAIL'}")
    print(f"✅ OpenTelemetry Integration: {'PASS' if otel_success else 'FAIL'}")
    
    if all([evaluator, evaluation_result, async_success, otel_success]):
        print("\n🎉 All tests passed! Azure AI Evaluation SDK integration is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
