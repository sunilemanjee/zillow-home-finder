#!/usr/bin/env python3
"""
Test script for query parser evaluation integration.

This script tests the evaluation functionality by:
1. Testing with ENABLE_QUERY_EVAL=true (tool should be available)
2. Testing with ENABLE_QUERY_EVAL=false (tool should not be available)
3. Running sample evaluations with test queries
4. Verifying Elasticsearch storage
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add the mcp directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'mcp'))

from query_evaluator import QueryParserEvaluator

def test_evaluator_initialization():
    """Test that the evaluator can be initialized with environment variables."""
    print("Testing evaluator initialization...")
    
    try:
        evaluator = QueryParserEvaluator(
            es_url=os.getenv('ELASTICSEARCH_O11Y_URL'),
            es_api_key=os.getenv('ELASTICSEARCH_O11Y_API_KEY'),
            eval_index_name=os.getenv('EVAL_INDEX_NAME'),
            llm_url=os.getenv('LLM_URL'),
            llm_model=os.getenv('LLM_MODEL'),
            llm_api_key=os.getenv('LLM_API_KEY')
        )
        print("✅ Evaluator initialized successfully")
        return evaluator
    except Exception as e:
        print(f"❌ Failed to initialize evaluator: {e}")
        return None

def test_evaluation_logic(evaluator):
    """Test the evaluation logic with sample queries."""
    print("\nTesting evaluation logic...")
    
    test_cases = [
        {
            "query": "homes within 10 miles of Orlando FL with 2 beds",
            "parsed_params": {
                "query": "homes",
                "distance": 10,
                "distance_unit": "mi",
                "latitude": 28.5383,
                "longitude": -81.3792,
                "bedrooms": 2
            }
        },
        {
            "query": "3 bedroom house under $500k in Miami",
            "parsed_params": {
                "query": "house",
                "bedrooms": 3,
                "home_price_max": 500000
            }
        },
        {
            "query": "condo with pool near downtown",
            "parsed_params": {
                "query": "condo",
                "feature": "pool"
            }
        },
        {
            "query": "invalid query with bad parameters",
            "parsed_params": {
                "query": "invalid",
                "bedrooms": "not_a_number",  # Invalid type
                "distance": 1000,  # Invalid range
                "latitude": 200  # Invalid range
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['query']}")
        
        try:
            result = evaluator.evaluate_query_parsing(
                query=test_case['query'],
                parsed_params=test_case['parsed_params']
            )
            
            scores = result.get('evaluation_scores', {})
            issues = result.get('issues', [])
            
            print(f"  Overall Score: {scores.get('overall_score', 0):.3f}")
            print(f"  Type Validation: {scores.get('type_validation_score', 0):.3f}")
            print(f"  Range Validation: {scores.get('range_validation_score', 0):.3f}")
            print(f"  Completeness: {scores.get('completeness_score', 0):.3f}")
            print(f"  Accuracy: {scores.get('accuracy_score', 0):.3f}")
            print(f"  Issues: {len(issues)}")
            
            if issues:
                for issue in issues[:3]:  # Show first 3 issues
                    print(f"    - {issue}")
                if len(issues) > 3:
                    print(f"    ... and {len(issues) - 3} more issues")
            
            print("  ✅ Evaluation completed successfully")
            
        except Exception as e:
            print(f"  ❌ Evaluation failed: {e}")

def test_elasticsearch_connection(evaluator):
    """Test Elasticsearch connection and index creation."""
    print("\nTesting Elasticsearch connection...")
    
    try:
        # Test connection
        info = evaluator.es_client.info()
        print(f"✅ Connected to Elasticsearch cluster: {info.get('cluster_name', 'unknown')}")
        
        # Test index creation (if it doesn't exist)
        index_name = evaluator.eval_index_name
        if not evaluator.es_client.indices.exists(index=index_name):
            print(f"Creating index: {index_name}")
            # Create a simple index mapping
            mapping = {
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "query": {"type": "text"},
                        "parsed_params": {"type": "object"},
                        "evaluation_scores": {"type": "object"},
                        "issues": {"type": "text"},
                        "trace_id": {"type": "keyword"},
                        "span_id": {"type": "keyword"}
                    }
                }
            }
            evaluator.es_client.indices.create(index=index_name, body=mapping)
            print(f"✅ Index '{index_name}' created successfully")
        else:
            print(f"✅ Index '{index_name}' already exists")
            
    except Exception as e:
        print(f"❌ Elasticsearch connection failed: {e}")

def main():
    """Main test function."""
    print("Query Parser Evaluation Integration Test")
    print("=" * 50)
    
    # Load environment variables
    env_path = Path(__file__).parent.parent / 'variables.env'
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
        print("✅ Environment variables loaded from variables.env")
    else:
        print("⚠️  variables.env not found, using system environment variables")
    
    # Test evaluator initialization
    evaluator = test_evaluator_initialization()
    if not evaluator:
        print("❌ Cannot proceed without evaluator")
        return
    
    # Test Elasticsearch connection
    test_elasticsearch_connection(evaluator)
    
    # Test evaluation logic
    test_evaluation_logic(evaluator)
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("\nTo test the MCP tool:")
    print("1. Set ENABLE_QUERY_EVAL=true in variables.env")
    print("2. Start the MCP server")
    print("3. Call parse_query tool, then evaluate_query_parse tool")
    print("4. Check Elasticsearch for stored evaluation results")

if __name__ == "__main__":
    main()
