#!/usr/bin/env python3
"""
Offline Query Parser Evaluator

This script polls the EVAL_SOURCE_INDEX_NAME for unevaluated queries and processes them
using the QueryParserEvaluator. Results are stored in EVAL_INDEX_NAME.
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from elasticsearch import Elasticsearch
from dotenv import load_dotenv

# Add parent directory to path to import query_evaluator
sys.path.append(str(Path(__file__).parent.parent / 'mcp'))
from query_evaluator import QueryParserEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OfflineEvaluator:
    """
    Offline evaluator that polls for unevaluated queries and processes them.
    """
    
    def __init__(self):
        """Initialize the offline evaluator."""
        # Load environment variables
        env_path = Path(__file__).parent.parent / 'variables.env'
        load_dotenv(env_path)
        
        # Get configuration
        self.es_url = os.getenv('ELASTICSEARCH_O11Y_URL')
        self.es_api_key = os.getenv('ELASTICSEARCH_O11Y_API_KEY')
        self.eval_source_index = os.getenv('EVAL_SOURCE_INDEX_NAME', 'eval_source')
        self.eval_index = os.getenv('EVAL_INDEX_NAME', 'evals')
        
        if not self.es_url or not self.es_api_key:
            raise ValueError("Missing ELASTICSEARCH_O11Y_URL or ELASTICSEARCH_O11Y_API_KEY")
        
        # Initialize Elasticsearch client
        self.es_client = Elasticsearch(
            hosts=[self.es_url],
            api_key=self.es_api_key,
            verify_certs=True
        )
        
        # Initialize query parser evaluator
        self.query_evaluator = QueryParserEvaluator(
            es_url=self.es_url,
            es_api_key=self.es_api_key,
            eval_index_name=self.eval_index,
            llm_url=os.getenv('LLM_URL'),
            llm_model=os.getenv('LLM_MODEL'),
            llm_api_key=os.getenv('LLM_API_KEY'),
            llm_version=os.getenv('LLM_VERSION'),
            azure_openai_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            azure_openai_key=os.getenv('AZURE_OPENAI_KEY'),
            azure_openai_model=os.getenv('AZURE_OPENAI_MODEL'),
            azure_openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION')
        )
        
        self.running = True
        self.poll_interval = 5  # seconds
        
        logger.info(f"Offline evaluator initialized")
        logger.info(f"Source index: {self.eval_source_index}")
        logger.info(f"Results index: {self.eval_index}")
        logger.info(f"Poll interval: {self.poll_interval} seconds")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def get_unevaluated_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get unevaluated queries from the source index.
        
        Args:
            limit: Maximum number of queries to retrieve
            
        Returns:
            List of query documents
        """
        try:
            # Query for documents where evaluated=false
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"evaluated": False}}
                        ]
                    }
                },
                "sort": [{"timestamp": {"order": "asc"}}],
                "size": limit
            }
            
            response = self.es_client.search(
                index=self.eval_source_index,
                body=query
            )
            
            hits = response.get('hits', {}).get('hits', [])
            queries = []
            
            for hit in hits:
                doc = hit['_source']
                doc['_id'] = hit['_id']  # Include document ID for updates
                queries.append(doc)
            
            logger.info(f"Found {len(queries)} unevaluated queries")
            return queries
            
        except Exception as e:
            logger.error(f"Error retrieving unevaluated queries: {e}")
            return []
    
    def process_query(self, query_doc: Dict[str, Any]) -> bool:
        """
        Process a single query for evaluation.
        
        Args:
            query_doc: Query document from source index
            
        Returns:
            True if successful, False otherwise
        """
        doc_id = query_doc.get('_id')
        query = query_doc.get('query', '')
        parsed_params = query_doc.get('parsed_params', {})
        trace_id = query_doc.get('trace_id')
        span_id = query_doc.get('span_id')
        
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            # Run evaluation
            evaluation_result = self.query_evaluator.evaluate_query_parsing(
                query=query,
                parsed_params=parsed_params,
                trace_id=trace_id,
                span_id=span_id
            )
            
            logger.info(f"Evaluation completed. Overall score: {evaluation_result.get('evaluation_scores', {}).get('overall_score', 'N/A')}")
            
            # Mark as evaluated in source index
            self.es_client.update(
                index=self.eval_source_index,
                id=doc_id,
                body={
                    "doc": {
                        "evaluated": True,
                        "evaluated_at": datetime.utcnow().isoformat() + 'Z'
                    }
                }
            )
            
            logger.info(f"Marked query {doc_id} as evaluated")
            return True
            
        except Exception as e:
            logger.error(f"Error processing query {doc_id}: {e}")
            return False
    
    def run_polling_loop(self):
        """Run the main polling loop."""
        logger.info("Starting offline evaluation polling loop...")
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        processed_count = 0
        error_count = 0
        
        while self.running:
            try:
                # Get unevaluated queries
                queries = self.get_unevaluated_queries()
                
                if not queries:
                    logger.debug("No unevaluated queries found, sleeping...")
                    time.sleep(self.poll_interval)
                    continue
                
                # Process each query
                for query_doc in queries:
                    if not self.running:
                        break
                    
                    success = self.process_query(query_doc)
                    if success:
                        processed_count += 1
                    else:
                        error_count += 1
                
                logger.info(f"Processed {processed_count} queries, {error_count} errors")
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                error_count += 1
                time.sleep(self.poll_interval)
        
        logger.info(f"Offline evaluator stopped. Total processed: {processed_count}, Total errors: {error_count}")

def main():
    """Main entry point."""
    try:
        evaluator = OfflineEvaluator()
        evaluator.run_polling_loop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
