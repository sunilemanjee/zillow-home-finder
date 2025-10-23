"""
Query Parser Evaluation Module using Azure AI Evaluation SDK.

This module provides evaluation capabilities for the query parser to assess
how well natural language queries are parsed into structured search parameters.
Uses Azure AI Evaluation SDK with RelevanceEvaluator, GroundednessEvaluator, and F1ScoreEvaluator.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import re

from elasticsearch import Elasticsearch
from opentelemetry import trace

# Azure AI Evaluation SDK imports
from azure.ai.evaluation import RelevanceEvaluator, GroundednessEvaluator, F1ScoreEvaluator

logger = logging.getLogger(__name__)

# Get tracer from the global context
tracer = trace.get_tracer(__name__)

class QueryParserEvaluator:
    """
    Evaluator for query parser using Azure AI Evaluation SDK.
    Validates extracted parameters using RelevanceEvaluator, GroundednessEvaluator, and F1ScoreEvaluator.
    """
    
    def __init__(self, es_url: str, es_api_key: str, eval_index_name: str, 
                 llm_url: str, llm_model: str, llm_api_key: str, llm_version: str = None,
                 azure_openai_endpoint: str = None, azure_openai_key: str = None, 
                 azure_openai_model: str = None, azure_openai_api_version: str = None):
        """
        Initialize the query parser evaluator with Azure AI Evaluation SDK.
        
        Args:
            es_url: Elasticsearch URL for storing evaluation results
            es_api_key: Elasticsearch API key
            eval_index_name: Index name for evaluation results
            llm_url: LLM URL for AI-assisted validation (LiteLLM proxy)
            llm_model: LLM model name
            llm_api_key: LLM API key
            llm_version: LLM API version (optional)
            azure_openai_endpoint: Azure OpenAI endpoint URL
            azure_openai_key: Azure OpenAI API key
            azure_openai_model: Azure OpenAI model name
            azure_openai_api_version: Azure OpenAI API version
        """
        self.es_url = es_url
        self.es_api_key = es_api_key
        self.eval_index_name = eval_index_name
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_version = llm_version
        self.azure_openai_endpoint = azure_openai_endpoint
        self.azure_openai_key = azure_openai_key
        self.azure_openai_model = azure_openai_model
        self.azure_openai_api_version = azure_openai_api_version
        
        # Initialize Elasticsearch client
        self.es_client = Elasticsearch(
            hosts=[es_url],
            api_key=es_api_key,
            verify_certs=True
        )
        
        # Initialize Azure AI Evaluation SDK evaluators
        # Configure for standard OpenAI format (not Azure format) to work with LiteLLM proxy
        self.use_azure_sdk = True
        self.azure_evaluators = {}
        
        try:
            # Use Azure OpenAI format for Azure SDK compatibility
            if azure_openai_endpoint and azure_openai_key and azure_openai_model:
                # Use real Azure OpenAI endpoint
                self.model_config = {
                    "azure_endpoint": azure_openai_endpoint,
                    "api_key": azure_openai_key,
                    "azure_deployment": azure_openai_model,
                    "api_version": azure_openai_api_version or "2024-02-15-preview"
                }
                logger.info("✅ Using real Azure OpenAI endpoint for evaluation")
            else:
                # Fallback to LiteLLM proxy (may not work with Azure SDK)
                self.model_config = {
                    "azure_endpoint": llm_url,  # LiteLLM proxy endpoint (treated as Azure endpoint)
                    "api_key": llm_api_key,
                    "azure_deployment": llm_model,  # Model name (treated as Azure deployment)
                    "api_version": llm_version or "2024-02-15-preview"  # Use provided version or default
                }
                logger.warning("⚠️ Using LiteLLM proxy - Azure SDK may not work properly")
            
            # Initialize Azure SDK evaluators with OpenAI format
            self.azure_evaluators = {
                'relevance': RelevanceEvaluator(self.model_config),
                'groundedness': GroundednessEvaluator(self.model_config),
                'f1': F1ScoreEvaluator()
            }
            logger.info("✅ Azure AI Evaluation SDK evaluators initialized with OpenAI format")
            
        except Exception as e:
            logger.error(f"❌ Azure AI Evaluation SDK initialization failed: {e}")
            raise
        
        # Define valid search template parameters based on search-template.mustache
        self.valid_parameters = {
            'query': str,
            'distance': (int, float),
            'distance_unit': str,
            'latitude': (int, float),
            'longitude': (int, float),
            'bedrooms': (int, float),
            'bathrooms': (int, float),
            'tax': (int, float),
            'maintenance': (int, float),
            'square_footage': (int, float),
            'home_price_min': (int, float),
            'home_price_max': (int, float),
            'feature': str
        }
        
        # Define reasonable value ranges for validation
        self.parameter_ranges = {
            'bedrooms': (1, 20),
            'bathrooms': (1, 20),
            'distance': (0.1, 1000),
            'latitude': (-90, 90),
            'longitude': (-180, 180),
            'tax': (0, 10000000),
            'maintenance': (0, 10000),
            'square_footage': (100, 50000),
            'home_price_min': (0, 10000000),
            'home_price_max': (0, 10000000)
        }
        
        # Valid distance units
        self.valid_distance_units = {'mi', 'km', 'miles', 'kilometers'}
    
    def evaluate_query_parsing(self, query: str, parsed_params: Dict[str, Any], 
                             trace_id: Optional[str] = None, span_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of query parsing using Azure AI Evaluation SDK or custom logic.
        
        Args:
            query: Original user query
            parsed_params: Parsed parameters from the query parser
            trace_id: OpenTelemetry trace ID for correlation
            span_id: OpenTelemetry span ID for correlation
            
        Returns:
            Dictionary containing evaluation results and scores
        """
        with tracer.start_as_current_span("query_evaluation") as span:
            span.set_attribute("evaluation.query", query[:500])
            span.set_attribute("evaluation.parsed_params_count", len(parsed_params))
            span.set_attribute("evaluation.trace_id", trace_id or "")
            span.set_attribute("evaluation.span_id", span_id or "")
            span.set_attribute("evaluation.use_azure_sdk", self.use_azure_sdk)
            
            start_time = time.time()
            
            try:
                # Use Azure AI Evaluation SDK only
                return self._evaluate_with_azure_sdk(query, parsed_params, trace_id, span_id, start_time)
                    
            except Exception as e:
                logger.error(f"Error during query evaluation: {e}")
                span.record_exception(e)
                span.set_attribute("evaluation.success", False)
                span.set_attribute("evaluation.error", str(e))
                
                # Return error result
                return {
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'query': query,
                    'parsed_params': parsed_params,
                    'response': json.dumps(parsed_params),
                    'evaluation_scores': {
                        'overall_score': 0.0,
                        'error': str(e)
                    },
                    'issues': [f"Evaluation error: {str(e)}"],
                    'trace_id': trace_id,
                    'span_id': span_id
                }
    
    def _evaluate_with_azure_sdk(self, query: str, parsed_params: Dict[str, Any], 
                               trace_id: Optional[str], span_id: Optional[str], start_time: float) -> Dict[str, Any]:
        """Evaluate using Azure AI Evaluation SDK"""
        with tracer.start_as_current_span("azure_sdk_evaluation") as span:
            response_json = json.dumps(parsed_params)
            evaluation_results = {}
            all_issues = []
            
            # 1. Relevance Evaluation
            relevance_score = 0
            with tracer.start_as_current_span("relevance_evaluator") as rel_span:
                try:
                    # Azure SDK evaluators are synchronous
                    relevance_result = self.azure_evaluators['relevance'](
                        query=query,
                        response=response_json
                    )
                    relevance_score = relevance_result.get('gpt_relevance', 0)
                    evaluation_results['relevance'] = relevance_result
                    rel_span.set_attribute("evaluation.relevance_score", relevance_score)
                    logger.info(f"Relevance evaluation: {relevance_score}")
                except Exception as e:
                    logger.warning(f"Relevance evaluation failed: {e}")
                    all_issues.append(f"Relevance evaluation error: {str(e)}")
            
            # 2. Groundedness Evaluation
            groundedness_score = 0
            with tracer.start_as_current_span("groundedness_evaluator") as ground_span:
                try:
                    groundedness_result = self.azure_evaluators['groundedness'](
                        query=query,
                        context=query,
                        response=response_json
                    )
                    groundedness_score = groundedness_result.get('gpt_groundedness', 0)
                    evaluation_results['groundedness'] = groundedness_result
                    ground_span.set_attribute("evaluation.groundedness_score", groundedness_score)
                    logger.info(f"Groundedness evaluation: {groundedness_score}")
                except Exception as e:
                    logger.warning(f"Groundedness evaluation failed: {e}")
                    all_issues.append(f"Groundedness evaluation error: {str(e)}")
            
            # 3. F1 Score Evaluation
            f1_score = 0
            if 'ground_truth' in parsed_params:
                with tracer.start_as_current_span("f1_score_evaluator") as f1_span:
                    try:
                        f1_result = self.azure_evaluators['f1'](
                            response=response_json,
                            ground_truth=json.dumps(parsed_params['ground_truth'])
                        )
                        f1_score = f1_result.get('f1_score', 0)
                        evaluation_results['f1_score'] = f1_result
                        f1_span.set_attribute("evaluation.f1_score", f1_score)
                        logger.info(f"F1 score evaluation: {f1_score}")
                    except Exception as e:
                        logger.warning(f"F1 score evaluation failed: {e}")
                        all_issues.append(f"F1 score evaluation error: {str(e)}")
            
            # Calculate overall score
            overall_score = (
                relevance_score * 0.4 +
                groundedness_score * 0.4 +
                f1_score * 0.2
            )
            
            # Create evaluation result
            evaluation_result = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'query': query,
                'parsed_params': parsed_params,
                'response': response_json,
                'relevance_score': relevance_score,
                'groundedness_score': groundedness_score,
                'f1_score': f1_score,
                'evaluation_scores': {
                    'overall_score': round(overall_score, 3),
                    'relevance_score': round(relevance_score, 3),
                    'groundedness_score': round(groundedness_score, 3),
                    'f1_score': round(f1_score, 3)
                },
                'evaluator_results': evaluation_results,
                'issues': all_issues,
                'trace_id': trace_id,
                'span_id': span_id,
                'evaluation_metadata': {
                    'evaluator_version': '2.0-azure-sdk',
                    'azure_sdk_version': '1.11.2',
                    'model_used': self.llm_model,
                    'evaluation_time_ms': round((time.time() - start_time) * 1000, 2)
                }
            }
            
            # Store in Elasticsearch
            self._store_evaluation_result(evaluation_result)
            
            # Record metrics
            span.set_attribute("evaluation.overall_score", overall_score)
            span.set_attribute("evaluation.relevance_score", relevance_score)
            span.set_attribute("evaluation.groundedness_score", groundedness_score)
            span.set_attribute("evaluation.f1_score", f1_score)
            span.set_attribute("evaluation.issues_count", len(all_issues))
            span.set_attribute("evaluation.success", True)
            
            logger.info(f"Azure AI evaluation completed. Overall score: {overall_score:.3f}")
            return evaluation_result
    
    def _evaluate_with_custom_logic(self, query: str, parsed_params: Dict[str, Any], 
                                  trace_id: Optional[str], span_id: Optional[str], start_time: float) -> Dict[str, Any]:
        """Evaluate using custom logic as fallback"""
        with tracer.start_as_current_span("custom_evaluation") as span:
            # Run custom validation checks
            type_validation = self._validate_parameter_types(parsed_params)
            range_validation = self._validate_parameter_ranges(parsed_params)
            completeness_score = self._calculate_completeness_score(query, parsed_params)
            accuracy_score = self._calculate_accuracy_score(query, parsed_params)
            
            # Calculate overall score
            overall_score = (
                type_validation['score'] * 0.3 +
                range_validation['score'] * 0.2 +
                completeness_score * 0.3 +
                accuracy_score * 0.2
            )
            
            # Collect all issues
            all_issues = []
            all_issues.extend(type_validation['issues'])
            all_issues.extend(range_validation['issues'])
            
            # Create evaluation result
            evaluation_result = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'query': query,
                'parsed_params': parsed_params,
                'response': json.dumps(parsed_params),
                'relevance_score': 0,  # Not available in custom logic
                'groundedness_score': 0,  # Not available in custom logic
                'f1_score': 0,  # Not available in custom logic
                'evaluation_scores': {
                    'overall_score': round(overall_score, 3),
                    'type_validation_score': round(type_validation['score'], 3),
                    'range_validation_score': round(range_validation['score'], 3),
                    'completeness_score': round(completeness_score, 3),
                    'accuracy_score': round(accuracy_score, 3)
                },
                'issues': all_issues,
                'trace_id': trace_id,
                'span_id': span_id,
                'evaluation_metadata': {
                    'evaluator_version': '2.0-custom-fallback',
                    'model_used': self.llm_model,
                    'evaluation_time_ms': round((time.time() - start_time) * 1000, 2)
                }
            }
            
            # Store in Elasticsearch
            self._store_evaluation_result(evaluation_result)
            
            # Record metrics
            span.set_attribute("evaluation.overall_score", overall_score)
            span.set_attribute("evaluation.issues_count", len(all_issues))
            span.set_attribute("evaluation.success", True)
            
            logger.info(f"Custom evaluation completed. Overall score: {overall_score:.3f}")
            return evaluation_result
    
    def _validate_parameter_types(self, parsed_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that parameter types match expected types.
        
        Args:
            parsed_params: Parsed parameters to validate
            
        Returns:
            Dictionary with validation score and issues
        """
        issues = []
        correct_types = 0
        total_checked = 0
        
        for param_name, param_value in parsed_params.items():
            if param_name not in self.valid_parameters:
                continue
                
            total_checked += 1
            expected_type = self.valid_parameters[param_name]
            
            # Handle None values (allowed)
            if param_value is None:
                correct_types += 1
                continue
            
            # Check type compatibility
            if isinstance(expected_type, tuple):
                # Multiple types allowed
                if isinstance(param_value, expected_type):
                    correct_types += 1
                else:
                    issues.append(f"Parameter '{param_name}' has type {type(param_value).__name__}, expected one of {[t.__name__ for t in expected_type]}")
            else:
                # Single type expected
                if isinstance(param_value, expected_type):
                    correct_types += 1
                else:
                    issues.append(f"Parameter '{param_name}' has type {type(param_value).__name__}, expected {expected_type.__name__}")
        
        score = correct_types / total_checked if total_checked > 0 else 1.0
        
        return {
            'score': score,
            'issues': issues
        }
    
    def _validate_parameter_ranges(self, parsed_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that parameter values are within reasonable ranges.
        
        Args:
            parsed_params: Parsed parameters to validate
            
        Returns:
            Dictionary with validation score and issues
        """
        issues = []
        valid_ranges = 0
        total_checked = 0
        
        for param_name, param_value in parsed_params.items():
            if param_name not in self.parameter_ranges or param_value is None:
                continue
                
            total_checked += 1
            min_val, max_val = self.parameter_ranges[param_name]
            
            try:
                numeric_value = float(param_value)
                if min_val <= numeric_value <= max_val:
                    valid_ranges += 1
                else:
                    issues.append(f"Parameter '{param_name}' value {param_value} is outside valid range [{min_val}, {max_val}]")
            except (ValueError, TypeError):
                issues.append(f"Parameter '{param_name}' value {param_value} cannot be converted to numeric for range validation")
        
        # Special validation for distance_unit
        if 'distance_unit' in parsed_params and parsed_params['distance_unit'] is not None:
            total_checked += 1
            if parsed_params['distance_unit'].lower() in self.valid_distance_units:
                valid_ranges += 1
            else:
                issues.append(f"Parameter 'distance_unit' value '{parsed_params['distance_unit']}' is not valid. Expected one of: {self.valid_distance_units}")
        
        score = valid_ranges / total_checked if total_checked > 0 else 1.0
        
        return {
            'score': score,
            'issues': issues
        }
    
    def _calculate_completeness_score(self, query: str, parsed_params: Dict[str, Any]) -> float:
        """
        Calculate how many relevant parameters were extracted from the query.
        
        Args:
            query: Original user query
            parsed_params: Parsed parameters
            
        Returns:
            Completeness score between 0 and 1
        """
        # Define patterns to detect potential parameters in the query
        patterns = {
            'bedrooms': r'\b(\d+)\s*(?:bed|bedroom|beds)\b',
            'bathrooms': r'\b(\d+)\s*(?:bath|bathroom|baths)\b',
            'distance': r'\b(?:within|near|close to)\s*(\d+(?:\.\d+)?)\s*(?:mi|miles?|km|kilometers?)\b',
            'price': r'\b(?:under|below|less than|max|maximum)\s*\$?(\d+(?:,\d{3})*(?:k|K)?)\b',
            'square_footage': r'\b(\d+(?:,\d{3})*)\s*(?:sq\s*ft|square\s*feet|sqft)\b',
            'feature': r'\b(?:with|has|includes?)\s+([a-zA-Z\s]+?)(?:\s|$|,|\.)'
        }
        
        detected_params = set()
        for param_type, pattern in patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                detected_params.add(param_type)
        
        # Count how many detected parameters were actually extracted
        extracted_params = set()
        for param_name in detected_params:
            if param_name in parsed_params and parsed_params[param_name] is not None:
                extracted_params.add(param_name)
        
        # Also count other parameters that were extracted
        other_extracted = set(parsed_params.keys()) - detected_params - {'query'}
        other_extracted = {k for k in other_extracted if parsed_params[k] is not None}
        
        total_extractable = len(detected_params)
        total_extracted = len(extracted_params) + len(other_extracted)
        
        if total_extractable == 0:
            # If no parameters were detectable, score based on whether query was preserved
            return 1.0 if 'query' in parsed_params and parsed_params['query'] else 0.0
        
        # Score based on extraction rate
        extraction_rate = len(extracted_params) / total_extractable
        bonus_for_extra = min(0.2, len(other_extracted) * 0.1)  # Small bonus for extracting unexpected params
        
        return min(1.0, extraction_rate + bonus_for_extra)
    
    def _calculate_accuracy_score(self, query: str, parsed_params: Dict[str, Any]) -> float:
        """
        Calculate accuracy score based on whether extracted parameters make sense for the query.
        
        Args:
            query: Original user query
            parsed_params: Parsed parameters
            
        Returns:
            Accuracy score between 0 and 1
        """
        # Simple heuristics for accuracy checking
        accuracy_checks = []
        
        # Check if query text is preserved
        if 'query' in parsed_params and parsed_params['query']:
            if parsed_params['query'].lower() in query.lower() or any(word in query.lower() for word in parsed_params['query'].lower().split()):
                accuracy_checks.append(True)
            else:
                accuracy_checks.append(False)
        
        # Check if bedrooms/bathrooms are reasonable
        for param in ['bedrooms', 'bathrooms']:
            if param in parsed_params and parsed_params[param] is not None:
                try:
                    value = float(parsed_params[param])
                    if 1 <= value <= 20:  # Reasonable range
                        accuracy_checks.append(True)
                    else:
                        accuracy_checks.append(False)
                except (ValueError, TypeError):
                    accuracy_checks.append(False)
        
        # Check if price values are reasonable
        for param in ['home_price_min', 'home_price_max']:
            if param in parsed_params and parsed_params[param] is not None:
                try:
                    value = float(parsed_params[param])
                    if 0 <= value <= 10000000:  # Reasonable price range
                        accuracy_checks.append(True)
                    else:
                        accuracy_checks.append(False)
                except (ValueError, TypeError):
                    accuracy_checks.append(False)
        
        # Check if coordinates are reasonable (if present)
        if 'latitude' in parsed_params and 'longitude' in parsed_params:
            if parsed_params['latitude'] is not None and parsed_params['longitude'] is not None:
                try:
                    lat = float(parsed_params['latitude'])
                    lon = float(parsed_params['longitude'])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        accuracy_checks.append(True)
                    else:
                        accuracy_checks.append(False)
                except (ValueError, TypeError):
                    accuracy_checks.append(False)
        
        return sum(accuracy_checks) / len(accuracy_checks) if accuracy_checks else 1.0
    
    def _store_evaluation_result(self, evaluation_result: Dict[str, Any]) -> None:
        """
        Store evaluation result in Elasticsearch.
        
        Args:
            evaluation_result: Evaluation result to store
        """
        try:
            # Index the evaluation result with proper timeout
            response = self.es_client.index(
                index=self.eval_index_name,
                body=evaluation_result,
                timeout='30s'  # Explicit timeout
            )
            
            logger.info(f"Evaluation result stored in Elasticsearch with ID: {response['_id']}")
            
        except Exception as e:
            logger.error(f"Failed to store evaluation result in Elasticsearch: {e}")
            # Don't raise the exception - evaluation should continue even if storage fails
