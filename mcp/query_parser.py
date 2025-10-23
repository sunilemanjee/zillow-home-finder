"""
LLM-based query parser for extracting search parameters from natural language queries.
"""

import json
import logging
import requests
import time
from typing import Dict, Any, Optional

# OpenTelemetry imports
from opentelemetry import trace

logger = logging.getLogger(__name__)

# Get tracer from the global context
tracer = trace.get_tracer(__name__)

class QueryParser:
    """LLM-based parser for extracting search parameters from natural language queries."""
    
    def __init__(self, llm_url: str, llm_model: str, llm_api_key: str):
        """
        Initialize the query parser with LLM configuration.
        
        Args:
            llm_url: URL of the LLM API endpoint
            llm_model: Model name to use
            llm_api_key: API key for authentication
        """
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        
        # System prompt that defines the search template parameters
        self.system_prompt = """You are a real estate search assistant. Parse the user's natural language query about finding homes and extract the following parameters in JSON format.

Available search parameters:
- query: Semantic search text for property descriptions (string)
- distance: Distance from a location (number)
- distance_unit: Unit for distance - "mi" for miles, "km" for kilometers (string, defaults to "mi")
- location: Location name that needs to be geocoded (string) - DO NOT extract coordinates, only location names
- bedrooms: Minimum number of bedrooms (number)
- bathrooms: Minimum number of bathrooms (number)
- tax: Maximum tax assessed value (number)
- maintenance: Maximum HOA/maintenance fee (number)
- square_footage: Minimum square footage (number)
- home_price_min: Minimum home price (number)
- home_price_max: Maximum home price (number)
- feature: Property feature to search for (string)

IMPORTANT: When you detect a location name (like "Orlando FL", "Miami Beach", "New York City"), extract it as a "location" field. Do NOT try to provide latitude/longitude coordinates - those will be obtained by calling the geocode_location tool separately.

Examples:
- "homes within 10 miles of Orlando FL with 2 beds" -> {"query": "homes", "distance": 10, "distance_unit": "mi", "location": "Orlando FL", "bedrooms": 2}
- "homes within 15 kilometers of Miami" -> {"query": "homes", "distance": 15, "distance_unit": "km", "location": "Miami"}
- "3 bedroom house under $500k in Miami" -> {"query": "house", "bedrooms": 3, "home_price_max": 500000, "location": "Miami"}
- "condo with pool near downtown" -> {"query": "condo", "feature": "pool"}
- "homes in San Francisco with 2 bathrooms" -> {"query": "homes", "bathrooms": 2, "location": "San Francisco"}

Return ONLY valid JSON with the extracted parameters. Use null for missing parameters. Do not include explanations or additional text."""

    def parse_query(self, user_query: str) -> Dict[str, Any]:
        """
        Parse a natural language query into structured search parameters.
        
        Args:
            user_query: Natural language query from the user
            
        Returns:
            Dictionary with search parameters matching the search template
        """
        # Create span for query parsing
        with tracer.start_as_current_span("query_parser.parse_query") as span:
            span.set_attribute("query_parser.query", user_query[:500])  # Truncate long queries
            span.set_attribute("query_parser.query_length", len(user_query))
            span.set_attribute("query_parser.llm_model", self.llm_model)
            span.set_attribute("query_parser.llm_url", self.llm_url)
            span.set_attribute("query_parser.llm_endpoint", f"{self.llm_url}/chat/completions")
            
            start_time = time.time()
            
            try:
                logger.info(f"Parsing query: {user_query}")
                
                # Prepare the request payload
                payload = {
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500
                }
                
                headers = {
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json"
                }
                
                span.set_attribute("query_parser.temperature", 0.1)
                span.set_attribute("query_parser.max_tokens", 500)
                
                # Make request to LLM
                # Append /chat/completions to the base URL for direct HTTP requests
                llm_endpoint = f"{self.llm_url}/chat/completions"
                response = requests.post(
                    llm_endpoint,
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()
                
                span.set_attribute("query_parser.llm_response_status", response.status_code)
                
                # Extract response content
                data = response.json()
                content = data['choices'][0]['message']['content'].strip()
                
                span.set_attribute("query_parser.llm_response_length", len(content))
                
                # Parse JSON response
                try:
                    parsed_params = json.loads(content)
                    logger.info(f"Successfully parsed query into parameters: {parsed_params}")
                    
                    # Record successful parsing
                    span.set_attribute("query_parser.success", True)
                    span.set_attribute("query_parser.parsed_params_count", len(parsed_params))
                    span.set_attribute("query_parser.parsed_params", str(parsed_params)[:1000])  # Truncate long params
                    
                    # Record execution time
                    execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                    span.set_attribute("query_parser.execution_time_ms", execution_time)
                    
                    return parsed_params
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM response as JSON: {e}")
                    logger.error(f"LLM response: {content}")
                    span.record_exception(e)
                    span.set_attribute("query_parser.success", False)
                    span.set_attribute("query_parser.error", "json_parse_error")
                    span.set_attribute("query_parser.llm_response", content[:1000])  # Truncate long response
                    
                    return self._fallback_parse(user_query)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error during query parsing: {e}")
                span.record_exception(e)
                span.set_attribute("query_parser.success", False)
                span.set_attribute("query_parser.error", "network_error")
                return self._fallback_parse(user_query)
            except Exception as e:
                logger.error(f"Unexpected error during query parsing: {e}")
                span.record_exception(e)
                span.set_attribute("query_parser.success", False)
                span.set_attribute("query_parser.error", "unexpected_error")
                return self._fallback_parse(user_query)
    
    def _fallback_parse(self, user_query: str) -> Dict[str, Any]:
        """
        Fallback parsing using simple regex patterns when LLM fails.
        
        Args:
            user_query: Natural language query from the user
            
        Returns:
            Dictionary with basic search parameters
        """
        import re
        
        # Create span for fallback parsing
        with tracer.start_as_current_span("query_parser.fallback_parse") as span:
            span.set_attribute("query_parser.fallback_query", user_query[:500])  # Truncate long queries
            span.set_attribute("query_parser.fallback_query_length", len(user_query))
            
            logger.info("Using fallback parsing for query")
            
            params = {"query": user_query}
            extracted_params = []
            
            # Extract bedrooms
            bedroom_match = re.search(r'(\d+)\s*(?:bed|bedroom|beds)', user_query, re.IGNORECASE)
            if bedroom_match:
                params["bedrooms"] = int(bedroom_match.group(1))
                extracted_params.append("bedrooms")
            
            # Extract bathrooms
            bathroom_match = re.search(r'(\d+)\s*(?:bath|bathroom|baths)', user_query, re.IGNORECASE)
            if bathroom_match:
                params["bathrooms"] = int(bathroom_match.group(1))
                extracted_params.append("bathrooms")
            
            # Extract distance
            distance_match = re.search(r'within\s*(\d+)\s*miles?', user_query, re.IGNORECASE)
            if distance_match:
                params["distance"] = int(distance_match.group(1))
                extracted_params.append("distance")
            
            # Extract price
            price_match = re.search(r'under\s*\$?(\d+(?:,\d{3})*(?:k|K)?)', user_query, re.IGNORECASE)
            if price_match:
                price_str = price_match.group(1).replace(',', '')
                if price_str.lower().endswith('k'):
                    params["home_price_max"] = int(price_str[:-1]) * 1000
                else:
                    params["home_price_max"] = int(price_str)
                extracted_params.append("home_price_max")
            
            # Record fallback parsing results
            span.set_attribute("query_parser.fallback_success", True)
            span.set_attribute("query_parser.fallback_extracted_params", ",".join(extracted_params))
            span.set_attribute("query_parser.fallback_params_count", len(params))
            span.set_attribute("query_parser.fallback_params", str(params)[:1000])  # Truncate long params
            
            return params
