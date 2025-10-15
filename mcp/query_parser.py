"""
LLM-based query parser for extracting search parameters from natural language queries.
"""

import json
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

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
- latitude: Latitude coordinate (number)
- longitude: Longitude coordinate (number)
- bedrooms: Minimum number of bedrooms (number)
- bathrooms: Minimum number of bathrooms (number)
- tax: Maximum tax assessed value (number)
- maintenance: Maximum HOA/maintenance fee (number)
- square_footage: Minimum square footage (number)
- home_price_min: Minimum home price (number)
- home_price_max: Maximum home price (number)
- feature: Property feature to search for (string)

Examples:
- "homes within 10 miles of Orlando FL with 2 beds" -> {"query": "homes", "distance": 10, "distance_unit": "mi", "latitude": 28.5383, "longitude": -81.3792, "bedrooms": 2}
- "homes within 15 kilometers of Miami" -> {"query": "homes", "distance": 15, "distance_unit": "km", "latitude": 25.7617, "longitude": -80.1918}
- "3 bedroom house under $500k in Miami" -> {"query": "house", "bedrooms": 3, "home_price_max": 500000}
- "condo with pool near downtown" -> {"query": "condo", "feature": "pool"}

Return ONLY valid JSON with the extracted parameters. Use null for missing parameters. Do not include explanations or additional text."""

    def parse_query(self, user_query: str) -> Dict[str, Any]:
        """
        Parse a natural language query into structured search parameters.
        
        Args:
            user_query: Natural language query from the user
            
        Returns:
            Dictionary with search parameters matching the search template
        """
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
            
            # Make request to LLM
            response = requests.post(
                self.llm_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            # Extract response content
            data = response.json()
            content = data['choices'][0]['message']['content'].strip()
            
            # Parse JSON response
            try:
                parsed_params = json.loads(content)
                logger.info(f"Successfully parsed query into parameters: {parsed_params}")
                return parsed_params
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.error(f"LLM response: {content}")
                return self._fallback_parse(user_query)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during query parsing: {e}")
            return self._fallback_parse(user_query)
        except Exception as e:
            logger.error(f"Unexpected error during query parsing: {e}")
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
        
        logger.info("Using fallback parsing for query")
        
        params = {"query": user_query}
        
        # Extract bedrooms
        bedroom_match = re.search(r'(\d+)\s*(?:bed|bedroom|beds)', user_query, re.IGNORECASE)
        if bedroom_match:
            params["bedrooms"] = int(bedroom_match.group(1))
        
        # Extract bathrooms
        bathroom_match = re.search(r'(\d+)\s*(?:bath|bathroom|baths)', user_query, re.IGNORECASE)
        if bathroom_match:
            params["bathrooms"] = int(bathroom_match.group(1))
        
        # Extract distance
        distance_match = re.search(r'within\s*(\d+)\s*miles?', user_query, re.IGNORECASE)
        if distance_match:
            params["distance"] = int(distance_match.group(1))
        
        # Extract price
        price_match = re.search(r'under\s*\$?(\d+(?:,\d{3})*(?:k|K)?)', user_query, re.IGNORECASE)
        if price_match:
            price_str = price_match.group(1).replace(',', '')
            if price_str.lower().endswith('k'):
                params["home_price_max"] = int(price_str[:-1]) * 1000
            else:
                params["home_price_max"] = int(price_str)
        
        return params
