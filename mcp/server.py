"""
MCP Home Finder Server implementation.
"""

import os
import logging
from typing import Any, Dict, List
from pathlib import Path
from dotenv import load_dotenv

from mcp.server import Server
from mcp.types import Tool, TextContent

from query_parser import QueryParser
from geocoder import AzureMapsGeocoder
from search_service import ElasticsearchSearchService

logger = logging.getLogger(__name__)

def create_home_finder_server() -> Server:
    """
    Create and configure the MCP Home Finder Server.
    
    Returns:
        Configured MCP Server instance
    """
    # Load environment variables
    env_path = Path(__file__).parent.parent / 'variables.env'
    load_dotenv(env_path)
    
    # Initialize services
    query_parser = QueryParser(
        llm_url=os.getenv('LLM_URL'),
        llm_model=os.getenv('LLM_MODEL'),
        llm_api_key=os.getenv('LLM_API_KEY')
    )
    
    # Debug logging for geocoder initialization
    azure_maps_key = os.getenv('AZURE_MAPS_SUBSCRIPTION_KEY')
    logger.info(f"Azure Maps key loaded: {azure_maps_key is not None}")
    logger.info(f"Azure Maps key length: {len(azure_maps_key) if azure_maps_key else 0}")
    logger.info(f"Azure Maps key starts with: {azure_maps_key[:10] if azure_maps_key else None}...")
    
    geocoder = AzureMapsGeocoder(
        subscription_key=azure_maps_key
    )
    
    search_service = ElasticsearchSearchService(
        es_url=os.getenv('ELASTICSEARCH_URL'),
        es_api_key=os.getenv('ELASTICSEARCH_API_KEY'),
        index_name=os.getenv('INDEX_NAME'),
        search_template_name=os.getenv('SEARCH_TEMPLATE_NAME')
    )
    
    # Create MCP server
    server = Server("home-finder")
    
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available tools."""
        return [
            Tool(
                name="parse_query",
                description="Parse natural language home search query to extract parameters like bedrooms, bathrooms, price, location, distance, etc.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query to parse"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="geocode_location",
                description="Convert a location name to latitude/longitude coordinates using Azure Maps",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location to geocode (e.g., 'Orlando FL', 'Miami Beach')"
                        }
                    },
                    "required": ["location"]
                }
            ),
            Tool(
                name="search_homes",
                description="Search for homes in Elasticsearch with structured parameters. Always include the original query text for semantic search.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Original search text for semantic matching"},
                        "latitude": {"type": "number"},
                        "longitude": {"type": "number"},
                        "distance": {"type": "number", "description": "Distance in miles"},
                        "bedrooms": {"type": "number"},
                        "bathrooms": {"type": "number"},
                        "square_footage": {"type": "number"},
                        "home_price_min": {"type": "number"},
                        "home_price_max": {"type": "number"},
                        "tax": {"type": "number"},
                        "maintenance": {"type": "number"},
                        "feature": {"type": "string"}
                    },
                    "required": ["query"]
                }
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls."""
        if name == "parse_query":
            return await parse_query_tool(arguments, query_parser)
        elif name == "geocode_location":
            return await geocode_location_tool(arguments, geocoder)
        elif name == "search_homes":
            return await search_homes_tool(arguments, search_service)
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    return server

async def parse_query_tool(
    arguments: Dict[str, Any],
    query_parser: QueryParser
) -> List[TextContent]:
    """
    Handle the parse_query tool call.
    
    Args:
        arguments: Tool arguments containing the user query
        query_parser: Query parser instance
        
    Returns:
        List of TextContent with parsed parameters as JSON
    """
    try:
        user_query = arguments.get("query", "")
        if not user_query:
            return [TextContent(
                type="text",
                text="Please provide a query to parse."
            )]
        
        logger.info(f"Parsing query: {user_query}")
        
        # Parse the query using LLM
        search_params = query_parser.parse_query(user_query)
        logger.info(f"Parsed search parameters: {search_params}")
        
        # Return the parsed parameters as JSON
        import json
        response_text = json.dumps(search_params, indent=2)
        
        return [TextContent(
            type="text",
            text=response_text
        )]
        
    except Exception as e:
        logger.error(f"Error in parse_query_tool: {e}")
        return [TextContent(
            type="text",
            text=f"An error occurred while parsing the query: {str(e)}"
        )]

async def geocode_location_tool(
    arguments: Dict[str, Any],
    geocoder: AzureMapsGeocoder
) -> List[TextContent]:
    """
    Handle the geocode_location tool call.
    
    Args:
        arguments: Tool arguments containing the location to geocode
        geocoder: Geocoder instance
        
    Returns:
        List of TextContent with geocoded coordinates
    """
    try:
        location_query = arguments.get("location", "")
        if not location_query:
            return [TextContent(
                type="text",
                text="Please provide a location to geocode."
            )]
        
        logger.info(f"Geocoding location: {location_query}")
        
        # Geocode the location
        location = geocoder.geocode(location_query)
        
        if location:
            response_text = f"Location: {location.address}\nLatitude: {location.latitude}\nLongitude: {location.longitude}\nConfidence: {location.confidence}"
            logger.info(f"Successfully geocoded '{location_query}' to {location.latitude}, {location.longitude}")
        else:
            response_text = f"Failed to geocode location: {location_query}"
            logger.warning(f"Failed to geocode location: {location_query}")
        
        return [TextContent(
            type="text",
            text=response_text
        )]
        
    except Exception as e:
        logger.error(f"Error in geocode_location_tool: {e}")
        return [TextContent(
            type="text",
            text=f"An error occurred while geocoding the location: {str(e)}"
        )]

async def search_homes_tool(
    arguments: Dict[str, Any],
    search_service: ElasticsearchSearchService
) -> List[TextContent]:
    """
    Handle the search_homes tool call.
    
    Args:
        arguments: Tool arguments containing structured search parameters
        search_service: Search service instance
        
    Returns:
        List of TextContent with search results
    """
    try:
        # Extract the original query (required for semantic search)
        original_query = arguments.get("query", "")
        if not original_query:
            return [TextContent(
                type="text",
                text="Please provide the original query text for semantic search."
            )]
        
        # Use all provided arguments as search parameters
        search_params = {k: v for k, v in arguments.items() if v is not None}
        
        logger.info(f"Searching homes with parameters: {search_params}")
        
        # Execute search
        results = search_service.search_homes(search_params)
        
        # Format and return results
        if not results:
            response_text = "No homes found matching your criteria. Try adjusting your search parameters."
        else:
            response_text = search_service.format_results_for_display(results)
        
        return [TextContent(
            type="text",
            text=response_text
        )]
        
    except Exception as e:
        logger.error(f"Error in search_homes_tool: {e}")
        return [TextContent(
            type="text",
            text=f"An error occurred while searching for homes: {str(e)}"
        )]

