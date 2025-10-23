"""
MCP Home Finder Server implementation.
"""

import os
import logging
import time
from typing import Any, Dict, List
from pathlib import Path
from dotenv import load_dotenv

from mcp.server import Server
from mcp.types import Tool, TextContent
from elasticsearch import Elasticsearch

from query_parser import QueryParser
from geocoder import AzureMapsGeocoder
from search_service import ElasticsearchSearchService
# from query_evaluator import QueryParserEvaluator  # Removed - using offline evaluation

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.elasticsearch import ElasticsearchInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.trace import SpanContext, TraceFlags
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)

# Initialize OpenTelemetry
def initialize_opentelemetry():
    """Initialize OpenTelemetry with OTLP exporters and auto-instrumentation"""
    
    # Validate environment variables
    required_vars = [
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS", 
        "OTEL_RESOURCE_ATTRIBUTES"
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        print(f"[OTEL] Warning: Missing environment variables: {missing_vars}")
        print("[OTEL] OpenTelemetry will not be fully configured")
        return None, None
    
    try:
        # Create resource with attributes
        resource = Resource.create({
            "service.name": "mcp",
            "service.version": "1.0",
            "deployment.environment": "production"
        })
        
        # Configure OTLP exporters
        otlp_endpoint = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
        otlp_headers = os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
        
        # Parse headers - handle both comma-separated and single header formats
        # For gRPC, we need to ensure headers are in the correct format
        headers = {}
        if "," in otlp_headers:
            # Multiple headers separated by commas
            for header in otlp_headers.split(","):
                if "=" in header:
                    key, value = header.split("=", 1)
                    key = key.strip().lower()  # gRPC requires lowercase keys
                    value = value.strip()
                    headers[key] = value
        else:
            # Single header
            if "=" in otlp_headers:
                key, value = otlp_headers.split("=", 1)
                key = key.strip().lower()  # gRPC requires lowercase keys
                value = value.strip()
                headers[key] = value
        
        # Initialize span exporter
        span_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            headers=headers
        )
        
        # Initialize metric exporter
        metric_exporter = OTLPMetricExporter(
            endpoint=otlp_endpoint,
            headers=headers
        )
        
        # Set up tracer provider
        tracer_provider = TracerProvider(resource=resource)
        span_processor = BatchSpanProcessor(span_exporter)
        tracer_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(tracer_provider)
        
        # Set up meter provider
        metric_reader = PeriodicExportingMetricReader(
            exporter=metric_exporter,
            export_interval_millis=10000  # Export every 10 seconds
        )
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(meter_provider)
        
        # Auto-instrument libraries
        ElasticsearchInstrumentor().instrument(
            service_name="elasticsearch",
            service_version="9.1.1"
        )
        RequestsInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()
        
        print("[OTEL] OpenTelemetry initialized successfully")
        return trace.get_tracer(__name__), metrics.get_meter(__name__)
        
    except Exception as e:
        print(f"[OTEL] Error initializing OpenTelemetry: {str(e)}")
        return None, None

# Initialize OpenTelemetry
tracer, meter = initialize_opentelemetry()

# Create custom metrics
if meter:
    tool_calls_counter = meter.create_counter(
        name="mcp.tool_calls.total",
        description="Total number of tool calls made"
    )
    tool_execution_time_histogram = meter.create_histogram(
        name="mcp.tool_execution_time",
        description="Tool execution time distribution",
        unit="ms"
    )
    errors_counter = meter.create_counter(
        name="mcp.errors.total",
        description="Total number of errors encountered"
    )
    
    # Azure AI Evaluation metrics
    evaluation_counter = meter.create_counter(
        name="azure_ai.evaluations.total",
        description="Total number of Azure AI evaluations performed",
        unit="evaluations"
    )
    
    evaluation_score_histogram = meter.create_histogram(
        name="azure_ai.evaluation.score",
        description="Distribution of evaluation scores by evaluator type",
        unit="score"
    )
    
    evaluation_latency_histogram = meter.create_histogram(
        name="azure_ai.evaluation.latency",
        description="Azure AI evaluation latency by evaluator type",
        unit="ms"
    )
    
    evaluation_errors_counter = meter.create_counter(
        name="azure_ai.evaluation.errors",
        description="Total number of evaluation errors by type",
        unit="errors"
    )
else:
    # Create no-op metrics if OpenTelemetry is not available
    class NoOpMetric:
        def add(self, *args, **kwargs): pass
        def record(self, *args, **kwargs): pass
    
    tool_calls_counter = NoOpMetric()
    tool_execution_time_histogram = NoOpMetric()
    errors_counter = NoOpMetric()

def extract_trace_context_from_args(arguments: Dict[str, Any]):
    """
    Extract trace context from tool arguments if present.
    
    Args:
        arguments: Tool arguments that may contain _trace_context
        
    Returns:
        SpanContext if found, None otherwise
    """
    trace_context = arguments.get('_trace_context')
    if not trace_context:
        return None
    
    try:
        trace_id = int(trace_context['trace_id'], 16)
        span_id = int(trace_context['span_id'], 16)
        trace_flags = TraceFlags(trace_context.get('trace_flags', 1))
        
        span_context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=trace_flags,
            is_remote=True
        )
        
        print(f"[MCP] Extracted trace context: trace_id={trace_context['trace_id']}, span_id={trace_context['span_id']}")
        return span_context
    except (KeyError, ValueError, TypeError) as e:
        print(f"[MCP] Error extracting trace context: {e}")
        return None

def create_tool_span_with_context(tool_name: str, arguments: Dict[str, Any]):
    """
    Create a span for tool execution with proper parent context if available.
    
    Args:
        tool_name: Name of the tool being executed
        arguments: Tool arguments that may contain trace context
        
    Returns:
        Span context manager
    """
    if not tracer:
        # Return a no-op context manager if no tracer
        class NoOpContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return NoOpContext()
    
    # Extract trace context from arguments
    parent_span_context = extract_trace_context_from_args(arguments)
    
    if parent_span_context:
        # Create span with parent context
        span = tracer.start_span(
            f"{tool_name}_tool",
            context=trace.set_span_in_context(trace.NonRecordingSpan(parent_span_context))
        )
        print(f"[MCP] Created span with parent context for {tool_name}")
    else:
        # Create span without parent context
        span = tracer.start_span(f"{tool_name}_tool")
        print(f"[MCP] Created span without parent context for {tool_name}")
    
    return span

def reconstruct_user_query(arguments: Dict[str, Any]) -> str:
    """
    Reconstruct the full user query from conversation history and current context.
    
    Args:
        arguments: Tool arguments that may contain conversation history
        
    Returns:
        Reconstructed full user query string
    """
    # Check if we have conversation history
    conversation_history = arguments.get('_conversation_history', [])
    current_query = arguments.get('query', '')
    
    if not conversation_history:
        # No history available, just return the current query
        return current_query
    
    # Extract user messages from conversation history
    user_messages = []
    for msg in conversation_history:
        if isinstance(msg, dict) and msg.get('role') == 'user':
            content = msg.get('content', '')
            if content:
                user_messages.append(content)
    
    # Combine all user messages into a single query
    if user_messages:
        # Join with spaces, handling follow-up patterns
        full_query = ' '.join(user_messages)
        # Clean up common follow-up patterns
        full_query = full_query.replace(' and ', ' ').replace(' also ', ' ')
        return full_query.strip()
    
    # Fallback to current query if no user messages found
    return current_query

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
    
    # Initialize Elasticsearch client for source index storage
    from elasticsearch import Elasticsearch
    es_client = Elasticsearch(
        hosts=[os.getenv('ELASTICSEARCH_O11Y_URL')],
        api_key=os.getenv('ELASTICSEARCH_O11Y_API_KEY'),
        verify_certs=True
    )
    
    # Create MCP server
    server = Server("home-finder")
    
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available tools."""
        return [
            Tool(
                name="parse_query",
                description="Parse natural language home search query to extract parameters like bedrooms, bathrooms, price, location names, distance, etc. IMPORTANT: This tool extracts location NAMES (like 'Orlando FL', 'Miami Beach') but NOT coordinates. If a location is detected in the query, you MUST call the geocode_location tool next to get the latitude/longitude coordinates before calling search_homes.",
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
                description="CRITICAL WORKFLOW STEP: Convert a location name to latitude/longitude coordinates using Azure Maps. This tool MUST be called whenever parse_query returns a 'location' field in its output. The workflow is: 1) parse_query extracts location names, 2) geocode_location converts them to coordinates, 3) search_homes uses those coordinates for distance-based searches. Do NOT skip this step if you need to search within a distance of a location.",
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
                description="Search for homes in Elasticsearch with structured parameters. Always include the original query text for semantic search. IMPORTANT: For distance-based searches (when 'distance' parameter is provided), you MUST provide latitude and longitude coordinates. These coordinates should come from the geocode_location tool - do NOT call this tool without coordinates if you need to search within a distance of a location.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Original search text for semantic matching"},
                        "latitude": {"type": "number", "description": "Latitude coordinate (required for distance-based searches, obtained from geocode_location tool)"},
                        "longitude": {"type": "number", "description": "Longitude coordinate (required for distance-based searches, obtained from geocode_location tool)"},
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
            return await search_homes_tool(arguments, search_service, es_client)
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
    # Create span for tool execution with context
    with create_tool_span_with_context("parse_query", arguments) as span:
        # Debug: Log current trace context
        current_span = trace.get_current_span()
        if current_span:
            span_context = current_span.get_span_context()
            print(f"[MCP] parse_query_tool - Trace ID: {span_context.trace_id}, Span ID: {span_context.span_id}")
            print(f"[MCP] parse_query_tool - Trace flags: {span_context.trace_flags}")
            print(f"[MCP] parse_query_tool - Is remote: {span_context.is_remote}")
        else:
            print("[MCP] parse_query_tool - No active span found")
        
        if span:
            span.set_attribute("tool.name", "parse_query")
            span.set_attribute("tool.arguments", str(arguments)[:1000])  # Truncate long arguments
            
            # Increment tool calls counter
            tool_calls_counter.add(1, {"tool_name": "parse_query"})
        
        start_time = time.time()
        
        try:
            user_query = arguments.get("query", "")
            if not user_query:
                if span:
                    span.set_attribute("tool.error", "empty_query")
                return [TextContent(
                    type="text",
                    text="Please provide a query to parse."
                )]
            
            if span:
                span.set_attribute("tool.query_length", len(user_query))
                span.set_attribute("tool.query_text", user_query[:500])  # Truncate long queries
            
            logger.info(f"Parsing query: {user_query}")
            
            # Parse the query using LLM
            search_params = query_parser.parse_query(user_query)
            logger.info(f"Parsed search parameters: {search_params}")
            
            # Return the parsed parameters as JSON
            import json
            response_text = json.dumps(search_params, indent=2)
            
            # Record execution time
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            if span:
                tool_execution_time_histogram.record(execution_time, {"tool_name": "parse_query"})
                span.set_attribute("tool.execution_time_ms", execution_time)
                span.set_attribute("tool.success", True)
                span.set_attribute("tool.response_size", len(response_text))
            
            return [TextContent(
                type="text",
                text=response_text
            )]
            
        except Exception as e:
            logger.error(f"Error in parse_query_tool: {e}")
            if span:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.set_attribute("tool.success", False)
                errors_counter.add(1, {"error_type": type(e).__name__, "tool_name": "parse_query"})
            
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
    # Create span for tool execution with context
    with create_tool_span_with_context("geocode_location", arguments) as span:
        # Debug: Log current trace context
        current_span = trace.get_current_span()
        if current_span:
            span_context = current_span.get_span_context()
            print(f"[MCP] geocode_location_tool - Trace ID: {span_context.trace_id}, Span ID: {span_context.span_id}")
            print(f"[MCP] geocode_location_tool - Trace flags: {span_context.trace_flags}")
            print(f"[MCP] geocode_location_tool - Is remote: {span_context.is_remote}")
        else:
            print("[MCP] geocode_location_tool - No active span found")
        
        if span:
            span.set_attribute("tool.name", "geocode_location")
            span.set_attribute("tool.arguments", str(arguments)[:1000])  # Truncate long arguments
            
            # Increment tool calls counter
            tool_calls_counter.add(1, {"tool_name": "geocode_location"})
        
        start_time = time.time()
        
        try:
            location_query = arguments.get("location", "")
            if not location_query:
                if span:
                    span.set_attribute("tool.error", "empty_location")
                return [TextContent(
                    type="text",
                    text="Please provide a location to geocode."
                )]
            
            if span:
                span.set_attribute("tool.location_query", location_query)
            
            logger.info(f"Geocoding location: {location_query}")
            
            # Geocode the location
            location = geocoder.geocode(location_query)
            
            if location:
                response_text = f"Location: {location.address}\nLatitude: {location.latitude}\nLongitude: {location.longitude}\nConfidence: {location.confidence}"
                logger.info(f"Successfully geocoded '{location_query}' to {location.latitude}, {location.longitude}")
                
                # Record successful geocoding attributes
                if span:
                    span.set_attribute("tool.success", True)
                    span.set_attribute("tool.latitude", location.latitude)
                    span.set_attribute("tool.longitude", location.longitude)
                    span.set_attribute("tool.confidence", location.confidence)
                    span.set_attribute("tool.address", location.address)
            else:
                response_text = f"Failed to geocode location: {location_query}"
                logger.warning(f"Failed to geocode location: {location_query}")
                if span:
                    span.set_attribute("tool.success", False)
                    span.set_attribute("tool.error", "geocoding_failed")
            
            # Record execution time
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            if span:
                tool_execution_time_histogram.record(execution_time, {"tool_name": "geocode_location"})
                span.set_attribute("tool.execution_time_ms", execution_time)
                span.set_attribute("tool.response_size", len(response_text))
            
            return [TextContent(
                type="text",
                text=response_text
            )]
            
        except Exception as e:
            logger.error(f"Error in geocode_location_tool: {e}")
            if span:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.set_attribute("tool.success", False)
                errors_counter.add(1, {"error_type": type(e).__name__, "tool_name": "geocode_location"})
            
            return [TextContent(
                type="text",
                text=f"An error occurred while geocoding the location: {str(e)}"
            )]

async def search_homes_tool(
    arguments: Dict[str, Any],
    search_service: ElasticsearchSearchService,
    es_client: Elasticsearch = None
) -> List[TextContent]:
    """
    Handle the search_homes tool call.
    
    Args:
        arguments: Tool arguments containing structured search parameters
        search_service: Search service instance
        
    Returns:
        List of TextContent with search results
    """
    # Create span for tool execution with context
    with create_tool_span_with_context("search_homes", arguments) as span:
        # Debug: Log current trace context
        current_span = trace.get_current_span()
        if current_span:
            span_context = current_span.get_span_context()
            print(f"[MCP] search_homes_tool - Trace ID: {span_context.trace_id}, Span ID: {span_context.span_id}")
            print(f"[MCP] search_homes_tool - Trace flags: {span_context.trace_flags}")
            print(f"[MCP] search_homes_tool - Is remote: {span_context.is_remote}")
        else:
            print("[MCP] search_homes_tool - No active span found")
        
        if span:
            span.set_attribute("tool.name", "search_homes")
            span.set_attribute("tool.arguments", str(arguments)[:1000])  # Truncate long arguments
            
            # Increment tool calls counter
            tool_calls_counter.add(1, {"tool_name": "search_homes"})
        
        start_time = time.time()
        
        try:
            # Extract the original query (required for semantic search)
            original_query = arguments.get("query", "")
            if not original_query:
                if span:
                    span.set_attribute("tool.error", "empty_query")
                return [TextContent(
                    type="text",
                    text="Please provide the original query text for semantic search."
                )]
            
            # Use all provided arguments as search parameters (excluding trace context)
            search_params = {k: v for k, v in arguments.items() if v is not None and k != '_trace_context'}
            
            # Record search parameters as span attributes
            if span:
                span.set_attribute("tool.query", original_query)
                span.set_attribute("tool.query_length", len(original_query))
                span.set_attribute("tool.search_params_count", len(search_params))
                
                # Record specific search parameters
                for key, value in search_params.items():
                    if key != "query":  # Don't duplicate the query
                        span.set_attribute(f"tool.search_param.{key}", str(value))
            
            logger.info(f"Searching homes with parameters: {search_params}")
            
            # Store query data for offline evaluation
            if es_client is not None:
                try:
                    # Reconstruct the full user query from conversation history
                    full_user_query = reconstruct_user_query(arguments)
                    
                    # Extract trace context
                    trace_context = arguments.get('_trace_context', {})
                    trace_id = trace_context.get('trace_id')
                    span_id = trace_context.get('span_id')
                    
                    # Extract session ID from conversation history if available
                    session_id = "unknown"
                    conversation_history = arguments.get('_conversation_history', [])
                    
                    # Create document for offline evaluation
                    eval_source_doc = {
                        "query": full_user_query,
                        "parsed_params": search_params,
                        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "session_id": session_id,
                        "evaluated": False,
                        "conversation_history": conversation_history
                    }
                    
                    # Store in EVAL_SOURCE_INDEX_NAME
                    eval_source_index = os.getenv('EVAL_SOURCE_INDEX_NAME', 'eval_source')
                    es_client.index(
                        index=eval_source_index,
                        body=eval_source_doc
                    )
                    
                    logger.info(f"Stored query data for offline evaluation in {eval_source_index}")
                    
                    if span:
                        span.set_attribute("tool.eval_source_stored", True)
                        span.set_attribute("tool.eval_source_index", eval_source_index)
                    
                except Exception as storage_error:
                    logger.warning(f"Failed to store query data for evaluation: {storage_error}")
                    if span:
                        span.set_attribute("tool.eval_source_stored", False)
                        span.set_attribute("tool.eval_source_error", str(storage_error))
            else:
                logger.warning("Elasticsearch client not available for storing evaluation data")
                if span:
                    span.set_attribute("tool.eval_source_stored", False)
                    span.set_attribute("tool.eval_source_error", "es_client_not_available")
            
            # Execute search
            results = search_service.search_homes(search_params)
            
            # Record search results
            result_count = len(results) if results else 0
            if span:
                span.set_attribute("tool.result_count", result_count)
                span.set_attribute("tool.success", True)
            
            # Format and return results
            if not results:
                response_text = "No homes found matching your criteria. Try adjusting your search parameters."
            else:
                response_text = search_service.format_results_for_display(results)
            
            # Record execution time
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            if span:
                tool_execution_time_histogram.record(execution_time, {"tool_name": "search_homes"})
                span.set_attribute("tool.execution_time_ms", execution_time)
                span.set_attribute("tool.response_size", len(response_text))
            
            return [TextContent(
                type="text",
                text=response_text
            )]
            
        except Exception as e:
            logger.error(f"Error in search_homes_tool: {e}")
            if span:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.set_attribute("tool.success", False)
                errors_counter.add(1, {"error_type": type(e).__name__, "tool_name": "search_homes"})
            
            return [TextContent(
                type="text",
                text=f"An error occurred while searching for homes: {str(e)}"
            )]

# evaluate_query_parse_tool function removed - using offline evaluation instead
