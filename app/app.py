import json
from mcp import ClientSession
from mcp.types import TextContent, ImageContent
import os
import re
from aiohttp import ClientSession, ClientError
import chainlit as cl
from openai import AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI
import traceback
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import asyncio
import sniffio
import time
from typing import Optional

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.elasticsearch import ElasticsearchInstrumentor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import inject
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

load_dotenv("../variables.env")

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
            "service.name": "home-finder",
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
        OpenAIInstrumentor().instrument()
        ElasticsearchInstrumentor().instrument(
            service_name="elasticsearch",
            service_version="9.1.1"
        )
        AioHttpClientInstrumentor().instrument()
        AsyncioInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()
        
        print("[OTEL] OpenTelemetry initialized successfully")
        return trace.get_tracer(__name__), metrics.get_meter(__name__)
        
    except Exception as e:
        print(f"[OTEL] Error initializing OpenTelemetry: {str(e)}")
        return None, None

# Initialize OpenTelemetry
tracer, meter = initialize_opentelemetry()

# Test function removed - no longer needed for production

# Create custom metrics
if meter:
    messages_counter = meter.create_counter(
        name="chat.messages.total",
        description="Total number of messages processed"
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
    tool_calls_counter = meter.create_counter(
        name="chat.tool_calls.total", 
        description="Total number of tool calls made"
    )
    response_time_histogram = meter.create_histogram(
        name="chat.response_time",
        description="Response generation time distribution",
        unit="ms"
    )
    errors_counter = meter.create_counter(
        name="chat.errors.total",
        description="Total number of errors encountered"
    )
    message_length_histogram = meter.create_histogram(
        name="chat.message_length",
        description="Message length distribution",
        unit="chars"
    )
else:
    # Create no-op metrics if OpenTelemetry is not available
    class NoOpMetric:
        def add(self, *args, **kwargs): pass
        def record(self, *args, **kwargs): pass
    
    messages_counter = NoOpMetric()
    tool_calls_counter = NoOpMetric()
    response_time_histogram = NoOpMetric()
    errors_counter = NoOpMetric()
    message_length_histogram = NoOpMetric()

# Initialize Elasticsearch client with credentials
es_client = Elasticsearch(
    os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"),
    api_key=os.environ.get("ELASTICSEARCH_API_KEY"),
    request_timeout=300
)

SYSTEM_PROMPT = """You are a helpful home finder assistant. You have access to tools that can help you search for properties.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags.
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.

Example tool call:
<tool_call>
{"name": "search_properties", "arguments": {"location": "New York", "max_price": 500000}}
</tool_call>

# Important Instructions for Search Results
When you receive search results from the search_homes tool:
1. Display ALL results exactly as returned - do not filter or hide any properties
2. Do not add commentary about "other homes found" or ask if the user wants to see more
3. Present the complete list of properties that match the search criteria
4. The search results already include all properties that meet the minimum requirements (e.g., "2 bedrooms" means 2 or more bedrooms)
5. Do NOT say things like "The other homes found have 3 or more bedrooms" or "Would you like to see homes with 3 bedrooms as well"
6. Simply present the results and ask if the user needs help with anything else about these properties
7. Remember: if someone asks for "2 bedrooms", they want to see ALL homes with 2 or more bedrooms, not just exactly 2 bedrooms"""


class ChatClient:
    def __init__(self) -> None:
        # Create span for client initialization
        if tracer:
            with tracer.start_as_current_span("chat_client_init") as span:
                span.set_attribute("llm.model", os.environ.get("LLM_MODEL", "unknown"))
                span.set_attribute("llm.base_url", os.environ.get("LLM_URL", "unknown"))
                
                # Initialize with LLM configuration from variables.env
                self.client = AsyncOpenAI(
                    base_url=os.environ["LLM_URL"],
                    api_key=os.environ["LLM_API_KEY"]
                )
                self.model = os.environ["LLM_MODEL"]
                self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                self.system_prompt = SYSTEM_PROMPT
                self.active_streams = []  # Track active response streams
        else:
            # Initialize with LLM configuration from variables.env
            self.client = AsyncOpenAI(
                base_url=os.environ["LLM_URL"],
                api_key=os.environ["LLM_API_KEY"]
            )
            self.model = os.environ["LLM_MODEL"]
            self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            self.system_prompt = SYSTEM_PROMPT
            self.active_streams = []  # Track active response streams
        
    async def _cleanup_streams(self):
        """Helper method to clean up all active streams"""
        for stream in self.active_streams[:]:  # Create a copy of the list to avoid modification during iteration
            try:
                if hasattr(stream, 'aclose'):
                    try:
                        await stream.aclose()
                    except (RuntimeError, asyncio.CancelledError, sniffio._impl.AsyncLibraryNotFoundError) as e:
                        print(f"[Cleanup] Ignoring error during aclose: {str(e)}")
                elif hasattr(stream, 'close'):
                    try:
                        await stream.close()
                    except (RuntimeError, asyncio.CancelledError, sniffio._impl.AsyncLibraryNotFoundError) as e:
                        print(f"[Cleanup] Ignoring error during close: {str(e)}")
                # Add specific handling for HTTP streams
                elif hasattr(stream, '_stream'):
                    try:
                        await stream._stream.aclose()
                    except (RuntimeError, asyncio.CancelledError, sniffio._impl.AsyncLibraryNotFoundError) as e:
                        print(f"[Cleanup] Ignoring error during _stream.aclose: {str(e)}")
                # Add specific handling for HTTP11ConnectionByteStream
                elif hasattr(stream, '__aiter__'):
                    try:
                        await stream.aclose()
                    except (RuntimeError, asyncio.CancelledError, sniffio._impl.AsyncLibraryNotFoundError) as e:
                        print(f"[Cleanup] Ignoring error during __aiter__ aclose: {str(e)}")
            except Exception as e:
                # Log the error but don't raise it
                print(f"[Cleanup] Error during stream cleanup: {str(e)}")
            finally:
                # Always remove the stream from active_streams
                if stream in self.active_streams:
                    self.active_streams.remove(stream)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cleanup_streams()
        
    def _parse_json_response(self, content):
        """Parse JSON response from content field and convert to tool call format"""
        try:
            # Clean up the content string
            content = content.strip()
            if not content:  # Handle empty content
                print("[JSON] Empty content received")
                return None
                
            if content.startswith('<tool_call>'):
                content = content[11:]
            if content.endswith('</tool_call>'):
                content = content[:-12]
            
            # Additional content validation
            if not content or content.isspace():
                print("[JSON] Content is empty or whitespace after cleanup")
                return None
                
            # Try to parse the JSON
            try:
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    print(f"[JSON] Parsed content is not a dictionary: {type(parsed)}")
                    return None
                    
                if "name" in parsed and "arguments" in parsed:
                    return {
                        "id": f"call_{hash(str(parsed))}",  # Generate a unique ID
                        "type": "function",
                        "function": {
                            "name": parsed["name"],
                            "arguments": json.dumps(parsed["arguments"])
                        }
                    }
                else:
                    print(f"[JSON] Missing required fields in parsed content: {parsed}")
                    return None
            except json.JSONDecodeError as e:
                print(f"[JSON] Error parsing JSON content: {str(e)}")
                print(f"[JSON] Raw content: {content}")
                return None
        except Exception as e:
            print(f"[JSON] Unexpected error in _parse_json_response: {str(e)}")
            print(f"[JSON] Content that caused error: {content}")
            return None

    async def process_response_stream(self, response_stream, tools, temperature=0):
        """
        Process response stream to handle function calls without recursion.
        """
        # Create span for stream processing
        if tracer:
            with tracer.start_as_current_span("process_response_stream") as span:
                span.set_attribute("llm.temperature", temperature)
                span.set_attribute("chat.tools_count", len(tools))
                
                function_arguments = ""
                function_name = ""
                tool_call_id = ""
                is_collecting_function_args = False
                collected_messages = []
                tool_calls = []
                tool_called = False
                error_occurred = False
                tokens_collected = 0
                
                print(f"[Stream] Starting new response stream processing")
                # Clean up any existing streams before adding new one
                await self._cleanup_streams()
                # Add to active streams for cleanup if needed
                self.active_streams.append(response_stream)
                print(f"[Stream] Added to active streams. Total active streams: {len(self.active_streams)}")
                
                try:
                    print("[Stream] Beginning to iterate over response stream")
                    async with asyncio.timeout(30):  # Add timeout to prevent hanging connections
                        try:
                            async for part in response_stream:
                                if not part or not part.choices:
                                    print("[Stream] Received empty part or choices, continuing")
                                    continue
                                    
                                delta = part.choices[0].delta
                                finish_reason = part.choices[0].finish_reason
                                
                                # Process assistant content
                                if delta and delta.content:
                                    collected_messages.append(delta.content)
                                    tokens_collected += 1
                                    yield delta.content
                                
                                # Handle tool calls
                                if delta and delta.tool_calls:
                                    print(f"[Stream] Processing tool calls: {delta.tool_calls}")
                                    span.add_event("tool_calls_detected", {"count": len(delta.tool_calls)})
                                    for tc in delta.tool_calls:
                                        if len(tool_calls) <= tc.index:
                                            tool_calls.append({
                                                "id": "", "type": "function",
                                                "function": {"name": "", "arguments": ""}
                                            })
                                        tool_calls[tc.index] = {
                                            "id": (tool_calls[tc.index]["id"] + (tc.id or "")),
                                            "type": "function",
                                            "function": {
                                                "name": (tool_calls[tc.index]["function"]["name"] + (tc.function.name or "")),
                                                "arguments": (tool_calls[tc.index]["function"]["arguments"] + (tc.function.arguments or ""))
                                            }
                                        }
                                
                                # Check if we've reached the end of a tool call
                                if finish_reason == "tool_calls" and tool_calls:
                                    print(f"[Stream] Tool calls completed. Processing {len(tool_calls)} tool calls")
                                    span.add_event("tool_calls_completed", {"count": len(tool_calls)})
                                    for tool_call in tool_calls:
                                        await self._handle_tool_call(
                                            tool_call["function"]["name"],
                                            tool_call["function"]["arguments"],
                                            tool_call["id"]
                                        )
                                    tool_called = True
                                    break
                                
                                # Check if we've reached the end of assistant's response
                                if finish_reason == "stop":
                                    print("[Stream] Received stop signal, processing final content")
                                    span.add_event("response_completed", {"finish_reason": "stop"})
                                    # Try to parse the final content as a tool call
                                    final_content = ''.join([msg for msg in collected_messages if msg is not None])
                                    if final_content.strip():
                                        # First check if content looks like JSON (starts with { or [)
                                        if final_content.strip().startswith(('{', '[')):
                                            try:
                                                tool_call = self._parse_json_response(final_content)
                                                if tool_call:
                                                    print(f"[Stream] Parsed final content as tool call: {tool_call['function']['name']}")
                                                    await self._handle_tool_call(
                                                        tool_call["function"]["name"],
                                                        tool_call["function"]["arguments"],
                                                        tool_call["id"]
                                                    )
                                                    tool_called = True
                                                else:
                                                    print("[Stream] Content was not a valid tool call, adding as assistant message")
                                                    self.messages.append({"role": "assistant", "content": final_content})
                                            except json.JSONDecodeError as json_err:
                                                print(f"[Stream] JSON parsing error in final content: {json_err}")
                                                print(f"[Stream] Raw final content: {final_content}")
                                                span.record_exception(json_err)
                                                # If we have a tool call pending, don't add the content as a message
                                                if not tool_called:
                                                    self.messages.append({"role": "assistant", "content": final_content})
                                        else:
                                            # Content is not JSON, treat as regular message
                                            print("[Stream] Final content is not JSON, adding as assistant message")
                                            self.messages.append({"role": "assistant", "content": final_content})
                                    elif tool_called:
                                        # If we had a tool call but no content, add a default message
                                        self.messages.append({"role": "assistant", "content": "I've processed your request. Is there anything else you'd like to know about these properties?"})
                                    break
                        except asyncio.CancelledError:
                            print("[Stream] Stream cancelled")
                            span.add_event("stream_cancelled")
                            raise
                        finally:
                            # Ensure proper cleanup of the stream
                            if response_stream in self.active_streams:
                                self.active_streams.remove(response_stream)
                                try:
                                    if hasattr(response_stream, 'aclose'):
                                        await response_stream.aclose()
                                    elif hasattr(response_stream, 'close'):
                                        await response_stream.close()
                                except (RuntimeError, asyncio.CancelledError, sniffio._impl.AsyncLibraryNotFoundError) as e:
                                    print(f"[Stream] Error during stream cleanup: {str(e)}")
                except asyncio.TimeoutError:
                    print("[Stream] Stream connection timed out")
                    error_occurred = True
                    span.add_event("stream_timeout")
                    span.set_status(trace.Status(trace.StatusCode.ERROR, "Stream timeout"))
                except Exception as e:
                    print(f"[Stream] Error in process_response_stream: {str(e)}")
                    print(f"[Stream] Error type: {type(e)}")
                    traceback.print_exc()
                    error_occurred = True
                    self.last_error = str(e)
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    errors_counter.add(1, {"error_type": type(e).__name__})
                finally:
                    # Store result in instance variables
                    self.tool_called = tool_called
                    self.last_function_name = function_name if tool_called else None
                    
                    # Set span attributes
                    span.set_attribute("chat.tokens_collected", tokens_collected)
                    span.set_attribute("chat.tool_called", tool_called)
                    span.set_attribute("chat.error_occurred", error_occurred)
                    if self.last_function_name:
                        span.set_attribute("tool.name", self.last_function_name)
                    
                    print(f"[Stream] Stream processing completed. Tool called: {tool_called}, Function name: {self.last_function_name}, Error occurred: {error_occurred}")
        else:
            # Fallback without tracing
            function_arguments = ""
            function_name = ""
            tool_call_id = ""
            is_collecting_function_args = False
            collected_messages = []
            tool_calls = []
            tool_called = False
            error_occurred = False
            
            print(f"[Stream] Starting new response stream processing")
            # Clean up any existing streams before adding new one
            await self._cleanup_streams()
            # Add to active streams for cleanup if needed
            self.active_streams.append(response_stream)
            print(f"[Stream] Added to active streams. Total active streams: {len(self.active_streams)}")
            
            try:
                print("[Stream] Beginning to iterate over response stream")
                async with asyncio.timeout(30):  # Add timeout to prevent hanging connections
                    try:
                        async for part in response_stream:
                            if not part or not part.choices:
                                print("[Stream] Received empty part or choices, continuing")
                                continue
                                
                            delta = part.choices[0].delta
                            finish_reason = part.choices[0].finish_reason
                            
                            # Process assistant content
                            if delta and delta.content:
                                collected_messages.append(delta.content)
                                yield delta.content
                            
                            # Handle tool calls
                            if delta and delta.tool_calls:
                                print(f"[Stream] Processing tool calls: {delta.tool_calls}")
                                for tc in delta.tool_calls:
                                    if len(tool_calls) <= tc.index:
                                        tool_calls.append({
                                            "id": "", "type": "function",
                                            "function": {"name": "", "arguments": ""}
                                        })
                                    tool_calls[tc.index] = {
                                        "id": (tool_calls[tc.index]["id"] + (tc.id or "")),
                                        "type": "function",
                                        "function": {
                                            "name": (tool_calls[tc.index]["function"]["name"] + (tc.function.name or "")),
                                            "arguments": (tool_calls[tc.index]["function"]["arguments"] + (tc.function.arguments or ""))
                                        }
                                    }
                            
                            # Check if we've reached the end of a tool call
                            if finish_reason == "tool_calls" and tool_calls:
                                print(f"[Stream] Tool calls completed. Processing {len(tool_calls)} tool calls")
                                for tool_call in tool_calls:
                                    await self._handle_tool_call(
                                        tool_call["function"]["name"],
                                        tool_call["function"]["arguments"],
                                        tool_call["id"]
                                    )
                                tool_called = True
                                break
                            
                            # Check if we've reached the end of assistant's response
                            if finish_reason == "stop":
                                print("[Stream] Received stop signal, processing final content")
                                # Try to parse the final content as a tool call
                                final_content = ''.join([msg for msg in collected_messages if msg is not None])
                                if final_content.strip():
                                    # First check if content looks like JSON (starts with { or [)
                                    if final_content.strip().startswith(('{', '[')):
                                        try:
                                            tool_call = self._parse_json_response(final_content)
                                            if tool_call:
                                                print(f"[Stream] Parsed final content as tool call: {tool_call['function']['name']}")
                                                await self._handle_tool_call(
                                                    tool_call["function"]["name"],
                                                    tool_call["function"]["arguments"],
                                                    tool_call["id"]
                                                )
                                                tool_called = True
                                            else:
                                                print("[Stream] Content was not a valid tool call, adding as assistant message")
                                                self.messages.append({"role": "assistant", "content": final_content})
                                        except json.JSONDecodeError as json_err:
                                            print(f"[Stream] JSON parsing error in final content: {json_err}")
                                            print(f"[Stream] Raw final content: {final_content}")
                                            # If we have a tool call pending, don't add the content as a message
                                            if not tool_called:
                                                self.messages.append({"role": "assistant", "content": final_content})
                                    else:
                                        # Content is not JSON, treat as regular message
                                        print("[Stream] Final content is not JSON, adding as assistant message")
                                        self.messages.append({"role": "assistant", "content": final_content})
                                elif tool_called:
                                    # If we had a tool call but no content, add a default message
                                    self.messages.append({"role": "assistant", "content": "I've processed your request. Is there anything else you'd like to know about these properties?"})
                                break
                    except asyncio.CancelledError:
                        print("[Stream] Stream cancelled")
                        raise
                    finally:
                        # Ensure proper cleanup of the stream
                        if response_stream in self.active_streams:
                            self.active_streams.remove(response_stream)
                            try:
                                if hasattr(response_stream, 'aclose'):
                                    await response_stream.aclose()
                                elif hasattr(response_stream, 'close'):
                                    await response_stream.close()
                            except (RuntimeError, asyncio.CancelledError, sniffio._impl.AsyncLibraryNotFoundError) as e:
                                print(f"[Stream] Error during stream cleanup: {str(e)}")
            except asyncio.TimeoutError:
                print("[Stream] Stream connection timed out")
                error_occurred = True
            except Exception as e:
                print(f"[Stream] Error in process_response_stream: {str(e)}")
                print(f"[Stream] Error type: {type(e)}")
                traceback.print_exc()
                error_occurred = True
                self.last_error = str(e)
            finally:
                # Store result in instance variables
                self.tool_called = tool_called
                self.last_function_name = function_name if tool_called else None
                print(f"[Stream] Stream processing completed. Tool called: {tool_called}, Function name: {self.last_function_name}, Error occurred: {error_occurred}")

    async def _handle_tool_call(self, function_name, function_arguments, tool_call_id):
        """Handle a tool call by calling the appropriate MCP tool"""
        # Create span for tool call handling
        if tracer:
            with tracer.start_as_current_span("handle_tool_call") as span:
                span.set_attribute("tool.name", function_name)
                span.set_attribute("tool.call_id", tool_call_id)
                span.set_attribute("tool.arguments", function_arguments[:1000])  # Truncate long arguments
                
                # Increment tool calls counter
                tool_calls_counter.add(1, {"tool_name": function_name})
                
                print(f"function_name: {function_name} function_arguments: {function_arguments}")
                try:
                    function_args = json.loads(function_arguments)
                    span.set_attribute("tool.arguments_parsed", True)
                except json.JSONDecodeError as e:
                    print(f"[JSON] Error parsing function arguments: {str(e)}")
                    print(f"[JSON] Raw function arguments: {function_arguments}")
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, f"JSON parse error: {str(e)}"))
                    errors_counter.add(1, {"error_type": "JSONDecodeError", "tool_name": function_name})
                    return

                mcp_tools = cl.user_session.get("mcp_tools", {})
                mcp_name = None
                for connection_name, session_tools in mcp_tools.items():
                    if any(tool.get("name") == function_name for tool in session_tools):
                        mcp_name = connection_name
                        break

                span.set_attribute("mcp.connection", mcp_name or "unknown")
                span.set_attribute("mcp.tools_available", len(mcp_tools))

                # Add the assistant message with tool call
                self.messages.append({
                    "role": "assistant", 
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "function": {
                                "name": function_name,
                                "arguments": function_arguments
                            },
                            "type": "function"
                        }
                    ]
                })
                
                try:
                    # Call the tool and add response to messages
                    func_response = await call_tool(mcp_name, function_name, function_args)
                    print(f"[Tool] Raw function response: {func_response}")
                    
                    # Record response size
                    response_size = len(str(func_response))
                    span.set_attribute("tool.response_size", response_size)
                    
                    try:
                        parsed_response = json.loads(func_response)
                        print(f"[Tool] Parsed function response: {parsed_response}")
                        self.messages.append({
                            "tool_call_id": tool_call_id,
                            "role": "tool",
                            "name": function_name,
                            "content": parsed_response,
                        })
                        span.set_attribute("tool.response_parsed", True)
                    except json.JSONDecodeError as e:
                        print(f"[JSON] Error parsing tool response: {str(e)}")
                        print(f"[JSON] Raw tool response: {func_response}")
                        span.record_exception(e)
                        # Add the raw response as a string if JSON parsing fails
                        self.messages.append({
                            "tool_call_id": tool_call_id,
                            "role": "tool",
                            "name": function_name,
                            "content": func_response,
                        })
                        span.set_attribute("tool.response_parsed", False)
                except Exception as e:
                    print(f"[Tool] Error calling tool: {str(e)}")
                    traceback.print_exc()
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    errors_counter.add(1, {"error_type": type(e).__name__, "tool_name": function_name})
                    self.messages.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": function_name,
                        "content": f"Error: {str(e)}",
                    })
        else:
            # Fallback without tracing
            print(f"function_name: {function_name} function_arguments: {function_arguments}")
            try:
                function_args = json.loads(function_arguments)
            except json.JSONDecodeError as e:
                print(f"[JSON] Error parsing function arguments: {str(e)}")
                print(f"[JSON] Raw function arguments: {function_arguments}")
                return

            mcp_tools = cl.user_session.get("mcp_tools", {})
            mcp_name = None
            for connection_name, session_tools in mcp_tools.items():
                if any(tool.get("name") == function_name for tool in session_tools):
                    mcp_name = connection_name
                    break

            # Add the assistant message with tool call
            self.messages.append({
                "role": "assistant", 
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "function": {
                            "name": function_name,
                            "arguments": function_arguments
                        },
                        "type": "function"
                    }
                ]
            })
            
            try:
                # Call the tool and add response to messages
                func_response = await call_tool(mcp_name, function_name, function_args)
                print(f"[Tool] Raw function response: {func_response}")
                
                try:
                    parsed_response = json.loads(func_response)
                    print(f"[Tool] Parsed function response: {parsed_response}")
                    self.messages.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": function_name,
                        "content": parsed_response,
                    })
                except json.JSONDecodeError as e:
                    print(f"[JSON] Error parsing tool response: {str(e)}")
                    print(f"[JSON] Raw tool response: {func_response}")
                    # Add the raw response as a string if JSON parsing fails
                    self.messages.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": function_name,
                        "content": func_response,
                    })
            except Exception as e:
                print(f"[Tool] Error calling tool: {str(e)}")
                traceback.print_exc()
                self.messages.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name": function_name,
                    "content": f"Error: {str(e)}",
                })

    async def generate_response(self, human_input, tools, temperature=0):
        # Create span for response generation
        if tracer:
            with tracer.start_as_current_span("generate_response") as span:
                span.set_attribute("user.message.content", human_input[:1000])  # Truncate long messages
                span.set_attribute("llm.model", self.model)
                span.set_attribute("llm.temperature", temperature)
                span.set_attribute("chat.tools_count", len(tools))
                span.set_attribute("chat.message_length", len(human_input))
                
                # Record message length metric
                message_length_histogram.record(len(human_input))
                
                start_time = time.time()
                
                try:
                    self.messages.append({"role": "user", "content": human_input})
                    print(f"self.messages: {self.messages}")
                    print(f"Available tools: {tools}")  # Debug print
                    
                    # Handle multiple sequential function calls in a loop rather than recursively
                    while True:
                        try:
                            response_stream = await self.client.chat.completions.create(
                                model=self.model,
                                messages=self.messages,
                                tools=tools,
                                stream=True,
                                temperature=temperature
                            )
                            
                            # Stream and process the response
                            async for token in self._stream_and_process(response_stream, tools, temperature):
                                yield token
                            
                            # Check instance variables after streaming is complete
                            if not self.tool_called:
                                break
                            # Otherwise, loop continues for the next response that follows the tool call
                        except Exception as e:
                            print(f"Error in generate_response: {str(e)}")
                            span.record_exception(e)
                            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                            errors_counter.add(1, {"error_type": type(e).__name__})
                            
                            if "function calling" in str(e).lower():
                                # If function calling isn't supported, fall back to regular chat
                                response_stream = await self.client.chat.completions.create(
                                    model=self.model,
                                    messages=self.messages,
                                    stream=True,
                                    temperature=temperature
                                )
                                async for token in self._stream_and_process(response_stream, [], temperature):
                                    yield token
                                break
                            else:
                                raise
                    
                    # Record response time
                    response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                    response_time_histogram.record(response_time)
                    span.set_attribute("chat.response_time_ms", response_time)
                    
                except GeneratorExit:
                    # Ensure we clean up when the client disconnects
                    await self._cleanup_streams()
                    return
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    errors_counter.add(1, {"error_type": type(e).__name__})
                    raise
        else:
            # Fallback without tracing
            self.messages.append({"role": "user", "content": human_input})
            print(f"self.messages: {self.messages}")
            print(f"Available tools: {tools}")  # Debug print
            
            try:
                # Handle multiple sequential function calls in a loop rather than recursively
                while True:
                    try:
                        response_stream = await self.client.chat.completions.create(
                            model=self.model,
                            messages=self.messages,
                            tools=tools,
                            stream=True,
                            temperature=temperature
                        )
                        
                        # Stream and process the response
                        async for token in self._stream_and_process(response_stream, tools, temperature):
                            yield token
                        
                        # Check instance variables after streaming is complete
                        if not self.tool_called:
                            break
                        # Otherwise, loop continues for the next response that follows the tool call
                    except Exception as e:
                        print(f"Error in generate_response: {str(e)}")
                        if "function calling" in str(e).lower():
                            # If function calling isn't supported, fall back to regular chat
                            response_stream = await self.client.chat.completions.create(
                                model=self.model,
                                messages=self.messages,
                                stream=True,
                                temperature=temperature
                            )
                            async for token in self._stream_and_process(response_stream, [], temperature):
                                yield token
                            break
                        else:
                            raise
            except GeneratorExit:
                # Ensure we clean up when the client disconnects
                await self._cleanup_streams()
                return

    async def _stream_and_process(self, response_stream, tools, temperature):
        """Helper method to yield tokens and return process result"""
        # Initialize instance variables before processing
        self.tool_called = False
        self.last_function_name = None
        self.last_error = None
        
        async for token in self.process_response_stream(response_stream, tools, temperature):
            yield token

def flatten(xss):
    return [x for xs in xss for x in xs]

@cl.on_mcp_connect
async def on_mcp(connection, session: ClientSession):
    result = await session.list_tools()
    tools = [{
        "name": t.name,
        "description": t.description,
        "parameters": t.inputSchema,
        } for t in result.tools]
    
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools[connection.name] = tools
    cl.user_session.set("mcp_tools", mcp_tools)

@cl.step(type="tool") 
async def call_tool(mcp_name, function_name, function_args):
    # Set the step name dynamically based on the function being called
    cl.context.current_step.name = f"Using {function_name}"
    
    # Create additional OpenTelemetry span for tool execution
    if tracer:
        with tracer.start_as_current_span("call_tool") as span:
            span.set_attribute("mcp.connection", mcp_name or "unknown")
            span.set_attribute("tool.name", function_name)
            span.set_attribute("tool.arguments", json.dumps(function_args)[:1000])  # Truncate long args
            span.set_attribute("tool.argument_keys", list(function_args.keys()) if isinstance(function_args, dict) else "non_dict")
            
            try:
                resp_items = []
                print(f"[Tool] Function Name: {function_name}")
                print(f"[Tool] Function Args: {function_args}")
                
                # Check if MCP session exists
                if not hasattr(cl.context.session, 'mcp_sessions'):
                    raise ConnectionError("MCP sessions not initialized. Please connect to the MCP server first.")
                    
                mcp_session = cl.context.session.mcp_sessions.get(mcp_name)
                if mcp_session is None:
                    raise ConnectionError(f"No active connection to MCP server '{mcp_name}'. Please connect to the server first.")
                    
                mcp_session, _ = mcp_session  # Now safe to unpack
                
                # Debug: Log current trace context before MCP call
                current_span = trace.get_current_span()
                if current_span:
                    span_context = current_span.get_span_context()
                    print(f"[Tool] Current trace ID: {span_context.trace_id}, Span ID: {span_context.span_id}")
                    print(f"[Tool] Trace flags: {span_context.trace_flags}")
                    print(f"[Tool] Is remote: {span_context.is_remote}")
                else:
                    print("[Tool] No active span found")
                
                # Check if we can access the MCP session's HTTP client for header inspection
                print(f"[Tool] MCP session type: {type(mcp_session)}")
                if hasattr(mcp_session, '_transport'):
                    print(f"[Tool] MCP transport type: {type(mcp_session._transport)}")
                if hasattr(mcp_session, '_client'):
                    print(f"[Tool] MCP client type: {type(mcp_session._client)}")
                
                # Try to inject trace context and conversation history into the MCP session
                try:
                    # Get current trace context
                    current_span = trace.get_current_span()
                    if current_span:
                        span_context = current_span.get_span_context()
                        trace_id = format(span_context.trace_id, '032x')
                        span_id = format(span_context.span_id, '016x')
                        
                        # Add trace context to function arguments as fallback
                        enhanced_args = function_args.copy()
                        enhanced_args['_trace_context'] = {
                            'trace_id': trace_id,
                            'span_id': span_id,
                            'trace_flags': span_context.trace_flags
                        }
                    else:
                        enhanced_args = function_args.copy()
                    
                    # Inject conversation history for context-aware tools
                    try:
                        # Get recent conversation history from the client
                        client = ChatClient()
                        client.messages = cl.user_session.get("messages", [])
                        
                        # Extract recent user messages (last 5 messages to avoid too much context)
                        recent_messages = []
                        for msg in client.messages[-10:]:  # Get last 10 messages
                            if isinstance(msg, dict) and msg.get('role') in ['user', 'assistant']:
                                recent_messages.append({
                                    'role': msg['role'],
                                    'content': msg.get('content', '')
                                })
                        
                        if recent_messages:
                            enhanced_args['_conversation_history'] = recent_messages
                            print(f"[Tool] Injected conversation history with {len(recent_messages)} messages")
                        
                    except Exception as history_error:
                        print(f"[Tool] Error injecting conversation history: {history_error}")
                        # Continue without history if injection fails
                    
                    print(f"[Tool] Injected trace context: trace_id={trace_id if current_span else 'N/A'}, span_id={span_id if current_span else 'N/A'}")
                    
                    # Try to inject headers into MCP session if possible
                    if hasattr(mcp_session, '_client') and hasattr(mcp_session._client, 'headers'):
                        # Create trace context headers
                        headers = {}
                        inject(headers)
                        if 'traceparent' in headers:
                            mcp_session._client.headers.update(headers)
                            print(f"[Tool] Injected trace headers into MCP client: {headers}")
                    
                    func_response = await mcp_session.call_tool(function_name, enhanced_args)
                except Exception as e:
                    print(f"[Tool] Error injecting trace context: {e}")
                    # Fallback to original call with conversation history
                    fallback_args = function_args.copy()
                    try:
                        # Still try to inject conversation history even in fallback
                        client = ChatClient()
                        client.messages = cl.user_session.get("messages", [])
                        recent_messages = []
                        for msg in client.messages[-10:]:
                            if isinstance(msg, dict) and msg.get('role') in ['user', 'assistant']:
                                recent_messages.append({
                                    'role': msg['role'],
                                    'content': msg.get('content', '')
                                })
                        if recent_messages:
                            fallback_args['_conversation_history'] = recent_messages
                    except Exception:
                        pass  # Ignore errors in fallback
                    func_response = await mcp_session.call_tool(function_name, fallback_args)
                
                print(f"[Tool] Raw MCP response: {func_response}")
                
                if not func_response or not func_response.content:
                    print("[Tool] Empty response from MCP")
                    span.set_attribute("tool.response_empty", True)
                    return json.dumps([{"type": "text", "text": "No response received from tool"}])
                
                content_count = len(func_response.content)
                span.set_attribute("tool.response_content_count", content_count)
                
                for item in func_response.content:
                    if isinstance(item, TextContent):
                        resp_items.append({"type": "text", "text": item.text})
                    elif isinstance(item, ImageContent):
                        resp_items.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{item.mimeType};base64,{item.data}",
                            },
                        })
                    else:
                        print(f"[Tool] Unsupported content type: {type(item)}")
                        resp_items.append({"type": "text", "text": f"Unsupported content type: {type(item)}"})
                
                if not resp_items:
                    print("[Tool] No valid content items in response")
                    span.set_attribute("tool.response_valid", False)
                    return json.dumps([{"type": "text", "text": "No valid content in response"}])
                
                response_size = len(json.dumps(resp_items))
                span.set_attribute("tool.response_size", response_size)
                span.set_attribute("tool.response_valid", True)
                
                return json.dumps(resp_items)
                
            except Exception as e:
                print(f"[Tool] Error in MCP call: {str(e)}")
                traceback.print_exc()
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                errors_counter.add(1, {"error_type": type(e).__name__, "tool_name": function_name, "mcp_connection": mcp_name})
                return json.dumps([{"type": "text", "text": f"Error calling tool: {str(e)}"}])
                
            except ConnectionError as e:
                error_msg = str(e)
                print(f"[Tool] Connection Error: {error_msg}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
                errors_counter.add(1, {"error_type": "ConnectionError", "tool_name": function_name, "mcp_connection": mcp_name})
                return json.dumps([{"type": "text", "text": f"Error: {error_msg}. Please ensure the MCP server is running at http://localhost:8001/sse and try connecting again."}])
            except Exception as e:
                print(f"[Tool] Unexpected error: {str(e)}")
                traceback.print_exc()
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                errors_counter.add(1, {"error_type": type(e).__name__, "tool_name": function_name, "mcp_connection": mcp_name})
                return json.dumps([{"type": "text", "text": f"Unexpected error: {str(e)}"}])
    else:
        # Fallback without tracing
        try:
            resp_items = []
            print(f"[Tool] Function Name: {function_name}")
            print(f"[Tool] Function Args: {function_args}")
            
            # Check if MCP session exists
            if not hasattr(cl.context.session, 'mcp_sessions'):
                raise ConnectionError("MCP sessions not initialized. Please connect to the MCP server first.")
                
            mcp_session = cl.context.session.mcp_sessions.get(mcp_name)
            if mcp_session is None:
                raise ConnectionError(f"No active connection to MCP server '{mcp_name}'. Please connect to the server first.")
                
            mcp_session, _ = mcp_session  # Now safe to unpack
            
            try:
                # Debug: Log current trace context before MCP call (fallback)
                current_span = trace.get_current_span()
                if current_span:
                    span_context = current_span.get_span_context()
                    print(f"[Tool] Current trace ID (fallback): {span_context.trace_id}, Span ID: {span_context.span_id}")
                else:
                    print("[Tool] No active span found (fallback)")
                
                func_response = await mcp_session.call_tool(function_name, function_args)
                print(f"[Tool] Raw MCP response: {func_response}")
                
                if not func_response or not func_response.content:
                    print("[Tool] Empty response from MCP")
                    return json.dumps([{"type": "text", "text": "No response received from tool"}])
                
                for item in func_response.content:
                    if isinstance(item, TextContent):
                        resp_items.append({"type": "text", "text": item.text})
                    elif isinstance(item, ImageContent):
                        resp_items.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{item.mimeType};base64,{item.data}",
                            },
                        })
                    else:
                        print(f"[Tool] Unsupported content type: {type(item)}")
                        resp_items.append({"type": "text", "text": f"Unsupported content type: {type(item)}"})
                
                if not resp_items:
                    print("[Tool] No valid content items in response")
                    return json.dumps([{"type": "text", "text": "No valid content in response"}])
                    
                return json.dumps(resp_items)
                
            except Exception as e:
                print(f"[Tool] Error in MCP call: {str(e)}")
                traceback.print_exc()
                return json.dumps([{"type": "text", "text": f"Error calling tool: {str(e)}"}])
            
        except ConnectionError as e:
            error_msg = str(e)
            print(f"[Tool] Connection Error: {error_msg}")
            return json.dumps([{"type": "text", "text": f"Error: {error_msg}. Please ensure the MCP server is running at http://localhost:8001/sse and try connecting again."}])
        except Exception as e:
            print(f"[Tool] Unexpected error: {str(e)}")
            traceback.print_exc()
            return json.dumps([{"type": "text", "text": f"Unexpected error: {str(e)}"}])


@cl.on_chat_start
async def start_chat():
    # Create span for chat session start
    if tracer:
        with tracer.start_as_current_span("chat_session_start") as span:
            # Initialize with LLM configuration from variables.env
            client = ChatClient()
            cl.user_session.set("messages", [])
            cl.user_session.set("system_prompt", SYSTEM_PROMPT)
            
            # Get MCP tools info
            mcp_tools = cl.user_session.get("mcp_tools", {})
            total_tools = sum(len(tools) for tools in mcp_tools.values())
            
            # Set span attributes
            span.set_attribute("chat.session_id", str(cl.user_session.get("id", "unknown")))
            span.set_attribute("chat.mcp_connections", len(mcp_tools))
            span.set_attribute("chat.total_tools", total_tools)
            span.set_attribute("chat.system_prompt_length", len(SYSTEM_PROMPT))
            
            # Create welcome message
            welcome_content = """
# 🏠 Welcome to Zillow Home Finder

I'm your AI assistant for finding the perfect home! I can help you search for properties using advanced search capabilities.

"""
            
            # Add tools info if available
            if mcp_tools:
                tools_list = "\n".join([f"- **{name}**: {len(tools)} available tools" 
                                     for name, tools in mcp_tools.items()])
                welcome_content += f"\n\n### 🛠️ Available Tools:\n{tools_list}"
            
            # Send welcome message
            await cl.Message(
                content=welcome_content,
                author="System"
            ).send()
            
            # Test function removed - no longer needed
    else:
        # Fallback without tracing
        # Initialize with LLM configuration from variables.env
        client = ChatClient()
        cl.user_session.set("messages", [])
        cl.user_session.set("system_prompt", SYSTEM_PROMPT)
        
        # Create welcome message
        welcome_content = """
# 🏠 Welcome to Zillow Home Finder

I'm your AI assistant for finding the perfect home! I can help you search for properties using advanced search capabilities.

"""
        
        # Add tools info if available
        mcp_tools = cl.user_session.get("mcp_tools", {})
        if mcp_tools:
            tools_list = "\n".join([f"- **{name}**: {len(tools)} available tools" 
                                 for name, tools in mcp_tools.items()])
            welcome_content += f"\n\n### 🛠️ Available Tools:\n{tools_list}"
        
        # Send welcome message
        await cl.Message(
            content=welcome_content,
            author="System"
        ).send()
        
        # Test function removed - no longer needed



@cl.on_message
async def on_message(message: cl.Message):
    # Create span for message processing
    if tracer:
        with tracer.start_as_current_span("on_message") as span:
            span.set_attribute("user.message.content", message.content[:1000])  # Truncate long messages
            span.set_attribute("user.message.length", len(message.content))
            span.set_attribute("chat.session_id", str(cl.user_session.get("id", "unknown")))
            
            # Record message metrics
            messages_counter.add(1)
            message_length_histogram.record(len(message.content))
            
            start_time = time.time()
            
            try:
                mcp_tools = cl.user_session.get("mcp_tools", {})
                tools = flatten([tools for _, tools in mcp_tools.items()])
                tools = [{"type": "function", "function": tool} for tool in tools]
                
                span.set_attribute("chat.tools_available", len(tools))
                span.set_attribute("chat.mcp_connections", len(mcp_tools))
                
                # Create a fresh client instance for each message
                client = ChatClient()
                # Restore conversation history
                client.messages = cl.user_session.get("messages", [])
                
                span.set_attribute("chat.conversation_length", len(client.messages))
                
                msg = cl.Message(content="")
                async for text in client.generate_response(human_input=message.content, tools=tools):
                    await msg.stream_token(text)
                
                # Update the stored messages after processing
                cl.user_session.set("messages", client.messages)
                
                # Record total processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                span.set_attribute("chat.processing_time_ms", processing_time)
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                errors_counter.add(1, {"error_type": type(e).__name__})
                raise
    else:
        # Fallback without tracing
        mcp_tools = cl.user_session.get("mcp_tools", {})
        tools = flatten([tools for _, tools in mcp_tools.items()])
        tools = [{"type": "function", "function": tool} for tool in tools]
        
        # Create a fresh client instance for each message
        client = ChatClient()
        # Restore conversation history
        client.messages = cl.user_session.get("messages", [])
        
        msg = cl.Message(content="")
        async for text in client.generate_response(human_input=message.content, tools=tools):
            await msg.stream_token(text)
        
        # Update the stored messages after processing
        cl.user_session.set("messages", client.messages) 