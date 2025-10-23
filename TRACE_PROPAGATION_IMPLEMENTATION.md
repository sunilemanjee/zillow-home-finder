# Trace Propagation Implementation

## Overview

This document describes the implementation of OpenTelemetry trace context propagation between the home-finder app and MCP server to establish service map connectivity in Elastic APM.

## Problem Solved

The home-finder app and MCP server were not connected in the Elastic APM service map because trace context was not being propagated across the HTTP boundary when the app calls the MCP server.

## Solution Implemented

### Hybrid Approach: Arguments + HTTP Headers

We implemented a hybrid approach that:
1. **Primary**: Passes trace context through tool arguments (`_trace_context`)
2. **Fallback**: Attempts to inject HTTP headers if the MCP client supports it
3. **Server-side**: Extracts context from both sources and creates proper parent-child spans

## Implementation Details

### App Side (`app/app.py`)

#### Trace Context Injection
- **Location**: `call_tool()` function (lines 961-996)
- **Method**: Extracts current span context and injects it into tool arguments
- **Format**: 
  ```python
  enhanced_args['_trace_context'] = {
      'trace_id': trace_id,
      'span_id': span_id,
      'trace_flags': span_context.trace_flags
  }
  ```

#### HTTP Header Injection (Fallback)
- **Location**: `call_tool()` function (lines 980-987)
- **Method**: Attempts to inject trace headers into MCP client if accessible
- **Implementation**: Uses OpenTelemetry's `inject()` function

#### Elasticsearch Service Naming
- **Location**: `initialize_opentelemetry()` function (lines 115-118)
- **Method**: Configures Elasticsearch instrumentation with clear service name
- **Implementation**: `ElasticsearchInstrumentor().instrument(service_name="elasticsearch")`

#### Diagnostic Logging
- Logs current trace context before MCP calls
- Logs MCP session and transport types
- Logs successful context injection

### MCP Server Side (`mcp/server.py`)

#### Context Extraction Functions
- **`extract_trace_context_from_args()`**: Extracts trace context from tool arguments
- **`create_tool_span_with_context()`**: Creates spans with proper parent context

#### Elasticsearch Service Naming
- **Location**: `initialize_opentelemetry()` function (lines 114-117)
- **Method**: Configures Elasticsearch instrumentation with clear service name
- **Implementation**: `ElasticsearchInstrumentor().instrument(service_name="elasticsearch")`

#### Tool Function Updates
All three tool functions updated:
- `parse_query_tool()`
- `geocode_location_tool()`
- `search_homes_tool()`

Each now:
1. Uses `create_tool_span_with_context()` instead of direct span creation
2. Extracts trace context from arguments
3. Creates proper parent-child relationships
4. Excludes `_trace_context` from actual tool parameters

#### Server Setup (`mcp/__main__.py`)
- **Location**: Lines 47-55
- **Purpose**: Creates FastMCP server with argument-based trace context propagation
- **Functionality**: 
  - Creates FastMCP server instance
  - Logs server creation with trace context propagation method
  - Uses argument-based propagation (more reliable than HTTP headers)

## Key Features

### 1. Robust Context Propagation
- **Primary path**: Tool arguments (always works, more reliable)
- **Fallback path**: HTTP headers (if MCP client supports it)
- **Server extraction**: From arguments (primary), HTTP headers (fallback)

### 2. Proper Parent-Child Relationships
- MCP server spans are created as children of app spans
- Uses OpenTelemetry's `SpanContext` and `NonRecordingSpan` for proper hierarchy
- Ensures service map connectivity

### 3. Comprehensive Diagnostics
- Detailed logging at both app and server sides
- Trace ID and span ID logging for verification
- Transport layer inspection
- Header capture and analysis

### 4. Error Handling
- Graceful fallbacks if context injection fails
- No-op implementations when OpenTelemetry is not available
- Exception handling with proper error logging

## Files Modified

### App Side
- `app/app.py`: Added trace context injection and diagnostic logging

### MCP Server Side
- `mcp/server.py`: Added context extraction and span creation functions
- `mcp/__main__.py`: Added diagnostic middleware

### Test and Documentation
- `test_trace_propagation.py`: Comprehensive test instructions
- `TRACE_PROPAGATION_IMPLEMENTATION.md`: This documentation

## Testing

### Manual Testing Steps
1. Start MCP server: `cd mcp && python __main__.py`
2. Start Chainlit app: `cd app && chainlit run app.py --port 8022`
3. Open http://localhost:8022 in browser
4. Connect to MCP server
5. Send message: "Find homes in Orlando with 2 bedrooms"
6. Check console logs for trace propagation diagnostics

### Expected Log Patterns

#### App Side
```
[Tool] Current trace ID: ..., Span ID: ...
[Tool] Injected trace context: trace_id=..., span_id=...
[Tool] MCP session type: ...
```

#### MCP Server Side
```
[MCP] Incoming request headers: ...
[MCP] Extracted trace context: trace_id=..., span_id=...
[MCP] Created span with parent context for parse_query
[MCP] parse_query_tool - Trace ID: ...
```

### Elastic APM Verification
1. Check traces with `service.name='home-finder'`
2. Verify traces include spans from `mcp` service
3. Confirm service map shows: `home-finder → mcp → elasticsearch`
4. Verify trace waterfall shows proper parent-child relationships

## Benefits

1. **Service Map Connectivity**: App and MCP server now appear connected in Elastic APM
2. **Complete Trace Visibility**: End-to-end tracing from user request through MCP to Elasticsearch
3. **Proper Span Hierarchy**: Parent-child relationships enable accurate performance analysis
4. **Robust Implementation**: Multiple fallback mechanisms ensure reliability
5. **Comprehensive Diagnostics**: Detailed logging for troubleshooting

## Technical Notes

- Uses W3C Trace Context standard for HTTP header propagation
- Leverages OpenTelemetry's built-in propagators
- Maintains compatibility with existing MCP SDK
- No breaking changes to tool interfaces
- Graceful degradation when OpenTelemetry is not available
