# OpenTelemetry Observability Implementation

This document describes the OpenTelemetry instrumentation added to the Zillow Home Finder Chainlit application.

## Overview

The application now includes comprehensive OpenTelemetry instrumentation that captures:
- **Traces**: Detailed request flows and function calls
- **Metrics**: Performance counters and histograms
- **Custom Attributes**: Rich context about user interactions, tool usage, and system behavior

All telemetry data is sent to Elasticsearch via the managed OTLP endpoint.

## Configuration

### Environment Variables

The following environment variables are required in `variables.env`:

```env
# OpenTelemetry Configuration
OTEL_EXPORTER_OTLP_ENDPOINT="https://your-motlp-endpoint.ingest.region.elastic.cloud:443"
OTEL_EXPORTER_OTLP_HEADERS="Authorization=ApiKey your-api-key"
OTEL_RESOURCE_ATTRIBUTES="service.name=home-finder,service.version=1.0,deployment.environment=production"
```

### Dependencies

The following OpenTelemetry packages have been added to `requirements.txt`:

```
elastic-opentelemetry
opentelemetry-instrumentation-openai
opentelemetry-instrumentation-elasticsearch
opentelemetry-instrumentation-aiohttp-client
opentelemetry-instrumentation-asyncio
```

## Instrumentation Details

### Auto-Instrumentation

The following libraries are automatically instrumented:
- **OpenAI SDK**: Captures LLM API calls, request/response details
- **Elasticsearch**: Captures search operations and performance
- **aiohttp**: Captures HTTP client requests
- **asyncio**: Captures async operation performance

### Custom Spans

#### ChatClient Class
- `chat_client_init`: Client initialization with model and URL attributes
- `generate_response`: Response generation with user input, model, and tool attributes
- `process_response_stream`: Stream processing with token collection and tool call tracking
- `handle_tool_call`: Tool execution with function name, arguments, and MCP connection details

#### Global Functions
- `call_tool`: MCP tool execution with connection and response details
- `on_message`: Message processing with user input and conversation context
- `chat_session_start`: Session initialization with available tools and MCP connections

### Custom Metrics

#### Counters
- `chat.messages.total`: Total messages processed
- `chat.tool_calls.total`: Total tool calls made (with tool name attribute)
- `chat.errors.total`: Total errors encountered (with error type attribute)

#### Histograms
- `chat.response_time`: Response generation time distribution (in milliseconds)
- `chat.message_length`: Message length distribution (in characters)

### Custom Attributes

#### User Context
- `user.message.content`: User message text (truncated to 1000 chars)
- `user.message.length`: Message length in characters

#### LLM Context
- `llm.model`: Model name being used
- `llm.temperature`: Temperature parameter
- `llm.response.finish_reason`: How the response completed

#### Tool Context
- `tool.name`: Tool/function being called
- `tool.arguments`: Tool arguments (as JSON string, truncated)
- `tool.call_id`: Unique tool call identifier
- `tool.response_size`: Size of tool response
- `tool.response_valid`: Whether response was successfully parsed

#### MCP Context
- `mcp.connection`: MCP connection name
- `mcp.tools_available`: Number of available tools

#### Chat Context
- `chat.session_id`: Chainlit session ID
- `chat.tools_count`: Number of available tools
- `chat.tokens_collected`: Number of tokens collected from stream
- `chat.tool_called`: Whether a tool was called
- `chat.error_occurred`: Whether an error occurred
- `chat.processing_time_ms`: Total message processing time

## Error Tracking

All exceptions are captured with:
- Exception details recorded as span events
- Span status set to ERROR
- Error counters incremented with error type attributes
- Full stack traces preserved

## Testing

### Test Script

Run the test script to verify OpenTelemetry setup:

```bash
cd app
python test_otel.py
```

This will test:
- OpenTelemetry imports
- Environment variable configuration
- OpenTelemetry initialization
- Basic tracing functionality
- Basic metrics functionality

### Manual Testing

1. Start the Chainlit application:
   ```bash
   chainlit run app.py --port 8022
   ```

2. Send messages and use tools to generate telemetry data

3. Check your Elasticsearch instance for:
   - Traces in the `traces-generic.otel-default` data stream
   - Metrics in the `metrics-generic.otel-default` data stream

## Data Streams in Elasticsearch

The telemetry data will appear in these Elasticsearch data streams:
- `traces-generic.otel-default`: All trace data
- `metrics-generic.otel-default`: All metric data

## Performance Impact

The instrumentation is designed to be lightweight:
- Spans are batched and exported asynchronously
- Metrics are exported every 10 seconds
- Long text content is truncated to prevent large payloads
- No-op fallbacks are provided when OpenTelemetry is not available

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**: Check that all required OTEL environment variables are set
2. **Import Errors**: Ensure all OpenTelemetry packages are installed
3. **Connection Issues**: Verify the OTLP endpoint URL and API key are correct
4. **No Data in Elasticsearch**: Check network connectivity and endpoint configuration

### Debug Mode

The application includes debug logging for OpenTelemetry operations. Look for log messages prefixed with `[OTEL]`.

### Fallback Behavior

If OpenTelemetry initialization fails, the application will continue to work normally with no-op instrumentation, ensuring no impact on functionality.
