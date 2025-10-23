#!/usr/bin/env python3
"""
Test script to verify trace propagation between app and MCP server.
Run this to test the trace context propagation implementation.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

async def test_trace_propagation():
    """Test trace propagation by making a simple MCP call"""
    
    print("=== Trace Propagation Test ===")
    print("This script will help verify if trace context is being propagated")
    print("between the home-finder app and MCP server.")
    print()
    
    print("Instructions:")
    print("1. Start the MCP server: cd mcp && python __main__.py")
    print("2. Start the Chainlit app: cd app && chainlit run app.py --port 8022")
    print("3. Open http://localhost:8022 in your browser")
    print("4. Connect to the MCP server (if not auto-connected)")
    print("5. Send a message like: 'Find homes in Orlando with 2 bedrooms'")
    print("6. Check the console logs for trace propagation diagnostics")
    print()
    
    print("Look for these log patterns:")
    print("=== APP SIDE ===")
    print("- [Tool] Current trace ID: ... (shows current span context)")
    print("- [Tool] Injected trace context: trace_id=..., span_id=... (shows context injection)")
    print("- [Tool] MCP session type: ... (shows MCP client type)")
    print("- [Tool] Injected trace headers into MCP client: ... (if headers were injected)")
    print()
    print("=== MCP SERVER SIDE ===")
    print("- [MCP] FastMCP server created - using argument-based trace context propagation")
    print("- [MCP] Extracted trace context: trace_id=..., span_id=... (from arguments)")
    print("- [MCP] Created span with parent context for ... (shows parent-child relationship)")
    print("- [MCP] parse_query_tool - Trace ID: ... (shows MCP tool trace ID)")
    print()
    
    print("Expected outcomes:")
    print("✅ SUCCESS: App and MCP will have matching trace IDs")
    print("✅ SUCCESS: MCP spans will be children of app spans")
    print("✅ SUCCESS: Service map will show: home-finder → mcp → elasticsearch")
    print("✅ SUCCESS: Elasticsearch will show as 'elasticsearch' instead of hostname")
    print()
    print("❌ FAILURE indicators:")
    print("- Different trace IDs between app and MCP")
    print("- 'Created span without parent context' in MCP logs")
    print("- No connection in Elastic APM service map")
    print()
    
    print("Implementation details:")
    print("- App injects trace context via tool arguments (_trace_context)")
    print("- App also tries to inject HTTP headers if MCP client supports it")
    print("- MCP server extracts context from arguments and creates parent-child spans")
    print("- Uses argument-based propagation (more reliable than HTTP headers)")
    print()
    
    print("To verify in Elastic APM:")
    print("1. Go to your Elastic APM dashboard")
    print("2. Look for traces with service.name='home-finder'")
    print("3. Check that traces include spans from 'mcp' service")
    print("4. Verify service map shows connection between home-finder and mcp")
    print("5. Check trace waterfall shows proper parent-child relationships")

if __name__ == "__main__":
    asyncio.run(test_trace_propagation())
