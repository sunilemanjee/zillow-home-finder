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
load_dotenv("../variables.env")

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



@cl.on_message
async def on_message(message: cl.Message):
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