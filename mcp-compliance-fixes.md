# MCP Specification Compliance Fixes for Postizz

## Issue Summary
Postizz tools are visible in MCP Jam but not in Letta due to stricter MCP protocol compliance enforcement in Letta. This document provides specific fixes to ensure full MCP specification compliance.

## Root Cause Analysis
- **Letta**: Enforces strict JSON-RPC 2.0 and MCP specification compliance
- **MCP Jam**: More forgiving of protocol deviations
- **Postizz**: Likely has minor specification violations that Letta rejects

## Critical MCP Requirements

### 1. JSON-RPC 2.0 Format Compliance
**Required Message Structure:**
```json
{
  "jsonrpc": "2.0",
  "id": <request_id>,
  "method": "<method_name>",
  "params": { ... }
}
```

**Response Structure:**
```json
{
  "jsonrpc": "2.0", 
  "id": <request_id>,
  "result": { ... }
}
```

### 2. Initialization Sequence
**Step 1: Client Initialize Request**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {"tools": {}},
    "clientInfo": {"name": "letta", "version": "1.0.0"}
  }
}
```

**Step 2: Server Initialize Response**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {"listChanged": true}
    },
    "serverInfo": {
      "name": "postizz-mcp",
      "version": "1.0.0"
    }
  }
}
```

**Step 3: Client Initialized Notification**
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized",
  "params": {}
}
```

### 3. Tool Discovery Implementation
**Tools List Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}
```

**Tools List Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "tool_name",
        "description": "Tool description",
        "inputSchema": {
          "type": "object",
          "properties": {
            "param1": {
              "type": "string",
              "description": "Parameter description"
            }
          },
          "required": ["param1"]
        }
      }
    ]
  }
}
```

## Implementation Fixes

### Fix 1: Strict JSON-RPC 2.0 Validation
```python
def validate_jsonrpc_request(request):
    """Validate incoming JSON-RPC 2.0 request"""
    if not isinstance(request, dict):
        raise ValueError("Request must be a JSON object")
    
    if request.get("jsonrpc") != "2.0":
        raise ValueError("jsonrpc field must be '2.0'")
    
    if "id" not in request:
        raise ValueError("id field is required")
    
    if "method" not in request:
        raise ValueError("method field is required")
    
    return True

def create_jsonrpc_response(request_id, result=None, error=None):
    """Create properly formatted JSON-RPC 2.0 response"""
    response = {
        "jsonrpc": "2.0",
        "id": request_id
    }
    
    if error:
        response["error"] = error
    else:
        response["result"] = result
    
    return response
```

### Fix 2: Proper Initialization Handler
```python
class MCPServer:
    def __init__(self):
        self.initialized = False
        self.capabilities = {
            "tools": {"listChanged": True}
        }
    
    def handle_initialize(self, request):
        """Handle MCP initialization request"""
        params = request.get("params", {})
        client_version = params.get("protocolVersion", "2024-11-05")
        
        # Validate protocol version compatibility
        if client_version not in ["2024-11-05", "2025-03-26"]:
            return create_jsonrpc_response(
                request["id"],
                error={
                    "code": -32602,
                    "message": f"Unsupported protocol version: {client_version}"
                }
            )
        
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": self.capabilities,
            "serverInfo": {
                "name": "postizz-mcp-server",
                "version": "1.0.0"
            }
        }
        
        return create_jsonrpc_response(request["id"], result)
    
    def handle_initialized(self, request):
        """Handle initialized notification"""
        self.initialized = True
        # No response for notifications
        return None
```

### Fix 3: Tool Schema Validation
```python
def validate_tool_schema(tool):
    """Validate tool against MCP specification"""
    required_fields = ["name", "description", "inputSchema"]
    
    for field in required_fields:
        if field not in tool:
            raise ValueError(f"Tool missing required field: {field}")
    
    # Validate inputSchema
    input_schema = tool["inputSchema"]
    if not isinstance(input_schema, dict):
        raise ValueError("inputSchema must be an object")
    
    if input_schema.get("type") != "object":
        raise ValueError("inputSchema.type must be 'object'")
    
    # Validate properties if present
    if "properties" in input_schema:
        if not isinstance(input_schema["properties"], dict):
            raise ValueError("inputSchema.properties must be an object")
    
    # Validate required array if present
    if "required" in input_schema:
        if not isinstance(input_schema["required"], list):
            raise ValueError("inputSchema.required must be an array")
    
    return True

def get_validated_tools():
    """Return all tools with validated schemas"""
    tools = []
    
    # Example tool - replace with your actual tools
    example_tool = {
        "name": "postizz_action",
        "description": "Execute a Postizz action",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to perform"
                },
                "parameters": {
                    "type": "object",
                    "description": "Action parameters"
                }
            },
            "required": ["action"]
        }
    }
    
    # Validate each tool before adding
    try:
        validate_tool_schema(example_tool)
        tools.append(example_tool)
    except ValueError as e:
        print(f"Tool validation failed: {e}")
    
    return tools
```

### Fix 4: Complete Request Handler
```python
def handle_mcp_request(request_data):
    """Main MCP request handler with full compliance"""
    try:
        # Parse JSON
        if isinstance(request_data, str):
            request = json.loads(request_data)
        else:
            request = request_data
        
        # Validate JSON-RPC format
        validate_jsonrpc_request(request)
        
        method = request["method"]
        request_id = request["id"]
        
        # Handle different methods
        if method == "initialize":
            return server.handle_initialize(request)
        
        elif method == "notifications/initialized":
            return server.handle_initialized(request)
        
        elif method == "tools/list":
            if not server.initialized:
                return create_jsonrpc_response(
                    request_id,
                    error={
                        "code": -32002,
                        "message": "Server not initialized"
                    }
                )
            
            tools = get_validated_tools()
            return create_jsonrpc_response(
                request_id,
                {"tools": tools}
            )
        
        elif method == "tools/call":
            # Handle tool execution
            return handle_tool_call(request)
        
        else:
            return create_jsonrpc_response(
                request_id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            )
    
    except json.JSONDecodeError:
        return create_jsonrpc_response(
            None,
            error={
                "code": -32700,
                "message": "Parse error"
            }
        )
    
    except ValueError as e:
        return create_jsonrpc_response(
            request.get("id"),
            error={
                "code": -32602,
                "message": str(e)
            }
        )
    
    except Exception as e:
        return create_jsonrpc_response(
            request.get("id"),
            error={
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        )
```

## Testing and Validation

### 1. Use MCP Inspector
```bash
# Install and run MCP Inspector
npm install -g @modelcontextprotocol/inspector
mcp-inspector --server-command "python postizz_mcp_server.py"
```

### 2. Test with Letta
```bash
# Add server to Letta with verbose logging
letta mcp add postizz --transport http http://localhost:8080/mcp --verbose
```

### 3. Validation Checklist
- [ ] JSON-RPC 2.0 format compliance
- [ ] Proper initialization sequence
- [ ] Tool schema validation
- [ ] Error handling with standard codes
- [ ] Protocol version compatibility
- [ ] Content-Type headers (application/json)

## Common Error Codes
- `-32700`: Parse error (invalid JSON)
- `-32600`: Invalid Request (malformed JSON-RPC)
- `-32601`: Method not found
- `-32602`: Invalid params
- `-32603`: Internal error

## Next Steps
1. Implement the fixes above in Postizz MCP server
2. Test with MCP Inspector for validation
3. Test with both Letta and MCP Jam
4. Monitor logs for any remaining compliance issues
5. Add comprehensive error logging for debugging
