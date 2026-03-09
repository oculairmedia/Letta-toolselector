# Agent ID Header Migration Guide

## Overview

This document outlines the migration strategy for the Letta Tool Selector to use the `x-agent-id` header instead of requiring explicit `agent_id` parameters in tool calls. This aligns with Letta's MCP server integration patterns and simplifies agent tool management.

## Current State Analysis

### How Agent IDs Are Currently Handled

1. **MCP Server (`src/index.js`)**
   - `find_tools` function requires explicit `agent_id` parameter
   - Parameter is extracted from `req.body.params.arguments.agent_id`
   - No header processing currently implemented

2. **HTTP Transport (`src/transports/http-transport.js`)**
   - Handles MCP protocol requests at `/mcp` endpoint
   - Passes tool arguments to handlers but ignores request headers
   - No access to `x-agent-id` header in tool handlers

3. **Backend Services**
   - Python API server (`tool-selector-api/api_server.py`) expects `agent_id` parameter
   - Tool management functions require agent ID for tool attachment/detachment
   - All downstream services are parameter-based

### Current Tool Call Flow

```
Letta Agent → MCP Server (/mcp) → Tool Handler → Python Backend
           ↓
    {agent_id: "agent-123"} (in request body)
```

## Migration Strategy

### Phase 1: Header Extraction and Dual Support

**Objective:** Enable header-based agent ID while maintaining backward compatibility

**Changes Required:**

1. **HTTP Transport Modification (`src/transports/http-transport.js`)**

```javascript
// In tools/call handler (line ~50)
else if (req.body.method === 'tools/call') {
    const toolName = req.body.params.name;
    const tool = toolServer.server.getTool(toolName);
    
    if (!tool) {
        // ... error handling
    }
    
    // Extract x-agent-id header
    const agentIdHeader = req.headers['x-agent-id'];
    
    // Pass both arguments and context to tool handler
    const result = await tool.handler(
        req.body.params.arguments || {}, 
        { 
            headers: req.headers,
            agentId: agentIdHeader 
        }
    );
    
    res.json({
        jsonrpc: '2.0',
        id: req.body.id,
        result: result
    });
}
```

2. **Tool Handler Update (`src/index.js`)**

```javascript
// Update handleFindTools method signature (line ~144)
async handleFindTools(args, context = {}) {
    try {
        // Extract agent ID with priority: header > parameter
        const agentIdFromHeader = context.agentId;
        const agentIdFromParam = args.agent_id;
        
        // Resolve agent ID with header taking priority
        const resolvedAgentId = agentIdFromHeader || agentIdFromParam;
        
        // Validation: if both provided, they should match
        if (agentIdFromHeader && agentIdFromParam && 
            agentIdFromHeader !== agentIdFromParam) {
            throw new Error(
                `Agent ID mismatch: header=${agentIdFromHeader}, param=${agentIdFromParam}`
            );
        }
        
        // Log the source of agent ID for debugging
        if (agentIdFromHeader) {
            console.log(`[find_tools] Using agent ID from header: ${agentIdFromHeader}`);
        } else if (agentIdFromParam) {
            console.log(`[find_tools] Using agent ID from parameter: ${agentIdFromParam}`);
        } else {
            console.warn(`[find_tools] No agent ID provided in header or parameter`);
        }
        
        const sanitizedLimit = sanitizeLimit(args.limit);
        const sanitizedMinScore = sanitizeMinScore(args.min_score);
        const requestHeartbeat = sanitizeBoolean(args.request_heartbeat);

        const workerPayload = {
            query: args.query ?? null,
            agent_id: resolvedAgentId, // Use resolved agent ID
            keep_tools: args.keep_tools ?? null,
            limit: sanitizedLimit,
            min_score: sanitizedMinScore,
            request_heartbeat: requestHeartbeat,
        };

        // ... rest of existing logic
    } catch (error) {
        return {
            content: [
                {
                    type: 'text',
                    text: `Error in find_tools: ${error.message}`,
                },
            ],
        };
    }
}
```

3. **Schema Update (`src/index.js`)**

```javascript
// Update tool schema (line ~111)
agent_id: {
    type: 'string',
    description: 'Your agent ID (optional if x-agent-id header is provided)',
    // Remove required constraint
},
```

### Phase 2: Enhanced Validation and Logging

**Objective:** Add robust validation and comprehensive logging

**Additional Changes:**

1. **Enhanced Validation Function**

```javascript
// Add to ToolSelectorServer class
validateAgentId(headerAgentId, paramAgentId) {
    if (!headerAgentId && !paramAgentId) {
        throw new Error('Agent ID must be provided either in x-agent-id header or agent_id parameter');
    }
    
    if (headerAgentId && paramAgentId && headerAgentId !== paramAgentId) {
        throw new Error(
            `Agent ID mismatch: header '${headerAgentId}' != parameter '${paramAgentId}'`
        );
    }
    
    const resolvedId = headerAgentId || paramAgentId;
    
    // Basic format validation (UUID-like)
    if (!/^[a-zA-Z0-9\-_]+$/.test(resolvedId)) {
        throw new Error(`Invalid agent ID format: ${resolvedId}`);
    }
    
    return resolvedId;
}
```

2. **Request Logging Enhancement**

```javascript
// Update logging in http-transport.js
app.use((req, res, next) => {
    const agentId = req.headers['x-agent-id'];
    const logMessage = agentId 
        ? `[${new Date().toISOString()}] ${req.method} ${req.path} (agent: ${agentId})`
        : `[${new Date().toISOString()}] ${req.method} ${req.path}`;
    console.log(logMessage);
    next();
});
```

## Backward Compatibility Strategy

### Dual Support Implementation

The migration maintains full backward compatibility by:

1. **Priority System:** Header takes precedence over parameter
2. **Fallback Logic:** If no header, use parameter
3. **Validation:** Ensure consistency when both are provided
4. **Error Handling:** Clear error messages for mismatches

### Migration Timeline

**Phase 1 (Immediate):** 
- Implement header extraction and dual support
- Deploy with backward compatibility
- Update documentation

**Phase 2 (1-2 weeks later):**
- Add enhanced validation and logging
- Monitor usage patterns
- Identify agents still using parameters

**Phase 3 (Future):**
- Consider deprecating parameter-based approach
- Update client libraries and documentation
- Eventually remove parameter support (optional)

## Testing Strategy

### Unit Tests

1. **Header-only scenarios**
2. **Parameter-only scenarios** 
3. **Both provided (matching)**
4. **Both provided (mismatching)**
5. **Neither provided**

### Integration Tests

1. **Real Letta agent calls with header**
2. **Legacy agent calls with parameter**
3. **Mixed environment testing**

### Test Cases

```javascript
// Test scenarios to implement
const testCases = [
    {
        name: "Header only",
        headers: { 'x-agent-id': 'agent-123' },
        params: {},
        expected: 'agent-123'
    },
    {
        name: "Parameter only", 
        headers: {},
        params: { agent_id: 'agent-456' },
        expected: 'agent-456'
    },
    {
        name: "Both matching",
        headers: { 'x-agent-id': 'agent-789' },
        params: { agent_id: 'agent-789' },
        expected: 'agent-789'
    },
    {
        name: "Both mismatching",
        headers: { 'x-agent-id': 'agent-111' },
        params: { agent_id: 'agent-222' },
        expected: 'error'
    },
    {
        name: "Neither provided",
        headers: {},
        params: {},
        expected: 'error'
    }
];
```

## Benefits of Migration

### For Letta Integration
- **Standardized Pattern:** Follows Letta's recommended MCP server patterns
- **Automatic Context:** Agents don't need to explicitly pass their ID
- **Cleaner API:** Reduces parameter complexity in tool calls

### For Tool Selector
- **Simplified Usage:** Agents can call `find_tools()` without parameters
- **Better Security:** Agent ID comes from trusted header vs user input
- **Improved Logging:** Automatic agent context in all requests

### For Developers
- **Less Boilerplate:** No need to pass agent ID in every tool call
- **Consistent Experience:** Matches other Letta MCP integrations
- **Future-Proof:** Aligns with Letta's architectural direction

## Implementation Checklist

- [ ] Update HTTP transport to extract `x-agent-id` header
- [ ] Modify tool handler to accept context parameter
- [ ] Implement agent ID resolution logic with priority
- [ ] Add validation for agent ID consistency
- [ ] Update tool schema to make `agent_id` optional
- [ ] Add comprehensive logging for debugging
- [ ] Create unit tests for all scenarios
- [ ] Test with real Letta agents
- [ ] Update documentation and examples
- [ ] Deploy with monitoring

## Monitoring and Rollback

### Metrics to Track
- **Header usage rate:** Percentage of requests using header vs parameter
- **Error rates:** Validation failures and mismatches
- **Performance impact:** Any latency changes from header processing

### Rollback Strategy
If issues arise, the migration can be easily rolled back by:
1. Reverting to parameter-only logic
2. Removing header extraction code
3. Restoring original tool schema

The dual-support approach ensures zero downtime during rollback.

## Code Examples

### Before Migration (Current)

```javascript
// Agent must explicitly pass agent_id
const result = await find_tools({
    query: "file management tools",
    agent_id: "agent-123e4567-e89b-12d3-a456-426614174000",
    limit: 5
});
```

### After Migration (Header-based)

```javascript
// Agent ID automatically extracted from x-agent-id header
const result = await find_tools({
    query: "file management tools",
    limit: 5
});
// No need to pass agent_id - comes from header
```

### Letta MCP Server Configuration

When configuring the Tool Selector as an MCP server in Letta:

```python
from letta_client import Letta
from letta_client.types import StreamableHTTPServerConfig, MCPServerType

client = Letta(token="LETTA_API_KEY")

# Configure Tool Selector MCP server
config = StreamableHTTPServerConfig(
    server_name="letta-tool-selector",
    type=MCPServerType.STREAMABLE_HTTP,
    server_url="http://localhost:3020/mcp",
    # No auth needed for local deployment
    custom_headers={"X-API-Version": "v1"}
)

client.tools.add_mcp_server(request=config)
```

Letta will automatically include the `x-agent-id` header in all tool calls.

## Environment Variables

Add these optional environment variables for migration control:

```bash
# Enable header-based agent ID (default: true)
ENABLE_AGENT_ID_HEADER=true

# Require agent ID validation (default: true)
REQUIRE_AGENT_ID=true

# Log agent ID source for debugging (default: false)
DEBUG_AGENT_ID_SOURCE=false

# Strict validation mode (default: false)
STRICT_AGENT_ID_VALIDATION=false
```

## Error Handling

### Common Error Scenarios

1. **No Agent ID Provided**
```json
{
    "jsonrpc": "2.0",
    "id": "req-123",
    "error": {
        "code": -32602,
        "message": "Agent ID must be provided either in x-agent-id header or agent_id parameter"
    }
}
```

2. **Agent ID Mismatch**
```json
{
    "jsonrpc": "2.0",
    "id": "req-124",
    "error": {
        "code": -32602,
        "message": "Agent ID mismatch: header 'agent-111' != parameter 'agent-222'"
    }
}
```

3. **Invalid Agent ID Format**
```json
{
    "jsonrpc": "2.0",
    "id": "req-125",
    "error": {
        "code": -32602,
        "message": "Invalid agent ID format: invalid@id"
    }
}
```

## Security Considerations

### Header Trust
- The `x-agent-id` header is set by Letta and should be trusted
- Parameter-based agent ID comes from user input and needs validation
- Header-based approach is more secure as it's controlled by the platform

### Validation Requirements
- Agent ID format validation (alphanumeric, hyphens, underscores)
- Length limits (reasonable UUID-like format)
- Consistency checks when both header and parameter provided

### Audit Logging
- Log all agent ID sources for security auditing
- Track parameter vs header usage patterns
- Monitor for potential spoofing attempts

## Performance Impact

### Expected Changes
- **Minimal Overhead:** Header extraction adds negligible latency
- **Reduced Payload:** Smaller request bodies without agent_id parameter
- **Better Caching:** Agent context available for request-level caching

### Benchmarking
- Measure request processing time before/after migration
- Monitor memory usage for header processing
- Track error rates during transition period

## Documentation Updates Required

### API Documentation
- Update tool schema to reflect optional agent_id parameter
- Document header-based approach as preferred method
- Provide migration examples for existing integrations

### Integration Guides
- Update Letta integration documentation
- Provide code examples for both approaches
- Document environment variable configuration

### Troubleshooting Guide
- Common error scenarios and solutions
- Debugging steps for agent ID issues
- Migration troubleshooting checklist

## Future Enhancements

### Agent Context Expansion
Once header-based agent ID is established, consider expanding to include:
- Agent name in `x-agent-name` header
- Agent capabilities in `x-agent-capabilities` header
- Request correlation ID in `x-request-id` header

### Advanced Validation
- Agent existence validation against Letta API
- Permission-based tool access control
- Rate limiting per agent

### Monitoring Integration
- Agent-specific metrics and dashboards
- Tool usage analytics per agent
- Performance monitoring by agent

## Conclusion

This migration aligns the Letta Tool Selector with Letta's MCP server patterns while maintaining full backward compatibility. The phased approach ensures a smooth transition with minimal risk and maximum flexibility.

The header-based approach simplifies agent tool management, improves security, and provides a foundation for future enhancements to the tool selector system.
