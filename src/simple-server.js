import express from 'express';
import cors from 'cors';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import crypto from 'crypto';

const ENABLE_AGENT_ID_HEADER = (process.env.ENABLE_AGENT_ID_HEADER ?? 'true').toLowerCase() !== 'false';
const REQUIRE_AGENT_ID = (process.env.REQUIRE_AGENT_ID ?? 'true').toLowerCase() === 'true';
const STRICT_AGENT_ID_VALIDATION = (process.env.STRICT_AGENT_ID_VALIDATION ?? 'false').toLowerCase() === 'true';
const DEBUG_AGENT_ID_SOURCE = (process.env.DEBUG_AGENT_ID_SOURCE ?? 'false').toLowerCase() === 'true';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const sessions = new Map();

app.use(cors({
    origin: ['http://localhost', 'http://127.0.0.1', 'http://192.168.50.90', 'https://letta.oculair.ca', 'https://letta2.oculair.ca'],
    credentials: true,
    allowedHeaders: ['Content-Type', 'Mcp-Session-Id', 'Accept'],
    exposedHeaders: ['Mcp-Session-Id']
}));
app.use(express.json({ limit: '10mb' }));

app.use((req, res, next) => {
    const agentId = req.headers['x-agent-id'] ?? req.headers['X-Agent-Id'];
    const logMessage = agentId
        ? `[${new Date().toISOString()}] ${req.method} ${req.path} (agent: ${agentId})`
        : `[${new Date().toISOString()}] ${req.method} ${req.path}`;
    console.log(logMessage);
    next();
});

const normalizeAgentIdValue = (value) => {
    if (Array.isArray(value)) {
        return normalizeAgentIdValue(value[0]);
    }

    if (value === undefined || value === null) {
        return undefined;
    }

    if (typeof value === 'string') {
        const trimmed = value.trim();
        return trimmed.length ? trimmed : undefined;
    }

    const stringified = String(value).trim();
    return stringified.length ? stringified : undefined;
};

const resolveAgentId = (headerAgentId, paramAgentId) => {
    const normalizedHeader = normalizeAgentIdValue(headerAgentId);
    const normalizedParam = normalizeAgentIdValue(paramAgentId);
    const hasHeader = Boolean(normalizedHeader);
    const hasParam = Boolean(normalizedParam);

    if (hasHeader && hasParam && normalizedHeader !== normalizedParam) {
        const message = `Agent ID mismatch: header '${normalizedHeader}' != parameter '${normalizedParam}'`;
        console.warn(`[find_tools] ${message}`);
        throw new Error(message);
    }

    if (hasHeader) {
        if (DEBUG_AGENT_ID_SOURCE) {
            console.log(`[find_tools] Using agent ID from header: ${normalizedHeader}`);
        }
    } else if (hasParam) {
        if (DEBUG_AGENT_ID_SOURCE) {
            console.log(`[find_tools] Using agent ID from parameter: ${normalizedParam}`);
        }
    } else {
        console.warn('[find_tools] No agent ID provided in header or parameter');
        if (REQUIRE_AGENT_ID) {
            throw new Error('Agent ID must be provided either in x-agent-id header or agent_id parameter');
        }
        return null;
    }

    const resolvedId = normalizedHeader || normalizedParam;

    if (resolvedId && !/^[a-zA-Z0-9\-_]+$/.test(resolvedId)) {
        const message = `Invalid agent ID format: ${resolvedId}`;
        console.warn(`[find_tools] ${message}`);
        if (STRICT_AGENT_ID_VALIDATION || REQUIRE_AGENT_ID) {
            throw new Error(message);
        }
    }

    return resolvedId;
};

async function executeFindTools(args) {
    try {
        const scriptPath = path.join(__dirname, '..', 'find_tools_enhanced.py');
        
        const pythonArgs = ['python3', scriptPath];
        
        if (args.query) pythonArgs.push('--query', args.query);
        if (args.agent_id) pythonArgs.push('--agent_id', args.agent_id);
        if (args.keep_tools) pythonArgs.push('--keep_tools', args.keep_tools);
        if (args.limit !== undefined) pythonArgs.push('--limit', args.limit.toString());
        if (args.min_score !== undefined) pythonArgs.push('--min_score', args.min_score.toString());
        if (args.request_heartbeat) pythonArgs.push('--request_heartbeat', 'true');
        if (args.detailed_response) pythonArgs.push('--detailed', 'true');
        if (args.apply_rules !== undefined) pythonArgs.push('--apply_rules', args.apply_rules.toString());

        return new Promise((resolve, reject) => {
            const process = spawn(pythonArgs[0], pythonArgs.slice(1));
            let stdout = '';
            let stderr = '';

            process.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            process.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            process.on('close', (code) => {
                if (code !== 0) {
                    resolve({
                        content: [
                            {
                                type: 'text',
                                text: `Error executing find_tools: ${stderr || 'Unknown error'}`,
                            },
                        ],
                    });
                } else {
                    // The Python script may output debug info to stderr and result to stdout
                    // We need to handle multi-line JSON output
                    const output = stdout.trim();
                    
                    // Try to parse as JSON for detailed responses
                    try {
                        // First try to parse the entire output as JSON
                        const jsonResult = JSON.parse(output);
                        if (jsonResult.status && jsonResult.summary) {
                            // It's a detailed response, format it nicely
                            let formattedText = jsonResult.summary;
                            
                            if (jsonResult.recommendations && jsonResult.recommendations.length > 0) {
                                formattedText += '\n\nRecommendations:\n' + 
                                    jsonResult.recommendations.map(r => `• ${r}`).join('\n');
                            }
                            
                            resolve({
                                content: [
                                    {
                                        type: 'text',
                                        text: formattedText,
                                    },
                                ],
                                // Include full details in metadata for agents that can use it
                                _meta: {
                                    'letta/tool-details': jsonResult
                                }
                            });
                        } else {
                            // JSON but not our expected format
                            resolve({
                                content: [
                                    {
                                        type: 'text',
                                        text: result,
                                    },
                                ],
                            });
                        }
                    } catch (e) {
                        // Not JSON or can't parse - try to extract just the last line
                        const lines = output.split('\n');
                        const lastLine = lines[lines.length - 1];
                        
                        resolve({
                            content: [
                                {
                                    type: 'text',
                                    text: lastLine || output,
                                },
                            ],
                        });
                    }
                }
            });

            process.on('error', (error) => {
                resolve({
                    content: [
                        {
                            type: 'text',
                            text: `Failed to execute find_tools: ${error.message}`,
                        },
                    ],
                });
            });
        });
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

app.post('/mcp', async (req, res) => {
    try {
        const sessionId = req.headers['mcp-session-id'] || req.headers['Mcp-Session-Id'] || crypto.randomUUID();
        
        if (req.body.method === 'initialize') {
            sessions.set(sessionId, { initialized: true });
            
            res.setHeader('Mcp-Session-Id', sessionId);
            res.json({
                jsonrpc: '2.0',
                id: req.body.id,
                result: {
                    protocolVersion: '2024-11-05',
                    capabilities: {
                        tools: {}
                    },
                    serverInfo: {
                        name: 'letta-tools-selector',
                        version: '1.0.0'
                    }
                }
            });
        } else if (req.body.method === 'tools/list') {
            res.json({
                jsonrpc: '2.0',
                id: req.body.id,
                result: {
                    tools: [
                        {
                            name: 'find_tools',
                            description: 'Silently manage tools for the agent by searching and attaching relevant tools',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    query: {
                                        type: 'string',
                                        description: 'Your search query - what kind of tool are you looking for?',
                                    },
                                    agent_id: {
                                        type: 'string',
                                        description: 'Your agent ID',
                                    },
                                    keep_tools: {
                                        type: 'string',
                                        description: 'Comma-separated list of tool IDs to preserve',
                                    },
                                    limit: {
                                        type: 'number',
                                        description: 'Maximum number of tools to find (default: 10)',
                                        default: 10,
                                    },
                                    min_score: {
                                        type: 'number',
                                        description: 'Minimum match score 0-100 (default: 50.0)',
                                        default: 50.0,
                                    },
                                    request_heartbeat: {
                                        type: 'boolean',
                                        description: 'Whether to request an immediate heartbeat (default: false)',
                                        default: false,
                                    },
                                    detailed_response: {
                                        type: 'boolean',
                                        description: 'Return detailed information about tool changes (default: false)',
                                        default: false,
                                    },
                                    apply_rules: {
                                        type: 'boolean',
                                        description: 'Apply tool dependency and exclusion rules (default: true)',
                                        default: true,
                                    },
                                },
                                required: ['query'],
                            },
                        },
                    ],
                }
            });
        } else if (req.body.method === 'tools/call') {
            const toolName = req.body.params.name;
            
            if (toolName === 'find_tools') {
                const args = req.body.params.arguments || {};
                const headerAgentId = ENABLE_AGENT_ID_HEADER
                    ? (req.headers['x-agent-id'] ?? req.headers['X-Agent-Id'])
                    : undefined;

                let resolvedAgentId;
                try {
                    resolvedAgentId = resolveAgentId(headerAgentId, args.agent_id);
                } catch (error) {
                    res.json({
                        jsonrpc: '2.0',
                        id: req.body.id,
                        error: {
                            code: -32602,
                            message: error.message,
                        },
                    });
                    return;
                }

                const result = await executeFindTools({
                    ...args,
                    agent_id: resolvedAgentId,
                });
                res.json({
                    jsonrpc: '2.0',
                    id: req.body.id,
                    result: result
                });
            } else {
                res.json({
                    jsonrpc: '2.0',
                    id: req.body.id,
                    error: {
                        code: -32602,
                        message: `Tool not found: ${toolName}`
                    }
                });
            }
        } else {
            res.json({
                jsonrpc: '2.0',
                id: req.body.id,
                error: {
                    code: -32601,
                    message: `Method not found: ${req.body.method}`
                }
            });
        }
    } catch (error) {
        console.error('Error handling request:', error);
        res.json({
            jsonrpc: '2.0',
            id: req.body ? req.body.id : null,
            error: {
                code: -32603,
                message: error.message
            }
        });
    }
});

app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        transport: 'http',
        protocol_version: '2024-11-05',
        sessions: sessions.size,
        timestamp: new Date().toISOString()
    });
});

const PORT = process.env.PORT || 3020;
const HOST = '0.0.0.0';

app.listen(PORT, HOST, () => {
    console.log(`Tool Selector MCP server running on ${HOST}:${PORT}`);
    console.log(`MCP endpoint: http://localhost:${PORT}/mcp`);
    console.log(`Health check: http://localhost:${PORT}/health`);
});
