import express from 'express';
import cors from 'cors';
import { randomUUID } from 'crypto';

export async function runHTTP(toolServer) {
    const app = express();
    const sessions = new Map();

    app.use(cors({
        origin: ['http://localhost', 'http://127.0.0.1', 'http://192.168.50.90', 'https://letta.oculair.ca', 'https://letta2.oculair.ca'],
        credentials: true
    }));
    app.use(express.json({ limit: '10mb' }));

    app.use((req, res, next) => {
        const agentId = req.headers['x-agent-id'];
        const logMessage = agentId
            ? `[${new Date().toISOString()}] ${req.method} ${req.path} (agent: ${agentId})`
            : `[${new Date().toISOString()}] ${req.method} ${req.path}`;
        console.log(logMessage);
        next();
    });

    app.post('/mcp', async (req, res) => {
        try {
            const sessionId = req.headers['mcp-session-id'] || randomUUID();
            
            if (req.body.method === 'initialize') {
                sessions.set(sessionId, { initialized: true });
                
                res.setHeader('mcp-session-id', sessionId);
                res.json({
                    jsonrpc: '2.0',
                    id: req.body.id,
                    result: {
                        protocolVersion: '2025-06-18',
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
                        tools: toolServer.server.getTools()
                    }
                });
            } else if (req.body.method === 'tools/call') {
                const toolName = req.body.params.name;
                const tool = toolServer.server.getTool(toolName);
                
                if (!tool) {
                    res.json({
                        jsonrpc: '2.0',
                        id: req.body.id,
                        error: {
                            code: -32602,
                            message: `Tool not found: ${toolName}`
                        }
                    });
                    return;
                }

                const agentIdHeader = req.headers['x-agent-id'];
                const result = await tool.handler(
                    req.body.params.arguments || {},
                    {
                        headers: req.headers,
                        agentId: agentIdHeader,
                    },
                );
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
            protocol_version: '2025-06-18',
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
}
