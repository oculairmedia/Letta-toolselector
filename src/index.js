import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { runHTTP } from './transports/http-transport.js';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const DEFAULT_LIMIT = 10;
const DEFAULT_MIN_SCORE = 50;
const MAX_LIMIT = Number.isFinite(Number(process.env.FIND_TOOLS_MAX_LIMIT))
    ? Math.max(1, Math.floor(Number(process.env.FIND_TOOLS_MAX_LIMIT)))
    : 25;
const PROCESS_TIMEOUT_MS = Number.isFinite(Number(process.env.FIND_TOOLS_PROCESS_TIMEOUT))
    ? Math.max(5000, Math.floor(Number(process.env.FIND_TOOLS_PROCESS_TIMEOUT)))
    : 30000;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const sanitizeLimit = (value) => {
    if (value === undefined || value === null || value === '') {
        return DEFAULT_LIMIT;
    }

    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
        return DEFAULT_LIMIT;
    }

    const bounded = Math.max(1, Math.floor(parsed));
    return Math.min(bounded, MAX_LIMIT);
};

const sanitizeMinScore = (value) => {
    if (value === undefined || value === null || value === '') {
        return DEFAULT_MIN_SCORE;
    }

    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
        return DEFAULT_MIN_SCORE;
    }

    if (parsed < 0) {
        return 0;
    }

    if (parsed > 100) {
        return 100;
    }

    return parsed;
};

const sanitizeBoolean = (value) => {
    if (typeof value === 'boolean') {
        return value;
    }

    if (typeof value === 'string') {
        return value.toLowerCase() === 'true';
    }

    return Boolean(value);
};

class ToolSelectorServer {
    constructor() {
        this.server = new Server(
            {
                name: 'letta-tools-selector',
                version: '1.0.0',
            },
            {
                capabilities: {
                    tools: {},
                },
            }
        );

        this.setupHandlers();
        this.setupToolsList();
    }

    setupToolsList() {
        this.server.addTool({
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
                },
            },
            handler: async (args) => this.handleFindTools(args),
        });
    }

    setupHandlers() {
        // No need for manual request handlers with the new API
    }

    async handleFindTools(args) {
        try {
            const scriptPath = path.join(__dirname, '..', 'find_tools.py');
            
            const pythonArgs = ['python3', scriptPath];

            if (args.query) pythonArgs.push('--query', args.query);
            if (args.agent_id) pythonArgs.push('--agent_id', args.agent_id);
            if (args.keep_tools) pythonArgs.push('--keep_tools', args.keep_tools);

            const sanitizedLimit = sanitizeLimit(args.limit);
            const sanitizedMinScore = sanitizeMinScore(args.min_score);
            const requestHeartbeat = sanitizeBoolean(args.request_heartbeat);

            pythonArgs.push('--limit', sanitizedLimit.toString());
            pythonArgs.push('--min_score', sanitizedMinScore.toString());
            if (requestHeartbeat) {
                pythonArgs.push('--request_heartbeat', 'true');
            }

            return new Promise((resolve) => {
                const child = spawn(pythonArgs[0], pythonArgs.slice(1));
                let stdout = '';
                let stderr = '';
                let timedOut = false;

                const timeoutHandle = setTimeout(() => {
                    timedOut = true;
                    child.kill('SIGKILL');
                }, PROCESS_TIMEOUT_MS);

                child.stdout.on('data', (data) => {
                    stdout += data.toString();
                });

                child.stderr.on('data', (data) => {
                    stderr += data.toString();
                });

                child.on('close', (code) => {
                    clearTimeout(timeoutHandle);

                    if (timedOut) {
                        resolve({
                            content: [
                                {
                                    type: 'text',
                                    text: 'The find_tools operation timed out. Please try again with narrower parameters.',
                                },
                            ],
                        });
                        return;
                    }

                    if (code !== 0) {
                        const errorMessage = stderr.trim() || 'Unknown error';
                        resolve({
                            content: [
                                {
                                    type: 'text',
                                    text: `Error executing find_tools: ${errorMessage}`,
                                },
                            ],
                        });
                    } else {
                        const lines = stdout.trim().split('\n');
                        const result = lines[lines.length - 1];
                        const content = [];
                        if (result) {
                            content.push({ type: 'text', text: result });
                        }
                        if (stderr.trim()) {
                            content.push({ type: 'text', text: `Warnings: ${stderr.trim()}` });
                        }
                        resolve({
                            content: content.length ? content : [
                                {
                                    type: 'text',
                                    text: 'find_tools completed with no output.'
                                }
                            ],
                        });
                    }
                });

                child.on('error', (error) => {
                    clearTimeout(timeoutHandle);
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
}

async function main() {
    const server = new ToolSelectorServer();
    
    const useHTTP = process.argv.includes('--http');
    
    if (useHTTP) {
        console.log('Starting Tool Selector MCP server with HTTP transport');
        await runHTTP(server);
    } else {
        console.log('Starting Tool Selector MCP server with stdio transport');
        const transport = new StdioServerTransport();
        await server.server.connect(transport);
    }
}

main().catch((error) => {
    console.error('Server error:', error);
    process.exit(1);
});
