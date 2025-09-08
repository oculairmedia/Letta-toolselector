import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { runHTTP } from './transports/http-transport.js';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

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
            if (args.limit !== undefined) pythonArgs.push('--limit', args.limit.toString());
            if (args.min_score !== undefined) pythonArgs.push('--min_score', args.min_score.toString());
            if (args.request_heartbeat) pythonArgs.push('--request_heartbeat', 'true');

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
                        const lines = stdout.trim().split('\n');
                        const result = lines[lines.length - 1];
                        
                        resolve({
                            content: [
                                {
                                    type: 'text',
                                    text: result,
                                },
                            ],
                        });
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