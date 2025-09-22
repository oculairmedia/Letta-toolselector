import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { runHTTP } from './transports/http-transport.js';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import axios from 'axios';

const DEFAULT_LIMIT = 10;
const DEFAULT_MIN_SCORE = 50;
const MAX_LIMIT = Number.isFinite(Number(process.env.FIND_TOOLS_MAX_LIMIT))
    ? Math.max(1, Math.floor(Number(process.env.FIND_TOOLS_MAX_LIMIT)))
    : 25;
const PROCESS_TIMEOUT_MS = Number.isFinite(Number(process.env.FIND_TOOLS_PROCESS_TIMEOUT))
    ? Math.max(5000, Math.floor(Number(process.env.FIND_TOOLS_PROCESS_TIMEOUT)))
    : 30000;
const WORKER_SERVICE_URL = process.env.WORKER_SERVICE_URL || 'http://worker-service:3021';
const WORKER_REQUEST_TIMEOUT_MS = Number.isFinite(Number(process.env.WORKER_REQUEST_TIMEOUT_MS))
    ? Math.max(1000, Math.floor(Number(process.env.WORKER_REQUEST_TIMEOUT_MS)))
    : 15000;
const WORKER_HEALTH_TIMEOUT_MS = Number.isFinite(Number(process.env.WORKER_HEALTH_TIMEOUT_MS))
    ? Math.max(500, Math.floor(Number(process.env.WORKER_HEALTH_TIMEOUT_MS)))
    : 5000;
const WORKER_HEALTH_CHECK_INTERVAL_MS = Number.isFinite(Number(process.env.WORKER_HEALTH_CHECK_INTERVAL_MS))
    ? Math.max(1000, Math.floor(Number(process.env.WORKER_HEALTH_CHECK_INTERVAL_MS)))
    : 30000;
const DISABLE_WORKER_SERVICE = (process.env.DISABLE_WORKER_SERVICE || '').toLowerCase() === 'true';

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

        this.workerServiceAvailable = !DISABLE_WORKER_SERVICE;
        this.lastWorkerHealthCheck = 0;

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
            const sanitizedLimit = sanitizeLimit(args.limit);
            const sanitizedMinScore = sanitizeMinScore(args.min_score);
            const requestHeartbeat = sanitizeBoolean(args.request_heartbeat);

            const workerPayload = {
                query: args.query ?? null,
                agent_id: args.agent_id ?? null,
                keep_tools: args.keep_tools ?? null,
                limit: sanitizedLimit,
                min_score: sanitizedMinScore,
                request_heartbeat: requestHeartbeat,
            };

            const workerResult = await this.handleFindToolsViaWorker(workerPayload);
            if (workerResult) {
                return workerResult;
            }

            return this.handleFindToolsViaProcess(
                args,
                sanitizedLimit,
                sanitizedMinScore,
                requestHeartbeat,
            );
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

    serializeWorkerResult(data) {
        if (data === null || data === undefined) {
            return '';
        }

        if (typeof data === 'string') {
            return data;
        }

        try {
            return JSON.stringify(data);
        } catch (error) {
            return String(data);
        }
    }

    describeAxiosError(error) {
        if (axios.isAxiosError?.(error)) {
            if (error.response) {
                const payload = typeof error.response.data === 'string'
                    ? error.response.data
                    : JSON.stringify(error.response.data);
                return `HTTP ${error.response.status} ${payload}`;
            }

            if (error.request) {
                return 'No response received from worker service';
            }
        }

        return error?.message || String(error);
    }

    async ensureWorkerAvailable() {
        if (DISABLE_WORKER_SERVICE) {
            return false;
        }

        const now = Date.now();
        if (this.workerServiceAvailable && now - this.lastWorkerHealthCheck < WORKER_HEALTH_CHECK_INTERVAL_MS) {
            return true;
        }

        this.lastWorkerHealthCheck = now;

        try {
            await axios.get(`${WORKER_SERVICE_URL}/health`, { timeout: WORKER_HEALTH_TIMEOUT_MS });
            this.workerServiceAvailable = true;
        } catch (error) {
            this.workerServiceAvailable = false;
            console.warn('Worker service health check failed:', this.describeAxiosError(error));
        }

        return this.workerServiceAvailable;
    }

    async handleFindToolsViaWorker(payload) {
        const available = await this.ensureWorkerAvailable();
        if (!available) {
            return null;
        }

        try {
            const response = await axios.post(
                `${WORKER_SERVICE_URL}/find_tools`,
                payload,
                { timeout: WORKER_REQUEST_TIMEOUT_MS },
            );

            const serialized = this.serializeWorkerResult(response.data);
            return {
                content: [
                    {
                        type: 'text',
                        text: serialized || JSON.stringify({ status: 'error', message: 'Empty worker response' }),
                    },
                ],
            };
        } catch (error) {
            this.workerServiceAvailable = false;
            console.warn('Worker service request failed, falling back to legacy script:', this.describeAxiosError(error));
            return null;
        }
    }

    async handleFindToolsViaProcess(args, sanitizedLimit, sanitizedMinScore, requestHeartbeat) {
        const scriptPath = path.join(__dirname, '..', 'find_tools.py');

        const pythonArgs = ['python3', scriptPath];

        if (args.query) pythonArgs.push('--query', args.query);
        if (args.agent_id) pythonArgs.push('--agent_id', args.agent_id);
        if (args.keep_tools) pythonArgs.push('--keep_tools', args.keep_tools);

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
                        content: content.length
                            ? content
                            : [
                                  {
                                      type: 'text',
                                      text: 'find_tools completed with no output.',
                                  },
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
