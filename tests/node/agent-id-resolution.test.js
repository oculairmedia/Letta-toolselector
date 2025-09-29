import test from 'node:test';
import assert from 'node:assert/strict';

process.env.ENABLE_AGENT_ID_HEADER = 'true';
process.env.REQUIRE_AGENT_ID = 'true';
process.env.STRICT_AGENT_ID_VALIDATION = 'true';
process.env.DEBUG_AGENT_ID_SOURCE = 'false';

const { ToolSelectorServer } = await import('../../src/index.js');
const { validateAgentId, extractAgentIdFromContext } = ToolSelectorServer.prototype;

test('validateAgentId returns header value when provided', () => {
    const resolved = validateAgentId.call({}, 'agent-123', undefined);
    assert.strictEqual(resolved, 'agent-123');
});

test('validateAgentId falls back to parameter value', () => {
    const resolved = validateAgentId.call({}, undefined, 'agent-456');
    assert.strictEqual(resolved, 'agent-456');
});

test('validateAgentId enforces matching header and parameter values', () => {
    assert.throws(
        () => validateAgentId.call({}, 'agent-111', 'agent-222'),
        /Agent ID mismatch/,
    );
});

test('validateAgentId rejects missing agent identifiers', () => {
    assert.throws(
        () => validateAgentId.call({}, undefined, undefined),
        /Agent ID must be provided/,
    );
});

test('validateAgentId enforces basic format rules', () => {
    assert.throws(
        () => validateAgentId.call({}, 'invalid@id', undefined),
        /Invalid agent ID format/,
    );
});

test('extractAgentIdFromContext reads x-agent-id header when enabled', () => {
    const extracted = extractAgentIdFromContext.call({}, {
        headers: { 'x-agent-id': 'agent-789' },
    });
    assert.strictEqual(extracted, 'agent-789');
});
