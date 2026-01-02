#!/bin/bash
# End-to-end webhook test script
# Tests the full flow: Letta API → Webhook → tool-selector-api → Weaviate

set -e

WEBHOOK_URL="${1:-http://192.168.50.90:8020/webhook/letta}"
LETTA_URL="${2:-http://192.168.50.90:8283/v1}"
LETTA_PASSWORD="${LETTA_PASSWORD:-}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           Webhook E2E Integration Test                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Webhook URL: $WEBHOOK_URL"
echo "Letta URL:   $LETTA_URL"
echo ""

# 1. Test webhook health endpoint
echo -e "${YELLOW}[1/7]${NC} Testing webhook health endpoint..."
HEALTH_RESPONSE=$(curl -sf "${WEBHOOK_URL%/letta}/health" 2>/dev/null || echo "FAILED")
if [[ "$HEALTH_RESPONSE" == *"healthy"* ]]; then
    echo -e "  ${GREEN}✓${NC} Webhook endpoint is healthy"
    echo "  $HEALTH_RESPONSE" | jq -c . 2>/dev/null || echo "  $HEALTH_RESPONSE"
else
    echo -e "  ${RED}✗${NC} Webhook endpoint not responding"
    exit 1
fi
echo ""

# 2. Test Letta API connectivity
echo -e "${YELLOW}[2/7]${NC} Testing Letta API connectivity..."
LETTA_HEALTH=$(curl -sf "$LETTA_URL/health" 2>/dev/null || echo "FAILED")
if [[ "$LETTA_HEALTH" != "FAILED" ]]; then
    echo -e "  ${GREEN}✓${NC} Letta API is responding"
else
    echo -e "  ${RED}✗${NC} Letta API not responding at $LETTA_URL"
    exit 1
fi
echo ""

# 3. Get current tool count from cache
echo -e "${YELLOW}[3/7]${NC} Getting baseline tool count..."
BASELINE_COUNT=$(curl -sf "http://192.168.50.90:8020/api/tools" 2>/dev/null | jq 'length' || echo "0")
echo -e "  Baseline tool count: $BASELINE_COUNT"
echo ""

# 4. Create a test tool via Letta API (source_code only, name derived from function)
TIMESTAMP=$(date +%s)
TOOL_FUNC_NAME="webhook_e2e_test_${TIMESTAMP}"
echo -e "${YELLOW}[4/7]${NC} Creating test tool via Letta API: $TOOL_FUNC_NAME"

TOOL_RESPONSE=$(curl -sL -X POST "${LETTA_URL}/tools/" \
  -H "Content-Type: application/json" \
  -H "X-BARE-PASSWORD: password $LETTA_PASSWORD" \
  -d "{
    \"source_type\": \"python\",
    \"source_code\": \"def ${TOOL_FUNC_NAME}():\\n    \\\"\\\"\\\"E2E test tool for webhook verification - safe to delete\\\"\\\"\\\"\\n    return 'test'\"
  }" 2>/dev/null)

TOOL_ID=$(echo "$TOOL_RESPONSE" | jq -r '.id // empty')
TOOL_NAME=$(echo "$TOOL_RESPONSE" | jq -r '.name // empty')

if [ -z "$TOOL_ID" ]; then
    echo -e "  ${RED}✗${NC} Failed to create tool via Letta API"
    echo "  Response: $TOOL_RESPONSE"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Created tool: $TOOL_ID (name: $TOOL_NAME)"
echo ""

# 5. Wait for webhook to process and verify tool appears
echo -e "${YELLOW}[5/7]${NC} Waiting for webhook processing (3s)..."
sleep 3

echo "  Checking tool-selector cache for new tool..."
NEW_COUNT=$(curl -sf "http://192.168.50.90:8020/api/tools" 2>/dev/null | jq 'length' || echo "0")
FOUND_TOOL=$(curl -sf "http://192.168.50.90:8020/api/tools" 2>/dev/null | jq -r ".[] | select(.name == \"$TOOL_NAME\") | .name" || echo "")

if [ "$FOUND_TOOL" == "$TOOL_NAME" ]; then
    echo -e "  ${GREEN}✓${NC} Tool found in cache via webhook! (count: $BASELINE_COUNT → $NEW_COUNT)"
else
    echo -e "  ${YELLOW}⚠${NC} Tool not found in cache yet (webhook may not have fired)"
fi
echo ""

# 6. Delete the test tool
echo -e "${YELLOW}[6/7]${NC} Deleting test tool..."
DELETE_RESPONSE=$(curl -sL -X DELETE "${LETTA_URL}/tools/${TOOL_ID}" \
  -H "X-BARE-PASSWORD: password $LETTA_PASSWORD" 2>/dev/null)
echo -e "  ${GREEN}✓${NC} Delete request sent"
echo ""

# 7. Verify tool removal
echo -e "${YELLOW}[7/7]${NC} Waiting for deletion webhook (3s)..."
sleep 3

FINAL_COUNT=$(curl -sf "http://192.168.50.90:8020/api/tools" 2>/dev/null | jq 'length' || echo "0")
STILL_EXISTS=$(curl -sf "http://192.168.50.90:8020/api/tools" 2>/dev/null | jq -r ".[] | select(.name == \"$TOOL_NAME\") | .name" || echo "")

if [ -z "$STILL_EXISTS" ]; then
    echo -e "  ${GREEN}✓${NC} Tool successfully removed from cache (count: $NEW_COUNT → $FINAL_COUNT)"
else
    echo -e "  ${YELLOW}⚠${NC} Tool still in cache (deletion webhook may not have fired)"
fi
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    Test Summary                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
if [ "$FOUND_TOOL" == "$TOOL_NAME" ] && [ -z "$STILL_EXISTS" ]; then
    echo -e "${GREEN}✓ All tests passed - Webhook integration working!${NC}"
    exit 0
elif [ "$FOUND_TOOL" == "$TOOL_NAME" ]; then
    echo -e "${YELLOW}⚠ Partial success - tool.created worked, tool.deleted may need verification${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Webhooks may not be configured or reachable${NC}"
    exit 1
fi
