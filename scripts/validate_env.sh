#!/bin/bash
# Validate that all required environment variables are set
# Usage: ./scripts/validate_env.sh [.env file]

set -e

ENV_FILE="${1:-.env}"
ENV_EXAMPLE=".env.example"

if [ ! -f "$ENV_EXAMPLE" ]; then
    echo "ERROR: $ENV_EXAMPLE not found"
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "WARNING: $ENV_FILE not found, checking environment variables only"
    CHECK_FILE=false
else
    CHECK_FILE=true
fi

# Required variables (must be set and non-empty)
REQUIRED_VARS=(
    "LETTA_API_URL"
    "WEAVIATE_URL"
    "EMBEDDING_PROVIDER"
)

# Optional but recommended
RECOMMENDED_VARS=(
    "LETTA_PASSWORD"
    "WEAVIATE_API_KEY"
    "NEVER_DETACH_TOOLS"
    "RERANKER_URL"
)

echo "=== Environment Validation ==="
echo ""

ERRORS=0
WARNINGS=0

# Check required variables
echo "Checking required variables..."
for var in "${REQUIRED_VARS[@]}"; do
    if [ "$CHECK_FILE" = true ]; then
        value=$(grep "^${var}=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'")
    else
        value="${!var}"
    fi
    
    if [ -z "$value" ] || [[ "$value" == *"your-"* ]] || [[ "$value" == *"sk-your"* ]]; then
        echo "  ERROR: $var is not set or has placeholder value"
        ((ERRORS++))
    else
        echo "  OK: $var"
    fi
done

echo ""
echo "Checking recommended variables..."
for var in "${RECOMMENDED_VARS[@]}"; do
    if [ "$CHECK_FILE" = true ]; then
        value=$(grep "^${var}=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'")
    else
        value="${!var}"
    fi
    
    if [ -z "$value" ] || [[ "$value" == *"your-"* ]]; then
        echo "  WARNING: $var is not set"
        ((WARNINGS++))
    else
        echo "  OK: $var"
    fi
done

echo ""
echo "=== Summary ==="
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "FAILED: Fix required variables before deploying"
    exit 1
fi

echo ""
echo "PASSED: Environment is valid"
exit 0
